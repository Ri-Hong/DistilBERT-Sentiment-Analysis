"""
DistilBERT fine-tuning script for sentiment analysis on IMDB dataset.
Includes mixed precision training and MLflow tracking.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Tuple

import mlflow
import numpy as np
import torch
from accelerate import Accelerator
from datasets import load_from_disk
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (
    AdamW,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
    DataCollatorWithPadding,
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
MODEL_NAME = "distilbert-base-uncased"
MAX_LENGTH = 512
BATCH_SIZE = 16
LEARNING_RATE = 2e-5
NUM_EPOCHS = 3
WARMUP_STEPS = 0
WEIGHT_DECAY = 0.01


def load_data(data_dir: str) -> Tuple[DataLoader, DataLoader]:
    """
    Load and prepare IMDB dataset for training.
    """
    # Load tokenized datasets
    train_dataset = load_from_disk(str(Path(data_dir) / "train"))
    test_dataset = load_from_disk(str(Path(data_dir) / "test"))

    # Create dataloaders
    train_dataloader = DataLoader(
        train_dataset,
        shuffle=True,
        batch_size=BATCH_SIZE,
        collate_fn=default_data_collator,
    )
    eval_dataloader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        collate_fn=default_data_collator,
    )

    return train_dataloader, eval_dataloader


def load_data(data_dir: str) -> Tuple[DataLoader, DataLoader]:
    """
    Load and prepare IMDB dataset for training.
    """
    # Load tokenizer for data collation
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # Load tokenized datasets
    train_dataset = load_from_disk(str(Path(data_dir) / "train"))
    test_dataset = load_from_disk(str(Path(data_dir) / "test"))

    # Create dataloaders
    train_dataloader = DataLoader(
        train_dataset,
        shuffle=True,
        batch_size=BATCH_SIZE,
        collate_fn=data_collator,
    )
    eval_dataloader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        collate_fn=data_collator,
    )

    return train_dataloader, eval_dataloader


def train_epoch(
    model: torch.nn.Module,
    train_dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    lr_scheduler: torch.optim.lr_scheduler,
    accelerator: Accelerator,
    epoch: int,
) -> float:
    """
    Train for one epoch and return average loss.
    """
    model.train()
    total_loss = 0
    progress_bar = tqdm(
        train_dataloader,
        desc=f"Training epoch {epoch}",
        disable=not accelerator.is_local_main_process,
    )

    for batch in progress_bar:
        with accelerator.accumulate(model):
            outputs = model(**batch)
            loss = outputs.loss
            accelerator.backward(loss)
            
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            
            total_loss += loss.detach().float()

        progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})

    return total_loss.item() / len(train_dataloader)


def evaluate(
    model: torch.nn.Module,
    eval_dataloader: DataLoader,
    accelerator: Accelerator,
) -> Dict[str, float]:
    """
    Evaluate model and return metrics.
    """
    model.eval()
    all_predictions = []
    all_labels = []

    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        with torch.no_grad():
            outputs = model(**batch)
            predictions = outputs.logits.argmax(dim=-1)
            
            predictions = accelerator.gather(predictions)
            labels = accelerator.gather(batch["labels"])
            
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    metrics = {
        "accuracy": accuracy_score(all_labels, all_predictions),
        "f1": f1_score(all_labels, all_predictions, average="weighted"),
    }
    
    return metrics


from app.training.mlflow_config import setup_mlflow, log_model_metrics, ARTIFACTS_DIR

def main():
    # Initialize MLflow
    setup_mlflow()

    # Initialize accelerator
    # Don't use mixed precision on MPS (Apple Silicon)
    if torch.backends.mps.is_available():
        accelerator = Accelerator()
        logger.info("Using MPS (Apple Silicon) device")
    else:
        accelerator = Accelerator(mixed_precision="fp16")
        logger.info(f"Using device: {accelerator.device}")

    # Load model and tokenizer
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=2,
    )
    
    # Load data
    train_dataloader, eval_dataloader = load_data("data/processed")

    # Prepare optimizer and scheduler
    optimizer = AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )
    
    num_training_steps = NUM_EPOCHS * len(train_dataloader)
    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=WARMUP_STEPS,
        num_training_steps=num_training_steps,
    )

    # Prepare for distributed training
    model, optimizer, train_dataloader, eval_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader, lr_scheduler
    )

    # Start MLflow run
    with mlflow.start_run():
        # Log parameters
        mlflow.log_params({
            "model_name": MODEL_NAME,
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "num_epochs": NUM_EPOCHS,
            "max_length": MAX_LENGTH,
            "weight_decay": WEIGHT_DECAY,
        })

        # Training loop
        best_f1 = 0.0
        for epoch in range(NUM_EPOCHS):
            # Train
            avg_loss = train_epoch(
                model,
                train_dataloader,
                optimizer,
                lr_scheduler,
                accelerator,
                epoch,
            )
            mlflow.log_metric("train_loss", avg_loss, step=epoch)
            
            # Evaluate
            metrics = evaluate(model, eval_dataloader, accelerator)
            for name, value in metrics.items():
                mlflow.log_metric(f"eval_{name}", value, step=epoch)
            
            logger.info(
                f"Epoch {epoch}: loss={avg_loss:.4f}, "
                f"accuracy={metrics['accuracy']:.4f}, "
                f"f1={metrics['f1']:.4f}"
            )

            # Save best model
            if metrics["f1"] > best_f1:
                best_f1 = metrics["f1"]
                if accelerator.is_local_main_process:
                    unwrapped_model = accelerator.unwrap_model(model)
                    model_path = ARTIFACTS_DIR / "best"
                    unwrapped_model.save_pretrained(str(model_path))
                    mlflow.log_artifacts(str(model_path), artifact_path="model")
                    
                    # Check if model meets promotion criteria
                    if log_model_metrics(metrics):
                        logger.info("Model meets promotion criteria!")

        logger.info("Training completed!")
        logger.info(f"Best F1 score: {best_f1:.4f}")


if __name__ == "__main__":
    main()
