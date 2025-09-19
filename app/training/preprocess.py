"""
Preprocess IMDB dataset for DistilBERT fine-tuning.
Tokenizes and saves the datasets in a format ready for training.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Tuple

from datasets import Dataset
from transformers import AutoTokenizer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
MODEL_NAME = "distilbert-base-uncased"
MAX_LENGTH = 512


def load_jsonl_dataset(file_path: str) -> Dataset:
    """
    Load dataset from a JSONL file (one JSON object per line).
    """
    texts = []
    labels = []
    
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line.strip())
            texts.append(item["text"])
            labels.append(item["label"])
    
    return Dataset.from_dict({
        "text": texts,
        "label": labels,
    })


def tokenize_function(examples: Dict, tokenizer: AutoTokenizer) -> Dict:
    """
    Tokenize text data using the DistilBERT tokenizer.
    """
    tokenized = tokenizer(
        examples["text"],
        padding="max_length",
        truncation=True,
        max_length=MAX_LENGTH,
        return_tensors=None,  # Don't return tensors here, Dataset will handle it
    )
    
    # Ensure we have the right label format
    tokenized["labels"] = examples["label"]
    return tokenized


def main():
    # Create output directory
    output_dir = Path("data/processed")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize tokenizer
    logger.info(f"Loading tokenizer: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # Load and process training data
    logger.info("Processing training data...")
    train_dataset = load_jsonl_dataset("data/raw/imdb_train.jsonl")
    train_tokenized = train_dataset.map(
        lambda x: tokenize_function(x, tokenizer),
        batched=True,
        remove_columns=["text"],
    )
    train_tokenized.save_to_disk(str(output_dir / "train"))
    logger.info(f"Saved processed training data: {len(train_tokenized)} examples")

    # Load and process test data
    logger.info("Processing test data...")
    test_dataset = load_jsonl_dataset("data/raw/imdb_test.jsonl")
    test_tokenized = test_dataset.map(
        lambda x: tokenize_function(x, tokenizer),
        batched=True,
        remove_columns=["text"],
    )
    test_tokenized.save_to_disk(str(output_dir / "test"))
    logger.info(f"Saved processed test data: {len(test_tokenized)} examples")

    logger.info("Data preprocessing completed!")


if __name__ == "__main__":
    main()
