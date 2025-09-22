"""
Register a trained model in MLflow Model Registry.
"""

import logging
import mlflow
from pathlib import Path
from mlflow.tracking import MlflowClient

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
MODEL_NAME = "sentiment-analysis-distilbert"
DESCRIPTION = """
DistilBERT-based sentiment analysis model fine-tuned on IMDB dataset.
- Architecture: DistilBERT (distilbert-base-uncased)
- Task: Binary sentiment classification (positive/negative)
- Dataset: IMDB Reviews
- Performance:
  * Accuracy: >93%
  * F1 Score: >93%
- Latency Requirements:
  * P95: 100ms
  * P99: 200ms
- Minimum Throughput: 10 QPS
"""

def register_best_model(run_id: str, model_name: str = MODEL_NAME) -> None:
    """
    Register the best model from a given run in MLflow Model Registry.
    """
    client = MlflowClient()
    
    # Check if model already exists in registry
    try:
        model = client.get_registered_model(model_name)
        logger.info(f"Model {model_name} already exists in registry")
    except:
        model = client.create_registered_model(
            name=model_name,
            description=DESCRIPTION,
            tags={
                "task_type": "sentiment_analysis",
                "model_type": "distilbert",
                "dataset": "imdb",
                "framework": "pytorch",
            }
        )
        logger.info(f"Created new model {model_name} in registry")
    
    # Get the run and its metrics
    run = client.get_run(run_id)
    metrics = run.data.metrics
    
    # Register the model version
    model_uri = f"runs:/{run_id}/model"
    version = mlflow.register_model(
        model_uri=model_uri,
        name=model_name,
        tags={
            "accuracy": f"{metrics.get('eval_accuracy', 0):.4f}",
            "f1_score": f"{metrics.get('eval_f1', 0):.4f}",
            "source_commit": run.data.tags.get("mlflow.source.git.commit", "unknown"),
        }
    )
    logger.info(f"Registered model version: {version.version}")
    
    # Transition to Staging if metrics meet criteria
    if metrics.get("eval_accuracy", 0) >= 0.85 and metrics.get("eval_f1", 0) >= 0.85:
        client.transition_model_version_stage(
            name=model_name,
            version=version.version,
            stage="Staging",
            archive_existing_versions=True
        )
        logger.info(f"Transitioned version {version.version} to Staging")

if __name__ == "__main__":
    # Get the run ID from the best performing model
    run_id = "32524e04adb3455c86e8a162b8b9095e"  # This is our best run
    register_best_model(run_id)
