"""
MLflow configuration for experiment tracking.
"""

import os
from pathlib import Path

# MLflow tracking configuration
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "/app/mlruns/tracking")
EXPERIMENT_NAME = "sentiment-analysis-training"

# Ensure MLflow directories exist
Path("/app/mlruns/tracking").mkdir(parents=True, exist_ok=True)
Path("/app/model/artifacts").mkdir(parents=True, exist_ok=True)

# Model artifacts configuration
ARTIFACTS_DIR = Path("/app/model/artifacts")
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

# Import metrics configuration
from app.training.metrics_config import BASELINE_METRICS, evaluate_metrics

def setup_mlflow():
    """
    Configure MLflow tracking.
    """
    import mlflow
    
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)
    
    # Create experiment if it doesn't exist
    experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
    if experiment is None:
        mlflow.create_experiment(EXPERIMENT_NAME)

def log_model_metrics(metrics: dict) -> bool:
    """
    Log model metrics and check if they meet promotion criteria.
    Returns True if the model meets all baseline metrics.
    """
    evaluation = evaluate_metrics(metrics)
    
    # Log evaluation results to MLflow
    import mlflow
    mlflow.log_dict(evaluation, "metric_evaluation.json")
    
    return evaluation["meets_baseline"]
