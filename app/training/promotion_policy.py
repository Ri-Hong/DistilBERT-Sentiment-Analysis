"""
Model promotion policies and procedures.

This module defines the criteria and workflows for promoting models through
different stages (Development → Staging → Production).
"""

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import mlflow
from mlflow.tracking import MlflowClient

# Constants
SOAK_TEST_DURATION = timedelta(days=7)  # Time required in staging before production
ERROR_RATE_THRESHOLD = 0.01  # Maximum acceptable error rate (1%)
LATENCY_P95_THRESHOLD = 100  # Maximum acceptable P95 latency (ms)
LATENCY_P99_THRESHOLD = 200  # Maximum acceptable P99 latency (ms)
MIN_THROUGHPUT = 10  # Minimum queries per second
MIN_INFERENCE_CALLS = 10000  # Minimum number of successful inferences in staging

@dataclass
class PromotionCriteria:
    """Criteria that must be met for model promotion."""
    # Accuracy metrics (from training)
    min_accuracy: float = 0.85
    min_f1_score: float = 0.85
    
    # Performance metrics (from staging)
    max_error_rate: float = ERROR_RATE_THRESHOLD
    max_p95_latency: float = LATENCY_P95_THRESHOLD
    max_p99_latency: float = LATENCY_P99_THRESHOLD
    min_qps: float = MIN_THROUGHPUT
    min_inference_calls: int = MIN_INFERENCE_CALLS
    
    # Soak testing
    soak_duration: timedelta = SOAK_TEST_DURATION

def get_model_metrics(model_name: str, version: int) -> Dict[str, float]:
    """
    Get all relevant metrics for a model version.
    """
    client = MlflowClient()
    model_version = client.get_model_version(model_name, version)
    
    # Get training metrics from tags
    metrics = {
        "accuracy": float(model_version.tags.get("accuracy", 0)),
        "f1_score": float(model_version.tags.get("f1_score", 0)),
    }
    
    # In a real system, we would also fetch:
    # - Error rates from monitoring system
    # - Latency metrics from Prometheus
    # - Throughput data from load balancer
    # For now, we'll just use training metrics
    
    return metrics

def check_promotion_criteria(
    model_name: str,
    version: int,
    criteria: Optional[PromotionCriteria] = None
) -> Dict[str, bool]:
    """
    Check if a model version meets promotion criteria.
    Returns a dictionary of criteria and whether they were met.
    """
    if criteria is None:
        criteria = PromotionCriteria()
    
    metrics = get_model_metrics(model_name, version)
    
    return {
        "accuracy_threshold_met": metrics["accuracy"] >= criteria.min_accuracy,
        "f1_score_threshold_met": metrics["f1_score"] >= criteria.min_f1_score,
        # In production, we would also check:
        # "error_rate_acceptable": error_rate <= criteria.max_error_rate,
        # "latency_p95_acceptable": p95_latency <= criteria.max_p95_latency,
        # "latency_p99_acceptable": p99_latency <= criteria.max_p99_latency,
        # "throughput_acceptable": qps >= criteria.min_qps,
        # "enough_inference_calls": total_calls >= criteria.min_inference_calls,
        # "soak_time_met": time_in_staging >= criteria.soak_duration,
    }

def promote_to_production(model_name: str, version: int) -> bool:
    """
    Promote a model version to production if it meets all criteria.
    Returns True if promotion was successful.
    """
    # Check if model meets promotion criteria
    results = check_promotion_criteria(model_name, version)
    
    if not all(results.values()):
        failed_criteria = [k for k, v in results.items() if not v]
        print(f"Model failed promotion criteria: {failed_criteria}")
        return False
    
    # Promote the model
    client = MlflowClient()
    try:
        client.transition_model_version_stage(
            name=model_name,
            version=version,
            stage="Production",
            archive_existing_versions=True
        )
        print(f"Successfully promoted model {model_name} version {version} to Production")
        return True
    except Exception as e:
        print(f"Failed to promote model: {e}")
        return False

if __name__ == "__main__":
    # Example usage
    MODEL_NAME = "sentiment-analysis-distilbert"
    VERSION = 1  # Our latest version
    
    # Check and promote if criteria met
    results = check_promotion_criteria(MODEL_NAME, VERSION)
    print(f"Promotion criteria check results: {results}")
    
    if all(results.values()):
        promote_to_production(MODEL_NAME, VERSION)
