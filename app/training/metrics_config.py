"""
Configuration for model evaluation metrics and baselines.
"""

from typing import Dict, Any

# Baseline metrics that a model must meet to be considered for production
BASELINE_METRICS = {
    "eval_accuracy": {
        "threshold": 0.85,
        "description": "Minimum classification accuracy on test set",
        "higher_is_better": True,
    },
    "eval_f1": {
        "threshold": 0.85,
        "description": "Minimum F1 score (weighted average) on test set",
        "higher_is_better": True,
    },
}

# Additional metrics to track (but not used for baseline comparison)
TRACKING_METRICS = {
    "train_loss": {
        "description": "Training loss per epoch",
        "higher_is_better": False,
    },
    "eval_precision": {
        "description": "Precision score per class",
        "higher_is_better": True,
    },
    "eval_recall": {
        "description": "Recall score per class",
        "higher_is_better": True,
    },
}

# Performance requirements
PERFORMANCE_REQUIREMENTS = {
    "latency_ms": {
        "p95": 100,  # 95th percentile latency in milliseconds
        "p99": 200,  # 99th percentile latency in milliseconds
    },
    "throughput": {
        "min_qps": 10,  # Minimum queries per second
    },
}

def evaluate_metrics(metrics: Dict[str, float]) -> Dict[str, Any]:
    """
    Evaluate if metrics meet baseline requirements.
    Returns a dictionary with evaluation results.
    """
    results = {
        "meets_baseline": True,
        "failed_metrics": [],
        "metrics_summary": {},
    }
    
    for metric_name, config in BASELINE_METRICS.items():
        if metric_name in metrics:
            metric_value = metrics[metric_name]
            threshold = config["threshold"]
            meets_threshold = (
                metric_value >= threshold if config["higher_is_better"]
                else metric_value <= threshold
            )
            
            results["metrics_summary"][metric_name] = {
                "value": metric_value,
                "threshold": threshold,
                "meets_threshold": meets_threshold,
            }
            
            if not meets_threshold:
                results["meets_baseline"] = False
                results["failed_metrics"].append(metric_name)
    
    return results
