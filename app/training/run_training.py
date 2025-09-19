#!/usr/bin/env python3
"""
CLI script to launch and monitor model training.
"""

import argparse
import logging
import subprocess
import time
from pathlib import Path

import mlflow
from mlflow.tracking import MlflowClient

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def start_mlflow_server():
    """Start MLflow tracking server locally."""
    try:
        # Create mlruns directory if it doesn't exist
        Path("mlruns").mkdir(exist_ok=True)
        
        # Check if MLflow UI is already running
        result = subprocess.run(
            ["lsof", "-i", ":5000"],
            capture_output=True,
            text=True
        )
        if "mlflow" not in result.stdout:
            logger.info("Starting MLflow server...")
            # Start MLflow server in a new process
            server_process = subprocess.Popen(
                ["./venv/bin/mlflow", "server", 
                 "--backend-store-uri", "mlruns",
                 "--default-artifact-root", "mlruns",
                 "--host", "0.0.0.0", 
                 "--port", "5000"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            
            # Wait for server to start
            time.sleep(5)
            
            # Check if server started successfully
            if server_process.poll() is not None:
                stderr = server_process.stderr.read().decode()
                raise Exception(f"MLflow server failed to start: {stderr}")
            
            logger.info("MLflow server started successfully")
            
    except Exception as e:
        logger.error(f"Failed to start MLflow server: {e}")
        raise

def fetch_data():
    """Fetch IMDB dataset."""
    logger.info("Fetching IMDB dataset...")
    subprocess.run(["./venv/bin/python3", "-m", "data.fetch_data"], check=True)

def run_preprocessing():
    """Run data preprocessing step."""
    logger.info("Running data preprocessing...")
    # First fetch data if it doesn't exist
    if not (Path("data/raw/imdb_train.jsonl").exists() and 
            Path("data/raw/imdb_test.jsonl").exists()):
        fetch_data()
    subprocess.run(["./venv/bin/python3", "-m", "app.training.preprocess"], check=True)

def run_local_training(gpu: bool = False):
    """Run training locally."""
    env = {"CUDA_VISIBLE_DEVICES": "0"} if gpu else {"CUDA_VISIBLE_DEVICES": ""}
    logger.info(f"Starting local training {'with' if gpu else 'without'} GPU...")
    subprocess.run(
        ["./venv/bin/python3", "-m", "app.training.train"],
        env=env,
        check=True
    )

def run_k8s_training():
    """Launch training job on Kubernetes."""
    logger.info("Launching Kubernetes training job...")
    subprocess.run(
        ["kubectl", "apply", "-f", "infra/k8s/training-job.yaml"],
        check=True
    )

def monitor_training():
    """Monitor training progress through MLflow."""
    client = MlflowClient()
    experiment = client.get_experiment_by_name("sentiment-analysis-training")
    
    if experiment:
        runs = client.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=["attributes.start_time DESC"]
        )
        
        if runs:
            latest_run = runs[0]
            logger.info("\nLatest training run metrics:")
            for key, value in latest_run.data.metrics.items():
                logger.info(f"{key}: {value:.4f}")
            
            logger.info(f"\nMLflow UI: http://localhost:5000")
            logger.info(f"Run ID: {latest_run.info.run_id}")
        else:
            logger.info("No training runs found.")
    else:
        logger.info("No experiment found.")

def main():
    parser = argparse.ArgumentParser(description="Run and monitor model training")
    parser.add_argument(
        "--mode",
        choices=["local", "local-gpu", "k8s"],
        default="local",
        help="Training mode: local CPU, local GPU, or Kubernetes"
    )
    parser.add_argument(
        "--skip-preprocessing",
        action="store_true",
        help="Skip data preprocessing step"
    )
    parser.add_argument(
        "--monitor-only",
        action="store_true",
        help="Only monitor existing training without starting new one"
    )
    
    args = parser.parse_args()
    
    # Start MLflow server
    start_mlflow_server()
    
    if not args.monitor_only:
        # Run preprocessing if not skipped
        if not args.skip_preprocessing:
            run_preprocessing()
        
        # Start training based on mode
        if args.mode == "k8s":
            run_k8s_training()
        else:
            run_local_training(gpu=(args.mode == "local-gpu"))
    
    # Monitor training progress
    monitor_training()

if __name__ == "__main__":
    main()
