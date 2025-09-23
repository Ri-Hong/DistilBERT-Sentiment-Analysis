"""
Plot training and evaluation metrics from MLflow runs.
"""

import mlflow
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from mlflow.tracking import MlflowClient

# Set style for better visualization
plt.style.use('seaborn-v0_8')
sns.set_theme()

def get_run_metrics(run_id: str) -> pd.DataFrame:
    """
    Get metrics from a specific run and convert to DataFrame.
    """
    client = MlflowClient()
    run = client.get_run(run_id)
    
    # Get metrics history
    metrics = {}
    for key in ['train_loss', 'eval_accuracy', 'eval_f1']:
        history = client.get_metric_history(run_id, key)
        metrics[key] = {h.step: h.value for h in history}
    
    # Convert to DataFrame
    df = pd.DataFrame(metrics)
    df.index.name = 'epoch'
    return df

def plot_metrics(run_id: str, save_path: str = 'images/training_metrics.png'):
    """
    Create a multi-plot figure showing training metrics.
    """
    df = get_run_metrics(run_id)
    
    # Create a figure with multiple subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    fig.suptitle('Training and Evaluation Metrics Over Time', fontsize=16)
    
    # Plot training loss
    ax1.plot(df.index, df['train_loss'], marker='o', linewidth=2, label='Training Loss')
    ax1.set_title('Training Loss vs Epoch')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.grid(True)
    ax1.legend()
    
    # Plot evaluation metrics
    ax2.plot(df.index, df['eval_accuracy'], marker='o', linewidth=2, label='Accuracy')
    ax2.plot(df.index, df['eval_f1'], marker='s', linewidth=2, label='F1 Score')
    ax2.set_title('Evaluation Metrics vs Epoch')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Score')
    ax2.grid(True)
    ax2.legend()
    
    # Add baseline thresholds
    ax2.axhline(y=0.85, color='r', linestyle='--', label='Baseline Threshold (0.85)')
    ax2.legend()
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Metrics plot saved to {save_path}")
    
    # Print summary statistics
    print("\nMetrics Summary:")
    print("-" * 50)
    print("Training Loss:")
    print(f"  Initial: {df['train_loss'].iloc[0]:.4f}")
    print(f"  Final: {df['train_loss'].iloc[-1]:.4f}")
    print(f"  Improvement: {(1 - df['train_loss'].iloc[-1]/df['train_loss'].iloc[0])*100:.1f}%")
    print("\nEvaluation Metrics:")
    print(f"  Final Accuracy: {df['eval_accuracy'].iloc[-1]:.4f}")
    print(f"  Final F1 Score: {df['eval_f1'].iloc[-1]:.4f}")
    print(f"  Accuracy Improvement: {(df['eval_accuracy'].iloc[-1] - df['eval_accuracy'].iloc[0])*100:.1f}%")
    print(f"  F1 Score Improvement: {(df['eval_f1'].iloc[-1] - df['eval_f1'].iloc[0])*100:.1f}%")

if __name__ == "__main__":
    # Use our best run
    RUN_ID = "32524e04adb3455c86e8a162b8b9095e"
    plot_metrics(RUN_ID)
