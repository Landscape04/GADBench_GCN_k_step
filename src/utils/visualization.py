# visualization.py
"""
Visualization and Display Utilities
"""


def print_metrics(metrics, dataset_name, model_name):
    """
    Format and print evaluation metrics
    
    Args:
        metrics: Metrics dictionary
        dataset_name: Dataset name
        model_name: Model name
    """
    print(f"\n=== {dataset_name.upper()} - {model_name.upper()} Results ===")
    print(f"AUROC:   {metrics.get('AUROC', 0.0):.4f}")
    print(f"AUPRC:   {metrics.get('AUPRC', 0.0):.4f}")
    print(f"REC@50:  {metrics.get('REC@50', 0.0):.4f}")
    print(f"REC@100: {metrics.get('REC@100', 0.0):.4f}")
    print("=" * 50)