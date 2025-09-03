# metrics.py
"""
Evaluation Metrics for Graph Anomaly Detection
"""

import torch
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score


def calculate_metrics(y_true, y_scores, k_values=[50, 100]):
    """
    Calculate GADBench-style evaluation metrics
    
    Args:
        y_true: True labels (numpy array or torch tensor)
        y_scores: Prediction scores (numpy array or torch tensor)
        k_values: List of k values for REC@K
    
    Returns:
        dict: Dictionary containing AUROC, AUPRC, REC@K metrics
    """
    # Convert to numpy format
    if torch.is_tensor(y_true):
        y_true = y_true.cpu().numpy()
    if torch.is_tensor(y_scores):
        y_scores = y_scores.cpu().numpy()
    
    metrics = {}
    
    try:
        # AUROC (Area Under ROC Curve)
        auroc = roc_auc_score(y_true, y_scores)
        metrics['AUROC'] = auroc
        
        # AUPRC (Area Under Precision-Recall Curve)
        auprc = average_precision_score(y_true, y_scores)
        metrics['AUPRC'] = auprc
        
        # REC@K (Recall at K)
        for k in k_values:
            rec_at_k = recall_at_k(y_true, y_scores, k)
            metrics[f'REC@{k}'] = rec_at_k
            
    except Exception as e:
        print(f"Error calculating metrics: {str(e)}")
        # Return default values
        metrics['AUROC'] = 0.5
        metrics['AUPRC'] = np.mean(y_true)
        for k in k_values:
            metrics[f'REC@{k}'] = 0.0
    
    return metrics


def recall_at_k(y_true, y_scores, k):
    """
    Calculate Recall@K metric
    
    Args:
        y_true: True labels
        y_scores: Prediction scores
        k: Top-k value
    
    Returns:
        float: Recall@K value
    """
    if len(y_true) == 0 or k <= 0:
        return 0.0
    
    # Sort by prediction scores and get top-k indices
    top_k_indices = np.argsort(y_scores)[-k:]
    
    # Calculate anomalies in top-k
    top_k_anomalies = np.sum(y_true[top_k_indices])
    
    # Calculate total anomalies
    total_anomalies = np.sum(y_true)
    
    if total_anomalies == 0:
        return 0.0
    
    # Calculate Recall@K
    recall = top_k_anomalies / total_anomalies
    return recall