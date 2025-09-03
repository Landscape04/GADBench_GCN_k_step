# early_stopping.py
"""
Early Stopping Implementation
"""

import torch


class EarlyStopper:
    """Early stopping mechanism for training"""
    
    def __init__(self, initial_model, patience=10, delta=0.001, 
                 warmup_epochs=5, smooth_window=3):
        self.patience = patience
        self.delta = delta
        self.warmup = warmup_epochs
        self.smooth_window = smooth_window
        
        self.counter = 0
        self.best_metric = -float('inf')
        self.metric_history = []
        self.improvement_history = []
        
        # Additional stopping criteria
        self.min_epochs = max(10, warmup_epochs)  # Minimum epochs to train
        self.convergence_threshold = 0.0001  # Threshold for convergence detection
        self.convergence_window = 5  # Window to check convergence
        
        # Initialize with initial model
        self.best_model = {}
        if initial_model is not None:
            self._save_model(initial_model)

    def _save_model(self, model):
        """Safe model saving method"""
        self.best_model = {
            k: v.cpu().clone().detach()
            for k, v in model.state_dict().items()
        }

    def step(self, current_metric, model=None):
        """
        Enhanced early stopping check with multiple criteria
        
        Args:
            current_metric: Current validation metric
            model: Model to save if improvement found
            
        Returns:
            bool: True if should stop, False otherwise
        """
        if not self.best_model and model is not None:
            self._save_model(model)
            
        self.metric_history.append(current_metric)
        current_epoch = len(self.metric_history)
        
        # Calculate smoothed metric
        smoothed = current_metric
        if len(self.metric_history) >= self.smooth_window:
            smoothed = sum(self.metric_history[-self.smooth_window:]) / self.smooth_window
        
        # Track improvement
        if len(self.metric_history) > 1:
            improvement = current_metric - self.metric_history[-2]
            self.improvement_history.append(improvement)
        
        # Update best metric and counter
        if current_epoch >= self.warmup and smoothed > self.best_metric + self.delta:
            self.best_metric = smoothed
            self.counter = 0
            if model is not None:
                self._save_model(model)
        else:
            self.counter += 1
        
        # Multiple stopping criteria
        should_stop = False
        
        # 1. Standard patience-based stopping
        if self.counter >= self.patience and current_epoch >= self.min_epochs:
            should_stop = True
        
        # 2. Convergence detection - very small improvements over recent epochs
        if (current_epoch >= self.min_epochs and 
            len(self.improvement_history) >= self.convergence_window):
            recent_improvements = self.improvement_history[-self.convergence_window:]
            avg_improvement = sum(recent_improvements) / len(recent_improvements)
            if abs(avg_improvement) < self.convergence_threshold:
                should_stop = True
        
        # 3. Performance plateau detection - no significant change in recent epochs
        if (current_epoch >= self.min_epochs and 
            len(self.metric_history) >= self.convergence_window):
            recent_metrics = self.metric_history[-self.convergence_window:]
            metric_std = (sum([(x - sum(recent_metrics)/len(recent_metrics))**2 
                              for x in recent_metrics]) / len(recent_metrics)) ** 0.5
            if metric_std < self.convergence_threshold:
                should_stop = True
        
        return should_stop
    
    def get_stop_reason(self):
        """Get the reason for stopping"""
        current_epoch = len(self.metric_history)
        
        if self.counter >= self.patience:
            return f"Patience exceeded ({self.counter}/{self.patience})"
        
        if (len(self.improvement_history) >= self.convergence_window and
            current_epoch >= self.min_epochs):
            recent_improvements = self.improvement_history[-self.convergence_window:]
            avg_improvement = sum(recent_improvements) / len(recent_improvements)
            if abs(avg_improvement) < self.convergence_threshold:
                return f"Converged (avg improvement: {avg_improvement:.6f})"
        
        if (len(self.metric_history) >= self.convergence_window and
            current_epoch >= self.min_epochs):
            recent_metrics = self.metric_history[-self.convergence_window:]
            metric_std = (sum([(x - sum(recent_metrics)/len(recent_metrics))**2 
                              for x in recent_metrics]) / len(recent_metrics)) ** 0.5
            if metric_std < self.convergence_threshold:
                return f"Performance plateau (std: {metric_std:.6f})"
        
        return "Training completed"