# trainer.py
"""
Model Training Implementation
"""

import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from .early_stopping import EarlyStopper


class Trainer:
    """Model trainer with early stopping and adaptive model support"""
    
    def __init__(self, model, optimizer, device, config, recorder=None):
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.config = config
        self.recorder = recorder
        
        self.stopper = EarlyStopper(
            initial_model=model,
            patience=config.get('patience', 10),
            delta=config.get('delta', 0.001),
            warmup_epochs=config.get('warmup_epochs', 5),
            smooth_window=config.get('smooth_window', 3)
        )
        
    def train_epoch(self, data, pos_weight):
        """Train one epoch"""
        self.model.train()
        self.optimizer.zero_grad()
        
        out = self.model(data.x, data.edge_index)
        loss = F.binary_cross_entropy_with_logits(
            out[data.train_mask],
            data.y[data.train_mask].float(),
            pos_weight=pos_weight
        )
        
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    @torch.no_grad()
    def validate(self, data):
        """Validate model performance"""
        self.model.eval()
        out = self.model(data.x, data.edge_index)
        val_pred = torch.sigmoid(out[data.val_mask])
        val_auc = roc_auc_score(data.y[data.val_mask].cpu(), val_pred.cpu())
        return val_auc
    
    @torch.no_grad()
    def test(self, data):
        """Test model"""
        self.model.eval()
        out = self.model(data.x, data.edge_index)
        return out.cpu()
    
    def train(self, data, trial_num):
        """Complete training process"""
        import time
        
        # Calculate class weights
        train_labels = data.y[data.train_mask]
        num_pos = train_labels.sum().item()
        num_neg = len(train_labels) - num_pos
        
        if num_pos == 0 or num_neg == 0:
            raise ValueError(f"Class imbalance error: pos={num_pos}, neg={num_neg}")
            
        pos_weight = torch.tensor([num_neg / num_pos], dtype=torch.float32, device=self.device)
        
        # Check if adaptive model
        is_adaptive_model = hasattr(self.model, 'adaptive_adjust_degree_percentile')
        
        for epoch in range(self.config['max_epochs']):
            epoch_start_time = time.time()
            
            # Train and validate
            loss = self.train_epoch(data, pos_weight)
            val_auc = self.validate(data)
            
            epoch_time = time.time() - epoch_start_time
            
            # Record epoch results if recorder is available
            if self.recorder:
                # Calculate validation AP (AUPRC) if needed
                val_ap = None
                try:
                    from sklearn.metrics import average_precision_score
                    self.model.eval()
                    with torch.no_grad():
                        out = self.model(data.x, data.edge_index)
                        val_pred = torch.sigmoid(out[data.val_mask])
                        val_ap = average_precision_score(data.y[data.val_mask].cpu(), val_pred.cpu())
                except:
                    val_ap = None
                
                self.recorder.record_epoch(trial_num, epoch+1, loss, val_auc, val_ap, epoch_time)
            
            # Print progress every 10 epochs or if verbose
            if epoch % 10 == 0 or epoch < 5:
                print(f"Epoch {epoch+1}/{self.config['max_epochs']}: loss={loss:.4f}, val_auc={val_auc:.4f}, time={epoch_time:.2f}s")
            
            # For adaptive models, call adaptive adjustment
            if is_adaptive_model:
                self.model.adaptive_adjust_degree_percentile(val_auc)
                
                # Print adaptive statistics every 20 epochs
                if epoch % 20 == 0 and epoch > 0:
                    stats = self.model.get_adaptation_stats()
                    print(f"Epoch {epoch}: degree_percentile={stats['current_degree_percentile']:.4f}")
            
            # Early stopping check
            if self.stopper.step(val_auc, self.model):
                stop_reason = self.stopper.get_stop_reason()
                final_stats = ""
                if is_adaptive_model:
                    stats = self.model.get_adaptation_stats()
                    final_stats = f", Final degree_percentile: {stats['current_degree_percentile']:.4f}"
                
                print(f"Trial {trial_num}: Early stopping at epoch {epoch+1} - {stop_reason}")
                print(f"Best Val AUC: {self.stopper.best_metric:.4f}{final_stats}")
                break
        
        # Load best model
        if self.stopper.best_model:
            self.model.load_state_dict(self.stopper.best_model)
        
        return epoch + 1