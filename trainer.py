import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
import copy
import time

class EarlyStopper:
    """早停机制"""
    def __init__(self, initial_model, patience=20, delta=0.00005, 
                 warmup_epochs=10, smooth_window=3):
        self.patience = patience
        self.delta = delta
        self.warmup = warmup_epochs
        self.smooth_window = smooth_window
        
        self.counter = 0
        self.best_metric = -float('inf')
        self.metric_history = []
        
        # 初始化时保存初始模型
        self.best_model = {}
        if initial_model is not None:
            self._save_model(initial_model)

    def _save_model(self, model):
        """安全的模型保存方法"""
        self.best_model = {
            k: v.cpu().clone().detach()
            for k, v in model.state_dict().items()
        }

    def step(self, current_metric, model=None):
        if not self.best_model and model is not None:
            self._save_model(model)
            
        self.metric_history.append(current_metric)
        
        # 计算平滑指标
        smoothed = current_metric
        if len(self.metric_history) >= self.smooth_window:
            smoothed = sum(self.metric_history[-self.smooth_window:])/self.smooth_window
            
        # 更新最佳指标
        if (len(self.metric_history) >= self.warmup) and (smoothed > self.best_metric + self.delta):
            self.best_metric = smoothed
            self.counter = 0
            if model is not None:
                self._save_model(model)
        else:
            self.counter += 1
            
        return self.counter >= self.patience and len(self.metric_history) >= self.warmup

class Trainer:
    """模型训练器"""
    def __init__(self, model, optimizer, device, config):
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.config = config
        
        self.stopper = EarlyStopper(
            initial_model=model,
            patience=config.get('patience', 20),
            delta=config.get('delta', 0.00005),
            warmup_epochs=config.get('warmup_epochs', 10),
            smooth_window=config.get('smooth_window', 3)
        )
        
    def train_epoch(self, data, pos_weight):
        """训练一个epoch"""
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
        """验证模型性能"""
        self.model.eval()
        out = self.model(data.x, data.edge_index)
        val_pred = torch.sigmoid(out[data.val_mask])
        val_auc = roc_auc_score(data.y[data.val_mask].cpu(), val_pred.cpu())
        return val_auc
    
    @torch.no_grad()
    def test(self, data):
        """测试模型"""
        self.model.eval()
        out = self.model(data.x, data.edge_index)
        return out.cpu()
    
    def train(self, data, trial_num):
        """完整训练流程"""
        # 计算类别权重
        train_labels = data.y[data.train_mask]
        num_pos = train_labels.sum().item()
        num_neg = len(train_labels) - num_pos
        
        if num_pos == 0 or num_neg == 0:
            raise ValueError(f"类别不平衡异常: 正样本={num_pos}, 负样本={num_neg}")
            
        pos_weight = torch.tensor([num_neg / num_pos], dtype=torch.float32, device=self.device)
        
        for epoch in range(self.config['max_epochs']):
            # 训练和验证
            loss = self.train_epoch(data, pos_weight)
            val_auc = self.validate(data)
            
            # 早停检查
            if self.stopper.step(val_auc, self.model):
                print(f"Trial {trial_num}: Early stopping at epoch {epoch+1}, Best AUC: {self.stopper.best_metric:.4f}")
                break
        
        # 加载最佳模型
        if self.stopper.best_model:
            self.model.load_state_dict(self.stopper.best_model)
        
        return epoch + 1 