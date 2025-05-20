"""
训练器模块 - 提供模型训练、验证和测试的完整功能

主要组件:
1. EarlyStopper - 早停机制，避免过拟合
2. Trainer - 模型训练器，封装完整训练流程

特点:
- 支持早停机制
- 自动处理类别不平衡
- 梯度裁剪和NaN检测
- 模型状态保存和恢复
"""

import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
import copy
import time

class EarlyStopper:
    """早停机制

    特点：
    1. 支持模型状态保存和恢复
    2. 平滑指标计算，减少波动影响
    3. 支持预热期，避免训练初期不稳定阶段触发早停
    """
    def __init__(self, initial_model=None, patience=20, delta=0.00005,
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
            k: v.cpu().clone().detach()  # 确保数据不在计算图中
            for k, v in model.state_dict().items()
        }

    def step(self, current_metric, model=None):
        """评估当前指标并决定是否应该早停

        Args:
            current_metric: 当前评估指标值
            model: 当前模型状态，如果需要保存

        Returns:
            bool: 是否应该停止训练
        """
        # 首次运行确保模型已保存
        if not self.best_model and model is not None:
            self._save_model(model)

        self.metric_history.append(current_metric)

        # 计算平滑指标
        smoothed = current_metric
        if len(self.metric_history) >= self.smooth_window:
            smoothed = sum(self.metric_history[-self.smooth_window:])/self.smooth_window

        # 更新最佳指标逻辑
        if (len(self.metric_history) >= self.warmup) and (smoothed > self.best_metric + self.delta):
            self.best_metric = smoothed
            self.counter = 0
            if model is not None:
                self._save_model(model)
        else:
            self.counter += 1

        # Warmup阶段不触发
        if len(self.metric_history) < self.warmup:
            return False

        return self.counter >= self.patience

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

    def train_epoch(self, data, pos_weight, epoch=0):
        """训练一个epoch

        Args:
            data: 训练数据
            pos_weight: 正样本权重
            epoch: 当前epoch数，用于日志

        Returns:
            float: 训练损失
            float: 训练准确率
        """
        self.model.train()
        self.optimizer.zero_grad()

        out = self.model(data.x, data.edge_index)
        loss = F.binary_cross_entropy_with_logits(
            out[data.train_mask],
            data.y[data.train_mask].float(),
            pos_weight=pos_weight
        )

        # 梯度检查
        if torch.isnan(loss).any():
            raise RuntimeError(f"Epoch {epoch}: 损失值出现NaN")

        loss.backward()

        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)

        self.optimizer.step()

        # 计算训练准确率
        with torch.no_grad():
            pred = (torch.sigmoid(out) > 0.5).long()
            correct = (pred[data.train_mask] == data.y[data.train_mask]).sum()
            acc = correct.float() / data.train_mask.sum()

        return loss.item(), acc.item()

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
        """完整训练流程

        Args:
            data: 训练数据
            trial_num: 当前实验编号

        Returns:
            int: 训练的总epoch数
        """
        # 计算类别权重
        train_labels = data.y[data.train_mask]
        num_pos = train_labels.sum().item()
        num_neg = len(train_labels) - num_pos

        if num_pos == 0 or num_neg == 0:
            raise ValueError(f"类别不平衡异常: 正样本={num_pos}, 负样本={num_neg}")

        pos_weight = torch.tensor([num_neg / num_pos], dtype=torch.float32, device=self.device)

        for epoch in range(self.config['max_epochs']):
            # 训练和验证
            loss, acc = self.train_epoch(data, pos_weight, epoch)
            val_auc = self.validate(data)

            # 早停检查
            if self.stopper.step(val_auc, self.model):
                break

        # 加载最佳模型
        if self.stopper.best_model:
            self.model.load_state_dict(self.stopper.best_model)

        return epoch + 1