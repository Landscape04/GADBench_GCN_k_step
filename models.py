# model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, SAGEConv
from torch_geometric.utils import add_self_loops, degree

class GCN(nn.Module):
    def __init__(self, in_dim, hidden_dim=128, dropout=0.3):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, 1)
        self.dropout = dropout
        # 参数初始化
        self.reset_parameters()
        
    def reset_parameters(self):
        nn.init.kaiming_normal_(self.conv1.lin.weight)
        nn.init.kaiming_normal_(self.conv2.lin.weight)
        nn.init.constant_(self.conv1.bias, 0)
        nn.init.constant_(self.conv2.bias, 0)
        
    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return x.squeeze()
    
class GAT(nn.Module):
    def __init__(self, nfeat, nhid, nclass, heads=8, dropout=0.6):
        super().__init__()
        self.gat1 = GATConv(nfeat, nhid, heads=heads)
        # 第二层将多头注意力的输出合并为单个向量
        self.gat2 = GATConv(nhid * heads, 1, heads=1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index):
        x = self.dropout(x)
        x = F.elu(self.gat1(x, edge_index))
        x = self.dropout(x)
        x = self.gat2(x, edge_index)
        return x.squeeze(1)  # 移除最后一个维度，使输出形状与二分类任务匹配

class GraphSAGE(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout=0.5):
        super().__init__()
        self.sage1 = SAGEConv(nfeat, nhid)
        self.sage2 = SAGEConv(nhid, 1)  # 改为输出1维
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index):
        x = self.dropout(x)
        x = F.relu(self.sage1(x, edge_index))
        x = self.dropout(x)
        x = self.sage2(x, edge_index)
        return x.squeeze(1)  # 确保输出维度正确

class AnomalyGCN(nn.Module):
    def __init__(self, in_dim, hidden_dim=128, k_steps=2, dropout=0.3):
        """异常感知的GCN模型
        
        特点：
        1. 自适应邻居筛选：基于节点特征相似度筛选可信邻居
        2. 双重信息流：分别维护原始特征和转换特征，避免信息过度稀释
        3. 残差连接：保持节点的原始特征，防止异常信号被完全抹除
        
        Args:
            in_dim: 输入特征维度
            hidden_dim: 隐藏层维度
            k_steps: 考虑的邻居步数
            dropout: Dropout率
        """
        super().__init__()
        self.k_steps = k_steps
        
        # 特征转换层
        self.feature_transform = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # 邻居相似度计算层
        self.similarity_net = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        
        # 每一步的GCN层
        self.step_convs = nn.ModuleList([
            GCNConv(hidden_dim, hidden_dim) for _ in range(k_steps)
        ])
        
        # 异常检测层
        self.anomaly_detector = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),  # 原始特征 + 转换特征 + 邻居特征
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )
        
        self.dropout = dropout
        self.reset_parameters()
        
    def reset_parameters(self):
        """初始化模型参数"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, GCNConv):
                m.reset_parameters()
    
    def _compute_neighbor_similarity(self, h_i, h_j):
        """计算节点对之间的相似度
        
        Args:
            h_i: 源节点特征
            h_j: 目标节点特征
            
        Returns:
            节点对的相似度分数 (0~1)
        """
        # 拼接特征
        pair_feat = torch.cat([h_i, h_j], dim=-1)
        # 计算相似度分数
        sim_score = self.similarity_net(pair_feat)
        return sim_score
    
    def _filter_neighbors(self, x, edge_index):
        """基于节点相似度筛选可信邻居
        
        Args:
            x: 节点特征
            edge_index: 边索引
            
        Returns:
            筛选后的边索引和对应的权重
        """
        row, col = edge_index
        
        # 计算节点对相似度
        h_row = x[row]  # 源节点特征
        h_col = x[col]  # 目标节点特征
        edge_weights = self._compute_neighbor_similarity(h_row, h_col)
        
        # 设置相似度阈值（可以是固定值或自适应值）
        threshold = edge_weights.mean() + edge_weights.std()
        
        # 筛选高相似度的边
        mask = edge_weights.squeeze() > threshold
        filtered_edge_index = edge_index[:, mask]
        filtered_weights = edge_weights[mask]
        
        return filtered_edge_index, filtered_weights
    
    def forward(self, x, edge_index):
        """前向传播
        
        Args:
            x: 节点特征
            edge_index: 边索引
            
        Returns:
            节点的异常分数
        """
        # 1. 特征转换
        h_transform = self.feature_transform(x)
        
        # 2. 基于节点相似度筛选邻居
        filtered_edge_index, edge_weights = self._filter_neighbors(h_transform, edge_index)
        
        # 3. 迭代聚合邻居信息
        h_neighbor = h_transform
        for step in range(self.k_steps):
            # 添加自环
            cur_edge_index, _ = add_self_loops(filtered_edge_index)
            
            # 计算归一化系数
            row, col = cur_edge_index
            deg = degree(row, x.size(0), dtype=x.dtype)
            deg_inv_sqrt = deg.pow(-0.5)
            deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
            norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
            
            # 聚合邻居信息
            h_neighbor = self.step_convs[step](h_neighbor, cur_edge_index)
            h_neighbor = F.relu(h_neighbor)
            h_neighbor = F.dropout(h_neighbor, p=self.dropout, training=self.training)
        
        # 4. 特征融合
        # 将原始特征、转换特征和邻居特征拼接
        final_h = torch.cat([
            self.feature_transform(x),  # 原始特征的转换
            h_transform,  # 初始转换特征
            h_neighbor    # 邻居聚合特征
        ], dim=1)
        
        # 5. 异常检测
        anomaly_scores = self.anomaly_detector(final_h)
        
        return anomaly_scores.squeeze()