# model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, SAGEConv

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