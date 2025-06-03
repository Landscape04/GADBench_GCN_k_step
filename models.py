# model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, SAGEConv
from torch_geometric.utils import add_self_loops, degree

class GCN(nn.Module):
    def __init__(self, in_dim, hidden_dim=128, dropout=0.0):
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
    def __init__(self, nfeat, nhid, nclass, heads=8, dropout=0.0):
        super().__init__()
        self.gat1 = GATConv(nfeat, nhid, heads=heads, dropout=dropout)
        # 第二层将多头注意力的输出合并为单个向量
        self.gat2 = GATConv(nhid * heads, 1, heads=1, dropout=dropout)
        self.dropout = dropout

    def forward(self, x, edge_index):
        x = F.elu(self.gat1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.gat2(x, edge_index)
        return x.squeeze(1)  # 移除最后一个维度，使输出形状与二分类任务匹配

class GraphSAGE(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout=0.0):
        super().__init__()
        self.sage1 = SAGEConv(nfeat, nhid)
        self.sage2 = SAGEConv(nhid, 1)  # 改为输出1维
        self.dropout = dropout

    def forward(self, x, edge_index):
        x = F.relu(self.sage1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.sage2(x, edge_index)
        return x.squeeze(1)  # 确保输出维度正确

class MultiHopGCN(nn.Module):
    """基础多跳聚合模型
    
    特点：
    1. 简单的多跳邻居信息聚合
    2. 每一跳使用独立的GCN层
    3. 通过跳数控制感受野大小
    """
    def __init__(self, in_dim, hidden_dim=128, k_hops=2, dropout=0.0):
        super().__init__()
        self.k_hops = k_hops
        self.dropout = dropout
        
        # 初始特征转换
        self.input_transform = nn.Linear(in_dim, hidden_dim)
        
        # k跳GCN层
        self.hop_convs = nn.ModuleList([
            GCNConv(hidden_dim, hidden_dim) for _ in range(k_hops)
        ])
        
        # 输出层
        self.output = nn.Linear(hidden_dim * (k_hops + 1), 1)  # +1是为了包含初始特征
        
        self.reset_parameters()
        
    def reset_parameters(self):
        """初始化模型参数"""
        nn.init.kaiming_normal_(self.input_transform.weight)
        nn.init.zeros_(self.input_transform.bias)
        nn.init.kaiming_normal_(self.output.weight)
        nn.init.zeros_(self.output.bias)
        for conv in self.hop_convs:
            conv.reset_parameters()
    
    def forward(self, x, edge_index):
        # 1. 初始特征转换
        h = F.relu(self.input_transform(x))
        h = F.dropout(h, p=self.dropout, training=self.training)
        
        # 存储每一跳的特征
        all_hop_features = [h]
        
        # 2. 逐跳聚合邻居信息
        current_h = h
        for hop in range(self.k_hops):
            # 使用GCN层聚合邻居
            current_h = self.hop_convs[hop](current_h, edge_index)
            current_h = F.relu(current_h)
            current_h = F.dropout(current_h, p=self.dropout, training=self.training)
            all_hop_features.append(current_h)
        
        # 3. 拼接所有跳数的特征
        final_h = torch.cat(all_hop_features, dim=1)
        
        # 4. 输出异常分数
        out = self.output(final_h)
        return out.squeeze()

class SelectiveMultiHopGCN(nn.Module):
    """带节点筛选的多跳聚合模型
    
    特点：
    1. 继承基础多跳聚合的优点
    2. 增加节点重要性评估
    3. 基于节点重要性进行邻居筛选
    4. 自适应调整每个节点的聚合范围
    """
    def __init__(self, in_dim, hidden_dim=128, k_hops=2, dropout=0.0):
        super().__init__()
        self.k_hops = k_hops
        self.dropout = dropout
        
        # 初始特征转换
        self.input_transform = nn.Linear(in_dim, hidden_dim)
        
        # 节点重要性评估网络
        self.importance_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
        # k跳GCN层
        self.hop_convs = nn.ModuleList([
            GCNConv(hidden_dim, hidden_dim) for _ in range(k_hops)
        ])
        
        # 输出层
        self.output = nn.Linear(hidden_dim * (k_hops + 1), 1)
        
        self.reset_parameters()
        
    def reset_parameters(self):
        """初始化模型参数"""
        nn.init.kaiming_normal_(self.input_transform.weight)
        nn.init.zeros_(self.input_transform.bias)
        nn.init.kaiming_normal_(self.output.weight)
        nn.init.zeros_(self.output.bias)
        for m in self.importance_net.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        for conv in self.hop_convs:
            conv.reset_parameters()
    
    def _compute_node_importance(self, x):
        """计算节点重要性分数"""
        return self.importance_net(x)
    
    def _filter_neighbors(self, x, edge_index):
        """基于节点重要性筛选邻居"""
        # 计算所有节点的重要性分数
        importance_scores = self._compute_node_importance(x)  # [N, 1]
        
        # 获取边的源节点和目标节点
        row, col = edge_index  # [2, E]
        
        # 计算边的权重（基于源节点和目标节点的重要性）
        edge_weights = (importance_scores[row] + importance_scores[col]) / 2  # [E, 1]
        
        # 设置自适应阈值（使用均值加标准差）
        threshold = edge_weights.mean() + edge_weights.std()
        
        # 筛选重要的边
        mask = edge_weights.squeeze() > threshold  # [E]
        filtered_edge_index = edge_index[:, mask]  # [2, E']
        
        return filtered_edge_index, edge_weights[mask]

    def forward(self, x, edge_index):
        # 1. 初始特征转换
        h = F.relu(self.input_transform(x))  # [N, hidden_dim]
        h = F.dropout(h, p=self.dropout, training=self.training)
        
        # 存储每一跳的特征
        all_hop_features = [h]  # [[N, hidden_dim], ...]
        
        # 2. 基于节点重要性筛选邻居
        filtered_edge_index, _ = self._filter_neighbors(h, edge_index)  # [2, E'], [E']
        
        # 3. 逐跳聚合邻居信息
        current_h = h
        for hop in range(self.k_hops):
            # 使用筛选后的边进行消息传递
            current_h = self.hop_convs[hop](current_h, filtered_edge_index)  # [N, hidden_dim]
            current_h = F.relu(current_h)
            current_h = F.dropout(current_h, p=self.dropout, training=self.training)
            all_hop_features.append(current_h)
        
        # 4. 拼接所有跳数的特征
        final_h = torch.cat(all_hop_features, dim=1)  # [N, hidden_dim * (k_hops + 1)]
        
        # 5. 输出异常分数
        out = self.output(final_h)  # [N, 1]
        return out.squeeze()  # [N]

class NeighborhoodSimilarityGCN(nn.Module):
    """思路一：基于邻域分布相似度的远程邻居选择GCN（修正版）
    
    核心思路：
    1. 保持标准两层GCN架构不变
    2. 将选中的3跳节点加入到2跳邻域中（而非直接连中心节点）
    3. 在第二层GCN时让这些远程节点发挥作用
    
    优化：简化显示，提升性能
    """
    def __init__(self, in_dim, hidden_dim=128, top_k=2, dropout=0.0):
        super().__init__()
        
        # 保持标准两层GCN架构
        self.conv1 = GCNConv(in_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, 1)
        
        # 用于计算邻域分布的特征变换
        self.feature_transform = nn.Linear(in_dim, hidden_dim)
        
        self.top_k = top_k  # 减少选择数量
        self.dropout = dropout
        self.reset_parameters()
        
    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()
        nn.init.kaiming_normal_(self.feature_transform.weight)
        nn.init.zeros_(self.feature_transform.bias)
    
    def _build_adjacency_dict(self, edge_index, num_nodes):
        """构建邻接字典"""
        adj = [set() for _ in range(num_nodes)]
        for src, dst in edge_index.T:
            adj[src.item()].add(dst.item())
            adj[dst.item()].add(src.item())
        return adj
    
    def _get_2hop_neighbors(self, node_id, adj):
        """获取2跳邻居"""
        one_hop = adj[node_id]
        two_hop = set()
        for neighbor in one_hop:
            two_hop.update(adj[neighbor])
        two_hop -= one_hop  # 移除1跳邻居
        two_hop.discard(node_id)  # 移除自己
        return two_hop
    
    def _get_3hop_neighbors_fast(self, node_id, adj, max_candidates=15):
        """快速获取3跳邻居（限制数量）"""
        one_hop = adj[node_id]
        two_hop = set()
        for neighbor in one_hop:
            two_hop.update(adj[neighbor])
        two_hop -= one_hop
        two_hop.discard(node_id)
        
        three_hop = set()
        for neighbor in two_hop:
            three_hop.update(adj[neighbor])
            if len(three_hop) > max_candidates:
                break
        three_hop -= two_hop
        three_hop -= one_hop
        three_hop.discard(node_id)
        
        # 限制候选数量
        if len(three_hop) > max_candidates:
            three_hop = set(list(three_hop)[:max_candidates])
        
        return three_hop
    
    def _compute_node_similarity_fast(self, center_node, remote_node, h, adj):
        """快速计算节点相似度"""
        # 主要使用特征相似度，简化计算
        center_features = h[center_node]
        remote_features = h[remote_node]
        similarity = F.cosine_similarity(
            center_features.unsqueeze(0), 
            remote_features.unsqueeze(0)
        ).item()
        return similarity
    
    def _enhance_graph_structure(self, x, edge_index):
        """核心思路：将与中心节点最相似的3跳节点连接到1跳节点（升级为2跳节点）"""
        num_nodes = x.size(0)
        h = torch.relu(self.conv1.lin(x))  # 预计算特征
        adj = self._build_adjacency_dict(edge_index, num_nodes)
        
        enhanced_edges = []
        
        # 进一步减少采样数量以提升速度
        sample_size = min(8, int(num_nodes * 0.01))  # 从15减少到8，比例从0.02降到0.01
        sampled_nodes = torch.randperm(num_nodes)[:sample_size]
        
        for center_node in sampled_nodes:
            center_node = center_node.item()
            
            # 获取1跳和3跳邻居
            one_hop_neighbors = adj[center_node]
            three_hop_neighbors = self._get_3hop_neighbors_fast(center_node, adj, max_candidates=20)  # 增加候选数量
            
            if len(one_hop_neighbors) == 0 or len(three_hop_neighbors) == 0:
                continue
            
            # 计算3跳邻居与中心节点的相似度
            three_hop_list = list(three_hop_neighbors)
            
            # 批量计算相似度，提升效率
            center_h = h[center_node].unsqueeze(0)
            three_hop_h = h[three_hop_list]
            
            # 使用余弦相似度，更高效
            similarities = F.cosine_similarity(center_h, three_hop_h, dim=1)
            
            # 选择最相似的top-k个3跳邻居
            if len(similarities) > 0:
                num_select = min(self.top_k, len(three_hop_list))
                _, top_indices = torch.topk(similarities, k=num_select)
                
                # 核心思路：将选中的3跳节点连接到1跳邻居（升级为2跳节点）
                selected_3hop = [three_hop_list[idx] for idx in top_indices]
                one_hop_list = list(one_hop_neighbors)
                
                # 为每个选中的3跳节点随机选择一个1跳邻居进行连接
                for remote_3hop in selected_3hop:
                    if len(one_hop_list) > 0:
                        # 随机选择一个1跳邻居作为桥梁
                        bridge_1hop = one_hop_list[torch.randint(0, len(one_hop_list), (1,)).item()]
                        enhanced_edges.extend([
                            [remote_3hop, bridge_1hop],
                            [bridge_1hop, remote_3hop]
                        ])
        
        if enhanced_edges:
            return torch.tensor(enhanced_edges, dtype=torch.long).T
        else:
            return None

    def forward(self, x, edge_index):
        # 增强图结构（核心思路：3跳→2跳升级）
        remote_edges = self._enhance_graph_structure(x, edge_index)
        
        # 构建增强图
        if remote_edges is not None:
            enhanced_edge_index = torch.cat([edge_index, remote_edges], dim=1)
            # 简化输出显示
            if not hasattr(self, '_printed_once'):
                print(f"增强图结构：添加了 {remote_edges.size(1)//2} 条3跳→2跳连接")
                self._printed_once = True
        else:
            enhanced_edge_index = edge_index
        
        # 标准的两层GCN
        h = self.conv1(x, enhanced_edge_index)
        h = F.relu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)
        out = self.conv2(h, enhanced_edge_index)
        
        return out.squeeze()

class LearnableSelectionGCN(nn.Module):
    """思路二：可学习的远程邻居筛选网络GCN（修正版）
    
    核心思路：
    1. 保持标准两层GCN架构不变
    2. 通过可学习网络评估3跳节点的重要性
    3. 将重要的3跳节点连接到2跳邻域中
    
    优化：简化显示，提升性能
    """
    def __init__(self, in_dim, hidden_dim=128, top_k=3, dropout=0.0):
        super().__init__()
        
        # 保持标准两层GCN架构
        self.conv1 = GCNConv(in_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, 1)
        
        # 特征变换用于重要性计算
        self.feature_transform = nn.Linear(in_dim, hidden_dim)
        
        # 可学习的远程邻居重要性评估网络（简化版）
        self.importance_evaluator = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        
        self.top_k = top_k
        self.dropout = dropout
        self.reset_parameters()
        
    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()
        nn.init.kaiming_normal_(self.feature_transform.weight)
        nn.init.zeros_(self.feature_transform.bias)
        
        for m in self.importance_evaluator.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                nn.init.zeros_(m.bias)
    
    def _build_adjacency_dict(self, edge_index, num_nodes):
        """构建邻接字典"""
        adj = [set() for _ in range(num_nodes)]
        for src, dst in edge_index.T:
            adj[src.item()].add(dst.item())
            adj[dst.item()].add(src.item())
        return adj
    
    def _get_2hop_neighbors(self, node_id, adj):
        """获取2跳邻居"""
        one_hop = adj[node_id]
        two_hop = set()
        for neighbor in one_hop:
            two_hop.update(adj[neighbor])
        two_hop -= one_hop
        two_hop.discard(node_id)
        return two_hop
    
    def _get_3hop_neighbors_fast(self, node_id, adj, max_candidates=10):
        """快速获取3跳邻居（限制数量）"""
        one_hop = adj[node_id]
        two_hop = set()
        for neighbor in one_hop:
            two_hop.update(adj[neighbor])
        two_hop -= one_hop
        two_hop.discard(node_id)
        
        three_hop = set()
        for neighbor in two_hop:
            three_hop.update(adj[neighbor])
            if len(three_hop) > max_candidates:
                break
        three_hop -= two_hop
        three_hop -= one_hop
        three_hop.discard(node_id)
        
        if len(three_hop) > max_candidates:
            three_hop = set(list(three_hop)[:max_candidates])
        
        return three_hop
    
    def _learn_remote_connections(self, x, edge_index):
        """核心思路：通过可学习网络选择与中心节点最重要的3跳节点，连接到1跳节点（升级为2跳节点）"""
        num_nodes = x.size(0)
        h = torch.relu(self.feature_transform(x))
        adj = self._build_adjacency_dict(edge_index, num_nodes)
        
        enhanced_edges = []
        
        # 进一步减少采样数量以提升速度
        sample_size = min(8, int(num_nodes * 0.01))  # 从15减少到8，比例从0.02降到0.01
        sampled_nodes = torch.randperm(num_nodes)[:sample_size]
        
        for center_node in sampled_nodes:
            center_node = center_node.item()
            
            # 获取1跳和3跳邻居
            one_hop_neighbors = adj[center_node]
            three_hop_neighbors = self._get_3hop_neighbors_fast(center_node, adj, max_candidates=20)  # 增加候选数量
            
            if len(one_hop_neighbors) == 0 or len(three_hop_neighbors) == 0:
                continue
            
            # 使用可学习网络评估3跳邻居的重要性
            center_features = h[center_node].unsqueeze(0)
            three_hop_list = list(three_hop_neighbors)
            three_hop_features = h[three_hop_list]
            
            # 构造输入特征
            combined_features = torch.cat([
                center_features.expand(len(three_hop_list), -1),
                three_hop_features
            ], dim=-1)
            
            importance_scores = self.importance_evaluator(combined_features).squeeze(-1)  # 只squeeze最后一维
            
            # 确保importance_scores至少是1维的
            if importance_scores.dim() == 0:
                importance_scores = importance_scores.unsqueeze(0)
            
            # 选择Top-K个重要的3跳邻居
            num_select = min(self.top_k, len(three_hop_list))
            if len(three_hop_list) == 1:
                # 如果只有一个候选，直接选择
                top_indices = torch.tensor([0])
            else:
                _, top_indices = torch.topk(importance_scores, k=num_select)
            
            # 核心思路：将选中的3跳节点连接到1跳邻居（升级为2跳节点）
            selected_3hop = [three_hop_list[idx] for idx in top_indices]
            one_hop_list = list(one_hop_neighbors)
            
            # 为每个选中的3跳节点随机选择一个1跳邻居进行连接
            for remote_3hop in selected_3hop:
                if len(one_hop_list) > 0:
                    # 随机选择一个1跳邻居作为桥梁
                    bridge_1hop = one_hop_list[torch.randint(0, len(one_hop_list), (1,)).item()]
                    enhanced_edges.extend([
                        [remote_3hop, bridge_1hop],
                        [bridge_1hop, remote_3hop]
                    ])
        
        if enhanced_edges:
            return torch.tensor(enhanced_edges, dtype=torch.long).T
        else:
            return None

    def forward(self, x, edge_index):
        """前向传播：可学习筛选思路实现（修正版）"""
        num_nodes = x.size(0)
        
        # 核心思路：学习3跳→2跳升级
        remote_edges = self._learn_remote_connections(x, edge_index)
        
        # 构建增强图
        if remote_edges is not None:
            enhanced_edge_index = torch.cat([edge_index, remote_edges], dim=1)
            # 简化输出显示
            if not hasattr(self, '_printed_once'):
                print(f"可学习增强：添加了 {remote_edges.size(1)//2} 条3跳→2跳连接")
                self._printed_once = True
        else:
            enhanced_edge_index = edge_index
        
        # 标准的两层GCN
        h = self.conv1(x, enhanced_edge_index)
        h = F.relu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)
        out = self.conv2(h, enhanced_edge_index)
        
        return out.squeeze()