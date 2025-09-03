# enhanced_gcn.py
"""
Enhanced GCN Models with Neighborhood Similarity
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from .base_enhanced import BaseNeighborhoodSimilarity, BaseAdaptiveNeighborhoodSimilarity


class NeighborhoodSimilarityGCN(BaseNeighborhoodSimilarity):
    """Enhanced GCN with Degree-Aware Neighborhood Similarity"""
    
    def __init__(self, in_dim, hidden_dim=128, degree_percentile=0.1, max_candidates=20, dropout=0.0):
        super().__init__(in_dim, hidden_dim, degree_percentile, max_candidates, dropout)
        
        # Two-layer GCN architecture
        self.conv1 = GCNConv(in_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, 1)
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()

    def forward(self, x, edge_index):
        enhanced_edges = self._enhance_graph_structure_degree_aware(x, edge_index, "GCN")
        
        if enhanced_edges is not None:
            enhanced_edge_index = torch.cat([edge_index, enhanced_edges], dim=1)
        else:
            enhanced_edge_index = edge_index
        
        h = self.conv1(x, enhanced_edge_index)
        h = F.relu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)
        out = self.conv2(h, enhanced_edge_index)
        
        return out.squeeze()


class AdaptiveNeighborhoodSimilarityGCN(BaseAdaptiveNeighborhoodSimilarity):
    """Adaptive GCN version with dynamic degree_percentile adjustment"""
    
    def __init__(self, in_dim, hidden_dim=128, initial_degree_percentile=0.1, max_candidates=20, 
                 dropout=0.0, adaptation_strategy='performance', min_percentile=0.05, max_percentile=0.3):
        super().__init__(in_dim, hidden_dim, initial_degree_percentile, max_candidates, 
                         dropout, adaptation_strategy, min_percentile, max_percentile)
        
        # Two-layer GCN architecture
        self.conv1 = GCNConv(in_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, 1)
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()

    def forward(self, x, edge_index):
        enhanced_edges = self._enhance_graph_structure_degree_aware(x, edge_index, "GCN")
        
        if enhanced_edges is not None:
            enhanced_edge_index = torch.cat([edge_index, enhanced_edges], dim=1)
        else:
            enhanced_edge_index = edge_index
        
        h = self.conv1(x, enhanced_edge_index)
        h = F.relu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)
        out = self.conv2(h, enhanced_edge_index)
        
        return out.squeeze()