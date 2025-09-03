# enhanced_gat.py
"""
Enhanced GAT Models with Neighborhood Similarity
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from .base_enhanced import BaseNeighborhoodSimilarity, BaseAdaptiveNeighborhoodSimilarity


class NeighborhoodSimilarityGAT(BaseNeighborhoodSimilarity):
    """Enhanced GAT with Degree-Aware Neighborhood Similarity"""
    
    def __init__(self, in_dim, hidden_dim=128, heads=8, degree_percentile=0.1, max_candidates=20, dropout=0.0):
        super().__init__(in_dim, hidden_dim, degree_percentile, max_candidates, dropout)
        
        # Two-layer GAT architecture
        self.gat1 = GATConv(in_dim, hidden_dim, heads=heads, dropout=dropout)
        self.gat2 = GATConv(hidden_dim * heads, 1, heads=1, dropout=dropout)

    def forward(self, x, edge_index):
        enhanced_edges = self._enhance_graph_structure_degree_aware(x, edge_index, "GAT")
        
        if enhanced_edges is not None:
            enhanced_edge_index = torch.cat([edge_index, enhanced_edges], dim=1)
        else:
            enhanced_edge_index = edge_index
        
        h = F.elu(self.gat1(x, enhanced_edge_index))
        h = F.dropout(h, p=self.dropout, training=self.training)
        out = self.gat2(h, enhanced_edge_index)
        
        return out.squeeze()


class AdaptiveNeighborhoodSimilarityGAT(BaseAdaptiveNeighborhoodSimilarity):
    """Adaptive GAT version with dynamic degree_percentile adjustment"""
    
    def __init__(self, in_dim, hidden_dim=128, heads=8, initial_degree_percentile=0.1, max_candidates=20, 
                 dropout=0.0, adaptation_strategy='performance', min_percentile=0.05, max_percentile=0.3):
        super().__init__(in_dim, hidden_dim, initial_degree_percentile, max_candidates, 
                         dropout, adaptation_strategy, min_percentile, max_percentile)
        
        # Two-layer GAT architecture
        self.gat1 = GATConv(in_dim, hidden_dim, heads=heads, dropout=dropout)
        self.gat2 = GATConv(hidden_dim * heads, 1, heads=1, dropout=dropout)

    def forward(self, x, edge_index):
        enhanced_edges = self._enhance_graph_structure_degree_aware(x, edge_index, "GAT")
        
        if enhanced_edges is not None:
            enhanced_edge_index = torch.cat([edge_index, enhanced_edges], dim=1)
        else:
            enhanced_edge_index = edge_index
        
        h = F.elu(self.gat1(x, enhanced_edge_index))
        h = F.dropout(h, p=self.dropout, training=self.training)
        out = self.gat2(h, enhanced_edge_index)
        
        return out.squeeze()