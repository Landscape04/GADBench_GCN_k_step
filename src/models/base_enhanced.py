# base_enhanced.py
"""
Base classes and shared functionality for enhanced models
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class BaseNeighborhoodSimilarity(nn.Module):
    """
    Base class for Neighborhood Similarity enhancement
    """
    
    def __init__(self, in_dim, hidden_dim=128, degree_percentile=0.1, max_candidates=20, dropout=0.0):
        super().__init__()
        
        # Feature transformation for neighborhood distribution computation
        self.feature_transform = nn.Linear(in_dim, hidden_dim)
        
        self.degree_percentile = degree_percentile
        self.max_candidates = max_candidates
        self.dropout = dropout
        
        nn.init.kaiming_normal_(self.feature_transform.weight)
        nn.init.zeros_(self.feature_transform.bias)
    
    def _build_adjacency_dict(self, edge_index, num_nodes):
        """Build adjacency dictionary for fast neighbor lookup"""
        adj = [set() for _ in range(num_nodes)]
        for src, dst in edge_index.T:
            adj[src.item()].add(dst.item())
            adj[dst.item()].add(src.item())
        return adj
    
    def _get_3hop_neighbors_fast(self, node_id, adj):
        """Fast 3-hop neighbor discovery"""
        one_hop = adj[node_id]
        two_hop = set()
        for neighbor in one_hop:
            two_hop.update(adj[neighbor])
        two_hop -= one_hop
        two_hop.discard(node_id)
        
        three_hop = set()
        for neighbor in two_hop:
            three_hop.update(adj[neighbor])
            if len(three_hop) > self.max_candidates:
                break
        three_hop -= two_hop
        three_hop -= one_hop
        three_hop.discard(node_id)
        
        if len(three_hop) > self.max_candidates:
            three_hop = set(list(three_hop)[:self.max_candidates])
        
        return three_hop
    
    def _compute_degree_based_neighbors(self, node_id, adj):
        """Compute number of neighbors to add based on node degree percentile"""
        degree = len(adj[node_id])
        num_neighbors_to_add = max(1, int(degree * self.degree_percentile))
        return num_neighbors_to_add
    
    def _enhance_graph_structure_degree_aware(self, x, edge_index, model_name=""):
        """Degree-aware all-node processing for graph enhancement"""
        num_nodes = x.size(0)
        h = torch.relu(self.feature_transform(x))
        adj = self._build_adjacency_dict(edge_index, num_nodes)
        
        enhanced_edges = []
        total_connections_added = 0
        
        for center_node in range(num_nodes):
            one_hop_neighbors = adj[center_node]
            three_hop_neighbors = self._get_3hop_neighbors_fast(center_node, adj)
            
            if len(one_hop_neighbors) == 0 or len(three_hop_neighbors) == 0:
                continue
            
            num_neighbors_to_add = self._compute_degree_based_neighbors(center_node, adj)
            three_hop_list = list(three_hop_neighbors)
            
            center_h = h[center_node].unsqueeze(0)
            three_hop_h = h[three_hop_list]
            similarities = F.cosine_similarity(center_h, three_hop_h, dim=1)
            
            if len(similarities) > 0:
                num_select = min(num_neighbors_to_add, len(three_hop_list))
                _, top_indices = torch.topk(similarities, k=num_select)
                
                selected_3hop = [three_hop_list[idx] for idx in top_indices]
                one_hop_list = list(one_hop_neighbors)
                
                for remote_3hop in selected_3hop:
                    if len(one_hop_list) > 0:
                        bridge_1hop = one_hop_list[torch.randint(0, len(one_hop_list), (1,)).item()]
                        enhanced_edges.extend([
                            [remote_3hop, bridge_1hop],
                            [bridge_1hop, remote_3hop]
                        ])
                        total_connections_added += 1
        
        if enhanced_edges:
            enhanced_edge_tensor = torch.tensor(enhanced_edges, dtype=torch.long, device=x.device).T
            
            if not hasattr(self, '_printed_once'):
                print(f"{model_name} enhancement: nodes={num_nodes}, connections_added={total_connections_added}, degree_percentile={self.degree_percentile}")
                self._printed_once = True
            
            return enhanced_edge_tensor
        else:
            return None


class BaseAdaptiveNeighborhoodSimilarity(BaseNeighborhoodSimilarity):
    """
    Base class for Adaptive Neighborhood Similarity enhancement
    """
    
    def __init__(self, in_dim, hidden_dim=128, initial_degree_percentile=0.1, max_candidates=20, 
                 dropout=0.0, adaptation_strategy='performance', min_percentile=0.05, max_percentile=0.3):
        super().__init__(in_dim, hidden_dim, initial_degree_percentile, max_candidates, dropout)
        
        # Learnable degree_percentile parameter
        self.degree_percentile = nn.Parameter(torch.tensor(initial_degree_percentile, dtype=torch.float32))
        
        # Adaptive parameters
        self.adaptation_strategy = adaptation_strategy
        self.min_percentile = min_percentile
        self.max_percentile = max_percentile
        
        # Performance tracking
        self.performance_history = []
        self.epoch_count = 0
        self.last_adjustment_epoch = 0
        self.adjustment_frequency = 5
        self.performance_patience = 3
        self.adjustment_factor = 0.95
        self.stability_threshold = 0.001
    
    def get_current_degree_percentile(self):
        return torch.clamp(self.degree_percentile, self.min_percentile, self.max_percentile).item()
    
    def _compute_degree_based_neighbors(self, node_id, adj):
        """Override to use dynamic degree percentile"""
        degree = len(adj[node_id])
        current_percentile = self.get_current_degree_percentile()
        num_neighbors_to_add = max(1, int(degree * current_percentile))
        return num_neighbors_to_add
    
    def adaptive_adjust_degree_percentile(self, current_performance=None):
        """Adaptively adjust degree_percentile parameter"""
        if current_performance is not None:
            self.performance_history.append(float(current_performance))
            self.epoch_count += 1
            if len(self.performance_history) > 20:
                self.performance_history = self.performance_history[-20:]
        
        if not self._should_adjust_parameter():
            return
        
        if self.adaptation_strategy == 'performance':
            self._performance_driven_adjustment()
        
        self.last_adjustment_epoch = self.epoch_count
    
    def _should_adjust_parameter(self):
        if len(self.performance_history) < self.performance_patience:
            return False
        if (self.epoch_count - self.last_adjustment_epoch) < self.adjustment_frequency:
            return False
        return True
    
    def _performance_driven_adjustment(self):
        if len(self.performance_history) < 6:
            return
            
        recent_performance = np.mean(self.performance_history[-3:])
        previous_performance = np.mean(self.performance_history[-6:-3])
        performance_trend = recent_performance - previous_performance
        
        if abs(performance_trend) < self.stability_threshold:
            adjustment = 1.02 if len(self.performance_history) % 2 == 0 else 0.98
            with torch.no_grad():
                self.degree_percentile.data *= adjustment
        elif performance_trend < -self.stability_threshold:
            with torch.no_grad():
                self.degree_percentile.data *= self.adjustment_factor
    
    def set_adaptation_config(self, strategy='performance', adjustment_frequency=5, 
                            adjustment_factor=0.95, print_adjustments=False):
        self.adaptation_strategy = strategy
        self.adjustment_frequency = adjustment_frequency
        self.adjustment_factor = adjustment_factor
        self._print_adjustments = print_adjustments
    
    def _enhance_graph_structure_degree_aware(self, x, edge_index, model_name=""):
        """Override to use dynamic percentile and add logging"""
        num_nodes = x.size(0)
        h = torch.relu(self.feature_transform(x))
        adj = self._build_adjacency_dict(edge_index, num_nodes)
        
        enhanced_edges = []
        total_connections_added = 0
        current_percentile = self.get_current_degree_percentile()
        
        for center_node in range(num_nodes):
            one_hop_neighbors = adj[center_node]
            three_hop_neighbors = self._get_3hop_neighbors_fast(center_node, adj)
            
            if len(one_hop_neighbors) == 0 or len(three_hop_neighbors) == 0:
                continue
            
            num_neighbors_to_add = self._compute_degree_based_neighbors(center_node, adj)
            three_hop_list = list(three_hop_neighbors)
            center_h = h[center_node].unsqueeze(0)
            three_hop_h = h[three_hop_list]
            similarities = F.cosine_similarity(center_h, three_hop_h, dim=1)
            
            if len(similarities) > 0:
                num_select = min(num_neighbors_to_add, len(three_hop_list))
                _, top_indices = torch.topk(similarities, k=num_select)
                
                selected_3hop = [three_hop_list[idx] for idx in top_indices]
                one_hop_list = list(one_hop_neighbors)
                
                for remote_3hop in selected_3hop:
                    if len(one_hop_list) > 0:
                        bridge_1hop = one_hop_list[torch.randint(0, len(one_hop_list), (1,)).item()]
                        enhanced_edges.extend([
                            [remote_3hop, bridge_1hop],
                            [bridge_1hop, remote_3hop]
                        ])
                        total_connections_added += 1
        
        if enhanced_edges:
            enhanced_edge_tensor = torch.tensor(enhanced_edges, dtype=torch.long, device=x.device).T
            
            if not hasattr(self, '_last_printed_percentile'):
                self._last_printed_percentile = current_percentile
                print(f"Adaptive {model_name} enhancement: nodes={num_nodes}, connections_added={total_connections_added}, "
                      f"dynamic_degree_percentile={current_percentile:.4f}")
            elif abs(current_percentile - self._last_printed_percentile) > 0.01:
                self._last_printed_percentile = current_percentile
                print(f"{model_name} Parameter adjustment: dynamic_degree_percentile={current_percentile:.4f}")
            
            return enhanced_edge_tensor
        else:
            return None
    
    def get_adaptation_stats(self):
        """Get current adaptation statistics"""
        return {
            'current_degree_percentile': self.get_current_degree_percentile(),
            'epoch_count': self.epoch_count,
            'last_adjustment_epoch': self.last_adjustment_epoch,
            'performance_history_length': len(self.performance_history),
            'recent_performance': np.mean(self.performance_history[-3:]) if len(self.performance_history) >= 3 else 0.0,
            'adaptation_strategy': self.adaptation_strategy,
            'adjustment_frequency': self.adjustment_frequency
        }