# config.py
"""
Configuration Settings for Enhanced Graph Anomaly Detection
"""

# Default model configurations
DEFAULT_CONFIG = {
    'hidden_dim': 64,
    'dropout': 0.1,
    'learning_rate': 0.01,
    'weight_decay': 1e-4,
    'max_epochs': 100,
    'patience': 10,  # Reduced from 15 to 10
    'delta': 0.001,  # Increased minimum improvement threshold
    'warmup_epochs': 5,  # Reduced from 10 to 5
    'smooth_window': 3,  # Keep smoothing window
    'degree_percentile': 0.1,
    'max_candidates': 20,
    'initial_degree_percentile': 0.1,
    'adaptation_strategy': 'performance',
    'min_percentile': 0.05,
    'max_percentile': 0.3,
    'adjustment_frequency': 5,
    'adjustment_factor': 0.95
}

# Fast training configuration for quick experiments
FAST_CONFIG = {
    'hidden_dim': 64,
    'dropout': 0.1,
    'learning_rate': 0.01,
    'weight_decay': 1e-4,
    'max_epochs': 50,  # Reduced max epochs
    'patience': 7,     # More aggressive patience
    'delta': 0.002,    # Higher improvement threshold
    'warmup_epochs': 3, # Shorter warmup
    'smooth_window': 2, # Smaller smoothing window
    'degree_percentile': 0.1,
    'max_candidates': 20,
    'initial_degree_percentile': 0.1,
    'adaptation_strategy': 'performance',
    'min_percentile': 0.05,
    'max_percentile': 0.3,
    'adjustment_frequency': 3,  # More frequent adjustments
    'adjustment_factor': 0.9    # Larger adjustment steps
}

# Model information with short aliases
MODEL_INFO = {
    # Baseline models
    'gcn': {
        'name': 'Graph Convolutional Network',
        'description': 'Standard GCN baseline model',
        'full_name': 'gcn'
    },
    'gat': {
        'name': 'Graph Attention Network',
        'description': 'Attention-based graph neural network',
        'full_name': 'gat'
    },
    'graphsage': {
        'name': 'GraphSAGE',
        'description': 'Inductive representation learning on large graphs',
        'full_name': 'graphsage'
    },
    
    # Enhanced GCN models
    'ns-gcn': {
        'name': 'Neighborhood Similarity GCN',
        'description': 'Enhanced GCN with degree-aware 3-hop to 2-hop promotion',
        'full_name': 'neighborhoodsimilaritygcn'
    },
    'ans-gcn': {
        'name': 'Adaptive Neighborhood Similarity GCN',
        'description': 'Self-adaptive GCN with dynamic parameter adjustment',
        'full_name': 'adaptiveneighborhoodsimilaritygcn'
    },
    
    # Enhanced GAT models
    'ns-gat': {
        'name': 'Neighborhood Similarity GAT',
        'description': 'Enhanced GAT with degree-aware 3-hop to 2-hop promotion',
        'full_name': 'neighborhoodsimilaritygat'
    },
    'ans-gat': {
        'name': 'Adaptive Neighborhood Similarity GAT',
        'description': 'Self-adaptive GAT with dynamic parameter adjustment',
        'full_name': 'adaptiveneighborhoodsimilaritygat'
    },
    
    # Enhanced GraphSAGE models
    'ns-sage': {
        'name': 'Neighborhood Similarity GraphSAGE',
        'description': 'Enhanced GraphSAGE with degree-aware 3-hop to 2-hop promotion',
        'full_name': 'neighborhoodsimilaritygraphsage'
    },
    'ans-sage': {
        'name': 'Adaptive Neighborhood Similarity GraphSAGE',
        'description': 'Self-adaptive GraphSAGE with dynamic parameter adjustment',
        'full_name': 'adaptiveneighborhoodsimilaritygraphsage'
    }
}

# Model aliases for backward compatibility and convenience
MODEL_ALIASES = {
    # Short aliases
    'neighborhoodsimilaritygcn': 'ns-gcn',
    'adaptiveneighborhoodsimilaritygcn': 'ans-gcn',
    'neighborhoodsimilaritygat': 'ns-gat',
    'adaptiveneighborhoodsimilaritygat': 'ans-gat',
    'neighborhoodsimilaritygraphsage': 'ns-sage',
    'adaptiveneighborhoodsimilaritygraphsage': 'ans-sage',
    
    # Alternative short forms
    'nsgcn': 'ns-gcn',
    'ansgcn': 'ans-gcn',
    'nsgat': 'ns-gat',
    'ansgat': 'ans-gat',
    'nssage': 'ns-sage',
    'anssage': 'ans-sage'
}