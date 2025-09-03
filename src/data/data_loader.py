# data_loader.py
"""
Data Loading and Preprocessing Module
"""

import os
import torch
import numpy as np
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected
from sklearn.preprocessing import StandardScaler
from torch_geometric.data.storage import GlobalStorage
from torch.serialization import add_safe_globals

# Add PyG data types to safe loading list
add_safe_globals([GlobalStorage])


def prepare_dataset(name, root='datasets'):
    """
    Prepare dataset for loading
    
    Args:
        name: Dataset name
        root: Data storage path
    
    Returns:
        str: Dataset file path
    """
    name = name.lower()
    processed_path = os.path.join(root, f"{name}.pt")
    
    if not os.path.exists(processed_path):
        raise FileNotFoundError(
            f"Dataset file {processed_path} does not exist.\n"
            f"Please download datasets to {root} directory first.\n"
            "You can use download_gadbench_datasets.py script to download datasets."
        )
    
    # Validate dataset format
    try:
        data_dict = torch.load(processed_path, map_location='cpu', weights_only=True)
        if not all(k in data_dict for k in ['x', 'edge_index', 'y']):
            raise ValueError("Invalid dataset format, missing required fields")
    except Exception as e:
        raise ValueError(f"Dataset file corrupted or invalid format: {str(e)}")
    
    return processed_path


def load_pt_dataset(file_path):
    """
    Load .pt format dataset
    
    Args:
        file_path: Dataset file path
    
    Returns:
        PyG data object
    """
    data_dict = torch.load(file_path, map_location='cpu', weights_only=True)
    
    # Convert to PyG data format
    x = data_dict['x']
    edge_index = data_dict['edge_index']
    y = data_dict['y']
    
    # Ensure edges are undirected
    edge_index = to_undirected(edge_index)
    
    # Standardize features
    scaler = StandardScaler()
    x = torch.FloatTensor(scaler.fit_transform(x))
    
    return Data(x=x, edge_index=edge_index, y=y)


def load_and_split(data_path, train_ratio=0.4, seed=42, show_stats=True):
    """
    Load and split dataset
    
    Split strategy:
    - Training set: 40%
    - Validation set: 30%
    - Test set: 30%
    
    Args:
        data_path: Dataset path
        train_ratio: Training set ratio, default 0.4
        seed: Random seed
        show_stats: Whether to show dataset statistics
    
    Returns:
        PyG data object with train, validation, and test masks
    """
    try:
        # First try loading with weights_only=True
        data_dict = torch.load(data_path, map_location='cpu', weights_only=True)
    except Exception as e:
        # If failed, try full loading
        print("Warning: weights_only loading failed, trying full loading...")
        data_dict = torch.load(data_path, map_location='cpu')
    
    # Data validation
    required_keys = ['x', 'edge_index', 'y']
    assert all(k in data_dict for k in required_keys), "Dataset missing required fields"
    
    x = data_dict['x'].float()
    edge_index = data_dict['edge_index'].long()
    y = data_dict['y'].long()
    
    num_nodes = x.size(0)
    indices = torch.randperm(num_nodes, generator=torch.Generator().manual_seed(seed))
    
    # Split ratios: 40% train, 30% val, 30% test
    train_size = int(num_nodes * train_ratio)
    val_size = (num_nodes - train_size) // 2
    
    train_mask = torch.zeros(num_nodes, dtype=bool)
    val_mask = torch.zeros(num_nodes, dtype=bool)
    test_mask = torch.zeros(num_nodes, dtype=bool)
    
    train_mask[indices[:train_size]] = True
    val_mask[indices[train_size:train_size+val_size]] = True
    test_mask[indices[train_size+val_size:]] = True
    
    # Print dataset statistics
    if show_stats:
        print("\n=== Dataset Statistics ===")
        print(f"Nodes: {num_nodes}")
        print(f"Edges: {edge_index.shape[1]}")
        print(f"Feature dimension: {x.shape[1]}")
        print(f"Anomaly ratio: {y.float().mean().item():.2%}")
        print(f"Train/Val/Test samples: {train_mask.sum().item()}/{val_mask.sum().item()}/{test_mask.sum().item()}")
        print(f"Train ratio: {train_mask.sum().item()/num_nodes:.1%}")
        print(f"Val ratio: {val_mask.sum().item()/num_nodes:.1%}")
        print(f"Test ratio: {test_mask.sum().item()/num_nodes:.1%}")
    
    return Data(x=x, 
               edge_index=edge_index, 
               y=y,
               train_mask=train_mask, 
               val_mask=val_mask, 
               test_mask=test_mask)