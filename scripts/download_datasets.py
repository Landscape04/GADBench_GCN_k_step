#!/usr/bin/env python3
"""
Dataset Download Script

Support downloading tolokers and questions datasets from DGL, following GADBench preprocessing methods
"""

import os
import sys
import torch
import numpy as np
from sklearn.preprocessing import StandardScaler
import json

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

try:
    import dgl
except ImportError:
    print("Error: DGL not installed. Please install with: pip install dgl")
    sys.exit(1)


def download_tolokers_dataset(save_dir='datasets'):
    """Download and preprocess Tolokers dataset"""
    print("Downloading Tolokers dataset...")
    
    try:
        # Download dataset from DGL
        dataset = dgl.data.TolokersDataset()
        graph = dataset[0]
        
        # Get node features and labels
        node_features = graph.ndata['feat'].numpy()
        labels = graph.ndata['label'].numpy()
        
        # Get edge index
        src, dst = graph.edges()
        edge_index = torch.stack([src, dst], dim=0)
        
        # Standardize features
        scaler = StandardScaler()
        node_features = scaler.fit_transform(node_features)
        
        # Convert to torch format
        x = torch.FloatTensor(node_features)
        y = torch.LongTensor(labels)
        
        # Save dataset
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, 'tolokers.pt')
        torch.save({
            'x': x,
            'edge_index': edge_index,
            'y': y,
            'num_nodes': x.size(0),
            'num_edges': edge_index.size(1),
            'num_features': x.size(1),
            'num_classes': len(torch.unique(y))
        }, save_path)
        
        print(f"Tolokers dataset download completed:")
        print(f"  Nodes: {x.size(0)}")
        print(f"  Edges: {edge_index.size(1)}")
        print(f"  Feature dimension: {x.size(1)}")
        print(f"  Anomaly ratio: {y.float().mean().item():.2%}")
        print(f"  Saved to: {save_path}")
        
        return save_path
        
    except Exception as e:
        print(f"Failed to download Tolokers dataset: {str(e)}")
        return None


def download_questions_dataset(save_dir='datasets'):
    """Download and preprocess Questions dataset"""
    print("Downloading Questions dataset...")
    
    try:
        # Download dataset from DGL
        dataset = dgl.data.QuestionsDataset()
        graph = dataset[0]
        
        # Get node features and labels
        node_features = graph.ndata['feat'].numpy()
        labels = graph.ndata['label'].numpy()
        
        # Get edge index
        src, dst = graph.edges()
        edge_index = torch.stack([src, dst], dim=0)
        
        # Standardize features
        scaler = StandardScaler()
        node_features = scaler.fit_transform(node_features)
        
        # Convert to torch format
        x = torch.FloatTensor(node_features)
        y = torch.LongTensor(labels)
        
        # Save dataset
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, 'questions.pt')
        torch.save({
            'x': x,
            'edge_index': edge_index,
            'y': y,
            'num_nodes': x.size(0),
            'num_edges': edge_index.size(1),
            'num_features': x.size(1),
            'num_classes': len(torch.unique(y))
        }, save_path)
        
        print(f"Questions dataset download completed:")
        print(f"  Nodes: {x.size(0)}")
        print(f"  Edges: {edge_index.size(1)}")
        print(f"  Feature dimension: {x.size(1)}")
        print(f"  Anomaly ratio: {y.float().mean().item():.2%}")
        print(f"  Saved to: {save_path}")
        
        return save_path
        
    except Exception as e:
        print(f"Failed to download Questions dataset: {str(e)}")
        return None


def check_dataset_exists(dataset_name, save_dir='datasets'):
    """Check if dataset already exists"""
    dataset_path = os.path.join(save_dir, f'{dataset_name}.pt')
    return os.path.exists(dataset_path)


def load_or_download_dataset(dataset_name, save_dir='datasets'):
    """Load dataset, download if not exists"""
    if check_dataset_exists(dataset_name, save_dir):
        print(f"{dataset_name} dataset already exists, skipping download")
        return os.path.join(save_dir, f'{dataset_name}.pt')
    
    if dataset_name == 'tolokers':
        return download_tolokers_dataset(save_dir)
    elif dataset_name == 'questions':
        return download_questions_dataset(save_dir)
    else:
        print(f"Unsupported dataset: {dataset_name}")
        return None


def update_download_results(dataset_name, success, save_dir='datasets'):
    """Update download results record"""
    results_file = os.path.join(save_dir, 'download_results.json')
    
    # Read existing results
    if os.path.exists(results_file):
        with open(results_file, 'r') as f:
            results = json.load(f)
    else:
        results = {}
    
    # Update results
    results[dataset_name] = {
        'success': success,
        'path': os.path.join(save_dir, f'{dataset_name}.pt') if success else None
    }
    
    # Save results
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)


def main():
    """Main function"""
    save_dir = 'datasets'
    datasets_to_download = ['tolokers', 'questions']
    
    print("=== Dataset Download Tool ===")
    print(f"Save directory: {save_dir}")
    
    for dataset_name in datasets_to_download:
        print(f"\n--- Processing {dataset_name} dataset ---")
        
        try:
            result_path = load_or_download_dataset(dataset_name, save_dir)
            success = result_path is not None
            update_download_results(dataset_name, success, save_dir)
            
            if success:
                print(f"✓ {dataset_name} processed successfully")
            else:
                print(f"✗ {dataset_name} processing failed")
                
        except Exception as e:
            print(f"✗ {dataset_name} processing error: {str(e)}")
            update_download_results(dataset_name, False, save_dir)
    
    print("\n=== Download completed ===")


if __name__ == "__main__":
    main()