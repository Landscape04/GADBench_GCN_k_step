#!/usr/bin/env python3
"""
Enhanced Graph Anomaly Detection - Main Entry Point

This is the main entry point for running graph anomaly detection experiments
with enhanced GCN models that promote 3-hop neighbors to 2-hop based on similarity.

Usage:
    python run_experiment.py --model neighborhoodsimilaritygcn --dataset tolokers --trials 5
    python run_experiment.py --model adaptiveneighborhoodsimilaritygcn --dataset reddit --trials 3
    python run_experiment.py --model all --dataset all --trials 1
"""

import os
import sys
import time
import torch
import torch.optim as optim
import argparse
from sklearn.metrics import roc_auc_score

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.data import load_and_split, prepare_dataset
from src.models import (
    GCN, GAT, GraphSAGE, 
    NeighborhoodSimilarityGCN, AdaptiveNeighborhoodSimilarityGCN,
    NeighborhoodSimilarityGAT, AdaptiveNeighborhoodSimilarityGAT,
    NeighborhoodSimilarityGraphSAGE, AdaptiveNeighborhoodSimilarityGraphSAGE
)
from config import MODEL_INFO, MODEL_ALIASES
from src.training import Trainer
from src.utils import calculate_metrics, get_results_manager, print_metrics


def resolve_model_name(model_name):
    """
    Resolve model name from alias to full name
    """
    # Check if it's an alias
    if model_name in MODEL_ALIASES:
        return MODEL_ALIASES[model_name]
    
    # Check if it's a short name in MODEL_INFO
    if model_name in MODEL_INFO:
        return MODEL_INFO[model_name]['full_name']
    
    # Return as-is if not found (for backward compatibility)
    return model_name


def run_experiment(model_name, dataset_name, trial_num, config, filename=None):
    """
    Run experiment with specified model and dataset
    
    Args:
        model_name: Model name
        dataset_name: Dataset name
        trial_num: Number of trials
        config: Configuration parameters
        filename: Excel filename for results
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Resolve model name from alias
    resolved_model_name = resolve_model_name(model_name)
    
    # Create display name
    if resolved_model_name in ['neighborhoodsimilaritygcn', 'neighborhoodsimilaritygat', 'neighborhoodsimilaritygraphsage']:
        model_display_name = f"{model_name.upper()}_degree{config['degree_percentile']}"
    elif resolved_model_name in ['adaptiveneighborhoodsimilaritygcn', 'adaptiveneighborhoodsimilaritygat', 'adaptiveneighborhoodsimilaritygraphsage']:
        model_display_name = f"{model_name.upper()}_adaptive"
    else:
        model_display_name = model_name.upper()
    
    # Load dataset
    dataset_path = os.path.join('datasets', f"{dataset_name}.pt")
    if not os.path.exists(dataset_path):
        dataset_path = prepare_dataset(dataset_name)
    
    # First load shows complete statistics
    data = load_and_split(dataset_path, seed=42, show_stats=True)
    print(f"\n=== Starting Experiment ===")
    print(f"Model: {model_display_name}")
    print(f"Dataset: {dataset_name.upper()}")
    print(f"Total trials: {trial_num}\n")
    
    all_trial_results = []
    best_metrics = {
        'AUROC': 0.0,
        'AUPRC': 0.0,
        'REC@50': 0.0,
        'REC@100': 0.0
    }

    for trial in range(trial_num):
        trial_start_time = time.time()
        
        try:
            # Re-split data for each trial (no statistics display)
            data = load_and_split(dataset_path, seed=trial+1, show_stats=False)
            data = data.to(device)
            
            # Create model based on resolved name
            if resolved_model_name == 'gcn':
                model = GCN(in_dim=data.x.shape[1], hidden_dim=128, dropout=config.get('dropout', 0.0)).to(device)
            elif resolved_model_name == 'gat':
                model = GAT(nfeat=data.x.shape[1], nhid=128, 
                           nclass=1, heads=8, dropout=config.get('dropout', 0.0)).to(device)
            elif resolved_model_name == 'graphsage':
                model = GraphSAGE(nfeat=data.x.shape[1], nhid=128, 
                                 nclass=1, dropout=config.get('dropout', 0.0)).to(device)
            
            # Enhanced GCN models
            elif resolved_model_name == 'neighborhoodsimilaritygcn':
                model = NeighborhoodSimilarityGCN(
                    in_dim=data.x.shape[1],
                    hidden_dim=config['hidden_dim'],
                    degree_percentile=config['degree_percentile'],
                    max_candidates=config.get('max_candidates', 20),
                    dropout=config.get('dropout', 0.0)
                ).to(device)
            elif resolved_model_name == 'adaptiveneighborhoodsimilaritygcn':
                model = AdaptiveNeighborhoodSimilarityGCN(
                    in_dim=data.x.shape[1],
                    hidden_dim=config['hidden_dim'],
                    initial_degree_percentile=config.get('initial_degree_percentile', 0.1),
                    max_candidates=config.get('max_candidates', 20),
                    dropout=config.get('dropout', 0.0),
                    adaptation_strategy=config.get('adaptation_strategy', 'performance'),
                    min_percentile=config.get('min_percentile', 0.05),
                    max_percentile=config.get('max_percentile', 0.3)
                ).to(device)
                
                # Configure adaptive model
                model.set_adaptation_config(
                    strategy=config.get('adaptation_strategy', 'performance'),
                    adjustment_frequency=config.get('adjustment_frequency', 5),
                    adjustment_factor=config.get('adjustment_factor', 0.95),
                    print_adjustments=config.get('print_adjustments', False)
                )
            
            # Enhanced GAT models
            elif resolved_model_name == 'neighborhoodsimilaritygat':
                model = NeighborhoodSimilarityGAT(
                    in_dim=data.x.shape[1],
                    hidden_dim=config['hidden_dim'],
                    heads=8,
                    degree_percentile=config['degree_percentile'],
                    max_candidates=config.get('max_candidates', 20),
                    dropout=config.get('dropout', 0.0)
                ).to(device)
            elif resolved_model_name == 'adaptiveneighborhoodsimilaritygat':
                model = AdaptiveNeighborhoodSimilarityGAT(
                    in_dim=data.x.shape[1],
                    hidden_dim=config['hidden_dim'],
                    heads=8,
                    initial_degree_percentile=config.get('initial_degree_percentile', 0.1),
                    max_candidates=config.get('max_candidates', 20),
                    dropout=config.get('dropout', 0.0),
                    adaptation_strategy=config.get('adaptation_strategy', 'performance'),
                    min_percentile=config.get('min_percentile', 0.05),
                    max_percentile=config.get('max_percentile', 0.3)
                ).to(device)
                
                # Configure adaptive model
                model.set_adaptation_config(
                    strategy=config.get('adaptation_strategy', 'performance'),
                    adjustment_frequency=config.get('adjustment_frequency', 5),
                    adjustment_factor=config.get('adjustment_factor', 0.95),
                    print_adjustments=config.get('print_adjustments', False)
                )
            
            # Enhanced GraphSAGE models
            elif resolved_model_name == 'neighborhoodsimilaritygraphsage':
                model = NeighborhoodSimilarityGraphSAGE(
                    in_dim=data.x.shape[1],
                    hidden_dim=config['hidden_dim'],
                    degree_percentile=config['degree_percentile'],
                    max_candidates=config.get('max_candidates', 20),
                    dropout=config.get('dropout', 0.0)
                ).to(device)
            elif resolved_model_name == 'adaptiveneighborhoodsimilaritygraphsage':
                model = AdaptiveNeighborhoodSimilarityGraphSAGE(
                    in_dim=data.x.shape[1],
                    hidden_dim=config['hidden_dim'],
                    initial_degree_percentile=config.get('initial_degree_percentile', 0.1),
                    max_candidates=config.get('max_candidates', 20),
                    dropout=config.get('dropout', 0.0),
                    adaptation_strategy=config.get('adaptation_strategy', 'performance'),
                    min_percentile=config.get('min_percentile', 0.05),
                    max_percentile=config.get('max_percentile', 0.3)
                ).to(device)
                
                # Configure adaptive model
                model.set_adaptation_config(
                    strategy=config.get('adaptation_strategy', 'performance'),
                    adjustment_frequency=config.get('adjustment_frequency', 5),
                    adjustment_factor=config.get('adjustment_factor', 0.95),
                    print_adjustments=config.get('print_adjustments', False)
                )
            else:
                raise ValueError(f"Unsupported model type: {resolved_model_name} (original: {model_name})")
            
            # Initialize optimizer
            optimizer = optim.AdamW(model.parameters(), 
                                  lr=config['learning_rate'],
                                  weight_decay=config['weight_decay'])
            
            # Train model
            print(f"\nTraining {model_name} on {dataset_name}, trial {trial+1}...")
            print(f"Device: {device}")
            
            # 添加进度显示
            print(f"Starting training with max_epochs={config['max_epochs']}")
            print(f"Model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
            
            # 记录开始时间
            start_time = time.time()
            trainer = Trainer(model, optimizer, device, config)
            epochs = trainer.train(data, trial+1)
            
            # 显示训练完成时间
            training_time = time.time() - start_time
            print(f"Training completed in {training_time:.2f} seconds")
            
            # Test model
            with torch.no_grad():
                model.eval()
                test_logits = model(data.x, data.edge_index)
                test_probs = torch.sigmoid(test_logits)
                
                # Calculate GADBench-style metrics
                test_metrics = calculate_metrics(
                    y_true=data.y[data.test_mask],
                    y_scores=test_probs[data.test_mask],
                    k_values=[50, 100]
                )
                
                # Update best metrics
                for metric in ['AUROC', 'AUPRC', 'REC@50', 'REC@100']:
                    if test_metrics[metric] > best_metrics[metric]:
                        best_metrics[metric] = test_metrics[metric]
                
                # Save trial result
                trial_result = {
                    'trial': trial + 1,
                    'dataset': dataset_name.upper(),
                    'model': model_display_name,
                    'AUROC': test_metrics['AUROC'],
                    'AUPRC': test_metrics['AUPRC'],
                    'REC@50': test_metrics['REC@50'],
                    'REC@100': test_metrics['REC@100'],
                    'epochs': epochs,
                    'time': round(time.time() - trial_start_time, 3),
                    'timestamp': time.strftime("%Y-%m-%d %H:%M:%S")
                }
                all_trial_results.append(trial_result)
                
                # Save result in real-time
                save_results_realtime(trial_result, filename)
                
                # Print trial result
                print(f"Trial {trial+1}: AUROC: {test_metrics['AUROC']:.3f}, AUPRC: {test_metrics['AUPRC']:.3f}, REC@50: {test_metrics['REC@50']:.3f}, Epochs: {epochs}")
            
        except Exception as e:
            print(f"Trial {trial+1} failed: {str(e)}")
            # Record failed trial
            failed_result = {
                'trial': trial + 1,
                'dataset': dataset_name.upper(),
                'model': model_display_name,
                'AUROC': 0.0,
                'AUPRC': 0.0,
                'REC@50': 0.0,
                'REC@100': 0.0,
                'epochs': 0,
                'time': round(time.time() - trial_start_time, 3),
                'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
                'status': 'FAILED',
                'error': str(e)
            }
            save_results_realtime(failed_result, filename)
            continue
    
    # Calculate and display final results
    if len(all_trial_results) > 0:
        avg_auroc = sum(r['AUROC'] for r in all_trial_results) / len(all_trial_results)
        avg_auprc = sum(r['AUPRC'] for r in all_trial_results) / len(all_trial_results)
        avg_rec50 = sum(r['REC@50'] for r in all_trial_results) / len(all_trial_results)
        avg_rec100 = sum(r['REC@100'] for r in all_trial_results) / len(all_trial_results)
        
        print(f"\n{'='*60}")
        print(f"Experiment Complete! {model_display_name} on {dataset_name.upper()}:")
        print(f"Average AUROC: {avg_auroc:.4f}")
        print(f"Average AUPRC: {avg_auprc:.4f}")
        print(f"Average REC@50: {avg_rec50:.4f}")
        print(f"Average REC@100: {avg_rec100:.4f}")
        print(f"Successful trials: {len(all_trial_results)}/{trial_num}")
        
        # Print best metrics
        print(f"Best AUROC: {best_metrics['AUROC']:.4f}")
        print(f"Best AUPRC: {best_metrics['AUPRC']:.4f}")
        print(f"Best REC@50: {best_metrics['REC@50']:.4f}")
        print(f"Best REC@100: {best_metrics['REC@100']:.4f}")
        print(f"{'='*60}")
        
        # Save summary result
        summary_result = {
            'dataset': dataset_name.upper(),
            'model': model_display_name,
            'trial_count': trial_num,
            'avg_AUROC': avg_auroc,
            'avg_AUPRC': avg_auprc,
            'avg_REC@50': avg_rec50,
            'avg_REC@100': avg_rec100,
            'best_AUROC': best_metrics['AUROC'],
            'best_AUPRC': best_metrics['AUPRC'],
            'best_REC@50': best_metrics['REC@50'],
            'best_REC@100': best_metrics['REC@100'],
            'success_trials': len(all_trial_results),
            'total_trials': trial_num,
            'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
            'type': 'SUMMARY'
        }
        save_results_realtime(summary_result, filename)
    
    return all_trial_results


def run_experiments_with_filename(models, datasets, trial_num, config):
    """Run experiments with dynamic filename"""
    # Generate filename
    filename = generate_experiment_filename(models, datasets, trial_num)
    print(f"\nExperiment results will be saved to: {filename}")
    
    # Ensure models and datasets are lists
    if isinstance(models, str):
        models = [models]
    if isinstance(datasets, str):
        datasets = [datasets]
    
    all_results = {}
    total_combinations = len(models) * len(datasets)
    current_combination = 0
    
    for model_name in models:
        all_results[model_name] = {}
        for dataset_name in datasets:
            current_combination += 1
            print(f"\n{'='*60}")
            print(f"Progress: {current_combination}/{total_combinations}")
            print(f"Current combination: {model_name.upper()} - {dataset_name.upper()}")
            print(f"{'='*60}")
            
            try:
                results = run_experiment(model_name, dataset_name, trial_num, config, filename)
                all_results[model_name][dataset_name] = results
            except Exception as e:
                print(f"Experiment failed: {model_name} - {dataset_name}: {str(e)}")
                all_results[model_name][dataset_name] = []
                continue
    
    return all_results


def parse_model_list(model_arg):
    """Parse model argument, support comma-separated multiple models"""
    # All available models (using short aliases)
    all_models = ['gcn', 'gat', 'graphsage', 'ns-gcn', 'ans-gcn', 'ns-gat', 'ans-gat', 'ns-sage', 'ans-sage']
    
    if model_arg == 'all':
        return all_models
    else:
        # Support comma-separated multiple models
        models = [m.strip() for m in model_arg.split(',')]
        # Validate model names (check both short and full names)
        for model in models:
            resolved_name = resolve_model_name(model)
            if model not in all_models and model not in MODEL_ALIASES and resolved_name not in [
                'gcn', 'gat', 'graphsage', 
                'neighborhoodsimilaritygcn', 'adaptiveneighborhoodsimilaritygcn',
                'neighborhoodsimilaritygat', 'adaptiveneighborhoodsimilaritygat',
                'neighborhoodsimilaritygraphsage', 'adaptiveneighborhoodsimilaritygraphsage'
            ]:
                raise ValueError(f"Unsupported model: {model}")
        return models


def parse_dataset_list(dataset_arg):
    """Parse dataset argument, support comma-separated multiple datasets"""
    all_datasets = ['reddit', 'weibo', 'tolokers', 'questions']
    
    if dataset_arg == 'all':
        return all_datasets
    else:
        # Support comma-separated multiple datasets
        datasets = [d.strip() for d in dataset_arg.split(',')]
        # Validate dataset names
        for dataset in datasets:
            if dataset not in all_datasets:
                raise ValueError(f"Unsupported dataset: {dataset}")
        return datasets


def main():
    parser = argparse.ArgumentParser(description='Enhanced Graph Anomaly Detection')
    parser.add_argument('--model', type=str, default='ns-gcn', 
                      help='Model to use (all for all models, comma-separated for multiple). ' +
                           'Available: gcn, gat, graphsage, ns-gcn, ans-gcn, ns-gat, ans-gat, ns-sage, ans-sage')
    parser.add_argument('--dataset', type=str, default='tolokers',
                      help='Dataset to use (all for all datasets, comma-separated for multiple)')
    parser.add_argument('--trials', type=int, default=5,
                      help='Number of experiment trials')
    parser.add_argument('--dropout', type=float, default=0.1,
                      help='Dropout probability')
    args = parser.parse_args()
    
    # Default configuration
    config = {
        'hidden_dim': 64,
        'learning_rate': 0.01,
        'degree_percentile': 0.1,  # Optimal value from hyperparameter tuning
        'patience': 15,
        'delta': 0.001,
        'warmup_epochs': 5,
        'smooth_window': 2,
        'max_epochs': 100,
        'weight_decay': 1e-4,
        'dropout': args.dropout,
        
        # Adaptive model parameters
        'initial_degree_percentile': 0.1,
        'max_candidates': 20,
        'adaptation_strategy': 'performance',
        'min_percentile': 0.05,
        'max_percentile': 0.3,
        'adjustment_frequency': 5,
        'adjustment_factor': 0.95,
        'print_adjustments': False
    }
    
    try:
        # Parse model and dataset lists
        models = parse_model_list(args.model)
        datasets = parse_dataset_list(args.dataset)
        
        print(f"Selected models: {models}")
        print(f"Selected datasets: {datasets}")
        print(f"Number of trials: {args.trials}")
        print(f"Dropout: {args.dropout}")
        
        # Run experiments
        run_experiments_with_filename(models, datasets, args.trials, config)
        
    except ValueError as e:
        print(f"Parameter error: {e}")
        print("Available models:")
        print("  Baseline: gcn, gat, graphsage")
        print("  Enhanced: ns-gcn, ans-gcn, ns-gat, ans-gat, ns-sage, ans-sage")
        print("  (ns = NeighborhoodSimilarity, ans = AdaptiveNeighborhoodSimilarity)")
        print("Available datasets: reddit, weibo, tolokers, questions")


if __name__ == "__main__":
    main()