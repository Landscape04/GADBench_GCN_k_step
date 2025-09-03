#!/usr/bin/env python3
"""
Enhanced Graph Anomaly Detection - Simplified Entry Point
"""

import os
import sys
import time
import torch
import torch.optim as optim
import argparse
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.data import load_and_split, prepare_dataset
from src.models import (
    GCN, GAT, GraphSAGE, 
    NeighborhoodSimilarityGCN, AdaptiveNeighborhoodSimilarityGCN,
    NeighborhoodSimilarityGAT, AdaptiveNeighborhoodSimilarityGAT,
    NeighborhoodSimilarityGraphSAGE, AdaptiveNeighborhoodSimilarityGraphSAGE
)
from config import MODEL_INFO, MODEL_ALIASES, DEFAULT_CONFIG, FAST_CONFIG
from src.training import Trainer
from src.utils import calculate_metrics, get_recorder


def resolve_model_name(model_name):
    """Resolve model name from alias to full name"""
    if model_name in MODEL_ALIASES:
        return MODEL_ALIASES[model_name]
    if model_name in MODEL_INFO:
        return MODEL_INFO[model_name]['full_name']
    return model_name


def create_model(resolved_model_name, in_dim, config):
    """Create model instance based on resolved name"""
    if resolved_model_name == 'gcn':
        return GCN(in_dim=in_dim, hidden_dim=128, dropout=config.get('dropout', 0.0))
    elif resolved_model_name == 'gat':
        return GAT(nfeat=in_dim, nhid=128, nclass=1, heads=8, dropout=config.get('dropout', 0.0))
    elif resolved_model_name == 'graphsage':
        return GraphSAGE(nfeat=in_dim, nhid=128, nclass=1, dropout=config.get('dropout', 0.0))
    
    # Enhanced models
    elif resolved_model_name == 'neighborhoodsimilaritygcn':
        return NeighborhoodSimilarityGCN(
            in_dim=in_dim, hidden_dim=config['hidden_dim'],
            degree_percentile=config['degree_percentile'],
            max_candidates=config.get('max_candidates', 20),
            dropout=config.get('dropout', 0.0)
        )
    elif resolved_model_name == 'adaptiveneighborhoodsimilaritygcn':
        model = AdaptiveNeighborhoodSimilarityGCN(
            in_dim=in_dim, hidden_dim=config['hidden_dim'],
            initial_degree_percentile=config.get('initial_degree_percentile', 0.1),
            max_candidates=config.get('max_candidates', 20),
            dropout=config.get('dropout', 0.0),
            adaptation_strategy=config.get('adaptation_strategy', 'performance'),
            min_percentile=config.get('min_percentile', 0.05),
            max_percentile=config.get('max_percentile', 0.3)
        )
        model.set_adaptation_config(
            strategy=config.get('adaptation_strategy', 'performance'),
            adjustment_frequency=config.get('adjustment_frequency', 5),
            adjustment_factor=config.get('adjustment_factor', 0.95)
        )
        return model
    
    # Enhanced GAT models
    elif resolved_model_name == 'neighborhoodsimilaritygat':
        return NeighborhoodSimilarityGAT(
            in_dim=in_dim, hidden_dim=config['hidden_dim'], heads=8,
            degree_percentile=config['degree_percentile'],
            max_candidates=config.get('max_candidates', 20),
            dropout=config.get('dropout', 0.0)
        )
    elif resolved_model_name == 'adaptiveneighborhoodsimilaritygat':
        model = AdaptiveNeighborhoodSimilarityGAT(
            in_dim=in_dim, hidden_dim=config['hidden_dim'], heads=8,
            initial_degree_percentile=config.get('initial_degree_percentile', 0.1),
            max_candidates=config.get('max_candidates', 20),
            dropout=config.get('dropout', 0.0),
            adaptation_strategy=config.get('adaptation_strategy', 'performance'),
            min_percentile=config.get('min_percentile', 0.05),
            max_percentile=config.get('max_percentile', 0.3)
        )
        model.set_adaptation_config(
            strategy=config.get('adaptation_strategy', 'performance'),
            adjustment_frequency=config.get('adjustment_frequency', 5),
            adjustment_factor=config.get('adjustment_factor', 0.95)
        )
        return model
    
    # Enhanced GraphSAGE models
    elif resolved_model_name == 'neighborhoodsimilaritygraphsage':
        return NeighborhoodSimilarityGraphSAGE(
            in_dim=in_dim, hidden_dim=config['hidden_dim'],
            degree_percentile=config['degree_percentile'],
            max_candidates=config.get('max_candidates', 20),
            dropout=config.get('dropout', 0.0)
        )
    elif resolved_model_name == 'adaptiveneighborhoodsimilaritygraphsage':
        model = AdaptiveNeighborhoodSimilarityGraphSAGE(
            in_dim=in_dim, hidden_dim=config['hidden_dim'],
            initial_degree_percentile=config.get('initial_degree_percentile', 0.1),
            max_candidates=config.get('max_candidates', 20),
            dropout=config.get('dropout', 0.0),
            adaptation_strategy=config.get('adaptation_strategy', 'performance'),
            min_percentile=config.get('min_percentile', 0.05),
            max_percentile=config.get('max_percentile', 0.3)
        )
        model.set_adaptation_config(
            strategy=config.get('adaptation_strategy', 'performance'),
            adjustment_frequency=config.get('adjustment_frequency', 5),
            adjustment_factor=config.get('adjustment_factor', 0.95)
        )
        return model
    else:
        raise ValueError(f"Unsupported model type: {resolved_model_name}")


def run_experiment(model_name, dataset_name, trial_num, config, resume=True):
    """Run experiment with specified model and dataset"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Get results recorder
    recorder = get_recorder()
    
    # Resolve model name
    resolved_model_name = resolve_model_name(model_name)
    
    # Start experiment recording (with resume capability)
    filename, is_completed = recorder.start_experiment(model_name, dataset_name, trial_num, config, resume=resume)
    
    # Skip if experiment is already completed
    if is_completed:
        return []
    
    # Load dataset
    dataset_path = os.path.join('datasets', f"{dataset_name}.pt")
    if not os.path.exists(dataset_path):
        dataset_path = prepare_dataset(dataset_name)
    
    data = load_and_split(dataset_path, seed=42, show_stats=True)
    print(f"\nRunning {model_name.upper()} on {dataset_name.upper()}, {trial_num} trials")
    print(f"Results file: {filename}")
    
    all_results = []
    
    for trial in range(trial_num):
        # Skip if trial already completed
        if recorder.should_skip_trial(trial + 1):
            print(f"⏭️ Skipping trial {trial+1}/{trial_num} (already completed)")
            continue
            
        trial_start_time = time.time()
        
        try:
            print(f"\n--- Trial {trial+1}/{trial_num} ---")
            # Re-split data for each trial
            data = load_and_split(dataset_path, seed=trial+1, show_stats=False)
            data = data.to(device)
            
            # Create model
            model = create_model(resolved_model_name, data.x.shape[1], config).to(device)
            
            # Initialize optimizer
            optimizer = optim.AdamW(model.parameters(), 
                                  lr=config['learning_rate'],
                                  weight_decay=config['weight_decay'])
            
            # Train model with recorder
            trainer = Trainer(model, optimizer, device, config, recorder=recorder)
            epochs = trainer.train(data, trial+1)
            
            # Test model
            with torch.no_grad():
                model.eval()
                test_logits = model(data.x, data.edge_index)
                test_probs = torch.sigmoid(test_logits)
                
                test_metrics = calculate_metrics(
                    y_true=data.y[data.test_mask],
                    y_scores=test_probs[data.test_mask],
                    k_values=[50, 100]
                )
                
                # Calculate training time
                training_time = time.time() - trial_start_time
                
                # Record final trial results
                recorder.record_trial_final(trial+1, test_metrics, epochs, training_time)
                
                all_results.append(test_metrics)
                print(f"✓ Trial {trial+1}: AUROC={test_metrics['AUROC']:.4f}, "
                      f"AUPRC={test_metrics['AUPRC']:.4f}, Epochs={epochs}, Time={training_time:.1f}s")
            
        except Exception as e:
            # Record failed trial
            recorder.record_failed_trial(trial+1, str(e))
            print(f"✗ Trial {trial+1} failed: {str(e)}")
            continue
    
    # Finish experiment
    recorder.finish_experiment()
    
    # Calculate averages
    if all_results:
        avg_auroc = sum(r['AUROC'] for r in all_results) / len(all_results)
        avg_auprc = sum(r['AUPRC'] for r in all_results) / len(all_results)
        print(f"\n{'='*50}")
        print(f"Final Results: AUROC={avg_auroc:.4f}, AUPRC={avg_auprc:.4f}")
        print(f"Results saved to: {filename}")
        print(f"{'='*50}")
    
    return all_results


def parse_model_list(model_arg):
    """Parse model argument"""
    all_models = ['gcn', 'gat', 'graphsage', 'ns-gcn', 'ans-gcn', 'ns-gat', 'ans-gat', 'ns-sage', 'ans-sage']
    
    if model_arg == 'all':
        return all_models
    else:
        models = [m.strip() for m in model_arg.split(',')]
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
    """Parse dataset argument"""
    all_datasets = ['reddit', 'weibo', 'tolokers', 'questions']
    
    if dataset_arg == 'all':
        return all_datasets
    else:
        datasets = [d.strip() for d in dataset_arg.split(',')]
        for dataset in datasets:
            if dataset not in all_datasets:
                raise ValueError(f"Unsupported dataset: {dataset}")
        return datasets


def main():
    parser = argparse.ArgumentParser(description='Enhanced Graph Anomaly Detection - Simplified')
    parser.add_argument('--model', type=str, default='ns-gcn', 
                      help='Model: gcn, gat, graphsage, ns-gcn, ans-gcn, ns-gat, ans-gat, ns-sage, ans-sage')
    parser.add_argument('--dataset', type=str, default='tolokers',
                      help='Dataset: reddit, weibo, tolokers, questions')
    parser.add_argument('--trials', type=int, default=5, help='Number of trials')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout probability')
    parser.add_argument('--fast', action='store_true', help='Use fast training configuration (shorter epochs, more aggressive early stopping)')
    parser.add_argument('--no-resume', action='store_true', help='Disable resume functionality, always start new experiments')
    args = parser.parse_args()
    
    # Configuration
    config = FAST_CONFIG.copy() if args.fast else DEFAULT_CONFIG.copy()
    config['dropout'] = args.dropout
    
    try:
        models = parse_model_list(args.model)
        datasets = parse_dataset_list(args.dataset)
        
        print(f"{'='*60}")
        print(f"EXPERIMENT CONFIGURATION")
        print(f"{'='*60}")
        print(f"Models: {models}")
        print(f"Datasets: {datasets}")
        print(f"Trials: {args.trials}")
        print(f"Dropout: {args.dropout}")
        print(f"Fast mode: {args.fast}")
        print(f"Resume mode: {not args.no_resume}")
        if args.fast:
            print(f"Fast config: max_epochs={config['max_epochs']}, patience={config['patience']}")
        print(f"{'='*60}")
        
        # Get recorder for comparison
        recorder = get_recorder()
        recorder.clear_comparison()  # Clear previous comparison data
        
        # Check for existing results
        results_dir = Path("results")
        if results_dir.exists():
            existing_files = list(results_dir.glob("*.xlsx"))
            if existing_files:
                print(f"📂 Found {len(existing_files)} existing result files in results/ directory")
                print("💡 New experiments will create separate files with timestamps")
        
        # Run experiments
        total_experiments = len(models) * len(datasets)
        current_exp = 0
        
        for model_name in models:
            for dataset_name in datasets:
                current_exp += 1
                print(f"\n[{current_exp}/{total_experiments}] Starting {model_name} on {dataset_name}")
                run_experiment(model_name, dataset_name, args.trials, config, resume=not args.no_resume)
        
        # Save comparison if multiple experiments
        if len(models) > 1 or len(datasets) > 1:
            comparison_file = recorder.save_comparison()
            if comparison_file:
                print(f"\n{'='*60}")
                print(f"COMPARISON SAVED")
                print(f"{'='*60}")
                print(f"Comparison file: {comparison_file}")
                print(f"This file contains summary results for all experiments.")
        
    except ValueError as e:
        print(f"Error: {e}")
        print("Available models: gcn, gat, graphsage, ns-gcn, ans-gcn, ns-gat, ans-gat, ns-sage, ans-sage")
        print("Available datasets: reddit, weibo, tolokers, questions")
    except KeyboardInterrupt:
        print("\n🛑 Experiment interrupted by user (Ctrl+C)")
        print("📁 All results have been automatically saved up to this point.")
        print("💡 You can resume by running the same command - completed experiments will be skipped.")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        print("📁 All results have been automatically saved up to this point.")
        print("💡 Check the error and try running again.")


if __name__ == "__main__":
    main()