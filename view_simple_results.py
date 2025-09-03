#!/usr/bin/env python3
"""
Simple Results Viewer
"""

import os
import pandas as pd
import argparse
from pathlib import Path


def list_results():
    """List all result files"""
    results_dir = Path("results")
    if not results_dir.exists():
        print("No results directory found.")
        return
    
    excel_files = list(results_dir.glob("*.xlsx"))
    if not excel_files:
        print("No result files found.")
        return
    
    print(f"{'='*80}")
    print("AVAILABLE RESULT FILES")
    print(f"{'='*80}")
    
    # Separate individual experiments and comparisons
    experiments = [f for f in excel_files if not f.name.startswith('comparison_')]
    comparisons = [f for f in excel_files if f.name.startswith('comparison_')]
    
    if experiments:
        print("\nINDIVIDUAL EXPERIMENTS:")
        print(f"{'ID':<3} {'Filename':<50} {'Size':<10} {'Modified':<20}")
        print(f"{'-'*80}")
        
        for i, file in enumerate(sorted(experiments, key=lambda x: x.stat().st_mtime, reverse=True), 1):
            size = f"{file.stat().st_size / 1024:.1f}KB"
            modified = pd.Timestamp(file.stat().st_mtime, unit='s').strftime('%Y-%m-%d %H:%M:%S')
            print(f"{i:<3} {file.name:<50} {size:<10} {modified:<20}")
    
    if comparisons:
        print("\nCOMPARISON FILES:")
        print(f"{'ID':<3} {'Filename':<50} {'Size':<10} {'Modified':<20}")
        print(f"{'-'*80}")
        
        for i, file in enumerate(sorted(comparisons, key=lambda x: x.stat().st_mtime, reverse=True), 1):
            size = f"{file.stat().st_size / 1024:.1f}KB"
            modified = pd.Timestamp(file.stat().st_mtime, unit='s').strftime('%Y-%m-%d %H:%M:%S')
            print(f"{i:<3} {file.name:<50} {size:<10} {modified:<20}")


def view_experiment(filename):
    """View details of a specific experiment"""
    results_dir = Path("results")
    file_path = results_dir / filename
    
    if not file_path.exists():
        print(f"File not found: {filename}")
        return
    
    try:
        # Read all sheets
        excel_data = pd.read_excel(file_path, sheet_name=None)
        
        print(f"{'='*80}")
        print(f"EXPERIMENT DETAILS: {filename}")
        print(f"{'='*80}")
        
        # Show config if available
        if 'Config' in excel_data:
            config_df = excel_data['Config']
            print("\nCONFIGURATION:")
            for _, row in config_df.iterrows():
                for col, val in row.items():
                    if pd.notna(val):
                        print(f"  {col}: {val}")
        
        # Show summary if available
        if 'Summary' in excel_data:
            summary_df = excel_data['Summary']
            print(f"\n{'='*60}")
            print("TRIAL SUMMARY")
            print(f"{'='*60}")
            
            successful_trials = summary_df[summary_df['status'] == 'SUCCESS']
            failed_trials = summary_df[summary_df['status'] == 'FAILED']
            
            print(f"Total trials: {len(summary_df)}")
            print(f"Successful: {len(successful_trials)}")
            print(f"Failed: {len(failed_trials)}")
            
            if len(successful_trials) > 0:
                print(f"\nSUCCESSFUL TRIALS STATISTICS:")
                print(f"Average AUROC: {successful_trials['final_auroc'].mean():.4f} ± {successful_trials['final_auroc'].std():.4f}")
                print(f"Average AUPRC: {successful_trials['final_auprc'].mean():.4f} ± {successful_trials['final_auprc'].std():.4f}")
                print(f"Best AUROC: {successful_trials['final_auroc'].max():.4f}")
                print(f"Best AUPRC: {successful_trials['final_auprc'].max():.4f}")
                print(f"Average Epochs: {successful_trials['total_epochs'].mean():.1f}")
                print(f"Average Time: {successful_trials['total_time'].mean():.1f}s")
            
            # Show individual trial results
            print(f"\nINDIVIDUAL TRIAL RESULTS:")
            print(f"{'Trial':<6} {'AUROC':<8} {'AUPRC':<8} {'REC@50':<8} {'REC@100':<9} {'Epochs':<7} {'Time(s)':<8} {'Status':<8}")
            print(f"{'-'*70}")
            
            for _, row in summary_df.iterrows():
                if row['status'] == 'SUCCESS':
                    print(f"{row['trial']:<6} {row['final_auroc']:<8.4f} {row['final_auprc']:<8.4f} "
                          f"{row['final_rec50']:<8.4f} {row['final_rec100']:<9.4f} "
                          f"{row['total_epochs']:<7} {row['total_time']:<8.1f} {'SUCCESS':<8}")
                else:
                    print(f"{row['trial']:<6} {'FAILED':<8} {'FAILED':<8} {'FAILED':<8} {'FAILED':<9} "
                          f"{row.get('total_epochs', 0):<7} {row.get('total_time', 0):<8.1f} {'FAILED':<8}")
        
        # Show epoch details for first few trials if available
        if 'Results' in excel_data:
            results_df = excel_data['Results']
            epoch_data = results_df[results_df['epoch'] != 'FINAL'].head(20)  # First 20 epochs
            
            if not epoch_data.empty:
                print(f"\n{'='*60}")
                print("EPOCH DETAILS (First 20 epochs)")
                print(f"{'='*60}")
                print(f"{'Trial':<6} {'Epoch':<6} {'Loss':<8} {'Val AUC':<8} {'Val AP':<8} {'Time(s)':<8}")
                print(f"{'-'*50}")
                
                for _, row in epoch_data.iterrows():
                    loss = f"{row['train_loss']:.4f}" if pd.notna(row['train_loss']) else "N/A"
                    auc = f"{row['val_auc']:.4f}" if pd.notna(row['val_auc']) else "N/A"
                    ap = f"{row['val_ap']:.4f}" if pd.notna(row['val_ap']) else "N/A"
                    time_val = f"{row['epoch_time']:.2f}" if pd.notna(row['epoch_time']) else "N/A"
                    
                    print(f"{row['trial']:<6} {row['epoch']:<6} {loss:<8} {auc:<8} {ap:<8} {time_val:<8}")
    
    except Exception as e:
        print(f"Error reading file: {e}")


def view_comparison(filename):
    """View comparison results"""
    results_dir = Path("results")
    file_path = results_dir / filename
    
    if not file_path.exists():
        print(f"File not found: {filename}")
        return
    
    try:
        df = pd.read_excel(file_path)
        
        print(f"{'='*100}")
        print(f"COMPARISON RESULTS: {filename}")
        print(f"{'='*100}")
        
        print(f"{'Model':<15} {'Dataset':<12} {'Trials':<7} {'Avg AUROC':<12} {'Avg AUPRC':<12} "
              f"{'Best AUROC':<12} {'Best AUPRC':<12} {'Filename':<30}")
        print(f"{'-'*100}")
        
        for _, row in df.iterrows():
            print(f"{row['model']:<15} {row['dataset']:<12} {row['trials']:<7} "
                  f"{row['avg_auroc']:<12.4f} {row['avg_auprc']:<12.4f} "
                  f"{row['best_auroc']:<12.4f} {row['best_auprc']:<12.4f} "
                  f"{row['filename']:<30}")
        
        # Show best performing models
        print(f"\n{'='*60}")
        print("TOP PERFORMERS")
        print(f"{'='*60}")
        
        best_auroc = df.loc[df['avg_auroc'].idxmax()]
        best_auprc = df.loc[df['avg_auprc'].idxmax()]
        
        print(f"Best Average AUROC: {best_auroc['model']} on {best_auroc['dataset']} ({best_auroc['avg_auroc']:.4f})")
        print(f"Best Average AUPRC: {best_auprc['model']} on {best_auprc['dataset']} ({best_auprc['avg_auprc']:.4f})")
    
    except Exception as e:
        print(f"Error reading comparison file: {e}")


def main():
    parser = argparse.ArgumentParser(description='Simple Results Viewer')
    parser.add_argument('--list', action='store_true', help='List all result files')
    parser.add_argument('--view', type=str, help='View specific experiment file')
    parser.add_argument('--compare', type=str, help='View comparison file')
    
    args = parser.parse_args()
    
    if args.list:
        list_results()
    elif args.view:
        view_experiment(args.view)
    elif args.compare:
        view_comparison(args.compare)
    else:
        print("Please specify an action:")
        print("  --list: List all result files")
        print("  --view <filename>: View experiment details")
        print("  --compare <filename>: View comparison results")
        print("\nExample:")
        print("  python view_simple_results.py --list")
        print("  python view_simple_results.py --view ns-gcn_tolokers_trials5_20241224_143022.xlsx")
        print("  python view_simple_results.py --compare comparison_20241224_143022.xlsx")


if __name__ == "__main__":
    main()