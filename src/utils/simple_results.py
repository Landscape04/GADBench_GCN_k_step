# simple_results.py
"""
Simplified Results Recording System
"""

import os
import pandas as pd
import json
from datetime import datetime
from pathlib import Path


class SimpleResultsRecorder:
    """
    Simple results recorder for experiments
    """
    
    def __init__(self, base_dir="results"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(exist_ok=True)
        
        # Current experiment info
        self.current_file = None
        self.current_data = []
        self.experiment_info = {}
        
        # Comparison data for multiple models
        self.comparison_data = []
    
    def start_experiment(self, model_name, dataset_name, trials, config=None, resume=False):
        """Start recording a new experiment or resume existing one"""
        # Add special parameters to filename if present
        param_str = ""
        if config:
            if 'degree_percentile' in config and config['degree_percentile'] != 0.1:
                param_str += f"_dp{config['degree_percentile']}"
            if 'dropout' in config and config['dropout'] != 0.1:
                param_str += f"_drop{config['dropout']}"
        
        # Check for existing experiment to resume
        if resume:
            pattern = f"{model_name}_{dataset_name}_trials{trials}{param_str}_*.xlsx"
            existing_files = list(self.base_dir.glob(pattern))
            
            if existing_files:
                # Find the most recent file
                latest_file = max(existing_files, key=lambda x: x.stat().st_mtime)
                self.current_file = latest_file
                
                # Load existing data
                try:
                    existing_data = pd.read_excel(self.current_file, sheet_name='Results')
                    self.current_data = existing_data.to_dict('records')
                    
                    # Count completed trials
                    completed_trials = len([d for d in self.current_data if d.get('epoch') in ['FINAL', 'FAILED']])
                    
                    if completed_trials >= trials:
                        print(f"✅ Experiment already completed: {self.current_file.name}")
                        print(f"📊 All {trials} trials finished. Skipping...")
                        return self.current_file.name, True  # Return completion flag
                    
                    print(f"🔄 Resuming experiment: {self.current_file.name}")
                    print(f"📈 Progress: {completed_trials}/{trials} trials completed")
                    
                    # Load experiment info
                    config_data = pd.read_excel(self.current_file, sheet_name='Config')
                    self.experiment_info = config_data.iloc[0].to_dict()
                    
                    return self.current_file.name, False  # Return not completed
                    
                except Exception as e:
                    print(f"⚠️ Could not resume from {latest_file.name}: {e}")
                    print("🆕 Starting new experiment...")
        
        # Create new experiment
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{model_name}_{dataset_name}_trials{trials}{param_str}_{timestamp}.xlsx"
        self.current_file = self.base_dir / filename
        
        # Reset data
        self.current_data = []
        self.experiment_info = {
            'model': model_name,
            'dataset': dataset_name,
            'trials': trials,
            'config': config or {},
            'start_time': datetime.now().isoformat(),
            'filename': filename
        }
        
        print(f"🆕 Recording results to: {filename}")
        return filename, False
    
    def record_epoch(self, trial, epoch, train_loss, val_auc=None, val_ap=None, epoch_time=None):
        """Record epoch-level results"""
        self.current_data.append({
            'trial': trial, 'epoch': epoch, 'train_loss': train_loss,
            'val_auc': val_auc, 'val_ap': val_ap, 'epoch_time': epoch_time,
            'timestamp': datetime.now().isoformat()
        })
    
    def record_trial_final(self, trial, final_metrics, total_epochs, total_time, status='SUCCESS'):
        """Record final trial results"""
        self.current_data.append({
            'trial': trial, 'epoch': 'FINAL', 'train_loss': None, 'val_auc': None, 'val_ap': None, 'epoch_time': None,
            'final_auroc': final_metrics.get('AUROC', 0), 'final_auprc': final_metrics.get('AUPRC', 0),
            'final_rec50': final_metrics.get('REC@50', 0), 'final_rec100': final_metrics.get('REC@100', 0),
            'total_epochs': total_epochs, 'total_time': total_time, 'status': status,
            'timestamp': datetime.now().isoformat()
        })
        
        # Auto-save every 5 trials
        if trial % 5 == 0:
            self.save_current_results()
            print(f"📁 Auto-saved after trial {trial}")
    
    def record_failed_trial(self, trial, error_message):
        """Record failed trial"""
        failed_data = {
            'trial': trial,
            'epoch': 'FAILED',
            'train_loss': None,
            'val_auc': None,
            'val_ap': None,
            'epoch_time': None,
            'final_auroc': 0,
            'final_auprc': 0,
            'final_rec50': 0,
            'final_rec100': 0,
            'total_epochs': 0,
            'total_time': 0,
            'status': 'FAILED',
            'error': error_message,
            'timestamp': datetime.now().isoformat()
        }
        self.current_data.append(failed_data)
        
        # Save immediately after failed trial to preserve error info
        self.save_current_results()
        print(f"📁 Auto-saved after failed trial {trial}")
    
    def save_current_results(self):
        """Save current results to Excel"""
        if not self.current_data or not self.current_file:
            return
        
        try:
            df = pd.DataFrame(self.current_data)
            
            # Create Excel writer with multiple sheets
            with pd.ExcelWriter(self.current_file, engine='openpyxl') as writer:
                # Main data sheet
                df.to_excel(writer, sheet_name='Results', index=False)
                
                # Summary sheet for final results only
                final_results = df[df['epoch'] == 'FINAL'].copy()
                if not final_results.empty:
                    summary_cols = ['trial', 'final_auroc', 'final_auprc', 'final_rec50', 
                                  'final_rec100', 'total_epochs', 'total_time', 'status']
                    final_results[summary_cols].to_excel(writer, sheet_name='Summary', index=False)
                
                # Config sheet
                config_df = pd.DataFrame([self.experiment_info])
                config_df.to_excel(writer, sheet_name='Config', index=False)
            
            print(f"Results saved: {len(self.current_data)} records")
            
        except Exception as e:
            print(f"Error saving results: {e}")
    
    def finish_experiment(self):
        """Finish current experiment"""
        if self.current_data:
            self.save_current_results()
            
            # Add to comparison data
            final_results = [d for d in self.current_data if d.get('epoch') == 'FINAL' and d.get('status') == 'SUCCESS']
            if final_results:
                avg_auroc = sum(r['final_auroc'] for r in final_results) / len(final_results)
                avg_auprc = sum(r['final_auprc'] for r in final_results) / len(final_results)
                
                comparison_entry = {
                    'model': self.experiment_info['model'],
                    'dataset': self.experiment_info['dataset'],
                    'trials': len(final_results),
                    'avg_auroc': avg_auroc,
                    'avg_auprc': avg_auprc,
                    'best_auroc': max(r['final_auroc'] for r in final_results),
                    'best_auprc': max(r['final_auprc'] for r in final_results),
                    'filename': self.experiment_info['filename'],
                    'timestamp': datetime.now().isoformat()
                }
                self.comparison_data.append(comparison_entry)
        
        print(f"Experiment completed: {self.experiment_info.get('filename', 'Unknown')}")
    
    def save_comparison(self, comparison_filename=None):
        """Save comparison results"""
        if not self.comparison_data:
            return
        
        if comparison_filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            comparison_filename = f"comparison_{timestamp}.xlsx"
        
        comparison_file = self.base_dir / comparison_filename
        
        try:
            df = pd.DataFrame(self.comparison_data)
            df.to_excel(comparison_file, index=False)
            print(f"Comparison saved: {comparison_filename}")
            return comparison_filename
        except Exception as e:
            print(f"Error saving comparison: {e}")
            return None
    
    def clear_comparison(self):
        """Clear comparison data"""
        self.comparison_data = []
    
    def get_completed_trials(self):
        """Get number of completed trials"""
        if not self.current_data:
            return 0
        return len([d for d in self.current_data if d.get('epoch') in ['FINAL', 'FAILED']])
    
    def should_skip_trial(self, trial_num):
        """Check if trial should be skipped (already completed)"""
        completed_trials = self.get_completed_trials()
        return trial_num <= completed_trials


# Global instance
_recorder = SimpleResultsRecorder()

def get_recorder():
    """Get the global recorder instance"""
    return _recorder