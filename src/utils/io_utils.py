# io_utils.py
"""
Input/Output Utility Functions
"""

import os
import pandas as pd
from datetime import datetime


def generate_experiment_filename(models, datasets, trial_num):
    """
    Generate Excel filename based on experiment scope
    
    Args:
        models: Model list or single model name
        datasets: Dataset list or single dataset name
        trial_num: Number of trials
    
    Returns:
        str: Generated filename
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Handle model names
    if isinstance(models, list):
        if len(models) == 1:
            model_str = models[0].upper()
        elif len(models) <= 3:
            model_str = "_".join([m.upper() for m in models])
        else:
            model_str = f"ALL{len(models)}MODELS"
    else:
        model_str = models.upper()
    
    # Handle dataset names
    if isinstance(datasets, list):
        if len(datasets) == 1:
            dataset_str = datasets[0].upper()
        elif len(datasets) <= 3:
            dataset_str = "_".join([d.upper() for d in datasets])
        else:
            dataset_str = f"ALL{len(datasets)}DATASETS"
    else:
        dataset_str = datasets.upper()
    
    # Generate filename
    filename = f"results/{model_str}_{dataset_str}_trials{trial_num}_{timestamp}.xlsx"
    return filename


def save_results_realtime(result_data, filename=None):
    """
    Save single experiment result to Excel file in real-time
    
    Args:
        result_data: Single experiment result dictionary
        filename: Excel file path, use default if None or empty
    """
    if filename is None or filename == "":
        filename = "results/experiment_results.xlsx"
    
    # Ensure results directory exists
    dirname = os.path.dirname(filename)
    if dirname:
        os.makedirs(dirname, exist_ok=True)
    
    # Create DataFrame
    df_new = pd.DataFrame([result_data])
    
    # If file exists, append data
    if os.path.exists(filename):
        try:
            df_existing = pd.read_excel(filename)
            df_combined = pd.concat([df_existing, df_new], ignore_index=True)
        except Exception as e:
            print(f"Failed to read existing file, creating new: {str(e)}")
            df_combined = df_new
    else:
        df_combined = df_new
    
    # Save to Excel (silent save)
    try:
        df_combined.to_excel(filename, index=False)
    except Exception as e:
        print(f"Failed to save results: {str(e)}")