import os
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
from datetime import datetime
import numpy as np

def evaluate(y_true, logits, mask):
    """评估模型性能
    
    Args:
        y_true: 真实标签
        logits: 模型输出
        mask: 评估掩码
    """
    y = y_true[mask].numpy()
    prob = torch.sigmoid(logits[mask]).numpy()
    pred = (prob > 0.5).astype(int)
    
    # 处理全0或全1的情况
    if len(torch.unique(y_true[mask])) == 1:
        print("[Warning] 验证集只有单一类别")
        return {'auc': 0.5, 'ap': 0.5, 'f1': 0.0}
    
    return {
        'auc': roc_auc_score(y, prob),
        'ap': average_precision_score(y, prob),
        'f1': f1_score(y, pred)
    }

def calculate_metrics(y_true, y_scores, k_values=[50, 100]):
    """
    计算GADBench风格的评估指标
    
    Args:
        y_true: 真实标签 (numpy array or torch tensor)
        y_scores: 预测分数 (numpy array or torch tensor)
        k_values: REC@K的k值列表
    
    Returns:
        dict: 包含AUROC、AUPRC、REC@K等指标的字典
    """
    # 转换为numpy格式
    if torch.is_tensor(y_true):
        y_true = y_true.cpu().numpy()
    if torch.is_tensor(y_scores):
        y_scores = y_scores.cpu().numpy()
    
    metrics = {}
    
    try:
        # AUROC (Area Under ROC Curve)
        auroc = roc_auc_score(y_true, y_scores)
        metrics['AUROC'] = auroc
        
        # AUPRC (Area Under Precision-Recall Curve)
        auprc = average_precision_score(y_true, y_scores)
        metrics['AUPRC'] = auprc
        
        # REC@K (Recall at K)
        for k in k_values:
            rec_at_k = recall_at_k(y_true, y_scores, k)
            metrics[f'REC@{k}'] = rec_at_k
            
    except Exception as e:
        print(f"计算指标时出错: {str(e)}")
        # 返回默认值
        metrics['AUROC'] = 0.5
        metrics['AUPRC'] = np.mean(y_true)
        for k in k_values:
            metrics[f'REC@{k}'] = 0.0
    
    return metrics

def recall_at_k(y_true, y_scores, k):
    """
    计算Recall@K指标
    
    Args:
        y_true: 真实标签
        y_scores: 预测分数
        k: top-k的k值
    
    Returns:
        float: Recall@K值
    """
    if len(y_true) == 0 or k <= 0:
        return 0.0
    
    # 根据预测分数排序，获取top-k的索引
    top_k_indices = np.argsort(y_scores)[-k:]
    
    # 计算top-k中的异常数量
    top_k_anomalies = np.sum(y_true[top_k_indices])
    
    # 计算总异常数量
    total_anomalies = np.sum(y_true)
    
    if total_anomalies == 0:
        return 0.0
    
    # 计算Recall@K
    recall = top_k_anomalies / total_anomalies
    return recall

def generate_experiment_filename(models, datasets, trial_num):
    """
    根据实验范围动态生成Excel文件名
    
    Args:
        models: 模型列表或单个模型名
        datasets: 数据集列表或单个数据集名
        trial_num: 试验次数
    
    Returns:
        str: 生成的文件名
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 处理模型名称
    if isinstance(models, list):
        if len(models) == 1:
            model_str = models[0].upper()
        elif len(models) <= 3:
            model_str = "_".join([m.upper() for m in models])
        else:
            model_str = f"ALL{len(models)}MODELS"
    else:
        model_str = models.upper()
    
    # 处理数据集名称
    if isinstance(datasets, list):
        if len(datasets) == 1:
            dataset_str = datasets[0].upper()
        elif len(datasets) <= 3:
            dataset_str = "_".join([d.upper() for d in datasets])
        else:
            dataset_str = f"ALL{len(datasets)}DATASETS"
    else:
        dataset_str = datasets.upper()
    
    # 生成文件名
    filename = f"results/{model_str}_{dataset_str}_trials{trial_num}_{timestamp}.xlsx"
    return filename

def save_results_realtime(result_data, filename=None):
    """
    实时保存单条实验结果到Excel文件
    
    Args:
        result_data: 单条实验结果字典
        filename: Excel文件路径，如果为None则使用默认文件名
    """
    if filename is None:
        filename = "results/experiment_results.xlsx"
    
    # 确保结果目录存在
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    # 创建DataFrame
    df_new = pd.DataFrame([result_data])
    
    # 如果文件已存在，追加数据
    if os.path.exists(filename):
        try:
            df_existing = pd.read_excel(filename)
            df_combined = pd.concat([df_existing, df_new], ignore_index=True)
        except Exception as e:
            print(f"读取现有文件失败，创建新文件: {str(e)}")
            df_combined = df_new
    else:
        df_combined = df_new
    
    # 保存到Excel（静默保存，不打印信息）
    try:
        df_combined.to_excel(filename, index=False)
    except Exception as e:
        print(f"保存结果失败: {str(e)}")

def save_results(results, dataset_name, model_name, trial_num, filename=None):
    """
    保存实验结果到Excel文件，使用详细的文件命名（保留兼容性）
    
    Args:
        results: 包含指标的字典（支持平均指标和最佳指标）
        dataset_name: 数据集名称
        model_name: 模型名称（已包含top_k标注）
        trial_num: 试验次数
        filename: 输出文件名，如果为None则自动生成
    """
    if filename is None:
        # 生成详细的文件名：模型_数据集_试验数_时间戳.xlsx
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"results/{model_name}_{dataset_name}_trials{trial_num}_{timestamp}.xlsx"
    
    # 确保结果目录存在
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    # 准备数据
    result_data = {
        'Dataset': dataset_name.upper(),
        'Model': model_name,
        'Trial_Count': trial_num,
        'Avg_AUROC': results.get('AUROC', 0.0),
        'Avg_AUPRC': results.get('AUPRC', 0.0),
        'Avg_REC@50': results.get('REC@50', 0.0),
        'Avg_REC@100': results.get('REC@100', 0.0),
        'Best_AUROC': results.get('best_AUROC', 0.0),
        'Best_AUPRC': results.get('best_AUPRC', 0.0),
        'Best_REC@50': results.get('best_REC@50', 0.0),
        'Best_REC@100': results.get('best_REC@100', 0.0),
        'Success_Trials': results.get('trials', 0),
        'Total_Trials': results.get('total_trials', 0),
        'Timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    # 创建DataFrame
    df_new = pd.DataFrame([result_data])
    
    # 如果文件已存在，追加数据
    if os.path.exists(filename):
        try:
            df_existing = pd.read_excel(filename)
            df_combined = pd.concat([df_existing, df_new], ignore_index=True)
        except Exception as e:
            print(f"读取现有文件失败，创建新文件: {str(e)}")
            df_combined = df_new
    else:
        df_combined = df_new
    
    # 保存到Excel
    try:
        df_combined.to_excel(filename, index=False)
        print(f"结果已保存到: {filename}")
    except Exception as e:
        print(f"保存结果失败: {str(e)}")

def print_metrics(metrics, dataset_name, model_name):
    """
    格式化打印评估指标
    
    Args:
        metrics: 指标字典
        dataset_name: 数据集名称
        model_name: 模型名称
    """
    print(f"\n=== {dataset_name.upper()} - {model_name.upper()} 评估结果 ===")
    print(f"AUROC:  {metrics.get('AUROC', 0.0):.4f}")
    print(f"AUPRC:  {metrics.get('AUPRC', 0.0):.4f}")
    print(f"REC@50: {metrics.get('REC@50', 0.0):.4f}")
    print(f"REC@100:{metrics.get('REC@100', 0.0):.4f}")
    print("=" * 50) 