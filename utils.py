import os
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
from datetime import datetime

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

def save_results(results):
    """保存实验结果到CSV文件
    
    Args:
        results: 包含实验结果的字典列表
    """
    # 确保results目录存在
    os.makedirs('results', exist_ok=True)
    
    # 转换为DataFrame
    df = pd.DataFrame(results)
    
    # 处理失败的trial，填充缺失值
    metrics = ['test_auc', 'test_ap', 'test_f1', 'epochs']
    for metric in metrics:
        if metric not in df.columns:
            df[metric] = None
    
    # 重新排序列
    columns_order = [
        'trial', 'model', 'dataset', 'status',
        'test_auc', 'test_ap', 'test_f1', 'epochs',
        'time',
        'best_auc', 'best_auc_trial',
        'best_ap', 'best_ap_trial',
        'best_f1', 'best_f1_trial'
    ]
    
    # 只选择存在的列
    existing_columns = [col for col in columns_order if col in df.columns]
    df = df[existing_columns]
    
    # 生成文件名
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    model_name = results[0]['model'].lower()
    dataset_name = results[0]['dataset'].lower()
    filename = f'results/{model_name}_{dataset_name}_{timestamp}.csv'
    
    # 保存结果
    df.to_csv(filename, index=False)
    print(f"\n结果已保存到: {filename}") 