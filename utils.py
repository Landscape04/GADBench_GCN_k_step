import os
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score

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

def save_results(results, output_dir='results'):
    """保存实验结果
    
    Args:
        results: 结果列表
        output_dir: 输出目录
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 创建DataFrame
    df = pd.DataFrame(results)
    
    # 重新排列列的顺序
    columns_order = [
        'trial', 'model', 'dataset', 'status',
        'test_auc', 'test_ap', 'test_f1',
        'best_auc', 'best_auc_trial',
        'best_ap', 'best_ap_trial',
        'best_f1', 'best_f1_trial',
        'epochs', 'time'
    ]
    df = df[columns_order]
    
    # 筛选完成的试验
    completed = df[df['status'] == 'completed']
    
    if completed.empty:
        print("\n[Warning] 没有成功完成的试验")
        return
    
    # 保存结果
    output_path = os.path.join(
        output_dir, 
        f'results_{completed["model"].iloc[0]}_{completed["dataset"].iloc[0]}.xlsx')
    df.to_excel(output_path, index=False, float_format='%.3f')
    
    # 打印统计信息
    print("\n=== 最终统计 ===")
    print(f"模型: {completed['model'].iloc[0]}")
    print(f"数据集: {completed['dataset'].iloc[0]}")
    print(f"完成试验数: {len(completed)}/{len(results)}")
    
    if not completed.empty:
        print(f"平均AUC: {completed['test_auc'].mean():.3f} ± {completed['test_auc'].std():.3f}")
        print(f"平均AP: {completed['test_ap'].mean():.3f} ± {completed['test_ap'].std():.3f}")
        print(f"平均F1: {completed['test_f1'].mean():.3f} ± {completed['test_f1'].std():.3f}")
        print(f"平均耗时: {completed['time'].mean():.3f}s/试验")
        print("\n最佳性能:")
        print(f"最高AUC: {completed['best_auc'].max():.3f} (Trial {completed['best_auc_trial'].iloc[0]})")
        print(f"最高AP: {completed['best_ap'].max():.3f} (Trial {completed['best_ap_trial'].iloc[0]})")
        print(f"最高F1: {completed['best_f1'].max():.3f} (Trial {completed['best_f1_trial'].iloc[0]})") 