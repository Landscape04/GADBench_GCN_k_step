#!/usr/bin/env python3
"""
模型性能对比脚本
基于results文件夹中的实验结果生成详细对比分析
"""

import pandas as pd
import numpy as np
from pathlib import Path

def analyze_results():
    """分析所有实验结果"""
    
    # 手动整理的关键结果数据 (基于查看的实验文件)
    results_data = {
        'Model': ['GraphSAGE', 'GCN', 'GAT', 'NS-GCN'],
        'Dataset': ['Tolokers', 'Tolokers', 'Tolokers', 'Tolokers'],
        'Avg_AUROC': [0.8004, 0.7510, 0.7360, 0.7337],
        'Std_AUROC': [0.0099, 0.0102, 0.0253, 0.0152],
        'Avg_AUPRC': [0.4800, 0.4381, 0.3941, 0.4107],
        'Std_AUPRC': [0.0264, 0.0187, 0.0360, 0.0212],
        'Best_AUROC': [0.8202, 0.7651, 0.7742, 0.7618],
        'Best_AUPRC': [0.5461, 0.4688, 0.4616, 0.4437],
        'Avg_Epochs': [33.5, 31.3, 36.5, 24.6],
        'Avg_Time_per_Trial': [0.6, 0.8, 7.5, 4838.6],
        'Trials': [20, 20, 20, 20]
    }
    
    # Reddit数据集结果 (基于GCN的结果)
    reddit_data = {
        'Model': ['GCN'],
        'Dataset': ['Reddit'],
        'Avg_AUROC': [0.6143],
        'Std_AUROC': [0.0195],
        'Avg_AUPRC': [0.0531],
        'Std_AUPRC': [0.0079],
        'Best_AUROC': [0.6453],
        'Best_AUPRC': [0.0664],
        'Avg_Epochs': [15.0],
        'Avg_Time_per_Trial': [0.2],
        'Trials': [20]
    }
    
    # 合并数据
    all_data = []
    for i in range(len(results_data['Model'])):
        all_data.append({k: v[i] for k, v in results_data.items()})
    
    for i in range(len(reddit_data['Model'])):
        all_data.append({k: v[i] for k, v in reddit_data.items()})
    
    df = pd.DataFrame(all_data)
    
    print("=" * 80)
    print("🏆 GADBench 模型性能对比分析")
    print("=" * 80)
    
    # Tolokers数据集分析
    tolokers_df = df[df['Dataset'] == 'Tolokers'].copy()
    tolokers_df = tolokers_df.sort_values('Avg_AUROC', ascending=False)
    
    print("\n📊 Tolokers数据集性能排名:")
    print("-" * 80)
    print(f"{'排名':<4} {'模型':<12} {'AUROC':<8} {'±std':<8} {'AUPRC':<8} {'±std':<8} {'训练时间':<10}")
    print("-" * 80)
    
    for idx, (_, row) in enumerate(tolokers_df.iterrows(), 1):
        print(f"{idx:<4} {row['Model']:<12} {row['Avg_AUROC']:.4f} ±{row['Std_AUROC']:.4f} "
              f"{row['Avg_AUPRC']:.4f} ±{row['Std_AUPRC']:.4f} {row['Avg_Time_per_Trial']:.1f}s")
    
    print("\n🎯 关键发现:")
    best_model = tolokers_df.iloc[0]
    print(f"• 最佳模型: {best_model['Model']} (AUROC: {best_model['Avg_AUROC']:.4f})")
    print(f"• 最快训练: {tolokers_df.loc[tolokers_df['Avg_Time_per_Trial'].idxmin(), 'Model']} "
          f"({tolokers_df['Avg_Time_per_Trial'].min():.1f}s)")
    print(f"• 最稳定: {tolokers_df.loc[tolokers_df['Std_AUROC'].idxmin(), 'Model']} "
          f"(std: {tolokers_df['Std_AUROC'].min():.4f})")
    
    # 数据集难度对比
    print("\n📈 数据集难度对比:")
    print("-" * 50)
    reddit_row = df[df['Dataset'] == 'Reddit'].iloc[0]
    tolokers_gcn = df[(df['Dataset'] == 'Tolokers') & (df['Model'] == 'GCN')].iloc[0]
    
    print(f"GCN在Tolokers: AUROC={tolokers_gcn['Avg_AUROC']:.4f}, AUPRC={tolokers_gcn['Avg_AUPRC']:.4f}")
    print(f"GCN在Reddit:   AUROC={reddit_row['Avg_AUROC']:.4f}, AUPRC={reddit_row['Avg_AUPRC']:.4f}")
    print(f"性能下降:      AUROC↓{(tolokers_gcn['Avg_AUROC']-reddit_row['Avg_AUROC'])/tolokers_gcn['Avg_AUROC']*100:.1f}%, "
          f"AUPRC↓{(tolokers_gcn['Avg_AUPRC']-reddit_row['Avg_AUPRC'])/tolokers_gcn['Avg_AUPRC']*100:.1f}%")
    
    # 效率分析
    print("\n⚡ 训练效率分析:")
    print("-" * 50)
    tolokers_df_sorted = tolokers_df.sort_values('Avg_Time_per_Trial')
    for _, row in tolokers_df_sorted.iterrows():
        if row['Avg_Time_per_Trial'] < 1:
            efficiency = "⭐⭐⭐⭐⭐"
        elif row['Avg_Time_per_Trial'] < 10:
            efficiency = "⭐⭐⭐⭐"
        elif row['Avg_Time_per_Trial'] < 100:
            efficiency = "⭐⭐⭐"
        else:
            efficiency = "⭐"
        
        print(f"{row['Model']:<12}: {row['Avg_Time_per_Trial']:>8.1f}s {efficiency}")
    
    # 推荐建议
    print("\n💡 模型选择建议:")
    print("-" * 50)
    print("🏆 生产环境推荐: GraphSAGE")
    print("   - 最高性能 (AUROC: 0.8004)")
    print("   - 训练快速 (0.6s/trial)")
    print("   - 结果稳定 (std: 0.0099)")
    
    print("\n🚀 快速原型推荐: GCN")
    print("   - 性能良好 (AUROC: 0.7510)")
    print("   - 极速训练 (0.8s/trial)")
    print("   - 实现简单")
    
    print("\n🔬 研究实验推荐: GAT")
    print("   - 注意力机制可解释")
    print("   - 需要进一步调优")
    print("   - 适合深入分析")
    
    return df

if __name__ == "__main__":
    df = analyze_results()
    
    # 保存详细结果
    df.to_csv('model_performance_summary.csv', index=False)
    print(f"\n📁 详细结果已保存到: model_performance_summary.csv")