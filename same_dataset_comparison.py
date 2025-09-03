#!/usr/bin/env python3
"""
同数据集下Baseline vs Enhanced模型对比分析
专门对比GCN vs NS-GCN, GAT vs NS-GAT在相同数据集上的性能
"""

import pandas as pd
import numpy as np

def analyze_same_dataset_comparison():
    """同数据集下baseline与enhanced模型对比"""
    
    # 基于实际实验结果的完整数据
    results_data = [
        # Tolokers数据集
        {'Model': 'GCN', 'Type': 'Baseline', 'Dataset': 'Tolokers', 'AUROC': 0.7510, 'AUPRC': 0.4381, 'Std_AUROC': 0.0102, 'Time': 0.8, 'Epochs': 31.3},
        {'Model': 'NS-GCN', 'Type': 'Enhanced', 'Dataset': 'Tolokers', 'AUROC': 0.7337, 'AUPRC': 0.4107, 'Std_AUROC': 0.0152, 'Time': 4838.6, 'Epochs': 24.6},
        
        {'Model': 'GAT', 'Type': 'Baseline', 'Dataset': 'Tolokers', 'AUROC': 0.7360, 'AUPRC': 0.3941, 'Std_AUROC': 0.0253, 'Time': 7.5, 'Epochs': 36.5},
        {'Model': 'NS-GAT', 'Type': 'Enhanced', 'Dataset': 'Tolokers', 'AUROC': 0.7388, 'AUPRC': 0.3996, 'Std_AUROC': 0.0230, 'Time': 6609.7, 'Epochs': 33.9},
        
        # Reddit数据集
        {'Model': 'GCN', 'Type': 'Baseline', 'Dataset': 'Reddit', 'AUROC': 0.6143, 'AUPRC': 0.0531, 'Std_AUROC': 0.0195, 'Time': 0.2, 'Epochs': 15.0},
        {'Model': 'NS-GCN', 'Type': 'Enhanced', 'Dataset': 'Reddit', 'AUROC': 0.6106, 'AUPRC': 0.0510, 'Std_AUROC': 0.0180, 'Time': 534.9, 'Epochs': 18.1},
        
        {'Model': 'GAT', 'Type': 'Baseline', 'Dataset': 'Reddit', 'AUROC': 0.6611, 'AUPRC': 0.0558, 'Std_AUROC': 0.0244, 'Time': 0.8, 'Epochs': 18.7},
        {'Model': 'NS-GAT', 'Type': 'Enhanced', 'Dataset': 'Reddit', 'AUROC': 0.6599, 'AUPRC': 0.0553, 'Std_AUROC': 0.0202, 'Time': 601.8, 'Epochs': 20.0},
    ]
    
    df = pd.DataFrame(results_data)
    
    print("=" * 80)
    print("🔬 同数据集下 BASELINE vs ENHANCED 模型对比分析")
    print("=" * 80)
    
    # 按数据集分组分析
    for dataset in ['Tolokers', 'Reddit']:
        dataset_df = df[df['Dataset'] == dataset].copy()
        
        print(f"\n📊 {dataset}数据集详细对比:")
        print("=" * 60)
        
        # GCN vs NS-GCN对比
        gcn_data = dataset_df[dataset_df['Model'] == 'GCN'].iloc[0]
        ns_gcn_data = dataset_df[dataset_df['Model'] == 'NS-GCN'].iloc[0]
        
        print(f"\n🔵 GCN vs NS-GCN 对比:")
        print("-" * 40)
        print(f"{'指标':<12} {'GCN (Baseline)':<15} {'NS-GCN (Enhanced)':<18} {'变化':<10}")
        print("-" * 40)
        
        auroc_change = ((ns_gcn_data['AUROC'] - gcn_data['AUROC']) / gcn_data['AUROC']) * 100
        auprc_change = ((ns_gcn_data['AUPRC'] - gcn_data['AUPRC']) / gcn_data['AUPRC']) * 100
        time_ratio = ns_gcn_data['Time'] / gcn_data['Time']
        
        print(f"{'AUROC':<12} {gcn_data['AUROC']:.4f}         {ns_gcn_data['AUROC']:.4f}            {auroc_change:+.1f}%")
        print(f"{'AUPRC':<12} {gcn_data['AUPRC']:.4f}         {ns_gcn_data['AUPRC']:.4f}            {auprc_change:+.1f}%")
        print(f"{'Std_AUROC':<12} {gcn_data['Std_AUROC']:.4f}         {ns_gcn_data['Std_AUROC']:.4f}            {'N/A'}")
        print(f"{'训练时间':<12} {gcn_data['Time']:.1f}s           {ns_gcn_data['Time']:.1f}s           {time_ratio:.0f}x")
        print(f"{'平均轮数':<12} {gcn_data['Epochs']:.1f}           {ns_gcn_data['Epochs']:.1f}             {((ns_gcn_data['Epochs']-gcn_data['Epochs'])/gcn_data['Epochs']*100):+.1f}%")
        
        # GAT vs NS-GAT对比
        gat_data = dataset_df[dataset_df['Model'] == 'GAT'].iloc[0]
        ns_gat_data = dataset_df[dataset_df['Model'] == 'NS-GAT'].iloc[0]
        
        print(f"\n🟠 GAT vs NS-GAT 对比:")
        print("-" * 40)
        print(f"{'指标':<12} {'GAT (Baseline)':<15} {'NS-GAT (Enhanced)':<18} {'变化':<10}")
        print("-" * 40)
        
        auroc_change = ((ns_gat_data['AUROC'] - gat_data['AUROC']) / gat_data['AUROC']) * 100
        auprc_change = ((ns_gat_data['AUPRC'] - gat_data['AUPRC']) / gat_data['AUPRC']) * 100
        time_ratio = ns_gat_data['Time'] / gat_data['Time']
        
        print(f"{'AUROC':<12} {gat_data['AUROC']:.4f}         {ns_gat_data['AUROC']:.4f}            {auroc_change:+.1f}%")
        print(f"{'AUPRC':<12} {gat_data['AUPRC']:.4f}         {ns_gat_data['AUPRC']:.4f}            {auprc_change:+.1f}%")
        print(f"{'Std_AUROC':<12} {gat_data['Std_AUROC']:.4f}         {ns_gat_data['Std_AUROC']:.4f}            {'N/A'}")
        print(f"{'训练时间':<12} {gat_data['Time']:.1f}s           {ns_gat_data['Time']:.1f}s           {time_ratio:.0f}x")
        print(f"{'平均轮数':<12} {gat_data['Epochs']:.1f}           {ns_gat_data['Epochs']:.1f}             {((ns_gat_data['Epochs']-gat_data['Epochs'])/gat_data['Epochs']*100):+.1f}%")
        
        # 数据集总结
        print(f"\n📈 {dataset}数据集总结:")
        print("-" * 30)
        
        if auroc_change > 0:
            gcn_result = f"✅ NS-GCN性能提升{abs(auroc_change):.1f}%"
        else:
            gcn_result = f"❌ NS-GCN性能下降{abs(auroc_change):.1f}%"
            
        gat_auroc_change = ((ns_gat_data['AUROC'] - gat_data['AUROC']) / gat_data['AUROC']) * 100
        if gat_auroc_change > 0:
            gat_result = f"✅ NS-GAT性能提升{abs(gat_auroc_change):.1f}%"
        else:
            gat_result = f"❌ NS-GAT性能下降{abs(gat_auroc_change):.1f}%"
        
        print(f"• {gcn_result}")
        print(f"• {gat_result}")
        print(f"• 计算成本大幅增加: NS-GCN {ns_gcn_data['Time']/gcn_data['Time']:.0f}x, NS-GAT {ns_gat_data['Time']/gat_data['Time']:.0f}x")
    
    # 跨数据集分析
    print(f"\n🌐 跨数据集分析:")
    print("=" * 50)
    
    print("📊 性能变化汇总:")
    print("-" * 30)
    
    # 计算所有对比的性能变化
    comparisons = [
        ('Tolokers', 'GCN', 'NS-GCN', -2.3, -6.3),
        ('Tolokers', 'GAT', 'NS-GAT', +0.4, +1.4),
        ('Reddit', 'GCN', 'NS-GCN', -0.6, -4.0),
        ('Reddit', 'GAT', 'NS-GAT', -0.2, -0.9),
    ]
    
    for dataset, baseline, enhanced, auroc_change, auprc_change in comparisons:
        status = "✅" if auroc_change > 0 else "❌"
        print(f"{status} {dataset:<8} {baseline:<3} → {enhanced:<6}: AUROC{auroc_change:+.1f}%, AUPRC{auprc_change:+.1f}%")
    
    print(f"\n🎯 关键发现:")
    print("-" * 20)
    print("✅ 微小提升: NS-GAT在Tolokers上略有改善")
    print("❌ 多数下降: 3/4的对比显示性能下降")
    print("❌ 成本激增: 训练时间增加500-6000倍")
    print("❌ 效果有限: 最大提升仅0.4% AUROC")
    
    print(f"\n💡 结论与建议:")
    print("=" * 30)
    print("🔍 增强策略评估:")
    print("  • 基于节点度的3阶邻居增强效果不明显")
    print("  • 大部分情况下性能持平或下降")
    print("  • 计算成本与性能收益不成正比")
    
    print("\n🚀 实用建议:")
    print("  • 继续使用baseline模型 (GCN, GAT)")
    print("  • 重新设计增强策略")
    print("  • 考虑更轻量级的图增强方法")
    print("  • 优化邻居采样算法")
    
    return df

if __name__ == "__main__":
    df = analyze_same_dataset_comparison()
    df.to_csv('same_dataset_baseline_vs_enhanced.csv', index=False)
    print(f"\n📁 详细对比数据已保存到: same_dataset_baseline_vs_enhanced.csv")