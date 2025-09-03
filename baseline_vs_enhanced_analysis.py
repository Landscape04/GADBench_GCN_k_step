#!/usr/bin/env python3
"""
Baseline vs Enhanced Models Analysis
对比基础模型与增强版本的性能差异
"""

import pandas as pd
import numpy as np

def analyze_baseline_vs_enhanced():
    """分析baseline与enhanced模型的性能对比"""
    
    # 基于实际查看的实验结果整理的数据
    results_data = [
        # Tolokers数据集 - Baseline模型
        {'Model': 'GCN', 'Type': 'Baseline', 'Dataset': 'Tolokers', 'AUROC': 0.7510, 'AUPRC': 0.4381, 'Std_AUROC': 0.0102, 'Time': 0.8, 'Trials': 20},
        {'Model': 'GAT', 'Type': 'Baseline', 'Dataset': 'Tolokers', 'AUROC': 0.7360, 'AUPRC': 0.3941, 'Std_AUROC': 0.0253, 'Time': 7.5, 'Trials': 20},
        {'Model': 'GraphSAGE', 'Type': 'Baseline', 'Dataset': 'Tolokers', 'AUROC': 0.8004, 'AUPRC': 0.4800, 'Std_AUROC': 0.0099, 'Time': 0.6, 'Trials': 20},
        
        # Tolokers数据集 - Enhanced模型
        {'Model': 'ANS-GCN', 'Type': 'Enhanced', 'Dataset': 'Tolokers', 'AUROC': 0.7313, 'AUPRC': 0.4096, 'Std_AUROC': 0.0155, 'Time': 3554.0, 'Trials': 10},
        {'Model': 'NS-GAT', 'Type': 'Enhanced', 'Dataset': 'Tolokers', 'AUROC': 0.7388, 'AUPRC': 0.3996, 'Std_AUROC': 0.0230, 'Time': 6609.7, 'Trials': 20},
        {'Model': 'NS-GCN', 'Type': 'Enhanced', 'Dataset': 'Tolokers', 'AUROC': 0.7337, 'AUPRC': 0.4107, 'Std_AUROC': 0.0152, 'Time': 4838.6, 'Trials': 20},
        
        # Reddit数据集 - Baseline模型
        {'Model': 'GCN', 'Type': 'Baseline', 'Dataset': 'Reddit', 'AUROC': 0.6143, 'AUPRC': 0.0531, 'Std_AUROC': 0.0195, 'Time': 0.2, 'Trials': 20},
        
        # Reddit数据集 - Enhanced模型  
        {'Model': 'ANS-GCN', 'Type': 'Enhanced', 'Dataset': 'Reddit', 'AUROC': 0.6120, 'AUPRC': 0.0517, 'Std_AUROC': 0.0184, 'Time': 597.4, 'Trials': 20},
    ]
    
    df = pd.DataFrame(results_data)
    
    print("=" * 80)
    print("🔬 BASELINE vs ENHANCED 模型性能对比分析")
    print("=" * 80)
    
    # 按数据集分组分析
    for dataset in ['Tolokers', 'Reddit']:
        dataset_df = df[df['Dataset'] == dataset].copy()
        
        print(f"\n📊 {dataset}数据集分析:")
        print("-" * 60)
        
        # 创建对比表格
        baseline_df = dataset_df[dataset_df['Type'] == 'Baseline'].copy()
        enhanced_df = dataset_df[dataset_df['Type'] == 'Enhanced'].copy()
        
        print(f"{'模型':<12} {'类型':<10} {'AUROC':<8} {'AUPRC':<8} {'训练时间':<10} {'试验数':<6}")
        print("-" * 60)
        
        # 显示baseline结果
        for _, row in baseline_df.iterrows():
            print(f"{row['Model']:<12} {row['Type']:<10} {row['AUROC']:.4f} {row['AUPRC']:.4f} {row['Time']:>8.1f}s {row['Trials']:>4}")
        
        print()
        # 显示enhanced结果
        for _, row in enhanced_df.iterrows():
            print(f"{row['Model']:<12} {row['Type']:<10} {row['AUROC']:.4f} {row['AUPRC']:.4f} {row['Time']:>8.1f}s {row['Trials']:>4}")
        
        # 计算改进情况
        print(f"\n🎯 {dataset}数据集改进分析:")
        print("-" * 40)
        
        if dataset == 'Tolokers':
            # GCN vs ANS-GCN
            gcn_baseline = baseline_df[baseline_df['Model'] == 'GCN'].iloc[0]
            ans_gcn = enhanced_df[enhanced_df['Model'] == 'ANS-GCN'].iloc[0]
            
            auroc_change = ((ans_gcn['AUROC'] - gcn_baseline['AUROC']) / gcn_baseline['AUROC']) * 100
            auprc_change = ((ans_gcn['AUPRC'] - gcn_baseline['AUPRC']) / gcn_baseline['AUPRC']) * 100
            
            print(f"GCN → ANS-GCN:")
            print(f"  AUROC: {gcn_baseline['AUROC']:.4f} → {ans_gcn['AUROC']:.4f} ({auroc_change:+.1f}%)")
            print(f"  AUPRC: {gcn_baseline['AUPRC']:.4f} → {ans_gcn['AUPRC']:.4f} ({auprc_change:+.1f}%)")
            print(f"  时间成本: {gcn_baseline['Time']:.1f}s → {ans_gcn['Time']:.1f}s ({ans_gcn['Time']/gcn_baseline['Time']:.0f}x)")
            
            # GAT vs NS-GAT
            gat_baseline = baseline_df[baseline_df['Model'] == 'GAT'].iloc[0]
            ns_gat = enhanced_df[enhanced_df['Model'] == 'NS-GAT'].iloc[0]
            
            auroc_change = ((ns_gat['AUROC'] - gat_baseline['AUROC']) / gat_baseline['AUROC']) * 100
            auprc_change = ((ns_gat['AUPRC'] - gat_baseline['AUPRC']) / gat_baseline['AUPRC']) * 100
            
            print(f"\nGAT → NS-GAT:")
            print(f"  AUROC: {gat_baseline['AUROC']:.4f} → {ns_gat['AUROC']:.4f} ({auroc_change:+.1f}%)")
            print(f"  AUPRC: {gat_baseline['AUPRC']:.4f} → {ns_gat['AUPRC']:.4f} ({auprc_change:+.1f}%)")
            print(f"  时间成本: {gat_baseline['Time']:.1f}s → {ns_gat['Time']:.1f}s ({ns_gat['Time']/gat_baseline['Time']:.0f}x)")
            
        elif dataset == 'Reddit':
            # GCN vs ANS-GCN (Reddit)
            gcn_baseline = baseline_df[baseline_df['Model'] == 'GCN'].iloc[0]
            ans_gcn = enhanced_df[enhanced_df['Model'] == 'ANS-GCN'].iloc[0]
            
            auroc_change = ((ans_gcn['AUROC'] - gcn_baseline['AUROC']) / gcn_baseline['AUROC']) * 100
            auprc_change = ((ans_gcn['AUPRC'] - gcn_baseline['AUPRC']) / gcn_baseline['AUPRC']) * 100
            
            print(f"GCN → ANS-GCN:")
            print(f"  AUROC: {gcn_baseline['AUROC']:.4f} → {ans_gcn['AUROC']:.4f} ({auroc_change:+.1f}%)")
            print(f"  AUPRC: {gcn_baseline['AUPRC']:.4f} → {ans_gcn['AUPRC']:.4f} ({auprc_change:+.1f}%)")
            print(f"  时间成本: {gcn_baseline['Time']:.1f}s → {ans_gcn['Time']:.1f}s ({ans_gcn['Time']/gcn_baseline['Time']:.0f}x)")
    
    # 总体结论
    print(f"\n🎉 总体结论:")
    print("=" * 50)
    
    print("📈 性能改进情况:")
    print("• Tolokers数据集:")
    print("  - GCN → ANS-GCN: AUROC下降2.6%, AUPRC下降6.5%")
    print("  - GAT → NS-GAT: AUROC提升0.4%, AUPRC提升1.4%")
    print("• Reddit数据集:")
    print("  - GCN → ANS-GCN: AUROC下降0.4%, AUPRC下降2.6%")
    
    print("\n⚡ 计算成本:")
    print("• 增强版本训练时间显著增加:")
    print("  - ANS-GCN: 4000+倍时间成本")
    print("  - NS-GAT: 880倍时间成本")
    print("  - NS-GCN: 6000+倍时间成本")
    
    print("\n💡 关键发现:")
    print("❌ 基于节点度的3阶邻居增强结构未带来显著性能提升")
    print("❌ 大部分情况下性能略有下降")
    print("❌ 计算成本大幅增加，性价比低")
    print("✅ 只有NS-GAT在Tolokers上有微小提升")
    
    print("\n🔧 建议:")
    print("• 重新评估增强策略的有效性")
    print("• 考虑优化邻居采样算法")
    print("• 探索更高效的图增强方法")
    print("• 在当前设置下，建议使用baseline模型")
    
    return df

if __name__ == "__main__":
    df = analyze_baseline_vs_enhanced()
    df.to_csv('baseline_vs_enhanced_comparison.csv', index=False)
    print(f"\n📁 详细对比数据已保存到: baseline_vs_enhanced_comparison.csv")