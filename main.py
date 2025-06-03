# main.py
import os
import time
import torch
import torch.optim as optim
from data import load_and_split, prepare_dataset
from models import GCN, GAT, GraphSAGE, MultiHopGCN, SelectiveMultiHopGCN, NeighborhoodSimilarityGCN, LearnableSelectionGCN
from trainer import Trainer
from utils import calculate_metrics, save_results, print_metrics, save_results_realtime, generate_experiment_filename
import argparse
from sklearn.metrics import roc_auc_score

def run_experiment(model_name, dataset_name, trial_num, config, filename=None):
    """运行实验
    
    Args:
        model_name: 模型名称
        dataset_name: 数据集名称
        trial_num: 实验重复次数
        config: 配置参数
        filename: Excel文件名
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 根据模型名称添加top_k标注
    if model_name in ['neighborhoodsimilaritygcn', 'learnableselectiongcn']:
        model_display_name = f"{model_name.upper()}_top{config['top_k']}"
    else:
        model_display_name = model_name.upper()
    
    # 预先加载数据
    dataset_path = os.path.join('datasets', f"{dataset_name}.pt")
    if not os.path.exists(dataset_path):
        dataset_path = prepare_dataset(dataset_name)
    
    # 首次加载显示完整统计
    data = load_and_split(dataset_path, seed=42, show_stats=True)
    print(f"\n=== 开始实验 ===")
    print(f"模型: {model_display_name}")
    print(f"数据集: {dataset_name.upper()}")
    print(f"总试验次数: {trial_num}\n")
    
    all_trial_results = []
    best_metrics = {
        'AUROC': 0.0,
        'AUPRC': 0.0,
        'REC@50': 0.0,
        'REC@100': 0.0
    }

    for trial in range(trial_num):
        trial_start_time = time.time()
        
        try:
            # 为每个trial重新分割数据（不显示统计信息）
            data = load_and_split(dataset_path, seed=trial+1, show_stats=False)
            data = data.to(device)
            
            # 创建模型（添加dropout支持）
            if model_name == 'gcn':
                model = GCN(in_dim=data.x.shape[1], hidden_dim=128, dropout=config.get('dropout', 0.0)).to(device)
            elif model_name == 'gat':
                model = GAT(nfeat=data.x.shape[1], nhid=128, 
                           nclass=1, heads=8, dropout=config.get('dropout', 0.0)).to(device)
            elif model_name == 'graphsage':
                model = GraphSAGE(nfeat=data.x.shape[1], nhid=128, 
                                 nclass=1, dropout=config.get('dropout', 0.0)).to(device)
            elif model_name == 'multihopgcn':
                model = MultiHopGCN(in_dim=data.x.shape[1], 
                                   hidden_dim=config['hidden_dim'],
                                   k_hops=config['k_hops'],
                                   dropout=config.get('dropout', 0.0)).to(device)
            elif model_name == 'selectivegcn':
                model = SelectiveMultiHopGCN(in_dim=data.x.shape[1], 
                                           hidden_dim=config['hidden_dim'],
                                           k_hops=config['k_hops'],
                                           dropout=config.get('dropout', 0.0)).to(device)
            elif model_name == 'neighborhoodsimilaritygcn':
                model = NeighborhoodSimilarityGCN(
                    in_dim=data.x.shape[1],
                    hidden_dim=config['hidden_dim'],
                    top_k=config['top_k'],
                    dropout=config.get('dropout', 0.0)
                ).to(device)
            elif model_name == 'learnableselectiongcn':
                model = LearnableSelectionGCN(
                    in_dim=data.x.shape[1],
                    hidden_dim=config['hidden_dim'],
                    top_k=config['top_k'],
                    dropout=config.get('dropout', 0.0)
                ).to(device)
            else:
                raise ValueError(f"不支持的模型类型: {model_name}")
            
            # 初始化优化器
            optimizer = optim.AdamW(model.parameters(), 
                                  lr=config['learning_rate'],
                                  weight_decay=config['weight_decay'])
            
            # 训练模型（所有模型都使用早停机制）
            trainer = Trainer(model, optimizer, device, config)
            epochs = trainer.train(data, trial+1)
            
            # 测试模型
            with torch.no_grad():
                model.eval()
                test_logits = model(data.x, data.edge_index)
                test_probs = torch.sigmoid(test_logits)
                
                # 计算GADBench风格的评估指标
                test_metrics = calculate_metrics(
                    y_true=data.y[data.test_mask],
                    y_scores=test_probs[data.test_mask],
                    k_values=[50, 100]
                )
                
                # 更新最佳指标
                for metric in ['AUROC', 'AUPRC', 'REC@50', 'REC@100']:
                    if test_metrics[metric] > best_metrics[metric]:
                        best_metrics[metric] = test_metrics[metric]
                
                # 保存试验结果到列表
                trial_result = {
                    'trial': trial + 1,
                    'dataset': dataset_name.upper(),
                    'model': model_display_name,
                    'AUROC': test_metrics['AUROC'],
                    'AUPRC': test_metrics['AUPRC'],
                    'REC@50': test_metrics['REC@50'],
                    'REC@100': test_metrics['REC@100'],
                    'epochs': epochs,
                    'time': round(time.time() - trial_start_time, 3),
                    'timestamp': time.strftime("%Y-%m-%d %H:%M:%S")
                }
                all_trial_results.append(trial_result)
                
                # 实时保存每个trial的结果到指定Excel文件
                save_results_realtime(trial_result, filename)
                
                # 打印试验结果
                print(f"Trial {trial+1}: AUROC: {test_metrics['AUROC']:.3f}, AUPRC: {test_metrics['AUPRC']:.3f}, REC@50: {test_metrics['REC@50']:.3f}, Epochs: {epochs}")
            
        except Exception as e:
            print(f"Trial {trial+1} 失败: {str(e)}")
            # 失败的trial也要记录
            failed_result = {
                'trial': trial + 1,
                'dataset': dataset_name.upper(),
                'model': model_display_name,
                'AUROC': 0.0,
                'AUPRC': 0.0,
                'REC@50': 0.0,
                'REC@100': 0.0,
                'epochs': 0,
                'time': round(time.time() - trial_start_time, 3),
                'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
                'status': 'FAILED',
                'error': str(e)
            }
            save_results_realtime(failed_result, filename)
            continue
    
    # 计算并显示最终结果
    if len(all_trial_results) > 0:
        avg_auroc = sum(r['AUROC'] for r in all_trial_results) / len(all_trial_results)
        avg_auprc = sum(r['AUPRC'] for r in all_trial_results) / len(all_trial_results)
        avg_rec50 = sum(r['REC@50'] for r in all_trial_results) / len(all_trial_results)
        avg_rec100 = sum(r['REC@100'] for r in all_trial_results) / len(all_trial_results)
        
        print(f"\n{'='*60}")
        print(f"实验完成! {model_display_name} 在 {dataset_name.upper()} 上的结果:")
        print(f"平均 AUROC: {avg_auroc:.4f}")
        print(f"平均 AUPRC: {avg_auprc:.4f}")
        print(f"平均 REC@50: {avg_rec50:.4f}")
        print(f"平均 REC@100: {avg_rec100:.4f}")
        print(f"成功trials: {len(all_trial_results)}/{trial_num}")
        
        # 打印最佳指标
        print(f"最佳 AUROC: {best_metrics['AUROC']:.4f}")
        print(f"最佳 AUPRC: {best_metrics['AUPRC']:.4f}")
        print(f"最佳 REC@50: {best_metrics['REC@50']:.4f}")
        print(f"最佳 REC@100: {best_metrics['REC@100']:.4f}")
        print(f"{'='*60}")
        
        # 保存汇总结果
        summary_result = {
            'dataset': dataset_name.upper(),
            'model': model_display_name,
            'trial_count': trial_num,
            'avg_AUROC': avg_auroc,
            'avg_AUPRC': avg_auprc,
            'avg_REC@50': avg_rec50,
            'avg_REC@100': avg_rec100,
            'best_AUROC': best_metrics['AUROC'],
            'best_AUPRC': best_metrics['AUPRC'],
            'best_REC@50': best_metrics['REC@50'],
            'best_REC@100': best_metrics['REC@100'],
            'success_trials': len(all_trial_results),
            'total_trials': trial_num,
            'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
            'type': 'SUMMARY'
        }
        save_results_realtime(summary_result, filename)
    
    return all_trial_results

def run_experiments_with_filename(models, datasets, trial_num, config):
    """运行指定模型和数据集的实验，使用动态文件名"""
    # 生成文件名
    filename = generate_experiment_filename(models, datasets, trial_num)
    print(f"\n实验结果将保存到: {filename}")
    
    # 确保models和datasets是列表
    if isinstance(models, str):
        models = [models]
    if isinstance(datasets, str):
        datasets = [datasets]
    
    all_results = {}
    total_combinations = len(models) * len(datasets)
    current_combination = 0
    
    for model_name in models:
        all_results[model_name] = {}
        for dataset_name in datasets:
            current_combination += 1
            print(f"\n{'='*60}")
            print(f"进度: {current_combination}/{total_combinations}")
            print(f"当前组合: {model_name.upper()} - {dataset_name.upper()}")
            print(f"{'='*60}")
            
            try:
                results = run_experiment(model_name, dataset_name, trial_num, config, filename)
                all_results[model_name][dataset_name] = results
            except Exception as e:
                print(f"实验失败: {model_name} - {dataset_name}: {str(e)}")
                all_results[model_name][dataset_name] = []
                continue
    
    return all_results

def parse_model_list(model_arg):
    """解析模型参数，支持逗号分隔的多个模型"""
    all_models = ['gcn', 'gat', 'graphsage', 'multihopgcn', 'selectivegcn', 
                  'neighborhoodsimilaritygcn', 'learnableselectiongcn']
    
    if model_arg == 'all':
        return all_models
    else:
        # 支持逗号分隔的多个模型
        models = [m.strip() for m in model_arg.split(',')]
        # 验证模型名称
        for model in models:
            if model not in all_models:
                raise ValueError(f"不支持的模型: {model}")
        return models

def parse_dataset_list(dataset_arg):
    """解析数据集参数，支持逗号分隔的多个数据集"""
    all_datasets = ['reddit', 'weibo', 'tolokers', 'questions']
    
    if dataset_arg == 'all':
        return all_datasets
    else:
        # 支持逗号分隔的多个数据集
        datasets = [d.strip() for d in dataset_arg.split(',')]
        # 验证数据集名称
        for dataset in datasets:
            if dataset not in all_datasets:
                raise ValueError(f"不支持的数据集: {dataset}")
        return datasets

def main():
    parser = argparse.ArgumentParser(description='图神经网络异常检测')
    parser.add_argument('--model', type=str, default='gcn', 
                      help='选择要使用的模型（all表示所有模型，可用逗号分隔多个模型，如gcn,gat,graphsage）')
    parser.add_argument('--dataset', type=str, default='reddit',
                      help='选择要使用的数据集（all表示所有数据集，可用逗号分隔多个数据集，如reddit,weibo）')
    parser.add_argument('--trials', type=int, default=10,
                      help='实验重复次数')
    parser.add_argument('--dropout', type=float, default=0.1,
                      help='Dropout概率')
    args = parser.parse_args()
    
    # 默认配置
    config = {
        'hidden_dim': 64,   # 隐藏层维度
        'learning_rate': 0.01,  # 学习率
        'k_hops': 3,        # k跳邻居数
        'top_k': 3,         # 选择的top-k邻居数
        'patience': 15,     # 早停耐心值（从50改为15，更严格）
        'delta': 0.001,     # 早停改善阈值（从0.00005改为0.001，更严格）
        'warmup_epochs': 5,   # 热身epoch数（从10改为5）
        'smooth_window': 2,   # 平滑窗口大小（从3改为2）
        'max_epochs': 100,   # 最大训练轮数
        'weight_decay': 1e-4,
        'dropout': args.dropout,  # 添加dropout参数
    }
    
    try:
        # 解析模型和数据集列表
        models = parse_model_list(args.model)
        datasets = parse_dataset_list(args.dataset)
        
        print(f"选择的模型: {models}")
        print(f"选择的数据集: {datasets}")
        print(f"试验次数: {args.trials}")
        print(f"Dropout: {args.dropout}")
        
        # 运行实验
        run_experiments_with_filename(models, datasets, args.trials, config)
        
    except ValueError as e:
        print(f"参数错误: {e}")
        print("可用模型: gcn, gat, graphsage, multihopgcn, selectivegcn, neighborhoodsimilaritygcn, learnableselectiongcn")
        print("可用数据集: reddit, weibo, tolokers, questions")

if __name__ == "__main__":
    main()