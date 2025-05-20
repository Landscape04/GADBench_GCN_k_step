# main.py
import os
import time
import torch
import torch.optim as optim
from data import load_and_split, prepare_dataset
from models import GCN, GAT, GraphSAGE, MultiHopGCN, SelectiveMultiHopGCN, AnomalyGCN
from trainer import Trainer
from utils import evaluate, save_results

def run_experiment(model_name, dataset_name, trial_num, config):
    """运行实验

    Args:
        model_name: 模型名称
        dataset_name: 数据集名称
        trial_num: 实验重复次数
        config: 配置参数
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 预先加载数据
    dataset_path = os.path.join('datasets', f"{dataset_name}.pt")
    if not os.path.exists(dataset_path):
        dataset_path = prepare_dataset(dataset_name)

    # 首次加载显示完整统计
    data = load_and_split(dataset_path, seed=42, show_stats=True)
    print(f"\n=== 开始实验 ===")
    print(f"模型: {model_name.upper()}")
    print(f"数据集: {dataset_name.upper()}")
    print(f"总试验次数: {trial_num}\n")

    results = []
    best_metrics = {
        'auc': {'value': 0.0, 'trial': 0},
        'ap': {'value': 0.0, 'trial': 0},
        'f1': {'value': 0.0, 'trial': 0}
    }

    for trial in range(trial_num):
        trial_start_time = time.time()
        trial_result = {
            'trial': trial+1,
            'status': 'pending',
            'model': model_name.upper(),
            'dataset': dataset_name.upper()
        }

        try:
            # 为每个trial重新分割数据（不显示统计信息）
            data = load_and_split(dataset_path, seed=trial+1, show_stats=False)
            data = data.to(device)

            # 初始化模型
            if model_name.lower() == 'gcn':
                model = GCN(data.x.size(1), config['hidden_dim']).to(device)
            elif model_name.lower() == 'gat':
                model = GAT(nfeat=data.x.size(1),
                          nhid=32,
                          nclass=1,
                          dropout=0.5).to(device)
            elif model_name.lower() == 'sage':
                model = GraphSAGE(nfeat=data.x.size(1),
                                nhid=32,
                                nclass=1,
                                dropout=0.5).to(device)
            elif model_name.lower() == 'multihop':
                model = MultiHopGCN(in_dim=data.x.size(1),
                                  hidden_dim=config['hidden_dim'],
                                  k_hops=config['k_hops'],
                                  dropout=config['dropout']).to(device)
            elif model_name.lower() == 'selective':
                model = SelectiveMultiHopGCN(in_dim=data.x.size(1),
                                           hidden_dim=config['hidden_dim'],
                                           k_hops=config['k_hops'],
                                           dropout=config['dropout']).to(device)
            elif model_name.lower() == 'anomaly':
                model = AnomalyGCN(in_dim=data.x.size(1),
                                hidden_dim=config['hidden_dim'],
                                k_steps=config['k_steps'],
                                dropout=config['dropout']).to(device)
            else:
                raise ValueError(f"不支持的模型类型: {model_name}")

            # 初始化优化器
            optimizer = optim.AdamW(model.parameters(),
                                  lr=config['lr'],
                                  weight_decay=config['weight_decay'])

            # 训练模型
            trainer = Trainer(model, optimizer, device, config)
            epochs = trainer.train(data, trial+1)

            # 测试评估
            test_logits = trainer.test(data)
            test_metrics = evaluate(data.y.cpu(), test_logits, data.test_mask.cpu())

            # 更新最佳指标
            for metric in ['auc', 'ap', 'f1']:
                if test_metrics[metric] > best_metrics[metric]['value']:
                    best_metrics[metric]['value'] = test_metrics[metric]
                    best_metrics[metric]['trial'] = trial + 1

            # 打印当前trial的结果
            trial_time = time.time() - trial_start_time
            print(f"Trial {trial+1}: AUC: {test_metrics['auc']:.3f}, AP: {test_metrics['ap']:.3f}, F1: {test_metrics['f1']:.3f}, Epochs: {epochs}, Time: {trial_time:.2f}s")

            trial_result.update({
                'status': 'completed',
                'test_auc': round(test_metrics['auc'], 3),
                'test_ap': round(test_metrics['ap'], 3),
                'test_f1': round(test_metrics['f1'], 3),
                'epochs': epochs,
                'time': round(time.time() - trial_start_time, 3)
            })

        except Exception as e:
            trial_result.update({
                'status': f'failed: {str(e)}',
                'time': round(time.time() - trial_start_time, 3)
            })
            print(f"Trial {trial+1}: 失败 - {str(e)}")

        results.append(trial_result)

    # 计算平均指标
    avg_metrics = {
        'auc': 0.0,
        'ap': 0.0,
        'f1': 0.0,
        'epochs': 0,
        'time': 0.0,
        'completed_trials': 0
    }

    for result in results:
        if result['status'] == 'completed':
            avg_metrics['auc'] += result['test_auc']
            avg_metrics['ap'] += result['test_ap']
            avg_metrics['f1'] += result['test_f1']
            avg_metrics['epochs'] += result['epochs']
            avg_metrics['time'] += result['time']
            avg_metrics['completed_trials'] += 1

    # 计算平均值
    if avg_metrics['completed_trials'] > 0:
        for key in ['auc', 'ap', 'f1', 'epochs', 'time']:
            avg_metrics[key] /= avg_metrics['completed_trials']

    # 添加最佳指标到每个结果中
    for result in results:
        result.update({
            'best_auc': round(best_metrics['auc']['value'], 3),
            'best_auc_trial': best_metrics['auc']['trial'],
            'best_ap': round(best_metrics['ap']['value'], 3),
            'best_ap_trial': best_metrics['ap']['trial'],
            'best_f1': round(best_metrics['f1']['value'], 3),
            'best_f1_trial': best_metrics['f1']['trial']
        })

    # 打印汇总结果
    print("\n=== 实验结果汇总 ===")
    print(f"完成试验数: {avg_metrics['completed_trials']}/{trial_num}")
    print(f"平均 AUC: {avg_metrics['auc']:.3f}, 最佳 AUC: {best_metrics['auc']['value']:.3f} (Trial {best_metrics['auc']['trial']})")
    print(f"平均 AP: {avg_metrics['ap']:.3f}, 最佳 AP: {best_metrics['ap']['value']:.3f} (Trial {best_metrics['ap']['trial']})")
    print(f"平均 F1: {avg_metrics['f1']:.3f}, 最佳 F1: {best_metrics['f1']['value']:.3f} (Trial {best_metrics['f1']['trial']})")
    print(f"平均训练轮次: {avg_metrics['epochs']:.1f}, 平均耗时: {avg_metrics['time']:.2f}s")

    # 保存结果
    save_results(results)

def main():
    import argparse
    parser = argparse.ArgumentParser(description='图神经网络异常检测')
    parser.add_argument('--model', type=str, default='gcn',
                      choices=['gcn', 'gat', 'sage', 'multihop', 'selective', 'anomaly'],
                      help='选择要使用的模型 (gcn, gat, sage, multihop, selective, anomaly)')
    parser.add_argument('--dataset', type=str, default='reddit',
                      choices=['tolokers', 'reddit', 'questions', 'weibo', 'amazon', 'yelpchi'],
                      help='选择要使用的数据集 (tolokers, reddit, questions, weibo, amazon, yelpchi)')
    parser.add_argument('--trials', type=int, default=10,
                      help='实验重复次数')
    args = parser.parse_args()

    # 配置参数
    config = {
        'hidden_dim': 128,
        'lr': 0.005,
        'weight_decay': 1e-4,
        'patience': 20,
        'delta': 0.00005,
        'warmup_epochs': 10,
        'smooth_window': 3,
        'max_epochs': 100,
        # 多跳模型参数
        'k_hops': 2,      # 多跳模型的跳数
        'k_steps': 2,     # AnomalyGCN的步数
        'dropout': 0.3,   # dropout率
    }

    run_experiment(args.model, args.dataset, args.trials, config)

if __name__ == "__main__":
    main()