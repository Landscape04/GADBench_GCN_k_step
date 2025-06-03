"""
数据加载和预处理模块

支持的数据集:
1. Reddit (10,984节点)
2. Weibo (8,405节点)
3. Tolokers (11,758节点)
4. Questions (48,921节点)
"""

import os
import torch
import numpy as np
import argparse
from torch_geometric.data import Data
from torch_geometric.transforms import NormalizeFeatures
from torch_geometric.utils import to_undirected
from sklearn.preprocessing import StandardScaler
from torch_geometric.data.storage import GlobalStorage
from torch.serialization import add_safe_globals

# 添加PyG的数据类型到安全加载列表
add_safe_globals([GlobalStorage])

def prepare_dataset(name, root='datasets'):
    """准备数据集
    
    Args:
        name: 数据集名称
        root: 数据存储路径
    
    Returns:
        str: 数据集文件路径
    """
    name = name.lower()
    processed_path = os.path.join(root, f"{name}.pt")
    
    if not os.path.exists(processed_path):
        raise FileNotFoundError(
            f"数据集文件 {processed_path} 不存在。\n"
            f"请先下载数据集到 {root} 目录。\n"
            "您可以使用 download_gadbench_datasets.py 脚本下载数据集。"
        )
    
    # 验证数据集格式
    try:
        data_dict = torch.load(processed_path, map_location='cpu', weights_only=True)
        if not all(k in data_dict for k in ['x', 'edge_index', 'y']):
            raise ValueError("数据集格式不正确，缺少必要的字段")
    except Exception as e:
        raise ValueError(f"数据集文件损坏或格式不正确: {str(e)}")
    
    return processed_path

def load_pt_dataset(file_path):
    """加载.pt格式的数据集
    
    Args:
        file_path: 数据集文件路径
    
    Returns:
        PyG数据对象
    """
    data_dict = torch.load(file_path, map_location='cpu', weights_only=True)
    
    # 转换为PyG数据格式
    x = data_dict['x']
    edge_index = data_dict['edge_index']
    y = data_dict['y']
    
    # 确保边是无向的
    edge_index = to_undirected(edge_index)
    
    # 标准化特征
    scaler = StandardScaler()
    x = torch.FloatTensor(scaler.fit_transform(x))
    
    return Data(x=x, edge_index=edge_index, y=y)

def load_and_split(data_path, train_ratio=0.4, seed=42, show_stats=True):
    """加载并分割数据集
    
    分割策略:
    - 训练集占40%
    - 验证集占30%
    - 测试集占30%
    
    Args:
        data_path: 数据集路径
        train_ratio: 训练集比例，默认0.4
        seed: 随机种子
        show_stats: 是否显示数据集统计信息
    
    Returns:
        PyG数据对象，包含训练、验证和测试掩码
    """
    try:
        # 首先尝试使用weights_only=True加载
        data_dict = torch.load(data_path, map_location='cpu', weights_only=True)
    except Exception as e:
        # 如果失败，尝试完整加载
        print("警告: weights_only加载失败，尝试完整加载...")
        data_dict = torch.load(data_path, map_location='cpu')
    
    # 数据验证
    required_keys = ['x', 'edge_index', 'y']
    assert all(k in data_dict for k in required_keys), "数据集缺少必要字段"
    
    x = data_dict['x'].float()
    edge_index = data_dict['edge_index'].long()
    y = data_dict['y'].long()
    
    num_nodes = x.size(0)
    indices = torch.randperm(num_nodes, generator=torch.Generator().manual_seed(seed))
    
    # 分割比例: 40%训练，30%验证，30%测试
    train_size = int(num_nodes * train_ratio)
    val_size = (num_nodes - train_size) // 2
    
    train_mask = torch.zeros(num_nodes, dtype=bool)
    val_mask = torch.zeros(num_nodes, dtype=bool)
    test_mask = torch.zeros(num_nodes, dtype=bool)
    
    train_mask[indices[:train_size]] = True
    val_mask[indices[train_size:train_size+val_size]] = True
    test_mask[indices[train_size+val_size:]] = True
    
    # 打印数据统计
    if show_stats:
        print("\n=== 数据集统计 ===")
        print(f"节点数: {num_nodes}")
        print(f"边数: {edge_index.shape[1]}")
        print(f"特征维度: {x.shape[1]}")
        print(f"异常比例: {y.float().mean().item():.2%}")
        print(f"训练/验证/测试样本数: {train_mask.sum().item()}/{val_mask.sum().item()}/{test_mask.sum().item()}")
        print(f"训练集比例: {train_mask.sum().item()/num_nodes:.1%}")
        print(f"验证集比例: {val_mask.sum().item()/num_nodes:.1%}")
        print(f"测试集比例: {test_mask.sum().item()/num_nodes:.1%}")
    
    return Data(x=x, 
               edge_index=edge_index, 
               y=y,
               train_mask=train_mask, 
               val_mask=val_mask, 
               test_mask=test_mask)

def get_available_datasets():
    """获取所有可用的数据集列表"""
    return ['reddit', 'weibo', 'tolokers', 'questions']

def get_available_models():
    """获取所有可用的模型列表"""
    return ['gcn', 'gat', 'graphsage', 'gin']

def process_single_dataset(dataset, models=None):
    """处理单个数据集
    
    Args:
        dataset: 数据集名称
        models: 要运行的模型列表，如果为None则不运行模型
    """
    try:
        print(f"\n开始处理 {dataset.upper()}")
        data_path = prepare_dataset(dataset)
        print(f"数据集处理完成: {data_path}")
        
        if models:
            print(f"\n在 {dataset} 上运行以下模型: {', '.join(models)}")
            # TODO: 这里可以添加模型训练的代码
            for model in models:
                print(f"运行模型 {model}...")
    except Exception as e:
        print(f"{dataset} 处理失败: {str(e)}\n")

def process_single_model(model, datasets=None):
    """使用单个模型处理多个数据集
    
    Args:
        model: 模型名称
        datasets: 要处理的数据集列表，如果为None则使用所有数据集
    """
    if datasets is None:
        datasets = get_available_datasets()
    
    print(f"\n使用模型 {model} 处理以下数据集: {', '.join(datasets)}")
    for dataset in datasets:
        try:
            print(f"\n处理数据集 {dataset.upper()}")
            data_path = prepare_dataset(dataset)
            print(f"数据集处理完成: {data_path}")
            # TODO: 这里可以添加模型训练的代码
            print(f"在 {dataset} 上运行 {model}...")
        except Exception as e:
            print(f"在 {dataset} 上运行 {model} 失败: {str(e)}\n")
            continue

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='数据集处理和模型训练工具')
    
    # 添加命令行参数
    parser.add_argument('--dataset', type=str, default=None,
                      help='指定要处理的数据集名称')
    parser.add_argument('--model', type=str, default=None,
                      help='指定要运行的模型名称')
    parser.add_argument('--all-datasets', action='store_true',
                      help='处理所有数据集')
    parser.add_argument('--all-models', action='store_true',
                      help='运行所有模型')
    parser.add_argument('--list', action='store_true',
                      help='列出所有可用的数据集和模型')
    
    args = parser.parse_args()
    
    # 显示可用选项
    if args.list:
        print("\n=== 可用的数据集 ===")
        for dataset in get_available_datasets():
            print(f"- {dataset}")
        
        print("\n=== 可用的模型 ===")
        for model in get_available_models():
            print(f"- {model}")
        exit(0)
    
    # 处理单个数据集上的所有模型
    if args.dataset:
        if args.dataset not in get_available_datasets():
            print(f"错误: 未知的数据集 '{args.dataset}'")
            exit(1)
        models = get_available_models() if args.all_models else ([args.model] if args.model else None)
        process_single_dataset(args.dataset, models)
    
    # 在所有数据集上运行单个模型
    elif args.model:
        if args.model not in get_available_models():
            print(f"错误: 未知的模型 '{args.model}'")
            exit(1)
        datasets = get_available_datasets() if args.all_datasets else None
        process_single_model(args.model, datasets)
    
    # 处理所有数据集
    elif args.all_datasets:
        datasets = get_available_datasets()
        print("\n=== 按复杂度顺序处理所有数据集 ===")
        for dataset in datasets:
            process_single_dataset(dataset, get_available_models() if args.all_models else None)
    
    # 默认行为：显示帮助信息
    else:
        parser.print_help()