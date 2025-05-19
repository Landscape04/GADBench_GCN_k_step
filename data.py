"""
数据加载和预处理模块

支持的数据集:
1. Cora
   - 节点数: 2708
   - 边数: 5429
   - 特征维度: 1433
   - 描述: 机器学习论文引文网络

2. Citeseer
   - 节点数: 3327
   - 边数: 4732
   - 特征维度: 3703
   - 描述: 计算机科学论文引文网络

3. PubMed
   - 节点数: 19717
   - 边数: 44338
   - 特征维度: 500
   - 描述: 生物医学论文引文网络

4. Amazon Computers
   - 节点数: 13752
   - 边数: 245861
   - 特征维度: 767
   - 描述: 计算机产品的共同购买网络
"""

import os
import importlib
import torch
import numpy as np
from torch_geometric.data import Data
from torch_geometric.datasets import Planetoid, Amazon
from torch_geometric.transforms import NormalizeFeatures
from torch_geometric.utils import to_undirected
from sklearn.preprocessing import StandardScaler

def allow_torch_geometric_globals():
    """安全注册PyG需要的全局类"""
    try:
        storage_module = importlib.import_module('torch_geometric.data.storage')
        classes = [
            getattr(storage_module, 'GlobalStorage'),
            getattr(storage_module, 'NodeStorage'),
            getattr(storage_module, 'EdgeStorage'),
        ]
        torch.serialization.add_safe_globals(classes)
        return True
    except Exception as e:
        print(f"[Warning] 安全类注册失败: {str(e)}")
        return False

def inject_anomalies(data, anomaly_ratio=0.1, seed=42):
    """注入异常样本，将数据集转换为异常检测任务
    
    步骤:
    1. 随机选择一部分节点作为异常样本
    2. 对异常节点的特征添加高斯噪声
    3. 标准化所有节点的特征
    
    Args:
        data: PyG数据对象
        anomaly_ratio: 异常样本比例，默认10%
        seed: 随机种子，确保结果可复现
    
    Returns:
        添加了异常标签和扰动特征的数据对象
    """
    rng = np.random.RandomState(seed)
    num_nodes = data.x.size(0)
    num_anomalies = int(num_nodes * anomaly_ratio)
    
    # 随机选择节点作为异常
    anomaly_idx = rng.choice(num_nodes, num_anomalies, replace=False)
    labels = torch.zeros(num_nodes, dtype=torch.long)
    labels[anomaly_idx] = 1
    
    # 对异常节点的特征进行扰动
    perturbed_features = data.x.clone()
    noise = torch.tensor(rng.normal(0, 2, size=perturbed_features[anomaly_idx].shape), 
                        dtype=torch.float32)
    perturbed_features[anomaly_idx] += noise
    
    # 标准化特征
    scaler = StandardScaler()
    perturbed_features = torch.tensor(
        scaler.fit_transform(perturbed_features), 
        dtype=torch.float32
    )
    
    data.x = perturbed_features
    data.y = labels
    return data

def prepare_dataset(name, root='datasets', anomaly_ratio=0.1, seed=42):
    """准备数据集（下载、处理和保存）
    
    Args:
        name: 数据集名称 ('cora', 'citeseer', 'pubmed', 'computers')
        root: 数据存储路径
        anomaly_ratio: 异常样本比例
        seed: 随机种子
    
    Returns:
        处理后的数据集文件路径
    """
    os.makedirs(root, exist_ok=True)
    name = name.lower()
    processed_path = os.path.join(root, f"{name}.pt")
    
    # 检查是否已存在处理好的数据集
    if os.path.exists(processed_path):
        print(f"\n发现本地缓存数据集: {processed_path}")
        try:
            # 验证缓存数据是否完整
            data_dict = torch.load(processed_path, map_location='cpu')
            if all(k in data_dict for k in ['x', 'edge_index', 'y']):
                print("缓存数据验证成功，直接加载本地数据集...")
                return processed_path
            else:
                print("缓存数据不完整，重新处理数据集...")
        except Exception as e:
            print(f"缓存数据加载失败 ({str(e)})，重新处理数据集...")
    else:
        print(f"\n未找到本地缓存，开始下载和处理数据集: {name}")
    
    transform = NormalizeFeatures()
    
    # 数据集基本信息
    dataset_info = {
        'cora': {'nodes': 2708, 'edges': 5429, 'features': 1433},
        'citeseer': {'nodes': 3327, 'edges': 4732, 'features': 3703},
        'pubmed': {'nodes': 19717, 'edges': 44338, 'features': 500},
        'computers': {'nodes': 13752, 'edges': 245861, 'features': 767}
    }
    
    if name in ['cora', 'citeseer', 'pubmed']:
        print(f"下载 {name} 数据集...")
        dataset = Planetoid(root=root, name=name, transform=transform)
        data = dataset[0]
        print("下载完成")
        
    elif name == 'computers':
        print(f"下载 Amazon Computers 数据集...")
        dataset = Amazon(root=root, name='Computers', transform=transform)
        data = dataset[0]
        print("下载完成")
        
    else:
        raise ValueError(f"未知的数据集: {name}")
    
    # 确保边是无向的
    print("处理数据集...")
    data.edge_index = to_undirected(data.edge_index)
    
    # 注入异常
    print(f"注入异常样本 (比例: {anomaly_ratio:.1%})...")
    data = inject_anomalies(data, anomaly_ratio=anomaly_ratio, seed=seed)
    
    # 打印数据集信息
    info = dataset_info.get(name, {})
    print(f"\n=== 数据集: {name.upper()} ===")
    print(f"预期规模:")
    print(f"  - 节点数: {info.get('nodes', '未知')}")
    print(f"  - 边数: {info.get('edges', '未知')}")
    print(f"  - 特征维度: {info.get('features', '未知')}")
    print(f"\n实际规模:")
    print(f"  - 节点数: {data.num_nodes}")
    print(f"  - 边数: {data.num_edges}")
    print(f"  - 特征维度: {data.num_features}")
    print(f"  - 异常比例: {data.y.float().mean().item():.2%}")
    
    # 保存处理后的数据集
    print(f"\n保存处理后的数据集...")
    torch.save({
        'x': data.x,
        'edge_index': data.edge_index,
        'y': data.y
    }, processed_path)
    
    print(f"数据集已保存至: {processed_path}")
    return processed_path

def load_and_split(data_path, train_ratio=0.4, seed=42, show_stats=True, trial=None):
    """加载并分割数据集
    
    Args:
        data_path: 数据集路径
        train_ratio: 训练集比例
        seed: 随机种子
        show_stats: 是否显示数据集统计信息
        trial: 当前试验编号
    
    Returns:
        PyG数据对象，包含训练、验证和测试掩码
    """
    # 尝试安全加载模式
    security_enabled = allow_torch_geometric_globals()
    
    try:
        data_dict = torch.load(data_path, map_location='cpu', 
                             weights_only=security_enabled)
    except Exception as e:
        print(f"[Info] 安全加载失败 ({str(e)}), 尝试非安全模式")
        data_dict = torch.load(data_path, map_location='cpu', weights_only=False)

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
        if trial is not None:
            print(f"\n=== Trial {trial} 数据集统计 ===")
        else:
            print("\n=== 数据集统计 ===")
        print(f"节点数: {num_nodes}")
        print(f"边数: {edge_index.shape[1]}")
        print(f"特征维度: {x.shape[1]}")
        print(f"异常比例: {y.float().mean().item():.2%}")
        print(f"训练/验证/测试样本数: {train_mask.sum().item()}/{val_mask.sum().item()}/{test_mask.sum().item()}")

    return Data(x=x, edge_index=edge_index, y=y, 
               train_mask=train_mask, val_mask=val_mask, test_mask=test_mask)

if __name__ == "__main__":
    # 下载并处理所有数据集
    datasets = ['cora', 'citeseer', 'pubmed', 'computers']
    for dataset in datasets:
        try:
            prepare_dataset(dataset)
            print(f"{dataset} 处理完成\n")
        except Exception as e:
            print(f"{dataset} 处理失败: {str(e)}\n")