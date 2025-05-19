"""
数据加载和预处理模块

支持的数据集(按计算复杂度从低到高排序):

1. Tolokers (复杂度: 低)
   - 节点数: 11,758
   - 边数: 519,000
   - 特征维度: 10
   - 异常比例: 21.8%
   - 描述: 工作协作关系异常检测
   - 特点: 低维特征，数据规模小，适合快速实验

2. Reddit (复杂度: 中低)
   - 节点数: 10,984
   - 边数: 168,016
   - 特征维度: 64
   - 异常比例: 3.3%
   - 描述: 社交网络异常用户检测
   - 特点: 特征维度适中，边密度适中

3. Questions (复杂度: 中)
   - 节点数: 48,921
   - 边数: 153,540
   - 特征维度: 301
   - 异常比例: 3.0%
   - 描述: 问答系统异常检测
   - 特点: 节点数适中，高维特征

4. Weibo (复杂度: 中高)
   - 节点数: 8,405
   - 边数: 407,963
   - 特征维度: 400
   - 异常比例: 10.3%
   - 描述: 微博平台异常用户检测
   - 特点: 超高维特征，边密度较高

5. Amazon (复杂度: 高)
   - 节点数: 11,944
   - 边数: 4,398,392
   - 特征维度: 25
   - 异常比例: 9.5%
   - 描述: 电商评论异常检测
   - 特点: 边密度极高，图结构复杂

6. YelpChi (复杂度: 极高)
   - 节点数: 45,954
   - 边数: 3,846,979
   - 特征维度: 32
   - 异常比例: 14.5%
   - 描述: 餐厅评论异常检测
   - 特点: 大规模节点，高边密度
"""

import os
import torch
import numpy as np
import argparse
from torch_geometric.data import Data
from torch_geometric.transforms import NormalizeFeatures
from torch_geometric.utils import to_undirected
from sklearn.preprocessing import StandardScaler
import urllib.request
import zipfile

def download_dataset(name, root='datasets'):
    """下载并解压数据集
    
    Args:
        name: 数据集名称 ('reddit', 'weibo', 'tolokers', 'questions', 'amazon', 'yelpchi')
        root: 数据存储路径
    """
    os.makedirs(root, exist_ok=True)
    
    # 数据集URL映射
    dataset_urls = {
        'reddit': 'https://raw.githubusercontent.com/pygod-team/data/main/reddit.pt.zip',
        'weibo': 'https://raw.githubusercontent.com/pygod-team/data/main/weibo.pt.zip',
        'tolokers': 'https://data.dgl.ai/dataset/tolokers.zip',
        'questions': 'https://data.dgl.ai/dataset/questions.zip',
        'amazon': 'https://data.dgl.ai/dataset/fraud/amazon.zip',
        'yelpchi': 'https://data.dgl.ai/dataset/fraud/yelp.zip'
    }
    
    if name.lower() not in dataset_urls:
        raise ValueError(f"不支持的数据集: {name}")
    
    url = dataset_urls[name.lower()]
    zip_filename = os.path.join(root, f"{name.lower()}.zip")
    pt_filename = os.path.join(root, f"{name.lower()}.pt")
    
    if not os.path.exists(pt_filename):
        print(f"下载数据集 {name}...")
        
        # 设置请求头
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        # 创建请求
        req = urllib.request.Request(url, headers=headers)
        
        try:
            # 下载文件
            with urllib.request.urlopen(req) as response, open(zip_filename, 'wb') as out_file:
                data = response.read()
                out_file.write(data)
            
            print("下载完成，正在解压...")
            
            # 解压zip文件
            with zipfile.ZipFile(zip_filename, 'r') as zip_ref:
                zip_ref.extractall(root)
                
            # 删除zip文件
            os.remove(zip_filename)
            print("解压完成")
            
        except urllib.error.HTTPError as e:
            print(f"下载失败 (HTTP {e.code}): {e.reason}")
            print("尝试使用备用下载源...")
            
            # 备用下载源
            backup_urls = {
                'tolokers': 'https://github.com/dmlc/dgl/raw/master/dataset/tolokers.zip',
                'questions': 'https://github.com/dmlc/dgl/raw/master/dataset/questions.zip',
                'amazon': 'https://github.com/dmlc/dgl/raw/master/dataset/fraud/amazon.zip',
                'yelpchi': 'https://github.com/dmlc/dgl/raw/master/dataset/fraud/yelp.zip'
            }
            
            if name.lower() in backup_urls:
                backup_url = backup_urls[name.lower()]
                req = urllib.request.Request(backup_url, headers=headers)
                
                try:
                    with urllib.request.urlopen(req) as response, open(zip_filename, 'wb') as out_file:
                        data = response.read()
                        out_file.write(data)
                    
                    print("备用源下载完成，正在解压...")
                    
                    with zipfile.ZipFile(zip_filename, 'r') as zip_ref:
                        zip_ref.extractall(root)
                    
                    os.remove(zip_filename)
                    print("解压完成")
                    
                except Exception as e2:
                    raise Exception(f"备用源下载也失败了: {str(e2)}")
            else:
                raise e
        
        except Exception as e:
            raise Exception(f"下载失败: {str(e)}")
    
    return pt_filename

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

def get_dataset_complexity(name):
    """获取数据集的复杂度信息
    
    Args:
        name: 数据集名称
    
    Returns:
        dict: 包含复杂度信息的字典
    """
    complexity_info = {
        'tolokers': {
            'complexity': 'low',
            'description': '低维特征，数据规模小，适合快速实验',
            'estimated_memory': '2GB'
        },
        'reddit': {
            'complexity': 'medium-low',
            'description': '特征维度适中，边密度适中',
            'estimated_memory': '4GB'
        },
        'questions': {
            'complexity': 'medium',
            'description': '节点数适中，高维特征',
            'estimated_memory': '4GB'
        },
        'weibo': {
            'complexity': 'medium-high',
            'description': '超高维特征，边密度较高',
            'estimated_memory': '8GB'
        },
        'amazon': {
            'complexity': 'high',
            'description': '边密度极高，图结构复杂',
            'estimated_memory': '8GB'
        },
        'yelpchi': {
            'complexity': 'very-high',
            'description': '大规模节点，高边密度',
            'estimated_memory': '16GB'
        }
    }
    
    name = name.lower()
    if name in complexity_info:
        return complexity_info[name]
    else:
        raise ValueError(f"未知数据集: {name}")

def prepare_dataset(name, root='datasets', check_resources=True):
    """准备数据集（下载、处理和保存）
    
    Args:
        name: 数据集名称
        root: 数据存储路径
        check_resources: 是否检查资源需求
    
    Returns:
        处理后的数据集文件路径
    """
    if check_resources:
        # 检查复杂度信息
        complexity = get_dataset_complexity(name)
        print(f"\n=== 数据集复杂度信息 ===")
        print(f"复杂度等级: {complexity['complexity']}")
        print(f"特点: {complexity['description']}")
        print(f"预估内存需求: {complexity['estimated_memory']}")
        
        if complexity['complexity'] in ['high', 'very-high']:
            print("\n注意: 该数据集计算复杂度较高，处理时间可能较长")
            if not torch.cuda.is_available():
                print("建议使用GPU进行加速")
            response = input("是否继续? [y/N]: ")
            if response.lower() != 'y':
                raise ValueError("用户取消操作")
    
    os.makedirs(root, exist_ok=True)
    name = name.lower()
    processed_path = os.path.join(root, f"{name}.pt")
    
    # 检查是否已存在处理好的数据集
    if os.path.exists(processed_path):
        print(f"\n发现本地缓存数据集: {processed_path}")
        try:
            # 验证缓存数据是否完整
            data_dict = torch.load(processed_path, map_location='cpu', weights_only=True)
            if all(k in data_dict for k in ['x', 'edge_index', 'y']):
                print("缓存数据验证成功，直接加载本地数据集...")
                return processed_path
            else:
                print("缓存数据不完整，重新处理数据集...")
        except Exception as e:
            print(f"缓存数据加载失败 ({str(e)})，重新处理数据集...")
    else:
        print(f"\n未找到本地缓存，开始下载和处理数据集: {name}")
    
    # 下载并加载数据集
    file_path = download_dataset(name, root)
    data = load_pt_dataset(file_path)
    
    # 打印数据集信息
    print(f"\n=== 数据集: {name.upper()} ===")
    print(f"节点数: {data.num_nodes}")
    print(f"边数: {data.num_edges}")
    print(f"特征维度: {data.num_features}")
    print(f"异常比例: {data.y.float().mean().item():.2%}")
    
    # 保存处理后的数据集
    print(f"\n保存处理后的数据集...")
    torch.save({
        'x': data.x,
        'edge_index': data.edge_index,
        'y': data.y
    }, processed_path)
    
    print(f"数据集已保存至: {processed_path}")
    return processed_path

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
    # 加载数据
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
    """获取所有可用的数据集列表，按复杂度排序"""
    return ['tolokers', 'reddit', 'questions', 'weibo', 'amazon', 'yelpchi']

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
        complexity = get_dataset_complexity(dataset)
        print(f"\n开始处理 {dataset.upper()} (复杂度: {complexity['complexity']})")
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
            complexity = get_dataset_complexity(dataset)
            print(f"\n处理数据集 {dataset.upper()} (复杂度: {complexity['complexity']})")
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
            complexity = get_dataset_complexity(dataset)
            print(f"- {dataset} (复杂度: {complexity['complexity']})")
        
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