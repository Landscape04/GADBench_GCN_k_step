"""
数据集下载脚本

支持从DGL下载tolokers和questions数据集，参考GADBench的预处理方法
"""

import os
import torch
import dgl
import numpy as np
from sklearn.preprocessing import StandardScaler
import json

def download_tolokers_dataset(save_dir='datasets'):
    """下载并预处理Tolokers数据集"""
    print("正在下载Tolokers数据集...")
    
    try:
        # 从DGL下载数据集
        dataset = dgl.data.TolokersDataset()
        graph = dataset[0]
        
        # 获取节点特征和标签
        node_features = graph.ndata['feat'].numpy()
        labels = graph.ndata['label'].numpy()
        
        # 获取边索引
        src, dst = graph.edges()
        edge_index = torch.stack([src, dst], dim=0)
        
        # 标准化特征
        scaler = StandardScaler()
        node_features = scaler.fit_transform(node_features)
        
        # 转换为torch格式
        x = torch.FloatTensor(node_features)
        y = torch.LongTensor(labels)
        
        # 保存数据集
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, 'tolokers.pt')
        torch.save({
            'x': x,
            'edge_index': edge_index,
            'y': y,
            'num_nodes': x.size(0),
            'num_edges': edge_index.size(1),
            'num_features': x.size(1),
            'num_classes': len(torch.unique(y))
        }, save_path)
        
        print(f"Tolokers数据集下载完成:")
        print(f"  节点数: {x.size(0)}")
        print(f"  边数: {edge_index.size(1)}")
        print(f"  特征维度: {x.size(1)}")
        print(f"  异常比例: {y.float().mean().item():.2%}")
        print(f"  保存路径: {save_path}")
        
        return save_path
        
    except Exception as e:
        print(f"下载Tolokers数据集失败: {str(e)}")
        return None

def download_questions_dataset(save_dir='datasets'):
    """下载并预处理Questions数据集"""
    print("正在下载Questions数据集...")
    
    try:
        # 从DGL下载数据集
        dataset = dgl.data.QuestionsDataset()
        graph = dataset[0]
        
        # 获取节点特征和标签
        node_features = graph.ndata['feat'].numpy()
        labels = graph.ndata['label'].numpy()
        
        # 获取边索引
        src, dst = graph.edges()
        edge_index = torch.stack([src, dst], dim=0)
        
        # 标准化特征
        scaler = StandardScaler()
        node_features = scaler.fit_transform(node_features)
        
        # 转换为torch格式
        x = torch.FloatTensor(node_features)
        y = torch.LongTensor(labels)
        
        # 保存数据集
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, 'questions.pt')
        torch.save({
            'x': x,
            'edge_index': edge_index,
            'y': y,
            'num_nodes': x.size(0),
            'num_edges': edge_index.size(1),
            'num_features': x.size(1),
            'num_classes': len(torch.unique(y))
        }, save_path)
        
        print(f"Questions数据集下载完成:")
        print(f"  节点数: {x.size(0)}")
        print(f"  边数: {edge_index.size(1)}")
        print(f"  特征维度: {x.size(1)}")
        print(f"  异常比例: {y.float().mean().item():.2%}")
        print(f"  保存路径: {save_path}")
        
        return save_path
        
    except Exception as e:
        print(f"下载Questions数据集失败: {str(e)}")
        return None

def check_dataset_exists(dataset_name, save_dir='datasets'):
    """检查数据集是否已存在"""
    dataset_path = os.path.join(save_dir, f'{dataset_name}.pt')
    return os.path.exists(dataset_path)

def load_or_download_dataset(dataset_name, save_dir='datasets'):
    """加载数据集，如果不存在则下载"""
    if check_dataset_exists(dataset_name, save_dir):
        print(f"{dataset_name}数据集已存在，跳过下载")
        return os.path.join(save_dir, f'{dataset_name}.pt')
    
    if dataset_name == 'tolokers':
        return download_tolokers_dataset(save_dir)
    elif dataset_name == 'questions':
        return download_questions_dataset(save_dir)
    else:
        print(f"不支持的数据集: {dataset_name}")
        return None

def update_download_results(dataset_name, success, save_dir='datasets'):
    """更新下载结果记录"""
    results_file = os.path.join(save_dir, 'download_results.json')
    
    # 读取现有结果
    if os.path.exists(results_file):
        with open(results_file, 'r') as f:
            results = json.load(f)
    else:
        results = {}
    
    # 更新结果
    results[dataset_name] = {
        'success': success,
        'path': os.path.join(save_dir, f'{dataset_name}.pt') if success else None
    }
    
    # 保存结果
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)

def main():
    """主函数"""
    save_dir = 'datasets'
    datasets_to_download = ['tolokers', 'questions']
    
    print("=== 数据集下载工具 ===")
    print(f"保存目录: {save_dir}")
    
    for dataset_name in datasets_to_download:
        print(f"\n--- 处理 {dataset_name} 数据集 ---")
        
        try:
            result_path = load_or_download_dataset(dataset_name, save_dir)
            success = result_path is not None
            update_download_results(dataset_name, success, save_dir)
            
            if success:
                print(f"✓ {dataset_name} 处理成功")
            else:
                print(f"✗ {dataset_name} 处理失败")
                
        except Exception as e:
            print(f"✗ {dataset_name} 处理出错: {str(e)}")
            update_download_results(dataset_name, False, save_dir)
    
    print("\n=== 下载完成 ===")

if __name__ == "__main__":
    main() 