import os
import time
from copy import deepcopy
from itertools import chain
import torch
import torch.nn.functional as F
from sklearn.manifold import TSNE
from torch import tensor
from torch.optim import Adam
from sklearn.model_selection import StratifiedKFold
from torch_geometric.data import DataLoader, DenseDataLoader as DenseLoader
from torch import Tensor
from torch_geometric.typing import OptTensor
from torch_geometric.nn.conv import MessagePassing
import torch.nn as nn
import numpy as np
import logging
from datetime import datetime
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.distributions.normal import Normal


def preprocess_dataset(dataset):
    valid_data = []
    for data in dataset:
        if data.x is not None and data.edge_index is not None:
            valid_data.append(data)
    return valid_data

def check_data(data):
    print("Checking data...")
    print("Nodes features shape:", data.x.shape if data.x is not None else "None")
    print("Edge index shape:", data.edge_index.shape if data.edge_index is not None else "None")
    print("Number of nodes:", data.num_nodes)
    print("Is edge index correct:", data.contains_isolated_nodes())
    print("Is edge index a valid edge index tensor:", data.is_coalesced())
    print("Does data contain self loops:", data.contains_self_loops())

    # 检查节点特征和边索引是否有None值
    if data.x is None or data.edge_index is None:
        print("Error: Node features or edge indices are None!")
    else:
        print("Data seems to be fine.")
def discrete_gaussian(nums, std=1):
    Dist = Normal(loc=0, scale=1)
    plen, halflen = std * 6 / nums, std * 3 / nums
    posx = torch.arange(-3 * std + halflen, 3 * std, plen)
    result = Dist.cdf(posx + halflen) - Dist.cdf(posx - halflen)
    return result / result.sum()

def get_prior(num_domain, dtype='uniform'):
    assert dtype in ['uniform', 'gaussian'], 'Invalid distribution type'
    # 在dtype为'uniform'的情况下，这行代码创建一个元素全为1的张量（长度为num_domain），
    # 然后将其除以num_domain，从而得到一个元素值都为1/num_domain的张量。这个张量代表了均匀分布的先验概率。
    if dtype == 'uniform':
        prior = torch.ones(num_domain) / num_domain
    else:
        # 在dtype为'gaussian'的情况下，这行代码调用一个名为discrete_gaussian的函数（该函数在代码段之外定义），
        # 并传入num_domain作为参数。这个函数预计会返回一个代表离散高斯分布的先验概率的张量
        prior = discrete_gaussian(num_domain)
    return prior

def set_masks(mask: Tensor, model: nn.Module):
    for module in model.modules():
        if isinstance(module, MessagePassing):
            module.__explain__ = True
            module.__edge_mask__ = mask

def clear_masks(model: nn.Module):
    for module in model.modules():
        if isinstance(module, MessagePassing):
            module.__explain__ = False
            module.__edge_mask__ = None

def num_graphs(data):
    if data.batch is not None:
        return data.num_graphs
    else:
        return data.x.size(0)


def split_into_groups(g):
    unique_groups, unique_counts = torch.unique(
        g, sorted=False, return_counts=True
    )
    group_indices = [
        torch.nonzero(g == group, as_tuple=True)[0]
        for group in unique_groups
    ]
    return unique_groups, group_indices, unique_counts

def plot_curves_ib(train_losses, val_losses, test_accuracies, num_epochs):
    epochs = range(1, num_epochs + 1)

    train_losses = [loss.cpu().item() if isinstance(loss, torch.Tensor) else loss for loss in train_losses]
    val_losses = [loss.cpu().item() if isinstance(loss, torch.Tensor) else loss for loss in val_losses]
    test_accuracies = [acc.cpu().item() if isinstance(acc, torch.Tensor) else acc for acc in test_accuracies]

    plt.figure(figsize=(12, 6))

    # 绘制损失曲线
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, 'b', label='Training Loss')
    plt.plot(epochs, val_losses, 'g', label='Validation Loss')
    plt.title('Losses')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # 绘制准确率曲线
    plt.subplot(1, 2, 2)
    plt.plot(epochs, test_accuracies, 'r', label='Test Accuracy')
    plt.title('Test Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.show()

def plot_curves(train_losses, val_losses, test_losses, val_accuracies, test_accuracies, test_aucs, num_epochs):
    epochs = range(1, num_epochs + 1)

    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, 'b', label='Training Loss')
    plt.plot(epochs, val_losses, 'g', label='Validation Loss')
    plt.plot(epochs, test_losses, 'r', label='Test Loss')
    plt.title('Losses')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, val_accuracies, 'g', label='Validation Accuracy')
    plt.plot(epochs, test_accuracies, 'r', label='Test Accuracy')
    plt.plot(epochs, test_aucs, 'm', label='Test AUC')
    plt.title('Accuracies and AUC')
    plt.xlabel('Epochs')
    plt.ylabel('Score')
    plt.legend()

    plt.tight_layout()
    plt.show()

def plot_curves_new(train_losses, val_accuracies, test_accuracies, test_aucs, num_epochs):
    epochs = range(1, num_epochs + 1)

    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, 'b', label='Training Loss')
    plt.title('Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, val_accuracies, 'g', label='Validation Accuracy')
    plt.plot(epochs, test_accuracies, 'r', label='Test Accuracy')
    plt.plot(epochs, test_aucs, 'm', label='Test AUC')
    plt.title('Accuracies and AUC')
    plt.xlabel('Epochs')
    plt.ylabel('Score')
    plt.ylim(0, 1)  # 将 y 轴的范围固定在 0 到 1 之间
    plt.legend()

    plt.tight_layout()
    plt.show()


def visualize_embeddings(model, loader, device):
    model.eval()

    all_embeddings = []
    all_labels = []
    i = 0

    with torch.no_grad():
        for data in loader:
            i +=1
            data = data.to(device)
            _,_, embeddings = model(data)

            # print(f"embeddings shape: {embeddings.shape}")
            # print(f"labels shape: {data.y.shape}")
            all_embeddings.append(embeddings.cpu().numpy())
            all_labels.append(data.y.cpu().numpy())
            if i == 30:
                break

    all_embeddings = np.concatenate(all_embeddings, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    print(f"Total embeddings shape: {all_embeddings.shape}")
    print(f"Total labels shape: {all_labels.shape}")

    tsne = TSNE(n_components=2, random_state=42)
    embeddings_2d = tsne.fit_transform(all_embeddings)

    plt.figure(figsize=(10, 10))
    scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=all_labels, cmap='viridis', s=15)
    plt.legend(*scatter.legend_elements(), title="Classes")
    plt.title('t-SNE visualization of node embeddings')
    plt.xlabel("t-SNE Dimension 1")
    plt.ylabel("t-SNE Dimension 2")
    plt.show()


def setup_logger(log_dir='./logs/', logger_name='training_logger'):
    # 获取当前时间戳，作为文件的唯一标识
    current_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

    # 创建日志目录（如果不存在）
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # 设置日志文件名，保存在指定的日志目录下
    log_filename = os.path.join(log_dir, f'training_log_{current_time}.log')

    # 检查 logger 是否已经存在，防止重复添加处理器
    logger = logging.getLogger(logger_name)
    if not logger.hasHandlers():
        logger.setLevel(logging.INFO)

        # 创建文件处理器并将日志写入文件
        fh = logging.FileHandler(log_filename)
        fh.setLevel(logging.INFO)

        # 创建格式化器并将其添加到处理器
        formatter = logging.Formatter('%(asctime)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
        fh.setFormatter(formatter)

        # 添加文件处理器到 logger
        logger.addHandler(fh)

    return logger