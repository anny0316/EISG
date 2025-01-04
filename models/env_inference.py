import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import device
from tqdm import tqdm

from informal.models.GIB import GIB
import torch
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU, BatchNorm1d as BN
from torch_geometric.nn import GINConv, global_mean_pool, JumpingKnowledge


class DomainClassifier(nn.Module):
    def __init__(self, backend_dim, backend, num_domain, num_task):
        super(DomainClassifier, self).__init__()
        self.backend = backend  # 图神经网络作为 backend
        self.num_task = num_task  # 任务数量
        self.predictor = nn.Linear(backend_dim + num_task, num_domain)  # 最终分类器

    def forward(self, data):
        # 使用 backend 提取图特征
        graph_feat = self.backend(data)

        # 从输入数据中获取 gt_label，并处理为适当的形状
        y_part = torch.nan_to_num(data.y).float()  #  data.y 是标签
        y_part = y_part.reshape(len(y_part), self.num_task)

        # 将图特征与标签结合，然后通过分类器
        combined_feat = torch.cat([graph_feat, y_part], dim=-1)
        return self.predictor(combined_feat)

    def reset_parameters(self):
        self.predictor.reset_parameters()
        if hasattr(self.backend, 'reset_parameters'):
            self.backend.reset_parameters()

class ConditionalGnn(nn.Module):
    def __init__(self, emb_dim, backend_dim, backend, num_domain, num_class):
        super(ConditionalGnn, self).__init__()
        self.emb_dim = emb_dim
        self.class_emb = nn.Parameter(torch.zeros(num_domain, emb_dim))
        self.backend = backend  
        self.predictor = nn.Linear(backend_dim + emb_dim, num_class)

    def forward(self, data, domains):
        # print(domains)
        # 根据域标签从 class_emb 中选择域相关的嵌入
        domain_feat = torch.index_select(self.class_emb, 0, domains)
        # print(domain_feat)
        # 通过 backend (GNN) 处理图数据
        graph_feat = self.backend(data)
        # print(f"graph_feat shape: {graph_feat.shape}")
        # 将图特征与域特征拼接
        combined_feat = torch.cat([graph_feat, domain_feat], dim=1)
        # print(f"combined_feat shape: {combined_feat.shape}")
        result = self.predictor(combined_feat)
        return result

    def reset_parameters(self):
        self.predictor.reset_parameters()
        # 也可以添加 backend 的参数重置，如果它有 reset_parameters 方法
        if hasattr(self.backend, 'reset_parameters'):
            self.backend.reset_parameters()

class GraphFeatureExtractor(nn.Module):
    def __init__(self, num_features, hidden_dim, num_layers, jk_mode=None):
        super(GraphFeatureExtractor, self).__init__()
        self.conv1 = GINConv(Sequential(
            Linear(num_features, hidden_dim),
            ReLU(),
            Linear(hidden_dim, hidden_dim),
            ReLU(),
            BN(hidden_dim)
        ))

        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers - 1):
            self.convs.append(GINConv(Sequential(
                Linear(hidden_dim, hidden_dim),
                ReLU(),
                Linear(hidden_dim, hidden_dim),
                ReLU(),
                BN(hidden_dim)
            )))

        # 如果需要跳跃连接，可以使用 Jumping Knowledge (JK)
        if jk_mode is not None:
            self.jk = JumpingKnowledge(mode=jk_mode, channels=hidden_dim, num_layers=num_layers)
        else:
            self.jk = None

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)

        xs = [x]
        for conv in self.convs:
            x = conv(x, edge_index)
            xs.append(x)

        if self.jk is not None:
            x = self.jk(xs)
        else:
            x = xs[-1]

        return global_mean_pool(x, data.batch)

    def reset_parameters(self):
        self.conv1.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
        if hasattr(self.jk, 'reset_parameters'):
            self.jk.reset_parameters()
class Predictor(nn.Module):
    def __init__(self, hidden_dim, num_classes, dropout_rate=0.3):
        super(Predictor, self).__init__()
        self.lin1 = nn.Linear(hidden_dim, hidden_dim)
        self.lin2 = nn.Linear(hidden_dim, num_classes)
        self.dropout_rate = dropout_rate

    def forward(self, x, domain_feat=None):
        if domain_feat is not None:
            x = torch.cat([x, domain_feat], dim=1)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=self.dropout_rate, training=self.training)
        return self.lin2(x)

    def reset_parameters(self):
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()