import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
from torch.nn import Sequential, Linear, ReLU

class GAT(torch.nn.Module):
    def __init__(self, num_features, num_classes, hidden_dim, num_layers, num_heads=8):
        super(GAT, self).__init__()
        self.num_layers = num_layers

        self.convs = torch.nn.ModuleList()
        self.convs.append(GATConv(num_features, hidden_dim, heads=num_heads, dropout=0.6))

        for _ in range(num_layers - 1):
            self.convs.append(GATConv(hidden_dim * num_heads, hidden_dim, heads=num_heads, dropout=0.6))

        self.lin1 = Linear(hidden_dim * num_heads, hidden_dim)
        self.lin2 = Linear(hidden_dim, num_classes)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        for conv in self.convs:
            x = F.dropout(x, p=0.6, training=self.training)
            x = F.elu(conv(x, edge_index))

        x = global_mean_pool(x, batch)  # 通过池化层将每个图的节点特征汇总为图级别特征

        x = F.dropout(x, p=0.6, training=self.training)
        x = F.elu(self.lin1(x))
        x = self.lin2(x)

        return F.log_softmax(x, dim=-1),x