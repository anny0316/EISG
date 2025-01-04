import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
from torch.nn import Linear

class GCN(torch.nn.Module):
    def __init__(self, num_features, num_classes, hidden_dim, num_layers):
        super(GCN, self).__init__()
        self.num_layers = num_layers

        # 创建 GCN 卷积层列表
        self.convs = torch.nn.ModuleList()
        # 第一个 GCN 卷积层，输入特征维度是 num_features
        self.convs.append(GCNConv(num_features, hidden_dim))

        # 剩下的 GCN 卷积层，输入特征维度是 hidden_dim
        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))

        # 两个全连接层
        self.lin1 = Linear(hidden_dim, hidden_dim)
        self.lin2 = Linear(hidden_dim, num_classes)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # 依次通过每个 GCN 卷积层
        for conv in self.convs:
            x = F.relu(conv(x, edge_index))
            x = F.dropout(x, p=0.5, training=self.training)

        # 全局均值池化，将每个图的节点特征汇总为图级别特征
        x = global_mean_pool(x, batch)

        x = F.relu(self.lin1(x))
        x = self.lin2(x)

        return F.log_softmax(x, dim=-1),x,x