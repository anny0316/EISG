import torch
import torch.nn.functional as F
from torch_geometric.nn import GINConv, global_mean_pool
from torch.nn import Sequential, Linear, ReLU

class GIN(torch.nn.Module):
    def __init__(self, num_features, num_classes, hidden_dim, num_layers):
        super(GIN, self).__init__()
        self.convs = torch.nn.ModuleList()
        self.convs.append(GINConv(Sequential(Linear(num_features, hidden_dim), ReLU(), Linear(hidden_dim, hidden_dim))))

        for _ in range(num_layers - 1):
            self.convs.append(GINConv(Sequential(Linear(hidden_dim, hidden_dim), ReLU(), Linear(hidden_dim, hidden_dim))))

        self.lin1 = Linear(hidden_dim, hidden_dim)
        self.lin2 = Linear(hidden_dim, num_classes)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        for conv in self.convs:
            x = F.relu(conv(x, edge_index))

        # print(f"Embeddings shape after conv layers: {x.shape}")
        x = global_mean_pool(x, batch)  # Pooling layer to aggregate node features into graph features

        x = F.relu(self.lin1(x))
        x = self.lin2(x)

        return F.log_softmax(x, dim=-1), x, x