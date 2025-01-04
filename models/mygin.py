import torch
import torch.nn as nn
from torch.nn import Linear, Sequential, ReLU, BatchNorm1d as BN
from torch_geometric.nn import GINConv, JumpingKnowledge, GCNConv, GATConv, SAGEConv

from EISG.models.GIB import JointGenerator
from EISG.utils import set_masks, clear_masks


class GraphFeatureExtractor(nn.Module):
    def __init__(self, num_features, hidden_dim, num_layers, jk_mode='last',gnn_type='GIN'):
        super(GraphFeatureExtractor, self).__init__()
        self.gnn_type = gnn_type
        if gnn_type == 'GIN':
            self.conv1 = GINConv(Sequential(
                Linear(num_features, hidden_dim),
                ReLU(),
                Linear(hidden_dim, hidden_dim),
                ReLU(),
                BN(hidden_dim)
            ))
        elif gnn_type == 'GCN':
            self.conv1 = GCNConv(num_features, hidden_dim)
        elif gnn_type == 'GAT':
            self.conv1 = GATConv(num_features, hidden_dim)
        elif gnn_type == 'GraphSAGE':
            self.conv1 = SAGEConv(num_features, hidden_dim)
        else:
            raise ValueError("Unsupported GNN type")

            # Initialize subsequent convolution layers
        self.convs = nn.ModuleList()
        for _ in range(num_layers - 1):
            if gnn_type == 'GIN':
                self.convs.append(GINConv(Sequential(
                    Linear(hidden_dim, hidden_dim),
                    ReLU(),
                    Linear(hidden_dim, hidden_dim),
                    ReLU(),
                    BN(hidden_dim),
                )))
            elif gnn_type == 'GCN':
                self.convs.append(GCNConv(hidden_dim, hidden_dim))
            elif gnn_type == 'GAT':
                self.convs.append(GATConv(hidden_dim, hidden_dim))
            elif gnn_type == 'GraphSAGE':
                self.convs.append(SAGEConv(hidden_dim, hidden_dim))
        #
        # self.conv1 = GINConv(Sequential(
        #     Linear(num_features, hidden_dim),
        #     ReLU(),
        #     Linear(hidden_dim, hidden_dim),
        #     ReLU(),
        #     BN(hidden_dim)
        # ),)
        # self.convs = nn.ModuleList()
        # for i in range(num_layers - 1):
        #     self.convs.append(GINConv(Sequential(
        #         Linear(hidden_dim, hidden_dim),
        #         ReLU(),
        #         Linear(hidden_dim, hidden_dim),
        #         ReLU(),
        #         BN(hidden_dim),
        #     ),))
        self.jk = JumpingKnowledge(mode=jk_mode, channels=hidden_dim, num_layers=num_layers)

    def reset_parameters(self):
        self.conv1.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
        self.jk.reset_parameters()

    def forward(self, data, mask=None):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        xs = [self.conv1(x, edge_index)]
        for conv in self.convs:
            xs.append(conv(xs[-1], edge_index))
        x = self.jk(xs)
        if mask is not None:
            x = mask * x
        return x,batch

import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool

class Predictor(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes,dropout_rate=0.3):
        super(Predictor, self).__init__()
        self.lin1 = Linear(input_dim, hidden_dim)
        self.lin2 = Linear(hidden_dim, num_classes)
        self.dropout_rate = dropout_rate

    def reset_parameters(self):
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, x, batch):
        x = global_mean_pool(x, batch)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=self.dropout_rate, training=self.training)
        x = self.lin2(x)
        x = torch.clamp(x, -10, 10)
        return F.log_softmax(x, dim=-1), x

class MYGIN(nn.Module):
    def __init__(self, num_features, hidden_dim, num_layers, num_classes, jk_mode='max',device=0):
        super(MYGIN, self).__init__()
        self.feature_extractor = GraphFeatureExtractor(num_features, hidden_dim, num_layers, jk_mode)
        self.device = device
        self.subgraph = JointGenerator(hidden_dim,device)
        if jk_mode == 'cat':
            input_dim = num_layers * hidden_dim
        else:
            input_dim = hidden_dim
        self.predictor = Predictor(input_dim, hidden_dim, num_classes)
    def reset_parameters(self):
        self.feature_extractor.reset_parameters()
        self.predictor.reset_parameters()

    def forward(self, data, mask=None, train=True):
        embeddings, batch = self.feature_extractor(data, mask)
        _, pre = self.predictor(embeddings, batch)
        if train:
            kld_loss, node_mask, edge_mask = self.subgraph(embeddings, data.edge_index, data.batch)
            set_masks(edge_mask, self.feature_extractor)
            sub_embedding, batch = self.feature_extractor(data, node_mask)
            clear_masks(self.feature_extractor)
            sub_pre, _ = self.predictor(sub_embedding, batch)
            return sub_pre, kld_loss, pre
        else:
            pre, x = self.predictor(embeddings, batch)
            return pre, x