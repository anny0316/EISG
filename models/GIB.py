import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn import Linear, Sequential, ReLU, BatchNorm1d as BN
from torch_geometric.nn import GINConv, global_mean_pool, global_max_pool, JumpingKnowledge
from torch_geometric.utils import to_dense_adj, to_dense_batch, add_self_loops
# import torch_scatter as tscatter

class GIB(torch.nn.Module):
    def __init__(self, num_features, num_classes, hidden_dim, num_layers, dropout_rate=0.3):
        super(GIB, self).__init__()

        self.conv1 = GINConv(Sequential(
            Linear(num_features, hidden_dim),
            ReLU(),
            Linear(hidden_dim, hidden_dim),
            ReLU(),
            BN(hidden_dim)
        ))

        self.convs = torch.nn.ModuleList()

        for i in range(num_layers - 1):
            self.convs.append(GINConv(Sequential(
                Linear(hidden_dim, hidden_dim),
                ReLU(),
                Linear(hidden_dim, hidden_dim),
                ReLU(),
                BN(hidden_dim),
            )))

        self.lin1 = Linear(hidden_dim, hidden_dim)
        self.lin2 = Linear(hidden_dim, num_classes)
        self.dropout_rate = dropout_rate  # 控制Dropout的比例

    def reset_parameters(self):
        self.conv1.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, data, mask=None):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        x = self.conv1(x, edge_index)
        for conv in self.convs:
            x = conv(x, edge_index)
        if mask is not None:
            embeds = mask * x
        else:
            embeds = x
        x = global_mean_pool(embeds, batch)
        x = self.lin1(x)
        x = F.dropout(x, p=self.dropout_rate)  # 使用传入的dropout_rate
        x = self.lin2(x)
        out = torch.clamp(x, -10, 10)

        return F.log_softmax(out, dim=-1), embeds, out

    def __repr__(self):
        return self.__class__.__name__



class JointGenerator(torch.nn.Module):
    def __init__(self, hidden_size, device):
        super(JointGenerator, self).__init__()


        self.input_size = hidden_size
        self.device = device
        self.hidden_size = hidden_size
        self.lin1 = torch.nn.Linear(self.input_size,self.hidden_size)
        self.lin2 = torch.nn.Linear(self.hidden_size, 1)
        self.relu = torch.nn.ReLU()
        self.rate = 0.7
        self.epsilon = 0.00000001
        self.sigmoid = nn.Sigmoid()
        # self.discriminator = Discriminator(input_size=self.input_size, hidden_size=4 * self.hidden_size)

    def reset_parameters(self):
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()
        self.discriminator.reset_parameters()  # 重置 Discriminator 参数

    def _kld(self, mask, prior_prob=0.5):
        pos = mask
        neg = 1 - mask
        # 使用传入的先验概率计算 KL 散度
        kld_loss = torch.mean(pos * torch.log((pos + self.epsilon) / (prior_prob + self.epsilon)) +
                              neg * torch.log((neg + self.epsilon) / (1 - prior_prob + self.epsilon)))

        return kld_loss


    def _sample_graph(self, sampling_weights, temperature=1.0, bias=0.5):
        bias = bias + 0.0001  # If bias is 0, we run into problems
        eps = (bias - (1 - bias)) * torch.rand(sampling_weights.size()) + (1 - bias)
        # print(eps)
        gate_inputs = torch.log(eps) - torch.log(1 - eps)
        # print(gate_inputs)
        gate_inputs = gate_inputs.to(self.device)
        # 计算采样权重的对数几率
        logit_sampling_weights = torch.log(sampling_weights) - torch.log(1 - sampling_weights)

        gate_inputs = (gate_inputs + logit_sampling_weights) / temperature
        graph = torch.sigmoid(gate_inputs)

        return graph

    def edge_sample(self, node_mask, edge_idx):
        src_val = node_mask[edge_idx[0]]
        dst_val = node_mask[edge_idx[1]]
        edge_val = 0.5 * (src_val + dst_val)

        return edge_val


    def forward(self, x_embeds, edges, batch):

        input_embs = x_embeds

        pre = self.relu(self.lin1(input_embs))

        pre = self.lin2(pre)
        pre = torch.clamp(pre, min=-10, max=10)
        sampling_weights = self.sigmoid(pre)

        node_mask = self._sample_graph(sampling_weights)  # 进行采样

        # node_mask = self._sample_graph(pre)
        kld_loss = self._kld(node_mask)
        edge_mask = self.edge_sample(node_mask, edges)
        # print(f"batch shape: {batch.shape}")
        # print(f"x_embeds shape: {x_embeds.shape}")
        # print(f"pre shape after lin1 and relu: {pre.shape}")
        # print(f"sampling_weights shape: {sampling_weights.shape}")
        # print(f"node_mask shape: {node_mask.shape}")
        # print(f"edge_mask shape: {edge_mask.shape}")
        # mi_loss = self.mi_loss(self.discriminator,x_embeds, node_mask, batch)

        return kld_loss, node_mask, edge_mask

class Discriminator(torch.nn.Module):

    def __init__(self, input_size, hidden_size):
        super(Discriminator, self).__init__()

        # self.args = args
        self.input_size = input_size*2
        self.hidden_size = hidden_size
        self.fc1 = torch.nn.Linear(self.input_size, self.hidden_size)
        self.fc2 = torch.nn.Linear(self.hidden_size, 1)
        self.relu = torch.nn.ReLU()


    def mi_loss(self, discriminator,embeddings, node_mask, batch):
        # 取出 mask 值较高的正样本
        positive_mask = node_mask.squeeze() > 0.5

        positive_mask = positive_mask.unsqueeze(-1).float()
        positive = embeddings * positive_mask  #
        # print(f"positive shape: {positive.shape}")

        shuffle_embeddings = embeddings[torch.randperm(embeddings.size(0))]
        # print(f"shuffle_embeddings shape: {shuffle_embeddings.shape}")
        # 计算正样本与整个图嵌入的关联性（joint）
        joint = torch.mean(torch.sum(positive * embeddings, dim=-1))
        margin = torch.mean(torch.sum(positive * shuffle_embeddings, dim=-1))

        # 计算互信息：joint 的均值减去 log(exp(margin)) 的均值
        mi_est = joint - torch.log(1 + torch.exp(margin))

        return mi_est

    def forward(self, embeddings, positive):
        cat_embeddings = torch.cat((embeddings, positive), dim=-1)
        pre = self.relu(self.fc1(cat_embeddings))
        pre = self.fc2(pre)

        return pre

    def reset_parameters(self):
        torch.nn.init.constant_(self.fc1.weight, 0.01)
        torch.nn.init.constant_(self.fc2.weight, 0.01)

