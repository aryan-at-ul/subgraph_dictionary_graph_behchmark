import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.utils import to_dense_batch, remove_self_loops
# from torch_geometric.nn.pool.topk_pool import topk, filter_adj
from math import ceil
import math
from torch_geometric.nn import MessagePassing
from torch.nn import Linear as Lin
from torch_geometric.data import Data

def topk(x, ratio, batch):
    num_nodes = x.size(0)
    batch_size = int(batch.max()) + 1
    perm = []
    for i in range(batch_size):
        mask = (batch == i)
        x_i = x[mask]
        num_nodes_i = x_i.size(0)
        k = max(int(ratio * num_nodes_i), 1)
        if k >= num_nodes_i:
            perm_i = torch.nonzero(mask).view(-1)
        else:
            x_i_score = x_i.view(-1)
            _, idx = torch.topk(x_i_score, k, largest=True)
            perm_i = torch.nonzero(mask).view(-1)[idx]
        perm.append(perm_i)
    perm = torch.cat(perm, dim=0)
    return perm
from torch_geometric.utils import subgraph

def filter_adj(edge_index, edge_attr, perm, num_nodes=None):
    # Subgraph function will return the filtered edge_index and edge_attr
    edge_index, edge_attr = subgraph(
        subset=perm,
        edge_index=edge_index,
        edge_attr=edge_attr,
        num_nodes=num_nodes,
        relabel_nodes=True
    )
    return edge_index, edge_attr









# --------------------- Dictionary Module --------------------- #

class DictionaryModule(nn.Module):
    def __init__(self, num_atoms, atom_size):
        super(DictionaryModule, self).__init__()
        self.num_atoms = num_atoms
        self.atom_size = atom_size
        # Initialize the dictionary atoms as learnable parameters
        self.dictionary = nn.Parameter(torch.randn(num_atoms, atom_size))

    def forward(self, x):
        # x: Node features [N, atom_size]
        # Compute similarity between node features and dictionary atoms
        similarity = torch.matmul(x, self.dictionary.t())  # [N, num_atoms]
        coefficients = F.softmax(similarity, dim=-1)       # [N, num_atoms]
        return coefficients

    def orthogonality_loss(self):
        # Compute D * D^T - I
        D = self.dictionary  # [num_atoms, atom_size]
        DT_D = torch.matmul(D, D.t())  # [num_atoms, num_atoms]
        I = torch.eye(self.num_atoms, device=D.device)
        # Orthogonality loss
        ortho_loss = torch.norm(DT_D - I, p='fro') ** 2
        return ortho_loss


# --------------------- Node Encoder --------------------- #

class SubGraphConv(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super(SubGraphConv, self).__init__(aggr='mean')
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.lin1 = Lin(in_channels, out_channels, bias=False)
        self.lin2 = Lin(in_channels, out_channels, bias=False)
        self.root = Lin(in_channels, out_channels)
        self.reset_parameters()

    def reset_parameters(self):
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()
        self.root.reset_parameters()

    def forward(self, x, edge_index):
        # 2 way message passing
        self.flow = 'source_to_target'
        out1 = self.propagate(edge_index, x=self.lin1(x))
        self.flow = 'target_to_source'
        out2 = self.propagate(edge_index, x=self.lin2(x))
        return self.root(x) + out1 + out2

    def message(self, x_j):
        return x_j

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels,
                                   self.out_channels)

class NodeEncoder(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_layers=1):
        super(NodeEncoder, self).__init__()
        self.convs = nn.ModuleList()
        self.convs.append(SubGraphConv(in_channels, hidden_channels))
        for _ in range(num_layers - 1):
            self.convs.append(SubGraphConv(hidden_channels, hidden_channels))

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, x, edge_index, subgraph_mask=None):
        # If subgraph_mask is provided, filter edge_index
        if subgraph_mask is not None:
            edge_index = edge_index[:, subgraph_mask]
        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.relu(x)
        return x




# --------------------- Graph Convolution --------------------- #

class Graph_convolution(nn.Module):
    def __init__(self, kernels, in_channel, out_channel, dictionary_module):
        super(Graph_convolution, self).__init__()
        self.kernels = kernels
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.g_list = nn.ModuleList()
        for _ in range(self.kernels):
            self.g_list.append(GCNConv(in_channel, out_channel))
        self.dictionary_module = dictionary_module  # Add dictionary module

    def reset_parameters(self):
        for gconv in self.g_list:
            gconv.reset_parameters()

    def forward(self, x, edge_index):
        # Compute dictionary coefficients
        coefficients = self.dictionary_module(x)  # [N, num_atoms]

        total_x = None
        for idx, gconv in enumerate(self.g_list):
            feature = gconv(x, edge_index)
            feature = F.relu(feature)
            # Weight the features using dictionary coefficients
            atom_coefficients = coefficients[:, idx % self.dictionary_module.num_atoms].unsqueeze(-1)
            weighted_feature = feature * atom_coefficients
            if total_x is None:
                total_x = weighted_feature
            else:
                total_x += weighted_feature
        return total_x

# --------------------- Pooling and Attention --------------------- #

class Topk_pool(nn.Module):
    def __init__(self, in_channels, alpha, ratio=0, non_linearity=torch.tanh):
        super(Topk_pool, self).__init__()
        self.in_channels = in_channels
        self.alpha = alpha
        self.ratio = ratio
        self.non_linearity = non_linearity
        self.score1 = nn.Linear(self.in_channels, 1)
        self.score2 = GCNConv(in_channels=self.in_channels, out_channels=1, add_self_loops=False)

    def reset_parameters(self):
        self.score1.reset_parameters()
        self.score2.reset_parameters()

    def forward(self, x, edge_index, edge_attr=None, batch=None, flag=0):
        if batch is None:
            batch = edge_index.new_zeros(x.size(0))

        x = x.unsqueeze(-1) if x.dim() == 1 else x

        edge_index1, _ = remove_self_loops(edge_index=edge_index, edge_attr=edge_attr)
        score = (self.alpha * self.score1(x) + (1 - self.alpha) * self.score2(x, edge_index1)).squeeze()

        if flag == 1:
            return score.view(-1, 1)
        else:
            perm = topk(score, self.ratio, batch)
            x = x[perm] * self.non_linearity(score[perm]).view(-1, 1)
            batch = batch[perm]
            edge_index, edge_attr = filter_adj(
                edge_index, edge_attr, perm, num_nodes=score.size(0))

            return x, edge_index, batch

class Attention_block(nn.Module):
    def __init__(self, dim_Q, dim_K, dim_V, num_heads):
        super(Attention_block, self).__init__()
        self.dim_V = dim_V
        self.num_heads = num_heads
        self.softmax_dim = 2

        self.fc_q = nn.Linear(dim_Q, dim_V)
        self.fc_k = GCNConv(dim_K, dim_V)
        self.fc_v = GCNConv(dim_K, dim_V)
        self.ln0 = nn.LayerNorm(dim_V)
        self.ln1 = nn.LayerNorm(dim_V)
        self.fc_o = nn.Linear(dim_V, dim_V)

    def reset_parameters(self):
        self.fc_q.reset_parameters()
        self.fc_k.reset_parameters()
        self.fc_v.reset_parameters()
        self.ln0.reset_parameters()
        self.ln1.reset_parameters()
        self.fc_o.reset_parameters()

    def forward(self, Q, graph=None):
        Q = self.fc_q(Q)

        (x, edge_index, batch) = graph
        K, V = self.fc_k(x, edge_index), self.fc_v(x, edge_index)
        K, mask = to_dense_batch(K, batch)
        V, _ = to_dense_batch(V, batch)
        attention_mask = mask.unsqueeze(1)
        attention_mask = attention_mask.to(dtype=next(self.parameters()).dtype)
        attention_mask = (1.0 - attention_mask) * -1e9

        dim_split = self.dim_V // self.num_heads
        Q_ = torch.cat(Q.split(dim_split, dim=2), 0)
        K_ = torch.cat(K.split(dim_split, dim=2), 0)
        V_ = torch.cat(V.split(dim_split, dim=2), 0)

        attention_mask = torch.cat([attention_mask for _ in range(self.num_heads)], 0)
        attention_score = Q_.bmm(K_.transpose(1, 2)) / math.sqrt(self.dim_V)
        A = torch.softmax(attention_mask + attention_score, self.softmax_dim)

        O = torch.cat((Q_ + A.bmm(V_)).split(Q.size(0), 0), 2)
        O = self.ln0(O)
        O = O + F.relu(self.fc_o(O))
        O = self.ln1(O)

        return O

# --------------------- Pool_Att and Classifier Classes --------------------- #

class GraphPooling(nn.Module):
    def __init__(self, nhid, alpha, ratio, num_heads):
        super(GraphPooling, self).__init__()
        self.ratio = ratio
        self.pool = Topk_pool(nhid, alpha, self.ratio)
        self.att = Attention_block(nhid, nhid, nhid, num_heads)
        self.readout = nn.Conv1d(self.ratio, 1, 1)

    def reset_parameters(self):
        self.pool.reset_parameters()
        self.att.reset_parameters()
        self.readout.reset_parameters()

    def forward(self, x, edge_index, batch):
        graph = (x, edge_index, batch)
        xp, _, batchp = self.pool(x=x, edge_index=edge_index, batch=batch)  # Select top-k nodes
        xp, _ = to_dense_batch(x=xp, batch=batchp, max_num_nodes=self.ratio, fill_value=0)
        xp = self.att(xp, graph)
        xp = self.readout(xp).squeeze()
        return xp




