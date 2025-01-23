# Standard library imports
import os
import math
from math import ceil

# PyTorch imports
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import timm

# PyTorch Geometric imports
import torch_geometric
from torch_geometric.data import Data, Batch, InMemoryDataset, DataLoader
from torch_geometric.nn import (
    MessagePassing,
    GCNConv,
    DynamicEdgeConv,
    global_mean_pool,
    global_add_pool,
    knn_graph
)
from torch_geometric.utils import (
    grid,
    to_dense_batch,
    to_dense_adj,
    remove_self_loops,
    to_undirected,
    subgraph,
    k_hop_subgraph,
    to_networkx
)

# Scientific computing and visualization
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pylab as pl
from sklearn.manifold import MDS
from sklearn.cluster import KMeans

# Graph related imports
import networkx as nx
from networkx.generators.community import stochastic_block_model as sbm

# Dataset imports
import medmnist
from medmnist.dataset import PathMNIST

def custom_topk(x, ratio, batch):
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

def custom_filter_adj(edge_index, edge_attr, perm, num_nodes=None):
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

# --------------------- Node Encoder with Subgraph Message Passing --------------------- #

from torch_geometric.nn import MessagePassing
from torch.nn import Linear as Lin


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
    def __init__(self, in_channels, hidden_channels, num_layers=3):
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
        print("passed x shape:", x.shape)
        if subgraph_mask is not None:
            edge_index = edge_index[:, subgraph_mask]
        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.relu(x)
        return x

# --------------------- Modified MKGC Class --------------------- #

class MKGC(nn.Module):
    def __init__(self, kernels, in_channel, out_channel, dictionary_module):
        super(MKGC, self).__init__()
        self.kernels = kernels
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.g_list = nn.ModuleList()
        for _ in range(self.kernels):
            # self.g_list.append(GCNConv(in_channel, out_channel))
            self.g_list.append(SubGraphConv(in_channel, out_channel))
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

# --------------------- GLAPool and MAB Classes --------------------- #
class GLAPool(nn.Module):
    def __init__(self, in_channels, alpha, ratio=0, non_linearity=torch.tanh):
        super(GLAPool, self).__init__()
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
            perm = custom_topk(score, self.ratio, batch)
            x = x[perm] * self.non_linearity(score[perm]).view(-1, 1)
            batch = batch[perm]
            edge_index, edge_attr = custom_filter_adj(
                edge_index, edge_attr, perm, num_nodes=score.size(0))
            return x, edge_index, batch


class MAB(nn.Module):
    def __init__(self, dim_Q, dim_K, dim_V, num_heads):
        super(MAB, self).__init__()
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

class Pool_Att(nn.Module):
    def __init__(self, nhid, alpha, ratio, num_heads):
        super(Pool_Att, self).__init__()
        self.ratio = ratio
        self.pool = GLAPool(nhid, alpha, self.ratio)
        self.att = MAB(nhid, nhid, nhid, num_heads)
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

class Classifier(nn.Module):
    def __init__(self, nhid, dropout_ratio, num_classes):
        super(Classifier, self).__init__()
        self.dropout_ratio = dropout_ratio
        self.lin1 = nn.Linear(nhid, nhid)
        self.lin2 = nn.Linear(nhid, num_classes)

    def reset_parameters(self):
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, x):
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=self.dropout_ratio, training=self.training)
        x = F.log_softmax(self.lin2(x), dim=-1)
        return x

# --------------------- Updated Net Class --------------------- #



def load_data(W, F):
    data_list = []
    for i in range(W.size(0)):
        adj = W[i]
        node_feats = F[i]

        edge_index = adj.nonzero(as_tuple=True)
        edge_index = torch.stack(edge_index, dim=0)  # Ensuring it's [2, num_edges]
        edge_weight = adj[edge_index[0], edge_index[1]]

        if isinstance(node_feats, torch.Tensor):
            node_feats = node_feats.clone().detach()
        else:
            node_feats = torch.tensor(node_feats, dtype=torch.float)

        data = pyg_data.Data(x=node_feats, edge_index=edge_index, edge_attr=edge_weight)
        data_list.append(data)

    return data_list


def create_adj(Fet, k, alpha=1, normalize=True):
    # FINAL WORKING
    device = torch.device('cuda')
    # print("shape of input features:", Fet.shape,type(Fet))


    # Fet = torch.from_numpy(Fet)
    Fet = Fet.to(device).float()
    batch_adj_matrices = []

    # Loop through each graph in the batch
    for i in range(Fet.shape[0]):
        # Extract the feature matrix for the current graph
        F_current = Fet[i]

        if isinstance(F_current, np.ndarray):
            F_current = torch.from_numpy(F_current)


        device = Fet.device  # Use the device of F
        F_current = F_current.to(device)


        # Use PyTorch Geometric to create a k-NN graph
        edge_index = knn_graph(F_current, k=k, loop=False, flow='source_to_target')

        # Create an adjacency matrix from the edge index
        num_nodes = F_current.size(0)
        W = torch.zeros((num_nodes, num_nodes), device=device)
        row, col = edge_index
        W[row, col] = 1  # This creates an unweighted graph for now

        if normalize:
            # Normalize the adjacency matrix
            row_sum = W.sum(dim=1, keepdim=True)
            W = W / row_sum.clamp(min=1)  # Avoid division by zero

        # Append the adjacency matrix to the list
        batch_adj_matrices.append(W)

    # Optionally, stack all adjacency matrices into a single tensor
    batch_adj_matrices = torch.stack(batch_adj_matrices, dim=0)

    return batch_adj_matrices




# class Net(nn.Module):
#     def __init__(self, args):
#         super(Net, self).__init__()
#         self.args = args
#         self.alpha = args.alpha
#         self.kernels = args.kernels
#         self.num_features = args.num_features
#         self.nhid = args.nhid
#         self.num_heads = args.num_heads
#         self.mean_num_nodes = args.mean_num_nodes
#         self.num_classes = args.num_classes
#         self.pooling_ratio = args.pooling_ratio
#         self.dropout_ratio = args.dropout_ratio
#         self.num_atoms = args.num_atoms # Number of atoms in the dictionary
#         self.ratio = ceil(int(self.mean_num_nodes * self.pooling_ratio))
#         self.device = torch.device('cuda')
#         # Instantiate the dictionary module
#         self.num_atoms = args.num_atoms  # Number of atoms in the dictionary
#         # self.dictionary_module = DictionaryModule(self.num_atoms, self.num_features)
#         self.dictionary_module = DictionaryModule(self.num_atoms, self.nhid)

#         self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#         self.pretrained_weights = './dino_deitsmall8_pretrain_full_checkpoint.pth'
#         self.layer = 11
#         self.facet = 'key'
#         self.stride = 8
#         self.log_bin = False


#         self.extractor = ViTExtractor('dino_vitbs8', self.stride, model_dir=self.pretrained_weights, device=self.device)
#         # Replace node_encoder with NodeEncoder
#         self.node_encoder = NodeEncoder(self.num_features, self.nhid, num_layers=3)
#         self.gconv1 = MKGC(self.kernels, self.nhid, self.nhid, self.dictionary_module)
#         self.gconv2 = MKGC(self.kernels, self.nhid, self.nhid, self.dictionary_module)
#         self.gconv3 = MKGC(self.kernels, self.nhid, self.nhid, self.dictionary_module)
#         self.weight = GLAPool(self.nhid, self.alpha)
#         self.pool_att = Pool_Att(self.nhid, self.alpha, self.ratio, self.num_heads)
#         self.classifier = Classifier(self.nhid, self.dropout_ratio, self.num_classes)

#     def reset_parameters(self):
#         self.node_encoder.reset_parameters()
#         self.gconv1.reset_parameters()
#         self.gconv2.reset_parameters()
#         self.gconv3.reset_parameters()
#         self.weight.reset_parameters()
#         self.pool_att.reset_parameters()
#         self.classifier.reset_parameters()
#         self.dictionary_module.dictionary.data = torch.randn(self.num_atoms, self.nhid)


#     def forward(self, x):

#         bs, c, H, W = x.shape

#         if c == 1:
#             x = x.repeat(1, 3, 1, 1)

#         x = x.to(self.device)
#         Fet = deep_features(x, self.extractor, self.layer, self.facet, bin=self.log_bin, device=self.device)
#         Fet = Fet.reshape(bs, -1, 384)
#         # print("shape of Fet:", Fet.shape)
#         W = create_adj(Fet, k=4, alpha=1).to(self.device)
#         data_list = load_data(W, Fet)
#         data = Batch.from_data_list(data_list).to(self.device)


#         x, edge_index, batch = data.x, data.edge_index, data.batch

#         # Encode node features
#         x = self.node_encoder(x, edge_index, None)
#         x = F.dropout(x, p=self.dropout_ratio, training=self.training)

#         # Compute dictionary coefficients using encoded features
#         coefficients = self.dictionary_module(x)  # x has shape (N, nhid)

#         # Assign nodes to subgraphs based on the highest coefficient
#         subgraph_assignments = torch.argmax(coefficients, dim=1)  # [N], values from 0 to num_atoms - 1

#         # Create a mask for edges within the same subgraph
#         src, dst = edge_index
#         src_subgraph = subgraph_assignments[src]
#         dst_subgraph = subgraph_assignments[dst]
#         subgraph_mask = (src_subgraph == dst_subgraph)

#         # Filter edge_index to include only intra-subgraph edges
#         edge_index_subgraph = edge_index[:, subgraph_mask]

#         # Proceed with the rest of the network using the subgraph edge_index
#         x1 = self.gconv1(x, edge_index_subgraph)
#         x1 = F.dropout(x1, p=self.dropout_ratio, training=self.training)
#         x2 = self.gconv2(x1, edge_index_subgraph)
#         x2 = F.dropout(x2, p=self.dropout_ratio, training=self.training)
#         x3 = self.gconv3(x2, edge_index_subgraph)

#         # Compute layer-wise weights using GLAPool
#         weight = torch.cat((self.weight(x1, edge_index_subgraph, None, batch, 1),
#                             self.weight(x2, edge_index_subgraph, None, batch, 1),
#                             self.weight(x3, edge_index_subgraph, None, batch, 1)), dim=-1)
#         weight = F.softmax(weight, dim=-1)
#         x = weight[:, 0].unsqueeze(-1) * x1 + weight[:, 1].unsqueeze(-1) * x2 + weight[:, 2].unsqueeze(-1) * x3

#         # Proceed with pooling and classification
#         x = self.pool_att(x, edge_index_subgraph, batch)
#         graph_feature = x
#         x = self.classifier(x)

#         return x, graph_feature


import torch
import torch.nn as nn
import torch.nn.functional as F
from math import ceil
from torch_geometric.data import Batch

# Assuming DictionaryModule, NodeEncoder, MKGC, GLAPool, Pool_Att, Classifier, deep_features, create_adj, load_data are defined elsewhere

import torch
import torch.nn as nn
import torch.nn.functional as F
from math import ceil
from torch_geometric.data import Batch





import torch
import torch.nn as nn
import timm

class ViTExtractor(nn.Module):
    def __init__(self, model_name='vit_base_patch16_224', pretrained=True, nhid=384):
        super(ViTExtractor, self).__init__()
        # Initialize the ViT model without the classification head
        self.vit = timm.create_model(model_name, pretrained=pretrained, num_classes=0)
        self.vit.eval()  # Set to evaluation mode
        for param in self.vit.parameters():
            param.requires_grad = False  # Freeze ViT parameters
        
        self.hidden_dim = self.vit.embed_dim  # Typically 768 for 'vit_base_patch16_224'
        
        # Add a projection layer if hidden_dim != nhid
        if self.hidden_dim != nhid:
            self.proj = nn.Linear(self.hidden_dim, nhid)
        else:
            self.proj = nn.Identity()
    
    def forward(self, x):
        with torch.no_grad():
            tokens = self.vit.forward_features(x)  # Shape: [batch_size, num_tokens, hidden_dim]
            tokens = self.proj(tokens)            # Shape: [batch_size, num_tokens, nhid]
        return tokens




class Net(nn.Module):
    def __init__(self, args):
        super(Net, self).__init__()
        self.args = args
        self.alpha = args.alpha
        self.kernels = args.kernels
        self.num_features = args.num_features
        self.nhid = args.nhid
        self.num_heads = args.num_heads
        self.mean_num_nodes = args.mean_num_nodes
        self.num_classes = args.num_classes
        self.pooling_ratio = args.pooling_ratio
        self.dropout_ratio = args.dropout_ratio
        self.num_atoms = args.num_atoms  # Number of atoms in the dictionary
        self.ratio = ceil(int(self.mean_num_nodes * self.pooling_ratio))
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Instantiate the dictionary module
        self.dictionary_module = DictionaryModule(self.num_atoms, self.nhid)
        
        # Initialize the ViT extractor with desired hidden dimension
        self.extractor = ViTExtractor(model_name='vit_base_patch16_224', pretrained=True, nhid=self.num_features).to(self.device)
        
        # Replace node_encoder with NodeEncoder
        # self.node_encoder = NodeEncoder(self.num_features, self.nhid, num_layers=3)
        self.bigcn = NodeEncoder(self.num_features, self.nhid, num_layers=4)
        self.node_encoder = torch.nn.Linear(self.num_features, self.nhid)
        
        # GCN layers
        self.gconv1 = MKGC(self.kernels, self.nhid, self.nhid, self.dictionary_module)
        self.gconv2 = MKGC(self.kernels, self.nhid, self.nhid, self.dictionary_module)
        self.gconv3 = MKGC(self.kernels, self.nhid, self.nhid, self.dictionary_module)
        
        # Pooling and classification
        self.weight = GLAPool(self.nhid, self.alpha)
        self.pool_att = Pool_Att(self.nhid, self.alpha, self.ratio, self.num_heads)
        self.classifier = Classifier(self.nhid, self.dropout_ratio, self.num_classes)
    
    def reset_parameters(self):
        self.node_encoder.reset_parameters()
        self.gconv1.reset_parameters()
        self.gconv2.reset_parameters()
        self.gconv3.reset_parameters()
        self.weight.reset_parameters()
        self.pool_att.reset_parameters()
        self.classifier.reset_parameters()
        self.dictionary_module.dictionary.data = torch.randn(self.num_atoms, self.nhid)
    
    def forward(self, x):
        bs, c, H, W = x.shape
        
        # If input is grayscale, replicate channels to make it 3-channel
        if c == 1:
            x = x.repeat(1, 3, 1, 1)
        
        x = x.to(self.device)
        
        # Extract features using ViT
        Fet = self.extractor(x)  # Shape: [batch_size, num_tokens, nhid]
        # print(f"Shape of Fet: {Fet.shape}")  # Debug: should be [batch_size, num_tokens, nhid]
        
        # Create adjacency matrix based on features
        W = create_adj(Fet, k=4, alpha=1).to(self.device)
        
        # Create graph data
        data_list = load_data(W, Fet)  # Custom function to create graph data
        data = Batch.from_data_list(data_list).to(self.device)
        
        # Extract node features, edge indices, and batch
        x, edge_index, batch = data.x, data.edge_index, data.batch
        self.mean_num_nodes = x.size(0) // bs
        self.ratio = ceil(int(self.mean_num_nodes * self.pooling_ratio))
        # Encode node features
        # x = self.node_encoder(x, edge_index, None) # this was bi directional, replaced with simple one
        x = self.node_encoder(x)
        
        x = F.dropout(x, p=self.dropout_ratio, training=self.training)
        
        # Compute dictionary coefficients using encoded features
        coefficients = self.dictionary_module(x)  # Shape: [num_nodes, nhid]
        
        # Assign nodes to subgraphs based on the highest coefficient
        subgraph_assignments = torch.argmax(coefficients, dim=1)  # Shape: [num_nodes]
        
        # Create a mask for edges within the same subgraph
        src, dst = edge_index
        src_subgraph = subgraph_assignments[src]
        dst_subgraph = subgraph_assignments[dst]
        subgraph_mask = (src_subgraph == dst_subgraph)
        
        # Filter edge_index to include only intra-subgraph edges
        edge_index_subgraph = edge_index[:, subgraph_mask]
        
        # GCN layers
        x1 = self.gconv1(x, edge_index_subgraph)
        # x1 = self.bigcn(x, edge_index_subgraph)
        x1 = F.dropout(x1, p=self.dropout_ratio, training=self.training)
        x2 = self.gconv2(x1, edge_index_subgraph)
        # x2 = self.bigcn(x1, edge_index_subgraph)
        x2 = F.dropout(x2, p=self.dropout_ratio, training=self.training)
        x3 = self.gconv3(x2, edge_index_subgraph)
        # x3 = self.bigcn(x2, edge_index_subgraph)
        
        # Compute layer-wise weights using GLAPool
        weight = torch.cat((
            self.weight(x1, edge_index_subgraph, None, batch, 1),
            self.weight(x2, edge_index_subgraph, None, batch, 1),
            self.weight(x3, edge_index_subgraph, None, batch, 1)
        ), dim=-1)
        weight = F.softmax(weight, dim=-1)
        x = weight[:, 0].unsqueeze(-1) * x1 + weight[:, 1].unsqueeze(-1) * x2 + weight[:, 2].unsqueeze(-1) * x3
        
        # Pooling and classification
        x = self.pool_att(x, edge_index_subgraph, batch)
        graph_feature = x
        x = self.classifier(x)
        
        return x, graph_feature
