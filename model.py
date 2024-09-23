import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import to_dense_batch, remove_self_loops
from torch_geometric.nn.pool.topk_pool import topk, filter_adj
from math import ceil
import math
from torch_geometric.nn import MessagePassing
from torch.nn import Linear as Lin
from torch_geometric.data import Data
from layers import DictionaryModule, Graph_convolution, GraphPooling, Topk_pool, NodeEncoder
from torch_geometric.nn import GCNConv, GATConv, GINConv, global_mean_pool

# --------------------- Classifier Head --------------------- #


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

# --------------------- GCNNet Class --------------------- #

class GCNNet(nn.Module):
    def __init__(self, args):
        super(GCNNet, self).__init__()
        self.args = args
        self.num_features = args.num_features
        self.nhid = args.nhid
        self.num_classes = args.num_classes
        self.dropout_ratio = args.dropout_ratio
        self.num_layers = 4  # Default number of layers

        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(self.num_features, self.nhid))
        for _ in range(self.num_layers - 1):
            self.convs.append(GCNConv(self.nhid, self.nhid))

        self.classifier = Classifier(self.nhid, self.dropout_ratio, self.num_classes)

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        self.classifier.reset_parameters()

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout_ratio, training=self.training)

        x = global_mean_pool(x, batch)
        x = self.classifier(x)
        return x

# --------------------- GATNet Class --------------------- #

class GATNet(nn.Module):
    def __init__(self, args):
        super(GATNet, self).__init__()
        self.args = args
        self.num_features = args.num_features
        self.nhid = args.nhid
        self.num_classes = args.num_classes
        self.dropout_ratio = args.dropout_ratio
        self.num_layers = 4  
        self.num_heads = args.num_heads

        self.convs = nn.ModuleList()
        self.convs.append(GATConv(self.num_features, self.nhid // self.num_heads, heads=self.num_heads, dropout=self.dropout_ratio))
        for _ in range(self.num_layers - 1):
            self.convs.append(GATConv(self.nhid, self.nhid // self.num_heads, heads=self.num_heads, dropout=self.dropout_ratio))

        self.classifier = Classifier(self.nhid, self.dropout_ratio, self.num_classes)

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        self.classifier.reset_parameters()

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.elu(x)
            x = F.dropout(x, p=self.dropout_ratio, training=self.training)

        x = global_mean_pool(x, batch)
        x = self.classifier(x)
        return x

# --------------------- GINNet Class --------------------- #

class GINNet(nn.Module):
    def __init__(self, args):
        super(GINNet, self).__init__()
        self.args = args
        self.num_features = args.num_features
        self.nhid = args.nhid
        self.num_classes = args.num_classes
        self.dropout_ratio = args.dropout_ratio
        self.num_layers = 4

        self.convs = nn.ModuleList()
        nn1 = nn.Sequential(nn.Linear(self.num_features, self.nhid), nn.ReLU(), nn.Linear(self.nhid, self.nhid))
        self.convs.append(GINConv(nn1))
        for _ in range(self.num_layers - 1):
            nnk = nn.Sequential(nn.Linear(self.nhid, self.nhid), nn.ReLU(), nn.Linear(self.nhid, self.nhid))
            self.convs.append(GINConv(nnk))

        self.classifier = Classifier(self.nhid, self.dropout_ratio, self.num_classes)

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        self.classifier.reset_parameters()

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout_ratio, training=self.training)

        x = global_mean_pool(x, batch)
        x = self.classifier(x)
        return x



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
        self.num_atoms = args.num_atoms 
        self.ratio = ceil(int(self.mean_num_nodes * self.pooling_ratio))

        self.dictionary_module = DictionaryModule(self.num_atoms, self.nhid)

  
        self.node_encoder = NodeEncoder(self.num_features, self.nhid, num_layers=2)
        self.gconv1 = Graph_convolution(self.kernels, self.nhid, self.nhid, self.dictionary_module)
        self.gconv2 = Graph_convolution(self.kernels, self.nhid, self.nhid, self.dictionary_module)
        self.gconv3 = Graph_convolution(self.kernels, self.nhid, self.nhid, self.dictionary_module)
        self.weight = Topk_pool(self.nhid, self.alpha)
        self.pool_att = GraphPooling(self.nhid, self.alpha, self.ratio, self.num_heads)
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


    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # Encode node features
        # x = self.node_encoder(x, edge_index, None)
        x = self.node_encoder(x, edge_index)
        x = F.dropout(x, p=self.dropout_ratio, training=self.training)

       
        coefficients = self.dictionary_module(x)  # x has shape (N, nhid)

        
        subgraph_assignments = torch.argmax(coefficients, dim=1)  # [N], values from 0 to num_atoms - 1

        
        src, dst = edge_index
        src_subgraph = subgraph_assignments[src]
        dst_subgraph = subgraph_assignments[dst]
        subgraph_mask = (src_subgraph == dst_subgraph)

        
        edge_index_subgraph = edge_index[:, subgraph_mask]

        
        x1 = self.gconv1(x, edge_index_subgraph)
        x1 = F.dropout(x1, p=self.dropout_ratio, training=self.training)
        x2 = self.gconv2(x1, edge_index_subgraph)
        x2 = F.dropout(x2, p=self.dropout_ratio, training=self.training)
        x3 = self.gconv3(x2, edge_index_subgraph)

        
        weight = torch.cat((self.weight(x1, edge_index_subgraph, None, batch, 1),
                            self.weight(x2, edge_index_subgraph, None, batch, 1),
                            self.weight(x3, edge_index_subgraph, None, batch, 1)), dim=-1)
        weight = F.softmax(weight, dim=-1)
        x = weight[:, 0].unsqueeze(-1) * x1 + weight[:, 1].unsqueeze(-1) * x2 + weight[:, 2].unsqueeze(-1) * x3

        
        x = self.pool_att(x, edge_index_subgraph, batch)
        graph_feature = x
        x = self.classifier(x)

        return x, graph_feature



# --------------------- Test --------------------- #

class Args:
    def __init__(self):
        self.alpha = 0.5  
        self.kernels = 3  
        self.num_features = 3 
        self.nhid = 64  
        self.num_heads = 4  
        self.mean_num_nodes = 30  
        self.num_classes = 6 
        self.pooling_ratio = 0.5  
        self.dropout_ratio = 0.5  
        self.num_atoms = 5  




if __name__ == '__main__':
    args = Args()
    model = Net(args)
    print(model)
    args = Args()
    num_nodes = 50
    num_edges = 100
    x = torch.randn(num_nodes, args.num_features) 
    edge_index = torch.randint(0, num_nodes, (2, num_edges))  
    batch = torch.zeros(num_nodes, dtype=torch.long)  
    data = Data(x=x, edge_index=edge_index, batch=batch)
    output, graph_feature = model(data)
    print("Output shape:", output.shape)
    print("Graph feature shape:", graph_feature.shape)
