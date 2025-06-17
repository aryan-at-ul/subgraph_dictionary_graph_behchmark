import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.data import Data, Batch
from torch_geometric.utils import to_dense_batch, k_hop_subgraph
import numpy as np
import timm
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import os
import argparse
from tqdm import tqdm
import time
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

# ===================== ViT Feature Extractor =====================
class ViTExtractor(nn.Module):
    def __init__(self, model_name='vit_base_patch16_224', pretrained=True, nhid=384):
        super(ViTExtractor, self).__init__()
        # Create a ViT model that outputs raw features (num_classes=0)
        self.vit = timm.create_model(model_name, pretrained=pretrained, num_classes=0)
        self.vit.eval()
        for param in self.vit.parameters():
            param.requires_grad = False
        
        self.hidden_dim = self.vit.embed_dim
        if self.hidden_dim != nhid:
            self.proj = nn.Linear(self.hidden_dim, nhid)
        else:
            self.proj = nn.Identity()
    
    def forward(self, x):
        # Extract patch‐token embeddings (including cls token) without gradient updates
        with torch.no_grad():
            tokens = self.vit.forward_features(x)  # shape: [B, num_tokens, embed_dim]
            tokens = self.proj(tokens)             # project to nhid if needed
        return tokens  # [B, N, nhid]


# ===================== Graph Construction Utilities =====================
def create_adj(Fet, k, alpha=1, normalize=True):
    """
    Given node features Fet of shape [B, N, F], build a k-NN adjacency for each graph in the batch.
    Returns a tensor of shape [B, N, N], where each slice is a row‐normalized adjacency (if normalize=True).
    """
    device = Fet.device
    batch_adj_matrices = []
    
    for i in range(Fet.shape[0]):
        F_current = Fet[i]  # [N, F]
        # Build k-NN graph (undirected by default; loop=False)
        edge_index = torch_geometric.nn.knn_graph(F_current, k=k, loop=False, flow='source_to_target')
        
        num_nodes = F_current.size(0)
        W = torch.zeros((num_nodes, num_nodes), device=device)
        row, col = edge_index
        W[row, col] = 1.0
        
        if normalize:
            row_sum = W.sum(dim=1, keepdim=True)  # degree for each node
            W = W / row_sum.clamp(min=1.0)
        
        batch_adj_matrices.append(W)
    
    return torch.stack(batch_adj_matrices, dim=0)  # [B, N, N]

def load_data(W, F):
    """
    Convert batched dense adjacency W ([B,N,N]) and features F ([B,N,Ff]) into a list of PyG Data objects.
    """
    data_list = []
    for i in range(W.size(0)):
        adj = W[i]                # [N, N]
        node_feats = F[i]         # [N, Ff]
        
        # Extract edge_index and edge_weight from dense adjacency
        edge_index = adj.nonzero(as_tuple=True)
        edge_index = torch.stack(edge_index, dim=0)
        edge_weight = adj[edge_index[0], edge_index[1]]
        
        if isinstance(node_feats, torch.Tensor):
            node_feats = node_feats.clone().detach()
        else:
            node_feats = torch.tensor(node_feats, dtype=torch.float)
        
        data = Data(x=node_feats, edge_index=edge_index, edge_attr=edge_weight)
        data_list.append(data)
    
    return data_list


# ===================== Model 1: GFK (Graph Frequency Kernels) =====================
class Dense(nn.Module):
    def __init__(self, in_features, out_features, bias='none'):
        super(Dense, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias == 'bn':
            self.bias = nn.BatchNorm1d(out_features)
        else:
            # If no BatchNorm, bias is a no-op
            self.bias = lambda x: x
        self.reset_parameters()
    
    def reset_parameters(self):
        stdv = 1.0 / np.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
    
    def forward(self, input):
        # Linear transformation
        output = torch.mm(input, self.weight)  # [N, out_features]
        output = self.bias(output)             # apply BN if requested
        if self.in_features == self.out_features:
            # residual connection if dims match
            output = output + input
        return output

class MLP(nn.Module):
    def __init__(self, nfeat, nlayers, nhidden, nclass, dropout, bias):
        super(MLP, self).__init__()
        self.fcs = nn.ModuleList()
        # First layer: nfeat -> nhidden
        self.fcs.append(Dense(nfeat, nhidden, bias))
        # Hidden layers (nhidden -> nhidden)
        for _ in range(nlayers - 2):
            self.fcs.append(Dense(nhidden, nhidden, bias))
        # Final layer: nhidden -> nclass
        self.fcs.append(Dense(nhidden, nclass))
        self.act_fn = nn.ReLU()
        self.dropout = dropout
    
    def forward(self, x):
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.act_fn(self.fcs[0](x))
        for fc in self.fcs[1:-1]:
            x = F.dropout(x, self.dropout, training=self.training)
            x = self.act_fn(fc(x))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.fcs[-1](x)
        return x

class GraphSpectralFilter(nn.Module):
    def __init__(self, K, in_features, out_features):
        """
        Applies K parallel spectral (GCNConv) filters to the same input features.
        Each GCNConv is independently applied to the original x.
        """
        super().__init__()
        self.K = K
        self.filters = nn.ModuleList([
            GCNConv(in_features, out_features, add_self_loops=True, normalize=True)
            for _ in range(K)
        ])
    
    def forward(self, x, edge_index, edge_weight=None):
        """
        x: [N, in_features]
        edge_index: [2, E]
        edge_weight: [E] or None
        Returns tensor of shape [N, K, out_features]
        """
        out = []
        for conv in self.filters:
            h_k = conv(x, edge_index, edge_weight)  # apply each filter to original x
            out.append(h_k)
        # Stack along a new dimension K
        return torch.stack(out, dim=1)  # [N, K, out_features]


class GFKNet(nn.Module):
    def __init__(self, args):
        super(GFKNet, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.num_features = args.num_features
        self.nhid = args.nhid
        self.num_classes = args.num_classes
        self.dropout = args.dropout_ratio
        self.K = args.gfk_K  # Number of spectral filters
        
        # ViT feature extractor (frozen)
        self.extractor = ViTExtractor(
            model_name='vit_base_patch16_224',
            pretrained=True,
            nhid=self.num_features
        ).to(self.device)
        
        # Graph spectral filtering
        self.spectral_filter = GraphSpectralFilter(self.K, self.num_features, self.nhid)
        
        # Combination weights: one scalar weight per filter (per-dimension)
        self.comb_weight = nn.Parameter(torch.ones((1, self.K, 1)))
        self.reset_comb_parameters()
        
        # Final MLP that takes pooled graph features -> class logits
        self.mlp = MLP(self.nhid, nlayers=3, nhidden=self.nhid,
                       nclass=self.num_classes, dropout=self.dropout, bias='bn')
    
    def reset_comb_parameters(self):
        # Initialize comb_weight to small random values in [-1/K, +1/K]
        bound = 1.0 / self.K
        TEMP = np.random.uniform(-bound, bound, self.K)
        self.comb_weight = nn.Parameter(torch.FloatTensor(TEMP).view(1, self.K, 1))
    
    def forward(self, x):
        """
        x: [B, C, H, W] input images
        Returns:
          out: [B, num_classes]
          graph_features: [B, nhid] (the pooled graph‐level embedding)
        """
        bs, c, H, W = x.shape
        if c == 1:
            # If grayscale, replicate to 3 channels
            x = x.repeat(1, 3, 1, 1)
        
        x = x.to(self.device)
        
        # 1) Extract patch token features from ViT
        Fet = self.extractor(x)  # [B, N_nodes, num_features]
        
        # 2) Build adjacency matrices per graph
        W = create_adj(Fet, k=4, alpha=1).to(self.device)  # [B, N_nodes, N_nodes]
        data_list = load_data(W, Fet)                      # list of PyG Data
        data = Batch.from_data_list(data_list).to(self.device)
        
        # 3) Apply spectral filters
        node_features, edge_index, batch = data.x, data.edge_index, data.batch
        # node_features: [total_nodes_in_batch, num_features]
        spectral_features = self.spectral_filter(node_features, edge_index, data.edge_attr)
        # [total_nodes, K, nhid]
        
        # 4) Weighted combination across K filters
        spectral_features = F.dropout(spectral_features, self.dropout, training=self.training)
        weighted_features = spectral_features * self.comb_weight  # broadcast [1,K,1] over [N,K,nhid]
        combined_features = torch.sum(weighted_features, dim=1)   # [total_nodes, nhid]
        
        # 5) Global mean pooling to get one nhid‐dim vector per graph
        graph_features = global_mean_pool(combined_features, batch)  # [B, nhid]
        
        # 6) Final MLP for classification
        out = self.mlp(graph_features)  # [B, num_classes]
        
        return out, graph_features


# ===================== Model 2: H2GCN (Beyond Homophily) =====================
class H2GCNConv(nn.Module):
    def __init__(self, in_channels, out_channels, use_relu=True):
        super().__init__()
        self.fc1 = nn.Linear(in_channels, out_channels)
        self.fc2 = nn.Linear(in_channels, out_channels)
        self.use_relu = use_relu
    
    def forward(self, x, edge_index):
        """
        x: [N, in_channels]
        edge_index: [2, E]
        Returns: [N, 2*out_channels] if use_relu=True, else same shape but without ReLU
        """
        row, col = edge_index
        
        # 1) Ego embedding (self‐feature)
        ego_embed = self.fc1(x)  # [N, out_channels]
        
        # 2) Neighbor aggregation
        out = torch.zeros_like(ego_embed)  # [N, out_channels]
        neighbor_messages = self.fc2(x[col])  # [E, out_channels]
        out.index_add_(0, row, neighbor_messages)  # sum over messages
        
        # Normalize by degree
        deg = torch.bincount(row, minlength=x.size(0)).float().clamp(min=1.0)  # [N]
        out = out / deg.view(-1, 1)
        
        # 3) Concatenate ego & neighbor embeddings
        out = torch.cat([ego_embed, out], dim=1)  # [N, 2*out_channels]
        
        if self.use_relu:
            out = F.relu(out)
        return out

class H2GCN(nn.Module):
    def __init__(self, args):
        super(H2GCN, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.num_features = args.num_features
        self.nhid = args.nhid
        self.num_classes = args.num_classes
        self.dropout = args.dropout_ratio
        self.num_layers = args.h2gcn_layers
        
        # ViT feature extractor (frozen)
        self.extractor = ViTExtractor(
            model_name='vit_base_patch16_224',
            pretrained=True,
            nhid=self.num_features
        ).to(self.device)
        
        # Build H2GCN layers
        self.convs = nn.ModuleList()
        # First H2GCNConv: from num_features -> nhid, after concat becomes 2*nhid
        self.convs.append(H2GCNConv(self.num_features, self.nhid))
        # Keep track of how many convs we have
        for _ in range(self.num_layers - 1):
            # Each subsequent conv takes input_dim = 2*nhid (output dimension of a previous conv)
            self.convs.append(H2GCNConv(2 * self.nhid, self.nhid))
        
        # Compute the total feature dimension after concatenating all conv outputs:
        # Each conv outputs a tensor of shape [N, 2*nhid], and we will concatenate across all self.num_layers of them.
        self.final_dim = self.num_layers * (2 * self.nhid)
        
        # Final MLP layers on pooled graph embeddings
        self.fc1 = nn.Linear(self.final_dim, self.nhid)
        self.fc2 = nn.Linear(self.nhid, self.num_classes)
        self.dropout_layer = nn.Dropout(self.dropout)
    
    def forward(self, x):
        """
        x: [B, C, H, W] input images
        Returns:
          out: [B, num_classes]
          graph_features: [B, final_dim]
        """
        bs, c, H, W = x.shape
        if c == 1:
            x = x.repeat(1, 3, 1, 1)
        
        x = x.to(self.device)
        
        # 1) Extract patch token features
        Fet = self.extractor(x)  # [B, N_nodes, num_features]
        
        # 2) Build adjacency and wrap in PyG Batch
        W = create_adj(Fet, k=4, alpha=1).to(self.device)  # [B, N_nodes, N_nodes]
        data_list = load_data(W, Fet)
        data = Batch.from_data_list(data_list).to(self.device)
        node_features, edge_index, batch = data.x, data.edge_index, data.batch
        
        # 3) Build 2-hop neighbor edge_index
        edge_index_2hop = self.get_2hop_neighbors(edge_index, node_features.size(0))
        
        # 4) Apply H2GCN layers
        h = node_features
        h_list = []
        for conv in self.convs:
            h = self.dropout_layer(h)
            h = conv(h, edge_index_2hop)  # each conv returns [N, 2*nhid]
            h_list.append(h)
        
        # Concatenate features from all layers along dimension=1
        if len(h_list) > 1:
            h = torch.cat(h_list, dim=1)  # [N, num_layers * 2*nhid]
        else:
            h = h_list[0]                 # [N, 2*nhid]
        
        # 5) Global mean pooling
        graph_features = global_mean_pool(h, batch)  # [B, final_dim]
        
        # 6) Final classification MLP on pooled features
        h_pool = self.dropout_layer(graph_features)
        h_pool = F.relu(self.fc1(h_pool))  # [B, nhid]
        h_pool = self.dropout_layer(h_pool)
        out = self.fc2(h_pool)             # [B, num_classes]
        
        return out, graph_features
    
    def get_2hop_neighbors(self, edge_index, num_nodes):
        """
        Compute 2-hop adjacency matrix from 1-hop edges, then return its edge_index.
        """
        device = edge_index.device
        row, col = edge_index
        
        # Build symmetric 1-hop adjacency
        adj = torch.zeros((num_nodes, num_nodes), device=device)
        adj[row, col] = 1.0
        adj[col, row] = 1.0
        
        # Compute A^2, then threshold to get 2-hop
        adj_2hop = torch.mm(adj, adj)
        adj_2hop.fill_diagonal_(0)      # remove self‐loops
        adj_2hop = (adj_2hop > 0).float()
        
        edge_index_2hop = adj_2hop.nonzero(as_tuple=True)
        edge_index_2hop = torch.stack(edge_index_2hop, dim=0)  # [2, E2]
        return edge_index_2hop


# ===================== Training and Evaluation Functions =====================
def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    
    for batch_idx, (data, target) in enumerate(tqdm(loader, desc="Training")):
        # data: batch of images, target: batch of labels
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        output, _ = model(data)            # forward pass
        loss = criterion(output, target)   # cross‐entropy
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        total += target.size(0)
    
    return total_loss / len(loader), correct / total

def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_targets = []
    all_probs = []
    
    with torch.no_grad():
        for data, target in tqdm(loader, desc="Evaluating"):
            data, target = data.to(device), target.to(device)
            output, _ = model(data)
            loss = criterion(output, target)
            
            total_loss += loss.item()
            probs = F.softmax(output, dim=1)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)
            
            all_preds.extend(pred.cpu().numpy().reshape(-1))
            all_targets.extend(target.cpu().numpy().reshape(-1))
            all_probs.append(probs.cpu().numpy())
    
    # Compute metrics
    acc = correct / total
    f1 = f1_score(all_targets, all_preds, average='macro')
    all_probs = np.concatenate(all_probs, axis=0)
    all_targets_np = np.array(all_targets)
    
    # Compute multiclass AUC (one-vs-rest, macro average)
    try:
        auc = roc_auc_score(all_targets_np, all_probs, multi_class='ovr', average='macro')
    except ValueError:
        # If only one class present in targets, AUC is undefined → set to nan
        auc = float('nan')
    
    return total_loss / len(loader), acc, f1, auc



def count_parameters(model: nn.Module):
    """
    Prints the number of trainable and non-trainable parameters in a PyTorch model.

    Args:
        model (torch.nn.Module): The PyTorch model.
    """
    trainable_params = 0
    non_trainable_params = 0
    for name, param in model.named_parameters():
        if param.requires_grad:
            trainable_params += param.numel()
        else:
            non_trainable_params += param.numel()
    print(f"Total Trainable Parameters: {trainable_params:,}")
    print(f"Total Non-Trainable Parameters: {non_trainable_params:,}")
    print(f"Total Parameters: {(trainable_params + non_trainable_params):,}")

# ===================== Main Training Script =====================
def main():
    parser = argparse.ArgumentParser(description='Heterophilic GNN Benchmark')
    
    # Dataset arguments /media/annatar/OLDHDD/all_random_datasets/archive/HAM10000_organized
    parser.add_argument('--data_dir', type=str, default='pascal', 
                        help='Path to dataset directory')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_workers', type=int, default=4)
    
    # Model arguments
    parser.add_argument('--model', type=str, default='h2gcn', 
                        choices=['gfk', 'h2gcn', 'original'],
                        help='Model to use')
    parser.add_argument('--num_features', type=int, default=384)
    parser.add_argument('--nhid', type=int, default=256)
    parser.add_argument('--dropout_ratio', type=float, default=0.5)
    
    # GFK specific
    parser.add_argument('--gfk_K', type=int, default=5, 
                        help='Number of spectral filters for GFK')
    
    # H2GCN specific
    parser.add_argument('--h2gcn_layers', type=int, default=3,
                        help='Number of layers for H2GCN')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', type=str, default='cuda')
    
    # Original model specific (if needed)
    parser.add_argument('--alpha', type=float, default=0.5)
    parser.add_argument('--kernels', type=int, default=3)
    parser.add_argument('--num_heads', type=int, default=4)
    parser.add_argument('--mean_num_nodes', type=int, default=196)
    parser.add_argument('--num_classes', type=int, default=20)
    parser.add_argument('--pooling_ratio', type=float, default=0.5)
    parser.add_argument('--num_atoms', type=int, default=4)
    
    args = parser.parse_args()
    
    # Set random seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    
    # Data transforms
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.3),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    
    # Load datasets
    train_dir = os.path.join(args.data_dir, "train")
    test_dir = os.path.join(args.data_dir, "test")
    if not os.path.exists(train_dir) or not os.path.exists(test_dir):
        raise ValueError(f"Dataset not found. Expected: {train_dir}, {test_dir}")
    
    full_train_dataset = datasets.ImageFolder(root=train_dir, transform=train_transform)
    test_dataset = datasets.ImageFolder(root=test_dir, transform=test_transform)
    
    n_classes = len(full_train_dataset.classes)
    args.num_classes = n_classes
    print(f"Dataset: {n_classes} classes")
    print(f"Classes: {full_train_dataset.classes}")
    
    # Split train into train/val
    train_size = int(0.8 * len(full_train_dataset))
    val_size = len(full_train_dataset) - train_size
    g = torch.Generator().manual_seed(args.seed)
    train_dataset, val_dataset = random_split(full_train_dataset, [train_size, val_size], generator=g)
    
    # Apply test transform to validation set
    val_dataset.dataset.transform = test_transform
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers, pin_memory=True)
    
    # Instantiate model
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    if args.model == 'gfk':
        model = GFKNet(args).to(device)
    elif args.model == 'h2gcn':
        model = H2GCN(args).to(device)
    else:
        # If using your "original" model, ensure attndictmodel.py is in the same directory
        from attndictmodel import Net
        model = Net(args).to(device)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=10, factor=0.5)
    

    count_parameters(model)

    # Training loop

    best_val_acc = 0.0
    best_model_state = None
    
    print(f"\nTraining {args.model.upper()} model...")
    print("=" * 50)
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        print("-" * 30)
        
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc, val_f1, val_auc = evaluate(model, val_loader, criterion, device)
        scheduler.step(val_acc)
        
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"Val Loss:   {val_loss:.4f}, Val Acc:   {val_acc:.4f}, Val F1: {val_f1:.4f}, Val AUC: {val_auc:.4f}")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict().copy()
            print(f"New best model! Val Acc: {val_acc:.4f}")
    
    # Load best model and evaluate on test set
    model.load_state_dict(best_model_state)
    test_loss, test_acc, test_f1, test_auc = evaluate(model, test_loader, criterion, device)
    
    print("\n" + "=" * 50)
    print(f"Final Test Results for {args.model.upper()}:")
    print(f"Test Loss:     {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")
    print(f"Test F1 Score: {test_f1:.4f}")
    print(f"Test AUC:      {test_auc:.4f}")
    print("=" * 50)
    
    # Save results to JSON
    results = {
        'model': args.model,
        'test_acc': test_acc,
        'test_f1': test_f1,
        'test_auc': test_auc,
        'best_val_acc': best_val_acc,
        'args': vars(args)
    }
    import json
    with open(f'results_{args.model}_{int(time.time())}.json', 'w') as f:
        json.dump(results, f, indent=4)


if __name__ == "__main__":
    main()

