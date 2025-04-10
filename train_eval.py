import time
import os
import torch
import numpy as np
import torch.nn.functional as F
from torch import tensor
from sklearn.model_selection import StratifiedKFold
from torch_geometric.loader import DataLoader, DenseDataLoader as DenseLoader
from sklearn.metrics import roc_auc_score 
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_auc_score

def cross_validation(dataset, model, args):

    accs, aucs, durations = [], [], []

    for fold, (train_idx, test_idx, val_idx) in enumerate(zip(*k_fold(dataset, args))):

        train_dataset = dataset[train_idx]
        test_dataset = dataset[test_idx]
        val_dataset = dataset[val_idx]

        if 'adj' in train_dataset[0]:
            train_loader = DenseLoader(train_dataset, args.batch_size, shuffle=True)
            val_loader = DenseLoader(val_dataset, args.batch_size, shuffle=False)
            test_loader = DenseLoader(test_dataset, args.batch_size, shuffle=False)
        else:
            train_loader = DataLoader(train_dataset, args.batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, args.batch_size, shuffle=False)
            test_loader = DataLoader(test_dataset, args.batch_size, shuffle=False)

        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        if torch.cuda.is_available():
            torch.cuda.synchronize(args.device)

        t_start = time.perf_counter()

        min_loss = 1e10
        max_patience = 0
        fold_train_acc, fold_valid_acc, fold_test_acc = 0, 0, 0
        fold_train_loss, fold_valid_loss, fold_test_loss = 0, 0, 0
        fold_val_auc, fold_test_auc = 0, 0
        best_epoch = 0
        for epoch in range(1, args.epochs + 1):
            train_loss, train_acc = train(model, optimizer, train_loader, args.device)
            val_loss, val_acc, val_auc = val_test(model, val_loader, args.device)
            test_loss, test_acc, test_auc = val_test(model, test_loader, args.device)

            print('{:02d}/{:03d}: train loss: {:.6f}, val loss: {:.6f}, test loss: {:.6f}; '
                  'train acc: {:.6f}, val acc: {:.6f}, test acc: {:.6f}; val auc: {:.6f}, test auc: {:.6f}'.format(
                fold+1, epoch, train_loss, val_loss, test_loss, train_acc, val_acc, test_acc, val_auc, test_auc))

            if val_loss < min_loss:
                print("Model saved at epoch {}".format(epoch))
                min_loss = val_loss
                max_patience = 0
                fold_train_acc, fold_valid_acc, fold_test_acc = train_acc, val_acc, test_acc
                fold_train_loss, fold_valid_loss, fold_test_loss = train_loss, val_loss, test_loss
                fold_val_auc, fold_test_auc = val_auc, test_auc
                best_epoch = epoch
            else:
                max_patience += 1
            if max_patience > args.patience:
                break

        if torch.cuda.is_available():
            torch.cuda.synchronize(args.device)

        t_end = time.perf_counter()
        durations.append(t_end - t_start)
        accs.append(fold_test_acc)
        aucs.append(fold_test_auc)
        print('For fold {}, test acc: {:.6f}, test auc: {:.6f}, best epoch: {}'.format(
            fold+1, fold_test_acc, fold_test_auc, best_epoch))

        with open(os.path.join(args.results, '{}_{}.txt'.format(args.experiment_number,args.model)), 'a') as f:
            f.write("fold {}: train acc: {:.4f}, valid acc: {:.4f}, test acc: {:.4f}, "
                    "train loss: {:.6f}, valid loss: {:.6f}, test loss: {:.6f}, "
                    "valid auc: {:.6f}, test auc: {:.6f}, best epoch: {}".format(
                        str(fold+1), fold_train_acc * 100, fold_valid_acc * 100, fold_test_acc * 100,
                        fold_train_loss, fold_valid_loss, fold_test_loss, fold_val_auc, fold_test_auc, best_epoch))
            f.write('\r\n')

    acc, auc, duration = tensor(accs), tensor(aucs), tensor(durations)
    acc_mean = acc.mean().item()
    acc_std = acc.std().item()
    auc_mean = auc.mean().item()
    auc_std = auc.std().item()
    duration_mean = duration.mean().item()
    overtime = time.strftime('%Y-%b-%d-%H:%M:%S', time.gmtime())
    print('Test Accuracy: {:.6f} ± {:.6f}, Test AUC: {:.6f} ± {:.6f}, Duration: {:.6f}'.format(
        acc_mean, acc_std, auc_mean, auc_std, duration_mean))
    with open(os.path.join(args.results, '{}_{}.txt'.format(args.experiment_number,args.model)), 'a') as f:
        f.write('Test Accuracy: {:.4f} ± {:.4f}, Test AUC: {:.4f} ± {:.4f}, Duration: {:.6f}'.format(
            acc_mean * 100, acc_std * 100, auc_mean, auc_std, duration_mean))
        f.write('\r\n')
        f.write('\r\n')

    return acc_mean, acc_std, duration_mean, overtime

def k_fold(dataset, args):
    skf = StratifiedKFold(args.folds, shuffle=True, random_state=args.seed)

    test_indices, train_indices = [], []
    for _, idx in skf.split(torch.zeros(len(dataset)), dataset.data.y):
        test_indices.append(torch.from_numpy(idx).to(torch.long))

    val_indices = [test_indices[i - 1] for i in range(args.folds)]

    for i in range(args.folds):
        train_mask = torch.ones(len(dataset), dtype=torch.bool)
        train_mask[test_indices[i]] = 0
        train_mask[val_indices[i]] = 0
        train_indices.append(train_mask.nonzero(as_tuple=False).view(-1))

    return train_indices, test_indices, val_indices


def num_graphs(data):
    if data.batch is not None:
        return data.num_graphs
    else:
        return data.x.size(0)


def train(model, optimizer, loader, device):
    model.train()
    train_loss = 0.
    train_correct = 0.
    for data in loader:
        data = data.to(device)
        out, cl = model(data)

        loss = F.nll_loss(out, data.y.view(-1))

        if hasattr(model, 'dictionary_module'):
            orthogonal_loss = model.dictionary_module.orthogonality_loss() * 0.1
            loss += orthogonal_loss

        pred = out.argmax(dim=1)
        train_loss += loss.item() * num_graphs(data)
        train_correct += pred.eq(data.y.view(-1)).sum().item()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    return train_loss / len(loader.dataset), train_correct / len(loader.dataset)


@torch.no_grad()
def val_test(model, loader, device):
    model.eval()
    correct = 0.
    loss = 0.
    y_true = []
    y_score = []
    for data in loader:
        data = data.to(device)
        out, cl = model(data)
        probs = out.exp() 
        pred = out.argmax(dim=1)
        # orthogonal_loss = model.dictionary_module.orthogonality_loss() * 0.2
        loss += F.nll_loss(out, data.y, reduction='sum').item() #+ orthogonal_loss.item()
        correct += pred.eq(data.y.view(-1)).sum().item()
        y_true.append(data.y.view(-1).cpu())
        y_score.append(probs.cpu())
    y_true = torch.cat(y_true).numpy().reshape(-1)
    y_score = torch.cat(y_score).numpy()
    n_classes = y_score.shape[1]
    try:
        if n_classes == 2:
            # Binary classification
            auc = roc_auc_score(y_true, y_score[:, 1])
        else:
            # Multiclass classification
            auc = roc_auc_score(y_true, y_score, multi_class='ovr')
    except ValueError as e:
        print("Error computing AUC:", e)
        auc = float('nan')
    return loss / len(loader.dataset), correct / len(loader.dataset), auc
