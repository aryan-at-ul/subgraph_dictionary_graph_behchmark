import os
import logging
import argparse
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data

from torch_geometric.data import Data, DataLoader, Batch
from torch_geometric.nn import GCNConv, global_add_pool

import torchvision.transforms as transforms

import medmnist

import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import label_binarize
from sklearn.cluster import KMeans

from features_extract import deep_features
from extractor import ViTExtractor
from attndictmodel import Net 


def setseed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False



def parse_arguments():
    parser = argparse.ArgumentParser(description='Training script')

    parser.add_argument('--alpha', type=float, default=0.1, help='Alpha value')
    parser.add_argument('--kernels', type=int, default=3, help='Number of kernels in MKGC')
    parser.add_argument('--num_features', type=int, default=768, help='Input feature dimension')
    parser.add_argument('--nhid', type=int, default=64, help='Hidden dimension')
    parser.add_argument('--num_heads', type=int, default=4, help='Number of heads for attention')
    parser.add_argument('--mean_num_nodes', type=int, default=30, help='Average number of nodes per graph')
    parser.add_argument('--pooling_ratio', type=float, default=0.5, help='Pooling ratio')
    parser.add_argument('--dropout_ratio', type=float, default=0.5, help='Dropout ratio')
    parser.add_argument('--num_atoms', type=int, default=10, help='Number of atoms in the dictionary')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to use')
    parser.add_argument('--data_flag', type=str, default='tissuemnist',
                        choices=['tissuemnist', 'pathmnist', 'chestmnist', 'dermamnist', 'octmnist', 'pnemoniamnist',
                                 'retinamnist', 'breastmnist', 'bloodmnist', 'organamnist', 'organcmnist',
                                 'organsmnist'], help='Dataset flag')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=1000, help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay')
    parser.add_argument('--seed', type=int, default=42, help='seed')
    parser.add_argument('--testing', action='store_true', default=False,
                        help='If true, load the saved model and run testing only')

    return parser.parse_args()


class Args:
    def __init__(self, args, num_classes):
        self.alpha = args.alpha
        self.kernels = args.kernels
        self.num_features = args.num_features
        self.nhid = args.nhid
        self.num_heads = args.num_heads
        self.mean_num_nodes = args.mean_num_nodes
        self.num_classes = num_classes
        self.pooling_ratio = args.pooling_ratio
        self.dropout_ratio = args.dropout_ratio
        self.num_atoms = args.num_atoms
        self.device = torch.device(args.device)


def calculate_label_percentage(dataset, n_classes, logger):
    label_counts = {i: 0 for i in range(n_classes)}
    total_samples = len(dataset)
    return_percent = {}

    for item in dataset:
        _, y = item
        label = y.item()
        label_counts[label] += 1

    for label, count in label_counts.items():
        percentage = (count / total_samples) * 100
        logger.info(f"Label {label}: {percentage:.2f}%")
        return_percent[label] = percentage

    return list(return_percent.values())


def train(model, device, train_loader, optimizer, criterion_cls, logger):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device).squeeze()

        optimizer.zero_grad()
        outputs, _ = model(images)
        loss = criterion_cls(outputs, labels)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        total_loss += loss.item()

    accuracy = 100. * correct / total
    logger.info(f"Training Loss: {total_loss / len(train_loader):.4f}, Accuracy: {accuracy:.2f}%")


def test(model, device, test_loader, criterion_cls, logger):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    all_labels = []
    all_preds = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device).squeeze()

            outputs, _ = model(images)
            loss = criterion_cls(outputs, labels)

            total_loss += loss.item()

            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            all_labels.append(labels.cpu())
            all_preds.append(outputs.cpu())

    all_labels = torch.cat(all_labels)
    all_preds = torch.cat(all_preds)

    num_classes = all_preds.size(1)
    all_labels_np = all_labels.numpy()
    all_preds_np = all_preds.numpy()

    if num_classes == 2:
        # Binary classification
        y_true = all_labels_np
        y_score = F.softmax(torch.from_numpy(all_preds_np), dim=1).numpy()[:, 1]
        try:
            auc = roc_auc_score(y_true, y_score)
        except ValueError:
            auc = float('nan')
    else:
        # Multi-class classification
        y_true = all_labels_np
        y_score = F.softmax(torch.from_numpy(all_preds_np), dim=1).numpy()
        y_true_bin = label_binarize(y_true, classes=np.arange(num_classes))
        try:
            auc = roc_auc_score(y_true_bin, y_score, multi_class='ovr')
        except ValueError:
            auc = float('nan')

    accuracy = 100. * correct / total
    logger.info(
        f"Test Loss: {total_loss / len(test_loader):.4f}, Accuracy: {accuracy:.2f}%, AUC: {auc:.4f}")
    return accuracy


def main():
    args = parse_arguments()

    setseed(args.seed)

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)


    device = torch.device(args.device)
    logger.info(f"Using device: {device}")


    data_flag = args.data_flag.lower()
    logger.info(f"Using dataset: {data_flag}")


    info = medmnist.INFO[data_flag]
    n_channels = info['n_channels']
    n_classes = len(info['label'])
    task = info['task']
    logger.info(f"Number of classes: {n_classes}, Number of channels: {n_channels}, Task: {task}")


    my_args = Args(args, n_classes)


    DataClass = getattr(medmnist, info['python_class'])
    download = True


    data_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])


    train_dataset = DataClass(split='train', transform=data_transform, download=download)
    test_dataset = DataClass(split='test', transform=data_transform, download=download)


    train_loader = data.DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = data.DataLoader(dataset=test_dataset, batch_size=2 * args.batch_size, shuffle=False)


    model = Net(my_args).to(device)
    logger.info(f"Initialized model with {sum(p.numel() for p in model.parameters() if p.requires_grad)} parameters.")


    label_percentages = calculate_label_percentage(train_dataset, n_classes, logger)


    label_proportions = [p / 100 for p in label_percentages]
    weights = [1.0 / p if p > 0 else 0.0 for p in label_proportions]
    sum_weights = sum(weights)
    normalized_weights = [n_classes * w / sum_weights for w in weights]
    weights_tensor = torch.FloatTensor(normalized_weights).to(device)
    logger.info(f"Weights for CrossEntropyLoss: {weights_tensor}")

    # criterion_cls = torch.nn.CrossEntropyLoss(weight=weights_tensor)
    criterion_cls = torch.nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    model_path = f"best_model_{data_flag}.pth"

    if args.testing:

        if os.path.exists(model_path):
            model.load_state_dict(torch.load(model_path, map_location=device))
            logger.info(f"Loaded model from {model_path}")
            test(model, device, test_loader, criterion_cls, logger)
        else:
            logger.error(f"Model file {model_path} does not exist. Cannot perform testing.")
    else:

        best_accuracy = 0.0

        for epoch in range(args.epochs):
            logger.info(f'Epoch {epoch + 1}/{args.epochs}')
            train(model, device, train_loader, optimizer, criterion_cls, logger)
            accuracy = test(model, device, test_loader, criterion_cls, logger)

            if accuracy > best_accuracy:
                logger.info(f"Saving new best model with accuracy: {accuracy:.2f}%")
                best_accuracy = accuracy
                torch.save(model.state_dict(), model_path)


if __name__ == '__main__':
    main()
