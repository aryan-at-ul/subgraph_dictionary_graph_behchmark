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
from torchvision.datasets import ImageFolder

import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import label_binarize
from sklearn.cluster import KMeans

# from features_extract import deep_features
# from extractor import ViTExtractor
from attndictmodel import Net

from tqdm import tqdm  # Import tqdm

def setseed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # torch.backends.cudnn.deterministic = True  
    # torch.backends.cudnn.benchmark = False     
    # torch.backends.cudnn.enabled = False       

def parse_arguments():
    parser = argparse.ArgumentParser(description='Training script')

    parser.add_argument('--alpha', type=float, default=0.1, help='Alpha value')
    parser.add_argument('--kernels', type=int, default=4, help='Number of kernels in MKGC')
    parser.add_argument('--num_features', type=int, default=768, help='Input feature dimension')
    parser.add_argument('--nhid', type=int, default=64, help='Hidden dimension')
    parser.add_argument('--num_heads', type=int, default=4, help='Number of heads for attention')
    parser.add_argument('--mean_num_nodes', type=int, default=30, help='Average number of nodes per graph')
    parser.add_argument('--pooling_ratio', type=float, default=0.5, help='Pooling ratio')
    parser.add_argument('--dropout_ratio', type=float, default=0.5, help='Dropout ratio')
    parser.add_argument('--num_atoms', type=int, default=30, help='Number of atoms in the dictionary')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to use')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay')
    parser.add_argument('--seed', type=int, default=42, help='Seed for reproducibility')
    parser.add_argument('--testing', action='store_true', default=False,
                        help='If true, load the saved model and run testing only')
    parser.add_argument('--data_dir', type=str, default='pascal',
                        help='Directory containing the Pascal VOC data organized in train and test folders')

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

    for _, y in dataset:
        label = y
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

    for images, labels in tqdm(train_loader, desc='Training', leave=False):
        images = images.to(device)
        labels = labels.to(device)

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

def test(model, device, test_loader, criterion_cls, logger, mode='Validation'):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    all_labels = []
    all_preds = []

    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc=f'{mode}', leave=False):
            images = images.to(device)
            labels = labels.to(device)

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

    y_true = all_labels_np
    y_score = F.softmax(torch.from_numpy(all_preds_np), dim=1).numpy()
    y_true_bin = label_binarize(y_true, classes=np.arange(num_classes))

    try:
        auc = roc_auc_score(y_true_bin, y_score, multi_class='ovr')
    except ValueError:
        auc = float('nan')

    accuracy = 100. * correct / total
    avg_loss = total_loss / len(test_loader)
    logger.info(
        f"{mode} Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%, AUC: {auc:.4f}")
    return accuracy, avg_loss

def main():
    args = parse_arguments()

    setseed(args.seed)

    log_filename = 'pascalvoc_gnn_trainval_loss.log'

    # Set up logging to work with tqdm
    class TqdmLoggingHandler(logging.Handler):
        def __init__(self, level=logging.NOTSET):
            super().__init__(level)

        def emit(self, record):
            try:
                msg = self.format(record)
                tqdm.write(msg)
                self.flush()
            except Exception:
                self.handleError(record)

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    c_handler = TqdmLoggingHandler()
    f_handler = logging.FileHandler(log_filename)
    c_handler.setLevel(logging.INFO)
    f_handler.setLevel(logging.INFO)

    c_format = logging.Formatter('%(message)s')
    f_format = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    c_handler.setFormatter(c_format)
    f_handler.setFormatter(f_format)

    logger.addHandler(c_handler)
    logger.addHandler(f_handler)

    device = torch.device(args.device)
    logger.info(f"Using device: {device}")

    logger.info(f"Using Pascal VOC dataset")

    data_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
    ])

    train_dir = os.path.join(args.data_dir, 'train')
    test_dir = os.path.join(args.data_dir, 'test')

    full_train_dataset = ImageFolder(root=train_dir, transform=data_transform)
    test_dataset = ImageFolder(root=test_dir, transform=data_transform)

    n_classes = len(full_train_dataset.classes)
    logger.info(f"Number of classes: {n_classes}")
    my_args = Args(args, n_classes)

    # Split the full train dataset into train and validation sets
    train_size = int(0.8 * len(full_train_dataset))
    val_size = len(full_train_dataset) - train_size
    generator = torch.Generator().manual_seed(args.seed)
    train_dataset, val_dataset = data.random_split(full_train_dataset, [train_size, val_size], generator=generator)

    # Create data loaders
    train_loader = data.DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = data.DataLoader(dataset=val_dataset, batch_size=2 * args.batch_size, shuffle=False)
    test_loader = data.DataLoader(dataset=test_dataset, batch_size=2 * args.batch_size, shuffle=False)

    model = Net(my_args).to(device)
    logger.info(f"Initialized model with {sum(p.numel() for p in model.parameters() if p.requires_grad)} parameters.")

    # Calculate label percentages on the training set
    label_percentages = calculate_label_percentage(train_dataset, n_classes, logger)

    label_proportions = [p / 100 for p in label_percentages]
    weights = [1.0 / p if p > 0 else 0.0 for p in label_proportions]
    sum_weights = sum(weights)
    normalized_weights = [n_classes * w / sum_weights for w in weights]
    weights_tensor = torch.FloatTensor(normalized_weights).to(device)
    logger.info(f"Weights for CrossEntropyLoss: {weights_tensor}")

    criterion_cls = torch.nn.CrossEntropyLoss()
    # criterion_cls = torch.nn.CrossEntropyLoss(weight=weights_tensor)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    model_path = f"best_model_pascalvoc_trainval_loss.pth"

    if args.testing:

        if os.path.exists(model_path):
            model.load_state_dict(torch.load(model_path, map_location=device))
            logger.info(f"Loaded model from {model_path}")
            test_accuracy, test_loss = test(model, device, test_loader, criterion_cls, logger, mode='Test')
            logger.info(f"Test accuracy: {test_accuracy:.2f}%, Loss: {test_loss:.4f}")
        else:
            logger.error(f"Model file {model_path} does not exist. Cannot perform testing.")
    else:

        best_val_loss = float('inf')

        for epoch in range(args.epochs):
            logger.info(f'Epoch {epoch + 1}/{args.epochs}')
            train(model, device, train_loader, optimizer, criterion_cls, logger)
            val_accuracy, val_loss = test(model, device, val_loader, criterion_cls, logger, mode='Validation')

            if val_loss < best_val_loss:
                logger.info(f"Saving new best model with validation loss: {val_loss:.4f}")
                best_val_loss = val_loss
                torch.save(model.state_dict(), model_path)

        # Load the best model and evaluate on the test set
        model.load_state_dict(torch.load(model_path, map_location=device))
        logger.info(f"Loaded best model from {model_path}")
        test_accuracy, test_loss = test(model, device, test_loader, criterion_cls, logger, mode='Test')
        logger.info(f"Final Test accuracy: {test_accuracy:.2f}%, Loss: {test_loss:.4f}")

if __name__ == '__main__':
    main()
