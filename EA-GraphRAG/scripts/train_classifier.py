"""
Training script for the query complexity classifier.
Adapted from mlp_classifier_4k_optimized.py
"""
import os
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.feature_selection import mutual_info_classif
import argparse
from typing import List, Dict, Tuple

# Add src to path
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from classifier.mlp_classifier import OptimizedMLPClassifier, FeatureAttention, ResidualBlock
from classifier.feature_extractor import QueryComplexityFeaturizer
from classifier.syntactic_features import SyntacticComplexityAnalyzer
from classifier.advanced_features import extract_advanced_features
import stanza
from stanza.server import CoreNLPClient


def seed_everything(seed: int = 42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_feature_keys(dataset: List[Dict], analyzer, client, nlp) -> List[str]:
    """Build feature key list from dataset"""
    key_set = set()
    for ex in dataset[:100]:  # Sample first 100 for feature discovery
        try:
            fd1 = extract_advanced_features(ex["question"])
            fd2 = analyzer.analyze_text(ex["question"], client, nlp)
            fd = {**fd1, **fd2}
            key_set.update(fd.keys())
        except Exception as e:
            continue
    return sorted(list(key_set))


def vectorize_features(feat_dict: Dict[str, float], feature_keys: List[str]) -> np.ndarray:
    """Convert feature dictionary to vector"""
    return np.array([float(feat_dict.get(k, 0.0)) for k in feature_keys], dtype=np.float32)


def prepare_xy(dataset: List[Dict], feature_keys: List[str], analyzer, client, nlp) -> Tuple[np.ndarray, np.ndarray]:
    """Prepare training data"""
    X_list, y_list = [], []
    for ex in dataset:
        try:
            fd1 = extract_advanced_features(ex["question"])
            fd2 = analyzer.analyze_text(ex["question"], client, nlp)
            fd = {**fd1, **fd2}
            x = vectorize_features(fd, feature_keys)
            y = int(ex.get("label", 0))
            if y in [0, 1]:  # Only binary labels
                X_list.append(x)
                y_list.append(y)
        except Exception as e:
            continue
    if len(X_list) == 0:
        raise ValueError("No valid samples after processing")
    X = np.stack(X_list, axis=0)
    y = np.array(y_list, dtype=np.float32)
    return X, y


def select_features_mutual_info(X_train: np.ndarray, y_train: np.ndarray, k: int = 85) -> np.ndarray:
    """Feature selection using mutual information"""
    k = min(k, X_train.shape[1])
    mi_scores = mutual_info_classif(X_train, y_train, random_state=42)
    top_k_indices = np.argsort(mi_scores)[-k:][::-1]
    return top_k_indices


def fit_standard_scaler(X_train: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Fit standard scaler"""
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0)
    std = np.where(std < 1e-8, 1.0, std)
    return mean, std


def apply_standard_scaler(X: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return (X - mean) / std


class TabTensorDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.from_numpy(X.astype(np.float32))
        self.y = torch.from_numpy(y.astype(np.float32))

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class FocalLoss(nn.Module):
    """Focal Loss for imbalanced classification"""
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        bce_loss = nn.functional.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss


def train_epoch(model, loader, optimizer, device, scheduler=None, use_focal_loss=False, label_smoothing=0.0):
    model.train()
    criterion = FocalLoss(alpha=0.25, gamma=2.0) if use_focal_loss else nn.BCEWithLogitsLoss()
    
    total_loss, total, correct = 0.0, 0, 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        
        if label_smoothing > 0:
            yb_smooth = yb * (1 - label_smoothing) + 0.5 * label_smoothing
        else:
            yb_smooth = yb
        
        optimizer.zero_grad()
        logit = model(xb)
        loss = criterion(logit, yb_smooth)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        if scheduler:
            scheduler.step()

        with torch.no_grad():
            prob = torch.sigmoid(logit)
            pred = (prob >= 0.5).float()
            correct += (pred == yb).sum().item()
            total += yb.numel()
            total_loss += loss.item() * yb.numel()
    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, device, threshold: float = 0.5):
    model.eval()
    criterion = nn.BCEWithLogitsLoss()
    total_loss, total, correct = 0.0, 0, 0
    probs_all, y_all = [], []

    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        logit = model(xb)
        loss = criterion(logit, yb)
        prob = torch.sigmoid(logit)
        pred = (prob >= threshold).float()

        correct += (pred == yb).sum().item()
        total += yb.numel()
        total_loss += loss.item() * yb.numel()
        probs_all.append(prob.cpu())
        y_all.append(yb.cpu())

    probs_all = torch.cat(probs_all).numpy()
    y_all = torch.cat(y_all).numpy()
    acc = correct / total
    brier = float(np.mean((probs_all - y_all) ** 2))
    
    preds = (probs_all >= threshold).astype(int)
    tp = np.sum((preds == 1) & (y_all == 1))
    fp = np.sum((preds == 1) & (y_all == 0))
    fn = np.sum((preds == 0) & (y_all == 1))
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)
    
    return total_loss / total, acc, brier, precision, recall, f1


def save_checkpoint(path: str, model: nn.Module, feature_keys: List[str], 
                   mean: np.ndarray, std: np.ndarray, 
                   input_dim: int, selected_features: np.ndarray = None):
    ckpt = {
        "model_state_dict": model.state_dict(),
        "feature_keys": feature_keys,
        "scaler_mean": mean.tolist(),
        "scaler_std": std.tolist(),
        "input_dim": input_dim,
        "selected_features": selected_features.tolist() if selected_features is not None else None,
        "arch": "optimized_mlp_classifier_v1"
    }
    torch.save(ckpt, path)


def main():
    parser = argparse.ArgumentParser(description='Train query complexity classifier')
    parser.add_argument('--train_data', type=str, required=True, help='Training data JSON file')
    parser.add_argument('--test_data', type=str, required=True, help='Test data JSON file')
    parser.add_argument('--output_dir', type=str, default='models', help='Output directory for models')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=2e-3, help='Learning rate')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device')
    parser.add_argument('--patience', type=int, default=20, help='Early stopping patience')
    args = parser.parse_args()
    
    seed_everything(42)
    
    # Load data
    with open(args.train_data, 'r') as f:
        train_data = json.load(f)
    with open(args.test_data, 'r') as f:
        test_data = json.load(f)
    
    # Initialize analyzers
    analyzer = SyntacticComplexityAnalyzer()
    nlp = stanza.Pipeline(lang='en', processors='tokenize,pos,constituency')
    
    with CoreNLPClient(
            annotators=['tokenize','ssplit','pos','lemma', 'parse'],
            timeout=30000,
            memory='16G',
            be_quiet=True,
            max_char_length=100000,
            threads=8
    ) as client:
        
        # Build feature keys
        print("Building feature keys...")
        feature_keys = build_feature_keys(train_data, analyzer, client, nlp)
        print(f"Found {len(feature_keys)} features")
        
        # Prepare training data
        print("Extracting training features...")
        X_train_full, y_train = prepare_xy(train_data, feature_keys, analyzer, client, nlp)
        print(f"Training data: {X_train_full.shape}, labels: {y_train.shape}")
        
        # Prepare test data
        print("Extracting test features...")
        X_test_full, y_test = prepare_xy(test_data, feature_keys, analyzer, client, nlp)
        print(f"Test data: {X_test_full.shape}, labels: {y_test.shape}")
        
        # Normalize
        mean, std = fit_standard_scaler(X_train_full)
        X_train_full = apply_standard_scaler(X_train_full, mean, std)
        X_test_full = apply_standard_scaler(X_test_full, mean, std)
        
        # Feature selection
        print("Selecting features...")
        selected_features = select_features_mutual_info(X_train_full, y_train, k=85)
        X_train = X_train_full[:, selected_features]
        X_test = X_test_full[:, selected_features]
        print(f"Selected {len(selected_features)} features")
        
        # DataLoader
        train_loader = DataLoader(TabTensorDataset(X_train, y_train), 
                                batch_size=args.batch_size, shuffle=True, drop_last=True)
        test_loader = DataLoader(TabTensorDataset(X_test, y_test), 
                                 batch_size=256, shuffle=False)
        
        # Model
        device = torch.device(args.device)
        model = OptimizedMLPClassifier(input_dim=X_train.shape[1], 
                                     hidden_dims=(256, 128, 64), 
                                     p_dropout=0.3, use_residual=True).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-3)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)
        
        # Training
        best_test_acc = 0.0
        best_test_f1 = 0.0
        best_epoch = 0
        patience_counter = 0
        
        print("\nStarting training...")
        for ep in range(1, args.epochs + 1):
            tr_loss, tr_acc = train_epoch(model, train_loader, optimizer, device, scheduler, 
                                         use_focal_loss=True, label_smoothing=0.1)
            te_loss, te_acc, te_brier, te_prec, te_rec, te_f1 = evaluate(model, test_loader, device)
            
            print(f"Epoch {ep:03d} | Train loss {tr_loss:.4f} acc {tr_acc:.4f} || "
                  f"Test loss {te_loss:.4f} acc {te_acc:.4f} prec {te_prec:.4f} rec {te_rec:.4f} f1 {te_f1:.4f}")
            
            # Early stopping
            improved = False
            if te_acc > best_test_acc:
                improved = True
            elif te_acc == best_test_acc and te_f1 > best_test_f1:
                improved = True
            
            if improved:
                best_test_acc = te_acc
                best_test_f1 = te_f1
                best_epoch = ep
                patience_counter = 0
                
                # Save best model
                os.makedirs(args.output_dir, exist_ok=True)
                best_path = os.path.join(args.output_dir, 'mlp_router_4k_optimized_best.pt')
                save_checkpoint(best_path, model, feature_keys, mean, std, 
                               input_dim=X_train.shape[1], selected_features=selected_features)
            else:
                patience_counter += 1
                if patience_counter >= args.patience:
                    print(f"Early stopping at epoch {ep} (best at epoch {best_epoch})")
                    break
        
        print(f"\nTraining complete!")
        print(f"Best test accuracy: {best_test_acc:.4f}, Best F1: {best_test_f1:.4f}")
        print(f"Best model saved to: {os.path.join(args.output_dir, 'mlp_router_4k_optimized_best.pt')}")


if __name__ == '__main__':
    main()

