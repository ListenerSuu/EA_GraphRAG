"""
MLP-based Query Complexity Classifier
Predicts query complexity to route between dense and graph-based retrieval.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Optional
import json


class FeatureAttention(nn.Module):
    """Feature-level attention mechanism"""
    def __init__(self, input_dim):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(input_dim // 2, input_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        weights = self.attention(x)
        return x * weights


class ResidualBlock(nn.Module):
    """Residual block for deep networks"""
    def __init__(self, dim: int, dropout: float = 0.2):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim)
        )
        self.activation = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        return self.dropout(self.activation(self.block(x) + x))


class OptimizedMLPClassifier(nn.Module):
    """
    Optimized MLP classifier with feature attention and residual connections.
    Predicts probability that graph-based retrieval should be used (vs dense retrieval).
    """
    def __init__(self, input_dim: int, hidden_dims=(256, 128, 64), p_dropout=0.3, use_residual=True):
        super().__init__()
        
        # Feature attention
        self.feature_attention = FeatureAttention(input_dim)
        
        layers = []
        prev = input_dim
        
        # Input projection
        layers.append(nn.Sequential(
            nn.Linear(prev, hidden_dims[0]),
            nn.BatchNorm1d(hidden_dims[0]),
            nn.ReLU(inplace=True),
            nn.Dropout(p_dropout * 0.5)
        ))
        prev = hidden_dims[0]
        
        # Hidden layers with residual connections
        for i, h in enumerate(hidden_dims[1:]):
            if use_residual and prev == h:
                layers.append(ResidualBlock(h, p_dropout))
            else:
                layers.append(nn.Sequential(
                    nn.Linear(prev, h),
                    nn.BatchNorm1d(h),
                    nn.ReLU(inplace=True),
                    nn.Dropout(p_dropout)
                ))
            prev = h
        
        self.feature_layers = nn.ModuleList(layers)
        
        # Output layer with attention
        self.output_attention = nn.Sequential(
            nn.Linear(prev, prev // 2),
            nn.Tanh(),
            nn.Linear(prev // 2, prev),
            nn.Sigmoid()
        )
        self.output = nn.Sequential(
            nn.Linear(prev, 32),
            nn.ReLU(inplace=True),
            nn.Dropout(p_dropout * 0.5),
            nn.Linear(32, 1)
        )
    
    def forward(self, x):
        # Apply feature attention
        x = self.feature_attention(x)
        
        # Pass through feature layers
        for layer in self.feature_layers:
            x = layer(x)
        
        # Output attention
        attn_weights = self.output_attention(x)
        x = x * attn_weights
        
        logit = self.output(x).squeeze(1)
        return logit


def load_classifier(checkpoint_path: str, device: str = 'cpu') -> Tuple[OptimizedMLPClassifier, dict]:
    """
    Load a trained classifier from checkpoint.
    
    Args:
        checkpoint_path: Path to the checkpoint file
        device: Device to load the model on
        
    Returns:
        Tuple of (model, metadata) where metadata contains feature_keys, scaler params, etc.
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Extract metadata
    feature_keys = checkpoint.get('feature_keys', [])
    scaler_mean = np.array(checkpoint.get('scaler_mean', []))
    scaler_std = np.array(checkpoint.get('scaler_std', []))
    input_dim = checkpoint.get('input_dim', len(feature_keys))
    selected_features = checkpoint.get('selected_features', None)
    
    # Create model
    model = OptimizedMLPClassifier(input_dim=input_dim, hidden_dims=(256, 128, 64), 
                                   p_dropout=0.3, use_residual=True)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    metadata = {
        'feature_keys': feature_keys,
        'scaler_mean': scaler_mean,
        'scaler_std': scaler_std,
        'selected_features': selected_features,
        'input_dim': input_dim
    }
    
    return model, metadata


def predict_complexity(model: OptimizedMLPClassifier, features: np.ndarray, 
                       threshold: float = 0.5) -> Tuple[float, str]:
    """
    Predict query complexity and routing decision.
    
    Args:
        model: Trained classifier model
        features: Feature vector (normalized)
        threshold: Decision threshold (default 0.5)
        
    Returns:
        Tuple of (probability, complexity_level) where complexity_level is 
        'low', 'medium', or 'high'
    """
    with torch.no_grad():
        x = torch.from_numpy(features.astype(np.float32)).unsqueeze(0)
        logit = model(x)
        prob = torch.sigmoid(logit).item()
    
    # Map probability to complexity levels
    # Low: prob < 0.3 -> use dense retrieval
    # Medium: 0.3 <= prob < 0.7 -> use fusion
    # High: prob >= 0.7 -> use graph retrieval
    if prob < 0.3:
        level = 'low'
    elif prob < 0.7:
        level = 'medium'
    else:
        level = 'high'
    
    return prob, level

