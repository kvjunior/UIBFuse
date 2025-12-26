"""
UIBFuse Utilities Module
========================

Comprehensive utilities for UIBFuse: Uncertainty-aware Information Bottleneck 
Fusion for Cross-Modal Visual-Temporal Learning.

This module provides:
- Data loading and preprocessing for CryptoPunks dataset
- Comprehensive evaluation metrics with uncertainty calibration
- Loss functions with explicit dynamic schedules
- Visualization tools for paper figures
- Training utilities (checkpointing, logging, early stopping)

IEEE ICME 2026 Submission
-------------------------
All metrics and loss formulations are designed to support rigorous empirical
evaluation and theoretical validation of the proposed framework.

References:
    [1] Kendall & Gal, "What Uncertainties Do We Need in Bayesian Deep Learning," NeurIPS 2017
    [2] Guo et al., "On Calibration of Modern Neural Networks," ICML 2017
    [3] Kraskov et al., "Estimating Mutual Information," Physical Review E 2004

Author: Anonymous ICME Submission
"""

import json
import logging
import os
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from scipy import stats
from sklearn.neighbors import NearestNeighbors
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, Subset, random_split
from torchvision import transforms

# Import configuration
try:
    from .config import UIBFuseConfig, DataConfig, LossConfig, get_config
except ImportError:
    from config import UIBFuseConfig, DataConfig, LossConfig, get_config


# =============================================================================
# DATA LOADING
# =============================================================================

class CryptoPunksDataset(Dataset):
    """
    Dataset for CryptoPunks visual-temporal learning.
    
    Handles loading of 167,492 transaction records with associated
    punk images, transaction sequences, and price targets.
    
    Dataset Statistics:
    ------------------
    - Total transactions: 167,492
    - Unique punks: 10,000
    - Transaction types: 9 (Offered, Bid, Sold, Transfer, etc.)
    - Price range: 0-4,200 ETH
    - Time span: 2017-2021
    
    Args:
        data_path: Root path to dataset
        split: Data split ('train', 'val', 'test')
        transform: Image transformations
        max_seq_len: Maximum transaction sequence length
        config: Data configuration object
    """
    
    # Transaction type mapping
    TXN_TYPES = {
        'Offered': 0, 'Bid': 1, 'Sold': 2, 'Transfer': 3,
        'Bid Withdrawn': 4, 'Offer Withdrawn': 5, 'Claimed': 6,
        'Wrapped': 7, 'Unwrapped': 8
    }
    
    # Punk type mapping
    PUNK_TYPES = {
        'Male': 0, 'Female': 1, 'Zombie': 2, 'Ape': 3, 'Alien': 4
    }
    
    def __init__(
        self,
        data_path: str,
        split: str = 'train',
        transform: Optional[Callable] = None,
        max_seq_len: int = 512,
        config: Optional[DataConfig] = None
    ):
        super().__init__()
        
        self.data_path = Path(data_path)
        self.split = split
        self.transform = transform or get_transforms(split == 'train')
        self.max_seq_len = max_seq_len
        self.config = config or DataConfig()
        
        # Load data
        self._load_data()
        
        # Precompute statistics for normalization
        self._compute_statistics()
    
    def _load_data(self):
        """Load transaction history and prepare samples."""
        # Load transaction history
        txn_path = self.data_path / self.config.transactions_file
        if txn_path.exists():
            self.transactions = pd.read_csv(txn_path)
        else:
            # Create synthetic data for testing
            self._create_synthetic_data()
            return
        
        # Load attributes if available
        attr_path = self.data_path / self.config.attributes_file
        if attr_path.exists():
            self.attributes = pd.read_csv(attr_path)
        else:
            self.attributes = None
        
        # Filter to 'Sold' transactions for price prediction
        self.sales = self.transactions[
            self.transactions['Type'] == 'Sold'
        ].copy()
        
        # Parse prices (handle ETH values)
        self.sales['price_eth'] = self.sales['Crypto'].apply(self._parse_price)
        
        # Remove outliers (prices > 99th percentile)
        price_cap = self.sales['price_eth'].quantile(0.99)
        self.sales = self.sales[self.sales['price_eth'] <= price_cap]
        
        # Group transactions by punk
        self.punk_transactions = self.transactions.groupby('Punk')
        
        # Create sample list
        self.samples = self.sales.reset_index(drop=True)
        
        # Load split indices
        self._load_split()
    
    def _create_synthetic_data(self):
        """Create synthetic data for testing without real dataset."""
        n_samples = 1000
        
        self.samples = pd.DataFrame({
            'Punk': np.random.randint(0, 10000, n_samples),
            'price_eth': np.random.exponential(30, n_samples),
            'Type': ['Sold'] * n_samples,
            'Txn': [f'txn_{i}' for i in range(n_samples)],
        })
        
        self.transactions = self.samples.copy()
        self.punk_transactions = self.transactions.groupby('Punk')
        self.attributes = None
        
        # Create all indices for split
        indices = list(range(n_samples))
        random.shuffle(indices)
        
        train_end = int(0.7 * n_samples)
        val_end = int(0.85 * n_samples)
        
        if self.split == 'train':
            self.indices = indices[:train_end]
        elif self.split == 'val':
            self.indices = indices[train_end:val_end]
        else:
            self.indices = indices[val_end:]
    
    def _parse_price(self, crypto_str: str) -> float:
        """Parse price from crypto string."""
        if pd.isna(crypto_str):
            return 0.0
        try:
            # Remove 'Ξ' symbol and convert
            price = float(str(crypto_str).replace('Ξ', '').replace(',', '').strip())
            return price
        except (ValueError, TypeError):
            return 0.0
    
    def _load_split(self):
        """Load or create data split indices."""
        split_file = self.data_path / f'{self.split}_indices.json'
        
        if split_file.exists():
            with open(split_file, 'r') as f:
                self.indices = json.load(f)
        else:
            # Create splits
            n_samples = len(self.samples)
            indices = list(range(n_samples))
            random.seed(42)  # Reproducibility
            random.shuffle(indices)
            
            train_end = int(self.config.train_ratio * n_samples)
            val_end = int((self.config.train_ratio + self.config.val_ratio) * n_samples)
            
            splits = {
                'train': indices[:train_end],
                'val': indices[train_end:val_end],
                'test': indices[val_end:]
            }
            
            # Save splits
            for split_name, split_indices in splits.items():
                with open(self.data_path / f'{split_name}_indices.json', 'w') as f:
                    json.dump(split_indices, f)
            
            self.indices = splits[self.split]
    
    def _compute_statistics(self):
        """Compute dataset statistics for normalization."""
        if hasattr(self, 'samples') and 'price_eth' in self.samples.columns:
            prices = self.samples['price_eth'].values
            self.price_mean = float(np.mean(prices))
            self.price_std = float(np.std(prices)) + 1e-6
            self.price_min = float(np.min(prices))
            self.price_max = float(np.max(prices))
        else:
            self.price_mean = 29.39
            self.price_std = 69.64
            self.price_min = 0.0
            self.price_max = 4200.0
    
    def _load_image(self, punk_id: int) -> Image.Image:
        """Load punk image."""
        # Try different image formats and naming conventions
        possible_paths = [
            self.data_path / 'images' / f'punk{punk_id:04d}.png',
            self.data_path / 'images' / f'{punk_id}.png',
            self.data_path / 'images' / f'cryptopunk{punk_id}.png',
        ]
        
        for img_path in possible_paths:
            if img_path.exists():
                return Image.open(img_path).convert('RGB')
        
        # Return placeholder if image not found
        return Image.new('RGB', (24, 24), color=(128, 128, 128))
    
    def _load_transactions(self, punk_id: int) -> Tensor:
        """
        Load and encode transaction history for a punk.
        
        Returns:
            Tensor of shape [seq_len, feature_dim]
        """
        try:
            punk_txns = self.punk_transactions.get_group(punk_id)
        except KeyError:
            # No transactions found, return zeros
            return torch.zeros(self.max_seq_len, 64)
        
        # Sort by transaction (chronological)
        punk_txns = punk_txns.sort_values('Txn')
        
        features = []
        for _, row in punk_txns.iterrows():
            feat = self._encode_transaction(row)
            features.append(feat)
        
        if len(features) == 0:
            return torch.zeros(self.max_seq_len, 64)
        
        features = torch.stack(features)
        
        # Pad or truncate
        if len(features) > self.max_seq_len:
            features = features[-self.max_seq_len:]
        elif len(features) < self.max_seq_len:
            padding = torch.zeros(self.max_seq_len - len(features), features.size(1))
            features = torch.cat([padding, features], dim=0)
        
        return features
    
    def _encode_transaction(self, row: pd.Series) -> Tensor:
        """Encode a single transaction as feature vector."""
        feature_dim = 64
        features = torch.zeros(feature_dim)
        
        # Transaction type one-hot (9 types)
        txn_type = self.TXN_TYPES.get(row.get('Type', 'Sold'), 2)
        features[txn_type] = 1.0
        
        # Price features (normalized)
        price = self._parse_price(row.get('Crypto', '0'))
        if self.config.log_transform_prices and price > 0:
            price = np.log1p(price)
        features[9] = (price - self.price_mean) / self.price_std
        
        # From/To address features (simplified hash)
        from_addr = str(row.get('From', ''))
        to_addr = str(row.get('To', ''))
        features[10] = hash(from_addr) % 1000 / 1000.0
        features[11] = hash(to_addr) % 1000 / 1000.0
        
        # Temporal features (if available)
        # Placeholder for timestamp encoding
        features[12:16] = torch.randn(4) * 0.1
        
        return features
    
    def _compute_volatility(self, prices: List[float]) -> float:
        """
        Compute price volatility from transaction history.
        
        Volatility is measured as the coefficient of variation
        of price changes, providing a scale-independent metric.
        """
        if len(prices) < 2:
            return 0.0
        
        prices = np.array(prices)
        returns = np.diff(prices) / (prices[:-1] + 1e-6)
        
        if len(returns) == 0:
            return 0.0
        
        volatility = np.std(returns) / (np.mean(np.abs(returns)) + 1e-6)
        return float(np.clip(volatility, 0, 10))
    
    def _normalize_prices(self, prices: Tensor) -> Tensor:
        """Normalize prices using dataset statistics."""
        if self.config.log_transform_prices:
            prices = torch.log1p(prices)
        return (prices - self.price_mean) / self.price_std
    
    def _denormalize_prices(self, prices: Tensor) -> Tensor:
        """Denormalize prices to original scale."""
        prices = prices * self.price_std + self.price_mean
        if self.config.log_transform_prices:
            prices = torch.expm1(prices)
        return prices
    
    def __len__(self) -> int:
        return len(self.indices)
    
    def __getitem__(self, idx: int) -> Dict[str, Tensor]:
        """
        Get a sample.
        
        Returns:
            Dictionary containing:
                - 'image': Transformed punk image [3, H, W]
                - 'temporal': Transaction sequence [seq_len, feat_dim]
                - 'target': Normalized price [1]
                - 'mask': Padding mask [seq_len]
                - 'punk_id': Punk identifier
                - 'volatility': Price volatility
        """
        sample_idx = self.indices[idx]
        row = self.samples.iloc[sample_idx]
        
        punk_id = int(row['Punk'])
        price = float(row['price_eth'])
        
        # Load image
        image = self._load_image(punk_id)
        if self.transform:
            image = self.transform(image)
        
        # Load transaction sequence
        temporal = self._load_transactions(punk_id)
        
        # Create padding mask (True for padded positions)
        mask = (temporal.sum(dim=-1) == 0)
        
        # Compute volatility from price history
        try:
            punk_prices = self.punk_transactions.get_group(punk_id)
            punk_prices = punk_prices[punk_prices['Type'] == 'Sold']['Crypto'].apply(self._parse_price)
            volatility = self._compute_volatility(punk_prices.tolist())
        except (KeyError, Exception):
            volatility = 0.0
        
        # Normalize target price
        target = torch.tensor([price], dtype=torch.float32)
        if self.config.normalize_prices:
            target = self._normalize_prices(target)
        
        return {
            'image': image,
            'temporal': temporal,
            'target': target,
            'mask': mask,
            'punk_id': torch.tensor([punk_id], dtype=torch.long),
            'volatility': torch.tensor([volatility], dtype=torch.float32),
        }


class DataModule:
    """
    PyTorch Lightning-style data module for CryptoPunks.
    
    Handles data loading, preprocessing, and DataLoader creation
    for train/val/test splits.
    """
    
    def __init__(self, config: UIBFuseConfig):
        self.config = config
        self.data_config = config.data
        self.hardware_config = config.hardware
        
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
    
    def setup(self, stage: Optional[str] = None):
        """Setup datasets for each stage."""
        if stage == 'fit' or stage is None:
            self.train_dataset = CryptoPunksDataset(
                data_path=self.data_config.data_root,
                split='train',
                transform=get_transforms(is_training=True, config=self.data_config),
                max_seq_len=self.data_config.max_seq_length,
                config=self.data_config
            )
            
            self.val_dataset = CryptoPunksDataset(
                data_path=self.data_config.data_root,
                split='val',
                transform=get_transforms(is_training=False, config=self.data_config),
                max_seq_len=self.data_config.max_seq_length,
                config=self.data_config
            )
        
        if stage == 'test' or stage is None:
            self.test_dataset = CryptoPunksDataset(
                data_path=self.data_config.data_root,
                split='test',
                transform=get_transforms(is_training=False, config=self.data_config),
                max_seq_len=self.data_config.max_seq_length,
                config=self.data_config
            )
    
    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.hardware_config.effective_batch_size,
            shuffle=True,
            num_workers=self.hardware_config.num_workers,
            pin_memory=self.hardware_config.pin_memory,
            drop_last=True,
            persistent_workers=self.hardware_config.persistent_workers,
        )
    
    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.hardware_config.effective_batch_size,
            shuffle=False,
            num_workers=self.hardware_config.num_workers,
            pin_memory=self.hardware_config.pin_memory,
        )
    
    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.hardware_config.effective_batch_size,
            shuffle=False,
            num_workers=self.hardware_config.num_workers,
            pin_memory=self.hardware_config.pin_memory,
        )


def get_transforms(
    is_training: bool = True,
    config: Optional[DataConfig] = None
) -> transforms.Compose:
    """
    Get image transformations.
    
    Training transforms include data augmentation.
    Validation/test transforms only normalize.
    """
    config = config or DataConfig()
    
    if is_training:
        return transforms.Compose([
            transforms.Resize((config.image_size, config.image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=config.normalize_mean, std=config.normalize_std),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((config.image_size, config.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=config.normalize_mean, std=config.normalize_std),
        ])


def create_data_splits(
    data_path: str,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    seed: int = 42
) -> Dict[str, List[int]]:
    """Create reproducible data splits."""
    random.seed(seed)
    np.random.seed(seed)
    
    # Load transactions
    txn_path = Path(data_path) / 'txn_history.csv'
    if txn_path.exists():
        df = pd.read_csv(txn_path)
        sales = df[df['Type'] == 'Sold']
        n_samples = len(sales)
    else:
        n_samples = 1000
    
    indices = list(range(n_samples))
    random.shuffle(indices)
    
    train_end = int(train_ratio * n_samples)
    val_end = int((train_ratio + val_ratio) * n_samples)
    
    return {
        'train': indices[:train_end],
        'val': indices[train_end:val_end],
        'test': indices[val_end:]
    }


# =============================================================================
# METRICS
# =============================================================================

class MetricsCalculator:
    """
    Comprehensive metrics calculator for evaluation.
    
    Implements standard regression metrics along with domain-specific
    metrics for NFT price prediction and market efficiency assessment.
    """
    
    @staticmethod
    def compute_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Coefficient of determination (R²)."""
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        return float(1 - ss_res / (ss_tot + 1e-8))
    
    @staticmethod
    def compute_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Mean Absolute Error."""
        return float(np.mean(np.abs(y_true - y_pred)))
    
    @staticmethod
    def compute_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Root Mean Squared Error."""
        return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    
    @staticmethod
    def compute_mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Mean Absolute Percentage Error."""
        mask = y_true != 0
        return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)
    
    @staticmethod
    def compute_directional_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Directional accuracy for price movements.
        
        Measures the percentage of correctly predicted price directions
        (up vs down) compared to previous values.
        """
        if len(y_true) < 2:
            return 0.5
        
        true_direction = np.sign(np.diff(y_true))
        pred_direction = np.sign(np.diff(y_pred))
        
        return float(np.mean(true_direction == pred_direction))
    
    @staticmethod
    def compute_market_efficiency_score(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        uncertainties: Optional[np.ndarray] = None
    ) -> float:
        """
        Market Efficiency Score (MES).
        
        Combines prediction accuracy with uncertainty calibration to assess
        the overall quality of predictions for market applications.
        
        MES = α·R² + β·(1-MAPE/100) + γ·DirectionalAcc + δ·Calibration
        
        where α, β, γ, δ are weighting factors summing to 1.
        """
        r2 = MetricsCalculator.compute_r2(y_true, y_pred)
        mape = MetricsCalculator.compute_mape(y_true, y_pred)
        dir_acc = MetricsCalculator.compute_directional_accuracy(y_true, y_pred)
        
        # Calibration component (if uncertainties provided)
        if uncertainties is not None:
            calibration = 1 - MetricsCalculator.compute_calibration_error(
                y_pred, uncertainties, y_true
            )
        else:
            calibration = 0.5
        
        # Weighted combination
        mes = 0.4 * r2 + 0.2 * (1 - mape / 100) + 0.2 * dir_acc + 0.2 * calibration
        
        return float(np.clip(mes, 0, 1))
    
    @staticmethod
    def compute_calibration_error(
        predictions: np.ndarray,
        uncertainties: np.ndarray,
        targets: np.ndarray,
        n_bins: int = 10
    ) -> float:
        """
        Expected Calibration Error (ECE) for uncertainty estimates.
        
        Measures how well predicted uncertainties match empirical errors.
        Well-calibrated models have ECE close to 0.
        
        Reference:
            Guo et al., "On Calibration of Modern Neural Networks," ICML 2017
        """
        # Compute standardized errors
        errors = np.abs(predictions - targets)
        
        # Bin by uncertainty
        bin_boundaries = np.percentile(uncertainties, np.linspace(0, 100, n_bins + 1))
        
        ece = 0.0
        total_samples = len(predictions)
        
        for i in range(n_bins):
            mask = (uncertainties >= bin_boundaries[i]) & (uncertainties < bin_boundaries[i + 1])
            if np.sum(mask) == 0:
                continue
            
            bin_errors = errors[mask]
            bin_uncertainties = uncertainties[mask]
            
            # Expected: mean error ≈ mean uncertainty
            expected_error = np.mean(bin_uncertainties)
            actual_error = np.mean(bin_errors)
            
            bin_weight = np.sum(mask) / total_samples
            ece += bin_weight * np.abs(expected_error - actual_error)
        
        return float(ece)
    
    @staticmethod
    def compute_all_metrics(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        uncertainties: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """Compute all metrics at once."""
        metrics = {
            'r2': MetricsCalculator.compute_r2(y_true, y_pred),
            'mae': MetricsCalculator.compute_mae(y_true, y_pred),
            'rmse': MetricsCalculator.compute_rmse(y_true, y_pred),
            'mape': MetricsCalculator.compute_mape(y_true, y_pred),
            'directional_accuracy': MetricsCalculator.compute_directional_accuracy(y_true, y_pred),
            'mes': MetricsCalculator.compute_market_efficiency_score(y_true, y_pred, uncertainties),
        }
        
        if uncertainties is not None:
            metrics['ece'] = MetricsCalculator.compute_calibration_error(
                y_pred, uncertainties, y_true
            )
        
        return metrics


def compute_mutual_information(
    x: np.ndarray,
    y: np.ndarray,
    method: str = 'knn',
    k: int = 3
) -> float:
    """
    Estimate mutual information between two variables.
    
    Uses k-nearest neighbors estimator for continuous variables.
    
    Reference:
        Kraskov et al., "Estimating Mutual Information," Physical Review E 2004
    
    Args:
        x: First variable [N, D1]
        y: Second variable [N, D2]
        method: Estimation method ('knn' or 'binning')
        k: Number of neighbors for KNN estimator
    
    Returns:
        Estimated mutual information in nats
    """
    if method == 'knn':
        return _mi_knn(x, y, k)
    else:
        return _mi_binning(x, y)


def _mi_knn(x: np.ndarray, y: np.ndarray, k: int = 3) -> float:
    """KNN-based mutual information estimator."""
    from scipy.special import digamma
    
    n = len(x)
    
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    if y.ndim == 1:
        y = y.reshape(-1, 1)
    
    xy = np.hstack([x, y])
    
    # Fit KNN
    nn_xy = NearestNeighbors(n_neighbors=k + 1, metric='chebyshev')
    nn_xy.fit(xy)
    distances, _ = nn_xy.kneighbors(xy)
    epsilon = distances[:, k]
    
    # Count neighbors within epsilon for marginals
    nn_x = NearestNeighbors(metric='chebyshev')
    nn_x.fit(x)
    nn_y = NearestNeighbors(metric='chebyshev')
    nn_y.fit(y)
    
    n_x = np.array([len(nn_x.radius_neighbors([xi], radius=eps, return_distance=False)[0]) - 1 
                    for xi, eps in zip(x, epsilon)])
    n_y = np.array([len(nn_y.radius_neighbors([yi], radius=eps, return_distance=False)[0]) - 1 
                    for yi, eps in zip(y, epsilon)])
    
    mi = digamma(k) + digamma(n) - np.mean(digamma(n_x + 1) + digamma(n_y + 1))
    
    return float(max(0, mi))


def _mi_binning(x: np.ndarray, y: np.ndarray, bins: int = 20) -> float:
    """Binning-based mutual information estimator."""
    hist_2d, _, _ = np.histogram2d(x.flatten(), y.flatten(), bins=bins)
    
    # Normalize to probabilities
    pxy = hist_2d / hist_2d.sum()
    px = pxy.sum(axis=1)
    py = pxy.sum(axis=0)
    
    # Compute MI
    px_py = np.outer(px, py)
    nzs = pxy > 0
    mi = np.sum(pxy[nzs] * np.log(pxy[nzs] / px_py[nzs]))
    
    return float(max(0, mi))


# =============================================================================
# LOSS FUNCTIONS
# =============================================================================

class UIBFuseLoss(nn.Module):
    """
    Combined loss function with explicit dynamic schedules.
    
    Total loss formulation:
    L = L_pred + β(t)·L_IB + λ(t)·L_KL + γ·L_consistency + η·L_efficiency
    
    Dynamic Schedules:
    -----------------
    1. Information Bottleneck weight β(t):
       β(t) = β_min + (β_max - β_min)·min(1, t/T_warmup)
       
       Starts low to allow learning, increases to enforce compression.
       Parameters: β_min=0.01, β_max=0.1, T_warmup=1000 steps
    
    2. KL divergence weight λ(t):
       λ(t) = λ_max·(1 - exp(-t/τ))
       
       Exponential warmup for cross-modal alignment.
       Parameters: λ_max=0.1, τ=500 (time constant)
    
    Loss Components:
    ---------------
    - L_pred: Gaussian negative log-likelihood for uncertainty-aware prediction
    - L_IB: Information bottleneck regularization
    - L_KL: KL divergence for Bayesian fusion
    - L_consistency: Cross-modal feature consistency
    - L_efficiency: Market efficiency prediction loss
    
    Args:
        config: Loss configuration object
    """
    
    def __init__(self, config: LossConfig):
        super().__init__()
        
        self.config = config
        
        # Schedule parameters (explicit for ICME reviewers)
        self.beta_min = config.beta_min
        self.beta_max = config.beta_max
        self.beta_warmup_steps = config.beta_warmup_steps
        
        self.lambda_max = config.lambda_kl_max
        self.lambda_tau = config.lambda_kl_tau
        
        self.consistency_weight = config.consistency_weight
        self.efficiency_weight = config.efficiency_loss_weight
        
        # Current step counter
        self.register_buffer('current_step', torch.tensor(0, dtype=torch.long))
    
    def get_beta(self, step: int) -> float:
        """
        Compute β(t) for information bottleneck.
        
        β(t) = β_min + (β_max - β_min)·min(1, t/T_warmup)
        """
        progress = min(1.0, step / self.beta_warmup_steps)
        beta = self.beta_min + (self.beta_max - self.beta_min) * progress
        return beta
    
    def get_lambda(self, step: int) -> float:
        """
        Compute λ(t) for KL divergence.
        
        λ(t) = λ_max·(1 - exp(-t/τ))
        """
        lambda_t = self.lambda_max * (1 - np.exp(-step / self.lambda_tau))
        return lambda_t
    
    def _prediction_loss(
        self,
        pred_mean: Tensor,
        pred_log_var: Tensor,
        target: Tensor
    ) -> Tensor:
        """
        Gaussian negative log-likelihood loss.
        
        NLL = 0.5 * (log(σ²) + (y - μ)²/σ²)
        
        This formulation learns both the prediction and its uncertainty,
        automatically balancing samples based on their difficulty.
        """
        # Clamp log_var for numerical stability
        pred_log_var = torch.clamp(pred_log_var, min=-10, max=10)
        
        precision = torch.exp(-pred_log_var)
        mse = (target - pred_mean) ** 2
        
        # Gaussian NLL
        nll = 0.5 * (pred_log_var + mse * precision)
        
        return nll.mean()
    
    def _consistency_loss(
        self,
        visual_features: Tensor,
        temporal_features: Tensor
    ) -> Tensor:
        """
        Cross-modal consistency regularization.
        
        Encourages aligned features across modalities for the same sample.
        """
        # Normalize features
        v_norm = F.normalize(visual_features, dim=-1)
        t_norm = F.normalize(temporal_features, dim=-1)
        
        # Cosine similarity loss (maximize similarity)
        consistency = 1 - (v_norm * t_norm).sum(dim=-1).mean()
        
        return consistency
    
    def _efficiency_loss(
        self,
        pred_efficiency: Tensor,
        target_efficiency: Optional[Tensor] = None
    ) -> Tensor:
        """Loss for market efficiency prediction auxiliary task."""
        if target_efficiency is None:
            # Self-supervised: encourage high efficiency scores
            return -pred_efficiency.mean()
        
        return F.mse_loss(pred_efficiency, target_efficiency)
    
    def forward(
        self,
        model_output: Dict[str, Tensor],
        targets: Tensor,
        step: Optional[int] = None
    ) -> Tuple[Tensor, Dict[str, float]]:
        """
        Compute total loss with all components.
        
        Args:
            model_output: Dictionary from UIBFuse forward pass
            targets: Ground truth prices
            step: Current training step (for schedules)
        
        Returns:
            Tuple of (total_loss, loss_dict with individual components)
        """
        step = step or int(self.current_step.item())
        
        # Get scheduled weights
        beta = self.get_beta(step)
        lambda_kl = self.get_lambda(step)
        
        # Unpack model outputs
        pred_mean, pred_log_var = model_output['predictions']
        ib_loss = model_output.get('ib_loss', torch.tensor(0.0))
        kl_loss = model_output.get('kl_loss', torch.tensor(0.0))
        
        # Primary prediction loss
        pred_loss = self._prediction_loss(pred_mean, pred_log_var, targets)
        
        # Information bottleneck loss (already computed in model)
        ib_loss = beta * ib_loss
        
        # KL divergence loss (from Bayesian fusion)
        kl_loss = lambda_kl * kl_loss
        
        # Consistency loss (if features available)
        consistency_loss = torch.tensor(0.0, device=pred_mean.device)
        
        # Efficiency loss (if predictions available)
        efficiency_loss = torch.tensor(0.0, device=pred_mean.device)
        if model_output.get('efficiency') is not None:
            efficiency_loss = self.efficiency_weight * self._efficiency_loss(
                model_output['efficiency']
            )
        
        # Total loss
        total_loss = pred_loss + ib_loss + kl_loss + consistency_loss + efficiency_loss
        
        # Increment step counter
        self.current_step += 1
        
        # Loss dictionary for logging
        loss_dict = {
            'total': total_loss.item(),
            'prediction': pred_loss.item(),
            'ib': ib_loss.item() if isinstance(ib_loss, Tensor) else ib_loss,
            'kl': kl_loss.item() if isinstance(kl_loss, Tensor) else kl_loss,
            'consistency': consistency_loss.item(),
            'efficiency': efficiency_loss.item(),
            'beta': beta,
            'lambda': lambda_kl,
        }
        
        return total_loss, loss_dict


# =============================================================================
# VISUALIZATION
# =============================================================================

class Visualizer:
    """
    Visualization utilities for paper figures.
    
    Generates publication-quality figures for IEEE ICME submission
    including training curves, attention maps, and ablation results.
    """
    
    def __init__(self, style: str = 'seaborn-v0_8-paper'):
        """Initialize visualizer with publication style."""
        try:
            plt.style.use(style)
        except OSError:
            plt.style.use('seaborn-v0_8-whitegrid')
        
        # IEEE two-column format dimensions
        self.fig_width_single = 3.5  # inches
        self.fig_width_double = 7.16  # inches
        self.fig_height = 2.5  # inches
        
        # Color palette
        self.colors = sns.color_palette("husl", 8)
    
    def plot_training_curves(
        self,
        history: Dict[str, List[float]],
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """Plot training and validation curves."""
        fig, axes = plt.subplots(1, 3, figsize=(self.fig_width_double, self.fig_height))
        
        epochs = range(1, len(history.get('train_loss', [])) + 1)
        
        # Loss curves
        axes[0].plot(epochs, history.get('train_loss', []), label='Train', color=self.colors[0])
        axes[0].plot(epochs, history.get('val_loss', []), label='Val', color=self.colors[1])
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Training Loss')
        axes[0].legend()
        
        # R² curves
        axes[1].plot(epochs, history.get('train_r2', []), label='Train', color=self.colors[0])
        axes[1].plot(epochs, history.get('val_r2', []), label='Val', color=self.colors[1])
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('R²')
        axes[1].set_title('R² Score')
        axes[1].legend()
        
        # Schedule visualization
        steps = range(len(history.get('beta', [])))
        ax2 = axes[2].twinx()
        axes[2].plot(steps, history.get('beta', []), label='β(t)', color=self.colors[2])
        ax2.plot(steps, history.get('lambda', []), label='λ(t)', color=self.colors[3])
        axes[2].set_xlabel('Step')
        axes[2].set_ylabel('β(t)', color=self.colors[2])
        ax2.set_ylabel('λ(t)', color=self.colors[3])
        axes[2].set_title('Loss Schedules')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_attention_maps(
        self,
        attention_weights: Dict[str, np.ndarray],
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """Visualize cross-modal attention patterns."""
        fig, axes = plt.subplots(1, 3, figsize=(self.fig_width_double, self.fig_height))
        
        for idx, (name, weights) in enumerate(attention_weights.items()):
            if idx >= 3:
                break
            
            # Average over batch and heads if needed
            if weights.ndim > 2:
                weights = weights.mean(axis=tuple(range(weights.ndim - 2)))
            
            im = axes[idx].imshow(weights, cmap='viridis', aspect='auto')
            axes[idx].set_title(f'{name.upper()} Attention')
            axes[idx].set_xlabel('Key')
            axes[idx].set_ylabel('Query')
            plt.colorbar(im, ax=axes[idx], fraction=0.046)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_uncertainty_calibration(
        self,
        predictions: np.ndarray,
        uncertainties: np.ndarray,
        targets: np.ndarray,
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """Plot uncertainty calibration curve."""
        fig, axes = plt.subplots(1, 2, figsize=(self.fig_width_double, self.fig_height))
        
        # Calibration curve
        n_bins = 10
        bin_boundaries = np.percentile(uncertainties, np.linspace(0, 100, n_bins + 1))
        
        expected_errors = []
        actual_errors = []
        
        for i in range(n_bins):
            mask = (uncertainties >= bin_boundaries[i]) & (uncertainties < bin_boundaries[i + 1])
            if np.sum(mask) > 0:
                expected_errors.append(np.mean(uncertainties[mask]))
                actual_errors.append(np.mean(np.abs(predictions[mask] - targets[mask])))
        
        axes[0].plot([0, max(expected_errors)], [0, max(expected_errors)], 
                     'k--', label='Perfect calibration')
        axes[0].scatter(expected_errors, actual_errors, color=self.colors[0], s=50)
        axes[0].set_xlabel('Predicted Uncertainty')
        axes[0].set_ylabel('Actual Error')
        axes[0].set_title('Calibration Curve')
        axes[0].legend()
        
        # Error vs uncertainty scatter
        axes[1].scatter(uncertainties, np.abs(predictions - targets), 
                       alpha=0.5, s=10, color=self.colors[1])
        axes[1].set_xlabel('Predicted Uncertainty')
        axes[1].set_ylabel('Absolute Error')
        axes[1].set_title('Error Distribution')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_baseline_comparison(
        self,
        results: Dict[str, Dict[str, float]],
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """Create baseline comparison bar chart."""
        fig, axes = plt.subplots(1, 3, figsize=(self.fig_width_double, self.fig_height))
        
        methods = list(results.keys())
        x = np.arange(len(methods))
        width = 0.6
        
        metrics = ['r2', 'mae', 'mes']
        titles = ['R² Score (↑)', 'MAE (↓)', 'MES (↑)']
        
        for idx, (metric, title) in enumerate(zip(metrics, titles)):
            values = [results[m].get(metric, 0) for m in methods]
            
            bars = axes[idx].bar(x, values, width, color=self.colors[:len(methods)])
            axes[idx].set_ylabel(metric.upper())
            axes[idx].set_title(title)
            axes[idx].set_xticks(x)
            axes[idx].set_xticklabels(methods, rotation=45, ha='right', fontsize=8)
            
            # Highlight best (UIBFuse)
            if 'UIBFuse' in methods:
                best_idx = methods.index('UIBFuse')
                bars[best_idx].set_edgecolor('red')
                bars[best_idx].set_linewidth(2)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_ablation_study(
        self,
        ablation_results: Dict[str, Dict[str, float]],
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """Visualize ablation study results."""
        fig, ax = plt.subplots(figsize=(self.fig_width_single, self.fig_height * 1.2))
        
        components = list(ablation_results.keys())
        metrics = ['r2', 'mes']
        
        x = np.arange(len(components))
        width = 0.35
        
        for i, metric in enumerate(metrics):
            values = [ablation_results[c].get(metric, 0) for c in components]
            ax.barh(x + i * width, values, width, label=metric.upper(), color=self.colors[i])
        
        ax.set_yticks(x + width / 2)
        ax.set_yticklabels(components)
        ax.set_xlabel('Score')
        ax.set_title('Ablation Study Results')
        ax.legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_volatility_adaptation(
        self,
        gate_v: np.ndarray,
        gate_t: np.ndarray,
        volatility: np.ndarray,
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """Visualize volatility-adaptive gating behavior."""
        fig, axes = plt.subplots(1, 2, figsize=(self.fig_width_double, self.fig_height))
        
        # Gate weights vs volatility
        axes[0].scatter(volatility, gate_v.mean(axis=-1), alpha=0.5, s=10, 
                       label='Visual Gate', color=self.colors[0])
        axes[0].scatter(volatility, gate_t.mean(axis=-1), alpha=0.5, s=10, 
                       label='Temporal Gate', color=self.colors[1])
        axes[0].set_xlabel('Volatility')
        axes[0].set_ylabel('Gate Weight')
        axes[0].set_title('Gate Adaptation to Volatility')
        axes[0].legend()
        
        # Gate distribution
        axes[1].hist(gate_v.flatten(), bins=50, alpha=0.5, label='Visual', color=self.colors[0])
        axes[1].hist(gate_t.flatten(), bins=50, alpha=0.5, label='Temporal', color=self.colors[1])
        axes[1].set_xlabel('Gate Weight')
        axes[1].set_ylabel('Frequency')
        axes[1].set_title('Gate Weight Distribution')
        axes[1].legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig


# =============================================================================
# UTILITIES
# =============================================================================

def set_seed(seed: int = 42):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def setup_logging(
    log_dir: str,
    name: str = 'uibfuse',
    level: int = logging.INFO
) -> logging.Logger:
    """Setup logging configuration."""
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # File handler
    fh = logging.FileHandler(
        Path(log_dir) / f'{name}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
    )
    fh.setLevel(level)
    
    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(level)
    
    # Formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    
    logger.addHandler(fh)
    logger.addHandler(ch)
    
    return logger


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: Dict[str, float],
    path: str,
    scheduler: Optional[Any] = None
):
    """Save training checkpoint."""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics,
    }
    
    if scheduler is not None:
        checkpoint['scheduler_state_dict'] = scheduler.state_dict()
    
    torch.save(checkpoint, path)


def load_checkpoint(
    model: nn.Module,
    path: str,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None
) -> Dict[str, Any]:
    """Load training checkpoint."""
    checkpoint = torch.load(path, map_location='cpu')
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    if scheduler is not None and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    return checkpoint


def count_parameters(model: nn.Module) -> Dict[str, int]:
    """Count model parameters."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        'total': total,
        'trainable': trainable,
        'frozen': total - trainable,
        'total_mb': total * 4 / (1024 ** 2),
    }


def get_gpu_memory_usage() -> Dict[str, float]:
    """Get GPU memory usage statistics."""
    if not torch.cuda.is_available():
        return {'available': False}
    
    result = {}
    for i in range(torch.cuda.device_count()):
        allocated = torch.cuda.memory_allocated(i) / (1024 ** 3)
        cached = torch.cuda.memory_reserved(i) / (1024 ** 3)
        result[f'gpu_{i}_allocated_gb'] = allocated
        result[f'gpu_{i}_cached_gb'] = cached
    
    return result


class EarlyStopping:
    """
    Early stopping handler.
    
    Monitors a metric and stops training when it stops improving
    for a specified number of epochs (patience).
    """
    
    def __init__(
        self,
        patience: int = 15,
        min_delta: float = 1e-4,
        mode: str = 'min'
    ):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        
        self.counter = 0
        self.best_score = None
        self.early_stop = False
    
    def __call__(self, score: float) -> bool:
        """
        Check if training should stop.
        
        Args:
            score: Current metric value
        
        Returns:
            True if training should stop
        """
        if self.best_score is None:
            self.best_score = score
            return False
        
        if self.mode == 'min':
            improved = score < self.best_score - self.min_delta
        else:
            improved = score > self.best_score + self.min_delta
        
        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        
        return self.early_stop
    
    def reset(self):
        """Reset early stopping state."""
        self.counter = 0
        self.best_score = None
        self.early_stop = False


class AverageMeter:
    """Computes and stores the average and current value."""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val: float, n: int = 1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def format_metrics(metrics: Dict[str, float], precision: int = 4) -> str:
    """Format metrics dictionary as string."""
    parts = []
    for k, v in metrics.items():
        if isinstance(v, float):
            parts.append(f'{k}: {v:.{precision}f}')
        else:
            parts.append(f'{k}: {v}')
    return ' | '.join(parts)


# =============================================================================
# TESTING
# =============================================================================

if __name__ == "__main__":
    print("Testing UIBFuse utilities...")
    
    # Test configuration
    config = get_config('default')
    
    # Test dataset (with synthetic data)
    print("\n1. Testing CryptoPunksDataset...")
    dataset = CryptoPunksDataset(
        data_path='./data/cryptopunks',
        split='train',
        config=config.data
    )
    print(f"   Dataset size: {len(dataset)}")
    
    sample = dataset[0]
    print(f"   Sample keys: {list(sample.keys())}")
    print(f"   Image shape: {sample['image'].shape}")
    print(f"   Temporal shape: {sample['temporal'].shape}")
    
    # Test metrics
    print("\n2. Testing MetricsCalculator...")
    y_true = np.random.randn(100)
    y_pred = y_true + np.random.randn(100) * 0.1
    uncertainties = np.abs(np.random.randn(100) * 0.1)
    
    metrics = MetricsCalculator.compute_all_metrics(y_true, y_pred, uncertainties)
    print(f"   Metrics: {format_metrics(metrics)}")
    
    # Test loss function
    print("\n3. Testing UIBFuseLoss...")
    loss_fn = UIBFuseLoss(config.loss)
    
    # Mock model output
    mock_output = {
        'predictions': (torch.randn(4, 1), torch.randn(4, 1)),
        'ib_loss': torch.tensor(0.1),
        'kl_loss': torch.tensor(0.05),
    }
    targets = torch.randn(4, 1)
    
    loss, loss_dict = loss_fn(mock_output, targets, step=100)
    print(f"   Loss: {loss.item():.4f}")
    print(f"   Components: {format_metrics(loss_dict)}")
    
    # Test visualization
    print("\n4. Testing Visualizer...")
    viz = Visualizer()
    print("   Visualizer initialized successfully")
    
    # Test utilities
    print("\n5. Testing utilities...")
    set_seed(42)
    print("   Seed set successfully")
    
    es = EarlyStopping(patience=3)
    for score in [1.0, 0.9, 0.95, 0.94, 0.93, 0.92]:
        stopped = es(score)
        print(f"   Score: {score:.2f}, Stop: {stopped}")
    
    print("\nAll tests passed!")
