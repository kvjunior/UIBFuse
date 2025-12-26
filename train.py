"""
UIBFuse Training Pipeline
=========================

Complete training pipeline for UIBFuse: Uncertainty-aware Information Bottleneck 
Fusion for Cross-Modal Visual-Temporal Learning.

This module provides:
- Multi-GPU distributed training (4x RTX 3090 support)
- Mixed precision training with gradient scaling
- Dynamic loss scheduling with β(t) and λ(t)
- Comprehensive logging and checkpointing
- Systematic ablation study framework

IEEE ICME 2026 Submission
-------------------------
Training procedure implements theoretically-grounded optimization with:
- Information bottleneck regularization warmup
- Bayesian uncertainty calibration
- Volatility-adaptive attention monitoring

Hardware Target:
---------------
- GPU: 4x NVIDIA GeForce RTX 3090 (24GB each)
- CPU: Intel Xeon Silver 4314 (64 cores @ 2.40GHz)
- RAM: 384GB DDR4

Usage:
------
    # Standard training
    python train.py --config default --gpus 4 --epochs 100
    
    # Debug mode
    python train.py --config debug --gpus 1
    
    # Ablation studies
    python train.py --ablation all --gpus 4
    
    # Resume training
    python train.py --resume checkpoints/best_model.pt

Author: Anonymous ICME Submission
"""

import argparse
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# Import project modules
try:
    from .config import UIBFuseConfig, get_config, FusionType
    from .models import UIBFuse, build_model, count_parameters
    from .utils import (
        CryptoPunksDataset, DataModule, UIBFuseLoss, MetricsCalculator,
        Visualizer, EarlyStopping, AverageMeter, set_seed, setup_logging,
        save_checkpoint, load_checkpoint, get_gpu_memory_usage, format_metrics,
        get_transforms
    )
except ImportError:
    from config import UIBFuseConfig, get_config, FusionType
    from models import UIBFuse, build_model, count_parameters
    from utils import (
        CryptoPunksDataset, DataModule, UIBFuseLoss, MetricsCalculator,
        Visualizer, EarlyStopping, AverageMeter, set_seed, setup_logging,
        save_checkpoint, load_checkpoint, get_gpu_memory_usage, format_metrics,
        get_transforms
    )


# =============================================================================
# TRAINING UTILITIES
# =============================================================================

def train_step(
    model: nn.Module,
    batch: Dict[str, torch.Tensor],
    criterion: UIBFuseLoss,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    device: torch.device,
    step: int,
    use_amp: bool = True
) -> Dict[str, float]:
    """
    Single training step with mixed precision support.
    
    Args:
        model: UIBFuse model
        batch: Batch dictionary from DataLoader
        criterion: UIBFuseLoss instance
        optimizer: Optimizer
        scaler: GradScaler for mixed precision
        device: Target device
        step: Current global step
        use_amp: Whether to use automatic mixed precision
    
    Returns:
        Dictionary with loss values
    """
    model.train()
    
    # Move batch to device
    images = batch['image'].to(device, non_blocking=True)
    temporal = batch['temporal'].to(device, non_blocking=True)
    targets = batch['target'].to(device, non_blocking=True)
    mask = batch['mask'].to(device, non_blocking=True)
    
    optimizer.zero_grad(set_to_none=True)
    
    # Forward pass with mixed precision
    with autocast(enabled=use_amp):
        outputs = model(images, temporal, mask)
        loss, loss_dict = criterion(outputs, targets, step=step)
    
    # Backward pass with gradient scaling
    scaler.scale(loss).backward()
    
    # Gradient clipping
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    
    # Optimizer step
    scaler.step(optimizer)
    scaler.update()
    
    return loss_dict


def validate_step(
    model: nn.Module,
    batch: Dict[str, torch.Tensor],
    criterion: UIBFuseLoss,
    device: torch.device
) -> Tuple[Dict[str, float], np.ndarray, np.ndarray, np.ndarray]:
    """
    Single validation step.
    
    Args:
        model: UIBFuse model
        batch: Batch dictionary
        criterion: Loss function
        device: Target device
    
    Returns:
        Tuple of (loss_dict, predictions, targets, uncertainties)
    """
    model.eval()
    
    images = batch['image'].to(device, non_blocking=True)
    temporal = batch['temporal'].to(device, non_blocking=True)
    targets = batch['target'].to(device, non_blocking=True)
    mask = batch['mask'].to(device, non_blocking=True)
    
    with torch.no_grad():
        outputs = model(images, temporal, mask)
        loss, loss_dict = criterion(outputs, targets)
    
    # Extract predictions and uncertainties
    pred_mean, pred_log_var = outputs['predictions']
    uncertainties = torch.exp(pred_log_var * 0.5)  # Standard deviation
    
    return (
        loss_dict,
        pred_mean.cpu().numpy(),
        targets.cpu().numpy(),
        uncertainties.cpu().numpy()
    )


def compute_gradient_stats(model: nn.Module) -> Dict[str, float]:
    """
    Compute gradient statistics for debugging.
    
    Monitors gradient flow through the network to detect
    vanishing/exploding gradients.
    """
    stats = {}
    
    total_norm = 0.0
    param_count = 0
    
    for name, param in model.named_parameters():
        if param.grad is not None:
            param_norm = param.grad.data.norm(2).item()
            total_norm += param_norm ** 2
            param_count += 1
            
            # Track specific modules
            if 'visual_encoder' in name:
                stats.setdefault('visual_grad_norm', 0.0)
                stats['visual_grad_norm'] += param_norm
            elif 'temporal_encoder' in name:
                stats.setdefault('temporal_grad_norm', 0.0)
                stats['temporal_grad_norm'] += param_norm
            elif 'bayesian_fusion' in name:
                stats.setdefault('fusion_grad_norm', 0.0)
                stats['fusion_grad_norm'] += param_norm
    
    stats['total_grad_norm'] = total_norm ** 0.5
    stats['avg_grad_norm'] = stats['total_grad_norm'] / max(param_count, 1)
    
    return stats


# =============================================================================
# TRAINER CLASS
# =============================================================================

class UIBFuseTrainer:
    """
    Training pipeline for UIBFuse with multi-GPU support.
    
    Implements information-theoretic training with dynamic schedules,
    designed for 4x RTX 3090 GPUs with mixed precision training.
    
    Features:
    ---------
    - Distributed Data Parallel (DDP) training
    - Mixed precision (FP16) with gradient scaling
    - Dynamic β(t) and λ(t) loss schedules
    - Comprehensive logging (TensorBoard + file)
    - Early stopping with patience
    - Best model checkpointing
    
    Args:
        config: UIBFuseConfig object
        model: UIBFuse model (or None to build from config)
        data_module: DataModule instance (or None to build from config)
    """
    
    def __init__(
        self,
        config: UIBFuseConfig,
        model: Optional[nn.Module] = None,
        data_module: Optional[DataModule] = None
    ):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Setup distributed training
        self.is_distributed = False
        self.local_rank = 0
        self.world_size = 1
        self._setup_distributed()
        
        # Build model
        if model is None:
            self.model = build_model('uibfuse', config)
        else:
            self.model = model
        
        self.model = self.model.to(self.device)
        
        # Wrap in DDP if distributed
        if self.is_distributed:
            self.model = DDP(
                self.model,
                device_ids=[self.local_rank],
                output_device=self.local_rank,
                find_unused_parameters=config.hardware.find_unused_parameters
            )
        
        # Setup data
        if data_module is None:
            self.data_module = DataModule(config)
        else:
            self.data_module = data_module
        
        # Setup training components
        self.criterion = UIBFuseLoss(config.loss)
        self.optimizer = self._setup_optimizer()
        self.scheduler = self._setup_scheduler()
        self.scaler = GradScaler(enabled=config.hardware.mixed_precision)
        
        # Early stopping
        self.early_stopping = EarlyStopping(
            patience=config.training.patience,
            min_delta=config.training.min_delta,
            mode='max'  # Maximize R²
        )
        
        # Logging
        self.logger = setup_logging(
            config.experiment.log_dir,
            name='uibfuse_train'
        )
        
        if self.local_rank == 0:
            self.writer = SummaryWriter(
                log_dir=Path(config.experiment.log_dir) / 'tensorboard'
            )
        else:
            self.writer = None
        
        # Visualization
        self.visualizer = Visualizer()
        
        # Training state
        self.global_step = 0
        self.best_metric = -float('inf')
        self.history = {
            'train_loss': [], 'val_loss': [],
            'train_r2': [], 'val_r2': [],
            'beta': [], 'lambda': []
        }
        
        # Log model info
        if self.local_rank == 0:
            params = count_parameters(self.model)
            self.logger.info(f"Model parameters: {params['total']:,}")
            self.logger.info(f"Trainable parameters: {params['trainable']:,}")
            self.logger.info(f"Model size: {params['total_mb']:.2f} MB")
    
    def _setup_distributed(self):
        """Setup distributed training for multi-GPU."""
        if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
            self.local_rank = int(os.environ.get('LOCAL_RANK', 0))
            self.world_size = int(os.environ.get('WORLD_SIZE', 1))
            
            if self.world_size > 1:
                dist.init_process_group(
                    backend=self.config.hardware.distributed_backend,
                    init_method='env://'
                )
                self.is_distributed = True
                torch.cuda.set_device(self.local_rank)
                self.device = torch.device(f'cuda:{self.local_rank}')
        
        elif torch.cuda.device_count() > 1 and self.config.hardware.num_gpus > 1:
            # Manual multi-GPU setup
            self.logger.info(f"Using DataParallel with {torch.cuda.device_count()} GPUs")
    
    def _setup_optimizer(self) -> torch.optim.Optimizer:
        """Setup optimizer with weight decay."""
        # Separate parameters for weight decay
        decay_params = []
        no_decay_params = []
        
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            
            if 'bias' in name or 'norm' in name or 'LayerNorm' in name:
                no_decay_params.append(param)
            else:
                decay_params.append(param)
        
        param_groups = [
            {'params': decay_params, 'weight_decay': self.config.training.weight_decay},
            {'params': no_decay_params, 'weight_decay': 0.0}
        ]
        
        if self.config.training.optimizer == 'adamw':
            optimizer = torch.optim.AdamW(
                param_groups,
                lr=self.config.training.learning_rate,
                betas=self.config.training.betas,
                eps=self.config.training.eps
            )
        elif self.config.training.optimizer == 'adam':
            optimizer = torch.optim.Adam(
                param_groups,
                lr=self.config.training.learning_rate,
                betas=self.config.training.betas,
                eps=self.config.training.eps
            )
        else:
            optimizer = torch.optim.SGD(
                param_groups,
                lr=self.config.training.learning_rate,
                momentum=0.9,
                weight_decay=self.config.training.weight_decay
            )
        
        return optimizer
    
    def _setup_scheduler(self) -> torch.optim.lr_scheduler._LRScheduler:
        """Setup learning rate scheduler."""
        total_steps = (
            self.config.training.num_epochs * 
            (self.config.data.num_samples // self.config.hardware.effective_batch_size)
        )
        
        if self.config.training.lr_schedule.value == 'cosine':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=total_steps,
                eta_min=self.config.training.lr_min
            )
        elif self.config.training.lr_schedule.value == 'warmup':
            # Linear warmup then cosine decay
            def lr_lambda(step):
                if step < self.config.training.lr_warmup_steps:
                    return step / self.config.training.lr_warmup_steps
                progress = (step - self.config.training.lr_warmup_steps) / (
                    total_steps - self.config.training.lr_warmup_steps
                )
                return 0.5 * (1 + np.cos(np.pi * progress))
            
            scheduler = torch.optim.lr_scheduler.LambdaLR(
                self.optimizer, lr_lambda
            )
        else:
            scheduler = torch.optim.lr_scheduler.StepLR(
                self.optimizer, step_size=30, gamma=0.1
            )
        
        return scheduler
    
    def _get_dataloaders(self) -> Tuple[DataLoader, DataLoader]:
        """Get train and validation dataloaders."""
        self.data_module.setup('fit')
        
        if self.is_distributed:
            train_sampler = DistributedSampler(
                self.data_module.train_dataset,
                num_replicas=self.world_size,
                rank=self.local_rank,
                shuffle=True
            )
            train_loader = DataLoader(
                self.data_module.train_dataset,
                batch_size=self.config.hardware.batch_size_per_gpu,
                sampler=train_sampler,
                num_workers=self.config.hardware.num_workers,
                pin_memory=self.config.hardware.pin_memory,
                drop_last=True
            )
        else:
            train_loader = self.data_module.train_dataloader()
            train_sampler = None
        
        val_loader = self.data_module.val_dataloader()
        
        return train_loader, val_loader, train_sampler
    
    def train_epoch(self, epoch: int, train_loader: DataLoader) -> Dict[str, float]:
        """
        Train for one epoch.
        
        Args:
            epoch: Current epoch number
            train_loader: Training DataLoader
        
        Returns:
            Dictionary with average metrics for the epoch
        """
        self.model.train()
        
        loss_meter = AverageMeter()
        metrics_meter = {
            'prediction': AverageMeter(),
            'ib': AverageMeter(),
            'kl': AverageMeter(),
        }
        
        pbar = tqdm(
            train_loader,
            desc=f'Epoch {epoch}/{self.config.training.num_epochs}',
            disable=self.local_rank != 0
        )
        
        for batch_idx, batch in enumerate(pbar):
            # Training step
            loss_dict = train_step(
                model=self.model,
                batch=batch,
                criterion=self.criterion,
                optimizer=self.optimizer,
                scaler=self.scaler,
                device=self.device,
                step=self.global_step,
                use_amp=self.config.hardware.mixed_precision
            )
            
            # Update meters
            loss_meter.update(loss_dict['total'])
            for key in metrics_meter:
                if key in loss_dict:
                    metrics_meter[key].update(loss_dict[key])
            
            # Update scheduler
            self.scheduler.step()
            
            # Update progress bar
            current_lr = self.optimizer.param_groups[0]['lr']
            pbar.set_postfix({
                'loss': f'{loss_meter.avg:.4f}',
                'β': f'{loss_dict["beta"]:.3f}',
                'λ': f'{loss_dict["lambda"]:.3f}',
                'lr': f'{current_lr:.2e}'
            })
            
            # Logging
            if self.global_step % self.config.training.log_every_n_steps == 0:
                self._log_step(loss_dict, epoch, 'train')
            
            # Track schedules
            self.history['beta'].append(loss_dict['beta'])
            self.history['lambda'].append(loss_dict['lambda'])
            
            self.global_step += 1
        
        # Compute epoch metrics
        epoch_metrics = {
            'loss': loss_meter.avg,
            **{k: v.avg for k, v in metrics_meter.items()}
        }
        
        return epoch_metrics
    
    @torch.no_grad()
    def validate(self, epoch: int, val_loader: DataLoader) -> Dict[str, float]:
        """
        Validate model.
        
        Args:
            epoch: Current epoch number
            val_loader: Validation DataLoader
        
        Returns:
            Dictionary with validation metrics
        """
        self.model.eval()
        
        loss_meter = AverageMeter()
        all_predictions = []
        all_targets = []
        all_uncertainties = []
        
        pbar = tqdm(
            val_loader,
            desc='Validation',
            disable=self.local_rank != 0
        )
        
        for batch in pbar:
            loss_dict, preds, targets, uncertainties = validate_step(
                self.model, batch, self.criterion, self.device
            )
            
            loss_meter.update(loss_dict['total'])
            all_predictions.append(preds)
            all_targets.append(targets)
            all_uncertainties.append(uncertainties)
        
        # Concatenate all predictions
        all_predictions = np.concatenate(all_predictions, axis=0)
        all_targets = np.concatenate(all_targets, axis=0)
        all_uncertainties = np.concatenate(all_uncertainties, axis=0)
        
        # Compute metrics
        metrics = MetricsCalculator.compute_all_metrics(
            all_targets.flatten(),
            all_predictions.flatten(),
            all_uncertainties.flatten()
        )
        
        metrics['loss'] = loss_meter.avg
        
        return metrics
    
    def _log_step(self, metrics: Dict[str, float], epoch: int, phase: str):
        """Log metrics for a step."""
        if self.writer is None:
            return
        
        for key, value in metrics.items():
            self.writer.add_scalar(f'{phase}/{key}', value, self.global_step)
        
        # Log learning rate
        lr = self.optimizer.param_groups[0]['lr']
        self.writer.add_scalar('train/learning_rate', lr, self.global_step)
    
    def _log_epoch(self, train_metrics: Dict, val_metrics: Dict, epoch: int):
        """Log metrics for an epoch."""
        if self.local_rank != 0:
            return
        
        # Console logging
        self.logger.info(
            f"Epoch {epoch}: "
            f"Train Loss={train_metrics['loss']:.4f}, "
            f"Val Loss={val_metrics['loss']:.4f}, "
            f"Val R²={val_metrics['r2']:.4f}, "
            f"Val MES={val_metrics['mes']:.4f}"
        )
        
        # TensorBoard
        if self.writer:
            self.writer.add_scalars('loss', {
                'train': train_metrics['loss'],
                'val': val_metrics['loss']
            }, epoch)
            
            for key in ['r2', 'mae', 'mes']:
                if key in val_metrics:
                    self.writer.add_scalar(f'val/{key}', val_metrics[key], epoch)
        
        # History
        self.history['train_loss'].append(train_metrics['loss'])
        self.history['val_loss'].append(val_metrics['loss'])
        self.history['train_r2'].append(train_metrics.get('r2', 0))
        self.history['val_r2'].append(val_metrics['r2'])
    
    def _save_best_model(self, metrics: Dict[str, float], epoch: int):
        """Save model if it's the best so far."""
        current_metric = metrics['r2']  # Primary metric
        
        if current_metric > self.best_metric:
            self.best_metric = current_metric
            
            if self.local_rank == 0:
                save_path = Path(self.config.experiment.checkpoint_dir) / 'best_model.pt'
                
                model_to_save = (
                    self.model.module if hasattr(self.model, 'module') else self.model
                )
                
                save_checkpoint(
                    model=model_to_save,
                    optimizer=self.optimizer,
                    epoch=epoch,
                    metrics=metrics,
                    path=str(save_path),
                    scheduler=self.scheduler
                )
                
                self.logger.info(f"New best model saved with R²={current_metric:.4f}")
    
    def train(self, num_epochs: Optional[int] = None) -> Dict[str, List]:
        """
        Main training loop.
        
        Args:
            num_epochs: Number of epochs (uses config if None)
        
        Returns:
            Training history dictionary
        """
        num_epochs = num_epochs or self.config.training.num_epochs
        
        if self.local_rank == 0:
            self.logger.info("=" * 60)
            self.logger.info("Starting UIBFuse Training")
            self.logger.info("=" * 60)
            self.logger.info(f"Epochs: {num_epochs}")
            self.logger.info(f"Batch size: {self.config.hardware.effective_batch_size}")
            self.logger.info(f"Learning rate: {self.config.training.learning_rate}")
            self.logger.info(f"Device: {self.device}")
            self.logger.info(f"Mixed precision: {self.config.hardware.mixed_precision}")
            self.logger.info("=" * 60)
        
        # Get dataloaders
        train_loader, val_loader, train_sampler = self._get_dataloaders()
        
        start_time = time.time()
        
        for epoch in range(1, num_epochs + 1):
            # Set epoch for distributed sampler
            if train_sampler is not None:
                train_sampler.set_epoch(epoch)
            
            # Train
            train_metrics = self.train_epoch(epoch, train_loader)
            
            # Validate
            if epoch % self.config.training.val_every_n_epochs == 0:
                val_metrics = self.validate(epoch, val_loader)
                
                # Log epoch metrics
                self._log_epoch(train_metrics, val_metrics, epoch)
                
                # Save best model
                self._save_best_model(val_metrics, epoch)
                
                # Early stopping
                if self.early_stopping(val_metrics['r2']):
                    self.logger.info(f"Early stopping at epoch {epoch}")
                    break
            
            # Periodic checkpoint
            if epoch % self.config.training.save_every_n_epochs == 0:
                if self.local_rank == 0:
                    save_path = Path(self.config.experiment.checkpoint_dir) / f'epoch_{epoch}.pt'
                    model_to_save = (
                        self.model.module if hasattr(self.model, 'module') else self.model
                    )
                    save_checkpoint(
                        model=model_to_save,
                        optimizer=self.optimizer,
                        epoch=epoch,
                        metrics=val_metrics,
                        path=str(save_path),
                        scheduler=self.scheduler
                    )
            
            # Log GPU memory
            if self.local_rank == 0 and epoch % 10 == 0:
                mem_stats = get_gpu_memory_usage()
                self.logger.info(f"GPU Memory: {mem_stats}")
        
        # Training complete
        total_time = time.time() - start_time
        
        if self.local_rank == 0:
            self.logger.info("=" * 60)
            self.logger.info("Training Complete!")
            self.logger.info(f"Total time: {total_time / 3600:.2f} hours")
            self.logger.info(f"Best R²: {self.best_metric:.4f}")
            self.logger.info("=" * 60)
            
            # Save training curves
            fig = self.visualizer.plot_training_curves(
                self.history,
                save_path=str(Path(self.config.experiment.figure_dir) / 'training_curves.pdf')
            )
            
            if self.writer:
                self.writer.close()
        
        return self.history


# =============================================================================
# ABLATION STUDIES
# =============================================================================

class AblationStudyRunner:
    """
    Systematic ablation experiment runner.
    
    Conducts controlled experiments to validate each component's
    contribution to the overall model performance.
    
    Ablation Studies:
    ----------------
    1. Pyramid levels: {1, 2, 3, 4} scales
    2. Latent dimension: {64, 128, 256, 512, 1024}
    3. Attention heads: {4, 8, 16, 32}
    4. Fusion mechanism: Bayesian vs GMU vs Early vs Late
    5. Uncertainty: With vs without learned variance
    6. Volatility gating: Adaptive vs static
    """
    
    def __init__(self, base_config: UIBFuseConfig):
        self.base_config = base_config
        self.results = {}
        self.logger = setup_logging(
            base_config.experiment.log_dir,
            name='ablation'
        )
    
    def run_single_ablation(
        self,
        config: UIBFuseConfig,
        name: str,
        epochs: int = 50
    ) -> Dict[str, float]:
        """Run a single ablation experiment."""
        self.logger.info(f"Running ablation: {name}")
        
        trainer = UIBFuseTrainer(config)
        trainer.train(num_epochs=epochs)
        
        # Get final validation metrics
        _, val_loader, _ = trainer._get_dataloaders()
        final_metrics = trainer.validate(epochs, val_loader)
        
        self.results[name] = final_metrics
        
        return final_metrics
    
    def ablate_pyramid_levels(self) -> Dict[str, Dict[str, float]]:
        """Ablate number of pyramid levels."""
        results = {}
        
        for num_levels in [1, 2, 3, 4]:
            config = get_config('default')
            
            # Modify pyramid configuration
            levels = [224 // (2 ** i) for i in range(num_levels)]
            channels = [64 * (2 ** i) for i in range(num_levels)]
            
            config.visual_encoder.pyramid_levels = levels
            config.visual_encoder.pyramid_channels = channels
            config.visual_encoder.pyramid_depths = [4 * (i + 1) for i in range(num_levels)]
            
            config.experiment.experiment_name = f'ablation_pyramid_{num_levels}'
            
            metrics = self.run_single_ablation(
                config, f'pyramid_levels_{num_levels}', epochs=30
            )
            results[f'{num_levels}_levels'] = metrics
        
        return results
    
    def ablate_latent_dimension(self) -> Dict[str, Dict[str, float]]:
        """Ablate latent space dimension."""
        results = {}
        
        for latent_dim in [64, 128, 256, 512, 1024]:
            config = get_config('default')
            config.info_theoretic.latent_dim = latent_dim
            config.experiment.experiment_name = f'ablation_latent_{latent_dim}'
            
            metrics = self.run_single_ablation(
                config, f'latent_dim_{latent_dim}', epochs=30
            )
            results[f'd_z={latent_dim}'] = metrics
        
        return results
    
    def ablate_attention_heads(self) -> Dict[str, Dict[str, float]]:
        """Ablate number of attention heads."""
        results = {}
        
        for num_heads in [4, 8, 16, 32]:
            config = get_config('default')
            config.volatility_attention.num_attention_heads = num_heads
            config.volatility_attention.attention_dim = 1024 // num_heads
            config.experiment.experiment_name = f'ablation_heads_{num_heads}'
            
            metrics = self.run_single_ablation(
                config, f'attention_heads_{num_heads}', epochs=30
            )
            results[f'H={num_heads}'] = metrics
        
        return results
    
    def ablate_fusion_mechanism(self) -> Dict[str, Dict[str, float]]:
        """Ablate fusion mechanism type."""
        results = {}
        
        fusion_types = [
            ('bayesian', 'uibfuse'),
            ('early', 'early_fusion'),
            ('late', 'late_fusion'),
            ('gated', 'gated_fusion'),
        ]
        
        for fusion_name, model_name in fusion_types:
            config = get_config('default')
            config.experiment.experiment_name = f'ablation_fusion_{fusion_name}'
            
            # Build specific model
            model = build_model(model_name, config)
            
            trainer = UIBFuseTrainer(config, model=model)
            trainer.train(num_epochs=30)
            
            _, val_loader, _ = trainer._get_dataloaders()
            metrics = trainer.validate(30, val_loader)
            
            results[fusion_name] = metrics
        
        return results
    
    def ablate_uncertainty(self) -> Dict[str, Dict[str, float]]:
        """Ablate uncertainty estimation."""
        results = {}
        
        # With uncertainty (default)
        config = get_config('default')
        config.prediction_head.predict_uncertainty = True
        config.experiment.experiment_name = 'ablation_uncertainty_with'
        
        metrics_with = self.run_single_ablation(
            config, 'with_uncertainty', epochs=30
        )
        results['with_uncertainty'] = metrics_with
        
        # Without uncertainty
        config = get_config('default')
        config.prediction_head.predict_uncertainty = False
        config.experiment.experiment_name = 'ablation_uncertainty_without'
        
        metrics_without = self.run_single_ablation(
            config, 'without_uncertainty', epochs=30
        )
        results['without_uncertainty'] = metrics_without
        
        return results
    
    def ablate_volatility_gating(self) -> Dict[str, Dict[str, float]]:
        """Ablate volatility-adaptive gating."""
        results = {}
        
        # With volatility gating (default)
        config = get_config('default')
        config.experiment.experiment_name = 'ablation_volatility_adaptive'
        
        metrics_adaptive = self.run_single_ablation(
            config, 'adaptive_gating', epochs=30
        )
        results['adaptive'] = metrics_adaptive
        
        # Static gating (λ = 0)
        config = get_config('default')
        config.volatility_attention.lambda_volatility = 0.0
        config.experiment.experiment_name = 'ablation_volatility_static'
        
        metrics_static = self.run_single_ablation(
            config, 'static_gating', epochs=30
        )
        results['static'] = metrics_static
        
        return results
    
    def run_all_ablations(self) -> Dict[str, Dict]:
        """Run all ablation studies."""
        self.logger.info("=" * 60)
        self.logger.info("Running Complete Ablation Study")
        self.logger.info("=" * 60)
        
        all_results = {}
        
        # Run each ablation
        ablations = [
            ('pyramid_levels', self.ablate_pyramid_levels),
            ('latent_dimension', self.ablate_latent_dimension),
            ('attention_heads', self.ablate_attention_heads),
            ('fusion_mechanism', self.ablate_fusion_mechanism),
            ('uncertainty', self.ablate_uncertainty),
            ('volatility_gating', self.ablate_volatility_gating),
        ]
        
        for name, ablation_fn in ablations:
            self.logger.info(f"\n{'='*40}")
            self.logger.info(f"Ablation: {name}")
            self.logger.info(f"{'='*40}")
            
            try:
                results = ablation_fn()
                all_results[name] = results
            except Exception as e:
                self.logger.error(f"Ablation {name} failed: {e}")
                all_results[name] = {'error': str(e)}
        
        # Save results
        self._save_results(all_results)
        
        return all_results
    
    def _save_results(self, results: Dict):
        """Save ablation results."""
        import json
        
        save_path = Path(self.base_config.experiment.output_dir) / 'ablation_results.json'
        
        # Convert numpy types to Python types
        def convert(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.float32, np.float64)):
                return float(obj)
            elif isinstance(obj, (np.int32, np.int64)):
                return int(obj)
            elif isinstance(obj, dict):
                return {k: convert(v) for k, v in obj.items()}
            return obj
        
        results_converted = convert(results)
        
        with open(save_path, 'w') as f:
            json.dump(results_converted, f, indent=2)
        
        self.logger.info(f"Ablation results saved to {save_path}")


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='UIBFuse Training Pipeline',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Configuration
    parser.add_argument(
        '--config', type=str, default='default',
        choices=['default', 'debug', 'ablation', 'baseline'],
        help='Configuration preset'
    )
    
    # Training
    parser.add_argument('--epochs', type=int, default=None, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=None, help='Batch size')
    parser.add_argument('--lr', type=float, default=None, help='Learning rate')
    
    # Hardware
    parser.add_argument('--gpus', type=int, default=4, help='Number of GPUs')
    parser.add_argument('--no-amp', action='store_true', help='Disable mixed precision')
    
    # Data
    parser.add_argument('--data-path', type=str, default='./data/cryptopunks', help='Data path')
    
    # Checkpointing
    parser.add_argument('--resume', type=str, default=None, help='Resume from checkpoint')
    parser.add_argument('--output-dir', type=str, default='./outputs', help='Output directory')
    
    # Ablation
    parser.add_argument(
        '--ablation', type=str, default=None,
        choices=['all', 'pyramid', 'latent', 'heads', 'fusion', 'uncertainty', 'volatility'],
        help='Run ablation study'
    )
    
    # Misc
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--debug', action='store_true', help='Debug mode')
    
    # Distributed
    parser.add_argument('--local_rank', type=int, default=0, help='Local rank for distributed')
    
    return parser.parse_args()


def main(args: argparse.Namespace):
    """Main training entry point."""
    # Set seed
    set_seed(args.seed)
    
    # Get configuration
    config_name = 'debug' if args.debug else args.config
    config = get_config(config_name)
    
    # Override config with command line arguments
    if args.epochs is not None:
        config.training.num_epochs = args.epochs
    if args.batch_size is not None:
        config.hardware.effective_batch_size = args.batch_size
    if args.lr is not None:
        config.training.learning_rate = args.lr
    if args.gpus is not None:
        config.hardware.num_gpus = args.gpus
    if args.no_amp:
        config.hardware.mixed_precision = False
    if args.data_path:
        config.data.data_root = args.data_path
    if args.output_dir:
        config.experiment.output_dir = args.output_dir
        config.experiment.checkpoint_dir = f'{args.output_dir}/checkpoints'
        config.experiment.log_dir = f'{args.output_dir}/logs'
        config.experiment.figure_dir = f'{args.output_dir}/figures'
    
    # Create output directories
    Path(config.experiment.output_dir).mkdir(parents=True, exist_ok=True)
    Path(config.experiment.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    Path(config.experiment.log_dir).mkdir(parents=True, exist_ok=True)
    Path(config.experiment.figure_dir).mkdir(parents=True, exist_ok=True)
    
    # Run ablation study if specified
    if args.ablation:
        ablation_runner = AblationStudyRunner(config)
        
        if args.ablation == 'all':
            ablation_runner.run_all_ablations()
        elif args.ablation == 'pyramid':
            ablation_runner.ablate_pyramid_levels()
        elif args.ablation == 'latent':
            ablation_runner.ablate_latent_dimension()
        elif args.ablation == 'heads':
            ablation_runner.ablate_attention_heads()
        elif args.ablation == 'fusion':
            ablation_runner.ablate_fusion_mechanism()
        elif args.ablation == 'uncertainty':
            ablation_runner.ablate_uncertainty()
        elif args.ablation == 'volatility':
            ablation_runner.ablate_volatility_gating()
        
        return
    
    # Standard training
    trainer = UIBFuseTrainer(config)
    
    # Resume from checkpoint if specified
    if args.resume:
        checkpoint = load_checkpoint(
            trainer.model if not hasattr(trainer.model, 'module') else trainer.model.module,
            args.resume,
            trainer.optimizer,
            trainer.scheduler
        )
        print(f"Resumed from epoch {checkpoint['epoch']}")
    
    # Train
    history = trainer.train()
    
    print("\nTraining completed!")
    print(f"Best R²: {trainer.best_metric:.4f}")


if __name__ == "__main__":
    args = parse_args()
    main(args)
