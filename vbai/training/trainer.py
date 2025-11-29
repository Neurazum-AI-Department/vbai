"""
Trainer Class for Vbai Models
"""

import os
import time
from pathlib import Path
from typing import Optional, Dict, List, Union, Callable
from dataclasses import dataclass, field

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Optimizer, Adam
from torch.optim.lr_scheduler import _LRScheduler, ReduceLROnPlateau

from .losses import MultiTaskLoss
from .callbacks import CallbackList, EarlyStopping, ModelCheckpoint


@dataclass
class TrainingHistory:
    """Container for training history."""
    train_loss: List[float] = field(default_factory=list)
    val_loss: List[float] = field(default_factory=list)
    dementia_acc: List[float] = field(default_factory=list)
    tumor_acc: List[float] = field(default_factory=list)
    val_dementia_acc: List[float] = field(default_factory=list)
    val_tumor_acc: List[float] = field(default_factory=list)
    lr: List[float] = field(default_factory=list)
    epoch_times: List[float] = field(default_factory=list)


class Trainer:
    """
    Keras-style trainer for Vbai models.
    
    Args:
        model: MultiTaskBrainModel instance
        optimizer: Optimizer (default: Adam)
        lr: Learning rate (default: 0.0005)
        loss_fn: Loss function (default: MultiTaskLoss)
        device: Device to train on ('cuda' or 'cpu')
        callbacks: List of callbacks
    
    Example:
        >>> model = vbai.MultiTaskBrainModel(variant='q')
        >>> trainer = vbai.Trainer(model=model, epochs=10, lr=0.0005)
        >>> history = trainer.fit(train_dataset, val_dataset)
        >>> trainer.save('model.pt')
    """
    
    def __init__(
        self,
        model: nn.Module,
        optimizer: Optional[Optimizer] = None,
        lr: float = 0.0005,
        loss_fn: Optional[nn.Module] = None,
        device: Optional[str] = None,
        callbacks: Optional[List] = None,
        scheduler: Optional[_LRScheduler] = None,
    ):
        # Device setup
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        self.model = model.to(self.device)
        self.lr = lr
        
        # Optimizer
        if optimizer is None:
            self.optimizer = Adam(self.model.parameters(), lr=lr)
        else:
            self.optimizer = optimizer
        
        # Loss function
        if loss_fn is None:
            self.loss_fn = MultiTaskLoss(ignore_index=-1)
        else:
            self.loss_fn = loss_fn
        
        # Scheduler
        self.scheduler = scheduler
        
        # Callbacks
        self.callbacks = CallbackList(callbacks or [])
        
        # Training state
        self.history = TrainingHistory()
        self.current_epoch = 0
        self.best_val_loss = float('inf')
    
    def fit(
        self,
        train_data: Union[DataLoader, 'UnifiedMRIDataset'],
        val_data: Optional[Union[DataLoader, 'UnifiedMRIDataset']] = None,
        epochs: int = 10,
        batch_size: int = 32,
        verbose: int = 1,
    ) -> TrainingHistory:
        """
        Train the model.
        
        Args:
            train_data: Training DataLoader or Dataset
            val_data: Validation DataLoader or Dataset (optional)
            epochs: Number of training epochs
            batch_size: Batch size (if Dataset provided)
            verbose: Verbosity level (0=silent, 1=progress, 2=detailed)
        
        Returns:
            TrainingHistory with loss/accuracy curves
        """
        # Create DataLoaders if needed
        if not isinstance(train_data, DataLoader):
            train_loader = DataLoader(
                train_data, batch_size=batch_size, shuffle=True,
                num_workers=4, pin_memory=True
            )
        else:
            train_loader = train_data
        
        if val_data is not None and not isinstance(val_data, DataLoader):
            val_loader = DataLoader(
                val_data, batch_size=batch_size, shuffle=False,
                num_workers=4, pin_memory=True
            )
        else:
            val_loader = val_data
        
        # Callbacks setup
        self.callbacks.on_train_begin(self)
        
        # Training loop
        for epoch in range(epochs):
            self.current_epoch = epoch
            epoch_start = time.time()
            
            self.callbacks.on_epoch_begin(epoch, self)
            
            # Train one epoch
            train_metrics = self._train_epoch(train_loader, verbose)
            
            # Validate if val_data provided
            if val_loader is not None:
                val_metrics = self._validate(val_loader)
                self.history.val_loss.append(val_metrics['loss'])
                self.history.val_dementia_acc.append(val_metrics['dementia_acc'])
                self.history.val_tumor_acc.append(val_metrics['tumor_acc'])
            else:
                val_metrics = None
            
            # Update history
            self.history.train_loss.append(train_metrics['loss'])
            self.history.dementia_acc.append(train_metrics['dementia_acc'])
            self.history.tumor_acc.append(train_metrics['tumor_acc'])
            self.history.lr.append(self.optimizer.param_groups[0]['lr'])
            self.history.epoch_times.append(time.time() - epoch_start)
            
            # Scheduler step
            if self.scheduler is not None:
                if isinstance(self.scheduler, ReduceLROnPlateau):
                    val_loss = val_metrics['loss'] if val_metrics else train_metrics['loss']
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()
            
            # Print progress
            if verbose >= 1:
                self._print_epoch(epoch, epochs, train_metrics, val_metrics)
            
            # Callbacks
            logs = {**train_metrics}
            if val_metrics:
                logs.update({f'val_{k}': v for k, v in val_metrics.items()})
            
            self.callbacks.on_epoch_end(epoch, logs, self)
            
            # Check for early stopping
            if self.callbacks.should_stop:
                if verbose >= 1:
                    print(f"Early stopping at epoch {epoch + 1}")
                break
        
        self.callbacks.on_train_end(self)
        
        return self.history
    
    def _train_epoch(self, loader: DataLoader, verbose: int) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        
        total_loss = 0.0
        dementia_correct = 0
        dementia_total = 0
        tumor_correct = 0
        tumor_total = 0
        
        for batch_idx, (images, dementia_labels, tumor_labels) in enumerate(loader):
            images = images.to(self.device)
            dementia_labels = dementia_labels.to(self.device)
            tumor_labels = tumor_labels.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            dementia_logits, tumor_logits = self.model(images)
            
            # Compute loss
            loss = self.loss_fn(dementia_logits, tumor_logits, dementia_labels, tumor_labels)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Metrics
            total_loss += loss.item()
            
            # Dementia accuracy (ignore -1 labels)
            d_mask = dementia_labels >= 0
            if d_mask.sum() > 0:
                d_pred = dementia_logits[d_mask].argmax(dim=1)
                dementia_correct += (d_pred == dementia_labels[d_mask]).sum().item()
                dementia_total += d_mask.sum().item()
            
            # Tumor accuracy
            t_mask = tumor_labels >= 0
            if t_mask.sum() > 0:
                t_pred = tumor_logits[t_mask].argmax(dim=1)
                tumor_correct += (t_pred == tumor_labels[t_mask]).sum().item()
                tumor_total += t_mask.sum().item()
        
        return {
            'loss': total_loss / len(loader),
            'dementia_acc': dementia_correct / max(dementia_total, 1),
            'tumor_acc': tumor_correct / max(tumor_total, 1),
        }
    
    def _validate(self, loader: DataLoader) -> Dict[str, float]:
        """Validate the model."""
        self.model.eval()
        
        total_loss = 0.0
        dementia_correct = 0
        dementia_total = 0
        tumor_correct = 0
        tumor_total = 0
        
        with torch.no_grad():
            for images, dementia_labels, tumor_labels in loader:
                images = images.to(self.device)
                dementia_labels = dementia_labels.to(self.device)
                tumor_labels = tumor_labels.to(self.device)
                
                dementia_logits, tumor_logits = self.model(images)
                loss = self.loss_fn(dementia_logits, tumor_logits, dementia_labels, tumor_labels)
                
                total_loss += loss.item()
                
                d_mask = dementia_labels >= 0
                if d_mask.sum() > 0:
                    d_pred = dementia_logits[d_mask].argmax(dim=1)
                    dementia_correct += (d_pred == dementia_labels[d_mask]).sum().item()
                    dementia_total += d_mask.sum().item()
                
                t_mask = tumor_labels >= 0
                if t_mask.sum() > 0:
                    t_pred = tumor_logits[t_mask].argmax(dim=1)
                    tumor_correct += (t_pred == tumor_labels[t_mask]).sum().item()
                    tumor_total += t_mask.sum().item()
        
        return {
            'loss': total_loss / len(loader),
            'dementia_acc': dementia_correct / max(dementia_total, 1),
            'tumor_acc': tumor_correct / max(tumor_total, 1),
        }
    
    def _print_epoch(
        self, 
        epoch: int, 
        total_epochs: int,
        train: Dict, 
        val: Optional[Dict]
    ):
        """Print epoch progress."""
        msg = (
            f"Epoch {epoch + 1}/{total_epochs} - "
            f"loss: {train['loss']:.4f} - "
            f"dem_acc: {train['dementia_acc']:.4f} - "
            f"tum_acc: {train['tumor_acc']:.4f}"
        )
        
        if val:
            msg += (
                f" - val_loss: {val['loss']:.4f} - "
                f"val_dem: {val['dementia_acc']:.4f} - "
                f"val_tum: {val['tumor_acc']:.4f}"
            )
        
        print(msg)
    
    def save(
        self, 
        path: str,
        save_optimizer: bool = True,
        save_history: bool = True
    ):
        """
        Save model checkpoint.
        
        Args:
            path: Path to save checkpoint
            save_optimizer: Whether to save optimizer state
            save_history: Whether to save training history
        """
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'config': {
                'variant': getattr(self.model, 'variant', 'q'),
                'num_dementia_classes': getattr(self.model, 'num_dementia_classes', 6),
                'num_tumor_classes': getattr(self.model, 'num_tumor_classes', 4),
            },
            'epoch': self.current_epoch,
        }
        
        if save_optimizer:
            checkpoint['optimizer_state_dict'] = self.optimizer.state_dict()
        
        if save_history:
            checkpoint['history'] = {
                'train_loss': self.history.train_loss,
                'val_loss': self.history.val_loss,
                'dementia_acc': self.history.dementia_acc,
                'tumor_acc': self.history.tumor_acc,
            }
        
        # Create directory if needed
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        
        torch.save(checkpoint, path)
        print(f"Model saved to {path}")
    
    def load(self, path: str, load_optimizer: bool = True):
        """
        Load model from checkpoint.
        
        Args:
            path: Path to checkpoint
            load_optimizer: Whether to load optimizer state
        """
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        if load_optimizer and 'optimizer_state_dict' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if 'epoch' in checkpoint:
            self.current_epoch = checkpoint['epoch']
        
        print(f"Model loaded from {path}")
    
    def evaluate(self, test_data: Union[DataLoader, 'UnifiedMRIDataset']) -> Dict[str, float]:
        """
        Evaluate model on test data.
        
        Args:
            test_data: Test DataLoader or Dataset
        
        Returns:
            Dictionary with test metrics
        """
        if not isinstance(test_data, DataLoader):
            test_loader = DataLoader(test_data, batch_size=32, shuffle=False)
        else:
            test_loader = test_data
        
        return self._validate(test_loader)
