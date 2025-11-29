"""
Training Callbacks for Vbai
"""

import os
from pathlib import Path
from typing import Optional, Dict, List, Any
from abc import ABC, abstractmethod

import torch


class Callback(ABC):
    """Base class for callbacks."""
    
    def on_train_begin(self, trainer):
        """Called at the beginning of training."""
        pass
    
    def on_train_end(self, trainer):
        """Called at the end of training."""
        pass
    
    def on_epoch_begin(self, epoch: int, trainer):
        """Called at the beginning of each epoch."""
        pass
    
    def on_epoch_end(self, epoch: int, logs: Dict, trainer):
        """Called at the end of each epoch."""
        pass
    
    def on_batch_begin(self, batch: int, trainer):
        """Called at the beginning of each batch."""
        pass
    
    def on_batch_end(self, batch: int, logs: Dict, trainer):
        """Called at the end of each batch."""
        pass


class CallbackList:
    """Container for managing multiple callbacks."""
    
    def __init__(self, callbacks: Optional[List[Callback]] = None):
        self.callbacks = callbacks or []
        self.should_stop = False
    
    def append(self, callback: Callback):
        """Add a callback."""
        self.callbacks.append(callback)
    
    def on_train_begin(self, trainer):
        for cb in self.callbacks:
            cb.on_train_begin(trainer)
    
    def on_train_end(self, trainer):
        for cb in self.callbacks:
            cb.on_train_end(trainer)
    
    def on_epoch_begin(self, epoch: int, trainer):
        for cb in self.callbacks:
            cb.on_epoch_begin(epoch, trainer)
    
    def on_epoch_end(self, epoch: int, logs: Dict, trainer):
        for cb in self.callbacks:
            cb.on_epoch_end(epoch, logs, trainer)
            if hasattr(cb, 'should_stop') and cb.should_stop:
                self.should_stop = True


class EarlyStopping(Callback):
    """
    Stop training when a monitored metric stops improving.
    
    Args:
        monitor: Metric to monitor (default: 'val_loss')
        min_delta: Minimum change to qualify as improvement
        patience: Number of epochs with no improvement before stopping
        mode: 'min' for loss, 'max' for accuracy
        verbose: Whether to print messages
    
    Example:
        >>> early_stop = EarlyStopping(monitor='val_loss', patience=5)
        >>> trainer = Trainer(model, callbacks=[early_stop])
    """
    
    def __init__(
        self,
        monitor: str = 'val_loss',
        min_delta: float = 0.0,
        patience: int = 5,
        mode: str = 'auto',
        verbose: bool = True
    ):
        self.monitor = monitor
        self.min_delta = min_delta
        self.patience = patience
        self.verbose = verbose
        
        # Determine mode
        if mode == 'auto':
            if 'loss' in monitor:
                self.mode = 'min'
            else:
                self.mode = 'max'
        else:
            self.mode = mode
        
        self.best = float('inf') if self.mode == 'min' else float('-inf')
        self.counter = 0
        self.should_stop = False
    
    def on_train_begin(self, trainer):
        self.best = float('inf') if self.mode == 'min' else float('-inf')
        self.counter = 0
        self.should_stop = False
    
    def on_epoch_end(self, epoch: int, logs: Dict, trainer):
        current = logs.get(self.monitor)
        
        if current is None:
            return
        
        if self.mode == 'min':
            improved = current < (self.best - self.min_delta)
        else:
            improved = current > (self.best + self.min_delta)
        
        if improved:
            self.best = current
            self.counter = 0
            if self.verbose:
                print(f"EarlyStopping: {self.monitor} improved to {current:.4f}")
        else:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping: No improvement for {self.counter}/{self.patience} epochs")
            
            if self.counter >= self.patience:
                self.should_stop = True
                if self.verbose:
                    print(f"EarlyStopping: Stopping training")


class ModelCheckpoint(Callback):
    """
    Save model checkpoints during training.
    
    Args:
        filepath: Path template for checkpoints (supports {epoch}, {val_loss})
        monitor: Metric to monitor for saving best model
        save_best_only: Only save when monitored metric improves
        mode: 'min' or 'max'
        verbose: Whether to print messages
    
    Example:
        >>> checkpoint = ModelCheckpoint(
        ...     filepath='checkpoints/model_{epoch:02d}.pt',
        ...     monitor='val_loss',
        ...     save_best_only=True
        ... )
    """
    
    def __init__(
        self,
        filepath: str = 'checkpoint.pt',
        monitor: str = 'val_loss',
        save_best_only: bool = True,
        mode: str = 'auto',
        verbose: bool = True
    ):
        self.filepath = filepath
        self.monitor = monitor
        self.save_best_only = save_best_only
        self.verbose = verbose
        
        if mode == 'auto':
            self.mode = 'min' if 'loss' in monitor else 'max'
        else:
            self.mode = mode
        
        self.best = float('inf') if self.mode == 'min' else float('-inf')
    
    def on_train_begin(self, trainer):
        self.best = float('inf') if self.mode == 'min' else float('-inf')
        
        # Create directory
        Path(self.filepath).parent.mkdir(parents=True, exist_ok=True)
    
    def on_epoch_end(self, epoch: int, logs: Dict, trainer):
        current = logs.get(self.monitor)
        
        # Format filepath
        filepath = self.filepath.format(
            epoch=epoch + 1,
            **{k: v for k, v in logs.items() if isinstance(v, (int, float))}
        )
        
        if self.save_best_only:
            if current is None:
                return
            
            if self.mode == 'min':
                improved = current < self.best
            else:
                improved = current > self.best
            
            if improved:
                self.best = current
                trainer.save(filepath, save_optimizer=True, save_history=False)
                if self.verbose:
                    print(f"ModelCheckpoint: Saved to {filepath} ({self.monitor}={current:.4f})")
        else:
            trainer.save(filepath, save_optimizer=True, save_history=False)
            if self.verbose:
                print(f"ModelCheckpoint: Saved to {filepath}")


class LearningRateLogger(Callback):
    """Log learning rate at each epoch."""
    
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.lrs: List[float] = []
    
    def on_epoch_end(self, epoch: int, logs: Dict, trainer):
        lr = trainer.optimizer.param_groups[0]['lr']
        self.lrs.append(lr)
        if self.verbose:
            print(f"Learning rate: {lr:.2e}")


class TensorBoardLogger(Callback):
    """
    Log metrics to TensorBoard.
    
    Args:
        log_dir: Directory for TensorBoard logs
    """
    
    def __init__(self, log_dir: str = './logs'):
        self.log_dir = log_dir
        self.writer = None
    
    def on_train_begin(self, trainer):
        try:
            from torch.utils.tensorboard import SummaryWriter
            self.writer = SummaryWriter(self.log_dir)
        except ImportError:
            print("TensorBoard not available. Install with: pip install tensorboard")
    
    def on_epoch_end(self, epoch: int, logs: Dict, trainer):
        if self.writer is None:
            return
        
        for key, value in logs.items():
            if isinstance(value, (int, float)):
                self.writer.add_scalar(key, value, epoch)
    
    def on_train_end(self, trainer):
        if self.writer is not None:
            self.writer.close()


class ProgressBar(Callback):
    """
    Display a progress bar during training.
    Requires tqdm.
    """
    
    def __init__(self):
        self.pbar = None
    
    def on_train_begin(self, trainer):
        try:
            from tqdm import tqdm
            self.tqdm = tqdm
        except ImportError:
            print("tqdm not available. Install with: pip install tqdm")
            self.tqdm = None
    
    def on_epoch_begin(self, epoch: int, trainer):
        if self.tqdm:
            # Progress bar is handled in trainer for batch-level updates
            pass
