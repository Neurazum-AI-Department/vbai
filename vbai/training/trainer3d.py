"""
3D Trainer Class for Vbai Models

Keras-style trainer adapted for 3D volumetric models.
"""

import time
from pathlib import Path
from typing import Optional, Dict, List, Union
from dataclasses import dataclass, field

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Optimizer, AdamW
from torch.optim.lr_scheduler import _LRScheduler, ReduceLROnPlateau

from .callbacks import CallbackList, EarlyStopping, ModelCheckpoint


@dataclass
class Training3DHistory:
    """Container for 3D model training history."""
    train_loss: List[float] = field(default_factory=list)
    val_loss: List[float] = field(default_factory=list)
    task_acc: Dict[str, List[float]] = field(default_factory=dict)
    val_task_acc: Dict[str, List[float]] = field(default_factory=dict)
    lr: List[float] = field(default_factory=list)
    epoch_times: List[float] = field(default_factory=list)


class Trainer3D:
    """
    Trainer for 3D volumetric brain MRI models.

    Handles multi-task training with dict-based label format and
    NIfTI dataset integration.

    Args:
        model: MultiTask3DBrainModel instance
        optimizer: Optimizer (default: AdamW)
        lr: Learning rate (default: 1e-4)
        loss_fn: Loss function (default: CrossEntropyLoss per task)
        device: Device to train on
        callbacks: List of callbacks
        scheduler: Learning rate scheduler
        class_weights: Dict mapping task names to class weight tensors
        mixed_precision: Whether to use AMP mixed precision training
        gradient_clip: Max gradient norm for clipping (0 to disable)

    Example:
        >>> model = vbai.MultiTask3DBrainModel(variant='q', tasks={'alzheimer': 3})
        >>> trainer = vbai.Trainer3D(model=model, lr=1e-4)
        >>> history = trainer.fit(train_loader, val_loader, epochs=25)
        >>> trainer.save('model_3d.pt')
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: Optional[Optimizer] = None,
        lr: float = 1e-4,
        loss_fn: Optional[nn.Module] = None,
        device: Optional[str] = None,
        callbacks: Optional[List] = None,
        scheduler: Optional[_LRScheduler] = None,
        class_weights: Optional[Dict[str, torch.Tensor]] = None,
        mixed_precision: bool = False,
        gradient_clip: float = 1.0,
    ):
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        self.model = model.to(self.device)
        self.lr = lr
        self.mixed_precision = mixed_precision
        self.gradient_clip = gradient_clip

        # Get task info from model
        self.task_names = getattr(model, 'task_names', ['default'])
        self.tasks = getattr(model, 'tasks', {'default': 2})

        # Optimizer
        if optimizer is None:
            self.optimizer = AdamW(self.model.parameters(), lr=lr, weight_decay=1e-4)
        else:
            self.optimizer = optimizer

        # Loss functions per task
        if loss_fn is not None:
            self.loss_fn = loss_fn
        else:
            self.task_losses = {}
            for task_name in self.task_names:
                weight = class_weights.get(task_name) if class_weights else None
                if weight is not None:
                    weight = weight.to(self.device)
                self.task_losses[task_name] = nn.CrossEntropyLoss(
                    weight=weight, ignore_index=-1
                )
            self.loss_fn = None

        # Scheduler
        self.scheduler = scheduler

        # Callbacks
        self.callbacks = CallbackList(callbacks or [])

        # AMP scaler
        self.scaler = torch.cuda.amp.GradScaler() if mixed_precision and torch.cuda.is_available() else None

        # Training state
        self.history = Training3DHistory()
        self.current_epoch = 0

    def _compute_loss(
        self,
        logits: Dict[str, torch.Tensor],
        labels: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Compute combined loss across all tasks."""
        if self.loss_fn is not None:
            return self.loss_fn(logits, labels)

        total_loss = torch.tensor(0.0, device=self.device)
        for task_name in self.task_names:
            if task_name in logits and task_name in labels:
                task_labels = labels[task_name]
                mask = task_labels >= 0
                if mask.sum() > 0:
                    loss = self.task_losses[task_name](
                        logits[task_name], task_labels
                    )
                    if not torch.isnan(loss):
                        total_loss = total_loss + loss
        return total_loss

    def fit(
        self,
        train_data: DataLoader,
        val_data: Optional[DataLoader] = None,
        epochs: int = 25,
        verbose: int = 1,
    ) -> Training3DHistory:
        """
        Train the 3D model.

        Args:
            train_data: Training DataLoader (yields volume, labels_dict)
            val_data: Validation DataLoader (optional)
            epochs: Number of training epochs
            verbose: Verbosity (0=silent, 1=progress, 2=detailed)

        Returns:
            Training3DHistory with loss/accuracy curves
        """
        # Initialize history
        for task_name in self.task_names:
            self.history.task_acc[task_name] = []
            self.history.val_task_acc[task_name] = []

        self.callbacks.on_train_begin(self)

        for epoch in range(epochs):
            self.current_epoch = epoch
            epoch_start = time.time()
            self.callbacks.on_epoch_begin(epoch, self)

            # Train
            train_metrics = self._train_epoch(train_data)

            # Validate
            if val_data is not None:
                val_metrics = self._validate(val_data)
                self.history.val_loss.append(val_metrics['loss'])
                for task_name in self.task_names:
                    acc_key = f'{task_name}_acc'
                    self.history.val_task_acc[task_name].append(
                        val_metrics.get(acc_key, 0.0)
                    )
            else:
                val_metrics = None

            # Update history
            self.history.train_loss.append(train_metrics['loss'])
            for task_name in self.task_names:
                acc_key = f'{task_name}_acc'
                self.history.task_acc[task_name].append(
                    train_metrics.get(acc_key, 0.0)
                )
            self.history.lr.append(self.optimizer.param_groups[0]['lr'])
            self.history.epoch_times.append(time.time() - epoch_start)

            # Scheduler
            if self.scheduler is not None:
                if isinstance(self.scheduler, ReduceLROnPlateau):
                    val_loss = val_metrics['loss'] if val_metrics else train_metrics['loss']
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()

            # Print
            if verbose >= 1:
                self._print_epoch(epoch, epochs, train_metrics, val_metrics)

            # Callbacks
            logs = {**train_metrics}
            if val_metrics:
                logs.update({f'val_{k}': v for k, v in val_metrics.items()})
            self.callbacks.on_epoch_end(epoch, logs, self)

            if self.callbacks.should_stop:
                if verbose >= 1:
                    print(f"Early stopping at epoch {epoch + 1}")
                break

        self.callbacks.on_train_end(self)
        return self.history

    def _parse_labels(self, labels) -> Dict[str, torch.Tensor]:
        """Convert labels from various formats to task-keyed dict of tensors."""
        label_tensors = {}
        if isinstance(labels, dict):
            # Multi-task format: {task_name: tensor_or_list}
            for task_name in self.task_names:
                if task_name in labels:
                    task_labels = labels[task_name]
                    if isinstance(task_labels, list):
                        task_labels = torch.tensor(task_labels)
                    label_tensors[task_name] = task_labels.to(self.device)
        elif isinstance(labels, (torch.Tensor, int)):
            # Single-task format: plain tensor/int - assign to first task
            if isinstance(labels, int):
                labels = torch.tensor([labels])
            label_tensors[self.task_names[0]] = labels.to(self.device)
        return label_tensors

    def _train_epoch(self, loader: DataLoader) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        task_correct = {t: 0 for t in self.task_names}
        task_total = {t: 0 for t in self.task_names}

        for batch_idx, (volumes, labels) in enumerate(loader):
            volumes = volumes.to(self.device)
            label_tensors = self._parse_labels(labels)

            self.optimizer.zero_grad()

            if self.scaler is not None:
                with torch.cuda.amp.autocast():
                    logits = self.model(volumes)
                    loss = self._compute_loss(logits, label_tensors)
                self.scaler.scale(loss).backward()
                if self.gradient_clip > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), max_norm=self.gradient_clip
                    )
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                logits = self.model(volumes)
                loss = self._compute_loss(logits, label_tensors)
                loss.backward()
                if self.gradient_clip > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), max_norm=self.gradient_clip
                    )
                self.optimizer.step()

            total_loss += loss.item()

            # Per-task accuracy
            for task_name in self.task_names:
                if task_name in logits and task_name in label_tensors:
                    t_labels = label_tensors[task_name]
                    mask = t_labels >= 0
                    if mask.sum() > 0:
                        preds = logits[task_name][mask].argmax(dim=1)
                        task_correct[task_name] += (preds == t_labels[mask]).sum().item()
                        task_total[task_name] += mask.sum().item()

        metrics = {'loss': total_loss / len(loader)}
        for task_name in self.task_names:
            metrics[f'{task_name}_acc'] = (
                task_correct[task_name] / max(task_total[task_name], 1)
            )
        return metrics

    def _validate(self, loader: DataLoader) -> Dict[str, float]:
        """Validate the model."""
        self.model.eval()
        total_loss = 0.0
        task_correct = {t: 0 for t in self.task_names}
        task_total = {t: 0 for t in self.task_names}

        with torch.no_grad():
            for volumes, labels in loader:
                volumes = volumes.to(self.device)
                label_tensors = self._parse_labels(labels)

                logits = self.model(volumes)
                loss = self._compute_loss(logits, label_tensors)
                total_loss += loss.item()

                for task_name in self.task_names:
                    if task_name in logits and task_name in label_tensors:
                        t_labels = label_tensors[task_name]
                        mask = t_labels >= 0
                        if mask.sum() > 0:
                            preds = logits[task_name][mask].argmax(dim=1)
                            task_correct[task_name] += (preds == t_labels[mask]).sum().item()
                            task_total[task_name] += mask.sum().item()

        metrics = {'loss': total_loss / max(len(loader), 1)}
        for task_name in self.task_names:
            metrics[f'{task_name}_acc'] = (
                task_correct[task_name] / max(task_total[task_name], 1)
            )
        return metrics

    def _print_epoch(
        self,
        epoch: int,
        total_epochs: int,
        train: Dict,
        val: Optional[Dict],
    ):
        """Print epoch progress."""
        msg = f"Epoch {epoch + 1}/{total_epochs} - loss: {train['loss']:.4f}"
        for task_name in self.task_names:
            acc_key = f'{task_name}_acc'
            msg += f" - {task_name}: {train.get(acc_key, 0):.4f}"

        if val:
            msg += f" - val_loss: {val['loss']:.4f}"
            for task_name in self.task_names:
                acc_key = f'{task_name}_acc'
                msg += f" - val_{task_name}: {val.get(acc_key, 0):.4f}"

        elapsed = self.history.epoch_times[-1] if self.history.epoch_times else 0
        msg += f" [{elapsed:.1f}s]"
        print(msg)

    def save(
        self,
        path: str,
        save_optimizer: bool = True,
        save_history: bool = True,
    ):
        """
        Save 3D model checkpoint.

        Args:
            path: Path to save checkpoint
            save_optimizer: Whether to save optimizer state
            save_history: Whether to save training history
        """
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'model_type': '3d',
            'config': {
                'variant': getattr(self.model, 'variant', 'q'),
                'tasks': getattr(self.model, 'tasks', {}),
                'input_shape': getattr(self.model, 'input_shape', (96, 96, 96)),
                'in_channels': getattr(self.model, 'in_channels', 1),
            },
            'epoch': self.current_epoch,
        }

        if save_optimizer:
            checkpoint['optimizer_state_dict'] = self.optimizer.state_dict()

        if save_history:
            checkpoint['history'] = {
                'train_loss': self.history.train_loss,
                'val_loss': self.history.val_loss,
                'task_acc': self.history.task_acc,
            }

        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(checkpoint, path)
        print(f"3D model saved to {path}")

    def load(self, path: str, load_optimizer: bool = True):
        """
        Load 3D model from checkpoint.

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

        print(f"3D model loaded from {path}")

    def evaluate(self, test_data: DataLoader) -> Dict[str, float]:
        """
        Evaluate model on test data.

        Args:
            test_data: Test DataLoader

        Returns:
            Dict with test metrics
        """
        return self._validate(test_data)
