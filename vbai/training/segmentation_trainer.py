"""
SegmentationTrainer — training loop for VbaiSegNet3D.

Features:
  - Mixed-precision training (AMP)
  - Gradient clipping
  - Deep supervision support
  - Early stopping and model checkpointing
  - Per-epoch Dice metric computation
  - TensorBoard logging (optional)
"""

from __future__ import annotations
import os
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .segmentation_losses import DeepSupervisionLoss, TumorSegmentationLoss


# ──────────────────────────────────────────────────────────────────────────────
# Metrics
# ──────────────────────────────────────────────────────────────────────────────

def batch_dice(logits: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5, smooth: float = 1e-5) -> torch.Tensor:
    """Compute mean Dice score over a batch (on GPU)."""
    preds = (torch.sigmoid(logits) > threshold).float()
    intersection = (preds * targets).sum(dim=list(range(1, preds.ndim)))
    denom = preds.sum(dim=list(range(1, preds.ndim))) + targets.sum(dim=list(range(1, targets.ndim)))
    return ((2 * intersection + smooth) / (denom + smooth)).mean()


def _fit_status_seg(train_dice: float, val_dice: float) -> str:
    """Diagnose underfitting / overfitting based on train vs val Dice gap."""
    gap = train_dice - val_dice  # positive => train >> val (overfitting)
    if train_dice < 0.35:
        return "Underfitting"
    if gap > 0.15:
        return "Overfitting"
    if gap > 0.07:
        return "Slight Overfitting"
    if val_dice < 0.45 and gap < 0.03:
        return "Slight Underfitting"
    return "Good Fit"


# ──────────────────────────────────────────────────────────────────────────────
# History
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class SegmentationHistory:
    train_loss: List[float] = field(default_factory=list)
    val_loss: List[float] = field(default_factory=list)
    train_dice: List[float] = field(default_factory=list)
    val_dice: List[float] = field(default_factory=list)
    lr_history: List[float] = field(default_factory=list)
    best_epoch: int = 0
    best_val_dice: float = 0.0


# ──────────────────────────────────────────────────────────────────────────────
# Trainer
# ──────────────────────────────────────────────────────────────────────────────

class SegmentationTrainer:
    """
    Trainer for VbaiSegNet3D.

    Args:
        model: VbaiSegNet3D instance.
        criterion: Loss module.  If None, defaults to TumorSegmentationLoss
                   wrapped in DeepSupervisionLoss.
        optimizer: If None, AdamW with lr=1e-4, wd=1e-5 is created.
        scheduler: LR scheduler.  If None, CosineAnnealingWarmRestarts is used.
        device: 'cuda', 'cpu', or 'auto'.
        use_amp: Enable mixed-precision training.
        grad_clip: Maximum gradient norm (0 disables clipping).
        early_stopping_patience: Stop if val Dice does not improve for N epochs
                                  (0 disables).
        checkpoint_dir: Directory to save checkpoints.
        checkpoint_name: Base filename for checkpoints.
        log_interval: Print progress every N batches.
        tensorboard: Enable TensorBoard logging.
        tensorboard_dir: TensorBoard log directory.

    Example::

        from vbai.models import VbaiSegNet3D
        from vbai.training import SegmentationTrainer

        model = VbaiSegNet3D(in_channels=2, out_channels=1)
        trainer = SegmentationTrainer(model, device='cuda')
        history = trainer.fit(train_loader, val_loader, epochs=50)
    """

    def __init__(
        self,
        model: nn.Module,
        criterion: Optional[nn.Module] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        device: str = 'auto',
        use_amp: bool = True,
        grad_clip: float = 1.0,
        early_stopping_patience: int = 15,
        checkpoint_dir: str = './checkpoints',
        checkpoint_name: str = 'vbai_seg',
        log_interval: int = 10,
        tensorboard: bool = False,
        tensorboard_dir: str = './logs',
    ) -> None:
        if device == 'auto':
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = torch.device(device)
        self.model = model.to(self.device)

        if criterion is None:
            criterion = DeepSupervisionLoss(TumorSegmentationLoss())
        self.criterion = criterion

        if optimizer is None:
            optimizer = torch.optim.AdamW(
                model.parameters(), lr=1e-4, weight_decay=1e-5
            )
        self.optimizer = optimizer

        if scheduler is None:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer, T_0=20, T_mult=2, eta_min=1e-7
            )
        self.scheduler = scheduler

        self.use_amp = use_amp and self.device.type == 'cuda'
        self.scaler = torch.amp.GradScaler('cuda', enabled=self.use_amp)
        self.grad_clip = grad_clip
        self.es_patience = early_stopping_patience
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_name = checkpoint_name
        self.log_interval = log_interval

        self.writer = None
        if tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter
                os.makedirs(tensorboard_dir, exist_ok=True)
                self.writer = SummaryWriter(tensorboard_dir)
            except ImportError:
                pass

        os.makedirs(checkpoint_dir, exist_ok=True)

    # ── Single epoch ──────────────────────────────────────────────────────────

    def _train_epoch(self, loader: DataLoader, epoch: int) -> Dict[str, float]:
        self.model.train()
        total_loss = total_dice = n = 0.0

        for i, batch in enumerate(loader):
            images = batch['image'].to(self.device, non_blocking=True)
            masks = batch['mask'].to(self.device, non_blocking=True)

            self.optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast('cuda', enabled=self.use_amp):
                use_ds = getattr(self.model, 'use_deep_supervision', False)
                if use_ds:
                    logits, aux = self.model(images, return_aux=True)
                    loss = self.criterion(logits, masks, aux)
                else:
                    logits = self.model(images)
                    aux = None
                    if hasattr(self.criterion, 'forward'):
                        import inspect
                        sig = inspect.signature(self.criterion.forward)
                        if len(sig.parameters) == 3:
                            loss = self.criterion(logits, masks, aux)
                        else:
                            loss = self.criterion(logits, masks)
                    else:
                        loss = self.criterion(logits, masks)

            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            if self.grad_clip > 0:
                nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            self.scaler.step(self.optimizer)
            self.scaler.update()

            dice = batch_dice(logits.detach(), masks).item()
            total_loss += loss.item()
            total_dice += dice
            n += 1

            if self.log_interval > 0 and (i + 1) % self.log_interval == 0:
                print(
                    f'  [Train] Epoch {epoch} | Batch {i+1}/{len(loader)} | '
                    f'Loss {loss.item():.4f} | Dice {dice:.4f}'
                )

        return {'loss': total_loss / max(n, 1), 'dice': total_dice / max(n, 1)}

    @torch.no_grad()
    def _val_epoch(self, loader: DataLoader) -> Dict[str, float]:
        self.model.eval()
        total_loss = total_dice = n = 0.0

        for batch in loader:
            images = batch['image'].to(self.device, non_blocking=True)
            masks = batch['mask'].to(self.device, non_blocking=True)

            with torch.amp.autocast('cuda', enabled=self.use_amp):
                use_ds = getattr(self.model, 'use_deep_supervision', False)
                if use_ds:
                    logits, aux = self.model(images, return_aux=True)
                    loss = self.criterion(logits, masks, aux)
                else:
                    logits = self.model(images)
                    loss_fn = getattr(self.criterion, 'base_loss', self.criterion)
                    loss = loss_fn(logits, masks)

            dice = batch_dice(logits, masks).item()
            total_loss += loss.item()
            total_dice += dice
            n += 1

        return {'loss': total_loss / max(n, 1), 'dice': total_dice / max(n, 1)}

    # ── Main training loop ────────────────────────────────────────────────────

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        epochs: int = 100,
        verbose: int = 1,
    ) -> SegmentationHistory:
        """
        Train the model.

        Args:
            train_loader: Training DataLoader (yields {'image', 'mask'}).
            val_loader:   Validation DataLoader.
            epochs:       Maximum training epochs.
            verbose:      0 = silent, 1 = per-epoch, 2 = per-batch.

        Returns:
            SegmentationHistory with recorded metrics.
        """
        history = SegmentationHistory()
        best_dice = 0.0
        no_improve = 0

        for epoch in range(1, epochs + 1):
            t0 = time.time()
            if verbose > 1:
                self.log_interval = 10
            else:
                self.log_interval = 0

            train_m = self._train_epoch(train_loader, epoch)
            history.train_loss.append(train_m['loss'])
            history.train_dice.append(train_m['dice'])

            val_m: Dict[str, float] = {}
            if val_loader is not None:
                val_m = self._val_epoch(val_loader)
                history.val_loss.append(val_m['loss'])
                history.val_dice.append(val_m['dice'])

            lr = self.optimizer.param_groups[0]['lr']
            history.lr_history.append(lr)

            if self.scheduler:
                self.scheduler.step()

            monitor_dice = val_m.get('dice', train_m['dice'])
            is_best = monitor_dice > best_dice
            if is_best:
                best_dice = monitor_dice
                history.best_val_dice = best_dice
                history.best_epoch = epoch
                no_improve = 0
                self._save_checkpoint(epoch, best_dice, is_best=True)
            else:
                no_improve += 1

            if verbose >= 1:
                elapsed = time.time() - t0
                val_str = (
                    f" | Val Loss {val_m['loss']:.4f} | Val Dice {val_m['dice']:.4f}"
                    if val_m else ''
                )
                status = _fit_status_seg(
                    train_m['dice'],
                    val_m.get('dice', train_m['dice']),
                )
                print(
                    f'Epoch {epoch:03d}/{epochs} | '
                    f'Train Loss {train_m["loss"]:.4f} | Train Dice {train_m["dice"]:.4f}'
                    f'{val_str} | LR {lr:.2e} | {elapsed:.1f}s'
                    + (' [best]' if is_best else '')
                    + f' | {status}'
                )

            if self.writer:
                self.writer.add_scalars('Loss', {'train': train_m['loss'], **(
                    {'val': val_m['loss']} if val_m else {})}, epoch)
                self.writer.add_scalars('Dice', {'train': train_m['dice'], **(
                    {'val': val_m['dice']} if val_m else {})}, epoch)
                self.writer.add_scalar('LR', lr, epoch)

            if self.es_patience > 0 and no_improve >= self.es_patience:
                if verbose >= 1:
                    print(f'Early stopping at epoch {epoch} (no improvement for {self.es_patience} epochs).')
                break

        return history

    # ── Checkpoint helpers ────────────────────────────────────────────────────

    def _save_checkpoint(self, epoch: int, metric: float, is_best: bool = False) -> None:
        suffix = 'best' if is_best else f'epoch{epoch:03d}'
        path = os.path.join(self.checkpoint_dir, f'{self.checkpoint_name}_{suffix}.pth')
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_metric': metric,
        }, path)

    def save(self, path: str) -> None:
        """Save model state dict."""
        torch.save(self.model.state_dict(), path)

    def load(self, path: str) -> None:
        """Load model state dict."""
        self.model.load_state_dict(
            torch.load(path, map_location=self.device, weights_only=False)
        )
