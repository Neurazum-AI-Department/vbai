"""
ProgressionTrainer — 3-phase training loop for VbaiProgressionNet.

Phase 1 — MRI encoder pretrain (MRI only, 40 epochs)
Phase 2 — Tabular encoder pretrain (biomarkers only, 60 epochs)
Phase 3 — Joint fusion training with modality dropout (both, 40 epochs)

Each phase uses a dedicated DataLoader and loss weighting schedule.
Differential learning rates in Phase 3 keep the pretrained encoders stable
while the fusion module and classification heads are trained aggressively.
"""

from __future__ import annotations
import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .progression_losses import VbaiProgressionLoss


# ──────────────────────────────────────────────────────────────────────────────
# History
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class ProgressionPhaseHistory:
    phase: int
    train_loss: List[float] = field(default_factory=list)
    val_loss: List[float] = field(default_factory=list)
    train_acc: List[float] = field(default_factory=list)
    val_acc: List[float] = field(default_factory=list)
    lr_history: List[float] = field(default_factory=list)
    best_epoch: int = 0
    best_val_acc: float = 0.0


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    return (logits.argmax(dim=-1) == labels).float().mean().item()


def _fit_status_prog(train_acc: float, val_acc: float) -> str:
    """Diagnose underfitting / overfitting based on train vs val accuracy gap."""
    gap = train_acc - val_acc  # positive => train >> val (overfitting)
    if train_acc < 0.45:
        return "Underfitting"
    if gap > 0.20:
        return "Overfitting"
    if gap > 0.10:
        return "Slight Overfitting"
    if val_acc < 0.55 and gap < 0.03:
        return "Slight Underfitting"
    return "Good Fit"


def _make_optimizer(
    param_groups: list,
    lr: float,
    weight_decay: float = 1e-4,
) -> torch.optim.Optimizer:
    return torch.optim.AdamW(param_groups, lr=lr, weight_decay=weight_decay)


def _make_scheduler(
    optimizer: torch.optim.Optimizer,
    epochs: int,
) -> torch.optim.lr_scheduler.LRScheduler:
    return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-7)


# ──────────────────────────────────────────────────────────────────────────────
# Trainer
# ──────────────────────────────────────────────────────────────────────────────

class ProgressionTrainer:
    """
    3-phase trainer for VbaiProgressionNet.

    Args:
        model: VbaiProgressionNet instance.
        device: 'cuda', 'cpu', or 'auto'.
        use_amp: Enable AMP mixed precision.
        grad_clip: Max gradient norm (0 to disable).
        checkpoint_dir: Directory for saved checkpoints.
        checkpoint_name: Base filename for checkpoints.
        tensorboard: Enable TensorBoard logging.
        tensorboard_dir: TensorBoard log directory.
        verbose: 0 = silent, 1 = per-epoch.

    Example::

        from vbai.models import VbaiProgressionNet
        from vbai.training import ProgressionTrainer

        model = VbaiProgressionNet()
        trainer = ProgressionTrainer(model, device='cuda')

        h1 = trainer.fit_phase1(mri_loader, mri_val_loader)
        h2 = trainer.fit_phase2(tab_loader, tab_val_loader)
        h3 = trainer.fit_phase3(full_loader, full_val_loader)
    """

    def __init__(
        self,
        model: nn.Module,
        device: str = 'auto',
        use_amp: bool = True,
        grad_clip: float = 1.0,
        checkpoint_dir: str = './checkpoints',
        checkpoint_name: str = 'vbai_prog',
        tensorboard: bool = False,
        tensorboard_dir: str = './logs',
        verbose: int = 1,
    ) -> None:
        if device == 'auto':
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.use_amp = use_amp and self.device.type == 'cuda'
        self.grad_clip = grad_clip
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_name = checkpoint_name
        self.verbose = verbose

        os.makedirs(checkpoint_dir, exist_ok=True)

        self.writer = None
        if tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter
                os.makedirs(tensorboard_dir, exist_ok=True)
                self.writer = SummaryWriter(tensorboard_dir)
            except ImportError:
                pass

    # ── Generic training / validation passes ──────────────────────────────────

    def _train_epoch(
        self,
        loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        scaler: torch.amp.GradScaler,
        criterion: VbaiProgressionLoss,
        mode: str,  # 'mri', 'tab', 'multi'
        epoch: int,
        total_epochs: int,
    ) -> Dict[str, float]:
        self.model.train()
        total_loss = total_acc = n = 0.0

        for batch in loader:
            mri, tab = self._unpack_batch(batch, mode)
            labels = batch['labels'].to(self.device)

            targets = {
                'labels': labels,
                'has_progression': batch.get('has_progression', torch.zeros(len(labels), dtype=torch.bool)).to(self.device),
                'will_progress': batch.get('will_progress', torch.zeros(len(labels))).to(self.device),
                'progression_months': batch.get('progression_months', torch.zeros(len(labels))).to(self.device),
            }

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda', enabled=self.use_amp):
                outputs = self.model(mri=mri, tab=tab)
                losses = criterion(outputs, targets)
                loss = losses['total']

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            if self.grad_clip > 0:
                nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            scaler.step(optimizer)
            scaler.update()

            acc = _accuracy(outputs['fused_logits'].detach(), labels)
            total_loss += loss.item()
            total_acc += acc
            n += 1

        return {'loss': total_loss / max(n, 1), 'acc': total_acc / max(n, 1)}

    @torch.no_grad()
    def _val_epoch(
        self,
        loader: DataLoader,
        criterion: VbaiProgressionLoss,
        mode: str,
    ) -> Dict[str, float]:
        self.model.eval()
        total_loss = total_acc = n = 0.0

        for batch in loader:
            mri, tab = self._unpack_batch(batch, mode)
            labels = batch['labels'].to(self.device)
            targets = {
                'labels': labels,
                'has_progression': batch.get('has_progression', torch.zeros(len(labels), dtype=torch.bool)).to(self.device),
                'will_progress': batch.get('will_progress', torch.zeros(len(labels))).to(self.device),
                'progression_months': batch.get('progression_months', torch.zeros(len(labels))).to(self.device),
            }

            with torch.amp.autocast('cuda', enabled=self.use_amp):
                outputs = self.model(mri=mri, tab=tab)
                losses = criterion(outputs, targets)

            acc = _accuracy(outputs['fused_logits'], labels)
            total_loss += losses['total'].item()
            total_acc += acc
            n += 1

        return {'loss': total_loss / max(n, 1), 'acc': total_acc / max(n, 1)}

    def _unpack_batch(self, batch: Dict, mode: str):
        mri = batch.get('mri', None)
        tab = batch.get('tab', None)

        if mri is not None:
            mri = mri.to(self.device, non_blocking=True)
        if tab is not None:
            tab = tab.to(self.device, non_blocking=True)

        if mode == 'mri':
            tab = None
        elif mode == 'tab':
            mri = None

        return mri, tab

    # ── Phase runners ─────────────────────────────────────────────────────────

    def _run_phase(
        self,
        phase: int,
        mode: str,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader],
        epochs: int,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler,
        criterion: VbaiProgressionLoss,
        es_patience: int = 20,
        min_epochs_es: int = 25,
    ) -> ProgressionPhaseHistory:
        hist = ProgressionPhaseHistory(phase=phase)
        scaler = torch.amp.GradScaler('cuda', enabled=self.use_amp)
        best_acc = 0.0
        no_improve = 0

        for epoch in range(1, epochs + 1):
            t0 = time.time()
            train_m = self._train_epoch(train_loader, optimizer, scaler, criterion, mode, epoch, epochs)
            hist.train_loss.append(train_m['loss'])
            hist.train_acc.append(train_m['acc'])

            val_m: Dict[str, float] = {}
            if val_loader:
                val_m = self._val_epoch(val_loader, criterion, mode)
                hist.val_loss.append(val_m['loss'])
                hist.val_acc.append(val_m['acc'])

            scheduler.step()
            lr = optimizer.param_groups[0]['lr']
            hist.lr_history.append(lr)

            monitor = val_m.get('acc', train_m['acc'])
            is_best = monitor > best_acc
            if is_best:
                best_acc = monitor
                hist.best_val_acc = best_acc
                hist.best_epoch = epoch
                no_improve = 0
                self._save(phase, epoch, best_acc)
            else:
                no_improve += 1

            if self.verbose >= 1:
                elapsed = time.time() - t0
                val_str = (
                    f" | Val Loss {val_m['loss']:.4f} | Val Acc {val_m['acc']:.4f}"
                    if val_m else ''
                )
                status = _fit_status_prog(
                    train_m['acc'],
                    val_m.get('acc', train_m['acc']),
                )
                print(
                    f'[Phase {phase}] Epoch {epoch:03d}/{epochs} | '
                    f'Train Loss {train_m["loss"]:.4f} | Train Acc {train_m["acc"]:.4f}'
                    f'{val_str} | LR {lr:.2e} | {elapsed:.1f}s'
                    + (' [best]' if is_best else '')
                    + f' | {status}'
                )

            if self.writer:
                self.writer.add_scalars(f'Phase{phase}/Loss', {'train': train_m['loss'], **(
                    {'val': val_m['loss']} if val_m else {})}, epoch)
                self.writer.add_scalars(f'Phase{phase}/Acc', {'train': train_m['acc'], **(
                    {'val': val_m['acc']} if val_m else {})}, epoch)

            if epoch >= min_epochs_es and es_patience > 0 and no_improve >= es_patience:
                if self.verbose >= 1:
                    print(f'[Phase {phase}] Early stopping at epoch {epoch}.')
                break

        return hist

    # ── Public phase API ──────────────────────────────────────────────────────

    def fit_phase1(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        epochs: int = 40,
        lr: float = 3e-4,
        weight_decay: float = 1e-4,
        es_patience: int = 20,
    ) -> ProgressionPhaseHistory:
        """
        Phase 1: Pretrain MRI encoder with classification signal only.

        DataLoader must yield batches with 'mri' and 'labels' keys.
        """
        if self.verbose >= 1:
            print('\n=== Phase 1: MRI Encoder Pretraining ===')

        params = list(self.model.mri_encoder.parameters()) + list(self.model.mri_cls.parameters())
        optimizer = _make_optimizer([{'params': params, 'lr': lr}], lr, weight_decay)
        scheduler = _make_scheduler(optimizer, epochs)
        criterion = VbaiProgressionLoss(w_fused=1.0, w_mri=0.0, w_tab=0.0, w_prog=0.0, w_contrastive=0.0)

        return self._run_phase(1, 'mri', train_loader, val_loader, epochs, optimizer, scheduler, criterion, es_patience, 25)

    def fit_phase2(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        epochs: int = 60,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        es_patience: int = 20,
    ) -> ProgressionPhaseHistory:
        """
        Phase 2: Pretrain tabular encoder with classification signal only.

        DataLoader must yield batches with 'tab' and 'labels' keys.
        """
        if self.verbose >= 1:
            print('\n=== Phase 2: Tabular Encoder Pretraining ===')

        params = list(self.model.tab_encoder.parameters()) + list(self.model.tab_cls.parameters())
        optimizer = _make_optimizer([{'params': params, 'lr': lr}], lr, weight_decay)
        scheduler = _make_scheduler(optimizer, epochs)
        criterion = VbaiProgressionLoss(w_fused=1.0, w_mri=0.0, w_tab=0.0, w_prog=0.0, w_contrastive=0.0)

        return self._run_phase(2, 'tab', train_loader, val_loader, epochs, optimizer, scheduler, criterion, es_patience, 30)

    def fit_phase3(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        epochs: int = 40,
        lr_backbone: float = 1e-5,
        lr_fusion: float = 5e-4,
        weight_decay: float = 1e-4,
        w_fused: float = 1.0,
        w_mri: float = 0.3,
        w_tab: float = 0.3,
        w_prog: float = 0.5,
        w_contrastive: float = 0.2,
        es_patience: int = 20,
    ) -> ProgressionPhaseHistory:
        """
        Phase 3: Joint fusion training with differential learning rates.

        Encoder backbone: lr_backbone (keep stable)
        Fusion + heads:   lr_fusion   (train aggressively)

        DataLoader should yield batches with 'mri', 'tab', 'labels', and
        optionally 'has_progression', 'will_progress', 'progression_months'.
        """
        if self.verbose >= 1:
            print('\n=== Phase 3: Joint Fusion Training ===')

        backbone_params = (
            list(self.model.mri_encoder.parameters())
            + list(self.model.tab_encoder.parameters())
        )
        fusion_params = (
            list(self.model.fusion.parameters())
            + list(self.model.fused_cls.parameters())
            + list(self.model.mri_cls.parameters())
            + list(self.model.tab_cls.parameters())
            + list(self.model.progression.parameters())
            + list(self.model.contrast_mri.parameters())
            + list(self.model.contrast_tab.parameters())
        )
        param_groups = [
            {'params': backbone_params, 'lr': lr_backbone},
            {'params': fusion_params, 'lr': lr_fusion},
        ]
        optimizer = _make_optimizer(param_groups, lr_fusion, weight_decay)
        scheduler = _make_scheduler(optimizer, epochs)
        criterion = VbaiProgressionLoss(w_fused, w_mri, w_tab, w_prog, w_contrastive)

        return self._run_phase(3, 'multi', train_loader, val_loader, epochs, optimizer, scheduler, criterion, es_patience, 25)

    # ── Convenience: train all phases ─────────────────────────────────────────

    def fit(
        self,
        mri_loader: DataLoader,
        tab_loader: DataLoader,
        full_loader: DataLoader,
        mri_val_loader: Optional[DataLoader] = None,
        tab_val_loader: Optional[DataLoader] = None,
        full_val_loader: Optional[DataLoader] = None,
    ) -> List[ProgressionPhaseHistory]:
        """Run all 3 phases sequentially and return their histories."""
        h1 = self.fit_phase1(mri_loader, mri_val_loader)
        h2 = self.fit_phase2(tab_loader, tab_val_loader)
        h3 = self.fit_phase3(full_loader, full_val_loader)
        return [h1, h2, h3]

    # ── Checkpoint ────────────────────────────────────────────────────────────

    def _save(self, phase: int, epoch: int, metric: float) -> None:
        path = os.path.join(
            self.checkpoint_dir,
            f'{self.checkpoint_name}_phase{phase}_best.pth',
        )
        torch.save({
            'epoch': epoch,
            'phase': phase,
            'model_state_dict': self.model.state_dict(),
            'best_metric': metric,
        }, path)

    def save(self, path: str) -> None:
        """Save full model state dict."""
        torch.save(self.model.state_dict(), path)

    def load(self, path: str) -> None:
        """Load model state dict."""
        self.model.load_state_dict(
            torch.load(path, map_location=self.device, weights_only=False)
        )
