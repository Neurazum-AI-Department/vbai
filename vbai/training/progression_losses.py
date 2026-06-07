"""
Loss functions for VbaiProgressionNet multimodal training.

Components:
  FocalLoss3Class     — multi-class focal loss for CN / MCI / AD classification
  ProgressionLoss     — BCE (will-progress) + Huber (time regression)
  InfoNCELoss         — symmetric cross-modal contrastive loss
  VbaiProgressionLoss — composite loss combining all components
"""

from __future__ import annotations
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss3Class(nn.Module):
    """
    Multi-class focal loss.

    Args:
        gamma: Focusing exponent.  0 = standard cross-entropy.
        label_smoothing: Smooth one-hot targets ∈ [0, 1).
    """

    def __init__(self, gamma: float = 1.0, label_smoothing: float = 0.05) -> None:
        super().__init__()
        self.gamma = gamma
        self.label_smoothing = label_smoothing

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits:  (B, C) raw logits.
            targets: (B,) integer class labels.
        """
        n_cls = logits.size(1)
        log_p = F.log_softmax(logits, dim=-1)
        p = log_p.exp()

        # Smooth one-hot
        one_hot = torch.zeros_like(logits).scatter_(1, targets.unsqueeze(1), 1)
        if self.label_smoothing > 0:
            one_hot = one_hot * (1 - self.label_smoothing) + self.label_smoothing / n_cls

        pt = (one_hot * p).sum(dim=-1)
        focal_w = (1 - pt) ** self.gamma
        ce = -(one_hot * log_p).sum(dim=-1)
        return (focal_w * ce).mean()


class ProgressionLoss(nn.Module):
    """
    Compound progression loss for MCI patients.

    ``loss = BCEWithLogits(will_progress) + Huber(time_to_conversion)``

    Only applied to samples where *has_progression* is True
    (i.e. MCI patients with known follow-up).
    """

    def __init__(self, huber_delta: float = 12.0) -> None:
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.huber = nn.HuberLoss(delta=huber_delta)

    def forward(
        self,
        prog_out: Dict[str, torch.Tensor],
        will_progress: torch.Tensor,
        progression_months: torch.Tensor,
        has_progression: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            prog_out:          Output dict from ProgressionHead.
            will_progress:     (B,) float labels {0.0, 1.0}.
            progression_months:(B,) months until conversion (0 if no conversion).
            has_progression:   (B,) bool mask — True for valid MCI samples.

        Returns:
            Scalar loss.  Returns 0 if no valid samples in batch.
        """
        mask = has_progression.bool()
        if not mask.any():
            return torch.tensor(0.0, device=will_progress.device, requires_grad=True)

        logits = prog_out['will_progress_logits'][mask].squeeze(-1)
        will_p = will_progress[mask]
        loss = self.bce(logits, will_p)

        converters = mask & (will_progress > 0.5)
        if converters.any():
            pred_time = prog_out['time_to_conversion'][converters].squeeze(-1)
            true_time = progression_months[converters]
            loss = loss + self.huber(pred_time, true_time)

        return loss


class InfoNCELoss(nn.Module):
    """
    Symmetric InfoNCE contrastive loss (both zm→zt and zt→zm directions).

    Aligns MRI and tabular embedding spaces so that paired samples form
    a common representation.

    Args:
        temperature: Softmax temperature (lower = sharper).
    """

    def __init__(self, temperature: float = 0.1) -> None:
        super().__init__()
        self.tau = temperature

    def forward(
        self, zm: torch.Tensor, zt: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            zm: (B, D) L2-normalised MRI embeddings.
            zt: (B, D) L2-normalised tabular embeddings.
        """
        b = zm.size(0)
        labels = torch.arange(b, device=zm.device)
        sim = torch.mm(zm, zt.t()) / self.tau
        loss = 0.5 * (
            F.cross_entropy(sim, labels) + F.cross_entropy(sim.t(), labels)
        )
        return loss


class VbaiProgressionLoss(nn.Module):
    """
    Composite training loss for VbaiProgressionNet.

    Components and default weights:
      w_fused (1.0) × focal(fused_logits, labels)
      w_mri   (0.3) × focal(mri_logits,   labels)   — auxiliary
      w_tab   (0.3) × focal(tab_logits,   labels)   — auxiliary
      w_prog  (0.5) × ProgressionLoss                — MCI only
      w_cont  (0.2) × InfoNCELoss                    — cross-modal alignment

    Only the loss terms present in *outputs* are computed (graceful fallback
    when operating with a single modality).
    """

    def __init__(
        self,
        w_fused: float = 1.0,
        w_mri: float = 0.3,
        w_tab: float = 0.3,
        w_prog: float = 0.5,
        w_contrastive: float = 0.2,
        focal_gamma: float = 1.0,
        label_smoothing: float = 0.05,
    ) -> None:
        super().__init__()
        self.focal = FocalLoss3Class(focal_gamma, label_smoothing)
        self.prog_loss = ProgressionLoss()
        self.nce = InfoNCELoss()

        self.w_fused = w_fused
        self.w_mri = w_mri
        self.w_tab = w_tab
        self.w_prog = w_prog
        self.w_cont = w_contrastive

    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            outputs: Model output dict from VbaiProgressionNet.forward().
            targets: dict with keys:
                'labels'             (B,) int class indices
                'has_progression'    (B,) bool
                'will_progress'      (B,) float {0,1}
                'progression_months' (B,) float

        Returns:
            dict with 'total' and individual component losses.
        """
        labels = targets['labels']
        losses: Dict[str, torch.Tensor] = {}

        if 'fused_logits' in outputs:
            losses['cls_fused'] = self.w_fused * self.focal(outputs['fused_logits'], labels)

        if 'mri_logits' in outputs:
            losses['cls_mri'] = self.w_mri * self.focal(outputs['mri_logits'], labels)

        if 'tab_logits' in outputs:
            losses['cls_tab'] = self.w_tab * self.focal(outputs['tab_logits'], labels)

        if 'progression' in outputs:
            prog_l = self.prog_loss(
                outputs['progression'],
                targets.get('will_progress', torch.zeros_like(labels, dtype=torch.float)),
                targets.get('progression_months', torch.zeros_like(labels, dtype=torch.float)),
                targets.get('has_progression', torch.zeros_like(labels, dtype=torch.bool)),
            )
            losses['progression'] = self.w_prog * prog_l

        if 'zm' in outputs and 'zt' in outputs and self.w_cont > 0:
            losses['contrastive'] = self.w_cont * self.nce(outputs['zm'], outputs['zt'])

        total = sum(losses.values())
        losses['total'] = total
        return losses
