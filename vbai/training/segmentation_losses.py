"""
Loss functions for 3D medical image segmentation.

Available:
  DiceLoss               — binary Dice (sigmoid-based)
  MulticlassDiceLoss     — macro-averaged multi-class Dice
  FocalLoss              — binary focal loss (α, γ)
  TumorSegmentationLoss  — 0.6 × Dice + 0.4 × Focal  (binary tumour task)
  TissueSegmentationLoss — 0.7 × Dice + 0.3 × MSE    (tissue probability task)
  DeepSupervisionLoss    — wraps any base loss with exponentially-weighted
                           auxiliary decoder outputs
"""

from __future__ import annotations
from typing import Callable, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    """
    Soft Dice loss for binary segmentation.

    Expects raw logits (before sigmoid).
    """

    def __init__(self, smooth: float = 1e-5) -> None:
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        preds = torch.sigmoid(logits)
        intersection = (preds * targets).sum()
        return 1 - (2 * intersection + self.smooth) / (
            preds.sum() + targets.sum() + self.smooth
        )


class MulticlassDiceLoss(nn.Module):
    """
    Macro-averaged Dice loss for C-channel probabilistic targets.

    Each channel is treated independently (sigmoid, not softmax).
    Targets should be float tensors in [0, 1].
    """

    def __init__(self, smooth: float = 1e-5) -> None:
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        preds = torch.sigmoid(logits)
        dice_per_ch = []
        for c in range(logits.shape[1]):
            p = preds[:, c]
            t = targets[:, c]
            inter = (p * t).sum()
            dice_c = (2 * inter + self.smooth) / (p.sum() + t.sum() + self.smooth)
            dice_per_ch.append(1 - dice_c)
        return torch.stack(dice_per_ch).mean()


class FocalLoss(nn.Module):
    """
    Binary focal loss (logits input).

    Args:
        alpha: Balancing factor for positive class.
        gamma: Focusing exponent (0 = BCEWithLogits).
    """

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0) -> None:
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        p = torch.sigmoid(logits)
        pt = targets * p + (1 - targets) * (1 - p)
        focal_w = self.alpha * (1 - pt) ** self.gamma
        return (focal_w * bce).mean()


class TumorSegmentationLoss(nn.Module):
    """
    Combined loss for binary tumour segmentation.

    ``loss = w_dice * DiceLoss + w_focal * FocalLoss``

    Default: 60 % Dice + 40 % Focal.
    """

    def __init__(
        self,
        w_dice: float = 0.6,
        w_focal: float = 0.4,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
    ) -> None:
        super().__init__()
        self.dice = DiceLoss()
        self.focal = FocalLoss(focal_alpha, focal_gamma)
        self.w_dice = w_dice
        self.w_focal = w_focal

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self.w_dice * self.dice(logits, targets) + self.w_focal * self.focal(logits, targets)


class TissueSegmentationLoss(nn.Module):
    """
    Combined loss for multi-class tissue probability segmentation.

    ``loss = w_dice * MulticlassDiceLoss + w_mse * MSELoss``

    Default: 70 % Dice + 30 % MSE.
    Soft (probabilistic) targets in [0, 1] are expected.
    """

    def __init__(self, w_dice: float = 0.7, w_mse: float = 0.3) -> None:
        super().__init__()
        self.dice = MulticlassDiceLoss()
        self.mse = nn.MSELoss()
        self.w_dice = w_dice
        self.w_mse = w_mse

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        preds = torch.sigmoid(logits)
        return self.w_dice * self.dice(logits, targets) + self.w_mse * self.mse(preds, targets)


class DeepSupervisionLoss(nn.Module):
    """
    Wraps a base segmentation loss with exponentially-weighted auxiliary outputs.

    During training the model can emit auxiliary logits from intermediate
    decoder levels.  Each auxiliary output is upsampled to the target
    resolution and penalised with decreasing weight.

    Args:
        base_loss: Any segmentation loss callable (logits, targets) → scalar.
        weights: Weights for [main, aux1, aux2, ...].
                 Defaults to [1.0, 0.5, 0.25, 0.125].
    """

    def __init__(
        self,
        base_loss: nn.Module,
        weights: Optional[List[float]] = None,
    ) -> None:
        super().__init__()
        self.base_loss = base_loss
        self.weights = weights or [1.0, 0.5, 0.25, 0.125]

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        aux_logits: Optional[List[torch.Tensor]] = None,
    ) -> torch.Tensor:
        loss = self.weights[0] * self.base_loss(logits, targets)

        if aux_logits:
            for i, aux in enumerate(aux_logits):
                w = self.weights[i + 1] if i + 1 < len(self.weights) else self.weights[-1]
                aux_up = F.interpolate(
                    aux, size=targets.shape[2:], mode='trilinear', align_corners=False
                )
                loss = loss + w * self.base_loss(aux_up, targets)

        return loss
