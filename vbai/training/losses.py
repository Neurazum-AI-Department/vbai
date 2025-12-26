"""
Loss Functions for Vbai Models
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class MultiTaskLoss(nn.Module):
    """
    Multi-task loss for simultaneous dementia and tumor classification.
    
    Combines CrossEntropyLoss for both tasks with configurable weights.
    Uses ignore_index to handle samples where only one label is available.
    
    Args:
        dementia_weight: Weight for dementia loss (default: 1.0)
        tumor_weight: Weight for tumor loss (default: 1.0)
        ignore_index: Label value to ignore (default: -1)
        label_smoothing: Label smoothing factor (default: 0.0)
    """
    
    def __init__(
        self,
        dementia_weight: float = 1.0,
        tumor_weight: float = 1.0,
        ignore_index: int = -1,
        label_smoothing: float = 0.0
    ):
        super().__init__()
        
        self.dementia_weight = dementia_weight
        self.tumor_weight = tumor_weight
        
        self.dementia_loss = nn.CrossEntropyLoss(
            ignore_index=ignore_index,
            label_smoothing=label_smoothing
        )
        self.tumor_loss = nn.CrossEntropyLoss(
            ignore_index=ignore_index,
            label_smoothing=label_smoothing
        )
    
    def forward(
        self,
        dementia_logits: torch.Tensor,
        tumor_logits: torch.Tensor,
        dementia_labels: torch.Tensor,
        tumor_labels: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute combined loss.

        Args:
            dementia_logits: Model predictions for dementia (B, num_dementia_classes) or None
            tumor_logits: Model predictions for tumor (B, num_tumor_classes) or None
            dementia_labels: Ground truth dementia labels (B,)
            tumor_labels: Ground truth tumor labels (B,)

        Returns:
            Combined weighted loss
        """
        # Handle None logits for single-task models
        if dementia_logits is not None:
            d_loss = self.dementia_loss(dementia_logits, dementia_labels)
            # Handle NaN when all labels are ignored
            if torch.isnan(d_loss):
                d_loss = torch.tensor(0.0, device=dementia_logits.device)
        else:
            d_loss = torch.tensor(0.0, device=tumor_logits.device if tumor_logits is not None else 'cpu')

        if tumor_logits is not None:
            t_loss = self.tumor_loss(tumor_logits, tumor_labels)
            # Handle NaN when all labels are ignored
            if torch.isnan(t_loss):
                t_loss = torch.tensor(0.0, device=tumor_logits.device)
        else:
            t_loss = torch.tensor(0.0, device=dementia_logits.device if dementia_logits is not None else 'cpu')

        return self.dementia_weight * d_loss + self.tumor_weight * t_loss


class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance.
    
    Reference: https://arxiv.org/abs/1708.02002
    
    Args:
        alpha: Class weights (optional)
        gamma: Focusing parameter (default: 2.0)
        ignore_index: Label to ignore (default: -1)
    """
    
    def __init__(
        self,
        alpha: Optional[torch.Tensor] = None,
        gamma: float = 2.0,
        ignore_index: int = -1
    ):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index
    
    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Compute focal loss."""
        # Filter ignored labels
        mask = labels != self.ignore_index
        if mask.sum() == 0:
            return torch.tensor(0.0, device=logits.device)
        
        logits = logits[mask]
        labels = labels[mask]
        
        ce_loss = F.cross_entropy(logits, labels, reduction='none')
        pt = torch.exp(-ce_loss)
        
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        
        if self.alpha is not None:
            alpha = self.alpha.to(logits.device)
            alpha_t = alpha[labels]
            focal_loss = alpha_t * focal_loss
        
        return focal_loss.mean()


class MultiTaskFocalLoss(nn.Module):
    """
    Multi-task version of Focal Loss.
    
    Args:
        dementia_weight: Weight for dementia loss
        tumor_weight: Weight for tumor loss
        gamma: Focusing parameter
        ignore_index: Label to ignore
    """
    
    def __init__(
        self,
        dementia_weight: float = 1.0,
        tumor_weight: float = 1.0,
        gamma: float = 2.0,
        ignore_index: int = -1
    ):
        super().__init__()
        
        self.dementia_weight = dementia_weight
        self.tumor_weight = tumor_weight
        
        self.dementia_loss = FocalLoss(gamma=gamma, ignore_index=ignore_index)
        self.tumor_loss = FocalLoss(gamma=gamma, ignore_index=ignore_index)
    
    def forward(
        self,
        dementia_logits: torch.Tensor,
        tumor_logits: torch.Tensor,
        dementia_labels: torch.Tensor,
        tumor_labels: torch.Tensor
    ) -> torch.Tensor:
        """Compute combined focal loss."""
        # Handle None logits for single-task models
        if dementia_logits is not None:
            d_loss = self.dementia_loss(dementia_logits, dementia_labels)
        else:
            d_loss = torch.tensor(0.0, device=tumor_logits.device if tumor_logits is not None else 'cpu')

        if tumor_logits is not None:
            t_loss = self.tumor_loss(tumor_logits, tumor_labels)
        else:
            t_loss = torch.tensor(0.0, device=dementia_logits.device if dementia_logits is not None else 'cpu')

        return self.dementia_weight * d_loss + self.tumor_weight * t_loss


class UncertaintyWeightedLoss(nn.Module):
    """
    Learns task weights automatically using uncertainty.
    
    Reference: Multi-Task Learning Using Uncertainty to Weigh Losses
    https://arxiv.org/abs/1705.07115
    """
    
    def __init__(self, ignore_index: int = -1):
        super().__init__()
        
        # Learnable log variances
        self.log_var_dementia = nn.Parameter(torch.zeros(1))
        self.log_var_tumor = nn.Parameter(torch.zeros(1))
        
        self.dementia_loss = nn.CrossEntropyLoss(ignore_index=ignore_index)
        self.tumor_loss = nn.CrossEntropyLoss(ignore_index=ignore_index)
    
    def forward(
        self,
        dementia_logits: torch.Tensor,
        tumor_logits: torch.Tensor,
        dementia_labels: torch.Tensor,
        tumor_labels: torch.Tensor
    ) -> torch.Tensor:
        """Compute uncertainty-weighted loss."""
        # Handle None logits for single-task models
        if dementia_logits is not None:
            d_loss = self.dementia_loss(dementia_logits, dementia_labels)
            # Handle NaN
            if torch.isnan(d_loss):
                d_loss = torch.tensor(0.0, device=dementia_logits.device)
            # Weighted loss with uncertainty
            precision_d = torch.exp(-self.log_var_dementia)
            d_component = precision_d * d_loss + self.log_var_dementia
        else:
            d_component = torch.tensor(0.0, device=tumor_logits.device if tumor_logits is not None else 'cpu')

        if tumor_logits is not None:
            t_loss = self.tumor_loss(tumor_logits, tumor_labels)
            # Handle NaN
            if torch.isnan(t_loss):
                t_loss = torch.tensor(0.0, device=tumor_logits.device)
            # Weighted loss with uncertainty
            precision_t = torch.exp(-self.log_var_tumor)
            t_component = precision_t * t_loss + self.log_var_tumor
        else:
            t_component = torch.tensor(0.0, device=dementia_logits.device if dementia_logits is not None else 'cpu')

        return d_component + t_component
