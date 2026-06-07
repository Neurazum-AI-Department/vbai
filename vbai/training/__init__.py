"""Vbai Training Module"""

from .trainer import Trainer
from .losses import MultiTaskLoss
from .callbacks import EarlyStopping, ModelCheckpoint, CallbackList

# 3D Classification Training
from .trainer3d import Trainer3D, Training3DHistory

# 3D Segmentation Training
from .segmentation_losses import (
    DiceLoss,
    MulticlassDiceLoss,
    FocalLoss as SegFocalLoss,
    TumorSegmentationLoss,
    TissueSegmentationLoss,
    DeepSupervisionLoss,
)
from .segmentation_trainer import SegmentationTrainer, SegmentationHistory

# 3D Progression Training
from .progression_losses import (
    FocalLoss3Class,
    ProgressionLoss,
    InfoNCELoss,
    VbaiProgressionLoss,
)
from .progression_trainer import ProgressionTrainer, ProgressionPhaseHistory

__all__ = [
    # 2D
    'Trainer', 'MultiTaskLoss', 'EarlyStopping', 'ModelCheckpoint', 'CallbackList',
    # 3D Classification
    'Trainer3D', 'Training3DHistory',
    # Segmentation Losses
    'DiceLoss', 'MulticlassDiceLoss', 'SegFocalLoss',
    'TumorSegmentationLoss', 'TissueSegmentationLoss', 'DeepSupervisionLoss',
    # Segmentation Trainer
    'SegmentationTrainer', 'SegmentationHistory',
    # Progression Losses
    'FocalLoss3Class', 'ProgressionLoss', 'InfoNCELoss', 'VbaiProgressionLoss',
    # Progression Trainer
    'ProgressionTrainer', 'ProgressionPhaseHistory',
]
