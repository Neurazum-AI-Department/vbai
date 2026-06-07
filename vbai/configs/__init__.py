"""Vbai Configuration Module"""

from .defaults import (
    ModelConfig,
    TrainingConfig,
    get_default_config,
    Model3DConfig,
    Training3DConfig,
    Full3DConfig,
    get_default_3d_config,
)

# Segmentation configs
from .segmentation_config import (
    SegmentationModelConfig,
    SegmentationTrainingConfig,
    FullSegmentationConfig,
    get_segmentation_config,
)

# Progression configs
from .progression_config import (
    ProgressionModelConfig,
    ProgressionTrainingConfig,
    FullProgressionConfig,
    get_progression_config,
)

__all__ = [
    # 2D / 3D Classification
    'ModelConfig', 'TrainingConfig', 'get_default_config',
    'Model3DConfig', 'Training3DConfig', 'Full3DConfig', 'get_default_3d_config',
    # Segmentation
    'SegmentationModelConfig', 'SegmentationTrainingConfig',
    'FullSegmentationConfig', 'get_segmentation_config',
    # Progression
    'ProgressionModelConfig', 'ProgressionTrainingConfig',
    'FullProgressionConfig', 'get_progression_config',
]
