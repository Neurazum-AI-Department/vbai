"""
Vbai - Visual Brain AI Library
A PyTorch-based library for multi-task brain MRI analysis.

Supports:
- Dementia classification (AD, Mild/Moderate/Very Mild Demented, Non Demented, PD)
- Brain tumor detection (Glioma, Meningioma, No Tumor, Pituitary)

Example:
    >>> import vbai
    >>> model = vbai.MultiTaskBrainModel(variant='q')
    >>> trainer = vbai.Trainer(model=model, epochs=10)
    >>> trainer.fit(dataset)
"""

__version__ = "0.1.9"
__author__ = "Neurazum"

# Models
from .models import (
    MultiTaskBrainModel,
    AttentionModule,
    SharedBackbone,
)

# Data
from .data import (
    MRIDataset,
    UnifiedMRIDataset,
    get_transforms,
    get_train_transforms,
    get_val_transforms,
)

# Training
from .training import (
    Trainer,
    MultiTaskLoss,
    EarlyStopping,
    ModelCheckpoint,
)

# Utils
from .utils import (
    VisualizationManager,
    BrainStructureAnalyzer,
    visualize_prediction,
    create_attention_heatmap,
)

# Configs
from .configs import (
    ModelConfig,
    TrainingConfig,
    get_default_config,
)


def load(path: str, device: str = 'cpu'):
    """Load a trained Vbai model from checkpoint."""
    import torch
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    config = checkpoint.get('config', {})
    variant = config.get('variant', 'q')
    tasks = config.get('tasks', ['dementia', 'tumor'])
    model = MultiTaskBrainModel(variant=variant, tasks=tasks)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    return model


def load_pretrained(model_name: str = 'vbai-2.5q', device: str = 'cpu'):
    """Load a pretrained Vbai model (not yet implemented)."""
    raise NotImplementedError(
        f"Pretrained model '{model_name}' download not yet implemented. "
        "Use vbai.load() with a local checkpoint path."
    )


DEMENTIA_CLASSES = [
    'AD_Alzheimer', 'AD_Mild_Demented', 'AD_Moderate_Demented',
    'AD_Very_Mild_Demented', 'CN_Non_Demented', 'PD_Parkinson'
]

TUMOR_CLASSES = ['Glioma', 'Meningioma', 'No_Tumor', 'Pituitary']

__all__ = [
    '__version__', 'MultiTaskBrainModel', 'AttentionModule', 'SharedBackbone',
    'MRIDataset', 'UnifiedMRIDataset', 'get_transforms', 'get_train_transforms',
    'get_val_transforms', 'Trainer', 'MultiTaskLoss', 'EarlyStopping',
    'ModelCheckpoint', 'VisualizationManager', 'BrainStructureAnalyzer',
    'visualize_prediction', 'create_attention_heatmap', 'ModelConfig',
    'TrainingConfig', 'get_default_config', 'load', 'load_pretrained',
    'DEMENTIA_CLASSES', 'TUMOR_CLASSES',
]
