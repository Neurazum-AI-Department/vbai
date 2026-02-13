"""
Vbai - Visual Brain AI Library
A PyTorch-based library for multi-task brain MRI analysis.

Supports:
- 2D MRI Analysis: Dementia classification, brain tumor detection (RGB images)
- 3D MRI Analysis: NIfTI (.nii/.nii.gz) volumetric brain MRI classification

Example (2D):
    >>> import vbai
    >>> model = vbai.MultiTaskBrainModel(variant='q')
    >>> trainer = vbai.Trainer(model=model)
    >>> trainer.fit(dataset)

Example (3D - NIfTI):
    >>> import vbai
    >>> model = vbai.MultiTask3DBrainModel(variant='q', tasks={'alzheimer': 3})
    >>> trainer = vbai.Trainer3D(model=model)
    >>> trainer.fit(train_loader, val_loader, epochs=25)
"""

__version__ = "0.2.3"
__author__ = "Neurazum"

# ── 2D Models ──
from .models import (
    MultiTaskBrainModel,
    AttentionModule,
    SharedBackbone,
)

# ── 3D Models ──
from .models import (
    MultiTask3DBrainModel,
    SharedBackbone3D,
    AttentionModule3D,
    Prediction3DResult,
)

# ── 2D Data ──
from .data import (
    MRIDataset,
    UnifiedMRIDataset,
    get_transforms,
    get_train_transforms,
    get_val_transforms,
)

# ── 3D Data (NIfTI) ──
from .data import (
    NIfTIDataset,
    UnifiedNIfTIDataset,
    create_3d_dataloaders,
    get_3d_train_transforms,
    get_3d_val_transforms,
)

# ── 2D Training ──
from .training import (
    Trainer,
    MultiTaskLoss,
    EarlyStopping,
    ModelCheckpoint,
)

# ── 3D Training ──
from .training import (
    Trainer3D,
    Training3DHistory,
)

# ── Utils ──
from .utils import (
    VisualizationManager,
    BrainStructureAnalyzer,
    visualize_prediction,
    create_attention_heatmap,
)

# ── Configs ──
from .configs import (
    ModelConfig,
    TrainingConfig,
    get_default_config,
    Model3DConfig,
    Training3DConfig,
    Full3DConfig,
    get_default_3d_config,
)

# ── Hub (Model Zoo & HuggingFace) ──
from .hub import (
    ModelInfo,
    MODEL_REGISTRY,
    list_models,
    get_model_info,
    register_model,
    download_from_hub,
    push_to_hub,
    from_hub,
    generate_model_card,
)

# ── Export (ONNX) ──
from .export import (
    export_onnx,
    ONNXModel,
)


# ── Loading Functions ──

def load(path: str, device: str = 'cpu'):
    """Load a trained Vbai 2D model from checkpoint."""
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


def load_3d(path: str, device: str = 'cpu'):
    """
    Load a trained Vbai 3D model from checkpoint.

    Args:
        path: Path to 3D model checkpoint (.pt/.pth)
        device: Device to load onto ('cpu', 'cuda')

    Returns:
        MultiTask3DBrainModel in eval mode

    Example:
        >>> model = vbai.load_3d('alzheimer_3d.pt', device='cuda')
        >>> result = model.predict('scan.nii.gz', task='alzheimer',
        ...                        class_names=['CN', 'MCI', 'AD'])
    """
    import torch
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    config = checkpoint.get('config', {})
    variant = config.get('variant', 'q')
    tasks = config.get('tasks', {'alzheimer': 3})
    input_shape = config.get('input_shape', (96, 96, 96))
    in_channels = config.get('in_channels', 1)

    if isinstance(input_shape, list):
        input_shape = tuple(input_shape)

    model = MultiTask3DBrainModel(
        variant=variant,
        tasks=tasks,
        in_channels=in_channels,
        input_shape=input_shape,
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    return model


def load_pretrained(model_name: str = 'vbai-3d-q', device: str = 'cpu'):
    """
    Load a pretrained model from the Vbai model zoo via HuggingFace Hub.

    Args:
        model_name: Name of the pretrained model. Use vbai.list_models() to see available models.
            Available: 'vbai-2d-q', 'vbai-2d-f', 'vbai-3d-q', 'vbai-3d-f'
        device: Device to load onto ('cpu', 'cuda')

    Returns:
        Pretrained model in eval mode

    Example:
        >>> model = vbai.load_pretrained('vbai-3d-q', device='cuda')
        >>> result = model.predict('scan.nii.gz', task='alzheimer',
        ...                        class_names=['CN', 'MCI', 'AD'])
    """
    info = get_model_info(model_name)

    if info.hub_id is None:
        raise ValueError(
            f"Model '{model_name}' has no Hub ID configured. "
            f"Use vbai.load() or vbai.load_3d() with a local checkpoint."
        )

    try:
        return from_hub(info.hub_id, device=device)
    except Exception as e:
        available = ', '.join(MODEL_REGISTRY.keys())
        raise RuntimeError(
            f"Failed to download pretrained model '{model_name}' from "
            f"'{info.hub_id}': {e}\n"
            f"This model may not have pretrained weights uploaded yet.\n"
            f"Available model names: {available}\n"
            f"Alternatively, use vbai.load() / vbai.load_3d() with a local checkpoint."
        ) from e


# ── Class Constants ──

DEMENTIA_CLASSES = [
    'AD_Alzheimer', 'AD_Mild_Demented', 'AD_Moderate_Demented',
    'AD_Very_Mild_Demented', 'CN_Non_Demented', 'PD_Parkinson'
]

TUMOR_CLASSES = ['Glioma', 'Meningioma', 'No_Tumor', 'Pituitary']

ALZHEIMER_3D_CLASSES = ['CN', 'MCI', 'AD']

__all__ = [
    # Core
    '__version__',
    # 2D Models
    'MultiTaskBrainModel', 'AttentionModule', 'SharedBackbone',
    # 3D Models
    'MultiTask3DBrainModel', 'SharedBackbone3D', 'AttentionModule3D', 'Prediction3DResult',
    # 2D Data
    'MRIDataset', 'UnifiedMRIDataset',
    'get_transforms', 'get_train_transforms', 'get_val_transforms',
    # 3D Data
    'NIfTIDataset', 'UnifiedNIfTIDataset', 'create_3d_dataloaders',
    'get_3d_train_transforms', 'get_3d_val_transforms',
    # 2D Training
    'Trainer', 'MultiTaskLoss', 'EarlyStopping', 'ModelCheckpoint',
    # 3D Training
    'Trainer3D', 'Training3DHistory',
    # Utils
    'VisualizationManager', 'BrainStructureAnalyzer',
    'visualize_prediction', 'create_attention_heatmap',
    # Configs
    'ModelConfig', 'TrainingConfig', 'get_default_config',
    'Model3DConfig', 'Training3DConfig', 'Full3DConfig', 'get_default_3d_config',
    # Hub
    'ModelInfo', 'MODEL_REGISTRY', 'list_models', 'get_model_info', 'register_model',
    'download_from_hub', 'push_to_hub', 'from_hub', 'generate_model_card',
    # Export
    'export_onnx', 'ONNXModel',
    # Loading
    'load', 'load_3d', 'load_pretrained',
    # Constants
    'DEMENTIA_CLASSES', 'TUMOR_CLASSES', 'ALZHEIMER_3D_CLASSES',
]
