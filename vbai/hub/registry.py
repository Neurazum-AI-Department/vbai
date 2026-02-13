"""
Model Registry for Vbai

Provides a registry of available pretrained models with metadata.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union


@dataclass
class ModelInfo:
    """Metadata for a registered model."""
    name: str
    description: str
    model_type: str  # '2d' or '3d'
    variant: str  # 'f' or 'q'
    tasks: Union[list, dict]
    default_input_shape: tuple
    hub_id: Optional[str] = None
    num_parameters: Optional[int] = None
    tags: List[str] = field(default_factory=list)


MODEL_REGISTRY: Dict[str, ModelInfo] = {
    'vbai-2d-q': ModelInfo(
        name='vbai-2d-q',
        description='Quality 2D multi-task brain MRI model (dementia + tumor)',
        model_type='2d',
        variant='q',
        tasks=['dementia', 'tumor'],
        default_input_shape=(3, 224, 224),
        hub_id='Neurazum/vbai-2d-q',
        tags=['brain-mri', '2d', 'multi-task', 'dementia', 'tumor'],
    ),
    'vbai-2d-f': ModelInfo(
        name='vbai-2d-f',
        description='Fast 2D multi-task brain MRI model (dementia + tumor)',
        model_type='2d',
        variant='f',
        tasks=['dementia', 'tumor'],
        default_input_shape=(3, 224, 224),
        hub_id='Neurazum/vbai-2d-f',
        tags=['brain-mri', '2d', 'multi-task', 'dementia', 'tumor', 'lightweight'],
    ),
    'vbai-3d-q': ModelInfo(
        name='vbai-3d-q',
        description='Quality 3D volumetric brain MRI model (NIfTI)',
        model_type='3d',
        variant='q',
        tasks={'alzheimer': 3},
        default_input_shape=(1, 96, 96, 96),
        hub_id='Neurazum/vbai-3d-q',
        tags=['brain-mri', '3d', 'nifti', 'alzheimer', 'volumetric'],
    ),
    'vbai-3d-f': ModelInfo(
        name='vbai-3d-f',
        description='Fast 3D volumetric brain MRI model (NIfTI)',
        model_type='3d',
        variant='f',
        tasks={'alzheimer': 3},
        default_input_shape=(1, 96, 96, 96),
        hub_id='Neurazum/vbai-3d-f',
        tags=['brain-mri', '3d', 'nifti', 'alzheimer', 'volumetric', 'lightweight'],
    ),
}


def list_models(model_type: Optional[str] = None) -> List[ModelInfo]:
    """
    List available models in the registry.

    Args:
        model_type: Filter by model type ('2d', '3d', or None for all)

    Returns:
        List of ModelInfo for matching models

    Example:
        >>> vbai.list_models()
        >>> vbai.list_models('3d')
    """
    models = list(MODEL_REGISTRY.values())
    if model_type is not None:
        models = [m for m in models if m.model_type == model_type]
    return models


def get_model_info(name: str) -> ModelInfo:
    """
    Get info for a specific registered model.

    Args:
        name: Model name (e.g., 'vbai-2d-q', 'vbai-3d-f')

    Returns:
        ModelInfo for the requested model

    Raises:
        KeyError: If model name not found in registry
    """
    if name not in MODEL_REGISTRY:
        available = ', '.join(MODEL_REGISTRY.keys())
        raise KeyError(
            f"Model '{name}' not found in registry. "
            f"Available models: {available}"
        )
    return MODEL_REGISTRY[name]


def register_model(name: str, info: ModelInfo) -> None:
    """
    Register a custom model in the registry.

    Args:
        name: Unique model name
        info: ModelInfo with model metadata

    Example:
        >>> info = vbai.ModelInfo(
        ...     name='my-custom-model',
        ...     description='My custom brain model',
        ...     model_type='3d',
        ...     variant='q',
        ...     tasks={'my_task': 5},
        ...     default_input_shape=(1, 96, 96, 96),
        ...     hub_id='myuser/my-custom-model',
        ... )
        >>> vbai.register_model('my-custom-model', info)
    """
    MODEL_REGISTRY[name] = info
