"""
HuggingFace Hub Integration for Vbai

Upload, download, and share models via HuggingFace Hub.
"""

import os
import tempfile
from pathlib import Path
from typing import Optional, Dict, Any

import torch


def _require_huggingface_hub():
    """Lazy import huggingface_hub with helpful error message."""
    try:
        import huggingface_hub
        return huggingface_hub
    except ImportError:
        raise ImportError(
            "huggingface_hub is required for Hub integration. "
            "Install with: pip install vbai[hub] "
            "or: pip install huggingface_hub>=0.14.0"
        )


def download_from_hub(
    repo_id: str,
    filename: str = "model.pt",
    revision: Optional[str] = None,
    cache_dir: Optional[str] = None,
) -> str:
    """
    Download a model checkpoint from HuggingFace Hub.

    Args:
        repo_id: Repository ID (e.g., 'Neurazum/vbai-3d-q')
        filename: Checkpoint filename in the repo
        revision: Git revision (branch, tag, or commit hash)
        cache_dir: Local cache directory

    Returns:
        Local path to the downloaded checkpoint file

    Example:
        >>> path = vbai.download_from_hub('Neurazum/vbai-3d-q')
        >>> model = vbai.load_3d(path, device='cuda')
    """
    hf_hub = _require_huggingface_hub()
    return hf_hub.hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        revision=revision,
        cache_dir=cache_dir,
    )


def push_to_hub(
    model: torch.nn.Module,
    repo_id: str,
    filename: str = "model.pt",
    commit_message: str = "Upload vbai model",
    private: bool = False,
    model_card: Optional[str] = None,
    token: Optional[str] = None,
    **save_kwargs,
) -> str:
    """
    Push a trained model to HuggingFace Hub.

    Saves the model checkpoint and uploads it along with an auto-generated
    model card (README.md).

    Args:
        model: Trained model (MultiTaskBrainModel or MultiTask3DBrainModel)
        repo_id: Target repository ID (e.g., 'username/my-brain-model')
        filename: Checkpoint filename
        commit_message: Git commit message
        private: Whether the repo should be private
        model_card: Custom model card markdown (auto-generated if None)
        token: HuggingFace API token (uses cached login if None)
        **save_kwargs: Extra kwargs passed to torch.save

    Returns:
        URL of the uploaded model on HuggingFace Hub

    Example:
        >>> model = vbai.MultiTask3DBrainModel(variant='q', tasks={'alzheimer': 3})
        >>> # ... train model ...
        >>> url = vbai.push_to_hub(model, 'myuser/alzheimer-model')
    """
    hf_hub = _require_huggingface_hub()

    # Build checkpoint
    checkpoint = _build_checkpoint(model)
    checkpoint.update(save_kwargs)

    # Create repo if needed
    api = hf_hub.HfApi(token=token)
    api.create_repo(repo_id=repo_id, exist_ok=True, private=private)

    with tempfile.TemporaryDirectory() as tmpdir:
        # Save checkpoint
        checkpoint_path = os.path.join(tmpdir, filename)
        torch.save(checkpoint, checkpoint_path)

        # Generate model card
        if model_card is None:
            from .model_card import generate_model_card
            model_card = generate_model_card(model)

        readme_path = os.path.join(tmpdir, "README.md")
        with open(readme_path, 'w', encoding='utf-8') as f:
            f.write(model_card)

        # Upload
        api.upload_folder(
            repo_id=repo_id,
            folder_path=tmpdir,
            commit_message=commit_message,
        )

    url = f"https://huggingface.co/{repo_id}"
    print(f"Model pushed to {url}")
    return url


def from_hub(
    repo_id: str,
    filename: str = "model.pt",
    device: str = "cpu",
    revision: Optional[str] = None,
    cache_dir: Optional[str] = None,
) -> torch.nn.Module:
    """
    Download and load a model from HuggingFace Hub.

    Auto-detects whether the model is 2D or 3D from the checkpoint.

    Args:
        repo_id: Repository ID (e.g., 'Neurazum/vbai-3d-q')
        filename: Checkpoint filename
        device: Device to load onto ('cpu', 'cuda')
        revision: Git revision
        cache_dir: Local cache directory

    Returns:
        Loaded model in eval mode

    Example:
        >>> model = vbai.from_hub('Neurazum/vbai-3d-q', device='cuda')
        >>> result = model.predict('scan.nii.gz', task='alzheimer',
        ...                        class_names=['CN', 'MCI', 'AD'])
    """
    checkpoint_path = download_from_hub(
        repo_id=repo_id,
        filename=filename,
        revision=revision,
        cache_dir=cache_dir,
    )

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint.get('config', {})
    model_type = checkpoint.get('model_type', _infer_model_type(config))

    if model_type == '3d':
        from ..models.multitask3d import MultiTask3DBrainModel
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
    else:
        from ..models.multitask import MultiTaskBrainModel
        variant = config.get('variant', 'q')
        tasks = config.get('tasks', ['dementia', 'tumor'])

        model = MultiTaskBrainModel(variant=variant, tasks=tasks)

    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    print(f"Loaded {'3D' if model_type == '3d' else '2D'} model from {repo_id}")
    return model


def _build_checkpoint(model: torch.nn.Module) -> Dict[str, Any]:
    """Build a checkpoint dict from a model."""
    from ..models.multitask3d import MultiTask3DBrainModel
    from ..models.multitask import MultiTaskBrainModel

    checkpoint = {
        'model_state_dict': model.state_dict(),
    }

    if isinstance(model, MultiTask3DBrainModel):
        checkpoint['model_type'] = '3d'
        checkpoint['config'] = {
            'variant': getattr(model, 'variant', 'q'),
            'tasks': getattr(model, 'tasks', {}),
            'input_shape': getattr(model, 'input_shape', (96, 96, 96)),
            'in_channels': getattr(model, 'in_channels', 1),
        }
    elif isinstance(model, MultiTaskBrainModel):
        checkpoint['model_type'] = '2d'
        checkpoint['config'] = {
            'variant': getattr(model, 'variant', 'q'),
            'tasks': getattr(model, 'tasks', ['dementia', 'tumor']),
            'num_dementia_classes': getattr(model, 'num_dementia_classes', 6),
            'num_tumor_classes': getattr(model, 'num_tumor_classes', 4),
        }
    else:
        checkpoint['config'] = {}

    return checkpoint


def _infer_model_type(config: dict) -> str:
    """Infer model type from checkpoint config."""
    tasks = config.get('tasks', None)
    if isinstance(tasks, dict):
        return '3d'
    if 'input_shape' in config or 'in_channels' in config:
        return '3d'
    return '2d'
