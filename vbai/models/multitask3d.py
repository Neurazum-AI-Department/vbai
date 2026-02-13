"""
3D Multi-Task Brain MRI Model for Vbai

Volumetric (.nii/.nii.gz) brain MRI analysis with multi-task classification.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional, Literal, Union, List, NamedTuple
import numpy as np

from .backbone3d import SharedBackbone3D
from .attention3d import AttentionModule3D


class Prediction3DResult(NamedTuple):
    """Result container for 3D model predictions."""
    predicted_class: str
    probabilities: torch.Tensor
    confidence: float
    attention_map: Optional[torch.Tensor] = None


class MultiTask3DBrainModel(nn.Module):
    """
    3D multi-task deep learning model for volumetric brain MRI analysis.

    Processes NIfTI (.nii/.nii.gz) volumes using a 3D ResNet-style backbone
    with SE attention blocks. Supports flexible multi-task classification
    where each task has its own attention module and classifier head.

    Args:
        variant: Model variant
            - 'f' (fast): Lightweight, faster training, less GPU memory
            - 'q' (quality): Deeper, higher accuracy
        tasks: Dict mapping task names to number of classes.
            Example: {'dementia': 3, 'tumor': 4}
        in_channels: Number of input channels (default: 1 for NIfTI)
        input_shape: Expected volume shape (D, H, W) - volumes are resized to this
        dropout: Dropout rate for classifier heads (default: 0.5)

    Example:
        >>> # Single task: Alzheimer classification (CN, MCI, AD)
        >>> model = MultiTask3DBrainModel(
        ...     variant='q',
        ...     tasks={'alzheimer': 3}
        ... )
        >>> x = torch.randn(1, 1, 96, 96, 96)
        >>> outputs = model(x)
        >>> print(outputs['alzheimer'].shape)  # (1, 3)

        >>> # Multi-task: Alzheimer + Tumor
        >>> model = MultiTask3DBrainModel(
        ...     variant='q',
        ...     tasks={'alzheimer': 3, 'tumor': 4}
        ... )
        >>> outputs = model(x)
        >>> print(outputs['alzheimer'].shape, outputs['tumor'].shape)

        >>> # Single prediction
        >>> result = model.predict('brain_scan.nii.gz', task='alzheimer',
        ...                        class_names=['CN', 'MCI', 'AD'])
    """

    def __init__(
        self,
        variant: Literal['f', 'q'] = 'q',
        tasks: Optional[Dict[str, int]] = None,
        in_channels: int = 1,
        input_shape: Tuple[int, int, int] = (96, 96, 96),
        dropout: float = 0.5,
    ):
        super().__init__()

        if tasks is None:
            tasks = {'alzheimer': 3}

        if not tasks:
            raise ValueError("At least one task must be specified")

        self.variant = variant
        self.tasks = tasks
        self.task_names = list(tasks.keys())
        self.in_channels = in_channels
        self.input_shape = input_shape

        # Shared 3D backbone
        self.backbone = SharedBackbone3D(variant=variant, in_channels=in_channels)
        backbone_channels = self.backbone.out_channels

        # Dual pooling (avg + max) for richer feature representation
        self.global_avg_pool = nn.AdaptiveAvgPool3d(1)
        self.global_max_pool = nn.AdaptiveMaxPool3d(1)
        pooled_size = backbone_channels * 2  # avg + max concatenated

        # Task-specific attention modules and classifier heads
        self.task_attentions = nn.ModuleDict()
        self.task_heads = nn.ModuleDict()

        for task_name, num_classes in tasks.items():
            self.task_attentions[task_name] = AttentionModule3D(backbone_channels)
            self.task_heads[task_name] = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(pooled_size, 256),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout * 0.6),
                nn.Linear(256, num_classes)
            )

        # Store attention maps for visualization
        self._attention_maps: Dict[str, Optional[torch.Tensor]] = {
            name: None for name in self.task_names
        }

    def forward(
        self,
        x: torch.Tensor,
        return_attention: bool = False,
    ) -> Union[Dict[str, torch.Tensor],
               Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]]:
        """
        Forward pass through the 3D model.

        Args:
            x: Input tensor of shape (B, 1, D, H, W)
            return_attention: Whether to return attention maps

        Returns:
            If return_attention=False:
                Dict mapping task names to logits tensors
            If return_attention=True:
                Tuple of (logits_dict, attention_dict)
        """
        # Extract 3D features
        features = self.backbone(x)

        logits = {}
        attention_maps = {}

        for task_name in self.task_names:
            # Task-specific attention
            attended_feat, attn_map = self.task_attentions[task_name](features)
            self._attention_maps[task_name] = attn_map

            # Dual pooling
            avg_pool = self.global_avg_pool(attended_feat).view(x.size(0), -1)
            max_pool = self.global_max_pool(attended_feat).view(x.size(0), -1)
            pooled = torch.cat([avg_pool, max_pool], dim=1)

            # Classification
            logits[task_name] = self.task_heads[task_name](pooled)
            attention_maps[task_name] = attn_map

        if return_attention:
            return logits, attention_maps
        return logits

    def predict(
        self,
        volume: Union[str, np.ndarray, torch.Tensor],
        task: Optional[str] = None,
        class_names: Optional[List[str]] = None,
        return_attention: bool = False,
    ) -> Union[Prediction3DResult, Dict[str, Prediction3DResult]]:
        """
        Make prediction on a single NIfTI volume.

        Args:
            volume: Input volume (file path, numpy array, or tensor)
            task: Specific task to predict (if None, predicts all tasks)
            class_names: List of class names for the task
            return_attention: Whether to include attention maps

        Returns:
            Prediction3DResult for single task, or dict of results for all tasks
        """
        x = self._preprocess(volume)
        x = x.to(next(self.parameters()).device)

        self.eval()
        with torch.no_grad():
            logits_dict, attn_dict = self(x, return_attention=True)

        tasks_to_predict = [task] if task else self.task_names

        results = {}
        for t_name in tasks_to_predict:
            if t_name not in logits_dict:
                raise ValueError(f"Unknown task '{t_name}'. Available: {self.task_names}")

            t_logits = logits_dict[t_name]
            probs = F.softmax(t_logits, dim=1)
            pred_idx = probs.argmax(dim=1).item()
            confidence = probs[0, pred_idx].item()

            if class_names and len(class_names) == self.tasks[t_name]:
                pred_class = class_names[pred_idx]
            else:
                pred_class = str(pred_idx)

            results[t_name] = Prediction3DResult(
                predicted_class=pred_class,
                probabilities=probs[0],
                confidence=confidence,
                attention_map=attn_dict.get(t_name) if return_attention else None,
            )

        if task:
            return results[task]
        return results

    def _preprocess(
        self,
        volume: Union[str, np.ndarray, torch.Tensor],
    ) -> torch.Tensor:
        """
        Preprocess a NIfTI volume for model input.

        Handles file loading, normalization, and resizing.
        """
        if isinstance(volume, str):
            try:
                import nibabel as nib
            except ImportError:
                raise ImportError(
                    "nibabel is required for loading NIfTI files. "
                    "Install it with: pip install nibabel"
                )
            img = nib.load(volume).get_fdata()
            img = np.nan_to_num(img, nan=0.0, posinf=0.0, neginf=0.0)
            volume = img

        if isinstance(volume, np.ndarray):
            # Brain mask normalization
            brain_mask = volume > volume.mean()
            if brain_mask.sum() > 0:
                brain_pixels = volume[brain_mask]
                p1, p99 = np.percentile(brain_pixels, [1, 99])
                volume = np.clip(volume, p1, p99)
                mean, std = brain_pixels.mean(), brain_pixels.std()
                if std > 1e-6:
                    volume = (volume - mean) / (std + 1e-8)
                else:
                    volume = volume - mean
            else:
                mean, std = volume.mean(), volume.std()
                if std > 1e-6:
                    volume = (volume - mean) / (std + 1e-8)

            # Min-max to [0, 1]
            v_min, v_max = volume.min(), volume.max()
            if abs(v_max - v_min) > 1e-6:
                volume = (volume - v_min) / (v_max - v_min + 1e-8)
            else:
                volume = np.zeros_like(volume)

            volume = np.clip(volume, 0, 1)
            volume = torch.from_numpy(volume).float().unsqueeze(0)  # (1, D, H, W)

        if isinstance(volume, torch.Tensor):
            if volume.dim() == 3:
                volume = volume.unsqueeze(0)  # (1, D, H, W)
            if volume.dim() == 4:
                volume = volume.unsqueeze(0)  # (1, 1, D, H, W)

            # Resize to expected shape
            volume = F.interpolate(
                volume, size=self.input_shape,
                mode='trilinear', align_corners=False
            )
            return volume

        raise ValueError(f"Unsupported volume type: {type(volume)}")

    def get_attention_maps(self) -> Dict[str, Optional[torch.Tensor]]:
        """Get the last computed attention maps for all tasks."""
        return self._attention_maps

    def count_parameters(self) -> Dict[str, int]:
        """Count model parameters."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)

        result = {
            'total': total,
            'trainable': trainable,
            'backbone': sum(p.numel() for p in self.backbone.parameters()),
        }

        for task_name in self.task_names:
            result[f'{task_name}_attention'] = sum(
                p.numel() for p in self.task_attentions[task_name].parameters()
            )
            result[f'{task_name}_head'] = sum(
                p.numel() for p in self.task_heads[task_name].parameters()
            )

        return result

    def push_to_hub(self, repo_id: str, **kwargs) -> str:
        """
        Push this model to HuggingFace Hub.

        Args:
            repo_id: Target repository (e.g., 'username/my-3d-model')
            **kwargs: Extra args passed to vbai.push_to_hub()

        Returns:
            URL of the uploaded model
        """
        from ..hub.hub_utils import push_to_hub
        return push_to_hub(self, repo_id, **kwargs)

    def export_onnx(self, output_path: str, **kwargs) -> str:
        """
        Export this model to ONNX format.

        Args:
            output_path: Path for the .onnx output file
            **kwargs: Extra args passed to vbai.export_onnx()

        Returns:
            Path to the exported ONNX file
        """
        from ..export.onnx_export import export_onnx
        return export_onnx(self, output_path, **kwargs)

    def __repr__(self):
        params = self.count_parameters()
        tasks_str = ', '.join(f'{k}={v}' for k, v in self.tasks.items())
        return (
            f"MultiTask3DBrainModel(\n"
            f"  variant='{self.variant}',\n"
            f"  tasks={{{tasks_str}}},\n"
            f"  input_shape={self.input_shape},\n"
            f"  total_params={params['total']:,}\n"
            f")"
        )


def create_3d_model(
    variant: Literal['f', 'q'] = 'q',
    **kwargs
) -> MultiTask3DBrainModel:
    """
    Factory function to create a MultiTask3DBrainModel.

    Args:
        variant: Model variant ('f' for fast, 'q' for quality)
        **kwargs: Additional arguments passed to MultiTask3DBrainModel

    Returns:
        Configured MultiTask3DBrainModel instance
    """
    return MultiTask3DBrainModel(variant=variant, **kwargs)
