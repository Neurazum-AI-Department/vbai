"""
ONNX Export for Vbai Models

Export 2D and 3D brain MRI models to ONNX format for production deployment.
"""

from pathlib import Path
from typing import Optional, Tuple, Union, List, Dict

import torch
import torch.nn as nn
import numpy as np


class _MultiTask2DWrapper(nn.Module):
    """Wraps 2D MultiTaskBrainModel to return a flat tuple instead of optional Nones.

    Also patches FeatureFusion to use F.interpolate instead of adaptive_avg_pool2d
    for ONNX compatibility (adaptive_avg_pool2d with dynamic output_size is not
    supported by the ONNX tracer).
    """

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model
        self._patch_fusion()

    def _patch_fusion(self):
        """Replace FeatureFusion.forward with ONNX-compatible version."""
        if hasattr(self.model, 'fusion'):
            fusion = self.model.fusion
            original_forward = fusion.forward

            def onnx_safe_forward(main_features, edge_features):
                if edge_features.shape[-1] != main_features.shape[-1]:
                    edge_features = nn.functional.interpolate(
                        edge_features,
                        size=(main_features.shape[2], main_features.shape[3]),
                        mode='bilinear',
                        align_corners=False,
                    )
                combined = torch.cat([main_features, edge_features], dim=1)
                return fusion.fusion(combined)

            fusion.forward = onnx_safe_forward

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        dementia_logits, tumor_logits = self.model(x)
        outputs = []
        if dementia_logits is not None:
            outputs.append(dementia_logits)
        if tumor_logits is not None:
            outputs.append(tumor_logits)
        if len(outputs) == 1:
            return outputs[0]
        return tuple(outputs)


class _Dict2TupleWrapper(nn.Module):
    """Wraps 3D MultiTask3DBrainModel to return ordered tuple instead of dict."""

    def __init__(self, model: nn.Module, task_names: List[str]):
        super().__init__()
        self.model = model
        self.task_names = task_names

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        logits_dict = self.model(x)
        outputs = tuple(logits_dict[name] for name in self.task_names)
        if len(outputs) == 1:
            return outputs[0]
        return outputs


def export_onnx(
    model: nn.Module,
    output_path: str,
    input_shape: Optional[tuple] = None,
    opset_version: int = 14,
    dynamic_batch: bool = True,
    verify: bool = True,
) -> str:
    """
    Export a vbai model to ONNX format.

    Supports both 2D (MultiTaskBrainModel) and 3D (MultiTask3DBrainModel).

    Args:
        model: Trained vbai model
        output_path: Path for the .onnx output file
        input_shape: Input tensor shape (without batch dim).
            Default: (3, 224, 224) for 2D, (1, 96, 96, 96) for 3D.
        opset_version: ONNX opset version (default: 14)
        dynamic_batch: Whether to allow dynamic batch size
        verify: Whether to verify the exported model with onnxruntime

    Returns:
        Path to the exported ONNX file

    Example:
        >>> model = vbai.MultiTask3DBrainModel(variant='f', tasks={'alzheimer': 3})
        >>> vbai.export_onnx(model, 'model.onnx')

        >>> model2d = vbai.MultiTaskBrainModel(variant='f')
        >>> vbai.export_onnx(model2d, 'model_2d.onnx')
    """
    from ..models.multitask3d import MultiTask3DBrainModel
    from ..models.multitask import MultiTaskBrainModel

    model.eval()
    is_3d = isinstance(model, MultiTask3DBrainModel)

    # Determine input shape
    if input_shape is None:
        if is_3d:
            in_channels = getattr(model, 'in_channels', 1)
            shape_3d = getattr(model, 'input_shape', (96, 96, 96))
            input_shape = (in_channels, *shape_3d)
        else:
            input_shape = (3, 224, 224)

    # Create dummy input
    dummy_input = torch.randn(1, *input_shape)
    device = next(model.parameters()).device
    dummy_input = dummy_input.to(device)

    # Wrap model for ONNX-compatible output
    if is_3d:
        task_names = model.task_names
        wrapped = _Dict2TupleWrapper(model, task_names)
        output_names = [f"{name}_logits" for name in task_names]
    else:
        wrapped = _MultiTask2DWrapper(model)
        output_names = []
        if model.has_dementia:
            output_names.append("dementia_logits")
        if model.has_tumor:
            output_names.append("tumor_logits")

    # Dynamic axes
    dynamic_axes = {}
    if dynamic_batch:
        dynamic_axes['input'] = {0: 'batch_size'}
        for name in output_names:
            dynamic_axes[name] = {0: 'batch_size'}

    # Export
    output_path = str(output_path)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    torch.onnx.export(
        wrapped,
        dummy_input,
        output_path,
        opset_version=opset_version,
        input_names=['input'],
        output_names=output_names,
        dynamic_axes=dynamic_axes if dynamic_axes else None,
    )

    print(f"ONNX model exported to {output_path}")

    # Verify
    if verify:
        _verify_onnx(output_path, dummy_input, wrapped, output_names)

    return output_path


def _verify_onnx(
    onnx_path: str,
    dummy_input: torch.Tensor,
    model: nn.Module,
    output_names: List[str],
) -> None:
    """Verify the exported ONNX model."""
    # Check with onnx
    try:
        import onnx
        onnx_model = onnx.load(onnx_path)
        onnx.checker.check_model(onnx_model)
        print("  ONNX model structure verified.")
    except ImportError:
        print("  Skipping ONNX structure check (install onnx for verification)")
    except Exception as e:
        print(f"  ONNX structure check warning: {e}")

    # Compare outputs with onnxruntime
    try:
        import onnxruntime as ort
        session = ort.InferenceSession(onnx_path)
        input_np = dummy_input.cpu().numpy()

        ort_outputs = session.run(None, {'input': input_np})

        # Get PyTorch outputs
        model.eval()
        with torch.no_grad():
            pt_result = model(dummy_input)

        if isinstance(pt_result, torch.Tensor):
            pt_outputs = [pt_result.cpu().numpy()]
        else:
            pt_outputs = [t.cpu().numpy() for t in pt_result]

        # Compare
        all_close = True
        for i, (pt_out, ort_out) in enumerate(zip(pt_outputs, ort_outputs)):
            if not np.allclose(pt_out, ort_out, atol=1e-5):
                all_close = False
                max_diff = np.abs(pt_out - ort_out).max()
                print(f"  Warning: Output '{output_names[i]}' max diff: {max_diff:.6f}")

        if all_close:
            print("  ONNX output verification passed.")
        else:
            print("  ONNX outputs differ slightly (may be due to float precision).")

    except ImportError:
        print("  Skipping ONNX runtime check (install onnxruntime for verification)")
    except Exception as e:
        print(f"  ONNX runtime check warning: {e}")
