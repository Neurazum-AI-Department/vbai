"""
ONNX Export for Vbai Models

Export 2D classification, 3D classification, 3D segmentation, and
multimodal progression models to ONNX format for production deployment.
"""

from pathlib import Path
from typing import Optional, Tuple, Union, List, Dict

import torch
import torch.nn as nn
import numpy as np


# ──────────────────────────────────────────────────────────────────────────────
# Wrappers for classification models
# ──────────────────────────────────────────────────────────────────────────────

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


# ──────────────────────────────────────────────────────────────────────────────
# Wrappers for VbaiSegNet3D
# ──────────────────────────────────────────────────────────────────────────────

class _SegNet3DWrapper(nn.Module):
    """Disables deep supervision so VbaiSegNet3D returns a single tensor."""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x, return_aux=False)


# ──────────────────────────────────────────────────────────────────────────────
# Wrappers for VbaiProgressionNet
# ──────────────────────────────────────────────────────────────────────────────

class _ProgressionMRIWrapper(nn.Module):
    """MRI-only inference — returns (class_logits, will_progress_logits, time_to_conversion)."""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, mri: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        out = self.model(mri=mri, tab=None)
        prog = out['progression']
        return (
            out['fused_logits'],
            prog['will_progress_logits'],
            prog['time_to_conversion'],
        )


class _ProgressionTabWrapper(nn.Module):
    """Tabular-only inference — returns (class_logits, will_progress_logits, time_to_conversion)."""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, tab: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        out = self.model(mri=None, tab=tab)
        prog = out['progression']
        return (
            out['fused_logits'],
            prog['will_progress_logits'],
            prog['time_to_conversion'],
        )


class _ProgressionMultiWrapper(nn.Module):
    """Multimodal inference — returns (class_logits, will_progress_logits, time_to_conversion)."""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(
        self,
        mri: torch.Tensor,
        tab: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        out = self.model(mri=mri, tab=tab)
        prog = out['progression']
        return (
            out['fused_logits'],
            prog['will_progress_logits'],
            prog['time_to_conversion'],
        )


# ──────────────────────────────────────────────────────────────────────────────
# Verification helper
# ──────────────────────────────────────────────────────────────────────────────

def _verify_onnx(
    onnx_path: str,
    dummy_inputs,          # tensor or tuple of tensors
    model: nn.Module,
    output_names: List[str],
    input_names: List[str],
) -> None:
    """Verify the exported ONNX model against PyTorch outputs."""
    try:
        import onnx
        onnx_model = onnx.load(onnx_path)
        onnx.checker.check_model(onnx_model)
        print("  ONNX model structure verified.")
    except ImportError:
        print("  Skipping ONNX structure check (install onnx for verification)")
    except Exception as e:
        print(f"  ONNX structure check warning: {e}")

    try:
        import onnxruntime as ort
        session = ort.InferenceSession(onnx_path)

        if isinstance(dummy_inputs, torch.Tensor):
            dummy_inputs = (dummy_inputs,)

        feed = {name: t.cpu().numpy() for name, t in zip(input_names, dummy_inputs)}
        ort_outputs = session.run(None, feed)

        model.eval()
        with torch.no_grad():
            pt_result = model(*dummy_inputs)

        if isinstance(pt_result, torch.Tensor):
            pt_outputs = [pt_result.cpu().numpy()]
        else:
            pt_outputs = [t.cpu().numpy() for t in pt_result]

        all_close = True
        for i, (pt_out, ort_out) in enumerate(zip(pt_outputs, ort_outputs)):
            if not np.allclose(pt_out, ort_out, atol=1e-4):
                all_close = False
                max_diff = np.abs(pt_out - ort_out).max()
                name = output_names[i] if i < len(output_names) else str(i)
                print(f"  Warning: Output '{name}' max diff: {max_diff:.6f}")

        if all_close:
            print("  ONNX output verification passed.")
        else:
            print("  ONNX outputs differ slightly (may be due to float precision).")

    except ImportError:
        print("  Skipping ONNX runtime check (install onnxruntime for verification)")
    except Exception as e:
        print(f"  ONNX runtime check warning: {e}")


# ──────────────────────────────────────────────────────────────────────────────
# Public API: classification models (2D and 3D)
# ──────────────────────────────────────────────────────────────────────────────

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

    Supports 2D classification (MultiTaskBrainModel), 3D classification
    (MultiTask3DBrainModel), 3D segmentation (VbaiSegNet3D), and multimodal
    progression (VbaiProgressionNet).

    For VbaiSegNet3D and VbaiProgressionNet use the dedicated helpers
    ``export_segmentation_onnx`` and ``export_progression_onnx`` for full
    control over export options.

    Args:
        model: Trained vbai model.
        output_path: Path for the .onnx output file.
        input_shape: Input tensor shape (without batch dim).
            Defaults: (3, 224, 224) for 2D, (1, 96, 96, 96) for 3D / segmentation.
        opset_version: ONNX opset version (default: 14).
        dynamic_batch: Whether to allow dynamic batch size.
        verify: Whether to verify with onnxruntime.

    Returns:
        Path to the exported ONNX file.

    Example::

        >>> model = vbai.MultiTask3DBrainModel(variant='f', tasks={'alzheimer': 3})
        >>> vbai.export_onnx(model, 'model.onnx')

        >>> seg = vbai.VbaiSegNet3D(in_channels=4, out_channels=1)
        >>> vbai.export_onnx(seg, 'seg.onnx')

        >>> prog = vbai.VbaiProgressionNet()
        >>> vbai.export_onnx(prog, 'prog.onnx')
    """
    from ..models.multitask3d import MultiTask3DBrainModel
    from ..models.multitask import MultiTaskBrainModel
    from ..models.segmentation3d import VbaiSegNet3D
    from ..models.progression3d import VbaiProgressionNet

    if isinstance(model, VbaiSegNet3D):
        return export_segmentation_onnx(
            model, output_path,
            input_shape=input_shape,
            opset_version=opset_version,
            dynamic_batch=dynamic_batch,
            verify=verify,
        )

    if isinstance(model, VbaiProgressionNet):
        return export_progression_onnx(
            model, output_path,
            mode='multi',
            mri_shape=input_shape,
            opset_version=opset_version,
            dynamic_batch=dynamic_batch,
            verify=verify,
        )

    model.eval()
    is_3d = isinstance(model, MultiTask3DBrainModel)

    if input_shape is None:
        if is_3d:
            in_channels = getattr(model, 'in_channels', 1)
            shape_3d = getattr(model, 'input_shape', (96, 96, 96))
            input_shape = (in_channels, *shape_3d)
        else:
            input_shape = (3, 224, 224)

    device = next(model.parameters()).device
    dummy_input = torch.randn(1, *input_shape).to(device)

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

    dynamic_axes: Dict = {}
    if dynamic_batch:
        dynamic_axes['input'] = {0: 'batch_size'}
        for name in output_names:
            dynamic_axes[name] = {0: 'batch_size'}

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

    if verify:
        _verify_onnx(output_path, dummy_input, wrapped, output_names, ['input'])

    return output_path


# ──────────────────────────────────────────────────────────────────────────────
# Public API: VbaiSegNet3D
# ──────────────────────────────────────────────────────────────────────────────

def export_segmentation_onnx(
    model: nn.Module,
    output_path: str,
    input_shape: Optional[tuple] = None,
    opset_version: int = 14,
    dynamic_batch: bool = True,
    verify: bool = True,
) -> str:
    """
    Export ``VbaiSegNet3D`` to ONNX format.

    Deep supervision is automatically disabled so the model outputs a single
    segmentation mask tensor ``(B, out_channels, D, H, W)``.

    Args:
        model: Trained ``VbaiSegNet3D`` instance.
        output_path: Destination ``.onnx`` file path.
        input_shape: ``(C, D, H, W)`` without batch dim.
            Defaults to ``(model.in_channels, 128, 128, 128)``.
        opset_version: ONNX opset (default 14).
        dynamic_batch: Export with dynamic batch axis.
        verify: Validate with onnxruntime after export.

    Returns:
        Path to the exported ONNX file.

    Example::

        from vbai.models import VbaiSegNet3D
        from vbai.export import export_segmentation_onnx

        model = VbaiSegNet3D(in_channels=4, out_channels=1)
        export_segmentation_onnx(model, 'tumor_seg.onnx')
    """
    model.eval()
    device = next(model.parameters()).device

    in_ch = getattr(model, 'in_channels', 1)
    if input_shape is None:
        input_shape = (in_ch, 128, 128, 128)

    dummy_input = torch.randn(1, *input_shape).to(device)
    wrapped = _SegNet3DWrapper(model)

    input_names = ['volume']
    output_names = ['segmentation_logits']

    dynamic_axes: Dict = {}
    if dynamic_batch:
        dynamic_axes['volume'] = {0: 'batch_size'}
        dynamic_axes['segmentation_logits'] = {0: 'batch_size'}

    output_path = str(output_path)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    torch.onnx.export(
        wrapped,
        dummy_input,
        output_path,
        opset_version=opset_version,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes if dynamic_axes else None,
    )
    print(f"Segmentation ONNX model exported to {output_path}")

    if verify:
        _verify_onnx(output_path, dummy_input, wrapped, output_names, input_names)

    return output_path


# ──────────────────────────────────────────────────────────────────────────────
# Public API: VbaiProgressionNet
# ──────────────────────────────────────────────────────────────────────────────

def export_progression_onnx(
    model: nn.Module,
    output_path: str,
    mode: str = 'multi',
    mri_shape: Optional[tuple] = None,
    tab_dim: int = 26,
    opset_version: int = 14,
    dynamic_batch: bool = True,
    verify: bool = True,
) -> str:
    """
    Export ``VbaiProgressionNet`` to ONNX format.

    Three export modes correspond to the three inference paths:

    * ``'mri'``   — MRI volume only (single input: ``volume``).
    * ``'tab'``   — Biomarker table only (single input: ``biomarkers``).
    * ``'multi'`` — Multimodal, both inputs (``volume`` + ``biomarkers``).

    All modes export three outputs:

    * ``class_logits``           — CN / MCI / AD classification logits ``(B, 3)``.
    * ``will_progress_logits``   — Binary progression logit ``(B, 1)``.
    * ``time_to_conversion``     — Estimated months to conversion ``(B, 1)``.

    Args:
        model: Trained ``VbaiProgressionNet`` instance.
        output_path: Destination ``.onnx`` file path.
        mode: ``'mri'``, ``'tab'``, or ``'multi'``.
        mri_shape: ``(C, D, H, W)`` without batch. Defaults to ``(1, 96, 96, 96)``.
        tab_dim: Tabular feature dimension (default 26 — 13 values + 13 masks).
        opset_version: ONNX opset (default 14).
        dynamic_batch: Export with dynamic batch axis.
        verify: Validate with onnxruntime after export.

    Returns:
        Path to the exported ONNX file.

    Example::

        from vbai.models import VbaiProgressionNet
        from vbai.export import export_progression_onnx

        model = VbaiProgressionNet()
        # Multimodal (MRI + biomarkers)
        export_progression_onnx(model, 'prog_multi.onnx', mode='multi')
        # MRI only
        export_progression_onnx(model, 'prog_mri.onnx', mode='mri')
        # Biomarkers only
        export_progression_onnx(model, 'prog_tab.onnx', mode='tab')
    """
    if mode not in ('mri', 'tab', 'multi'):
        raise ValueError(f"mode must be 'mri', 'tab', or 'multi', got '{mode}'")

    model.eval()
    device = next(model.parameters()).device

    if mri_shape is None:
        mri_shape = (1, 96, 96, 96)

    output_names = ['class_logits', 'will_progress_logits', 'time_to_conversion']

    if mode == 'mri':
        wrapped = _ProgressionMRIWrapper(model)
        dummy = torch.randn(1, *mri_shape).to(device)
        input_names = ['volume']
        dynamic_axes: Dict = {}
        if dynamic_batch:
            dynamic_axes['volume'] = {0: 'batch_size'}
            for n in output_names:
                dynamic_axes[n] = {0: 'batch_size'}
        torch_inputs = dummy

    elif mode == 'tab':
        wrapped = _ProgressionTabWrapper(model)
        dummy = torch.randn(1, tab_dim).to(device)
        input_names = ['biomarkers']
        dynamic_axes = {}
        if dynamic_batch:
            dynamic_axes['biomarkers'] = {0: 'batch_size'}
            for n in output_names:
                dynamic_axes[n] = {0: 'batch_size'}
        torch_inputs = dummy

    else:  # multi
        wrapped = _ProgressionMultiWrapper(model)
        dummy_mri = torch.randn(1, *mri_shape).to(device)
        dummy_tab = torch.randn(1, tab_dim).to(device)
        input_names = ['volume', 'biomarkers']
        dynamic_axes = {}
        if dynamic_batch:
            dynamic_axes['volume'] = {0: 'batch_size'}
            dynamic_axes['biomarkers'] = {0: 'batch_size'}
            for n in output_names:
                dynamic_axes[n] = {0: 'batch_size'}
        torch_inputs = (dummy_mri, dummy_tab)

    output_path = str(output_path)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    torch.onnx.export(
        wrapped,
        torch_inputs,
        output_path,
        opset_version=opset_version,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes if dynamic_axes else None,
    )
    print(f"Progression ONNX model ({mode} mode) exported to {output_path}")

    if verify:
        _verify_onnx(
            output_path,
            torch_inputs if isinstance(torch_inputs, tuple) else (torch_inputs,),
            wrapped,
            output_names,
            input_names,
        )

    return output_path
