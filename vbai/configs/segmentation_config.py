"""Configurations for VbaiSegNet3D segmentation."""

from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple, Any
import json
from pathlib import Path


@dataclass
class SegmentationModelConfig:
    """
    VbaiSegNet3D architecture configuration.

    Args:
        task: 'tumor' (binary) or 'tissue' (3-class CSF/GM/WM).
        in_channels: Number of input MRI modalities.
        out_channels: Number of output segmentation channels.
        base_channels: Stem output channels.
        channel_mult: Per-level channel multipliers.
        use_se: Squeeze-and-Excitation blocks.
        use_cbam: CBAM attention blocks.
        use_aspp: ASPP bottleneck.
        use_attn_gate: Attention gates on skip connections.
        use_deep_supervision: Auxiliary decoder outputs during training.
        dropout: Dropout rate inside residual blocks.
        input_shape: Target volume shape (D, H, W).
    """
    task: str = 'tumor'
    in_channels: int = 2
    out_channels: int = 1
    base_channels: int = 32
    channel_mult: Tuple[int, ...] = (1, 2, 4, 8, 10)
    use_se: bool = True
    use_cbam: bool = True
    use_aspp: bool = True
    use_attn_gate: bool = True
    use_deep_supervision: bool = True
    dropout: float = 0.1
    input_shape: Tuple[int, int, int] = (96, 96, 96)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d['channel_mult'] = list(d['channel_mult'])
        d['input_shape'] = list(d['input_shape'])
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'SegmentationModelConfig':
        valid = {k: v for k, v in d.items() if k in cls.__dataclass_fields__}
        if 'channel_mult' in valid:
            valid['channel_mult'] = tuple(valid['channel_mult'])
        if 'input_shape' in valid:
            valid['input_shape'] = tuple(valid['input_shape'])
        return cls(**valid)

    def save(self, path: str) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def load(cls, path: str) -> 'SegmentationModelConfig':
        return cls.from_dict(json.loads(Path(path).read_text()))


@dataclass
class SegmentationTrainingConfig:
    """Training configuration for SegmentationTrainer."""
    epochs: int = 100
    batch_size: int = 2
    lr: float = 1e-4
    weight_decay: float = 1e-5
    scheduler_T0: int = 20
    scheduler_T_mult: int = 2
    scheduler_eta_min: float = 1e-7
    grad_clip: float = 1.0
    use_amp: bool = True
    early_stopping_patience: int = 15
    checkpoint_dir: str = './checkpoints'
    checkpoint_name: str = 'vbai_seg'
    val_split: float = 0.15
    test_split: float = 0.15
    num_workers: int = 4
    pin_memory: bool = True
    deep_sup_weights: Tuple[float, ...] = (1.0, 0.5, 0.25, 0.125)
    tumor_dice_weight: float = 0.6
    tumor_focal_weight: float = 0.4
    tissue_dice_weight: float = 0.7
    tissue_mse_weight: float = 0.3

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d['deep_sup_weights'] = list(d['deep_sup_weights'])
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'SegmentationTrainingConfig':
        valid = {k: v for k, v in d.items() if k in cls.__dataclass_fields__}
        if 'deep_sup_weights' in valid:
            valid['deep_sup_weights'] = tuple(valid['deep_sup_weights'])
        return cls(**valid)


@dataclass
class FullSegmentationConfig:
    """Combined model + training configuration for segmentation."""
    model: SegmentationModelConfig = field(default_factory=SegmentationModelConfig)
    training: SegmentationTrainingConfig = field(default_factory=SegmentationTrainingConfig)

    def to_dict(self) -> Dict[str, Any]:
        return {'model': self.model.to_dict(), 'training': self.training.to_dict()}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'FullSegmentationConfig':
        return cls(
            model=SegmentationModelConfig.from_dict(d.get('model', {})),
            training=SegmentationTrainingConfig.from_dict(d.get('training', {})),
        )

    def save(self, path: str) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def load(cls, path: str) -> 'FullSegmentationConfig':
        return cls.from_dict(json.loads(Path(path).read_text()))


def get_segmentation_config(preset: str = 'tumor') -> FullSegmentationConfig:
    """
    Return a preset segmentation configuration.

    Presets:
      'tumor'  — binary tumour segmentation (2 ch in, 1 ch out)
      'tissue' — 3-class tissue (1 ch in, 3 ch out, soft labels)
      'fast'   — reduced model for quick experiments
      'debug'  — minimal config for unit tests
    """
    presets: Dict[str, FullSegmentationConfig] = {
        'tumor': FullSegmentationConfig(
            model=SegmentationModelConfig(task='tumor', in_channels=2, out_channels=1),
            training=SegmentationTrainingConfig(epochs=100),
        ),
        'tissue': FullSegmentationConfig(
            model=SegmentationModelConfig(
                task='tissue', in_channels=1, out_channels=3,
                use_deep_supervision=True,
            ),
            training=SegmentationTrainingConfig(
                epochs=100,
                tissue_dice_weight=0.7,
                tissue_mse_weight=0.3,
            ),
        ),
        'fast': FullSegmentationConfig(
            model=SegmentationModelConfig(
                task='tumor', in_channels=2, out_channels=1,
                base_channels=16, channel_mult=(1, 2, 4, 8),
                use_cbam=False, use_deep_supervision=False,
            ),
            training=SegmentationTrainingConfig(epochs=50, batch_size=4),
        ),
        'debug': FullSegmentationConfig(
            model=SegmentationModelConfig(
                task='tumor', in_channels=1, out_channels=1,
                base_channels=8, channel_mult=(1, 2, 4),
                use_se=False, use_cbam=False, use_aspp=False,
                use_attn_gate=False, use_deep_supervision=False,
                input_shape=(64, 64, 64),
            ),
            training=SegmentationTrainingConfig(
                epochs=2, batch_size=2, num_workers=0,
                early_stopping_patience=0, use_amp=False,
            ),
        ),
    }
    if preset not in presets:
        raise ValueError(f"Unknown preset '{preset}'. Available: {list(presets)}")
    return presets[preset]
