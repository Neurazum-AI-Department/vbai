"""Configurations for VbaiProgressionNet multimodal progression model."""

from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple, Any
import json
from pathlib import Path


@dataclass
class ProgressionModelConfig:
    """
    VbaiProgressionNet architecture configuration.

    Args:
        mri_in_channels: MRI input channels (1 for T1).
        n_tabular_features: Number of biomarker features (default 13).
        num_classes: Classification targets (default 3: CN/MCI/AD).
        mri_feature_dim: MRI encoder embedding dimension.
        tab_feature_dim: Tabular encoder embedding dimension.
        fusion_dim: Cross-modal fusion dimension.
        num_attn_heads: Attention heads in fusion.
        max_progression_months: Upper bound for time prediction.
        n_time_bins: Number of bins in time distribution head.
        mri_dropout: MRI encoder dropout.
        tab_dropout: Tabular encoder dropout.
        fusion_dropout: Fusion module dropout.
    """
    mri_in_channels: int = 1
    n_tabular_features: int = 13
    num_classes: int = 3
    mri_feature_dim: int = 512
    tab_feature_dim: int = 256
    fusion_dim: int = 512
    num_attn_heads: int = 8
    max_progression_months: int = 120
    n_time_bins: int = 24
    mri_dropout: float = 0.4
    tab_dropout: float = 0.3
    fusion_dropout: float = 0.3

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'ProgressionModelConfig':
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    def save(self, path: str) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def load(cls, path: str) -> 'ProgressionModelConfig':
        return cls.from_dict(json.loads(Path(path).read_text()))


@dataclass
class ProgressionTrainingConfig:
    """
    Training configuration for ProgressionTrainer (3-phase schedule).

    Phase 1 — MRI encoder pretrain
    Phase 2 — Tabular encoder pretrain
    Phase 3 — Joint fusion fine-tuning
    """
    # Phase 1
    phase1_epochs: int = 40
    phase1_lr: float = 3e-4
    phase1_weight_decay: float = 1e-4
    phase1_batch_size: int = 4
    phase1_es_patience: int = 20

    # Phase 2
    phase2_epochs: int = 60
    phase2_lr: float = 1e-3
    phase2_weight_decay: float = 1e-4
    phase2_batch_size: int = 64
    phase2_es_patience: int = 20

    # Phase 3
    phase3_epochs: int = 40
    phase3_lr_backbone: float = 1e-5
    phase3_lr_fusion: float = 5e-4
    phase3_weight_decay: float = 1e-4
    phase3_batch_size: int = 4
    phase3_es_patience: int = 20

    # Loss weights
    w_fused: float = 1.0
    w_mri: float = 0.3
    w_tab: float = 0.3
    w_prog: float = 0.5
    w_contrastive: float = 0.2
    focal_gamma: float = 1.0
    label_smoothing: float = 0.05

    # Data
    val_split: float = 0.15
    test_split: float = 0.15
    feature_mask_prob: float = 0.20
    pair_window_months: int = 6
    progression_horizon_months: int = 60

    # Infrastructure
    grad_clip: float = 1.0
    use_amp: bool = True
    num_workers: int = 4
    pin_memory: bool = True
    checkpoint_dir: str = './checkpoints'
    checkpoint_name: str = 'vbai_prog'

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'ProgressionTrainingConfig':
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class FullProgressionConfig:
    """Combined model + training configuration for progression model."""
    model: ProgressionModelConfig = field(default_factory=ProgressionModelConfig)
    training: ProgressionTrainingConfig = field(default_factory=ProgressionTrainingConfig)

    def to_dict(self) -> Dict[str, Any]:
        return {'model': self.model.to_dict(), 'training': self.training.to_dict()}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'FullProgressionConfig':
        return cls(
            model=ProgressionModelConfig.from_dict(d.get('model', {})),
            training=ProgressionTrainingConfig.from_dict(d.get('training', {})),
        )

    def save(self, path: str) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def load(cls, path: str) -> 'FullProgressionConfig':
        return cls.from_dict(json.loads(Path(path).read_text()))


def get_progression_config(preset: str = 'default') -> FullProgressionConfig:
    """
    Return a preset progression configuration.

    Presets:
      'default' — standard full-resolution training
      'fast'    — smaller embeddings, fewer epochs
      'debug'   — minimal config for unit tests
    """
    presets: Dict[str, FullProgressionConfig] = {
        'default': FullProgressionConfig(),
        'fast': FullProgressionConfig(
            model=ProgressionModelConfig(
                mri_feature_dim=256, tab_feature_dim=128, fusion_dim=256,
                num_attn_heads=4,
            ),
            training=ProgressionTrainingConfig(
                phase1_epochs=15, phase2_epochs=20, phase3_epochs=15,
            ),
        ),
        'debug': FullProgressionConfig(
            model=ProgressionModelConfig(
                mri_feature_dim=64, tab_feature_dim=32, fusion_dim=64,
                num_attn_heads=2,
            ),
            training=ProgressionTrainingConfig(
                phase1_epochs=2, phase2_epochs=2, phase3_epochs=2,
                phase1_batch_size=2, phase2_batch_size=2, phase3_batch_size=2,
                num_workers=0, use_amp=False,
            ),
        ),
    }
    if preset not in presets:
        raise ValueError(f"Unknown preset '{preset}'. Available: {list(presets)}")
    return presets[preset]
