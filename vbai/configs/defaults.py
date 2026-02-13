"""
Default Configurations for Vbai
"""

from dataclasses import dataclass, field, asdict
from typing import Optional, List, Dict, Any, Literal, Tuple
import json
import yaml
from pathlib import Path


@dataclass
class ModelConfig:
    """
    Model configuration.

    Args:
        variant: Model variant ('f' for fast, 'q' for quality)
        tasks: List of tasks to enable ('dementia', 'tumor', or both)
        num_dementia_classes: Number of dementia classes
        num_tumor_classes: Number of tumor classes
        use_edge_branch: Whether to use edge detection branch
        dropout: Dropout rate
        image_size: Input image size
    """
    variant: Literal['f', 'q'] = 'q'
    tasks: List[str] = field(default_factory=lambda: ['dementia', 'tumor'])
    num_dementia_classes: int = 6
    num_tumor_classes: int = 4
    use_edge_branch: bool = True
    dropout: float = 0.5
    image_size: int = 224

    # Advanced options
    attention_reduction: int = 16
    backbone_pretrained: bool = False

    def __post_init__(self):
        """Validate tasks configuration."""
        valid_tasks = {'dementia', 'tumor'}
        for task in self.tasks:
            if task not in valid_tasks:
                raise ValueError(f"Invalid task '{task}'. Valid tasks: {valid_tasks}")
        if not self.tasks:
            raise ValueError("At least one task must be specified")

    @property
    def has_dementia(self) -> bool:
        """Check if dementia task is enabled."""
        return 'dementia' in self.tasks

    @property
    def has_tumor(self) -> bool:
        """Check if tumor task is enabled."""
        return 'tumor' in self.tasks
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'ModelConfig':
        """Create from dictionary."""
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})
    
    def save(self, path: str):
        """Save config to file (JSON or YAML)."""
        path = Path(path)
        data = self.to_dict()
        
        if path.suffix == '.yaml' or path.suffix == '.yml':
            with open(path, 'w') as f:
                yaml.dump(data, f, default_flow_style=False)
        else:
            with open(path, 'w') as f:
                json.dump(data, f, indent=2)
    
    @classmethod
    def load(cls, path: str) -> 'ModelConfig':
        """Load config from file."""
        path = Path(path)
        
        if path.suffix == '.yaml' or path.suffix == '.yml':
            with open(path) as f:
                data = yaml.safe_load(f)
        else:
            with open(path) as f:
                data = json.load(f)
        
        return cls.from_dict(data)


@dataclass
class TrainingConfig:
    """
    Training configuration.
    
    Args:
        epochs: Number of training epochs
        batch_size: Training batch size
        lr: Learning rate
        weight_decay: Weight decay for optimizer
        scheduler: Learning rate scheduler type
        early_stopping_patience: Early stopping patience (0 to disable)
        save_best_only: Only save best model
        checkpoint_dir: Directory for checkpoints
    """
    epochs: int = 10
    batch_size: int = 32
    lr: float = 0.0005
    weight_decay: float = 0.0001
    
    # Scheduler
    scheduler: Optional[str] = 'plateau'  # 'plateau', 'step', 'cosine', None
    scheduler_patience: int = 3
    scheduler_factor: float = 0.5
    
    # Early stopping
    early_stopping_patience: int = 5
    early_stopping_min_delta: float = 0.001
    
    # Checkpointing
    save_best_only: bool = True
    checkpoint_dir: str = './checkpoints'
    checkpoint_monitor: str = 'val_loss'
    
    # Data
    val_split: float = 0.2
    num_workers: int = 4
    pin_memory: bool = True
    
    # Augmentation
    augmentation_strength: str = 'medium'  # 'light', 'medium', 'strong'
    
    # Loss
    dementia_loss_weight: float = 1.0
    tumor_loss_weight: float = 1.0
    label_smoothing: float = 0.0
    
    # Device
    device: str = 'auto'  # 'auto', 'cuda', 'cpu'
    mixed_precision: bool = False
    
    # Logging
    log_interval: int = 10
    tensorboard: bool = False
    tensorboard_dir: str = './logs'
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'TrainingConfig':
        """Create from dictionary."""
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})
    
    def save(self, path: str):
        """Save config to file."""
        path = Path(path)
        data = self.to_dict()
        
        if path.suffix in ['.yaml', '.yml']:
            with open(path, 'w') as f:
                yaml.dump(data, f, default_flow_style=False)
        else:
            with open(path, 'w') as f:
                json.dump(data, f, indent=2)
    
    @classmethod
    def load(cls, path: str) -> 'TrainingConfig':
        """Load config from file."""
        path = Path(path)
        
        if path.suffix in ['.yaml', '.yml']:
            with open(path) as f:
                data = yaml.safe_load(f)
        else:
            with open(path) as f:
                data = json.load(f)
        
        return cls.from_dict(data)


@dataclass
class FullConfig:
    """Combined model and training configuration."""
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'model': self.model.to_dict(),
            'training': self.training.to_dict()
        }
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'FullConfig':
        return cls(
            model=ModelConfig.from_dict(d.get('model', {})),
            training=TrainingConfig.from_dict(d.get('training', {}))
        )
    
    def save(self, path: str):
        """Save full config."""
        path = Path(path)
        data = self.to_dict()
        
        if path.suffix in ['.yaml', '.yml']:
            with open(path, 'w') as f:
                yaml.dump(data, f, default_flow_style=False)
        else:
            with open(path, 'w') as f:
                json.dump(data, f, indent=2)
    
    @classmethod
    def load(cls, path: str) -> 'FullConfig':
        """Load full config."""
        path = Path(path)
        
        if path.suffix in ['.yaml', '.yml']:
            with open(path) as f:
                data = yaml.safe_load(f)
        else:
            with open(path) as f:
                data = json.load(f)
        
        return cls.from_dict(data)


@dataclass
class Model3DConfig:
    """
    3D Model configuration for volumetric NIfTI processing.

    Args:
        variant: Model variant ('f' for fast, 'q' for quality)
        tasks: Dict mapping task names to number of classes
        in_channels: Number of input channels (1 for NIfTI)
        input_shape: Target volume shape (D, H, W)
        dropout: Dropout rate
    """
    variant: Literal['f', 'q'] = 'q'
    tasks: Dict[str, int] = field(default_factory=lambda: {'alzheimer': 3})
    in_channels: int = 1
    input_shape: Tuple[int, int, int] = (96, 96, 96)
    dropout: float = 0.5

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        d = asdict(self)
        d['input_shape'] = list(d['input_shape'])
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'Model3DConfig':
        """Create from dictionary."""
        valid = {k: v for k, v in d.items() if k in cls.__dataclass_fields__}
        if 'input_shape' in valid and isinstance(valid['input_shape'], list):
            valid['input_shape'] = tuple(valid['input_shape'])
        return cls(**valid)

    def save(self, path: str):
        """Save config to file."""
        path = Path(path)
        data = self.to_dict()
        if path.suffix in ['.yaml', '.yml']:
            with open(path, 'w') as f:
                yaml.dump(data, f, default_flow_style=False)
        else:
            with open(path, 'w') as f:
                json.dump(data, f, indent=2)

    @classmethod
    def load(cls, path: str) -> 'Model3DConfig':
        """Load config from file."""
        path = Path(path)
        if path.suffix in ['.yaml', '.yml']:
            with open(path) as f:
                data = yaml.safe_load(f)
        else:
            with open(path) as f:
                data = json.load(f)
        return cls.from_dict(data)


@dataclass
class Training3DConfig:
    """
    3D Training configuration.

    Defaults are tuned for 3D brain MRI (smaller batch size, lower lr).
    """
    epochs: int = 25
    batch_size: int = 4
    lr: float = 1e-4
    weight_decay: float = 1e-4
    scheduler: Optional[str] = 'cosine'
    scheduler_patience: int = 5
    scheduler_factor: float = 0.5
    early_stopping_patience: int = 10
    early_stopping_min_delta: float = 0.001
    save_best_only: bool = True
    checkpoint_dir: str = './checkpoints_3d'
    checkpoint_monitor: str = 'val_loss'
    val_split: float = 0.2
    num_workers: int = 4
    pin_memory: bool = True
    augmentation_strength: str = 'medium'
    device: str = 'auto'
    mixed_precision: bool = True
    gradient_clip: float = 1.0
    log_interval: int = 5

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'Training3DConfig':
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class Full3DConfig:
    """Combined 3D model and training configuration."""
    model: Model3DConfig = field(default_factory=Model3DConfig)
    training: Training3DConfig = field(default_factory=Training3DConfig)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'model': self.model.to_dict(),
            'training': self.training.to_dict()
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'Full3DConfig':
        return cls(
            model=Model3DConfig.from_dict(d.get('model', {})),
            training=Training3DConfig.from_dict(d.get('training', {}))
        )

    def save(self, path: str):
        path = Path(path)
        data = self.to_dict()
        if path.suffix in ['.yaml', '.yml']:
            with open(path, 'w') as f:
                yaml.dump(data, f, default_flow_style=False)
        else:
            with open(path, 'w') as f:
                json.dump(data, f, indent=2)

    @classmethod
    def load(cls, path: str) -> 'Full3DConfig':
        path = Path(path)
        if path.suffix in ['.yaml', '.yml']:
            with open(path) as f:
                data = yaml.safe_load(f)
        else:
            with open(path) as f:
                data = json.load(f)
        return cls.from_dict(data)


def get_default_3d_config(preset: str = 'default') -> Full3DConfig:
    """
    Get a preset 3D configuration.

    Args:
        preset: 'default', 'fast', 'quality', or 'debug'

    Returns:
        Full3DConfig with preset values
    """
    presets = {
        'default': Full3DConfig(
            model=Model3DConfig(variant='q', input_shape=(96, 96, 96)),
            training=Training3DConfig(epochs=25, batch_size=4, lr=1e-4)
        ),
        'fast': Full3DConfig(
            model=Model3DConfig(variant='f', input_shape=(80, 80, 80)),
            training=Training3DConfig(epochs=10, batch_size=6, lr=5e-4)
        ),
        'quality': Full3DConfig(
            model=Model3DConfig(variant='q', input_shape=(96, 96, 96), dropout=0.3),
            training=Training3DConfig(
                epochs=50, batch_size=4, lr=5e-5,
                augmentation_strength='strong',
                early_stopping_patience=15
            )
        ),
        'debug': Full3DConfig(
            model=Model3DConfig(variant='f', input_shape=(64, 64, 64)),
            training=Training3DConfig(
                epochs=2, batch_size=2,
                num_workers=0,
                early_stopping_patience=0,
                mixed_precision=False
            )
        ),
    }

    if preset not in presets:
        available = ', '.join(presets.keys())
        raise ValueError(f"Unknown 3D preset '{preset}'. Available: {available}")

    return presets[preset]


def get_default_config(preset: str = 'default') -> FullConfig:
    """
    Get a preset configuration.
    
    Args:
        preset: Configuration preset name
            - 'default': Balanced configuration
            - 'fast': Quick training with smaller model
            - 'quality': High quality with more epochs
            - 'debug': Minimal config for debugging
    
    Returns:
        FullConfig with preset values
    """
    presets = {
        'default': FullConfig(
            model=ModelConfig(variant='q'),
            training=TrainingConfig(epochs=10, batch_size=32, lr=0.0005)
        ),
        'fast': FullConfig(
            model=ModelConfig(variant='f'),
            training=TrainingConfig(epochs=5, batch_size=64, lr=0.001)
        ),
        'quality': FullConfig(
            model=ModelConfig(variant='q', dropout=0.3),
            training=TrainingConfig(
                epochs=30,
                batch_size=16,
                lr=0.0001,
                scheduler='cosine',
                augmentation_strength='strong',
                early_stopping_patience=10
            )
        ),
        'debug': FullConfig(
            model=ModelConfig(variant='f'),
            training=TrainingConfig(
                epochs=2,
                batch_size=4,
                num_workers=0,
                early_stopping_patience=0
            )
        ),
    }
    
    if preset not in presets:
        available = ', '.join(presets.keys())
        raise ValueError(f"Unknown preset '{preset}'. Available: {available}")
    
    return presets[preset]
