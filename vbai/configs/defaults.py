"""
Default Configurations for Vbai
"""

from dataclasses import dataclass, field, asdict
from typing import Optional, List, Dict, Any, Literal
import json
import yaml
from pathlib import Path


@dataclass
class ModelConfig:
    """
    Model configuration.
    
    Args:
        variant: Model variant ('f' for fast, 'q' for quality)
        num_dementia_classes: Number of dementia classes
        num_tumor_classes: Number of tumor classes
        use_edge_branch: Whether to use edge detection branch
        dropout: Dropout rate
        image_size: Input image size
    """
    variant: Literal['f', 'q'] = 'q'
    num_dementia_classes: int = 6
    num_tumor_classes: int = 4
    use_edge_branch: bool = True
    dropout: float = 0.5
    image_size: int = 224
    
    # Advanced options
    attention_reduction: int = 16
    backbone_pretrained: bool = False
    
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
