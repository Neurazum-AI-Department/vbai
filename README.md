# Vbai - Visual Brain AI

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.9+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A PyTorch-based deep learning library for multi-task brain MRI analysis. Train models for simultaneous dementia classification and brain tumor detection with just a few lines of code.

## Features

- 🧠 **Multi-Task Learning**: Simultaneous dementia and tumor classification
- ⚡ **Easy to Use**: Keras-like API for quick training
- 🎯 **Dual Attention**: Task-specific attention mechanisms
- 📊 **Visualization**: Built-in attention heatmap visualization
- 🔧 **Configurable**: YAML/JSON configuration support
- 📦 **Production Ready**: Export and deploy trained models

## Supported Classifications

**Dementia (6 classes):**
- AD Alzheimer's Disease
- AD Mild Demented
- AD Moderate Demented  
- AD Very Mild Demented
- CN Non-Demented (Cognitively Normal)
- PD Parkinson's Disease

**Brain Tumor (4 classes):**
- Glioma
- Meningioma
- No Tumor
- Pituitary

## Installation

```bash
# Basic installation
pip install vbai

# With all optional dependencies
pip install vbai[full]

# Development installation
git clone https://github.com/Neurazum-AI-Department/vbai.git
cd vbai
pip install -e .[dev]
```

## Quick Start

### Training a Model

```python
import vbai

# Create model
model = vbai.MultiTaskBrainModel(variant='q')  # 'q' for quality, 'f' for fast

# Prepare dataset
dataset = vbai.UnifiedMRIDataset(
    dementia_path='./data/dementia/train',
    tumor_path='./data/tumor/train',
    is_training=True
)

# Create trainer
trainer = vbai.Trainer(
    model=model,
    lr=0.0005,
    device='cuda'
)

# Train
history = trainer.fit(
    train_data=dataset,
    epochs=10,
    batch_size=32
)

# Save model
trainer.save('brain_model.pt')
```

### Making Predictions

```python
import vbai

# Load trained model
model = vbai.load('brain_model.pt', device='cuda')

# Single image prediction
result = model.predict('brain_scan.jpg')
print(f"Dementia: {result.dementia_class} ({result.dementia_confidence:.1%})")
print(f"Tumor: {result.tumor_class} ({result.tumor_confidence:.1%})")

# With attention visualization
result = model.predict('brain_scan.jpg', return_attention=True)
vis = vbai.VisualizationManager()
vis.visualize('brain_scan.jpg', result, save=True)
```

### Using Callbacks

```python
import vbai

model = vbai.MultiTaskBrainModel(variant='q')

# Setup callbacks
callbacks = [
    vbai.EarlyStopping(monitor='val_loss', patience=5),
    vbai.ModelCheckpoint(
        filepath='checkpoints/model_{epoch:02d}.pt',
        monitor='val_loss',
        save_best_only=True
    )
]

trainer = vbai.Trainer(model=model, callbacks=callbacks)
trainer.fit(train_data, val_data, epochs=50)
```

### Configuration

```python
import vbai

# Use preset configurations
config = vbai.get_default_config('quality')  # 'default', 'fast', 'quality', 'debug'

# Or customize
model_config = vbai.ModelConfig(
    variant='q',
    dropout=0.3,
    use_edge_branch=True
)

training_config = vbai.TrainingConfig(
    epochs=20,
    batch_size=16,
    lr=0.0001,
    scheduler='cosine'
)

# Save/Load configs
model_config.save('model_config.yaml')
loaded_config = vbai.ModelConfig.load('model_config.yaml')
```

## Model Variants

| Variant | Layers | Channels | Speed | Accuracy |
|---------|--------|----------|-------|----------|
| `f` (fast) | 3 | 32→64→128 | ⚡⚡⚡ | ⭐⭐ |
| `q` (quality) | 4 | 64→128→256→512 | ⚡ | ⭐⭐⭐ |

## Dataset Structure

Your dataset should be organized as follows:

```
data/
├── dementia/
│   ├── train/
│   │   ├── AD_Alzheimer/
│   │   ├── AD_Mild_Demented/
│   │   ├── AD_Moderate_Demented/
│   │   ├── AD_Very_Mild_Demented/
│   │   ├── CN_Non_Demented/
│   │   └── PD_Parkinson/
│   └── val/
│       └── ...
└── tumor/
    ├── train/
    │   ├── Glioma/
    │   ├── Meningioma/
    │   ├── No_Tumor/
    │   └── Pituitary/
    └── val/
        └── ...
```

## API Reference

### Core Classes

- `MultiTaskBrainModel` - Main model class
- `UnifiedMRIDataset` - Dataset for multi-task training
- `Trainer` - Training loop manager
- `VisualizationManager` - Attention heatmap visualization

### Configuration

- `ModelConfig` - Model architecture settings
- `TrainingConfig` - Training hyperparameters
- `get_default_config()` - Preset configurations

### Callbacks

- `EarlyStopping` - Stop when no improvement
- `ModelCheckpoint` - Save best/all checkpoints
- `TensorBoardLogger` - Log to TensorBoard

## Examples

See the `examples/` directory for complete examples:

- `train_basic.py` - Basic training example
- `train_advanced.py` - Advanced training with callbacks
- `inference.py` - Model inference
- `visualize.py` - Attention visualization

## Citation

If you use Vbai in your research, please cite:

```bibtex
@software{vbai,
  title = {Vbai: Visual Brain AI Library},
  author = {Neurazum},
  year = {2025},
  url = {https://github.com/Neurazum-AI-Department/vbai}
}
```

## License

MIT License - see [LICENSE](LICENSE) for details.

## Contributing

_Is being planned..._

### Support

- **Website**: [Neurazum](https://neurazum.com) - [HealFuture](https://healfuture.com)
- **Email**: [contact@neurazum.com](mailto:contact@neurazum.com)

---

<span style="color: #ff8d26; "><b>Neurazum</b> AI Department</span>
