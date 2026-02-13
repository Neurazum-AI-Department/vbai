# Vbai - Visual Brain AI

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.9+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A PyTorch-based deep learning library for multi-task brain MRI analysis. Train models for dementia classification, brain tumor detection, and 3D volumetric NIfTI analysis with just a few lines of code.

## Features

- **2D & 3D Support**: RGB image analysis and NIfTI (.nii/.nii.gz) volumetric processing
- **Flexible Task Selection**: Train for dementia only, tumor only, or both simultaneously
- **Easy to Use**: Keras-like API for quick training
- **Task-Specific Attention**: Separate attention mechanisms for each task
- **HuggingFace Hub**: Share and download models via HuggingFace
- **ONNX Export**: Production deployment for both 2D and 3D models
- **Model Zoo**: Registry of pretrained model configurations
- **Visualization**: Built-in attention heatmap visualization
- **Configurable**: YAML/JSON configuration support

## Supported Classifications

**2D - Dementia (6 classes):**
- AD Alzheimer's Disease
- AD Mild Demented
- AD Moderate Demented
- AD Very Mild Demented
- CN Non-Demented (Cognitively Normal)
- PD Parkinson's Disease

**2D - Brain Tumor (4 classes):**
- Glioma
- Meningioma
- No Tumor
- Pituitary

**3D - Alzheimer's (NIfTI):**
- CN (Cognitively Normal)
- MCI (Mild Cognitive Impairment)
- AD (Alzheimer's Disease)

## Installation

```bash
# Basic installation
pip install vbai

# With NIfTI (3D) support
pip install vbai[nifti]

# With HuggingFace Hub integration
pip install vbai[hub]

# With ONNX export
pip install vbai[onnx]

# With all optional dependencies
pip install vbai[full]

# Development installation
git clone https://github.com/Neurazum-AI-Department/vbai.git
cd vbai
pip install -e .[dev]
```

## Quick Start

### 2D Training (Dementia & Tumor)

```python
import vbai

# Create model for both tasks (default)
model = vbai.MultiTaskBrainModel(variant='q')  # 'q' for quality, 'f' for fast

# Prepare dataset
dataset = vbai.UnifiedMRIDataset(
    dementia_path='./data/dementia/train',
    tumor_path='./data/tumor/train',
    is_training=True
)

# Create trainer and train
trainer = vbai.Trainer(model=model, lr=0.0005, device='cuda')
history = trainer.fit(train_data=dataset, epochs=10, batch_size=32)
trainer.save('brain_model.pt')
```

### 3D NIfTI Training (Volumetric)

```python
import vbai

# Create 3D model
model = vbai.MultiTask3DBrainModel(
    variant='q',
    tasks={'alzheimer': 3},
    input_shape=(96, 96, 96)
)

# Create NIfTI dataset
dataset = vbai.NIfTIDataset(
    root='./data/alzheimer_3d',  # Subfolders: CN/, MCI/, AD/
    target_shape=(96, 96, 96),
    is_training=True
)

# Dataloaders
train_loader, val_loader = vbai.create_3d_dataloaders(
    root='./data/alzheimer_3d',
    batch_size=4, val_split=0.2
)

# Train
trainer = vbai.Trainer3D(model=model, lr=1e-4, device='cuda')
history = trainer.fit(train_loader, val_loader, epochs=25)
trainer.save('alzheimer_3d.pt')
```

### Single-Task Training

```python
import vbai

# Dementia only
model = vbai.MultiTaskBrainModel(variant='q', tasks=['dementia'])

# Tumor only
model = vbai.MultiTaskBrainModel(variant='q', tasks=['tumor'])
```

### Making Predictions

```python
import vbai

# 2D prediction
model = vbai.load('brain_model.pt', device='cuda')
result = model.predict('brain_scan.jpg')
print(f"Dementia: {result.dementia_class} ({result.dementia_confidence:.1%})")
print(f"Tumor: {result.tumor_class} ({result.tumor_confidence:.1%})")

# 3D NIfTI prediction
model_3d = vbai.load_3d('alzheimer_3d.pt', device='cuda')
result = model_3d.predict('scan.nii.gz', task='alzheimer',
                          class_names=['CN', 'MCI', 'AD'])
print(f"{result.predicted_class}: {result.confidence:.1%}")
```

### HuggingFace Hub

```python
import vbai

# List available models
models = vbai.list_models()        # All models
models_3d = vbai.list_models('3d') # Only 3D models

# Download and load from Hub
model = vbai.from_hub('Neurazum/vbai-3d-q', device='cuda')

# Push your trained model to Hub
model.push_to_hub('username/my-brain-model')

# Or use the functional API
vbai.push_to_hub(model, 'username/my-brain-model', private=True)
```

### ONNX Export

```python
import vbai

# Export 2D model
model_2d = vbai.MultiTaskBrainModel(variant='q')
model_2d.export_onnx('model_2d.onnx')

# Export 3D model
model_3d = vbai.MultiTask3DBrainModel(variant='q', tasks={'alzheimer': 3})
model_3d.export_onnx('model_3d.onnx')

# Or use the functional API
vbai.export_onnx(model_2d, 'model_2d.onnx')

# PyTorch-free inference with ONNX
onnx_model = vbai.ONNXModel('model_3d.onnx')
output = onnx_model.predict_nifti('brain_scan.nii.gz')
probs = onnx_model.softmax(output)
```

### Using Callbacks

```python
import vbai

model = vbai.MultiTaskBrainModel(variant='q')

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

# 3D configuration
config_3d = vbai.get_default_3d_config('quality')

# Custom config
model_config = vbai.ModelConfig(
    variant='q',
    tasks=['dementia', 'tumor'],
    dropout=0.3,
    use_edge_branch=True
)
```

## Command Line Interface

```bash
# Train 2D model
vbai-train --dementia_path ./data/dementia --tumor_path ./data/tumor --epochs 10

# Single-task training
vbai-train --dementia_path ./data/dementia --tasks dementia --epochs 10

# Prediction
vbai-predict --model brain_model.pt --image brain_scan.jpg

# With visualization
vbai-predict --model brain_model.pt --image brain_scan.jpg --visualize --output result.png
```

## Model Variants

### 2D Models

| Variant | Layers | Channels | Speed | Accuracy |
|---------|--------|----------|-------|----------|
| `f` (fast) | 3 | 32-64-128 | Fast | Good |
| `q` (quality) | 4 | 64-128-256-512 | Slower | Better |

### 3D Models (NIfTI)

| Variant | Stages | Channels | Speed | Accuracy |
|---------|--------|----------|-------|----------|
| `f` (fast) | 3x1 blocks | 32-64-128 | Fast | Good |
| `q` (quality) | 3x2 blocks | 64-128-256 | Slower | Better |

## Dataset Structure

### 2D (RGB Images)

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
└── tumor/
    ├── train/
    │   ├── Glioma/
    │   ├── Meningioma/
    │   ├── No_Tumor/
    │   └── Pituitary/
    └── val/
```

### 3D (NIfTI Volumes)

```
data/alzheimer_3d/
├── CN/
│   ├── subject_001.nii.gz
│   └── subject_002.nii.gz
├── MCI/
│   └── subject_003.nii.gz
└── AD/
    └── subject_004.nii.gz
```

## API Reference

### Core Classes

- `MultiTaskBrainModel` - 2D multi-task model (dementia + tumor)
- `MultiTask3DBrainModel` - 3D volumetric model (NIfTI)
- `Trainer` - 2D training loop manager
- `Trainer3D` - 3D training loop manager

### Data

- `UnifiedMRIDataset` - 2D dataset (RGB images)
- `NIfTIDataset` - 3D dataset (NIfTI volumes)
- `UnifiedNIfTIDataset` - 3D multi-task dataset

### Hub & Export

- `list_models()` / `get_model_info()` - Model zoo registry
- `from_hub()` / `push_to_hub()` - HuggingFace Hub integration
- `export_onnx()` - ONNX export (2D & 3D)
- `ONNXModel` - PyTorch-free ONNX inference

### Configuration

- `ModelConfig` / `Model3DConfig` - Architecture settings
- `TrainingConfig` / `Training3DConfig` - Training hyperparameters
- `get_default_config()` / `get_default_3d_config()` - Presets

### Callbacks

- `EarlyStopping` - Stop when no improvement
- `ModelCheckpoint` - Save best/all checkpoints
- `TensorBoardLogger` - Log to TensorBoard

## Examples

See the `examples/` directory:

- `train_basic.py` - Basic 2D training
- `train_advanced.py` - Advanced training with callbacks
- `train_3d.py` - 3D NIfTI training
- `inference.py` - Model inference

## Citation

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
