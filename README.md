# Vbai - Visual Brain AI

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.9+-red.svg)](https://pytorch.org/)
[![Version](https://img.shields.io/badge/version-0.3.0-green.svg)](https://github.com/Neurazum-AI-Department/vbai)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A professional PyTorch library for 3D brain MRI analysis.  
Train state-of-the-art models for **tumor/tissue segmentation** and
**multimodal Alzheimer's progression prediction** with a clean, Keras-like API.

---

## What's New in 1.2.2

- **VbaiSegNet3D** — 3D UNet with SE, CBAM, ASPP, Attention Gates, and Deep Supervision (~25M params)
- **VbaiProgressionNet** — Multimodal fusion of 3D MRI + 13 biomarkers for CN/MCI/AD prediction and progression timeline estimation (~16M params)
- **3-Phase Training Pipeline** — MRI pretraining → Tabular pretraining → Joint fusion with differential learning rates
- **Per-Epoch Fit Diagnosis** — Each epoch prints whether the model is Underfitting, Overfitting, or a Good Fit
- **Expanded ONNX Export** — Segmentation and progression models now exportable to ONNX (3 modes for progression)
- **Clinical Visualization** — Risk gauge, progression timeline histogram, biomarker radar chart, and printable report figure

---

## Features

| Capability | Details |
|------------|---------|
| **3D Tumor Segmentation** | VbaiSegNet3D: SE+CBAM+ASPP+AttGate+DeepSup, sliding-window inference |
| **Progression Prediction** | VbaiProgressionNet: MRI encoder + tabular encoder + cross-modal fusion |
| **2D Classification** | MultiTaskBrainModel: dementia (6 classes) + tumor (4 classes) |
| **3D Classification** | MultiTask3DBrainModel: CN / MCI / AD from NIfTI volumes |
| **MRI Augmentation** | Bias field, ghosting, spike noise, Rician noise, elastic deformation, MixUp, CutMix, AutoAugment |
| **Fit Diagnosis** | Underfitting / Overfitting / Slight Overfitting / Good Fit printed every epoch |
| **ONNX Export** | All model types — segmentation (single tensor), progression (3 modes) |
| **HuggingFace Hub** | Push / pull trained models |
| **Clinical Reports** | Matplotlib figures: risk gauge, timeline, biomarker radar |
| **Configurable** | YAML-friendly dataclass presets for every model type |

---

## Installation

```bash
# Core (PyTorch only)
pip install vbai

# With NIfTI / 3D support
pip install vbai[nifti]

# With ONNX export / inference
pip install vbai[onnx]

# With HuggingFace Hub
pip install vbai[hub]

# Everything
pip install vbai[full]

# Development
git clone https://github.com/Neurazum-AI-Department/vbai.git
cd vbai
pip install -e .[dev]
```

---

## Quick Start — 3D Tumor Segmentation

```python
import vbai

# Build model (4 MRI channels: T1, T1ce, T2, FLAIR)
model = vbai.VbaiSegNet3D(
    in_channels=4,
    out_channels=1,          # binary tumor mask
    base_channels=32,
    use_deep_supervision=True,
)

# Create datasets from NIfTI files
train_loader, val_loader, test_loader = vbai.create_segmentation_dataloaders(
    dataset=vbai.TumorSegmentationDataset(
        root='./data/tumor',
        modality_files=['T1.nii.gz', 'T1ce.nii.gz', 'T2.nii.gz', 'FLAIR.nii.gz'],
        mask_file='mask.nii.gz',
        target_shape=(128, 128, 128),
        is_training=True,
    ),
    val_split=0.15,
    test_split=0.10,
    batch_size=2,
)

# Train — fit status printed every epoch
trainer = vbai.SegmentationTrainer(model, device='cuda')
history = trainer.fit(train_loader, val_loader, epochs=100)

# Sliding-window inference on arbitrary volume
import nibabel as nib
import numpy as np
volume = nib.load('patient.nii.gz').get_fdata()
volume = (volume - volume.mean()) / (volume.std() + 1e-8)
mask = model.predict_volume(volume, threshold=0.5, patch_size=(128, 128, 128), overlap=0.5)
```

---

## Quick Start — Progression Prediction

```python
import vbai

# Build multimodal model
model = vbai.VbaiProgressionNet(
    mri_in_channels=1,
    num_classes=3,           # CN / MCI / AD
    max_time_months=120,
)

# Prepare records — one dict per subject visit
records = [
    {
        'ptid': 'sub-001',
        'mri_path': '/data/sub-001/T1.nii.gz',
        'label': 1,           # 0=CN, 1=MCI, 2=AD
        'has_progression': True,
        'will_progress': 1,
        'progression_months': 18,
        'Age': 72, 'Sex': 1, 'MMSE': 26.0,
        # ... other biomarkers (NaN for missing)
    },
    # ...
]

# Fit normalizer on training split
normalizer = vbai.TabularNormalizer()
normalizer.fit(records)

# Create dataloaders
loaders = vbai.create_progression_dataloaders(
    records, normalizer,
    mode='multi',            # 'mri', 'tab', or 'multi'
    batch_size=8,
)

# 3-phase training — each phase prints Underfitting / Overfitting / Good Fit
trainer = vbai.ProgressionTrainer(model, device='cuda')
trainer.fit(
    mri_loader=loaders['mri_train'],
    tab_loader=loaders['tab_train'],
    full_loader=loaders['multi_train'],
    mri_val_loader=loaders['mri_val'],
    tab_val_loader=loaders['tab_val'],
    full_val_loader=loaders['multi_val'],
)

# Clinical inference
import torch
mri_tensor = torch.randn(1, 1, 96, 96, 96)      # pre-processed NIfTI
tab_array  = normalizer.transform(records[0])    # (26,) numpy array
tab_tensor = torch.tensor(tab_array).unsqueeze(0)

prediction = model.predict(
    mri=mri_tensor,
    tab=tab_tensor,
    class_names=['CN', 'MCI', 'AD'],
)
# prediction = {
#   'predicted_class': 'MCI',
#   'class_probabilities': {'CN': 0.12, 'MCI': 0.71, 'AD': 0.17},
#   'will_progress': True,
#   'progression_probability': 0.83,
#   'estimated_months_to_conversion': 21.4,
#   'risk_category': 'High Risk',
# }

# Generate printable clinical report
vbai.plot_progression_report(
    prediction,
    biomarker_values={'Age': 72, 'MMSE': 26, 'APOE4_count': 1},
    subject_id='sub-001',
    scan_date='2026-06-07',
    save_path='report.png',
)
```

---

## Fit Diagnosis — Every Epoch

Both `SegmentationTrainer` and `ProgressionTrainer` automatically append a
fit status label to each epoch line:

```
Epoch 012/100 | Train Loss 0.4231 | Train Dice 0.6814 | Val Loss 0.5102 | Val Dice 0.5021 | LR 9.23e-05 | 14.3s | Slight Overfitting
Epoch 013/100 | Train Loss 0.3987 | Train Dice 0.7102 | Val Loss 0.4891 | Val Dice 0.6543 | LR 8.80e-05 | 14.1s [best] | Good Fit
```

| Status | Condition |
|--------|-----------|
| **Underfitting** | Train metric below learning threshold |
| **Slight Underfitting** | Both train and val are moderate, gap near zero |
| **Good Fit** | Healthy train/val gap |
| **Slight Overfitting** | Train-val gap is moderate (>7% Dice / >10% Acc) |
| **Overfitting** | Large train-val gap (>15% Dice / >20% Acc) |

---

## ONNX Export

```python
import vbai

# --- Segmentation ---
seg_model = vbai.VbaiSegNet3D(in_channels=4, out_channels=1)
# Deep supervision automatically disabled for export
vbai.export_segmentation_onnx(seg_model, 'tumor_seg.onnx')
# Output: segmentation_logits (B, 1, D, H, W)

# --- Progression (3 modes) ---
prog_model = vbai.VbaiProgressionNet()

# Multimodal (MRI + biomarkers)
vbai.export_progression_onnx(prog_model, 'prog_multi.onnx', mode='multi')

# MRI only
vbai.export_progression_onnx(prog_model, 'prog_mri.onnx', mode='mri')

# Biomarkers only
vbai.export_progression_onnx(prog_model, 'prog_tab.onnx', mode='tab')

# All modes output 3 tensors:
#   class_logits          (B, 3)
#   will_progress_logits  (B, 1)
#   time_to_conversion    (B, 1)

# --- Auto-dispatch (works for all model types) ---
vbai.export_onnx(seg_model, 'seg.onnx')
vbai.export_onnx(prog_model, 'prog.onnx')   # defaults to 'multi' mode

# --- Inference (no PyTorch needed) ---
onnx_model = vbai.ONNXModel('tumor_seg.onnx')
```

---

## Biomarker Reference

`VbaiProgressionNet` accepts 13 biomarkers (missing values → NaN → handled automatically).

| Index | Feature | Description |
|-------|---------|-------------|
| 0 | Age | Subject age in years |
| 1 | Sex | 0 = Female, 1 = Male |
| 2 | MMSE | Mini-Mental State Examination (0–30) |
| 3 | CDRSB | Clinical Dementia Rating Sum of Boxes |
| 4 | APOE4_count | APOE ε4 allele count (0, 1, 2) |
| 5 | CSF_ABETA42 | CSF Amyloid beta 1-42 (pg/mL) |
| 6 | CSF_TAU | CSF Total tau (pg/mL) |
| 7 | CSF_PTAU | CSF Phospho-tau 181 (pg/mL) |
| 8 | CSF_AB42_AB40 | CSF Abeta42/Abeta40 ratio |
| 9 | PLASMA_PTAU | Plasma Phospho-tau 217 (pg/mL) |
| 10 | PLASMA_NFL | Plasma Neurofilament light (pg/mL) |
| 11 | PLASMA_AB42_AB40 | Plasma Abeta42/Abeta40 ratio |
| 12 | PLASMA_GFAP | Plasma GFAP (pg/mL) |

The `TabularNormalizer` creates a 26-dimensional vector: 13 normalized values + 13 binary
missingness masks. Use `normalizer.fit(records)` then `normalizer.transform(record)`.

---

## Dataset Structure

### Tumor Segmentation

```
data/tumor/
  subject_001/
    T1.nii.gz
    T1ce.nii.gz
    T2.nii.gz
    FLAIR.nii.gz
    mask.nii.gz        # binary tumor mask
  subject_002/
    ...
```

### Tissue Segmentation

```
data/tissue/
  images/
    sub-001_T1.nii.gz
  masks/
    sub-001_GM.nii.gz   # grey matter soft label
    sub-001_WM.nii.gz   # white matter soft label
    sub-001_CSF.nii.gz  # CSF soft label
```

### Progression Records

```python
records = [
    {
        'ptid': 'sub-001',        # subject ID (for train/val/test split)
        'mri_path': 'T1.nii.gz', # path to NIfTI file
        'label': 1,               # 0=CN, 1=MCI, 2=AD
        'has_progression': True,  # is there a follow-up conversion event?
        'will_progress': 1,       # 1 if MCI->AD conversion occurred
        'progression_months': 18, # months until conversion (0 if no event)
        'Age': 72,
        'MMSE': 26.0,
        # ... other biomarkers (omit or set to NaN if unknown)
    }
]
```

### 2D Classification (legacy)

```
data/
  dementia/
    train/  AD_Alzheimer/ | AD_Mild_Demented/ | CN_Non_Demented/ | PD_Parkinson/ | ...
    val/    ...
  tumor/
    train/  Glioma/ | Meningioma/ | No_Tumor/ | Pituitary/
    val/    ...
```

---

## Configuration Presets

### Segmentation

```python
from vbai.configs import get_segmentation_config

config = get_segmentation_config('tumor')   # 'tumor' | 'tissue' | 'fast' | 'debug'
model  = config.build_model()
```

| Preset | Channels | Deep Supervision | Use Case |
|--------|----------|-----------------|---------|
| `tumor` | 32, stride patch 128 | Yes | Multi-modal tumor segmentation |
| `tissue` | 24, out_channels=3 | No | Grey matter / WM / CSF |
| `fast` | 16, patch 64 | No | Quick experiments |
| `debug` | 8, patch 32 | No | Unit tests / CI |

### Progression

```python
from vbai.configs import get_progression_config

config = get_progression_config('default')  # 'default' | 'fast' | 'debug'
model  = config.build_model()
```

---

## Model Architectures

### VbaiSegNet3D

3D encoder-decoder UNet variant (~25M parameters):

```
Input (B, C, D, H, W)
  └─ Stem Conv
  └─ EncoderBlock x4  [SE + CBAM + ResBlocks, stride pooling]
  └─ ASPP3D bottleneck (dilations: 1, 2, 4, 8)
  └─ DecoderBlock x4  [Attention Gate + transposed conv + skip]
  └─ Output head  → logits (B, out_channels, D, H, W)
     [+ 4 auxiliary heads for deep supervision during training]
```

### VbaiProgressionNet

Multimodal fusion network (~16M parameters):

```
MRI volume ──► MRIEncoder3D ──► 512-d embedding (zm)
                   (4-stage ResBlock3D + DropPath + ASPP3D)
Biomarkers ──► TabularEncoder ──► 256-d embedding (zt)
                   (MLP 26→128→256, LayerNorm)
         ┌─────────────────────────┐
         │    CrossModalFusion     │
         │  Bidirectional MHA      │
         │  + Gated blend → 512-d  │
         └─────────────────────────┘
                      │
         ┌────────────┼────────────┐
         ▼            ▼            ▼
  ClassHead       ProgressionHead  ContrastiveProj
  (CN/MCI/AD)  (will_progress,      (InfoNCE loss)
               time_to_conversion,
               time_distribution)
```

---

## API Reference

### Models

| Class | Task | Key Args |
|-------|------|---------|
| `VbaiSegNet3D` | 3D segmentation | `in_channels`, `out_channels`, `base_channels`, `use_deep_supervision` |
| `VbaiProgressionNet` | Multimodal progression | `mri_in_channels`, `num_classes`, `max_time_months` |
| `MultiTaskBrainModel` | 2D classification | `variant`, `tasks` |
| `MultiTask3DBrainModel` | 3D NIfTI classification | `variant`, `tasks`, `input_shape` |

### Training

| Class | Use For |
|-------|---------|
| `SegmentationTrainer` | VbaiSegNet3D |
| `ProgressionTrainer` | VbaiProgressionNet (3-phase) |
| `Trainer` | MultiTaskBrainModel (2D) |
| `Trainer3D` | MultiTask3DBrainModel |

### Losses

| Class | Purpose |
|-------|---------|
| `TumorSegmentationLoss` | Dice + Focal for binary tumor masks |
| `TissueSegmentationLoss` | Dice + MSE for soft tissue labels |
| `DeepSupervisionLoss` | Weighted multi-scale supervision |
| `VbaiProgressionLoss` | Combined fused/MRI/tabular/progression/InfoNCE |

### Data

| Class / Function | Purpose |
|-----------------|---------|
| `TumorSegmentationDataset` | NIfTI volumes + binary mask |
| `TissueSegmentationDataset` | NIfTI volumes + 3-channel soft masks |
| `ProgressionDataset` | MRI + tabular records |
| `TabularNormalizer` | Robust normalization + missingness masks |
| `create_segmentation_dataloaders` | Train / val / test split |
| `create_progression_dataloaders` | Subject-level split (no leakage) |

### Visualization

| Function | Output |
|---------|--------|
| `plot_segmentation_slices` | Axial / coronal / sagittal slices with overlay |
| `compute_segmentation_metrics` | Dice, IoU, Volume Similarity per class |
| `plot_training_curves` | Loss + Dice vs epoch |
| `plot_progression_report` | Full clinical figure (saves to file) |
| `create_report_figure` | Risk gauge + timeline + biomarker radar |

### Export

| Function | Purpose |
|----------|---------|
| `export_onnx` | Auto-dispatch for all model types |
| `export_segmentation_onnx` | VbaiSegNet3D → ONNX |
| `export_progression_onnx` | VbaiProgressionNet → ONNX (mode: mri / tab / multi) |
| `ONNXModel` | PyTorch-free ONNX inference wrapper |

---

## Legacy 2D Classification

```python
import vbai

# Dementia + Tumor (2D images)
model = vbai.MultiTaskBrainModel(variant='q')
trainer = vbai.Trainer(model=model, lr=5e-4, device='cuda')
history = trainer.fit(train_data=dataset, epochs=10, batch_size=32)

# Predict
result = model.predict('scan.jpg')
print(result.dementia_class, result.tumor_class)

# ONNX
vbai.export_onnx(model, 'model_2d.onnx')
```

---

## Project Structure

```
vbai/
  models/
    segmentation3d.py    VbaiSegNet3D
    progression3d.py     VbaiProgressionNet, MRIEncoder3D, TabularEncoder, CrossModalFusion
    multitask.py         MultiTaskBrainModel (2D)
    multitask3d.py       MultiTask3DBrainModel
  training/
    segmentation_trainer.py  SegmentationTrainer (fit diagnosis)
    progression_trainer.py   ProgressionTrainer 3-phase (fit diagnosis)
    segmentation_losses.py   Dice, Focal, TumorSeg, TissueSeg, DeepSupervision
    progression_losses.py    FocalLoss3Class, InfoNCE, VbaiProgressionLoss
  data/
    segmentation_dataset.py  TumorSeg / TissueSeg datasets + dataloaders
    progression_dataset.py   ProgressionDataset, TabularNormalizer, BIOMARKER_FEATURES
    dataset.py               UnifiedMRIDataset (2D)
    nifti_dataset.py         NIfTIDataset (3D classification)
  utils/
    segmentation_viz.py      Slice plots, metrics, training curves
    progression_viz.py       Clinical report, risk gauge, timeline, radar
    visualization.py         Attention heatmaps (2D)
  configs/
    segmentation_config.py   SegmentationModelConfig, get_segmentation_config()
    progression_config.py    ProgressionModelConfig, get_progression_config()
    config.py                ModelConfig, TrainingConfig (2D)
    config3d.py              Model3DConfig, Training3DConfig
  export/
    onnx_export.py           export_onnx, export_segmentation_onnx, export_progression_onnx
    onnx_inference.py        ONNXModel
  hub/
    hub.py                   push_to_hub, from_hub, list_models
tests/
  test_models.py             18 tests (2D / 3D classification)
  test_3d_modules.py         49 tests (segmentation + progression)
```

---

## Citation

```bibtex
@software{vbai,
  title  = {Vbai: Visual Brain AI Library},
  author = {Neurazum},
  year   = {2026},
  url    = {https://github.com/Neurazum-AI-Department/vbai}
}
```

## License

MIT License — see [LICENSE](LICENSE) for details.

### Support

- **Website**: [Neurazum](https://neurazum.com)
- **Email**: [contact@neurazum.com](mailto:contact@neurazum.com)

---

<span style="color: #ff8d26;"><b>Neurazum</b> AI Department</span>
