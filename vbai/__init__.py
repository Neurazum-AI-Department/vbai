"""
Vbai — Visual Brain AI Library
================================
PyTorch-based library for 3D brain MRI analysis.

Capabilities:
  1. 3D Tumour & Tissue Segmentation  — VbaiSegNet3D
  2. Multimodal Progression Prediction — VbaiProgressionNet (MRI + biomarkers)
  3. 3D Alzheimer's Classification     — MultiTask3DBrainModel
  4. 2D Multi-task Classification      — MultiTaskBrainModel (legacy)

Quick start — segmentation::

    import vbai
    model = vbai.VbaiSegNet3D(in_channels=2, out_channels=1)
    trainer = vbai.SegmentationTrainer(model, device='cuda')
    history = trainer.fit(train_loader, val_loader, epochs=50)

Quick start — progression prediction::

    import vbai
    model = vbai.VbaiProgressionNet()
    trainer = vbai.ProgressionTrainer(model, device='cuda')
    trainer.fit_phase3(full_loader, full_val_loader)

    result = model.predict(mri=volume, tab=biomarkers)
    fig = vbai.plot_progression_report(result, subject_id='ADNI-001')
    fig.savefig('report.pdf')
"""

__version__ = "1.2.2"
__author__ = "Neurazum"

# ── 3D Segmentation ────────────────────────────────────────────────────────────
from .models import (
    VbaiSegNet3D,
    ASPP3D,
    AttentionGate3D,
)

# ── 3D Progression (MRI + Biomarkers) ─────────────────────────────────────────
from .models import (
    VbaiProgressionNet,
    MRIEncoder3D,
    TabularEncoder,
    CrossModalFusion,
    ProgressionHead,
)

# ── 3D Classification ──────────────────────────────────────────────────────────
from .models import (
    MultiTask3DBrainModel,
    SharedBackbone3D,
    AttentionModule3D,
    Prediction3DResult,
)

# ── 2D Models ─────────────────────────────────────────────────────────────────
from .models import (
    MultiTaskBrainModel,
    AttentionModule,
    SharedBackbone,
)

# ── Segmentation Data ──────────────────────────────────────────────────────────
from .data import (
    TumorSegmentationDataset,
    TissueSegmentationDataset,
    create_segmentation_dataloaders,
)

# ── Progression Data ───────────────────────────────────────────────────────────
from .data import (
    TabularNormalizer,
    ProgressionDataset,
    create_progression_dataloaders,
    BIOMARKER_FEATURES,
    N_FEATURES,
    CLASS_NAMES,
)

# ── 3D NIfTI Data (classification) ────────────────────────────────────────────
from .data import (
    NIfTIDataset,
    UnifiedNIfTIDataset,
    create_3d_dataloaders,
    get_3d_train_transforms,
    get_3d_val_transforms,
)

# ── 2D Data ───────────────────────────────────────────────────────────────────
from .data import (
    MRIDataset,
    UnifiedMRIDataset,
    get_transforms,
    get_train_transforms,
    get_val_transforms,
)

# ── Advanced Augmentation ─────────────────────────────────────────────────────
from .data import (
    simulate_bias_field,
    simulate_ghosting,
    simulate_spike_noise,
    simulate_rician_noise,
    simulate_mri_artifacts,
    elastic_deformation_2d,
    elastic_deformation_3d,
    mixup,
    cutmix,
    MRIAutoAugment,
)

# ── Segmentation Training ──────────────────────────────────────────────────────
from .training import (
    DiceLoss,
    MulticlassDiceLoss,
    TumorSegmentationLoss,
    TissueSegmentationLoss,
    DeepSupervisionLoss,
    SegmentationTrainer,
    SegmentationHistory,
)

# ── Progression Training ───────────────────────────────────────────────────────
from .training import (
    FocalLoss3Class,
    ProgressionLoss,
    InfoNCELoss,
    VbaiProgressionLoss,
    ProgressionTrainer,
    ProgressionPhaseHistory,
)

# ── 2D / 3D Classification Training ───────────────────────────────────────────
from .training import (
    Trainer,
    Trainer3D,
    MultiTaskLoss,
    EarlyStopping,
    ModelCheckpoint,
    Training3DHistory,
)

# ── Visualization ──────────────────────────────────────────────────────────────
from .utils import (
    plot_segmentation_slices,
    plot_dice_per_class,
    compute_segmentation_metrics,
    plot_segmentation_training_curves,
    plot_progression_report,
    plot_risk_gauge,
    plot_time_distribution,
    plot_class_probabilities,
    plot_biomarker_radar,
    create_report_figure,
    VisualizationManager,
    BrainStructureAnalyzer,
    visualize_prediction,
    create_attention_heatmap,
)

# ── Configs ───────────────────────────────────────────────────────────────────
from .configs import (
    SegmentationModelConfig,
    SegmentationTrainingConfig,
    FullSegmentationConfig,
    get_segmentation_config,
    ProgressionModelConfig,
    ProgressionTrainingConfig,
    FullProgressionConfig,
    get_progression_config,
    ModelConfig,
    TrainingConfig,
    get_default_config,
    Model3DConfig,
    Training3DConfig,
    Full3DConfig,
    get_default_3d_config,
)

# ── Hub ───────────────────────────────────────────────────────────────────────
from .hub import (
    ModelInfo,
    MODEL_REGISTRY,
    list_models,
    get_model_info,
    register_model,
    download_from_hub,
    push_to_hub,
    from_hub,
    generate_model_card,
)

# ── Export (ONNX) ─────────────────────────────────────────────────────────────
from .export import (
    export_onnx,
    export_segmentation_onnx,
    export_progression_onnx,
    ONNXModel,
)


# ── Loading Functions ──────────────────────────────────────────────────────────

def load(path: str, device: str = 'cpu'):
    """Load a trained Vbai 2D model from checkpoint."""
    import torch
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    config = checkpoint.get('config', {})
    model = MultiTaskBrainModel(
        variant=config.get('variant', 'q'),
        tasks=config.get('tasks', ['dementia', 'tumor']),
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    return model.to(device).eval()


def load_3d(path: str, device: str = 'cpu'):
    """Load a trained Vbai 3D classification model from checkpoint."""
    import torch
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    config = checkpoint.get('config', {})
    input_shape = config.get('input_shape', (96, 96, 96))
    if isinstance(input_shape, list):
        input_shape = tuple(input_shape)
    model = MultiTask3DBrainModel(
        variant=config.get('variant', 'q'),
        tasks=config.get('tasks', {'alzheimer': 3}),
        in_channels=config.get('in_channels', 1),
        input_shape=input_shape,
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    return model.to(device).eval()


def load_segmentation(path: str, device: str = 'cpu'):
    """Load a trained VbaiSegNet3D from checkpoint."""
    return VbaiSegNet3D.load(path, device=device)


def load_progression(path: str, device: str = 'cpu'):
    """Load a trained VbaiProgressionNet from checkpoint."""
    return VbaiProgressionNet.load(path, device=device)


def load_pretrained(model_name: str = 'vbai-3d-q', device: str = 'cpu'):
    """Load a pretrained model from the Vbai model zoo via HuggingFace Hub."""
    info = get_model_info(model_name)
    if info.hub_id is None:
        raise ValueError(f"Model '{model_name}' has no Hub ID. Use load_*() with a local checkpoint.")
    try:
        return from_hub(info.hub_id, device=device)
    except Exception as e:
        available = ', '.join(MODEL_REGISTRY.keys())
        raise RuntimeError(
            f"Failed to download '{model_name}' from '{info.hub_id}': {e}\n"
            f"Available models: {available}"
        ) from e


# ── Constants ─────────────────────────────────────────────────────────────────

DEMENTIA_CLASSES = [
    'AD_Alzheimer', 'AD_Mild_Demented', 'AD_Moderate_Demented',
    'AD_Very_Mild_Demented', 'CN_Non_Demented', 'PD_Parkinson',
]
TUMOR_CLASSES = ['Glioma', 'Meningioma', 'No_Tumor', 'Pituitary']
ALZHEIMER_3D_CLASSES = ['CN', 'MCI', 'AD']

__all__ = [
    '__version__',
    # 3D Segmentation
    'VbaiSegNet3D', 'ASPP3D', 'AttentionGate3D',
    # 3D Progression
    'VbaiProgressionNet', 'MRIEncoder3D', 'TabularEncoder',
    'CrossModalFusion', 'ProgressionHead',
    # 3D Classification
    'MultiTask3DBrainModel', 'SharedBackbone3D', 'AttentionModule3D', 'Prediction3DResult',
    # 2D
    'MultiTaskBrainModel', 'AttentionModule', 'SharedBackbone',
    # Segmentation Data
    'TumorSegmentationDataset', 'TissueSegmentationDataset', 'create_segmentation_dataloaders',
    # Progression Data
    'TabularNormalizer', 'ProgressionDataset', 'create_progression_dataloaders',
    'BIOMARKER_FEATURES', 'N_FEATURES', 'CLASS_NAMES',
    # 3D Data
    'NIfTIDataset', 'UnifiedNIfTIDataset', 'create_3d_dataloaders',
    'get_3d_train_transforms', 'get_3d_val_transforms',
    # 2D Data
    'MRIDataset', 'UnifiedMRIDataset',
    'get_transforms', 'get_train_transforms', 'get_val_transforms',
    # Augmentation
    'simulate_bias_field', 'simulate_ghosting', 'simulate_spike_noise',
    'simulate_rician_noise', 'simulate_mri_artifacts',
    'elastic_deformation_2d', 'elastic_deformation_3d',
    'mixup', 'cutmix', 'MRIAutoAugment',
    # Segmentation Training
    'DiceLoss', 'MulticlassDiceLoss', 'TumorSegmentationLoss',
    'TissueSegmentationLoss', 'DeepSupervisionLoss',
    'SegmentationTrainer', 'SegmentationHistory',
    # Progression Training
    'FocalLoss3Class', 'ProgressionLoss', 'InfoNCELoss', 'VbaiProgressionLoss',
    'ProgressionTrainer', 'ProgressionPhaseHistory',
    # Classification Training
    'Trainer', 'Trainer3D', 'MultiTaskLoss', 'EarlyStopping', 'ModelCheckpoint',
    'Training3DHistory',
    # Visualization
    'plot_segmentation_slices', 'plot_dice_per_class',
    'compute_segmentation_metrics', 'plot_segmentation_training_curves',
    'plot_progression_report', 'plot_risk_gauge', 'plot_time_distribution',
    'plot_class_probabilities', 'plot_biomarker_radar', 'create_report_figure',
    'VisualizationManager', 'BrainStructureAnalyzer',
    'visualize_prediction', 'create_attention_heatmap',
    # Configs
    'SegmentationModelConfig', 'SegmentationTrainingConfig',
    'FullSegmentationConfig', 'get_segmentation_config',
    'ProgressionModelConfig', 'ProgressionTrainingConfig',
    'FullProgressionConfig', 'get_progression_config',
    'ModelConfig', 'TrainingConfig', 'get_default_config',
    'Model3DConfig', 'Training3DConfig', 'Full3DConfig', 'get_default_3d_config',
    # Hub
    'ModelInfo', 'MODEL_REGISTRY', 'list_models', 'get_model_info', 'register_model',
    'download_from_hub', 'push_to_hub', 'from_hub', 'generate_model_card',
    # Export
    'export_onnx', 'export_segmentation_onnx', 'export_progression_onnx', 'ONNXModel',
    # Loading
    'load', 'load_3d', 'load_segmentation', 'load_progression', 'load_pretrained',
    # Constants
    'DEMENTIA_CLASSES', 'TUMOR_CLASSES', 'ALZHEIMER_3D_CLASSES',
]
