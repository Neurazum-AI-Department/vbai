"""Vbai Models Module"""

from .backbone import SharedBackbone
from .attention import AttentionModule
from .multitask import MultiTaskBrainModel

# 3D Classification Models
from .backbone3d import SharedBackbone3D, ResBlock3D, SEBlock3D
from .attention3d import AttentionModule3D
from .multitask3d import MultiTask3DBrainModel, Prediction3DResult

# 3D Segmentation Model
from .segmentation3d import (
    VbaiSegNet3D,
    SEBlock3D as SegSEBlock3D,
    CBAM3D as SegCBAM3D,
    ResBlock3D as SegResBlock3D,
    ASPP3D,
    AttentionGate3D,
)

# 3D Progression Model (MRI + Biomarkers)
from .progression3d import (
    VbaiProgressionNet,
    MRIEncoder3D,
    TabularEncoder,
    CrossModalFusion,
    ProgressionHead,
)

__all__ = [
    # 2D
    'SharedBackbone', 'AttentionModule', 'MultiTaskBrainModel',
    # 3D Classification
    'SharedBackbone3D', 'ResBlock3D', 'SEBlock3D',
    'AttentionModule3D', 'MultiTask3DBrainModel', 'Prediction3DResult',
    # 3D Segmentation
    'VbaiSegNet3D', 'ASPP3D', 'AttentionGate3D',
    # 3D Progression
    'VbaiProgressionNet', 'MRIEncoder3D', 'TabularEncoder',
    'CrossModalFusion', 'ProgressionHead',
]
