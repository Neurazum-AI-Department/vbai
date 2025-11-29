"""
Multi-Task Brain MRI Model for Vbai
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional, Literal, Union, NamedTuple
from PIL import Image
import numpy as np

from .backbone import SharedBackbone, EdgeDetectionBranch, FeatureFusion
from .attention import AttentionModule, DualAttention


class PredictionResult(NamedTuple):
    """Result container for model predictions."""
    dementia_class: str
    dementia_probs: torch.Tensor
    dementia_confidence: float
    tumor_class: str
    tumor_probs: torch.Tensor
    tumor_confidence: float
    dementia_attention: Optional[torch.Tensor] = None
    tumor_attention: Optional[torch.Tensor] = None


class MultiTaskBrainModel(nn.Module):
    """
    Multi-task deep learning model for brain MRI analysis.
    
    Performs simultaneous classification of:
    - Dementia: 6 classes (AD Alzheimer, Mild/Moderate/Very Mild Demented, Non Demented, PD)
    - Brain Tumor: 4 classes (Glioma, Meningioma, No Tumor, Pituitary)
    
    Args:
        variant: Model variant
            - 'f' (fast): Lightweight model, faster training
            - 'q' (quality): Deeper model, higher accuracy
        pretrained: Whether to load pretrained weights (not yet implemented)
        num_dementia_classes: Number of dementia classes (default: 6)
        num_tumor_classes: Number of tumor classes (default: 4)
        use_edge_branch: Whether to use edge detection branch (default: True)
        dropout: Dropout rate for classifier heads (default: 0.5)
    
    Example:
        >>> model = MultiTaskBrainModel(variant='q')
        >>> x = torch.randn(1, 3, 224, 224)
        >>> dementia_logits, tumor_logits = model(x)
        >>> 
        >>> # Or get predictions with class names
        >>> result = model.predict('brain_scan.jpg')
        >>> print(result.dementia_class, result.tumor_class)
    """
    
    DEMENTIA_CLASSES = [
        'AD_Alzheimer', 'AD_Mild_Demented', 'AD_Moderate_Demented',
        'AD_Very_Mild_Demented', 'CN_Non_Demented', 'PD_Parkinson'
    ]
    
    TUMOR_CLASSES = ['Glioma', 'Meningioma', 'No_Tumor', 'Pituitary']
    
    def __init__(
        self,
        variant: Literal['f', 'q'] = 'q',
        pretrained: bool = False,
        num_dementia_classes: int = 6,
        num_tumor_classes: int = 4,
        use_edge_branch: bool = True,
        dropout: float = 0.5
    ):
        super().__init__()
        
        self.variant = variant
        self.num_dementia_classes = num_dementia_classes
        self.num_tumor_classes = num_tumor_classes
        self.use_edge_branch = use_edge_branch
        
        # Shared backbone
        self.backbone = SharedBackbone(variant=variant)
        backbone_channels = self.backbone.out_channels
        
        # Optional edge detection branch
        if use_edge_branch:
            self.edge_branch = EdgeDetectionBranch(in_channels=3, out_channels=32)
            self.fusion = FeatureFusion(
                main_channels=backbone_channels,
                edge_channels=32,
                out_channels=backbone_channels
            )
        
        # Dual attention for task-specific focus
        self.attention = DualAttention(backbone_channels)
        
        # Task-specific classifier heads
        feature_size = backbone_channels * (self.backbone.output_size ** 2)
        
        self.dementia_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(feature_size, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, num_dementia_classes)
        )
        
        self.tumor_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(feature_size, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, num_tumor_classes)
        )
        
        # Store attention maps for visualization
        self._dementia_attention = None
        self._tumor_attention = None
        
        # Load pretrained weights if requested
        if pretrained:
            self._load_pretrained()
        
        # ImageNet normalization (used for preprocessing)
        self.register_buffer(
            'mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        )
        self.register_buffer(
            'std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        )
    
    def forward(
        self, 
        x: torch.Tensor,
        return_attention: bool = False
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], 
               Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
        """
        Forward pass through the model.
        
        Args:
            x: Input tensor of shape (B, 3, 224, 224)
            return_attention: Whether to return attention maps
        
        Returns:
            If return_attention=False:
                (dementia_logits, tumor_logits)
            If return_attention=True:
                (dementia_logits, tumor_logits, dementia_attn, tumor_attn)
        """
        # Extract backbone features
        features = self.backbone(x)
        
        # Optional edge feature fusion
        if self.use_edge_branch:
            edge_features = self.edge_branch(x)
            features = self.fusion(features, edge_features)
        
        # Apply dual attention
        dementia_feat, tumor_feat, dementia_attn, tumor_attn = self.attention(features)
        
        # Store attention maps
        self._dementia_attention = dementia_attn
        self._tumor_attention = tumor_attn
        
        # Classification heads
        dementia_logits = self.dementia_head(dementia_feat)
        tumor_logits = self.tumor_head(tumor_feat)
        
        if return_attention:
            return dementia_logits, tumor_logits, dementia_attn, tumor_attn
        return dementia_logits, tumor_logits
    
    def predict(
        self, 
        image: Union[str, Image.Image, torch.Tensor, np.ndarray],
        return_attention: bool = False
    ) -> PredictionResult:
        """
        Make prediction on a single image.
        
        Args:
            image: Input image (file path, PIL Image, tensor, or numpy array)
            return_attention: Whether to include attention maps in result
        
        Returns:
            PredictionResult with class names, probabilities, and confidence scores
        """
        # Preprocess image
        x = self._preprocess(image)
        x = x.to(next(self.parameters()).device)
        
        # Forward pass
        self.eval()
        with torch.no_grad():
            dementia_logits, tumor_logits = self(x)
            
            dementia_probs = F.softmax(dementia_logits, dim=1)
            tumor_probs = F.softmax(tumor_logits, dim=1)
            
            dementia_idx = dementia_probs.argmax(dim=1).item()
            tumor_idx = tumor_probs.argmax(dim=1).item()
        
        result = PredictionResult(
            dementia_class=self.DEMENTIA_CLASSES[dementia_idx],
            dementia_probs=dementia_probs[0],
            dementia_confidence=dementia_probs[0, dementia_idx].item(),
            tumor_class=self.TUMOR_CLASSES[tumor_idx],
            tumor_probs=tumor_probs[0],
            tumor_confidence=tumor_probs[0, tumor_idx].item(),
            dementia_attention=self._dementia_attention if return_attention else None,
            tumor_attention=self._tumor_attention if return_attention else None
        )
        
        return result
    
    def _preprocess(
        self, 
        image: Union[str, Image.Image, torch.Tensor, np.ndarray]
    ) -> torch.Tensor:
        """Preprocess image for model input."""
        from torchvision import transforms
        
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        if isinstance(image, str):
            image = Image.open(image).convert('RGB')
        elif isinstance(image, np.ndarray):
            image = Image.fromarray(image).convert('RGB')
        elif isinstance(image, torch.Tensor):
            if image.dim() == 4:
                return image
            elif image.dim() == 3:
                return image.unsqueeze(0)
        
        x = transform(image)
        return x.unsqueeze(0)
    
    def _load_pretrained(self):
        """Load pretrained weights."""
        raise NotImplementedError(
            "Pretrained weights not yet available. "
            "Train your own model or use vbai.load() with a checkpoint."
        )
    
    def get_attention_maps(self) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Get the last computed attention maps."""
        return self._dementia_attention, self._tumor_attention
    
    def count_parameters(self) -> Dict[str, int]:
        """Count model parameters."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'total': total,
            'trainable': trainable,
            'backbone': sum(p.numel() for p in self.backbone.parameters()),
            'attention': sum(p.numel() for p in self.attention.parameters()),
            'dementia_head': sum(p.numel() for p in self.dementia_head.parameters()),
            'tumor_head': sum(p.numel() for p in self.tumor_head.parameters()),
        }
    
    def __repr__(self):
        params = self.count_parameters()
        return (
            f"MultiTaskBrainModel(\n"
            f"  variant='{self.variant}',\n"
            f"  dementia_classes={self.num_dementia_classes},\n"
            f"  tumor_classes={self.num_tumor_classes},\n"
            f"  total_params={params['total']:,}\n"
            f")"
        )


def create_model(
    variant: Literal['f', 'q'] = 'q',
    **kwargs
) -> MultiTaskBrainModel:
    """
    Factory function to create a MultiTaskBrainModel.
    
    Args:
        variant: Model variant ('f' for fast, 'q' for quality)
        **kwargs: Additional arguments passed to MultiTaskBrainModel
    
    Returns:
        Configured MultiTaskBrainModel instance
    """
    return MultiTaskBrainModel(variant=variant, **kwargs)
