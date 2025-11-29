"""
Shared Backbone Networks for Vbai Models
"""

import torch
import torch.nn as nn
from typing import Literal


class SharedBackbone(nn.Module):
    """
    Shared CNN backbone for multi-task brain MRI analysis.
    
    Args:
        variant: Model variant - 'f' (fast/lightweight) or 'q' (quality/deep)
        in_channels: Number of input channels (default: 3 for RGB)
    
    Variants:
        - 'f' (fast): 3 conv layers, 128 final channels - faster training
        - 'q' (quality): 4 conv layers, 512 final channels - better accuracy
    """
    
    # Variant configurations
    VARIANTS = {
        'f': {  # Fast variant
            'channels': [32, 64, 128],
            'description': 'Lightweight model for fast training'
        },
        'q': {  # Quality variant
            'channels': [64, 128, 256, 512],
            'description': 'Deep model for high accuracy'
        }
    }
    
    def __init__(
        self, 
        variant: Literal['f', 'q'] = 'q',
        in_channels: int = 3
    ):
        super().__init__()
        
        if variant not in self.VARIANTS:
            raise ValueError(f"Unknown variant '{variant}'. Choose from: {list(self.VARIANTS.keys())}")
        
        self.variant = variant
        self.config = self.VARIANTS[variant]
        channels = self.config['channels']
        
        # Build convolutional layers
        layers = []
        prev_channels = in_channels
        
        for i, out_channels in enumerate(channels):
            layers.extend([
                nn.Conv2d(prev_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2, 2)
            ])
            prev_channels = out_channels
        
        self.features = nn.Sequential(*layers)
        self.out_channels = channels[-1]
        
        # Calculate output spatial size (224 / 2^num_pools)
        self.num_pools = len(channels)
        self.output_size = 224 // (2 ** self.num_pools)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through backbone.
        
        Args:
            x: Input tensor of shape (B, C, H, W)
        
        Returns:
            Feature tensor of shape (B, out_channels, H', W')
        """
        return self.features(x)
    
    def get_output_shape(self, input_size: int = 224) -> tuple:
        """Get output tensor shape for given input size."""
        spatial = input_size // (2 ** self.num_pools)
        return (self.out_channels, spatial, spatial)
    
    def __repr__(self):
        return (
            f"SharedBackbone(variant='{self.variant}', "
            f"out_channels={self.out_channels}, "
            f"num_layers={self.num_pools})"
        )


class EdgeDetectionBranch(nn.Module):
    """
    Edge detection branch using learned convolutions.
    Mimics Canny edge detection behavior with learnable parameters.
    """
    
    def __init__(self, in_channels: int = 3, out_channels: int = 32):
        super().__init__()
        
        # Sobel-like edge detection (learnable)
        self.edge_conv = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        
        self.out_channels = out_channels
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Extract edge features from input image."""
        return self.edge_conv(x)


class FeatureFusion(nn.Module):
    """
    Fuses features from main backbone and edge detection branch.
    """
    
    def __init__(self, main_channels: int, edge_channels: int, out_channels: int):
        super().__init__()
        
        self.fusion = nn.Sequential(
            nn.Conv2d(main_channels + edge_channels, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, main_features: torch.Tensor, edge_features: torch.Tensor) -> torch.Tensor:
        """
        Fuse main and edge features.
        
        Args:
            main_features: Features from main backbone
            edge_features: Features from edge branch (will be downsampled)
        
        Returns:
            Fused feature tensor
        """
        # Downsample edge features to match main features
        if edge_features.shape[-1] != main_features.shape[-1]:
            edge_features = nn.functional.adaptive_avg_pool2d(
                edge_features, main_features.shape[-2:]
            )
        
        # Concatenate and fuse
        combined = torch.cat([main_features, edge_features], dim=1)
        return self.fusion(combined)
