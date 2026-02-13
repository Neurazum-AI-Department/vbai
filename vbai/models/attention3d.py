"""
3D Attention Modules for Vbai Models

Channel and spatial attention adapted for 3D volumetric data.
"""

import torch
import torch.nn as nn
from typing import Tuple


class AttentionModule3D(nn.Module):
    """
    3D spatial attention module for task-specific feature focusing.

    Uses channel-wise squeeze-excitation followed by 3D spatial attention
    to highlight relevant volumetric regions.

    Args:
        in_channels: Number of input feature channels
        reduction_ratio: Channel reduction ratio for SE attention (default: 16)
    """

    def __init__(self, in_channels: int, reduction_ratio: int = 16):
        super().__init__()

        self.in_channels = in_channels
        reduced = max(in_channels // reduction_ratio, 8)

        # Channel attention (3D squeeze-excitation)
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Flatten(),
            nn.Linear(in_channels, reduced),
            nn.ReLU(inplace=True),
            nn.Linear(reduced, in_channels),
            nn.Sigmoid()
        )

        # Spatial attention (3D)
        self.spatial_attention = nn.Sequential(
            nn.Conv3d(in_channels, 1, kernel_size=7, padding=3),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply 3D attention to input features.

        Args:
            x: Input tensor of shape (B, C, D, H, W)

        Returns:
            Tuple of (attended_features, attention_map)
            - attended_features: Shape (B, C, D, H, W)
            - attention_map: Shape (B, 1, D, H, W) for visualization
        """
        # Channel attention
        channel_weights = self.channel_attention(x)
        channel_weights = channel_weights.view(-1, self.in_channels, 1, 1, 1)
        x = x * channel_weights

        # Spatial attention
        attention_map = self.spatial_attention(x)
        attended = x * attention_map

        return attended, attention_map
