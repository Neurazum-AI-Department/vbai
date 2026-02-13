"""
3D Backbone Networks for Vbai Models

ResNet-style 3D CNN backbone for volumetric brain MRI analysis (.nii/.nii.gz).
Adapted from Vbai-3D architecture with SE attention blocks.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Literal


class ResBlock3D(nn.Module):
    """
    3D Residual Block with optional downsampling.

    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        stride: Convolution stride (2 for downsampling)
        dropout: Dropout rate for Dropout3d
    """

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1, dropout: float = 0.2):
        super().__init__()

        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(out_channels)

        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm3d(out_channels)
            )

        self.dropout = nn.Dropout3d(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.dropout(out)
        out += self.shortcut(residual)
        out = F.relu(out)
        return out


class SEBlock3D(nn.Module):
    """
    3D Squeeze-and-Excitation Block for channel attention.

    Args:
        channels: Number of input/output channels
        reduction: Channel reduction ratio
    """

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        reduced = max(channels // reduction, 8)
        self.fc = nn.Sequential(
            nn.Linear(channels, reduced, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(reduced, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1, 1)
        return x * y.expand_as(x)


class SharedBackbone3D(nn.Module):
    """
    3D ResNet-style backbone with SE attention for volumetric MRI analysis.

    Input: (B, 1, D, H, W) - single-channel NIfTI volumes
    Output: (B, out_channels, D', H', W') - 3D feature maps

    Args:
        variant: Model variant
            - 'f' (fast): 3 ResNet stages [32, 64, 128], faster training
            - 'q' (quality): 3 ResNet stages [64, 128, 256], higher accuracy
        in_channels: Number of input channels (default: 1 for NIfTI)
        dropout: Dropout rate for residual blocks

    Variants:
        - 'f': Lightweight, ~2M params, suitable for limited GPU memory
        - 'q': Deeper with more channels, ~8M params, better accuracy
    """

    VARIANTS = {
        'f': {
            'channels': [32, 64, 128],
            'num_blocks': [1, 1, 1],
            'description': 'Lightweight 3D model for fast training'
        },
        'q': {
            'channels': [64, 128, 256],
            'num_blocks': [2, 2, 2],
            'description': 'Deep 3D model for high accuracy'
        }
    }

    def __init__(
        self,
        variant: Literal['f', 'q'] = 'q',
        in_channels: int = 1,
        dropout: float = 0.2,
    ):
        super().__init__()

        if variant not in self.VARIANTS:
            raise ValueError(f"Unknown variant '{variant}'. Choose from: {list(self.VARIANTS.keys())}")

        self.variant = variant
        self.config = self.VARIANTS[variant]
        channels = self.config['channels']
        num_blocks = self.config['num_blocks']

        # Initial convolution stem
        self.stem = nn.Sequential(
            nn.Conv3d(in_channels, channels[0], kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm3d(channels[0]),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        )

        # ResNet stages with SE attention
        self.stages = nn.ModuleList()
        self.se_blocks = nn.ModuleList()

        prev_channels = channels[0]
        for i, (out_ch, n_blocks) in enumerate(zip(channels, num_blocks)):
            stride = 1 if i == 0 else 2
            stage = self._make_stage(prev_channels, out_ch, n_blocks, stride, dropout)
            self.stages.append(stage)
            self.se_blocks.append(SEBlock3D(out_ch))
            prev_channels = out_ch

        self.out_channels = channels[-1]

    def _make_stage(self, in_channels, out_channels, num_blocks, stride, dropout):
        layers = [ResBlock3D(in_channels, out_channels, stride, dropout)]
        for _ in range(1, num_blocks):
            layers.append(ResBlock3D(out_channels, out_channels, 1, dropout))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through 3D backbone.

        Args:
            x: Input tensor of shape (B, 1, D, H, W)

        Returns:
            Feature tensor of shape (B, out_channels, D', H', W')
        """
        x = self.stem(x)
        for stage, se in zip(self.stages, self.se_blocks):
            x = stage(x)
            x = se(x)
        return x

    def __repr__(self):
        return (
            f"SharedBackbone3D(variant='{self.variant}', "
            f"out_channels={self.out_channels})"
        )
