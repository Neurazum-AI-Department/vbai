"""
Attention Modules for Vbai Models
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class AttentionModule(nn.Module):
    """
    Spatial attention module for task-specific feature focusing.
    
    Uses channel-wise attention followed by spatial attention to
    highlight relevant regions for each task (dementia/tumor).
    
    Args:
        in_channels: Number of input feature channels
        reduction_ratio: Channel reduction ratio for attention (default: 16)
    """
    
    def __init__(self, in_channels: int, reduction_ratio: int = 16):
        super().__init__()
        
        self.in_channels = in_channels
        reduced_channels = max(in_channels // reduction_ratio, 8)
        
        # Channel attention (squeeze-excitation style)
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(in_channels, reduced_channels),
            nn.ReLU(inplace=True),
            nn.Linear(reduced_channels, in_channels),
            nn.Sigmoid()
        )
        
        # Spatial attention
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(in_channels, 1, kernel_size=7, padding=3),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply attention to input features.
        
        Args:
            x: Input tensor of shape (B, C, H, W)
        
        Returns:
            Tuple of (attended_features, attention_map)
            - attended_features: Shape (B, C, H, W)
            - attention_map: Shape (B, 1, H, W) for visualization
        """
        # Channel attention
        channel_weights = self.channel_attention(x)
        channel_weights = channel_weights.view(-1, self.in_channels, 1, 1)
        x = x * channel_weights
        
        # Spatial attention
        attention_map = self.spatial_attention(x)
        attended = x * attention_map
        
        return attended, attention_map
    
    def get_attention_map(self, x: torch.Tensor) -> torch.Tensor:
        """Get only the attention map without attended features."""
        _, attention_map = self.forward(x)
        return attention_map


class DualAttention(nn.Module):
    """
    Dual attention module for multi-task learning.
    
    Contains separate attention modules for dementia and tumor tasks,
    allowing each task to focus on different image regions.
    
    Args:
        in_channels: Number of input feature channels
        reduction_ratio: Channel reduction ratio (default: 16)
    """
    
    def __init__(self, in_channels: int, reduction_ratio: int = 16):
        super().__init__()
        
        self.dementia_attention = AttentionModule(in_channels, reduction_ratio)
        self.tumor_attention = AttentionModule(in_channels, reduction_ratio)
    
    def forward(
        self, 
        x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Apply dual attention for both tasks.
        
        Args:
            x: Shared features from backbone
        
        Returns:
            Tuple of (dementia_features, tumor_features, dementia_attn, tumor_attn)
        """
        dementia_features, dementia_attn = self.dementia_attention(x)
        tumor_features, tumor_attn = self.tumor_attention(x)
        
        return dementia_features, tumor_features, dementia_attn, tumor_attn


class CBAM(nn.Module):
    """
    Convolutional Block Attention Module (CBAM).
    
    Alternative attention mechanism combining channel and spatial attention
    in a sequential manner.
    
    Reference: https://arxiv.org/abs/1807.06521
    """
    
    def __init__(self, in_channels: int, reduction_ratio: int = 16):
        super().__init__()
        
        reduced = max(in_channels // reduction_ratio, 8)
        
        # Channel attention
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.channel_mlp = nn.Sequential(
            nn.Linear(in_channels, reduced),
            nn.ReLU(inplace=True),
            nn.Linear(reduced, in_channels)
        )
        
        # Spatial attention  
        self.spatial_conv = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply CBAM attention."""
        b, c, h, w = x.shape
        
        # Channel attention
        avg_out = self.channel_mlp(self.avg_pool(x).view(b, c))
        max_out = self.channel_mlp(self.max_pool(x).view(b, c))
        channel_attn = torch.sigmoid(avg_out + max_out).view(b, c, 1, 1)
        x = x * channel_attn
        
        # Spatial attention
        avg_spatial = x.mean(dim=1, keepdim=True)
        max_spatial = x.max(dim=1, keepdim=True)[0]
        spatial_input = torch.cat([avg_spatial, max_spatial], dim=1)
        spatial_attn = self.spatial_conv(spatial_input)
        
        attended = x * spatial_attn
        
        return attended, spatial_attn


class SelfAttention(nn.Module):
    """
    Self-attention module for capturing long-range dependencies.
    
    Useful for relating distant brain regions in MRI analysis.
    """
    
    def __init__(self, in_channels: int, num_heads: int = 8):
        super().__init__()
        
        self.num_heads = num_heads
        self.head_dim = in_channels // num_heads
        
        self.query = nn.Conv2d(in_channels, in_channels, 1)
        self.key = nn.Conv2d(in_channels, in_channels, 1)
        self.value = nn.Conv2d(in_channels, in_channels, 1)
        self.out = nn.Conv2d(in_channels, in_channels, 1)
        
        self.scale = self.head_dim ** -0.5
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply self-attention."""
        b, c, h, w = x.shape
        
        # Compute Q, K, V
        q = self.query(x).view(b, self.num_heads, self.head_dim, h * w)
        k = self.key(x).view(b, self.num_heads, self.head_dim, h * w)
        v = self.value(x).view(b, self.num_heads, self.head_dim, h * w)
        
        # Attention scores
        attn = torch.matmul(q.transpose(-2, -1), k) * self.scale
        attn = F.softmax(attn, dim=-1)
        
        # Apply attention
        out = torch.matmul(v, attn.transpose(-2, -1))
        out = out.view(b, c, h, w)
        out = self.out(out)
        
        # Create attention map for visualization (average across heads)
        attn_map = attn.mean(dim=1).view(b, h, w, h, w)
        attn_map = attn_map.mean(dim=(3, 4)).unsqueeze(1)  # (B, 1, H, W)
        
        return x + out, attn_map
