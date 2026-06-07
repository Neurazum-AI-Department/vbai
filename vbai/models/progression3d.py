"""
VbaiProgressionNet — Multimodal 3D Brain MRI + Biomarker Progression Model

Architecture:
  - MRIEncoder3D: 3D ResNet + CBAM + SE + ASPP + stochastic depth → 512-d
  - TabularEncoder: MLP with missing-mask handling → 256-d
  - CrossModalFusion: Bidirectional cross-attention + gated fusion → 512-d
  - ClassificationHead: CN / MCI / AD
  - ProgressionHead: will_progress (binary) · time_to_conversion · time_distribution

Key design decisions:
  - Explicit missing-mask (0/1) per biomarker prevents "missing ≠ low value" confusion
  - Modality dropout during training → robust to missing MRI or biomarkers at inference
  - InfoNCE contrastive head aligns MRI and tabular embedding spaces
  - Subject-level train/val/test split enforced in dataset (no data leakage)
  - Stochastic depth (DropPath) regularises the MRI encoder
"""

from __future__ import annotations
import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ──────────────────────────────────────────────────────────────────────────────
# Shared utilities
# ──────────────────────────────────────────────────────────────────────────────

class DropPath(nn.Module):
    """Stochastic depth (drop entire residual branch with probability *p*)."""

    def __init__(self, drop_prob: float = 0.0) -> None:
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.drop_prob == 0.0:
            return x
        keep = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = torch.empty(shape, device=x.device).bernoulli_(keep).div_(keep)
        return x * mask


class SEBlock3D(nn.Module):
    def __init__(self, ch: int, reduction: int = 16) -> None:
        super().__init__()
        mid = max(ch // reduction, 4)
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Sequential(
            nn.Linear(ch, mid, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid, ch, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c = x.shape[:2]
        w = self.fc(self.pool(x).view(b, c)).view(b, c, 1, 1, 1)
        return x * w


class CBAM3D(nn.Module):
    def __init__(self, ch: int, reduction: int = 16) -> None:
        super().__init__()
        mid = max(ch // reduction, 4)
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.max_pool = nn.AdaptiveMaxPool3d(1)
        self.ch_fc = nn.Sequential(
            nn.Linear(ch, mid, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid, ch, bias=False),
        )
        self.sp_conv = nn.Conv3d(2, 1, kernel_size=7, padding=3, bias=False)
        self.sp_bn = nn.BatchNorm3d(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c = x.shape[:2]
        avg = self.ch_fc(self.avg_pool(x).view(b, c))
        mx = self.ch_fc(self.max_pool(x).view(b, c))
        ch_w = torch.sigmoid(avg + mx).view(b, c, 1, 1, 1)
        x = x * ch_w
        sp_in = torch.cat([x.mean(1, keepdim=True), x.amax(1, keepdim=True)], 1)
        sp_w = torch.sigmoid(self.sp_bn(self.sp_conv(sp_in)))
        return x * sp_w


class ResBlock3D(nn.Module):
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        stride: int = 1,
        drop_path_rate: float = 0.0,
    ) -> None:
        super().__init__()
        self.conv1 = nn.Conv3d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(out_ch)
        self.conv2 = nn.Conv3d(out_ch, out_ch, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(out_ch)
        self.cbam = CBAM3D(out_ch)
        self.se = SEBlock3D(out_ch)
        self.drop_path = DropPath(drop_path_rate)

        self.skip = None
        if in_ch != out_ch or stride != 1:
            self.skip = nn.Sequential(
                nn.Conv3d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm3d(out_ch),
            )
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.skip(x) if self.skip else x
        out = self.act(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.cbam(out)
        out = self.se(out)
        return self.act(identity + self.drop_path(out))


class ASPP3D(nn.Module):
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        dilations: Tuple[int, ...] = (1, 6, 12, 18),
    ) -> None:
        super().__init__()
        mid = out_ch // (len(dilations) + 1)
        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv3d(in_ch, mid, 3, padding=d, dilation=d, bias=False),
                nn.BatchNorm3d(mid),
                nn.ReLU(inplace=True),
            )
            for d in dilations
        ])
        self.gp_conv = nn.Conv3d(in_ch, mid, 1, bias=False)
        total = mid * (len(dilations) + 1)
        self.project = nn.Sequential(
            nn.Conv3d(total, out_ch, 1, bias=False),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
            nn.Dropout3d(0.1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = [b(x) for b in self.branches]
        gp = F.adaptive_avg_pool3d(x, 1)
        gp = F.relu(self.gp_conv(gp), inplace=True)
        gp = F.interpolate(gp, size=x.shape[2:], mode='trilinear', align_corners=False)
        feats.append(gp)
        return self.project(torch.cat(feats, 1))


# ──────────────────────────────────────────────────────────────────────────────
# MRI Encoder
# ──────────────────────────────────────────────────────────────────────────────

class MRIEncoder3D(nn.Module):
    """
    3D ResNet encoder: stem → 4 stages → ASPP bottleneck → 512-d embedding.

    Stochastic depth rates increase linearly across stages (0 % → 15 %).
    """

    def __init__(
        self,
        in_channels: int = 1,
        feature_dim: int = 512,
        dropout: float = 0.4,
        drop_path_rates: Tuple[float, ...] = (0.0, 0.05, 0.10, 0.15),
    ) -> None:
        super().__init__()

        self.stem = nn.Sequential(
            nn.Conv3d(in_channels, 32, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm3d(32),
            nn.GELU(),
            nn.MaxPool3d(3, stride=2, padding=1),
        )

        self.stage1 = self._make_stage(32, 32, 2, drop_path_rates[0])
        self.stage2 = self._make_stage(32, 64, 2, drop_path_rates[1], stride=2)
        self.stage3 = self._make_stage(64, 128, 2, drop_path_rates[2], stride=2)
        self.stage4 = self._make_stage(128, 256, 2, drop_path_rates[3], stride=2)

        self.aspp = ASPP3D(256, feature_dim)
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.proj = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    @staticmethod
    def _make_stage(
        in_ch: int, out_ch: int, n: int, dpr: float, stride: int = 1
    ) -> nn.Sequential:
        layers = [ResBlock3D(in_ch, out_ch, stride=stride, drop_path_rate=dpr)]
        for _ in range(n - 1):
            layers.append(ResBlock3D(out_ch, out_ch, drop_path_rate=dpr))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns (B, feature_dim)."""
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.aspp(x)
        x = self.pool(x).flatten(1)
        return self.proj(x)


# ──────────────────────────────────────────────────────────────────────────────
# Tabular Encoder
# ──────────────────────────────────────────────────────────────────────────────

class TabularEncoder(nn.Module):
    """
    MLP encoder for clinical biomarkers.

    Input:  (B, n_features * 2)  — concatenation of z-scored values and
            binary missing-masks (0 = missing, 1 = present).
    Output: (B, feature_dim)

    Missing features are zeroed *before* normalisation; the accompanying mask
    tells the model explicitly which values are unknown.
    """

    def __init__(
        self,
        n_features: int = 13,
        hidden_dims: Tuple[int, ...] = (128, 256),
        feature_dim: int = 256,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        in_dim = n_features * 2  # values + masks
        layers: List[nn.Module] = []
        prev = in_dim
        for h in hidden_dims:
            layers += [
                nn.Linear(prev, h),
                nn.LayerNorm(h),
                nn.GELU(),
                nn.Dropout(dropout),
            ]
            prev = h
        layers.append(nn.Linear(prev, feature_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, n_features * 2)."""
        return self.net(x)


# ──────────────────────────────────────────────────────────────────────────────
# Cross-modal fusion
# ──────────────────────────────────────────────────────────────────────────────

class CrossModalFusion(nn.Module):
    """
    Bidirectional cross-attention between MRI and tabular embeddings,
    followed by gated fusion.

    - MRI queries tabular → tabular-attended MRI
    - Tabular queries MRI → MRI-attended tabular
    - Gated blend: α*m2 + (1-α)*t2 + linear projection of concat
    """

    def __init__(
        self,
        mri_dim: int = 512,
        tab_dim: int = 256,
        fusion_dim: int = 512,
        num_heads: int = 8,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.proj_m = nn.Linear(mri_dim, fusion_dim)
        self.proj_t = nn.Linear(tab_dim, fusion_dim)

        self.attn_m = nn.MultiheadAttention(fusion_dim, num_heads, dropout=dropout, batch_first=True)
        self.attn_t = nn.MultiheadAttention(fusion_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm_m = nn.LayerNorm(fusion_dim)
        self.norm_t = nn.LayerNorm(fusion_dim)

        self.gate = nn.Linear(fusion_dim * 2, fusion_dim)
        self.out = nn.Linear(fusion_dim * 2, fusion_dim)

    def forward(self, m_feat: torch.Tensor, t_feat: torch.Tensor) -> torch.Tensor:
        m1 = self.proj_m(m_feat).unsqueeze(1)   # (B, 1, D)
        t1 = self.proj_t(t_feat).unsqueeze(1)   # (B, 1, D)

        ma, _ = self.attn_m(m1, t1, t1)
        ta, _ = self.attn_t(t1, m1, m1)

        m2 = self.norm_m(m1 + ma).squeeze(1)    # (B, D)
        t2 = self.norm_t(t1 + ta).squeeze(1)    # (B, D)

        cat = torch.cat([m2, t2], dim=1)         # (B, 2D)
        gate = torch.sigmoid(self.gate(cat))
        fused = self.out(cat)
        return gate * m2 + (1 - gate) * t2 + fused


# ──────────────────────────────────────────────────────────────────────────────
# Prediction heads
# ──────────────────────────────────────────────────────────────────────────────

class ClassificationHead(nn.Module):
    def __init__(self, in_dim: int, num_classes: int = 3, dropout: float = 0.3) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(256, 128), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ProgressionHead(nn.Module):
    """
    Three-output progression head (MCI-only).

    Outputs:
        will_progress_logits : (B, 1)   raw logit for BCE loss
        time_to_conversion   : (B, 1)   predicted months [0, max_months]
        time_distribution    : (B, n_bins) probability over time bins
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 256,
        max_months: int = 120,
        n_bins: int = 24,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.trunk = nn.Sequential(nn.Linear(in_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout))
        self.will_progress = nn.Linear(hidden_dim, 1)
        self.time_regress = nn.Linear(hidden_dim, 1)
        self.time_dist = nn.Linear(hidden_dim, n_bins)
        self.max_months = max_months

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        h = self.trunk(x)
        raw_time = F.softplus(self.time_regress(h))
        time_clamped = torch.clamp(raw_time, 0, self.max_months)
        return {
            'will_progress_logits': self.will_progress(h),
            'time_to_conversion': time_clamped,
            'time_distribution': F.softmax(self.time_dist(h), dim=-1),
        }


# ──────────────────────────────────────────────────────────────────────────────
# Main model
# ──────────────────────────────────────────────────────────────────────────────

class VbaiProgressionNet(nn.Module):
    """
    Multimodal 3D MRI + biomarker model for Alzheimer's classification
    and MCI-to-AD progression prediction.

    Supports:
      - MRI only (tab=None)
      - Biomarkers only (mri=None)
      - Both modalities (full fusion)

    Args:
        mri_in_channels: Input MRI channels (1 for T1).
        n_tabular_features: Number of clinical biomarker features (default 13).
        num_classes: Classification targets (default 3: CN / MCI / AD).
        mri_feature_dim: MRI encoder output dimension.
        tab_feature_dim: Tabular encoder output dimension.
        fusion_dim: Cross-modal fusion dimension.
        num_attn_heads: Attention heads in fusion module.
        max_progression_months: Upper bound for time-to-conversion prediction.
        n_time_bins: Bins for time distribution head.
        mri_dropout: MRI encoder dropout rate.
        tab_dropout: Tabular encoder dropout rate.
        fusion_dropout: Fusion module dropout rate.

    Example::

        model = VbaiProgressionNet()

        # Full multimodal prediction
        out = model(mri=volume, tab=biomarkers)

        # MRI-only prediction
        out = model(mri=volume)

        # Classification probabilities
        probs = torch.softmax(out['fused_logits'], dim=-1)

        # Progression risk for MCI patients
        risk = torch.sigmoid(out['progression']['will_progress_logits'])
    """

    CLASS_NAMES = ['CN', 'MCI', 'AD']

    def __init__(
        self,
        mri_in_channels: int = 1,
        n_tabular_features: int = 13,
        num_classes: int = 3,
        mri_feature_dim: int = 512,
        tab_feature_dim: int = 256,
        fusion_dim: int = 512,
        num_attn_heads: int = 8,
        max_progression_months: int = 120,
        n_time_bins: int = 24,
        mri_dropout: float = 0.4,
        tab_dropout: float = 0.3,
        fusion_dropout: float = 0.3,
    ) -> None:
        super().__init__()

        self.n_tabular_features = n_tabular_features

        # Encoders
        self.mri_encoder = MRIEncoder3D(
            in_channels=mri_in_channels,
            feature_dim=mri_feature_dim,
            dropout=mri_dropout,
        )
        self.tab_encoder = TabularEncoder(
            n_features=n_tabular_features,
            feature_dim=tab_feature_dim,
            dropout=tab_dropout,
        )

        # Fusion
        self.fusion = CrossModalFusion(
            mri_dim=mri_feature_dim,
            tab_dim=tab_feature_dim,
            fusion_dim=fusion_dim,
            num_heads=num_attn_heads,
            dropout=fusion_dropout,
        )

        # Classification heads
        self.mri_cls = ClassificationHead(mri_feature_dim, num_classes, fusion_dropout)
        self.tab_cls = ClassificationHead(tab_feature_dim, num_classes, fusion_dropout)
        self.fused_cls = ClassificationHead(fusion_dim, num_classes, fusion_dropout)

        # Progression head
        self.progression = ProgressionHead(
            fusion_dim,
            max_months=max_progression_months,
            n_bins=n_time_bins,
            dropout=fusion_dropout,
        )

        # Contrastive projection heads (used during training only)
        self.contrast_mri = nn.Linear(mri_feature_dim, 128)
        self.contrast_tab = nn.Linear(tab_feature_dim, 128)

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, (nn.Linear,)):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    # ── Forward ───────────────────────────────────────────────────────────────

    def forward(
        self,
        mri: Optional[torch.Tensor] = None,
        tab: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            mri: (B, C, D, H, W) MRI volume. Optional.
            tab: (B, n_features * 2) tabular biomarkers [values | masks]. Optional.

        Returns:
            Dictionary with keys:
              'fused_logits'   (B, num_classes)    — primary prediction
              'mri_logits'     (B, num_classes)    — if mri given
              'tab_logits'     (B, num_classes)    — if tab given
              'progression'    dict                — if both modalities given
              'mri_features'   (B, mri_feature_dim)
              'tab_features'   (B, tab_feature_dim)
              'fused_features' (B, fusion_dim)
              'zm'             (B, 128) L2-normalised MRI contrastive emb.
              'zt'             (B, 128) L2-normalised tab contrastive emb.
        """
        out: Dict[str, torch.Tensor] = {}
        m_feat = t_feat = None

        if mri is not None:
            m_feat = self.mri_encoder(mri)
            out['mri_features'] = m_feat
            out['mri_logits'] = self.mri_cls(m_feat)

        if tab is not None:
            t_feat = self.tab_encoder(tab)
            out['tab_features'] = t_feat
            out['tab_logits'] = self.tab_cls(t_feat)

        if m_feat is not None and t_feat is not None:
            f = self.fusion(m_feat, t_feat)
            out['fused_features'] = f
            out['fused_logits'] = self.fused_cls(f)
            out['progression'] = self.progression(f)

            zm = F.normalize(self.contrast_mri(m_feat), dim=-1)
            zt = F.normalize(self.contrast_tab(t_feat), dim=-1)
            out['zm'] = zm
            out['zt'] = zt

        elif m_feat is not None:
            out['fused_logits'] = out['mri_logits']

        elif t_feat is not None:
            out['fused_logits'] = out['tab_logits']

        return out

    # ── Inference ─────────────────────────────────────────────────────────────

    @torch.no_grad()
    def predict(
        self,
        mri: Optional[torch.Tensor] = None,
        tab: Optional[torch.Tensor] = None,
        class_names: Optional[List[str]] = None,
    ) -> Dict:
        """
        High-level inference returning human-readable results.

        Args:
            mri: (1, C, D, H, W) or (C, D, H, W) volume tensor.
            tab: (1, n_features * 2) or (n_features * 2,) tabular tensor.
            class_names: Override default class names ['CN', 'MCI', 'AD'].

        Returns:
            dict with:
              'class_name'      — predicted class string
              'class_probs'     — {class: probability} dict
              'confidence'      — max probability
              'progression'     — progression sub-dict (if both modalities given)
        """
        self.eval()
        names = class_names or self.CLASS_NAMES
        device = next(self.parameters()).device

        if mri is not None:
            if mri.dim() == 4:
                mri = mri.unsqueeze(0)
            mri = mri.to(device)

        if tab is not None:
            if tab.dim() == 1:
                tab = tab.unsqueeze(0)
            tab = tab.to(device)

        out = self(mri, tab)

        logits = out['fused_logits']
        probs = torch.softmax(logits, dim=-1).squeeze(0).cpu()
        pred_idx = int(probs.argmax())

        result: Dict = {
            'class_name': names[pred_idx] if pred_idx < len(names) else str(pred_idx),
            'class_probs': {n: float(p) for n, p in zip(names, probs)},
            'confidence': float(probs.max()),
        }

        if 'progression' in out:
            prog = out['progression']
            will_prog = float(torch.sigmoid(prog['will_progress_logits'].squeeze()).cpu())
            time_conv = float(prog['time_to_conversion'].squeeze().cpu())
            time_dist = prog['time_distribution'].squeeze().cpu().tolist()
            risk_cat = (
                'High' if will_prog > 0.6
                else 'Moderate' if will_prog > 0.35
                else 'Low'
            )
            result['progression'] = {
                'will_progress_probability': will_prog,
                'estimated_months_to_conversion': time_conv,
                'time_bin_distribution': time_dist,
                'risk_category': risk_cat,
            }

        return result

    # ── Utilities ─────────────────────────────────────────────────────────────

    def count_parameters(self) -> Dict[str, int]:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {'total': total, 'trainable': trainable}

    def save(self, path: str) -> None:
        torch.save({
            'model_state_dict': self.state_dict(),
            'config': {
                'mri_in_channels': self.mri_encoder.stem[0].in_channels,
                'n_tabular_features': self.n_tabular_features,
            },
        }, path)

    @classmethod
    def load(cls, path: str, device: str = 'cpu', **override_kw) -> 'VbaiProgressionNet':
        ckpt = torch.load(path, map_location=device, weights_only=False)
        cfg = ckpt.get('config', {})
        cfg.update(override_kw)
        model = cls(**cfg)
        model.load_state_dict(ckpt['model_state_dict'])
        model.to(device).eval()
        return model
