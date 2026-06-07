"""
VbaiSegNet3D — 3D Medical Image Segmentation Network

Encoder-decoder (UNet-style) with:
  - Squeeze-and-Excitation (SE) channel attention
  - Convolutional Block Attention Module (CBAM)
  - Atrous Spatial Pyramid Pooling (ASPP)
  - Attention gates on skip connections
  - Optional deep supervision
  - Kaiming weight initialization

Supports:
  - Binary segmentation (e.g. tumor mask, 1 output channel)
  - Multi-class probabilistic segmentation (e.g. tissue: CSF / GM / WM)
"""

from __future__ import annotations
import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ──────────────────────────────────────────────────────────────────────────────
# Atomic blocks
# ──────────────────────────────────────────────────────────────────────────────

class SEBlock3D(nn.Module):
    """Squeeze-and-Excitation channel recalibration for 3-D feature maps."""

    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()
        mid = max(channels // reduction, 4)
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, mid, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c = x.shape[:2]
        w = self.pool(x).view(b, c)
        w = self.fc(w).view(b, c, 1, 1, 1)
        return x * w


class CBAM3D(nn.Module):
    """3-D Convolutional Block Attention Module (channel + spatial)."""

    def __init__(self, channels: int, reduction: int = 16, kernel: int = 7) -> None:
        super().__init__()
        mid = max(channels // reduction, 4)
        # Channel attention
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.max_pool = nn.AdaptiveMaxPool3d(1)
        self.ch_fc = nn.Sequential(
            nn.Linear(channels, mid, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid, channels, bias=False),
        )
        # Spatial attention
        pad = kernel // 2
        self.sp_conv = nn.Conv3d(2, 1, kernel_size=kernel, padding=pad, bias=False)
        self.sp_bn = nn.BatchNorm3d(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c = x.shape[:2]
        # channel
        avg = self.ch_fc(self.avg_pool(x).view(b, c))
        mx = self.ch_fc(self.max_pool(x).view(b, c))
        ch_w = torch.sigmoid(avg + mx).view(b, c, 1, 1, 1)
        x = x * ch_w
        # spatial
        sp_in = torch.cat([x.mean(1, keepdim=True), x.amax(1, keepdim=True)], dim=1)
        sp_w = torch.sigmoid(self.sp_bn(self.sp_conv(sp_in)))
        return x * sp_w


class ResBlock3D(nn.Module):
    """3-D residual block with optional SE and CBAM attention."""

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        stride: int = 1,
        use_se: bool = True,
        use_cbam: bool = True,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.conv1 = nn.Conv3d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(out_ch)
        self.drop = nn.Dropout3d(dropout)
        self.conv2 = nn.Conv3d(out_ch, out_ch, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(out_ch)
        self.act = nn.ReLU(inplace=True)

        self.se = SEBlock3D(out_ch) if use_se else None
        self.cbam = CBAM3D(out_ch) if use_cbam else None

        self.skip = None
        if in_ch != out_ch or stride != 1:
            self.skip = nn.Sequential(
                nn.Conv3d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm3d(out_ch),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.skip(x) if self.skip else x
        out = self.act(self.bn1(self.conv1(x)))
        out = self.drop(out)
        out = self.bn2(self.conv2(out))
        if self.se:
            out = self.se(out)
        if self.cbam:
            out = self.cbam(out)
        return self.act(out + identity)


class ASPP3D(nn.Module):
    """Atrous Spatial Pyramid Pooling for 3-D volumes."""

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        dilations: Tuple[int, ...] = (1, 3, 6),
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        mid = out_ch // (len(dilations) + 2)

        self.branches = nn.ModuleList()
        for d in dilations:
            self.branches.append(nn.Sequential(
                nn.Conv3d(in_ch, mid, 3, padding=d, dilation=d, bias=False),
                nn.BatchNorm3d(mid),
                nn.ReLU(inplace=True),
            ))
        # point-wise branch
        self.branches.append(nn.Sequential(
            nn.Conv3d(in_ch, mid, 1, bias=False),
            nn.BatchNorm3d(mid),
            nn.ReLU(inplace=True),
        ))
        # global average pool branch (no BN to avoid batch-size issues)
        self.gp_conv = nn.Conv3d(in_ch, mid, 1, bias=False)
        self.gp_act = nn.ReLU(inplace=True)

        total_mid = mid * (len(dilations) + 2)
        self.project = nn.Sequential(
            nn.Conv3d(total_mid, out_ch, 1, bias=False),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
            nn.Dropout3d(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = [b(x) for b in self.branches]
        gp = F.adaptive_avg_pool3d(x, 1)
        gp = self.gp_act(self.gp_conv(gp))
        gp = F.interpolate(gp, size=x.shape[2:], mode='trilinear', align_corners=False)
        feats.append(gp)
        return self.project(torch.cat(feats, dim=1))


class AttentionGate3D(nn.Module):
    """Attention gate that suppresses irrelevant skip-connection features."""

    def __init__(self, skip_ch: int, gate_ch: int) -> None:
        super().__init__()
        inter = max(skip_ch // 2, 1)
        self.Wskip = nn.Sequential(nn.Conv3d(skip_ch, inter, 1, bias=False), nn.BatchNorm3d(inter))
        self.Wgate = nn.Sequential(nn.Conv3d(gate_ch, inter, 1, bias=False), nn.BatchNorm3d(inter))
        self.psi = nn.Sequential(nn.Conv3d(inter, 1, 1, bias=False), nn.BatchNorm3d(1), nn.Sigmoid())

    def forward(self, skip: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        g_up = F.interpolate(gate, size=skip.shape[2:], mode='trilinear', align_corners=False)
        attn = self.psi(F.relu(self.Wskip(skip) + self.Wgate(g_up), inplace=True))
        return skip * attn


# ──────────────────────────────────────────────────────────────────────────────
# Encoder / Decoder blocks
# ──────────────────────────────────────────────────────────────────────────────

class EncoderBlock(nn.Module):
    """Two residual blocks followed by a strided-conv downsampler."""

    def __init__(self, in_ch: int, out_ch: int, **res_kw) -> None:
        super().__init__()
        self.res = nn.Sequential(
            ResBlock3D(in_ch, out_ch, **res_kw),
            ResBlock3D(out_ch, out_ch, **res_kw),
        )
        self.down = nn.Sequential(
            nn.Conv3d(out_ch, out_ch, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        skip = self.res(x)
        return skip, self.down(skip)


class DecoderBlock(nn.Module):
    """Upsample → optional attention gate → concat skip → two residual blocks."""

    def __init__(
        self,
        in_ch: int,
        skip_ch: int,
        out_ch: int,
        use_attn_gate: bool = True,
        **res_kw,
    ) -> None:
        super().__init__()
        self.gate = AttentionGate3D(skip_ch, in_ch) if use_attn_gate else None
        self.res = nn.Sequential(
            ResBlock3D(in_ch + skip_ch, out_ch, **res_kw),
            ResBlock3D(out_ch, out_ch, **res_kw),
        )

    def forward(
        self, x: torch.Tensor, skip: torch.Tensor
    ) -> torch.Tensor:
        x = F.interpolate(x, size=skip.shape[2:], mode='trilinear', align_corners=False)
        if self.gate:
            skip = self.gate(skip, x)
        return self.res(torch.cat([x, skip], dim=1))


# ──────────────────────────────────────────────────────────────────────────────
# Main model
# ──────────────────────────────────────────────────────────────────────────────

class VbaiSegNet3D(nn.Module):
    """
    3-D encoder-decoder segmentation network.

    Args:
        in_channels: Number of input MRI modalities.
        out_channels: Number of segmentation output channels
            (1 = binary, N = probabilistic multi-class).
        base_channels: Stem output channels (default 32).
        channel_mult: Per-level channel multipliers applied to *base_channels*.
        use_se: Enable Squeeze-and-Excitation blocks.
        use_cbam: Enable CBAM blocks.
        use_aspp: Enable ASPP bottleneck.
        use_attn_gate: Enable attention gates in decoder.
        use_deep_supervision: Return auxiliary decoder outputs during training.
        dropout: Dropout rate inside residual blocks.

    Example — binary tumor segmentation (2 modalities in, 1 out)::

        model = VbaiSegNet3D(in_channels=2, out_channels=1)
        logits = model(volume)            # (B, 1, D, H, W)

    Example — 3-class tissue segmentation (1 modality in, 3 out)::

        model = VbaiSegNet3D(in_channels=1, out_channels=3)
        logits, aux = model(volume, return_aux=True)
    """

    # Channel progression at each encoder level
    _DEFAULT_MULT = (1, 2, 4, 8, 10)   # → [32, 64, 128, 256, 320] with base=32

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        base_channels: int = 32,
        channel_mult: Tuple[int, ...] = _DEFAULT_MULT,
        use_se: bool = True,
        use_cbam: bool = True,
        use_aspp: bool = True,
        use_attn_gate: bool = True,
        use_deep_supervision: bool = True,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.use_deep_supervision = use_deep_supervision

        if len(channel_mult) < 5:
            raise ValueError(
                f"channel_mult must have at least 5 elements (4 encoder levels + 1 bottleneck), "
                f"got {len(channel_mult)}. Use base_channels to reduce model size instead."
            )
        ch = [base_channels * m for m in channel_mult]  # e.g. [32,64,128,256,320]
        res_kw = dict(use_se=use_se, use_cbam=use_cbam, dropout=dropout)

        # ── Stem ──────────────────────────────────────────────────────────────
        self.stem = nn.Sequential(
            nn.Conv3d(in_channels, ch[0], 3, padding=1, bias=False),
            nn.BatchNorm3d(ch[0]),
            nn.ReLU(inplace=True),
        )

        # ── Encoder (4 levels) ────────────────────────────────────────────────
        self.enc0 = EncoderBlock(ch[0], ch[0], **res_kw)
        self.enc1 = EncoderBlock(ch[0], ch[1], **res_kw)
        self.enc2 = EncoderBlock(ch[1], ch[2], **res_kw)
        self.enc3 = EncoderBlock(ch[2], ch[3], **res_kw)

        # ── Bottleneck ────────────────────────────────────────────────────────
        if use_aspp:
            self.bottleneck = ASPP3D(ch[3], ch[4], dropout=dropout)
        else:
            self.bottleneck = nn.Sequential(
                ResBlock3D(ch[3], ch[4], **res_kw),
                ResBlock3D(ch[4], ch[4], **res_kw),
            )

        # ── Decoder (4 levels) ────────────────────────────────────────────────
        dec_kw = dict(use_attn_gate=use_attn_gate, **res_kw)
        self.dec3 = DecoderBlock(ch[4], ch[3], ch[3], **dec_kw)
        self.dec2 = DecoderBlock(ch[3], ch[2], ch[2], **dec_kw)
        self.dec1 = DecoderBlock(ch[2], ch[1], ch[1], **dec_kw)
        self.dec0 = DecoderBlock(ch[1], ch[0], ch[0], **dec_kw)

        # ── Segmentation head ─────────────────────────────────────────────────
        self.head = nn.Conv3d(ch[0], out_channels, 1)

        # ── Deep supervision heads ────────────────────────────────────────────
        if use_deep_supervision:
            self.ds_head3 = nn.Conv3d(ch[3], out_channels, 1)
            self.ds_head2 = nn.Conv3d(ch[2], out_channels, 1)
            self.ds_head1 = nn.Conv3d(ch[1], out_channels, 1)

        self._init_weights()

    # ── Weight init ───────────────────────────────────────────────────────────

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    # ── Forward ───────────────────────────────────────────────────────────────

    def forward(
        self,
        x: torch.Tensor,
        return_aux: bool = False,
    ) -> torch.Tensor | Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Args:
            x: Input volume (B, C_in, D, H, W).
            return_aux: If True *and* deep supervision is enabled, returns
                (main_logits, [aux3, aux2, aux1]).

        Returns:
            main_logits (B, C_out, D, H, W), or (main_logits, aux_list).
        """
        # Stem
        x = self.stem(x)

        # Encoder
        s0, x = self.enc0(x)
        s1, x = self.enc1(x)
        s2, x = self.enc2(x)
        s3, x = self.enc3(x)

        # Bottleneck
        x = self.bottleneck(x)

        # Decoder
        d3 = self.dec3(x, s3)
        d2 = self.dec2(d3, s2)
        d1 = self.dec1(d2, s1)
        d0 = self.dec0(d1, s0)

        logits = self.head(d0)

        if return_aux and self.use_deep_supervision:
            aux = [self.ds_head3(d3), self.ds_head2(d2), self.ds_head1(d1)]
            return logits, aux

        return logits

    # ── Inference helpers ─────────────────────────────────────────────────────

    @torch.no_grad()
    def predict_volume(
        self,
        volume: torch.Tensor,
        threshold: float = 0.5,
        patch_size: Tuple[int, int, int] = (96, 96, 96),
        overlap: float = 0.5,
    ) -> torch.Tensor:
        """
        Sliding-window inference on an arbitrary-size volume.

        Args:
            volume: (1, C, D, H, W) or (C, D, H, W) tensor.
            threshold: Binarisation threshold for single-channel output.
            patch_size: Patch dimensions used during inference.
            overlap: Overlap fraction between adjacent patches [0, 1).

        Returns:
            Probability map (1, C_out, D, H, W) ∈ [0, 1].
        """
        self.eval()
        if volume.dim() == 4:
            volume = volume.unsqueeze(0)

        device = next(self.parameters()).device
        volume = volume.to(device)

        D, H, W = volume.shape[2:]
        pd, ph, pw = patch_size
        step = max(1, int(min(patch_size) * (1 - overlap)))

        out_ch = self.head.out_channels
        pred_sum = torch.zeros(1, out_ch, D, H, W, device=device)
        weight_sum = torch.zeros(1, 1, D, H, W, device=device)

        # Gaussian blending kernel
        def _gauss_w(size):
            c = (size - 1) / 2
            idx = torch.arange(size, dtype=torch.float32, device=device)
            return torch.exp(-0.5 * ((idx - c) / (size * 0.125)) ** 2)

        kd = _gauss_w(pd).view(-1, 1, 1)
        kh = _gauss_w(ph).view(1, -1, 1)
        kw = _gauss_w(pw).view(1, 1, -1)
        kernel = (kd * kh * kw).unsqueeze(0).unsqueeze(0)  # (1,1,pd,ph,pw)

        d_starts = list(range(0, max(D - pd + 1, 1), step))
        h_starts = list(range(0, max(H - ph + 1, 1), step))
        w_starts = list(range(0, max(W - pw + 1, 1), step))

        for d0 in d_starts:
            for h0 in h_starts:
                for w0 in w_starts:
                    d1, h1, w1 = min(d0 + pd, D), min(h0 + ph, H), min(w0 + pw, W)
                    d0_, h0_, w0_ = d1 - pd, h1 - ph, w1 - pw

                    patch = volume[:, :, d0_:d1, h0_:h1, w0_:w1]
                    logits = self(patch)
                    probs = torch.sigmoid(logits)

                    k = kernel[:, :, :d1 - d0_, :h1 - h0_, :w1 - w0_]
                    pred_sum[:, :, d0_:d1, h0_:h1, w0_:w1] += probs * k
                    weight_sum[:, :, d0_:d1, h0_:h1, w0_:w1] += k

        return pred_sum / (weight_sum + 1e-8)

    def count_parameters(self) -> dict:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {'total': total, 'trainable': trainable}

    def save(self, path: str) -> None:
        """Save model checkpoint."""
        torch.save({
            'model_state_dict': self.state_dict(),
            'config': {
                'in_channels': self.stem[0].in_channels,
                'out_channels': self.head.out_channels,
                'use_deep_supervision': self.use_deep_supervision,
            },
        }, path)

    @classmethod
    def load(cls, path: str, device: str = 'cpu', **override_kw) -> 'VbaiSegNet3D':
        """Load from checkpoint."""
        ckpt = torch.load(path, map_location=device, weights_only=False)
        cfg = ckpt.get('config', {})
        cfg.update(override_kw)
        model = cls(**cfg)
        model.load_state_dict(ckpt['model_state_dict'])
        model.to(device).eval()
        return model
