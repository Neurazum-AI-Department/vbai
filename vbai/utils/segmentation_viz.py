"""
Visualization utilities for VbaiSegNet3D segmentation results.

Functions:
  plot_segmentation_slices  — multi-axis MRI + prediction overlay
  plot_dice_per_class       — per-class Dice bar chart
  compute_segmentation_metrics — Dice, IoU, volume similarity
  plot_training_curves      — train/val loss & Dice history
"""

from __future__ import annotations
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    _MPL_AVAILABLE = True
except ImportError:
    _MPL_AVAILABLE = False


_TISSUE_COLOURS = ['#e74c3c', '#3498db', '#2ecc71']   # CSF, GM, WM
_TUMOR_COLOUR = '#e74c3c'


def _require_mpl():
    if not _MPL_AVAILABLE:
        raise ImportError("matplotlib required: pip install vbai[full]")


# ──────────────────────────────────────────────────────────────────────────────
# Slice viewer
# ──────────────────────────────────────────────────────────────────────────────

def plot_segmentation_slices(
    volume: np.ndarray,
    prediction: np.ndarray,
    ground_truth: Optional[np.ndarray] = None,
    channel: int = 0,
    modality: int = 0,
    n_slices: int = 5,
    alpha: float = 0.4,
    title: str = 'Segmentation',
    cmap_image: str = 'gray',
    save_path: Optional[str] = None,
    show: bool = False,
) -> 'plt.Figure':
    """
    Plot axial slices with segmentation overlays.

    Args:
        volume:       (C, D, H, W) or (D, H, W) float32 MRI volume.
        prediction:   (C_out, D, H, W) or (D, H, W) predicted probabilities ∈ [0, 1].
        ground_truth: Optional same shape as prediction (binary or proba).
        channel:      Output channel to visualise (for multi-class).
        modality:     Input modality channel to display.
        n_slices:     Number of evenly-spaced axial slices.
        alpha:        Overlay transparency.
        title:        Figure title.
        cmap_image:   Colormap for background image.
        save_path:    If given, save to this path.
        show:         Call plt.show().

    Returns:
        matplotlib Figure.
    """
    _require_mpl()

    if volume.ndim == 4:
        vol = volume[modality]
    else:
        vol = volume

    if prediction.ndim == 4:
        pred = prediction[channel]
    else:
        pred = prediction

    D = vol.shape[0]
    slice_indices = [int(D * (i + 1) / (n_slices + 1)) for i in range(n_slices)]

    n_rows = 2 if ground_truth is None else 3
    fig, axes = plt.subplots(n_rows, n_slices, figsize=(3 * n_slices, 3 * n_rows))
    if n_slices == 1:
        axes = [[ax] for ax in axes]

    for col, si in enumerate(slice_indices):
        img_sl = vol[si]
        pred_sl = pred[si]

        # Row 0: MRI image
        axes[0][col].imshow(img_sl, cmap=cmap_image, origin='lower')
        axes[0][col].axis('off')
        if col == 0:
            axes[0][col].set_ylabel('MRI', fontsize=9)
        axes[0][col].set_title(f'z={si}', fontsize=8)

        # Row 1: prediction overlay
        axes[1][col].imshow(img_sl, cmap=cmap_image, origin='lower')
        axes[1][col].imshow(pred_sl, cmap='Reds', alpha=alpha, vmin=0, vmax=1, origin='lower')
        axes[1][col].axis('off')
        if col == 0:
            axes[1][col].set_ylabel('Prediction', fontsize=9)

        # Row 2: ground truth overlay (optional)
        if ground_truth is not None:
            gt_ch = ground_truth[channel] if ground_truth.ndim == 4 else ground_truth
            axes[2][col].imshow(img_sl, cmap=cmap_image, origin='lower')
            axes[2][col].imshow(gt_ch[si], cmap='Blues', alpha=alpha, vmin=0, vmax=1, origin='lower')
            axes[2][col].axis('off')
            if col == 0:
                axes[2][col].set_ylabel('Ground Truth', fontsize=9)

    fig.suptitle(title, fontsize=13, fontweight='bold')
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, bbox_inches='tight', dpi=150)
    if show:
        plt.show()
    return fig


# ──────────────────────────────────────────────────────────────────────────────
# Metrics
# ──────────────────────────────────────────────────────────────────────────────

def compute_segmentation_metrics(
    prediction: np.ndarray,
    ground_truth: np.ndarray,
    threshold: float = 0.5,
    smooth: float = 1e-5,
) -> Dict[str, float]:
    """
    Compute segmentation quality metrics for a single volume.

    Works for both binary (C=1) and multi-class (C>1) cases.
    For multi-class, returns per-channel results under keys like 'dice_0'.

    Args:
        prediction:   (C, D, H, W) probability map ∈ [0, 1].
        ground_truth: (C, D, H, W) binary or probability map.
        threshold:    Binarisation threshold.
        smooth:       Numerical stability term.

    Returns:
        dict with 'dice', 'iou', 'vol_sim', and per-channel variants.
    """
    if prediction.ndim == 3:
        prediction = prediction[np.newaxis]
        ground_truth = ground_truth[np.newaxis]

    pred_bin = (prediction > threshold).astype(np.float32)
    gt_bin = (ground_truth > 0.5).astype(np.float32)

    metrics: Dict[str, float] = {}
    dice_list, iou_list, vs_list = [], [], []

    for c in range(pred_bin.shape[0]):
        p = pred_bin[c].ravel()
        g = gt_bin[c].ravel()

        inter = (p * g).sum()
        dsc = (2 * inter + smooth) / (p.sum() + g.sum() + smooth)
        iou = (inter + smooth) / (p.sum() + g.sum() - inter + smooth)
        vs = 1 - abs(p.sum() - g.sum()) / (p.sum() + g.sum() + smooth)

        dice_list.append(float(dsc))
        iou_list.append(float(iou))
        vs_list.append(float(vs))

        if pred_bin.shape[0] > 1:
            metrics[f'dice_{c}'] = float(dsc)
            metrics[f'iou_{c}'] = float(iou)
            metrics[f'vol_sim_{c}'] = float(vs)

    metrics['dice'] = float(np.mean(dice_list))
    metrics['iou'] = float(np.mean(iou_list))
    metrics['vol_sim'] = float(np.mean(vs_list))
    return metrics


# ──────────────────────────────────────────────────────────────────────────────
# Plotting helpers
# ──────────────────────────────────────────────────────────────────────────────

def plot_dice_per_class(
    metrics: Dict[str, float],
    class_names: Optional[List[str]] = None,
    save_path: Optional[str] = None,
    show: bool = False,
) -> 'plt.Figure':
    """Bar chart of per-class Dice scores."""
    _require_mpl()

    dice_keys = sorted(k for k in metrics if k.startswith('dice_'))
    if not dice_keys:
        dice_keys = ['dice']
    dices = [metrics[k] for k in dice_keys]
    labels = class_names or [k.replace('dice_', 'Class ') for k in dice_keys]

    colours = _TISSUE_COLOURS[:len(labels)] if len(labels) > 1 else [_TUMOR_COLOUR]

    fig, ax = plt.subplots(figsize=(max(4, len(labels) * 1.5), 3.5))
    bars = ax.bar(labels, dices, color=colours, edgecolor='white', linewidth=1.5)
    for bar, d in zip(bars, dices):
        ax.text(bar.get_x() + bar.get_width() / 2, d + 0.01, f'{d:.3f}',
                ha='center', fontsize=10, fontweight='bold')
    ax.set_ylim(0, 1.12)
    ax.set_ylabel('Dice Score')
    ax.set_title('Segmentation Quality (Dice)', fontweight='bold')
    ax.axhline(0.8, color='gray', linestyle='--', alpha=0.5, label='0.8 threshold')
    ax.legend(fontsize=8)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, bbox_inches='tight', dpi=150)
    if show:
        plt.show()
    return fig


def plot_training_curves(
    history,
    save_path: Optional[str] = None,
    show: bool = False,
) -> 'plt.Figure':
    """
    Plot loss and Dice curves from a SegmentationHistory object.

    Args:
        history: SegmentationHistory (from SegmentationTrainer.fit()).
        save_path: Optional save path.
        show: Call plt.show().
    """
    _require_mpl()

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Loss
    axes[0].plot(history.train_loss, label='Train', color='#3498db', lw=2)
    if history.val_loss:
        axes[0].plot(history.val_loss, label='Validation', color='#e74c3c', lw=2)
    if history.best_epoch:
        axes[0].axvline(history.best_epoch - 1, color='gray', linestyle='--', alpha=0.5, label=f'Best epoch {history.best_epoch}')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training Loss', fontweight='bold')
    axes[0].legend()
    axes[0].spines['top'].set_visible(False)
    axes[0].spines['right'].set_visible(False)

    # Dice
    axes[1].plot(history.train_dice, label='Train', color='#3498db', lw=2)
    if history.val_dice:
        axes[1].plot(history.val_dice, label='Validation', color='#e74c3c', lw=2)
    if history.best_epoch:
        axes[1].axvline(history.best_epoch - 1, color='gray', linestyle='--', alpha=0.5)
        axes[1].axhline(history.best_val_dice, color='#2ecc71', linestyle=':', alpha=0.7,
                        label=f'Best Dice {history.best_val_dice:.4f}')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Dice')
    axes[1].set_title('Dice Score', fontweight='bold')
    axes[1].set_ylim(0, 1)
    axes[1].legend()
    axes[1].spines['top'].set_visible(False)
    axes[1].spines['right'].set_visible(False)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches='tight', dpi=150)
    if show:
        plt.show()
    return fig
