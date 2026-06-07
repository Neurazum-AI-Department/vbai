"""
Professional clinical visualization for VbaiProgressionNet predictions.

Functions:
  plot_progression_report   — comprehensive clinical-style PDF/PNG report
  plot_risk_gauge           — circular risk gauge (matplotlib)
  plot_time_distribution    — time-to-conversion histogram
  plot_class_probabilities  — CN / MCI / AD bar chart
  plot_biomarker_radar      — radar chart of biomarker values
  create_report_figure      — assembles all panels into one figure

All functions return matplotlib Figure objects so the caller can
save (fig.savefig), display (plt.show), or embed in reports.
"""

from __future__ import annotations
import math
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.gridspec import GridSpec
    from matplotlib.patches import FancyArrowPatch
    _MPL_AVAILABLE = True
except ImportError:
    _MPL_AVAILABLE = False


# Colour palette (consistent across all panels)
_PALETTE = {
    'CN': '#2ecc71',      # green
    'MCI': '#f39c12',     # amber
    'AD': '#e74c3c',      # red
    'low': '#27ae60',
    'moderate': '#e67e22',
    'high': '#c0392b',
    'bg': '#f8f9fa',
    'border': '#dee2e6',
    'text': '#212529',
    'accent': '#3498db',
}


def _require_mpl():
    if not _MPL_AVAILABLE:
        raise ImportError(
            "matplotlib is required for visualization.  "
            "pip install vbai[full]"
        )


# ──────────────────────────────────────────────────────────────────────────────
# Individual panels
# ──────────────────────────────────────────────────────────────────────────────

def plot_class_probabilities(
    class_probs: Dict[str, float],
    ax=None,
) -> 'plt.Axes':
    """Horizontal bar chart of CN / MCI / AD probabilities."""
    _require_mpl()
    if ax is None:
        _, ax = plt.subplots(figsize=(5, 2.5))

    classes = ['CN', 'MCI', 'AD']
    probs = [class_probs.get(c, 0.0) for c in classes]
    colours = [_PALETTE[c] for c in classes]

    bars = ax.barh(classes, probs, color=colours, height=0.55, edgecolor='white', linewidth=1.5)
    for bar, p in zip(bars, probs):
        ax.text(
            min(p + 0.02, 0.97), bar.get_y() + bar.get_height() / 2,
            f'{p:.1%}', va='center', ha='left', fontsize=10,
            fontweight='bold', color=_PALETTE['text'],
        )

    ax.set_xlim(0, 1.12)
    ax.set_xlabel('Probability', fontsize=9)
    ax.set_title('Classification Probabilities', fontsize=11, fontweight='bold')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_facecolor(_PALETTE['bg'])
    return ax


def plot_risk_gauge(
    risk_prob: float,
    ax=None,
) -> 'plt.Axes':
    """
    Semicircular risk gauge for MCI-to-AD conversion probability.

    0 % (left/green) → 100 % (right/red).
    """
    _require_mpl()
    if ax is None:
        _, ax = plt.subplots(figsize=(4, 2.8), subplot_kw={'aspect': 'equal'})

    theta_range = (180, 0)  # degrees: left → right

    # Background arc segments: green / amber / red
    for (t0, t1), colour in [
        ((120, 180), _PALETTE['low']),
        ((60, 120), _PALETTE['moderate']),
        ((0, 60), _PALETTE['high']),
    ]:
        thetas = np.linspace(math.radians(t0), math.radians(t1), 100)
        ax.plot(np.cos(thetas) * 0.9, np.sin(thetas) * 0.9, lw=18, color=colour, alpha=0.25, solid_capstyle='round')

    # Needle
    needle_angle = math.radians(180 - risk_prob * 180)
    ax.annotate(
        '', xy=(math.cos(needle_angle) * 0.75, math.sin(needle_angle) * 0.75),
        xytext=(0, 0),
        arrowprops=dict(arrowstyle='->', color=_PALETTE['text'], lw=2.5),
    )
    # Centre dot
    ax.add_patch(plt.Circle((0, 0), 0.05, color=_PALETTE['text'], zorder=5))

    # Labels
    ax.text(-0.95, -0.15, '0%', ha='center', fontsize=8, color=_PALETTE['low'], fontweight='bold')
    ax.text(0.95, -0.15, '100%', ha='center', fontsize=8, color=_PALETTE['high'], fontweight='bold')
    ax.text(0, -0.35, f'{risk_prob:.1%}', ha='center', fontsize=16, fontweight='bold', color=_PALETTE['text'])

    risk_cat = 'High' if risk_prob > 0.6 else 'Moderate' if risk_prob > 0.35 else 'Low'
    cat_col = _PALETTE[risk_cat.lower()]
    ax.text(0, -0.55, f'Risk: {risk_cat}', ha='center', fontsize=11, color=cat_col, fontweight='bold')

    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-0.7, 1.1)
    ax.axis('off')
    ax.set_title('Progression Risk\n(MCI → AD within 5 years)', fontsize=10, fontweight='bold', pad=4)
    return ax


def plot_time_distribution(
    time_distribution: List[float],
    estimated_months: float,
    n_bins: int = 24,
    bin_months: int = 5,
    ax=None,
) -> 'plt.Axes':
    """Bar chart of the predicted time-to-conversion probability distribution."""
    _require_mpl()
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 3))

    bins = list(range(n_bins))
    bin_labels = [f'{i * bin_months}' for i in bins]
    probs = time_distribution[:n_bins]
    if len(probs) < n_bins:
        probs += [0.0] * (n_bins - len(probs))

    colours = [_PALETTE['accent']] * n_bins
    est_bin = int(estimated_months // bin_months)
    if 0 <= est_bin < n_bins:
        colours[est_bin] = _PALETTE['high']

    ax.bar(bins, probs, color=colours, edgecolor='white', linewidth=0.5)

    # Show every 5 ticks for readability
    tick_step = max(1, n_bins // 6)
    ax.set_xticks(range(0, n_bins, tick_step))
    ax.set_xticklabels([bin_labels[i] for i in range(0, n_bins, tick_step)], fontsize=8)
    ax.set_xlabel('Months', fontsize=9)
    ax.set_ylabel('Probability', fontsize=9)
    ax.set_title(
        f'Time-to-Conversion Distribution  (est. {estimated_months:.0f} mo)',
        fontsize=10, fontweight='bold',
    )
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_facecolor(_PALETTE['bg'])
    return ax


def plot_biomarker_radar(
    biomarker_values: Dict[str, float],
    reference_ranges: Optional[Dict[str, Tuple[float, float]]] = None,
    ax=None,
) -> 'plt.Axes':
    """
    Radar / spider chart of available biomarker values.

    Args:
        biomarker_values: {feature_name: z-scored value}.
        reference_ranges: {feature_name: (lo, hi)} in z-score units.
        ax: matplotlib Axes (PolarAxes).
    """
    _require_mpl()

    features = [k for k, v in biomarker_values.items() if v is not None and not math.isnan(v)]
    if len(features) < 3:
        if ax is None:
            fig, ax = plt.subplots()
        ax.text(0.5, 0.5, 'Insufficient biomarker\ndata for radar chart.',
                ha='center', va='center', transform=ax.transAxes, fontsize=10)
        ax.axis('off')
        return ax

    if ax is None:
        _, ax = plt.subplots(figsize=(4.5, 4.5), subplot_kw={'polar': True})

    N = len(features)
    angles = [n / N * 2 * math.pi for n in range(N)]
    angles += angles[:1]

    values = [float(biomarker_values[f]) for f in features]
    values_plot = values + values[:1]

    # Clip to ±3 for display
    vals_clipped = [max(-3, min(3, v)) for v in values_plot]

    ax.plot(angles, vals_clipped, 'o-', linewidth=2, color=_PALETTE['accent'])
    ax.fill(angles, vals_clipped, alpha=0.25, color=_PALETTE['accent'])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(features, size=7)
    ax.set_ylim(-3.5, 3.5)
    ax.set_yticks([-2, 0, 2])
    ax.set_yticklabels(['-2σ', '0', '+2σ'], size=7)
    ax.set_title('Biomarker Profile (z-score)', size=10, fontweight='bold', pad=14)
    ax.set_facecolor(_PALETTE['bg'])
    return ax


# ──────────────────────────────────────────────────────────────────────────────
# Composite report
# ──────────────────────────────────────────────────────────────────────────────

def create_report_figure(
    prediction: Dict,
    biomarker_values: Optional[Dict[str, float]] = None,
    subject_id: Optional[str] = None,
    scan_date: Optional[str] = None,
    figsize: Tuple[float, float] = (16, 10),
) -> 'plt.Figure':
    """
    Assemble a full clinical-style report figure from a model prediction dict.

    Args:
        prediction: Output from VbaiProgressionNet.predict().
        biomarker_values: Raw (or z-scored) biomarker values for radar chart.
        subject_id: Patient/subject identifier string (displayed in header).
        scan_date: Scan date string (displayed in header).
        figsize: Figure size in inches.

    Returns:
        matplotlib Figure.

    Example::

        result = model.predict(mri=vol, tab=tab_tensor)
        fig = create_report_figure(result, subject_id='ADNI-123', scan_date='2024-01')
        fig.savefig('report.pdf', bbox_inches='tight')
    """
    _require_mpl()

    fig = plt.figure(figsize=figsize, facecolor='white')
    gs = GridSpec(2, 4, figure=fig, hspace=0.45, wspace=0.4,
                  left=0.05, right=0.97, top=0.88, bottom=0.06)

    # ── Header ────────────────────────────────────────────────────────────────
    header_ax = fig.add_axes([0, 0.91, 1, 0.09])
    header_ax.set_facecolor(_PALETTE['accent'])
    header_ax.axis('off')

    pred_class = prediction.get('class_name', '—')
    conf = prediction.get('confidence', 0.0)
    class_col = _PALETTE.get(pred_class, 'white')

    header_ax.text(0.02, 0.5, 'VBAI Clinical Report', color='white',
                   fontsize=16, fontweight='bold', va='center', ha='left', transform=header_ax.transAxes)
    if subject_id:
        header_ax.text(0.35, 0.5, f'Subject: {subject_id}', color='white',
                       fontsize=11, va='center', ha='left', transform=header_ax.transAxes)
    if scan_date:
        header_ax.text(0.55, 0.5, f'Scan: {scan_date}', color='white',
                       fontsize=11, va='center', ha='left', transform=header_ax.transAxes)
    header_ax.text(0.78, 0.5, f'Prediction: {pred_class}  ({conf:.1%})',
                   color='white', fontsize=13, fontweight='bold',
                   va='center', ha='left', transform=header_ax.transAxes)

    # ── Panel 1: class probabilities ──────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0:2])
    plot_class_probabilities(prediction.get('class_probs', {}), ax=ax1)

    # ── Panel 2: progression risk gauge (only if available) ───────────────────
    prog = prediction.get('progression')
    if prog is not None:
        ax2 = fig.add_subplot(gs[0, 2:4])
        plot_risk_gauge(prog.get('will_progress_probability', 0.0), ax=ax2)
    else:
        ax2 = fig.add_subplot(gs[0, 2:4])
        ax2.text(0.5, 0.5, 'Progression prediction\nrequires multimodal input\n(MRI + biomarkers).',
                 ha='center', va='center', transform=ax2.transAxes, fontsize=10, color='gray')
        ax2.axis('off')
        ax2.set_title('Progression Risk', fontsize=11, fontweight='bold')

    # ── Panel 3: time distribution (bottom left) ──────────────────────────────
    if prog is not None and 'time_bin_distribution' in prog:
        ax3 = fig.add_subplot(gs[1, 0:3])
        plot_time_distribution(
            prog['time_bin_distribution'],
            prog.get('estimated_months_to_conversion', 0.0),
            ax=ax3,
        )
    else:
        ax3 = fig.add_subplot(gs[1, 0:3])
        ax3.text(0.5, 0.5, 'No progression timeline available.', ha='center', va='center',
                 transform=ax3.transAxes, fontsize=10, color='gray')
        ax3.axis('off')

    # ── Panel 4: biomarker radar (bottom right) ───────────────────────────────
    if biomarker_values:
        ax4 = fig.add_subplot(gs[1, 3], polar=True)
        plot_biomarker_radar(biomarker_values, ax=ax4)
    else:
        ax4 = fig.add_subplot(gs[1, 3])
        ax4.text(0.5, 0.5, 'No biomarker data.', ha='center', va='center',
                 transform=ax4.transAxes, fontsize=10, color='gray')
        ax4.axis('off')

    # ── Summary text box ──────────────────────────────────────────────────────
    summary_lines = [f'Diagnosis: {pred_class}  (confidence {conf:.1%})']
    if prog:
        risk_cat = prog.get('risk_category', '—')
        will_p = prog.get('will_progress_probability', 0.0)
        est_mo = prog.get('estimated_months_to_conversion', 0.0)
        summary_lines += [
            f'MCI→AD Risk: {risk_cat} ({will_p:.1%})',
            f'Est. time to conversion: {est_mo:.0f} months',
        ]
    fig.text(0.5, 0.005, '  |  '.join(summary_lines), ha='center', fontsize=9,
             color=_PALETTE['text'], style='italic')

    return fig


def plot_progression_report(
    prediction: Dict,
    save_path: Optional[str] = None,
    biomarker_values: Optional[Dict[str, float]] = None,
    subject_id: Optional[str] = None,
    scan_date: Optional[str] = None,
    show: bool = False,
    **figure_kw,
) -> 'plt.Figure':
    """
    High-level wrapper: create and optionally save / display the report.

    Args:
        prediction: Output from VbaiProgressionNet.predict().
        save_path: File path to save (supports .png, .pdf, .svg).
        biomarker_values: Optional biomarker dict for radar chart.
        subject_id: Subject identifier for header.
        scan_date: Scan date string for header.
        show: Call plt.show() after creating the figure.
        **figure_kw: Forwarded to create_report_figure().

    Returns:
        matplotlib Figure.
    """
    _require_mpl()
    fig = create_report_figure(
        prediction, biomarker_values, subject_id, scan_date, **figure_kw
    )
    if save_path:
        fig.savefig(save_path, bbox_inches='tight', dpi=150)
    if show:
        plt.show()
    return fig
