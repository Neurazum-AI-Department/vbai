"""Vbai Utilities Module"""

from .visualization import (
    VisualizationManager,
    visualize_prediction,
    create_attention_heatmap,
    plot_training_history,
)
from .analysis import BrainStructureAnalyzer

# Segmentation visualization
from .segmentation_viz import (
    plot_segmentation_slices,
    plot_dice_per_class,
    compute_segmentation_metrics,
    plot_training_curves as plot_segmentation_training_curves,
)

# Progression / clinical report visualization
from .progression_viz import (
    plot_progression_report,
    plot_risk_gauge,
    plot_time_distribution,
    plot_class_probabilities,
    plot_biomarker_radar,
    create_report_figure,
)

__all__ = [
    # 2D / general
    'VisualizationManager',
    'visualize_prediction',
    'create_attention_heatmap',
    'plot_training_history',
    'BrainStructureAnalyzer',
    # Segmentation
    'plot_segmentation_slices',
    'plot_dice_per_class',
    'compute_segmentation_metrics',
    'plot_segmentation_training_curves',
    # Progression / clinical report
    'plot_progression_report',
    'plot_risk_gauge',
    'plot_time_distribution',
    'plot_class_probabilities',
    'plot_biomarker_radar',
    'create_report_figure',
]
