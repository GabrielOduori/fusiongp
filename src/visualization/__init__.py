"""
Visualization utilities for FusionGP.

This module provides plotting functions for:
- Spatial prediction maps
- Uncertainty visualization
- Training diagnostics
- Calibration plots
- Residual analysis

Functions
---------
plot_predictions
    Create spatial maps of predictions.
plot_uncertainty
    Visualize prediction uncertainty.
plot_training_history
    Plot training curves.
plot_calibration
    Create calibration diagnostic plots.
plot_residuals
    Analyze prediction residuals.
"""

from src.visualization.maps import plot_predictions, plot_spatial_field
from src.visualization.uncertainty import plot_uncertainty, plot_confidence_intervals
from src.visualization.diagnostics import (
    plot_training_history,
    plot_calibration,
    plot_residuals,
    plot_pit_histogram,
)

__all__ = [
    "plot_predictions",
    "plot_spatial_field",
    "plot_uncertainty",
    "plot_confidence_intervals",
    "plot_training_history",
    "plot_calibration",
    "plot_residuals",
    "plot_pit_histogram",
]
