"""
Inference utilities for FusionGP.

This module provides prediction and uncertainty quantification utilities.

Classes
-------
Predictor
    Make gridded predictions with uncertainty estimates.
Predictions
    Container for prediction results.
GridPredictions
    Container for gridded prediction results with visualization.
"""

from src.inference.predictor import Predictor, Predictions, GridPredictions
from src.inference.kalman_smoother import KalmanSmoother, SmoothedPredictions
from src.inference.gp_kalman_filter import GPKalmanFilter, SequentialPredictions
from src.inference.st_svgp_filter import STSVGPFilter, FilterResult, SmoothResult

__all__ = [
    "Predictor", "Predictions", "GridPredictions",
    "KalmanSmoother", "SmoothedPredictions",
    "GPKalmanFilter", "SequentialPredictions",
    "STSVGPFilter", "FilterResult", "SmoothResult",
]
