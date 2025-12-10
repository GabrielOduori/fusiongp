"""
Evaluation utilities for FusionGP.

This module provides comprehensive evaluation metrics for probabilistic models:
- Point prediction metrics (RMSE, MAE, R², Bias)
- Probabilistic metrics (NLL, CRPS, DSS, Energy Score)
- Calibration diagnostics (coverage, sharpness, PIT)

Classes
-------
Evaluator
    Comprehensive model evaluator.

Functions
---------
compute_metrics
    Compute all metrics for predictions.
"""

from src.evaluation.metrics import (
    Evaluator,
    compute_metrics,
    rmse,
    mae,
    r_squared,
    bias,
    negative_log_likelihood,
    crps,
    dawid_sebastiani_score,
    calibration_error,
    sharpness,
)

__all__ = [
    "Evaluator",
    "compute_metrics",
    "rmse",
    "mae",
    "r_squared",
    "bias",
    "negative_log_likelihood",
    "crps",
    "dawid_sebastiani_score",
    "calibration_error",
    "sharpness",
]
