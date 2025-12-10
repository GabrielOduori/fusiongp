"""
Evaluation metrics for FusionGP.

This module provides comprehensive evaluation metrics for probabilistic
models, including both point prediction metrics and proper scoring rules
for probabilistic forecasts.

Metrics Categories
------------------
1. Point Prediction Metrics:
   - RMSE: Root Mean Squared Error
   - MAE: Mean Absolute Error
   - R²: Coefficient of Determination
   - Bias: Mean Error
   - MAPE: Mean Absolute Percentage Error

2. Probabilistic/Gaussian Metrics:
   - NLL: Negative Log-Likelihood
   - CRPS: Continuous Ranked Probability Score
   - DSS: Dawid-Sebastiani Score
   - Energy Score: Multivariate probabilistic metric
   - Calibration: Coverage at various levels
   - Sharpness: Average predictive uncertainty
   - PIT: Probability Integral Transform

References
----------
.. [1] Gneiting, T., & Raftery, A. E. (2007). Strictly proper scoring rules, 
       prediction, and estimation. JASA.
.. [2] Dawid, A. P., & Sebastiani, P. (1999). Coherent dispersion criteria 
       for optimal experimental design. Annals of Statistics.
.. [3] Hersbach, H. (2000). Decomposition of the continuous ranked probability 
       score for ensemble prediction systems. Weather and Forecasting.

Example
-------
>>> evaluator = Evaluator()
>>> metrics = evaluator.compute_all_metrics(predictions, observations)
>>> print(metrics)
{'rmse': 2.34, 'mae': 1.89, 'nll': 1.23, 'crps': 1.45, ...}
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import stats

logger = logging.getLogger(__name__)


# =============================================================================
# Point Prediction Metrics
# =============================================================================

def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Root Mean Squared Error.
    
    RMSE = sqrt(mean((y_true - y_pred)²))
    
    Parameters
    ----------
    y_true : np.ndarray
        True values.
    y_pred : np.ndarray
        Predicted values.
        
    Returns
    -------
    float
        RMSE value.
    """
    return np.sqrt(np.mean((y_true - y_pred) ** 2))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Mean Absolute Error.
    
    MAE = mean(|y_true - y_pred|)
    
    Parameters
    ----------
    y_true : np.ndarray
        True values.
    y_pred : np.ndarray
        Predicted values.
        
    Returns
    -------
    float
        MAE value.
    """
    return np.mean(np.abs(y_true - y_pred))


def r_squared(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Coefficient of Determination (R²).
    
    R² = 1 - SS_res / SS_tot
    
    Parameters
    ----------
    y_true : np.ndarray
        True values.
    y_pred : np.ndarray
        Predicted values.
        
    Returns
    -------
    float
        R² value.
    """
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    
    if ss_tot == 0:
        return 0.0
    
    return 1 - (ss_res / ss_tot)


def bias(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Mean Bias (Mean Error).
    
    Bias = mean(y_pred - y_true)
    
    Positive bias indicates over-prediction.
    
    Parameters
    ----------
    y_true : np.ndarray
        True values.
    y_pred : np.ndarray
        Predicted values.
        
    Returns
    -------
    float
        Bias value.
    """
    return np.mean(y_pred - y_true)


def mape(y_true: np.ndarray, y_pred: np.ndarray, epsilon: float = 1e-8) -> float:
    """
    Mean Absolute Percentage Error.
    
    MAPE = mean(|y_true - y_pred| / |y_true|) * 100
    
    Parameters
    ----------
    y_true : np.ndarray
        True values.
    y_pred : np.ndarray
        Predicted values.
    epsilon : float
        Small value to avoid division by zero.
        
    Returns
    -------
    float
        MAPE value (in percentage).
    """
    return np.mean(np.abs(y_true - y_pred) / (np.abs(y_true) + epsilon)) * 100


# =============================================================================
# Probabilistic Metrics
# =============================================================================

def negative_log_likelihood(
    y_true: np.ndarray,
    y_pred_mean: np.ndarray,
    y_pred_std: np.ndarray,
) -> float:
    """
    Negative Log-Likelihood for Gaussian predictions.
    
    NLL = -mean(log p(y|μ,σ))
        = mean(0.5 * log(2πσ²) + (y - μ)² / (2σ²))
    
    Lower is better.
    
    Parameters
    ----------
    y_true : np.ndarray
        True values.
    y_pred_mean : np.ndarray
        Predicted means.
    y_pred_std : np.ndarray
        Predicted standard deviations.
        
    Returns
    -------
    float
        NLL value.
        
    References
    ----------
    .. [1] Williams, C. K., & Rasmussen, C. E. (2006). 
           Gaussian processes for machine learning.
    """
    var = y_pred_std ** 2
    nll = 0.5 * np.log(2 * np.pi * var) + (y_true - y_pred_mean) ** 2 / (2 * var)
    return np.mean(nll)


def crps(
    y_true: np.ndarray,
    y_pred_mean: np.ndarray,
    y_pred_std: np.ndarray,
) -> float:
    """
    Continuous Ranked Probability Score for Gaussian predictions.
    
    CRPS measures the integrated squared difference between the predicted
    CDF and the empirical CDF. For Gaussian distributions:
    
    CRPS = σ * (z * (2Φ(z) - 1) + 2φ(z) - 1/√π)
    
    where z = (y - μ) / σ, Φ is the normal CDF, φ is the normal PDF.
    
    Lower is better.
    
    Parameters
    ----------
    y_true : np.ndarray
        True values.
    y_pred_mean : np.ndarray
        Predicted means.
    y_pred_std : np.ndarray
        Predicted standard deviations.
        
    Returns
    -------
    float
        Mean CRPS value.
        
    References
    ----------
    .. [1] Gneiting, T., & Raftery, A. E. (2007). Strictly proper scoring 
           rules, prediction, and estimation. JASA.
    .. [2] Hersbach, H. (2000). Decomposition of the continuous ranked 
           probability score. Weather and Forecasting.
    """
    # Standardize
    z = (y_true - y_pred_mean) / y_pred_std
    
    # Normal PDF and CDF at z
    phi_z = stats.norm.pdf(z)
    Phi_z = stats.norm.cdf(z)
    
    # CRPS formula for Gaussian
    crps_values = y_pred_std * (z * (2 * Phi_z - 1) + 2 * phi_z - 1 / np.sqrt(np.pi))
    
    return np.mean(crps_values)


def dawid_sebastiani_score(
    y_true: np.ndarray,
    y_pred_mean: np.ndarray,
    y_pred_std: np.ndarray,
) -> float:
    """
    Dawid-Sebastiani Score (DSS).
    
    DSS = log(σ²) + (y - μ)² / σ²
    
    This is a proper scoring rule that penalizes both bias and 
    miscalibrated uncertainty. Lower is better.
    
    Parameters
    ----------
    y_true : np.ndarray
        True values.
    y_pred_mean : np.ndarray
        Predicted means.
    y_pred_std : np.ndarray
        Predicted standard deviations.
        
    Returns
    -------
    float
        Mean DSS value.
        
    References
    ----------
    .. [1] Dawid, A. P., & Sebastiani, P. (1999). Coherent dispersion 
           criteria for optimal experimental design.
    """
    var = y_pred_std ** 2
    dss = np.log(var) + (y_true - y_pred_mean) ** 2 / var
    return np.mean(dss)


def energy_score(
    y_true: np.ndarray,
    samples: np.ndarray,
) -> float:
    """
    Energy Score computed from posterior samples.
    
    ES = E[||Y - y||] - 0.5 * E[||Y - Y'||]
    
    where Y, Y' are independent samples from the predictive distribution.
    Lower is better.
    
    Parameters
    ----------
    y_true : np.ndarray
        True values, shape (N,).
    samples : np.ndarray
        Posterior samples, shape (n_samples, N).
        
    Returns
    -------
    float
        Mean Energy Score.
        
    References
    ----------
    .. [1] Gneiting, T., & Raftery, A. E. (2007). Strictly proper 
           scoring rules, prediction, and estimation.
    """
    n_samples, n_points = samples.shape
    
    # E[||Y - y||]
    term1 = np.mean(np.abs(samples - y_true[np.newaxis, :]))
    
    # E[||Y - Y'||] - approximate with sample pairs
    # Use half the samples for Y and half for Y'
    half = n_samples // 2
    term2 = np.mean(np.abs(samples[:half] - samples[half:2*half]))
    
    return term1 - 0.5 * term2


def calibration_error(
    y_true: np.ndarray,
    y_pred_mean: np.ndarray,
    y_pred_std: np.ndarray,
    confidence_levels: List[float] = None,
) -> Dict[float, float]:
    """
    Calibration Error at various confidence levels.
    
    For a well-calibrated model, the fraction of observations falling
    within the p% prediction interval should be approximately p%.
    
    Calibration Error = |observed_coverage - expected_coverage|
    
    Parameters
    ----------
    y_true : np.ndarray
        True values.
    y_pred_mean : np.ndarray
        Predicted means.
    y_pred_std : np.ndarray
        Predicted standard deviations.
    confidence_levels : List[float]
        Confidence levels to evaluate.
        
    Returns
    -------
    Dict[float, float]
        Calibration error at each level.
    """
    if confidence_levels is None:
        confidence_levels = [0.5, 0.8, 0.9, 0.95]
    
    errors = {}
    
    for level in confidence_levels:
        z = stats.norm.ppf((1 + level) / 2)
        lower = y_pred_mean - z * y_pred_std
        upper = y_pred_mean + z * y_pred_std
        
        # Observed coverage
        in_interval = (y_true >= lower) & (y_true <= upper)
        observed_coverage = np.mean(in_interval)
        
        # Calibration error
        errors[level] = abs(observed_coverage - level)
    
    return errors


def coverage(
    y_true: np.ndarray,
    y_pred_mean: np.ndarray,
    y_pred_std: np.ndarray,
    confidence_levels: List[float] = None,
) -> Dict[float, float]:
    """
    Empirical coverage at various confidence levels.
    
    Coverage = fraction of observations within prediction interval.
    
    Parameters
    ----------
    y_true : np.ndarray
        True values.
    y_pred_mean : np.ndarray
        Predicted means.
    y_pred_std : np.ndarray
        Predicted standard deviations.
    confidence_levels : List[float]
        Confidence levels to evaluate.
        
    Returns
    -------
    Dict[float, float]
        Coverage at each level.
    """
    if confidence_levels is None:
        confidence_levels = [0.5, 0.8, 0.9, 0.95]
    
    coverages = {}
    
    for level in confidence_levels:
        z = stats.norm.ppf((1 + level) / 2)
        lower = y_pred_mean - z * y_pred_std
        upper = y_pred_mean + z * y_pred_std
        
        in_interval = (y_true >= lower) & (y_true <= upper)
        coverages[level] = np.mean(in_interval)
    
    return coverages


def sharpness(y_pred_std: np.ndarray) -> float:
    """
    Sharpness (average predictive uncertainty).
    
    Sharpness = mean(σ)
    
    Lower sharpness with good calibration is better.
    
    Parameters
    ----------
    y_pred_std : np.ndarray
        Predicted standard deviations.
        
    Returns
    -------
    float
        Mean predictive std.
    """
    return np.mean(y_pred_std)


def pit_values(
    y_true: np.ndarray,
    y_pred_mean: np.ndarray,
    y_pred_std: np.ndarray,
) -> np.ndarray:
    """
    Probability Integral Transform (PIT) values.
    
    PIT = F(y) where F is the predictive CDF.
    
    For a well-calibrated model, PIT values should be uniformly distributed.
    
    Parameters
    ----------
    y_true : np.ndarray
        True values.
    y_pred_mean : np.ndarray
        Predicted means.
    y_pred_std : np.ndarray
        Predicted standard deviations.
        
    Returns
    -------
    np.ndarray
        PIT values in [0, 1].
    """
    z = (y_true - y_pred_mean) / y_pred_std
    return stats.norm.cdf(z)


def pit_deviation(
    y_true: np.ndarray,
    y_pred_mean: np.ndarray,
    y_pred_std: np.ndarray,
) -> float:
    """
    PIT Deviation from uniformity.
    
    Measures how much the PIT histogram deviates from uniform distribution
    using the Kolmogorov-Smirnov statistic.
    
    Parameters
    ----------
    y_true : np.ndarray
        True values.
    y_pred_mean : np.ndarray
        Predicted means.
    y_pred_std : np.ndarray
        Predicted standard deviations.
        
    Returns
    -------
    float
        KS statistic (0 = perfect uniformity).
    """
    pit = pit_values(y_true, y_pred_mean, y_pred_std)
    ks_stat, _ = stats.kstest(pit, 'uniform')
    return ks_stat


def interval_score(
    y_true: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    alpha: float = 0.05,
) -> float:
    """
    Interval Score for prediction intervals.
    
    IS = (upper - lower) + (2/α) * (lower - y) * I(y < lower) 
                        + (2/α) * (y - upper) * I(y > upper)
    
    Penalizes both width and coverage failures.
    
    Parameters
    ----------
    y_true : np.ndarray
        True values.
    lower : np.ndarray
        Lower prediction bounds.
    upper : np.ndarray
        Upper prediction bounds.
    alpha : float
        Significance level (1 - confidence).
        
    Returns
    -------
    float
        Mean interval score.
        
    References
    ----------
    .. [1] Gneiting, T., & Raftery, A. E. (2007). Strictly proper 
           scoring rules.
    """
    width = upper - lower
    
    # Penalty for observations below lower bound
    below = y_true < lower
    penalty_below = (2 / alpha) * (lower - y_true) * below
    
    # Penalty for observations above upper bound
    above = y_true > upper
    penalty_above = (2 / alpha) * (y_true - upper) * above
    
    return np.mean(width + penalty_below + penalty_above)


# =============================================================================
# Evaluator Class
# =============================================================================

@dataclass
class MetricsResult:
    """
    Container for evaluation metrics.
    
    Attributes
    ----------
    point_metrics : Dict[str, float]
        Point prediction metrics.
    probabilistic_metrics : Dict[str, float]
        Probabilistic metrics.
    calibration_metrics : Dict[str, Dict[float, float]]
        Calibration metrics at various levels.
    per_source_metrics : Dict[str, Dict[str, float]]
        Metrics computed per source.
    """
    point_metrics: Dict[str, float]
    probabilistic_metrics: Dict[str, float]
    calibration_metrics: Dict[str, Dict[float, float]]
    per_source_metrics: Dict[str, Dict[str, float]]
    
    def to_dict(self) -> Dict[str, float]:
        """Flatten to single dictionary."""
        result = {}
        result.update(self.point_metrics)
        result.update(self.probabilistic_metrics)
        
        for metric_name, levels in self.calibration_metrics.items():
            for level, value in levels.items():
                result[f'{metric_name}_{int(level*100)}'] = value
        
        return result
    
    def summary(self) -> str:
        """Generate summary string."""
        lines = [
            "Evaluation Metrics Summary",
            "=" * 50,
            "",
            "Point Prediction Metrics:",
        ]
        for name, value in self.point_metrics.items():
            lines.append(f"  {name}: {value:.4f}")
        
        lines.extend(["", "Probabilistic Metrics:"])
        for name, value in self.probabilistic_metrics.items():
            lines.append(f"  {name}: {value:.4f}")
        
        lines.extend(["", "Calibration (Coverage @ Level):"])
        if 'coverage' in self.calibration_metrics:
            for level, value in self.calibration_metrics['coverage'].items():
                lines.append(f"  {int(level*100)}%: {value:.1%} (expected: {level:.1%})")
        
        return "\n".join(lines)


class Evaluator:
    """
    Comprehensive evaluator for FusionGP predictions.
    
    Computes all standard metrics for probabilistic predictions including
    point prediction metrics, proper scoring rules, and calibration diagnostics.
    
    Parameters
    ----------
    confidence_levels : List[float], optional
        Confidence levels for calibration evaluation.
        
    Attributes
    ----------
    confidence_levels : List[float]
        Confidence levels.
        
    Example
    -------
    >>> evaluator = Evaluator()
    >>> 
    >>> # Compute metrics
    >>> metrics = evaluator.evaluate(
    ...     y_true=observations,
    ...     y_pred_mean=predictions.mean,
    ...     y_pred_std=predictions.std
    ... )
    >>> 
    >>> print(metrics.summary())
    """
    
    def __init__(
        self,
        confidence_levels: List[float] = None,
    ):
        """
        Initialize the Evaluator.
        
        Parameters
        ----------
        confidence_levels : List[float]
            Confidence levels for calibration.
        """
        self.confidence_levels = confidence_levels or [0.5, 0.8, 0.9, 0.95]
        
        logger.info(f"Evaluator initialized with confidence_levels={self.confidence_levels}")
    
    def evaluate(
        self,
        y_true: np.ndarray,
        y_pred_mean: np.ndarray,
        y_pred_std: np.ndarray,
        samples: Optional[np.ndarray] = None,
    ) -> MetricsResult:
        """
        Compute all evaluation metrics.
        
        Parameters
        ----------
        y_true : np.ndarray
            True values.
        y_pred_mean : np.ndarray
            Predicted means.
        y_pred_std : np.ndarray
            Predicted standard deviations.
        samples : np.ndarray, optional
            Posterior samples for energy score.
            
        Returns
        -------
        MetricsResult
            All computed metrics.
        """
        # Remove NaN values
        mask = ~(np.isnan(y_true) | np.isnan(y_pred_mean) | np.isnan(y_pred_std))
        y_true = y_true[mask]
        y_pred_mean = y_pred_mean[mask]
        y_pred_std = y_pred_std[mask]
        
        if len(y_true) == 0:
            logger.warning("No valid observations for evaluation")
            return MetricsResult({}, {}, {}, {})
        
        # Point metrics
        point_metrics = {
            'rmse': rmse(y_true, y_pred_mean),
            'mae': mae(y_true, y_pred_mean),
            'r2': r_squared(y_true, y_pred_mean),
            'bias': bias(y_true, y_pred_mean),
            'mape': mape(y_true, y_pred_mean),
        }
        
        # Probabilistic metrics
        prob_metrics = {
            'nll': negative_log_likelihood(y_true, y_pred_mean, y_pred_std),
            'crps': crps(y_true, y_pred_mean, y_pred_std),
            'dss': dawid_sebastiani_score(y_true, y_pred_mean, y_pred_std),
            'sharpness': sharpness(y_pred_std),
            'pit_deviation': pit_deviation(y_true, y_pred_mean, y_pred_std),
        }
        
        # Energy score if samples provided
        if samples is not None:
            samples = samples[:, mask]
            prob_metrics['energy_score'] = energy_score(y_true, samples)
        
        # Interval score at 90% level
        z_90 = stats.norm.ppf(0.95)
        lower_90 = y_pred_mean - z_90 * y_pred_std
        upper_90 = y_pred_mean + z_90 * y_pred_std
        prob_metrics['interval_score_90'] = interval_score(
            y_true, lower_90, upper_90, alpha=0.1
        )
        
        # Calibration metrics
        calib_metrics = {
            'coverage': coverage(
                y_true, y_pred_mean, y_pred_std, self.confidence_levels
            ),
            'calibration_error': calibration_error(
                y_true, y_pred_mean, y_pred_std, self.confidence_levels
            ),
        }
        
        return MetricsResult(
            point_metrics=point_metrics,
            probabilistic_metrics=prob_metrics,
            calibration_metrics=calib_metrics,
            per_source_metrics={},
        )
    
    def evaluate_per_source(
        self,
        y_true: Dict[str, np.ndarray],
        y_pred_mean: np.ndarray,
        y_pred_std: np.ndarray,
        source_masks: Dict[str, np.ndarray],
    ) -> MetricsResult:
        """
        Compute metrics separately for each source.
        
        Parameters
        ----------
        y_true : Dict[str, np.ndarray]
            True values per source.
        y_pred_mean : np.ndarray
            Predicted means.
        y_pred_std : np.ndarray
            Predicted stds.
        source_masks : Dict[str, np.ndarray]
            Valid observation masks per source.
            
        Returns
        -------
        MetricsResult
            Metrics including per-source breakdown.
        """
        # Overall metrics (using EPA as reference if available)
        reference_source = 'epa' if 'epa' in y_true else list(y_true.keys())[0]
        ref_mask = source_masks[reference_source]
        
        overall = self.evaluate(
            y_true[reference_source][ref_mask],
            y_pred_mean[ref_mask],
            y_pred_std[ref_mask],
        )
        
        # Per-source metrics
        per_source = {}
        for source, obs in y_true.items():
            mask = source_masks[source]
            if mask.sum() > 0:
                source_metrics = self.evaluate(
                    obs[mask],
                    y_pred_mean[mask],
                    y_pred_std[mask],
                )
                per_source[source] = source_metrics.to_dict()
        
        overall.per_source_metrics = per_source
        
        return overall


def compute_metrics(
    y_true: np.ndarray,
    y_pred_mean: np.ndarray,
    y_pred_std: np.ndarray,
    confidence_levels: List[float] = None,
) -> Dict[str, float]:
    """
    Convenience function to compute all metrics.
    
    Parameters
    ----------
    y_true : np.ndarray
        True values.
    y_pred_mean : np.ndarray
        Predicted means.
    y_pred_std : np.ndarray
        Predicted standard deviations.
    confidence_levels : List[float]
        Confidence levels.
        
    Returns
    -------
    Dict[str, float]
        Dictionary of all metrics.
    """
    evaluator = Evaluator(confidence_levels=confidence_levels)
    result = evaluator.evaluate(y_true, y_pred_mean, y_pred_std)
    return result.to_dict()
