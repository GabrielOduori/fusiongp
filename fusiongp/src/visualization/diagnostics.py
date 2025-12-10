"""
Diagnostic visualization for FusionGP.

This module provides plotting functions for model diagnostics including
training curves, calibration plots, residual analysis, and PIT histograms.

Example
-------
>>> from src.visualization import plot_training_history, plot_calibration
>>> plot_training_history(history)
>>> plot_calibration(y_true, predictions)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

logger = logging.getLogger(__name__)


def plot_training_history(
    history: Dict[str, List[float]],
    title: str = "Training History",
    figsize: Tuple[int, int] = (14, 5),
    save_path: Optional[Union[str, Path]] = None,
) -> plt.Figure:
    """
    Plot training curves.
    
    Parameters
    ----------
    history : Dict[str, List[float]]
        Training history with keys like 'train_loss', 'val_loss', 'learning_rate'.
    title : str
        Overall title.
    figsize : Tuple[int, int]
        Figure size.
    save_path : str or Path, optional
        Save path.
        
    Returns
    -------
    plt.Figure
        The figure object.
    """
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    epochs = range(1, len(history.get('train_loss', [])) + 1)
    
    # Plot 1: Loss curves
    ax1 = axes[0]
    if 'train_loss' in history:
        ax1.plot(epochs, history['train_loss'], 'b-', label='Train')
    if 'val_loss' in history:
        # Validation might be at different intervals
        val_epochs = range(1, len(history['val_loss']) + 1)
        # Scale to match train epochs
        if len(history['val_loss']) != len(epochs):
            scale = len(epochs) / len(history['val_loss'])
            val_epochs = [int(i * scale) for i in range(1, len(history['val_loss']) + 1)]
        ax1.plot(val_epochs, history['val_loss'], 'r-', label='Validation')
    
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss (neg ELBO)')
    ax1.set_title('Training & Validation Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Learning rate
    ax2 = axes[1]
    if 'learning_rate' in history:
        ax2.plot(epochs, history['learning_rate'], 'g-')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Learning Rate')
        ax2.set_title('Learning Rate Schedule')
        ax2.set_yscale('log')
        ax2.grid(True, alpha=0.3)
    else:
        ax2.text(0.5, 0.5, 'No LR data', ha='center', va='center', transform=ax2.transAxes)
        ax2.set_title('Learning Rate Schedule')
    
    # Plot 3: Epoch time
    ax3 = axes[2]
    if 'epoch_time' in history:
        ax3.bar(epochs, history['epoch_time'], alpha=0.7)
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Time (seconds)')
        ax3.set_title('Epoch Duration')
        ax3.grid(True, alpha=0.3, axis='y')
    else:
        ax3.text(0.5, 0.5, 'No timing data', ha='center', va='center', transform=ax3.transAxes)
        ax3.set_title('Epoch Duration')
    
    fig.suptitle(title, fontsize=14, y=1.02)
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved figure to {save_path}")
    
    return fig


def plot_calibration(
    y_true: np.ndarray,
    y_pred_mean: np.ndarray,
    y_pred_std: np.ndarray,
    confidence_levels: List[float] = None,
    title: str = "Calibration Plot",
    figsize: Tuple[int, int] = (8, 8),
    save_path: Optional[Union[str, Path]] = None,
) -> plt.Figure:
    """
    Create a calibration plot comparing expected vs observed coverage.
    
    For a well-calibrated model, points should lie on the diagonal.
    
    Parameters
    ----------
    y_true : np.ndarray
        True values.
    y_pred_mean : np.ndarray
        Predicted means.
    y_pred_std : np.ndarray
        Predicted standard deviations.
    confidence_levels : List[float], optional
        Confidence levels to evaluate.
    title : str
        Plot title.
    figsize : Tuple[int, int]
        Figure size.
    save_path : str or Path, optional
        Save path.
        
    Returns
    -------
    plt.Figure
        The figure object.
    """
    if confidence_levels is None:
        confidence_levels = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
    
    fig, ax = plt.subplots(figsize=figsize)
    
    observed_coverages = []
    
    for level in confidence_levels:
        z = stats.norm.ppf((1 + level) / 2)
        lower = y_pred_mean - z * y_pred_std
        upper = y_pred_mean + z * y_pred_std
        
        in_interval = (y_true >= lower) & (y_true <= upper)
        observed_coverages.append(np.mean(in_interval))
    
    # Plot calibration curve
    ax.plot(confidence_levels, observed_coverages, 'bo-', markersize=8, label='Observed')
    ax.plot([0, 1], [0, 1], 'k--', label='Perfect calibration')
    
    # Confidence band for sampling variability
    n = len(y_true)
    for level, obs in zip(confidence_levels, observed_coverages):
        se = np.sqrt(level * (1 - level) / n)
        ax.fill_between(
            [level - 0.02, level + 0.02],
            [obs - 1.96*se, obs - 1.96*se],
            [obs + 1.96*se, obs + 1.96*se],
            alpha=0.2, color='blue'
        )
    
    ax.set_xlabel('Expected Coverage')
    ax.set_ylabel('Observed Coverage')
    ax.set_title(title)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved figure to {save_path}")
    
    return fig


def plot_residuals(
    y_true: np.ndarray,
    y_pred_mean: np.ndarray,
    y_pred_std: Optional[np.ndarray] = None,
    title: str = "Residual Analysis",
    figsize: Tuple[int, int] = (14, 10),
    save_path: Optional[Union[str, Path]] = None,
) -> plt.Figure:
    """
    Comprehensive residual analysis plots.
    
    Parameters
    ----------
    y_true : np.ndarray
        True values.
    y_pred_mean : np.ndarray
        Predicted means.
    y_pred_std : np.ndarray, optional
        Predicted standard deviations (for standardized residuals).
    title : str
        Overall title.
    figsize : Tuple[int, int]
        Figure size.
    save_path : str or Path, optional
        Save path.
        
    Returns
    -------
    plt.Figure
        The figure object.
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    
    residuals = y_true - y_pred_mean
    
    if y_pred_std is not None:
        std_residuals = residuals / y_pred_std
    else:
        std_residuals = residuals / np.std(residuals)
    
    # Plot 1: Residuals vs Predicted
    ax1 = axes[0, 0]
    ax1.scatter(y_pred_mean, residuals, alpha=0.3, s=10)
    ax1.axhline(y=0, color='r', linestyle='--')
    ax1.set_xlabel('Predicted')
    ax1.set_ylabel('Residual')
    ax1.set_title('Residuals vs Predicted')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Predicted vs Actual
    ax2 = axes[0, 1]
    ax2.scatter(y_true, y_pred_mean, alpha=0.3, s=10)
    lims = [
        min(y_true.min(), y_pred_mean.min()),
        max(y_true.max(), y_pred_mean.max())
    ]
    ax2.plot(lims, lims, 'r--', label='Perfect')
    ax2.set_xlabel('Actual')
    ax2.set_ylabel('Predicted')
    ax2.set_title('Predicted vs Actual')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Residual histogram
    ax3 = axes[1, 0]
    ax3.hist(std_residuals, bins=50, density=True, alpha=0.7, edgecolor='black')
    
    # Overlay normal distribution
    x = np.linspace(std_residuals.min(), std_residuals.max(), 100)
    ax3.plot(x, stats.norm.pdf(x), 'r-', linewidth=2, label='N(0,1)')
    
    ax3.set_xlabel('Standardized Residual')
    ax3.set_ylabel('Density')
    ax3.set_title('Residual Distribution')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Q-Q plot
    ax4 = axes[1, 1]
    stats.probplot(std_residuals, dist="norm", plot=ax4)
    ax4.set_title('Q-Q Plot (Standardized Residuals)')
    ax4.grid(True, alpha=0.3)
    
    fig.suptitle(title, fontsize=14, y=1.02)
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved figure to {save_path}")
    
    return fig


def plot_pit_histogram(
    y_true: np.ndarray,
    y_pred_mean: np.ndarray,
    y_pred_std: np.ndarray,
    n_bins: int = 20,
    title: str = "PIT Histogram",
    figsize: Tuple[int, int] = (8, 6),
    save_path: Optional[Union[str, Path]] = None,
) -> plt.Figure:
    """
    Plot Probability Integral Transform histogram.
    
    For a well-calibrated model, PIT values should be uniformly distributed,
    resulting in a flat histogram.
    
    Parameters
    ----------
    y_true : np.ndarray
        True values.
    y_pred_mean : np.ndarray
        Predicted means.
    y_pred_std : np.ndarray
        Predicted standard deviations.
    n_bins : int
        Number of histogram bins.
    title : str
        Plot title.
    figsize : Tuple[int, int]
        Figure size.
    save_path : str or Path, optional
        Save path.
        
    Returns
    -------
    plt.Figure
        The figure object.
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Calculate PIT values
    z = (y_true - y_pred_mean) / y_pred_std
    pit = stats.norm.cdf(z)
    
    # Plot histogram
    counts, bins, patches = ax.hist(
        pit, bins=n_bins, density=True,
        alpha=0.7, edgecolor='black'
    )
    
    # Reference line for uniform distribution
    ax.axhline(y=1, color='r', linestyle='--', linewidth=2, label='Uniform')
    
    # Add confidence bands
    n = len(pit)
    expected = 1.0
    se = np.sqrt(expected * (1 - 1/n_bins) / (n / n_bins))
    ax.fill_between(
        [0, 1], [expected - 1.96*se]*2, [expected + 1.96*se]*2,
        alpha=0.2, color='red', label='95% CI'
    )
    
    ax.set_xlabel('PIT Value')
    ax.set_ylabel('Density')
    ax.set_title(title)
    ax.set_xlim(0, 1)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Add KS test result
    ks_stat, p_value = stats.kstest(pit, 'uniform')
    ax.text(
        0.02, 0.98,
        f'KS stat: {ks_stat:.3f}\np-value: {p_value:.3f}',
        transform=ax.transAxes,
        verticalalignment='top',
        fontsize=10,
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
    )
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved figure to {save_path}")
    
    return fig


def plot_metrics_summary(
    metrics: Dict[str, float],
    title: str = "Evaluation Metrics Summary",
    figsize: Tuple[int, int] = (12, 6),
    save_path: Optional[Union[str, Path]] = None,
) -> plt.Figure:
    """
    Create a bar chart summary of evaluation metrics.
    
    Parameters
    ----------
    metrics : Dict[str, float]
        Dictionary of metric names and values.
    title : str
        Plot title.
    figsize : Tuple[int, int]
        Figure size.
    save_path : str or Path, optional
        Save path.
        
    Returns
    -------
    plt.Figure
        The figure object.
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Separate metrics by type
    point_metrics = {k: v for k, v in metrics.items() 
                    if k in ['rmse', 'mae', 'r2', 'bias', 'mape']}
    prob_metrics = {k: v for k, v in metrics.items() 
                   if k in ['nll', 'crps', 'dss', 'sharpness', 'interval_score_90']}
    
    # Plot 1: Point metrics
    ax1 = axes[0]
    if point_metrics:
        bars1 = ax1.bar(point_metrics.keys(), point_metrics.values(), color='steelblue')
        ax1.set_ylabel('Value')
        ax1.set_title('Point Prediction Metrics')
        ax1.tick_params(axis='x', rotation=45)
        
        # Add value labels
        for bar, val in zip(bars1, point_metrics.values()):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                    f'{val:.3f}', ha='center', va='bottom', fontsize=9)
    
    # Plot 2: Probabilistic metrics
    ax2 = axes[1]
    if prob_metrics:
        bars2 = ax2.bar(prob_metrics.keys(), prob_metrics.values(), color='coral')
        ax2.set_ylabel('Value')
        ax2.set_title('Probabilistic Metrics')
        ax2.tick_params(axis='x', rotation=45)
        
        for bar, val in zip(bars2, prob_metrics.values()):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                    f'{val:.3f}', ha='center', va='bottom', fontsize=9)
    
    fig.suptitle(title, fontsize=14, y=1.02)
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved figure to {save_path}")
    
    return fig
