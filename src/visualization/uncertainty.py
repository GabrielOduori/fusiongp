"""
Uncertainty visualization for FusionGP.

This module provides functions for visualizing prediction uncertainty,
including spatial uncertainty maps and confidence interval plots.

Example
-------
>>> from src.visualization import plot_uncertainty
>>> fig = plot_uncertainty(predictions)
>>> fig.savefig("uncertainty.png")
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np

logger = logging.getLogger(__name__)


def plot_uncertainty(
    coords: np.ndarray,
    std: np.ndarray,
    title: str = "Prediction Uncertainty",
    cmap: str = "YlOrRd",
    figsize: Tuple[int, int] = (10, 8),
    save_path: Optional[Union[str, Path]] = None,
    ax: Optional[plt.Axes] = None,
) -> plt.Figure:
    """
    Create a spatial map of prediction uncertainty.
    
    Parameters
    ----------
    coords : np.ndarray
        Coordinates, shape (N, 2) with columns [lat, lon].
    std : np.ndarray
        Standard deviation values, shape (N,).
    title : str
        Plot title.
    cmap : str
        Colormap name (sequential recommended).
    figsize : Tuple[int, int]
        Figure size.
    save_path : str or Path, optional
        Path to save figure.
    ax : plt.Axes, optional
        Existing axes.
        
    Returns
    -------
    plt.Figure
        The figure object.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()
    
    scatter = ax.scatter(
        coords[:, 1],
        coords[:, 0],
        c=std,
        cmap=cmap,
        s=20,
        edgecolors='none',
    )
    
    cbar = plt.colorbar(scatter, ax=ax, label='Std Dev (ppb)')
    
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title(title)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved figure to {save_path}")
    
    return fig


def plot_confidence_intervals(
    x: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    y_true: Optional[np.ndarray] = None,
    confidence_levels: list = [0.5, 0.9, 0.95],
    xlabel: str = "Index",
    ylabel: str = "NO₂ (ppb)",
    title: str = "Predictions with Confidence Intervals",
    figsize: Tuple[int, int] = (12, 6),
    save_path: Optional[Union[str, Path]] = None,
) -> plt.Figure:
    """
    Plot predictions with confidence intervals.
    
    Parameters
    ----------
    x : np.ndarray
        X-axis values (e.g., time or index).
    mean : np.ndarray
        Predicted means.
    std : np.ndarray
        Predicted standard deviations.
    y_true : np.ndarray, optional
        True values for comparison.
    confidence_levels : list
        Confidence levels to show.
    xlabel : str
        X-axis label.
    ylabel : str
        Y-axis label.
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
    from scipy import stats
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Sort by x for proper line plotting
    sort_idx = np.argsort(x)
    x_sorted = x[sort_idx]
    mean_sorted = mean[sort_idx]
    std_sorted = std[sort_idx]
    
    # Plot confidence intervals (from widest to narrowest)
    colors = plt.cm.Blues(np.linspace(0.3, 0.7, len(confidence_levels)))
    
    for level, color in zip(sorted(confidence_levels, reverse=True), colors):
        z = stats.norm.ppf((1 + level) / 2)
        lower = mean_sorted - z * std_sorted
        upper = mean_sorted + z * std_sorted
        
        ax.fill_between(
            x_sorted, lower, upper,
            alpha=0.3,
            color=color,
            label=f'{int(level*100)}% CI'
        )
    
    # Plot mean
    ax.plot(x_sorted, mean_sorted, 'b-', linewidth=2, label='Predicted Mean')
    
    # Plot true values if provided
    if y_true is not None:
        y_sorted = y_true[sort_idx]
        ax.scatter(
            x_sorted, y_sorted,
            c='red', s=20, alpha=0.7,
            label='Observations', zorder=5
        )
    
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved figure to {save_path}")
    
    return fig


def plot_uncertainty_vs_error(
    y_true: np.ndarray,
    y_pred_mean: np.ndarray,
    y_pred_std: np.ndarray,
    n_bins: int = 10,
    title: str = "Uncertainty vs. Absolute Error",
    figsize: Tuple[int, int] = (10, 8),
    save_path: Optional[Union[str, Path]] = None,
) -> plt.Figure:
    """
    Plot relationship between predicted uncertainty and actual error.
    
    For a well-calibrated model, higher uncertainty should correspond
    to higher errors on average.
    
    Parameters
    ----------
    y_true : np.ndarray
        True values.
    y_pred_mean : np.ndarray
        Predicted means.
    y_pred_std : np.ndarray
        Predicted standard deviations.
    n_bins : int
        Number of uncertainty bins.
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
    
    # Calculate absolute errors
    abs_error = np.abs(y_true - y_pred_mean)
    
    # Plot 1: Scatter plot
    ax1 = axes[0]
    ax1.scatter(y_pred_std, abs_error, alpha=0.3, s=10)
    ax1.set_xlabel('Predicted Std Dev')
    ax1.set_ylabel('Absolute Error')
    ax1.set_title('Uncertainty vs. Error (Scatter)')
    
    # Add diagonal reference line
    max_val = max(y_pred_std.max(), abs_error.max())
    ax1.plot([0, max_val], [0, max_val], 'r--', label='y=x')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Binned average
    ax2 = axes[1]
    
    # Bin by uncertainty
    bins = np.quantile(y_pred_std, np.linspace(0, 1, n_bins + 1))
    bin_indices = np.digitize(y_pred_std, bins[:-1])
    
    bin_centers = []
    bin_mean_errors = []
    bin_std_errors = []
    
    for i in range(1, n_bins + 1):
        mask = bin_indices == i
        if mask.sum() > 0:
            bin_centers.append(y_pred_std[mask].mean())
            bin_mean_errors.append(abs_error[mask].mean())
            bin_std_errors.append(abs_error[mask].std() / np.sqrt(mask.sum()))
    
    ax2.errorbar(
        bin_centers, bin_mean_errors, yerr=bin_std_errors,
        fmt='o-', capsize=3, capthick=1
    )
    ax2.set_xlabel('Mean Predicted Std Dev (binned)')
    ax2.set_ylabel('Mean Absolute Error')
    ax2.set_title('Uncertainty vs. Error (Binned)')
    
    # Reference line
    max_val = max(max(bin_centers), max(bin_mean_errors))
    ax2.plot([0, max_val], [0, max_val], 'r--', label='y=x')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    fig.suptitle(title, fontsize=14)
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved figure to {save_path}")
    
    return fig


def plot_coefficient_of_variation(
    coords: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    title: str = "Coefficient of Variation",
    figsize: Tuple[int, int] = (10, 8),
    save_path: Optional[Union[str, Path]] = None,
) -> plt.Figure:
    """
    Plot coefficient of variation (std/mean) spatially.
    
    CV indicates relative uncertainty - useful when absolute values vary.
    
    Parameters
    ----------
    coords : np.ndarray
        Coordinates, shape (N, 2).
    mean : np.ndarray
        Predicted means.
    std : np.ndarray
        Predicted standard deviations.
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
    
    # Calculate CV (avoid division by zero)
    cv = np.divide(std, np.abs(mean), where=np.abs(mean) > 1e-8)
    cv = np.clip(cv, 0, 2)  # Clip extreme values
    
    scatter = ax.scatter(
        coords[:, 1], coords[:, 0],
        c=cv, cmap='RdYlGn_r',
        s=20, edgecolors='none'
    )
    
    cbar = plt.colorbar(scatter, ax=ax, label='CV (std/|mean|)')
    
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title(title)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved figure to {save_path}")
    
    return fig
