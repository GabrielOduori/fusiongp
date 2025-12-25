"""
Spatial map visualization for FusionGP.

This module provides functions for creating spatial maps of predictions
and observations, including support for multiple time slices.

Example
-------
>>> from src.visualization import plot_predictions
>>> fig = plot_predictions(predictions, title="NO₂ Predictions")
>>> fig.savefig("predictions.png")
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

logger = logging.getLogger(__name__)


def plot_predictions(
    coords: np.ndarray,
    values: np.ndarray,
    title: str = "NO₂ Predictions",
    cmap: str = "RdYlBu_r",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    figsize: Tuple[int, int] = (10, 8),
    colorbar_label: str = "NO₂ (ppb)",
    marker_size: float = 20,
    save_path: Optional[Union[str, Path]] = None,
    ax: Optional[plt.Axes] = None,
) -> plt.Figure:
    """
    Create a spatial map of predictions.
    
    Parameters
    ----------
    coords : np.ndarray
        Coordinates, shape (N, 2) with columns [lat, lon].
    values : np.ndarray
        Values to plot, shape (N,).
    title : str
        Plot title.
    cmap : str
        Colormap name.
    vmin, vmax : float, optional
        Color scale limits.
    figsize : Tuple[int, int]
        Figure size.
    colorbar_label : str
        Label for colorbar.
    marker_size : float
        Size of scatter markers.
    save_path : str or Path, optional
        Path to save figure.
    ax : plt.Axes, optional
        Existing axes to plot on.
        
    Returns
    -------
    plt.Figure
        The figure object.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()
    
    # Handle vmin/vmax
    if vmin is None:
        vmin = np.nanpercentile(values, 2)
    if vmax is None:
        vmax = np.nanpercentile(values, 98)
    
    # Create scatter plot
    scatter = ax.scatter(
        coords[:, 1],  # longitude (x-axis)
        coords[:, 0],  # latitude (y-axis)
        c=values,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        s=marker_size,
        edgecolors='none',
    )
    
    # Colorbar
    cbar = plt.colorbar(scatter, ax=ax, label=colorbar_label)
    
    # Labels
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title(title)
    ax.set_aspect('equal')
    
    # Grid
    ax.grid(True, alpha=0.3)
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved figure to {save_path}")
    
    return fig


def plot_spatial_field(
    coords: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    observations: Optional[np.ndarray] = None,
    obs_coords: Optional[np.ndarray] = None,
    title: str = "NO₂ Field",
    figsize: Tuple[int, int] = (16, 6),
    save_path: Optional[Union[str, Path]] = None,
) -> plt.Figure:
    """
    Create a combined plot showing mean, uncertainty, and observations.
    
    Parameters
    ----------
    coords : np.ndarray
        Prediction coordinates, shape (N, 2).
    mean : np.ndarray
        Predicted mean, shape (N,).
    std : np.ndarray
        Predicted standard deviation, shape (N,).
    observations : np.ndarray, optional
        Observation values for comparison.
    obs_coords : np.ndarray, optional
        Observation coordinates.
    title : str
        Overall title.
    figsize : Tuple[int, int]
        Figure size.
    save_path : str or Path, optional
        Path to save figure.
        
    Returns
    -------
    plt.Figure
        The figure object.
    """
    n_plots = 3 if observations is not None else 2
    fig, axes = plt.subplots(1, n_plots, figsize=figsize)
    
    # Common color scale for mean/observations
    if observations is not None:
        all_values = np.concatenate([mean, observations[~np.isnan(observations)]])
    else:
        all_values = mean
    vmin, vmax = np.nanpercentile(all_values, [2, 98])
    
    # Plot 1: Mean
    ax1 = axes[0]
    scatter1 = ax1.scatter(
        coords[:, 1], coords[:, 0],
        c=mean, cmap='RdYlBu_r',
        vmin=vmin, vmax=vmax,
        s=20, edgecolors='none'
    )
    plt.colorbar(scatter1, ax=ax1, label='NO₂ (ppb)')
    ax1.set_title('Predicted Mean')
    ax1.set_xlabel('Longitude')
    ax1.set_ylabel('Latitude')
    ax1.set_aspect('equal')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Uncertainty
    ax2 = axes[1]
    scatter2 = ax2.scatter(
        coords[:, 1], coords[:, 0],
        c=std, cmap='YlOrRd',
        s=20, edgecolors='none'
    )
    plt.colorbar(scatter2, ax=ax2, label='Std Dev (ppb)')
    ax2.set_title('Prediction Uncertainty')
    ax2.set_xlabel('Longitude')
    ax2.set_ylabel('Latitude')
    ax2.set_aspect('equal')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Observations (if provided)
    if observations is not None and obs_coords is not None:
        ax3 = axes[2]
        mask = ~np.isnan(observations)
        scatter3 = ax3.scatter(
            obs_coords[mask, 1], obs_coords[mask, 0],
            c=observations[mask], cmap='RdYlBu_r',
            vmin=vmin, vmax=vmax,
            s=50, edgecolors='black', linewidths=0.5
        )
        plt.colorbar(scatter3, ax=ax3, label='NO₂ (ppb)')
        ax3.set_title('Observations')
        ax3.set_xlabel('Longitude')
        ax3.set_ylabel('Latitude')
        ax3.set_aspect('equal')
        ax3.grid(True, alpha=0.3)
    
    fig.suptitle(title, fontsize=14, y=1.02)
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved figure to {save_path}")
    
    return fig


def plot_time_series_maps(
    coords: np.ndarray,
    timestamps: np.ndarray,
    values: np.ndarray,
    n_times: int = 4,
    title: str = "NO₂ Time Series",
    figsize: Tuple[int, int] = (16, 12),
    save_path: Optional[Union[str, Path]] = None,
) -> plt.Figure:
    """
    Create a grid of maps for different time points.
    
    Parameters
    ----------
    coords : np.ndarray
        Coordinates, shape (N, 2).
    timestamps : np.ndarray
        Timestamps, shape (N,).
    values : np.ndarray
        Values, shape (N,).
    n_times : int
        Number of time points to show.
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
    # Get unique timestamps
    unique_times = np.unique(timestamps)
    
    # Select evenly spaced times
    if len(unique_times) > n_times:
        indices = np.linspace(0, len(unique_times) - 1, n_times, dtype=int)
        selected_times = unique_times[indices]
    else:
        selected_times = unique_times
        n_times = len(selected_times)
    
    # Create subplot grid
    n_cols = min(n_times, 2)
    n_rows = (n_times + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = np.atleast_1d(axes).flatten()
    
    # Common color scale
    vmin, vmax = np.nanpercentile(values, [2, 98])
    
    for i, t in enumerate(selected_times):
        ax = axes[i]
        mask = timestamps == t
        
        scatter = ax.scatter(
            coords[mask, 1], coords[mask, 0],
            c=values[mask], cmap='RdYlBu_r',
            vmin=vmin, vmax=vmax,
            s=30, edgecolors='none'
        )
        
        ax.set_title(f'Time: {t:.2f}')
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
    
    # Hide unused axes
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)
    
    # Add colorbar
    fig.subplots_adjust(right=0.85)
    cbar_ax = fig.add_axes([0.88, 0.15, 0.03, 0.7])
    cbar = fig.colorbar(scatter, cax=cbar_ax, label='NO₂ (ppb)')
    
    fig.suptitle(title, fontsize=14, y=0.98)
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved figure to {save_path}")
    
    return fig
