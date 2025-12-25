"""
Enhanced spatial mapping utilities for FusionGP.

Provides high-quality contour maps for predictions and uncertainty
across multiple timestamps.
"""

import os
from typing import Optional, Tuple, List

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors


def create_spatial_maps(
    predictor,
    scalers,
    lat_range: Tuple[float, float],
    lon_range: Tuple[float, float],
    timestamps: Optional[List[float]] = None,
    n_times: int = 4,
    nx: int = 80,
    ny: int = 80,
    output_dir: str = 'outputs/spatial_maps',
    cmap_mean: str = 'RdYlBu_r',
    cmap_std: str = 'plasma',
):
    """
    Create spatial prediction and uncertainty maps for multiple timestamps.

    Parameters
    ----------
    predictor : Predictor
        Trained FusionGP predictor.
    scalers : Scalers
        Data scalers for coordinate transformation.
    lat_range : Tuple[float, float]
        (min_lat, max_lat) in original coordinates.
    lon_range : Tuple[float, float]
        (min_lon, max_lon) in original coordinates.
    timestamps : List[float], optional
        Specific timestamps to visualize (original scale).
        If None, will select n_times evenly spaced.
    n_times : int, default=4
        Number of timestamps to visualize if timestamps not provided.
    nx : int, default=80
        Number of grid points in longitude direction.
    ny : int, default=80
        Number of grid points in latitude direction.
    output_dir : str
        Directory to save output figures.
    cmap_mean : str
        Colormap for mean predictions.
    cmap_std : str
        Colormap for uncertainty.

    Returns
    -------
    None
        Saves figures to output_dir.
    """
    os.makedirs(output_dir, exist_ok=True)

    lat_min, lat_max = lat_range
    lon_min, lon_max = lon_range

    # Create spatial grid
    lons = np.linspace(lon_min, lon_max, nx)
    lats = np.linspace(lat_min, lat_max, ny)
    lon_grid, lat_grid = np.meshgrid(lons, lats)

    # Select timestamps
    if timestamps is None:
        # Use normalized time range [0, 1] evenly spaced
        timestamps_norm = np.linspace(0, 1, n_times)
        # Convert to original scale
        timestamps = scalers.inverse_transform_time(timestamps_norm)
    else:
        timestamps = np.array(timestamps)

    print(f"Creating {len(timestamps)} spatial maps...")
    print(f"Lat range: [{lat_min:.4f}, {lat_max:.4f}]")
    print(f"Lon range: [{lon_min:.4f}, {lon_max:.4f}]")

    for t_idx, timestamp in enumerate(timestamps):
        print(f"  Processing timestamp {t_idx+1}/{len(timestamps)}: {timestamp:.2f}")

        # Create grid for this timestamp
        coords = np.column_stack([
            lat_grid.ravel(),
            lon_grid.ravel()
        ])
        times = np.full(len(coords), timestamp)

        # Make predictions
        predictions = predictor.predict_locations(
            coords=coords,
            timestamps=times,
            normalized=False,
            verbose=False
        )

        # Reshape to grid
        mean_map = predictions.mean.reshape(lat_grid.shape)
        std_map = predictions.std.reshape(lat_grid.shape)

        # Create figure
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Mean predictions
        im1 = axes[0].contourf(lon_grid, lat_grid, mean_map, levels=20, cmap=cmap_mean)
        axes[0].set_title(f'Predicted NO₂ (t={timestamp:.2f})', fontsize=12, fontweight='bold')
        axes[0].set_xlabel('Longitude', fontsize=11)
        axes[0].set_ylabel('Latitude', fontsize=11)
        cbar1 = plt.colorbar(im1, ax=axes[0], label='NO₂ (µg/m³)')
        cbar1.ax.tick_params(labelsize=10)

        # Uncertainty
        im2 = axes[1].contourf(lon_grid, lat_grid, std_map, levels=20, cmap=cmap_std)
        axes[1].set_title(f'Uncertainty (t={timestamp:.2f})', fontsize=12, fontweight='bold')
        axes[1].set_xlabel('Longitude', fontsize=11)
        axes[1].set_ylabel('Latitude', fontsize=11)
        cbar2 = plt.colorbar(im2, ax=axes[1], label='Std Dev (µg/m³)')
        cbar2.ax.tick_params(labelsize=10)

        # Make square and clean
        for ax in axes:
            ax.set_aspect('equal', adjustable='box')
            ax.tick_params(labelsize=10)
            ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)

        plt.tight_layout()

        # Save figure
        filename = f'spatial_map_t{t_idx:02d}_{timestamp:.2f}.png'
        filepath = os.path.join(output_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close(fig)

        print(f"    Saved: {filename}")

    print(f"\nAll maps saved to: {output_dir}")


def create_animation_frames(
    predictor,
    scalers,
    lat_range: Tuple[float, float],
    lon_range: Tuple[float, float],
    n_frames: int = 24,
    nx: int = 100,
    ny: int = 100,
    output_dir: str = 'outputs/animation_frames',
    show_observations: bool = False,
    obs_coords: Optional[np.ndarray] = None,
):
    """
    Create frames for temporal animation of predictions.

    Parameters
    ----------
    predictor : Predictor
        Trained predictor.
    scalers : Scalers
        Data scalers.
    lat_range : Tuple[float, float]
        Latitude bounds.
    lon_range : Tuple[float, float]
        Longitude bounds.
    n_frames : int
        Number of frames to generate.
    nx, ny : int
        Grid resolution.
    output_dir : str
        Output directory.
    show_observations : bool
        Whether to overlay observation locations.
    obs_coords : np.ndarray, optional
        Observation coordinates to overlay, shape (N, 2).
    """
    os.makedirs(output_dir, exist_ok=True)

    lat_min, lat_max = lat_range
    lon_min, lon_max = lon_range

    # Create grid
    lons = np.linspace(lon_min, lon_max, nx)
    lats = np.linspace(lat_min, lat_max, ny)
    lon_grid, lat_grid = np.meshgrid(lons, lats)
    coords = np.column_stack([lat_grid.ravel(), lon_grid.ravel()])

    # Time points
    timestamps_norm = np.linspace(0, 1, n_frames)
    timestamps = scalers.inverse_transform_time(timestamps_norm)

    # Get global min/max for consistent coloring
    print("Computing global value ranges...")
    all_means = []
    for timestamp in timestamps[:5]:  # Sample a few for range
        times = np.full(len(coords), timestamp)
        preds = predictor.predict_locations(coords, times, normalized=False, verbose=False)
        all_means.append(preds.mean)

    vmin = np.percentile(np.concatenate(all_means), 2)
    vmax = np.percentile(np.concatenate(all_means), 98)

    print(f"Creating {n_frames} animation frames...")
    print(f"Value range: [{vmin:.2f}, {vmax:.2f}]")

    for frame_idx, timestamp in enumerate(timestamps):
        times = np.full(len(coords), timestamp)
        predictions = predictor.predict_locations(
            coords=coords,
            timestamps=times,
            normalized=False,
            verbose=False
        )

        mean_map = predictions.mean.reshape(lat_grid.shape)

        fig, ax = plt.subplots(figsize=(10, 8))

        im = ax.contourf(
            lon_grid, lat_grid, mean_map,
            levels=20,
            cmap='RdYlBu_r',
            vmin=vmin,
            vmax=vmax
        )

        if show_observations and obs_coords is not None:
            ax.scatter(
                obs_coords[:, 1], obs_coords[:, 0],
                c='black', s=10, alpha=0.5, marker='x',
                label='Observations'
            )
            ax.legend(loc='upper right')

        ax.set_title(f'NO₂ Predictions - Time: {timestamp:.2f}', fontsize=14, fontweight='bold')
        ax.set_xlabel('Longitude', fontsize=12)
        ax.set_ylabel('Latitude', fontsize=12)
        ax.set_aspect('equal', adjustable='box')
        ax.grid(True, alpha=0.3)

        cbar = plt.colorbar(im, ax=ax, label='NO₂ (µg/m³)')
        cbar.ax.tick_params(labelsize=10)

        plt.tight_layout()

        filename = f'frame_{frame_idx:04d}.png'
        filepath = os.path.join(output_dir, filename)
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close(fig)

        if (frame_idx + 1) % 5 == 0:
            print(f"  Generated {frame_idx + 1}/{n_frames} frames")

    print(f"\nAll frames saved to: {output_dir}")
    print(f"To create video: ffmpeg -framerate 10 -i {output_dir}/frame_%04d.png -c:v libx264 -pix_fmt yuv420p output.mp4")
