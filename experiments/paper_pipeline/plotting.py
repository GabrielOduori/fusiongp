"""Figure generation: training curves, prediction/uncertainty maps, diagnostics."""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

from src.visualization import plot_predictions, plot_uncertainty, plot_confidence_intervals
from src.visualization.spatial_maps import create_spatial_maps


def plot_epa_timeseries_by_site(
    data,
    predictions,
    scalers,
    output_dir,
    n_sites=5,
):
    """
    Plot observed vs predicted EPA time series for selected grid sites.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Prefer raw timestamps for readability
    if hasattr(data, "raw_timestamps") and data.raw_timestamps is not None:
        x_time = pd.to_datetime(data.raw_timestamps, errors="coerce")
    else:
        x_time = pd.to_datetime(data.timestamps, errors="coerce")

    epa_mask = data.source_masks.get("epa")
    if epa_mask is None or epa_mask.sum() == 0:
        return

    df = pd.DataFrame({
        "grid_id": data.grid_ids,
        "timestamp": x_time,
        "epa_obs": data.observations["epa"],
        "pred_mean": predictions.mean,
    })
    df = df[epa_mask]

    # Select top sites by number of EPA observations
    top_sites = (
        df.groupby("grid_id")["epa_obs"]
        .count()
        .sort_values(ascending=False)
        .head(n_sites)
        .index
        .tolist()
    )
    if not top_sites:
        return

    df = df[df["grid_id"].isin(top_sites)].copy()
    df = df.sort_values("timestamp")

    fig, ax = plt.subplots(figsize=(12, 6))

    colors = plt.cm.tab10.colors
    for i, gid in enumerate(top_sites):
        site = df[df["grid_id"] == gid]
        color = colors[i % len(colors)]
        ax.plot(site["timestamp"], site["epa_obs"], label=f"EPA obs (grid {gid})", color=color, linewidth=1.0)
        ax.plot(site["timestamp"], site["pred_mean"], label=f"Pred (grid {gid})", color=color, linestyle="--", alpha=0.8)

    ax.set_title("EPA Observed vs Predicted (Top Sites)")
    ax.set_ylabel("NO₂")
    ax.set_xlabel("Time")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / "epa_timeseries_sites.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_training_curves(history, save_path):
    """
    Plot training and validation loss curves.

    Args:
        history: TrainingHistory object from trainer
        save_path: Path to save the plot
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Training and Validation Loss
    epochs_train = np.arange(1, len(history.train_loss) + 1)
    ax1.plot(epochs_train, history.train_loss, 'b-', linewidth=2, label='Training Loss', alpha=0.8)

    if history.val_loss:
        # Validation is computed every val_interval epochs
        val_interval = len(history.train_loss) // len(history.val_loss) if len(history.val_loss) > 1 else 5
        epochs_val = np.arange(val_interval, len(history.train_loss) + 1, val_interval)[:len(history.val_loss)]
        ax1.plot(epochs_val, history.val_loss, 'r-', linewidth=2, label='Validation Loss', alpha=0.8, marker='o')

    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss (Negative ELBO)', fontsize=12)
    ax1.set_title('Training Progress', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    # Highlight best epoch if validation exists
    if history.val_loss:
        best_epoch = history.best_epoch
        best_loss = history.best_val_loss
        ax1.axvline(x=epochs_val[best_epoch], color='green', linestyle='--', linewidth=1.5, alpha=0.7, label='Best Model')
        ax1.scatter([epochs_val[best_epoch]], [best_loss], color='green', s=100, zorder=5, marker='*')
        ax1.legend(fontsize=10)

    # Plot 2: Learning Rate Schedule
    ax2.plot(epochs_train, history.learning_rates, 'g-', linewidth=2, alpha=0.8)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Learning Rate', fontsize=12)
    ax2.set_title('Learning Rate Schedule', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.set_yscale('log')

    plt.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

    return fig


def create_prediction_plots(grid_predictions, predictions, epa_eval_mask, test_data, scalers, figures_dir):
    """Save predictions/uncertainty maps and the EPA confidence-interval plot."""
    plot_predictions(
        coords=grid_predictions.coords,
        values=grid_predictions.mean,
        save_path=str(figures_dir / "predictions.png"),
    )
    print(f"   ✓ Saved predictions plot")

    plot_uncertainty(
        coords=grid_predictions.coords,
        std=grid_predictions.std,
        save_path=str(figures_dir / "uncertainty.png"),
    )
    print(f"   ✓ Saved uncertainty plot")

    if epa_eval_mask.any():
        y_epa = test_data.observations["epa"][epa_eval_mask]
        if scalers.normalize_targets and "epa" in scalers.target_std:
            y_epa = y_epa * scalers.target_std["epa"] + scalers.target_mean["epa"]
        x_idx = np.arange(len(y_epa))
        fig_ci = plot_confidence_intervals(
            x=x_idx,
            mean=predictions.mean[epa_eval_mask],
            std=predictions.std[epa_eval_mask],
            y_true=y_epa,
            confidence_levels=[0.5, 0.8, 0.9, 0.95],
            xlabel="EPA test index",
            ylabel="NO₂ (µg/m³)",
            title="EPA Predictions with Confidence Intervals",
            save_path=str(figures_dir / "uncertainty_ci.png"),
        )
        plt.close(fig_ci)
        print(f"   ✓ Saved confidence intervals plot")


def create_spatial_maps_and_surfaces(
    predictor,
    scalers,
    lat_min,
    lat_max,
    lon_min,
    lon_max,
    primary_source,
    use_analysis_grid,
    analysis_grid,
    grid_predictions,
    figures_dir,
):
    """Save the enhanced spatial maps plus AnalysisGrid (or legacy) surface plots."""
    print("\n   Creating enhanced spatial maps...")
    create_spatial_maps(
        predictor=predictor,
        scalers=scalers,
        lat_range=(lat_min, lat_max),
        lon_range=(lon_min, lon_max),
        n_times=4,
        nx=100,
        ny=100,
        output_dir=str(figures_dir / 'spatial_maps'),
        cmap_mean='RdYlBu_r',
        cmap_std='plasma',
        task=primary_source,
    )
    print(f"   ✓ Saved spatial maps to: {figures_dir / 'spatial_maps'}")

    if use_analysis_grid and analysis_grid is not None:
        print("\n   Creating gridded surface plots (AnalysisGrid)...")

        fig = grid_predictions.plot_with_uncertainty(
            timestamp_idx=0,
            figsize=(14, 5),
            vmin=0,
            vmax=None,  # Auto-scale
        )
        fig.savefig(figures_dir / "predictions_surface.png", dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"   ✓ Saved mean + uncertainty surface plot")

        fig, ax = plt.subplots(figsize=(10, 8))
        grid_predictions.plot(
            'mean',
            timestamp_idx=0,
            ax=ax,
            cmap='RdYlBu_r',
            title=f'NO₂ Prediction (t={grid_predictions.timestamps[0]:.2f})',
            colorbar_label='NO₂ (µg/m³)',
        )
        fig.savefig(figures_dir / "mean_surface_grid.png", dpi=300, bbox_inches="tight")
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(10, 8))
        grid_predictions.plot(
            'std',
            timestamp_idx=0,
            ax=ax,
            cmap='magma',
            title=f'Prediction Uncertainty (t={grid_predictions.timestamps[0]:.2f})',
            colorbar_label='Std Dev (µg/m³)',
        )
        fig.savefig(figures_dir / "uncertainty_surface_grid.png", dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"   ✓ Saved individual surface plots")

    else:
        ts0 = np.unique(grid_predictions.timestamps)[0]
        mask = grid_predictions.timestamps == ts0
        lat_unique = np.unique(grid_predictions.coords[mask][:, 0])
        lon_unique = np.unique(grid_predictions.coords[mask][:, 1])

        mean_grid = grid_predictions.mean[mask].reshape(len(lat_unique), len(lon_unique))
        std_grid = grid_predictions.std[mask].reshape(len(lat_unique), len(lon_unique))

        fig, ax = plt.subplots(figsize=(8, 6))
        pcm = ax.pcolormesh(lon_unique, lat_unique, mean_grid, shading="auto", cmap="RdYlBu_r")
        fig.colorbar(pcm, ax=ax, label="Mean NO₂ (µg/m³)")
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.set_title(f"Gridded Mean Predictions (t={ts0:.2f})")
        fig.savefig(figures_dir / "predictions_surface.png", dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"   ✓ Saved mean surface plot")

        fig, ax = plt.subplots(figsize=(8, 6))
        pcm = ax.pcolormesh(lon_unique, lat_unique, std_grid, shading="auto", cmap="YlOrRd")
        fig.colorbar(pcm, ax=ax, label="Std Dev (µg/m³)")
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.set_title(f"Gridded Uncertainty (t={ts0:.2f})")
        fig.savefig(figures_dir / "uncertainty_surface.png", dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"   ✓ Saved uncertainty surface plot")


def create_grid_scatter_plots(grid_results_df, timestamps_for_export, figures_dir):
    """Scatter-plot mean/std predictions at grid centroids for the first 3 export timestamps."""
    for t in timestamps_for_export[:3]:  # Plot first 3 timestamps
        data_t = grid_results_df[grid_results_df.timestamp == t]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

        # Predictions (grid centroids as squares)
        scatter1 = ax1.scatter(
            data_t.longitude,
            data_t.latitude,
            c=data_t.predicted_mean,
            s=10,
            marker='s',  # Square markers for grid cells
            cmap='RdYlBu_r',
            alpha=0.95,
            edgecolors='none'
        )
        ax1.set_xlabel('Longitude', fontsize=12)
        ax1.set_ylabel('Latitude', fontsize=12)
        ax1.set_title(f'NO₂ Predictions at t={t:.1f} days (Grid Centroids)', fontsize=14, fontweight='bold')
        ax1.set_aspect('equal')
        ax1.grid(True, alpha=0.2)
        cbar1 = plt.colorbar(scatter1, ax=ax1, label='NO₂ (µg/m³)')
        cbar1.ax.tick_params(labelsize=10)

        # Uncertainty (grid centroids as squares)
        scatter2 = ax2.scatter(
            data_t.longitude,
            data_t.latitude,
            c=data_t.predicted_std,
            s=10,
            marker='s',
            cmap='viridis',
            alpha=0.95,
            edgecolors='none'
        )
        ax2.set_xlabel('Longitude', fontsize=12)
        ax2.set_ylabel('Latitude', fontsize=12)
        ax2.set_title(f'Uncertainty (Std) at t={t:.1f} days (Grid Centroids)', fontsize=14, fontweight='bold')
        ax2.set_aspect('equal')
        ax2.grid(True, alpha=0.2)
        cbar2 = plt.colorbar(scatter2, ax=ax2, label='Std (µg/m³)')
        cbar2.ax.tick_params(labelsize=10)

        plt.tight_layout()

        output_path_t = figures_dir / f"grid_predictions_t{t:.1f}.png"
        plt.savefig(output_path_t, dpi=250, bbox_inches='tight')
        plt.close()

    print(f"   ✓ Saved grid-based visualizations (3 timestamps)")


def create_diagnostic_plots(epa_eval_mask, test_data, epa_test_orig, predictions, figures_dir):
    """Time series with confidence bands, Q-Q plot of residuals, spatial uncertainty map."""
    print("      • Time series with confidence bands...")
    epa_coords = test_data.coords[epa_eval_mask]
    epa_times = test_data.timestamps[epa_eval_mask]
    epa_true = epa_test_orig
    epa_pred_mean = predictions.mean[epa_eval_mask]
    epa_pred_std = predictions.std[epa_eval_mask]

    sort_idx = np.argsort(epa_times)
    times_sorted = epa_times[sort_idx]
    true_sorted = epa_true[sort_idx]
    pred_sorted = epa_pred_mean[sort_idx]
    std_sorted = epa_pred_std[sort_idx]

    if len(times_sorted) > 1000:
        step = len(times_sorted) // 1000
        times_sorted = times_sorted[::step]
        true_sorted = true_sorted[::step]
        pred_sorted = pred_sorted[::step]
        std_sorted = std_sorted[::step]

    ci_lower = pred_sorted - 1.96 * std_sorted
    ci_upper = pred_sorted + 1.96 * std_sorted

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.fill_between(times_sorted, ci_lower, ci_upper,
                    alpha=0.3, color='gray',
                    label='Model Prediction (95% Confidence)')
    ax.plot(times_sorted, pred_sorted, 'k-', linewidth=2, label='Model Mean')
    ax.plot(times_sorted, true_sorted, 'ro', markersize=4, alpha=0.6, label='Actual (EPA)')
    ax.set_xlabel('Time Step', fontsize=12)
    ax.set_ylabel('NO₂ Concentration [µg/m³]', fontsize=12)
    ax.set_title('Model Predictions vs Actual Measurements (95% Confidence Interval)',
                 fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    timeseries_path = figures_dir / "timeseries_with_confidence.png"
    plt.savefig(timeseries_path, dpi=300, bbox_inches='tight')
    plt.close()

    print("      • Q-Q plot of residuals...")
    residuals = epa_true - epa_pred_mean
    standardized_residuals = residuals / epa_pred_std
    if len(standardized_residuals) > 1000:
        sample_idx = np.random.choice(len(standardized_residuals), 1000, replace=False)
        standardized_residuals_sample = standardized_residuals[sample_idx]
    else:
        standardized_residuals_sample = standardized_residuals

    fig, ax = plt.subplots(figsize=(7, 7))
    stats.probplot(standardized_residuals_sample, dist="norm", plot=ax)
    ax.axvline(x=-1.96, color='gray', linestyle='--', alpha=0.5, label='95% CI')
    ax.axvline(x=1.96, color='gray', linestyle='--', alpha=0.5)
    ax.set_title('Q-Q Plot of Standardized Residuals', fontsize=14, fontweight='bold')
    ax.set_xlabel('Theoretical Quantiles', fontsize=12)
    ax.set_ylabel('Sample Quantiles', fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    qq_path = figures_dir / "qq_plot_residuals.png"
    plt.savefig(qq_path, dpi=300, bbox_inches='tight')
    plt.close()

    print("      • Spatial uncertainty map...")
    fig, ax = plt.subplots(figsize=(10, 8))
    scatter = ax.scatter(
        epa_coords[:, 1],  # longitude
        epa_coords[:, 0],  # latitude
        c=epa_pred_std,
        s=25,
        cmap='YlOrRd',
        alpha=0.7,
        edgecolors='none',
        vmin=0,
        vmax=np.percentile(epa_pred_std, 95)
    )
    ax.set_xlabel('Longitude [°]', fontsize=12)
    ax.set_ylabel('Latitude [°]', fontsize=12)
    ax.set_title('Spatial Distribution of Prediction Uncertainty (Std Deviation)',
                 fontsize=14, fontweight='bold')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    cbar = plt.colorbar(scatter, ax=ax, label='Std Deviation [µg/m³]')
    cbar.ax.tick_params(labelsize=10)
    plt.tight_layout()
    spatial_path = figures_dir / "spatial_uncertainty_map.png"
    plt.savefig(spatial_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"   ✓ Saved additional evaluation visualizations")
