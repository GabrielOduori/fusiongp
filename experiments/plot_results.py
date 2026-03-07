"""
Plotting module for FusionGP experiment results.

Generates all standard figures from a trained model run.
Called from reproduce_paper_daily.py (or similar) with a single function.

Usage:
    from plot_results import generate_all_figures
    saved = generate_all_figures(figures_dir=..., history=..., ...)
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel as C

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.visualization.diagnostics import (
    plot_training_history,
    plot_calibration,
    plot_residuals,
)


def _plot_predicted_vs_actual(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metrics: Optional[dict] = None,
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """Scatter plot of predicted vs actual with 1:1 line and metrics."""
    fig, ax = plt.subplots(figsize=(6, 6))

    ax.scatter(y_true, y_pred, alpha=0.4, s=12, edgecolors="none", c="#4C72B0")

    lo = min(np.nanmin(y_true), np.nanmin(y_pred))
    hi = max(np.nanmax(y_true), np.nanmax(y_pred))
    margin = (hi - lo) * 0.05
    ax.plot([lo - margin, hi + margin], [lo - margin, hi + margin],
            "k--", linewidth=1, label="1:1 line")
    ax.set_xlim(lo - margin, hi + margin)
    ax.set_ylim(lo - margin, hi + margin)

    ax.set_xlabel("Observed EPA NO$_2$ ($\\mu$g/m$^3$)", fontsize=11)
    ax.set_ylabel("Predicted NO$_2$ ($\\mu$g/m$^3$)", fontsize=11)
    ax.set_title("Predicted vs Observed", fontsize=13, fontweight="bold")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.3, linestyle="--", linewidth=0.5)

    if metrics is not None:
        text = (
            f"R$^2$ = {metrics.get('r2', float('nan')):.3f}\n"
            f"RMSE = {metrics.get('rmse', float('nan')):.2f}\n"
            f"Bias = {metrics.get('bias', float('nan')):.2f}"
        )
        ax.text(0.05, 0.95, text, transform=ax.transAxes,
                fontsize=10, verticalalignment="top",
                bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.8))

    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
    return fig


def _plot_spatial_contour(
    predictor,
    scalers,
    timestamp_norm: float,
    nx: int = 80,
    ny: int = 80,
    task: Optional[str] = None,
    save_mean: Optional[Path] = None,
    save_std: Optional[Path] = None,
) -> tuple[plt.Figure, plt.Figure]:
    """Create contour maps of mean prediction and uncertainty."""
    lat_min, lon_min = scalers.coord_min
    lat_max, lon_max = scalers.coord_max

    lats = np.linspace(lat_min, lat_max, ny)
    lons = np.linspace(lon_min, lon_max, nx)
    lon_grid, lat_grid = np.meshgrid(lons, lats)

    coords = np.column_stack([lat_grid.ravel(), lon_grid.ravel()])
    times = np.full(len(coords), scalers.inverse_transform_time(
        np.array([timestamp_norm]))[0])

    preds = predictor.predict_locations(
        coords=coords, timestamps=times,
        normalized=False, verbose=False, task=task,
    )

    mean_map = preds.mean.reshape(lat_grid.shape)
    std_map = preds.std.reshape(lat_grid.shape)

    # Mean figure
    fig_mean, ax = plt.subplots(figsize=(8, 6))
    im = ax.contourf(lon_grid, lat_grid, mean_map, levels=20, cmap="RdYlBu_r")
    ax.set_title("Predicted NO$_2$ Concentration", fontsize=13, fontweight="bold")
    ax.set_xlabel("Longitude", fontsize=11)
    ax.set_ylabel("Latitude", fontsize=11)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.3, linestyle="--", linewidth=0.5)
    plt.colorbar(im, ax=ax, label="NO$_2$ ($\\mu$g/m$^3$)")
    fig_mean.tight_layout()
    if save_mean is not None:
        fig_mean.savefig(save_mean, dpi=300, bbox_inches="tight")
        plt.close(fig_mean)

    # Std figure
    fig_std, ax = plt.subplots(figsize=(8, 6))
    im = ax.contourf(lon_grid, lat_grid, std_map, levels=20, cmap="plasma")
    ax.set_title("Prediction Uncertainty (Std Dev)", fontsize=13, fontweight="bold")
    ax.set_xlabel("Longitude", fontsize=11)
    ax.set_ylabel("Latitude", fontsize=11)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.3, linestyle="--", linewidth=0.5)
    plt.colorbar(im, ax=ax, label="Std Dev ($\\mu$g/m$^3$)")
    fig_std.tight_layout()
    if save_std is not None:
        fig_std.savefig(save_std, dpi=300, bbox_inches="tight")
        plt.close(fig_std)

    return fig_mean, fig_std


def _select_mid_timestamp(df: pd.DataFrame, target_day: float) -> Tuple[str, float]:
    """
    Select the closest available timestamp to target_day (days since min).

    Returns (timestamp_str, day_value).
    """
    df = df.copy()
    df["timestamp_dt"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp_dt"])
    if df.empty:
        raise ValueError("No valid timestamps found in dataset")
    t0 = df["timestamp_dt"].min()
    df["day"] = (df["timestamp_dt"] - t0).dt.total_seconds() / (24 * 3600)
    idx = (df["day"] - target_day).abs().idxmin()
    row = df.loc[idx]
    return row["timestamp"], float(row["day"])


def _regression_kriging(
    df_day: pd.DataFrame,
    baseline_col: str,
    obs_col: str,
) -> Optional[pd.DataFrame]:
    """
    Regression-kriging: regress obs on baseline, krige residuals, add back.
    Returns a DataFrame with columns [latitude, longitude, fused].
    """
    needed = {"latitude", "longitude", baseline_col, obs_col}
    if not needed.issubset(df_day.columns):
        return None

    df_epa = df_day.dropna(subset=["latitude", "longitude", baseline_col, obs_col])
    if len(df_epa) < 5:
        return None

    # Linear regression: obs ~ baseline
    X = df_epa[[baseline_col]].to_numpy()
    y = df_epa[obs_col].to_numpy()
    lr = LinearRegression()
    lr.fit(X, y)
    resid = y - lr.predict(X)

    # Ordinary kriging surrogate with GP on residuals
    coords = df_epa[["latitude", "longitude"]].to_numpy()
    coord_scaler = StandardScaler()
    coords_scaled = coord_scaler.fit_transform(coords)
    kernel = C(1.0, (1e-2, 1e2)) * RBF(length_scale=1.0) + WhiteKernel(noise_level=1.0)
    gp = GaussianProcessRegressor(kernel=kernel, normalize_y=True)
    gp.fit(coords_scaled, resid)

    df_grid = df_day.dropna(subset=["latitude", "longitude", baseline_col])
    grid_coords = df_grid[["latitude", "longitude"]].to_numpy()
    grid_coords_scaled = coord_scaler.transform(grid_coords)

    resid_pred = gp.predict(grid_coords_scaled)
    base_pred = lr.predict(df_grid[[baseline_col]].to_numpy())
    fused = base_pred + resid_pred

    return pd.DataFrame({
        "latitude": df_grid["latitude"].to_numpy(),
        "longitude": df_grid["longitude"].to_numpy(),
        "fused": fused,
    })


def _plot_side_by_side_maps(
    maps: List[Tuple[str, pd.DataFrame]],
    fusion_map: pd.DataFrame,
    save_path: Path,
) -> None:
    """
    Plot side-by-side scatter maps with a shared color scale.
    """
    all_vals = []
    for _, df_map in maps:
        all_vals.append(df_map["fused"].to_numpy())
    all_vals.append(fusion_map["fused"].to_numpy())
    vmin = np.nanmin([np.nanmin(v) for v in all_vals])
    vmax = np.nanmax([np.nanmax(v) for v in all_vals])

    ncols = len(maps) + 1
    fig, axes = plt.subplots(1, ncols, figsize=(5 * ncols, 5), constrained_layout=True)
    if ncols == 1:
        axes = [axes]

    for ax, (title, df_map) in zip(axes, maps):
        sc = ax.scatter(
            df_map["longitude"], df_map["latitude"],
            c=df_map["fused"], s=8, cmap="RdYlBu_r", vmin=vmin, vmax=vmax,
            edgecolors="none",
        )
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.set_aspect("equal", adjustable="box")

    ax = axes[-1]
    sc = ax.scatter(
        fusion_map["longitude"], fusion_map["latitude"],
        c=fusion_map["fused"], s=8, cmap="RdYlBu_r", vmin=vmin, vmax=vmax,
        edgecolors="none",
    )
    ax.set_title("FusionGP", fontsize=12, fontweight="bold")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_aspect("equal", adjustable="box")

    fig.colorbar(sc, ax=axes, label="NO$_2$ ($\mu$g/m$^3$)")
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def generate_all_figures(
    figures_dir: Path,
    history,
    epa_test_orig: np.ndarray,
    pred_mean_at_epa: np.ndarray,
    pred_std_at_epa: np.ndarray,
    epa_metrics,
    epa_eval_mask: np.ndarray,
    test_data,
    train_data,
    scalers,
    predictor,
    per_source_metrics: Dict[str, dict],
    training_sources: List[str],
    learned_params: Optional[dict] = None,
    base_vs_epa_metrics=None,
    data_path: Optional[Path] = None,
    atmo_plan_path: Optional[Path] = None,
) -> List[Path]:
    """
    Generate all standard experiment figures.

    Parameters
    ----------
    figures_dir : Path
        Directory to save figures (created if needed).
    history : TrainingHistory
        Training history with .to_dict() returning
        {train_loss, val_loss, learning_rate, epoch_time}.
    epa_test_orig : np.ndarray
        EPA ground truth values (original scale).
    pred_mean_at_epa : np.ndarray
        Predicted means at EPA evaluation locations.
    pred_std_at_epa : np.ndarray
        Predicted std at EPA evaluation locations.
    epa_metrics : MetricsResult
        Evaluation result with .to_dict().
    epa_eval_mask : np.ndarray
        Boolean mask for EPA evaluation points in test_data.
    test_data : FusionData
        Test dataset.
    train_data : FusionData
        Training dataset.
    scalers : Scalers
        Normalization scalers.
    predictor : Predictor
        Trained predictor for spatial map generation.
    per_source_metrics : dict
        Per-source metrics dicts.
    training_sources : list of str
        Sources used during training.
    learned_params : dict, optional
        Learned hyperparameters.
    base_vs_epa_metrics : MetricsResult, optional
        Baseline vs EPA comparison metrics.

    Returns
    -------
    list of Path
        Paths to all saved figures.
    """
    figures_dir = Path(figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)
    saved = []

    print("\n   Generating figures...")

    # 1. Training curves
    path = figures_dir / "training_curves.png"
    plot_training_history(history.to_dict(), save_path=str(path))
    saved.append(path)
    print(f"   [1/6] Training curves")

    # 2. Predicted vs actual
    path = figures_dir / "predicted_vs_actual.png"
    _plot_predicted_vs_actual(
        epa_test_orig, pred_mean_at_epa,
        metrics=epa_metrics.to_dict(), save_path=path,
    )
    saved.append(path)
    print(f"   [2/6] Predicted vs actual")

    # 3. Residuals
    path = figures_dir / "residuals.png"
    plot_residuals(
        epa_test_orig, pred_mean_at_epa, pred_std_at_epa,
        save_path=str(path),
    )
    saved.append(path)
    print(f"   [3/6] Residuals")

    # 4. Calibration
    path = figures_dir / "calibration.png"
    plot_calibration(
        epa_test_orig, pred_mean_at_epa, pred_std_at_epa,
        save_path=str(path),
    )
    saved.append(path)
    print(f"   [4/6] Calibration")

    # 5 & 6. Spatial maps (mean + uncertainty) at mid-time
    path_mean = figures_dir / "spatial_prediction.png"
    path_std = figures_dir / "spatial_uncertainty.png"

    task = training_sources[0] if len(training_sources) == 1 else None
    if "epa" in training_sources:
        task = "epa"

    _plot_spatial_contour(
        predictor, scalers, timestamp_norm=0.5,
        task=task, save_mean=path_mean, save_std=path_std,
    )
    saved.append(path_mean)
    saved.append(path_std)
    print(f"   [5/6] Spatial prediction map")
    print(f"   [6/6] Spatial uncertainty map")

    # 7. Side-by-side maps: LUR regression-kriging, ATMO-Plan regression-kriging, FusionGP
    if data_path is not None and Path(data_path).exists():
        try:
            df = pd.read_csv(
                data_path,
                usecols=["grid_id", "latitude", "longitude", "timestamp", "epa_no2", "predicted_no2"],
            )
            # Determine mid-time date from scalers
            target_day = scalers.inverse_transform_time(np.array([0.5]))[0]
            timestamp_str, day_value = _select_mid_timestamp(df, target_day)
            df_day = df[df["timestamp"] == timestamp_str].copy()

            # LUR regression-kriging
            lur_map = _regression_kriging(df_day, "predicted_no2", "epa_no2")

            # ATMO-Plan regression-kriging
            atmo_map = None
            if atmo_plan_path is not None and Path(atmo_plan_path).exists():
                df_atmo = pd.read_csv(atmo_plan_path, usecols=["grid_id", "model_no2_10m"])
                df_day = df_day.merge(df_atmo, on="grid_id", how="left")
                atmo_map = _regression_kriging(df_day, "model_no2_10m", "epa_no2")

            # FusionGP predictions at grid coords for same date
            coords = df_day[["latitude", "longitude"]].to_numpy()
            times = np.full(len(coords), day_value)
            task = training_sources[0] if len(training_sources) == 1 else None
            if "epa" in training_sources:
                task = "epa"
            preds = predictor.predict_locations(
                coords=coords,
                timestamps=times,
                normalized=False,
                verbose=False,
                task=task,
            )
            fusion_map = pd.DataFrame({
                "latitude": df_day["latitude"].to_numpy(),
                "longitude": df_day["longitude"].to_numpy(),
                "fused": preds.mean,
            })

            maps = []
            if lur_map is not None:
                maps.append(("LUR + Regression-Kriging", lur_map))
            if atmo_map is not None:
                maps.append(("ATMO-Plan + Regression-Kriging", atmo_map))

            if maps:
                path = figures_dir / "comparison_regression_kriging_vs_fusiongp.png"
                _plot_side_by_side_maps(maps, fusion_map, path)
                saved.append(path)
                print(f"   [7/7] Regression-kriging vs FusionGP map")
        except Exception as exc:
            print(f"   ⚠ Regression-kriging comparison map skipped ({exc})")

    print(f"   Saved {len(saved)} figures to {figures_dir}")
    return saved
