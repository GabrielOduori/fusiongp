"""
Reproduce FusionGP paper results.

Multi-source NO2 fusion with evaluation against EPA ground truth.

Methodology:
- When USE_EPA_IN_TRAINING=False: Train on Satellite (and prior), evaluate on held-out EPA
- When USE_EPA_IN_TRAINING=True: Legacy mode with all sources (has data leakage)

Pre-calibration uses colocated EPA-LCS observations to learn a linear correction
before training, avoiding data leakage while bringing sources to EPA scale.

Usage:
    python reproduce_paper.py

Outputs saved to results/experiment_YYYYMMDD_HHMMSS/ with subdirectories:
    figures/ - Training curves, predictions, uncertainty maps
    models/  - Trained model checkpoint
    tables/  - Metrics CSVs and experiment summary
"""

import sys
import time
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
from scipy import stats
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data import DataLoader, DataPreprocessor, AnalysisGrid
from src.models import FusionSVGP, MultiTaskSVGP, ATMOPlanMean, GridPriorMean
from src.training import Trainer
from src.training.callbacks import EarlyStopping
from src.inference import Predictor
from src.evaluation import Evaluator, rmse, mae, r_squared, bias
from src.visualization import plot_predictions, plot_uncertainty, plot_confidence_intervals
from src.visualization.spatial_maps import create_spatial_maps


# =============================================================================
# Configuration
# =============================================================================

# Use the vetted real dataset (no synthetic/low-cost inputs) to avoid data leakage.
DATA_PATH = Path("/media/gabriel-oduori/SERVER/dev_space/FusionGP/data_test/FusionData.csv")

COVARIATE_COLUMNS = []
COVARIATE_DISTANCE_COLUMNS = []

MODEL_CONFIG = {
    "n_inducing": 100,
    "spatial_kernel_type": "matern32",
    "temporal_kernel_type": "exponential",
    "learn_inducing_locations": True,
    "learn_kernel_hyperparams": True,
    "learn_noise": True,
    "learn_calibration": True,
    "initial_lengthscales": {"spatial_x": 0.1, "spatial_y": 0.1, "temporal": 0.1},
    "initial_noise": {"epa": 0.3, "satellite": 2.0},
}

TRAINING_CONFIG = {
    "learning_rate": 0.01, #0.01
    "n_epochs": 50,#150
    "batch_size": 2048,
    "val_interval": 5,
    "gradient_clip": 1.0,  # Prevent NaN gradients from corrupting parameters
}

GRID_RESOLUTION = 0.01
GRID_RESOLUTION_M = 100
USE_ANALYSIS_GRID = True
USE_PRIOR_GRID_FOR_PREDICTION = True
GEOJSON_GRID_PATH = Path("/media/gabriel-oduori/SERVER/dev_space/data-tools/osm/run_20260130_074609/grid.geojson")
MAX_ANALYSIS_GRID_POINTS = 5_000_000


def build_multi_case_definitions():
    """Return case definitions for multi-case evaluation.

    Sources available: EPA monitors, satellite (TROPOMI), and LUR prior mean.
    """
    return [
        {"name": "Case 0 (EPA only)", "sources": ["epa"], "mode": "train", "use_prior": False},
        {"name": "Case 1 (LUR prior only)", "sources": [], "mode": "prior_only", "use_prior": True},
        {"name": "Case 2 (LUR prior + Satellite)", "sources": ["satellite"], "mode": "train", "use_prior": True},
        {"name": "Case 3 (LUR prior + EPA + Satellite)", "sources": ["epa", "satellite"], "mode": "train", "use_prior": True},
    ]

USE_EPA_HOLDOUT = True
HOLDOUT_EPA_FRAC = 0.2
HOLDOUT_EPA_BY_GRID = True
HOLDOUT_SEED = 42

USE_EPA_IN_TRAINING = False  # EPA is evaluation-only
RUN_MULTI_CASE_EVALUATION = False
USE_MULTITASK_MODEL = False  # Set to False to use FusionSVGP with ATMO-Plan prior
PRIMARY_SOURCE = "epa"
SATELLITE_KEEP_FRAC = 0.1
HAS_SATELLITE = True
RUN_CASE0_EPA_ONLY = False

AVAILABLE_SOURCES = ["epa"] + (["satellite"] if HAS_SATELLITE else [])

EPA_INTERP_COLUMN = "epa_interpolated"
USE_EPA_INTERPOLATED = False
PRE_CALIBRATE_SOURCES = False  # EPA evaluation-only; no calibration
EPA_OUTLIER_FILTER = True
EPA_OUTLIER_METHOD = "iqr"  # "iqr" or "zscore"
EPA_OUTLIER_IQR_MULT = 3.0
EPA_OUTLIER_Z_THRESH = 4.0
EPA_OUTLIER_ACTION = "nan"  # "nan", "clip", or "drop"
EPA_OUTLIER_CLEANED_SUFFIX = "_epa_cleaned"

# Prior mean configuration
# Option 1: Use GridPriorMean with CSV column (preferred for new datasets)
USE_GRID_PRIOR = True  # Set True to use grid-based prior from CSV column
GRID_PRIOR_COLUMN = "model_no2_10m"  # Column containing prior values (if using new data format)
GRID_PRIOR_LEARNABLE_BIAS = False  # Fixed at 0 to preserve raw ATMOS-Plan baseline
# USE_TROPOMI_PATTERN = False
# TROPOMI_COLUMN = "satellite_no2"
# TROPOMI_CALIBRATION_DAYS = 14
# TROPOMI_RESTRICTED = True
# AUGMENTED_PRIOR_COLUMN = "model_no2_10m_tropomi"

# # Option 2: Use ATMOPlanMean with GeoTIFF (for test_data_updated.csv)
# USE_ATMO_PLAN_PRIOR = False  # Use ATMO-Plan GeoTIFF as GP prior mean
# ATMO_PLAN_PATH = Path(__file__).parent.parent / "data/atmo_plan_dublin.tif"
# ATMO_PLAN_LEARNABLE_BIAS = True  # Learn bias correction for ATMO-Plan


# =============================================================================
# Helper Functions
# =============================================================================

def calibrate_sources_to_epa(data, sources_to_calibrate=('satellite',)):
    """
    Pre-calibrate LCS and satellite data to EPA scale using colocated observations.

    This learns a linear transformation (y = a*x + b) from each source to EPA
    using only colocated observations (where both EPA and source have valid data).

    The calibration is learned from colocated points, then applied to ALL points
    of that source. This corrects systematic bias before fusion.

    Parameters
    ----------
    data : FusionData
        Original fusion data with all sources.
    sources_to_calibrate : tuple of str
        Sources to calibrate (default: satellite).

    Returns
    -------
    FusionData
        Data with calibrated observations for specified sources.
    calibration_params : dict
        Calibration parameters {source: {'slope': a, 'intercept': b, 'n_points': n}}
    """
    from src.data.loader import FusionData

    calibration_params = {}
    calibrated_observations = dict(data.observations)  # Copy

    epa_mask = data.source_masks.get('epa', np.zeros(len(data.coords), dtype=bool))
    epa_vals = data.observations.get('epa', np.full(len(data.coords), np.nan))

    for source in sources_to_calibrate:
        if source not in data.source_masks or source not in data.observations:
            print(f"   ⚠ Source '{source}' not found, skipping calibration")
            continue

        source_mask = data.source_masks[source]
        source_vals = data.observations[source]

        # Find colocated points (both EPA and source have valid data)
        coloc_mask = epa_mask & source_mask
        n_coloc = coloc_mask.sum()

        if n_coloc < 10:
            print(f"   ⚠ Only {n_coloc} colocated points for '{source}', skipping calibration")
            calibration_params[source] = {'slope': 1.0, 'intercept': 0.0, 'n_points': n_coloc}
            continue

        # Get colocated values
        epa_coloc = epa_vals[coloc_mask]
        source_coloc = source_vals[coloc_mask]

        valid = ~np.isnan(epa_coloc) & ~np.isnan(source_coloc)
        epa_coloc = epa_coloc[valid]
        source_coloc = source_coloc[valid]

        if len(epa_coloc) < 10:
            print(f"   ⚠ Only {len(epa_coloc)} valid colocated points for '{source}', skipping")
            calibration_params[source] = {'slope': 1.0, 'intercept': 0.0, 'n_points': len(epa_coloc)}
            continue

        A = np.vstack([source_coloc, np.ones_like(source_coloc)]).T
        result = np.linalg.lstsq(A, epa_coloc, rcond=None)
        slope, intercept = result[0]

        predicted = slope * source_coloc + intercept
        ss_res = np.sum((epa_coloc - predicted) ** 2)
        ss_tot = np.sum((epa_coloc - epa_coloc.mean()) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

        calibration_params[source] = {
            'slope': slope,
            'intercept': intercept,
            'n_points': len(epa_coloc),
            'r2': r2,
            'source_mean_before': source_vals[source_mask].mean(),
            'epa_mean': epa_vals[epa_mask].mean(),
        }

        calibrated_vals = source_vals.copy()
        calibrated_vals[source_mask] = slope * source_vals[source_mask] + intercept
        calibrated_observations[source] = calibrated_vals

        print(f"   ✓ Calibrated '{source}': EPA = {slope:.4f} * {source} + {intercept:.4f}")
        print(f"      Colocated points: {len(epa_coloc)}, R²: {r2:.4f}")
        print(f"      Mean before: {calibration_params[source]['source_mean_before']:.4f}, "
              f"EPA mean: {calibration_params[source]['epa_mean']:.4f}")
        calibrated_mean = calibrated_vals[source_mask].mean()
        print(f"      Mean after calibration: {calibrated_mean:.4f}")

    # Create new FusionData with calibrated observations
    calibrated_data = FusionData(
        coords=data.coords,
        timestamps=data.timestamps,
        observations=calibrated_observations,
        source_masks=data.source_masks,
        grid_ids=data.grid_ids,
        raw_timestamps=data.raw_timestamps if hasattr(data, 'raw_timestamps') else None,
        covariates=data.covariates if hasattr(data, 'covariates') else None,
        metadata=data.metadata if hasattr(data, 'metadata') else {}
    )

    return calibrated_data, calibration_params

def filter_data_by_sources(data, source_names):
    """
    Filter FusionData to only include specified sources.

    Parameters
    ----------
    data : FusionData
        Original fusion data
    source_names : list of str
        Source names to keep (e.g., ['epa', 'low_cost'])

    Returns
    -------
    FusionData
        Filtered data with only specified sources
    """
    from src.data.loader import FusionData

    # Create masks for included sources
    keep_mask = np.zeros(len(data.coords), dtype=bool)
    for source in source_names:
        if source in data.source_masks:
            keep_mask |= data.source_masks[source]

    # Filter all attributes
    filtered_data = FusionData(
        coords=data.coords[keep_mask],
        timestamps=data.timestamps[keep_mask],
        observations={
            src: obs[keep_mask]
            for src, obs in data.observations.items()
            if src in source_names
        },
        source_masks={
            src: mask[keep_mask]
            for src, mask in data.source_masks.items()
            if src in source_names
        },
        grid_ids=data.grid_ids[keep_mask] if data.grid_ids is not None else None,
        raw_timestamps=data.raw_timestamps[keep_mask] if hasattr(data, 'raw_timestamps') and data.raw_timestamps is not None else None,
        covariates=data.covariates[keep_mask] if getattr(data, "covariates", None) is not None else None,
        metadata=data.metadata if hasattr(data, 'metadata') else {}
    )

    return filtered_data


def downsample_source(data, source, keep_frac=1.0, seed=42):
    """
    Randomly downsample a source by masking out a fraction of its observations.
    """
    if keep_frac >= 1.0:
        return data

    from src.data.loader import FusionData

    rng = np.random.default_rng(seed)
    mask = data.source_masks.get(source)
    if mask is None:
        return data

    source_idx = np.where(mask)[0]
    if len(source_idx) == 0:
        return data

    n_keep = max(1, int(len(source_idx) * keep_frac))
    keep_idx = rng.choice(source_idx, size=n_keep, replace=False)

    new_masks = data.source_masks.copy()
    new_obs = data.observations.copy()

    mask_new = np.zeros_like(mask, dtype=bool)
    mask_new[keep_idx] = True
    new_masks[source] = mask_new

    obs_new = new_obs[source].copy()
    obs_new[~mask_new] = np.nan
    new_obs[source] = obs_new

    return FusionData(
        coords=data.coords,
        timestamps=data.timestamps,
        observations=new_obs,
        source_masks=new_masks,
        grid_ids=data.grid_ids,
        raw_timestamps=data.raw_timestamps if hasattr(data, "raw_timestamps") else None,
        covariates=data.covariates if getattr(data, "covariates", None) is not None else None,
        metadata=data.metadata if hasattr(data, "metadata") else {}
    )


def build_tropomi_pattern_prior(
    df: pd.DataFrame,
    model_col: str,
    tropomi_col: str,
    window_days: int = 14,
    restricted: bool = True,
    output_col: str = "model_no2_10m_tropomi",
) -> pd.DataFrame:
    """
    Build a TROPOMI-augmented prior using a typical-pattern approach.

    The pattern is computed over a calibration window and then added to the
    model prior for all timestamps.
    """
    if model_col not in df.columns:
        raise ValueError(f"Missing model prior column: {model_col}")
    if tropomi_col not in df.columns:
        raise ValueError(f"Missing TROPOMI column: {tropomi_col}")

    ts = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
    if ts.isna().all():
        ts_num = pd.to_numeric(df["timestamp"], errors="coerce")
        if ts_num.isna().all():
            raise ValueError("Unable to parse timestamps for TROPOMI calibration window")
        ts = pd.to_datetime(ts_num, unit="s", errors="coerce", utc=True)

    start = ts.min()
    end = start + pd.Timedelta(days=window_days)
    calib_mask = (ts >= start) & (ts < end)

    if restricted:
        calib_mask = calib_mask & df[tropomi_col].notna()

    if calib_mask.sum() == 0:
        raise ValueError("No TROPOMI data available in calibration window")

    calib = df.loc[calib_mask, ["grid_id", "latitude", "longitude", model_col, tropomi_col]].copy()

    sat_mean = calib.groupby("grid_id", as_index=False)[tropomi_col].mean()
    model_mean = calib.groupby("grid_id", as_index=False)[model_col].mean()

    merged = pd.merge(sat_mean, model_mean, on="grid_id", how="inner")
    merged = merged.dropna()
    if len(merged) < 2:
        slope, intercept = 1.0, 0.0
    else:
        x = merged[tropomi_col].values
        y = merged[model_col].values
        A = np.vstack([x, np.ones_like(x)]).T
        slope, intercept = np.linalg.lstsq(A, y, rcond=None)[0]

    merged["pattern"] = slope * merged[tropomi_col] + intercept - merged[model_col]
    pattern_map = merged[["grid_id", "pattern"]]

    df = df.copy()
    df = df.merge(pattern_map, on="grid_id", how="left")
    df["pattern"] = df["pattern"].fillna(0.0)
    df[output_col] = df[model_col] + df["pattern"]
    df.drop(columns=["pattern"], inplace=True)
    return df


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


def _polygon_centroid(coords):
    """Compute centroid for a polygon ring (lon, lat)."""
    if len(coords) < 3:
        lons, lats = zip(*coords)
        return float(np.mean(lons)), float(np.mean(lats))
    if coords[0] != coords[-1]:
        coords = coords + [coords[0]]
    area = 0.0
    cx = 0.0
    cy = 0.0
    for i in range(len(coords) - 1):
        x0, y0 = coords[i]
        x1, y1 = coords[i + 1]
        cross = x0 * y1 - x1 * y0
        area += cross
        cx += (x0 + x1) * cross
        cy += (y0 + y1) * cross
    if area == 0.0:
        lons, lats = zip(*coords)
        return float(np.mean(lons)), float(np.mean(lats))
    area *= 0.5
    cx /= (6.0 * area)
    cy /= (6.0 * area)
    return float(cx), float(cy)


def load_geojson_grid_centroids(path: Path) -> pd.DataFrame:
    """Load grid centroids from a GeoJSON file with grid_id properties."""
    import json

    path = Path(path)
    with path.open("r") as f:
        geo = json.load(f)

    records = []
    for feature in geo.get("features", []):
        props = feature.get("properties", {})
        grid_id = props.get("grid_id")
        geom = feature.get("geometry", {})
        gtype = geom.get("type")
        coords = geom.get("coordinates", [])
        if gtype == "Polygon" and coords:
            ring = coords[0]
            lon, lat = _polygon_centroid(ring)
        elif gtype == "MultiPolygon" and coords:
            ring = coords[0][0]
            lon, lat = _polygon_centroid(ring)
        else:
            continue
        records.append({"grid_id": grid_id, "longitude": lon, "latitude": lat})

    return pd.DataFrame.from_records(records)


def flag_epa_outliers(
    df,
    column="epa_no2",
    method=EPA_OUTLIER_METHOD,
    iqr_mult=EPA_OUTLIER_IQR_MULT,
    z_thresh=EPA_OUTLIER_Z_THRESH,
    min_valid=20,
):
    """
    Flag outliers in raw EPA observations.

    Returns
    -------
    outlier_mask : np.ndarray (bool)
        True where values are flagged as outliers.
    info : dict
        Summary stats and thresholds used.
    """
    series = pd.to_numeric(df[column], errors="coerce").values
    valid_mask = np.isfinite(series)
    if valid_mask.sum() < min_valid:
        return np.zeros(len(df), dtype=bool), {"reason": "too_few_points"}

    values = series[valid_mask]
    info = {"method": method, "n_valid": int(valid_mask.sum())}

    if method == "iqr":
        q1 = np.nanpercentile(values, 25)
        q3 = np.nanpercentile(values, 75)
        iqr = q3 - q1
        if iqr == 0:
            return np.zeros(len(df), dtype=bool), {"reason": "zero_iqr", **info}
        lower = q1 - iqr_mult * iqr
        upper = q3 + iqr_mult * iqr
        info.update({"q1": q1, "q3": q3, "iqr": iqr, "lower": lower, "upper": upper})
        outliers_valid = (values < lower) | (values > upper)
    elif method == "zscore":
        median = np.nanmedian(values)
        mad = np.nanmedian(np.abs(values - median))
        if mad == 0:
            return np.zeros(len(df), dtype=bool), {"reason": "zero_mad", **info}
        robust_z = 0.6745 * (values - median) / mad
        info.update({"median": median, "mad": mad, "z_thresh": z_thresh})
        outliers_valid = np.abs(robust_z) > z_thresh
    else:
        raise ValueError(f"Unknown EPA outlier method: {method}")

    outlier_mask = np.zeros(len(df), dtype=bool)
    outlier_mask[np.where(valid_mask)[0]] = outliers_valid
    info["n_outliers"] = int(outlier_mask.sum())
    return outlier_mask, info



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


def _get_original_coords(data, scalers):
    """Return coordinates in original scale, handling preprocessed inputs."""
    if isinstance(getattr(data, "metadata", None), dict) and data.metadata.get("preprocessed"):
        return scalers.inverse_transform_coords(data.coords)
    return data.coords


# def create_latitude_diagnostics(data, test_data, y_true_orig, predictions, scalers, output_dir):
#     """
#     Create latitude-based diagnostics plots to check spatial gradients/bias.

#     Outputs:
#         - epa_vs_latitude.png
#         - epa_residuals_vs_latitude.png
#         - source_means_by_latitude.png
#         - source_counts_by_latitude.png
#     """
#     output_dir = Path(output_dir)
#     output_dir.mkdir(parents=True, exist_ok=True)

#     # Use original coordinates for consistent interpretation
#     data_coords_orig = _get_original_coords(data, scalers)
#     test_coords_orig = _get_original_coords(test_data, scalers)

#     # 1) EPA observations vs latitude (scatter)
#     epa_mask = data.source_masks['epa']
#     epa_lat = data_coords_orig[epa_mask][:, 0]
#     epa_vals = data.observations['epa'][epa_mask]

#     # Subsample for readability
#     if len(epa_vals) > 5000:
#         idx = np.random.choice(len(epa_vals), 5000, replace=False)
#         epa_lat = epa_lat[idx]
#         epa_vals = epa_vals[idx]

#     fig, ax = plt.subplots(figsize=(8, 6))
#     ax.scatter(epa_lat, epa_vals, s=10, alpha=0.4, edgecolors='none')
#     ax.set_xlabel("Latitude")
#     ax.set_ylabel("EPA NO₂ (µg/m³)")
#     ax.set_title("EPA Observations vs Latitude")
#     ax.grid(True, alpha=0.3)
#     fig.tight_layout()
#     fig.savefig(output_dir / "epa_vs_latitude.png", dpi=300, bbox_inches="tight")
#     plt.close(fig)

#     # 2) EPA residuals vs latitude (test set)
#     test_epa_mask = test_data.source_masks["epa"]
#     test_lat = test_coords_orig[test_epa_mask][:, 0]
#     test_true = test_data.observations["epa"][test_epa_mask]
#     if scalers.normalize_targets and "epa" in scalers.target_std:
#         test_true = test_true * scalers.target_std["epa"] + scalers.target_mean["epa"]
#     test_pred = predictions.mean[test_epa_mask]
#     residuals = test_true - test_pred

#     if len(residuals) > 5000:
#         idx = np.random.choice(len(residuals), 5000, replace=False)
#         test_lat = test_lat[idx]
#         residuals = residuals[idx]

#     fig, ax = plt.subplots(figsize=(8, 6))
#     ax.scatter(test_lat, residuals, s=10, alpha=0.4, edgecolors='none')
#     ax.axhline(0.0, color="black", linewidth=1.0, alpha=0.6)
#     ax.set_xlabel("Latitude")
#     ax.set_ylabel("EPA Residual (True - Pred)")
#     ax.set_title("EPA Residuals vs Latitude (Test)")
#     ax.grid(True, alpha=0.3)
#     fig.tight_layout()
#     fig.savefig(output_dir / "epa_residuals_vs_latitude.png", dpi=300, bbox_inches="tight")
#     plt.close(fig)

#     # 3) Per-source mean/median by latitude bins
#     n_bins = 12
#     lat_all = data_coords_orig[:, 0]
#     bins = np.quantile(lat_all, np.linspace(0, 1, n_bins + 1))
#     bin_centers = 0.5 * (bins[1:] + bins[:-1])

#     fig, ax = plt.subplots(figsize=(9, 6))
#     for source, color in [("epa", "tab:blue"), ("low_cost", "tab:orange"), ("satellite", "tab:green")]:
#         mask = data.source_masks[source]
#         lat = data_coords_orig[mask][:, 0]
#         vals = data.observations[source][mask]
#         if len(vals) == 0:
#             continue

#         means = []
#         medians = []
#         for i in range(n_bins):
#             bin_mask = (lat >= bins[i]) & (lat <= bins[i + 1])
#             if bin_mask.any():
#                 means.append(np.mean(vals[bin_mask]))
#                 medians.append(np.median(vals[bin_mask]))
#             else:
#                 means.append(np.nan)
#                 medians.append(np.nan)

#         ax.plot(bin_centers, means, color=color, linewidth=2, label=f"{source} mean")
#         ax.plot(bin_centers, medians, color=color, linestyle="--", linewidth=1.5, label=f"{source} median")

#     ax.set_xlabel("Latitude (binned)")
#     ax.set_ylabel("NO₂ (µg/m³)")
#     ax.set_title("Per-Source NO₂ by Latitude Bin")
#     ax.grid(True, alpha=0.3)
#     ax.legend(fontsize=9)
#     fig.tight_layout()
#     fig.savefig(output_dir / "source_means_by_latitude.png", dpi=300, bbox_inches="tight")
#     plt.close(fig)

#     # 4) Observation counts by latitude bins
#     fig, ax = plt.subplots(figsize=(9, 6))
#     width = (bins[1] - bins[0]) * 0.25
#     for i, (source, color, offset) in enumerate([
#         ("epa", "tab:blue", -width),
#         ("low_cost", "tab:orange", 0.0),
#         ("satellite", "tab:green", width),
#     ]):
#         mask = data.source_masks[source]
#         lat = data_coords_orig[mask][:, 0]
#         counts = []
#         for j in range(n_bins):
#             bin_mask = (lat >= bins[j]) & (lat <= bins[j + 1])
#             counts.append(int(bin_mask.sum()))
#         ax.bar(bin_centers + offset, counts, width=width, color=color, alpha=0.7, label=source)

#     ax.set_xlabel("Latitude (binned)")
#     ax.set_ylabel("Observation Count")
#     ax.set_title("Observation Density by Latitude Bin")
#     ax.grid(True, alpha=0.3)
#     ax.legend(fontsize=9)
#     fig.tight_layout()
#     fig.savefig(output_dir / "source_counts_by_latitude.png", dpi=300, bbox_inches="tight")
#     plt.close(fig)


# def save_north_south_summary(data, scalers, output_dir):
#     """
#     Save a north/south split summary table by latitude median.
#     """
#     output_dir = Path(output_dir)
#     output_dir.mkdir(parents=True, exist_ok=True)

#     coords_orig = _get_original_coords(data, scalers)
#     lat = coords_orig[:, 0]
#     lat_split = np.median(lat)

#     records = []
#     for region_name, region_mask in [
#         ("south", lat <= lat_split),
#         ("north", lat > lat_split),
#     ]:
#         for source in ["epa", "low_cost", "satellite"]:
#             src_mask = data.source_masks[source]
#             mask = region_mask & src_mask
#             vals = data.observations[source][mask]
#             if len(vals) == 0:
#                 stats = {"count": 0, "mean": np.nan, "median": np.nan, "std": np.nan}
#             else:
#                 stats = {
#                     "count": int(len(vals)),
#                     "mean": float(np.mean(vals)),
#                     "median": float(np.median(vals)),
#                     "std": float(np.std(vals)),
#                 }
#             records.append({
#                 "region": region_name,
#                 "source": source,
#                 **stats,
#             })

#     summary_df = pd.DataFrame(records)
#     output_path = output_dir / "north_south_summary.csv"
#     summary_df.to_csv(output_path, index=False)
#     return output_path


def save_experiment_summary(experiment_dir, model, trainer, learned_params,
                            epa_metrics, epa_test, scalers, timestamp, elapsed_time=None,
                            training_sources=None, base_vs_epa_metrics=None,
                            base_vs_fused_metrics=None, epa_baseline_metrics=None,
                            epa_bias_corrected_metrics=None,
                            predictor_noise_source=None,
                            prediction_task=None,
                            noise_std_summary=None):
    """Save experiment configuration and results to text file.

    Parameters
    ----------
    experiment_dir : Path
        Directory to save the summary file.
    model : FusionSVGP
        Trained model.
    trainer : Trainer
        Trainer instance with training configuration.
    learned_params : dict
        Learned hyperparameters.
    epa_metrics : EvaluationResults
        Evaluation metrics against EPA ground truth.
    epa_test : np.ndarray
        Test EPA observations (for statistics).
    scalers : Scalers
        Data scalers used for normalization.
    timestamp : str
        Experiment timestamp.
    elapsed_time : float, optional
        Total runtime in seconds.
    training_sources : list, optional
        List of sources used for training (e.g., ['low_cost', 'satellite']).
    """
    summary_file = experiment_dir / "experiment_summary.txt"

    if training_sources is None:
        training_sources = AVAILABLE_SOURCES

    with open(summary_file, 'w') as f:
        f.write(f"FusionGP Paper Reproduction Experiment\n")
        f.write(f"{'='*70}\n")
        f.write(f"Timestamp: {timestamp}\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        if elapsed_time is not None:
            hours, remainder = divmod(elapsed_time, 3600)
            minutes, seconds = divmod(remainder, 60)
            if hours > 0:
                f.write(f"Runtime: {int(hours)}h {int(minutes)}m {seconds:.2f}s ({elapsed_time:.2f} seconds)\n")
            elif minutes > 0:
                f.write(f"Runtime: {int(minutes)}m {seconds:.2f}s ({elapsed_time:.2f} seconds)\n")
            else:
                f.write(f"Runtime: {seconds:.2f} seconds\n")
        f.write(f"\n")

        f.write(f"Methodology:\n")
        f.write(f"{'-'*70}\n")
        f.write(f"Training sources: {', '.join(training_sources)}\n")
        f.write(f"Evaluation target: EPA (held out from training)\n")
        if USE_GRID_PRIOR:
            f.write(f"Prior mean: {GRID_PRIOR_COLUMN} (GridPriorMean, learnable_bias={GRID_PRIOR_LEARNABLE_BIAS})\n")
        else:
            f.write("Prior mean: None\n")
        if 'epa' not in training_sources:
            f.write(f"NOTE: EPA data was NOT used in training to avoid data leakage.\n")
            if HAS_SATELLITE:
                f.write(f"      Model fuses LCS + Satellite, evaluated against independent EPA.\n")
            else:
                f.write(f"      Model fuses LCS only, evaluated against independent EPA.\n")
        f.write(f"\n")

        f.write(f"Model Configuration:\n")
        f.write(f"{'-'*70}\n")
        f.write(f"n_inducing: {model.n_inducing}\n")
        f.write(f"kernel_type: {model.kernel_type}\n")
        f.write(f"n_covariates: {getattr(model, 'n_covariates', 0)}\n")
        f.write(f"learn_inducing_locations: {model.learn_inducing_locations}\n")
        f.write(f"learn_kernel_hyperparams: {MODEL_CONFIG['learn_kernel_hyperparams']}\n")
        f.write(f"learn_noise: {MODEL_CONFIG['learn_noise']}\n")
        f.write(f"learn_calibration: {MODEL_CONFIG['learn_calibration']}\n\n")

        f.write(f"Scaling / Noise Diagnostics:\n")
        f.write(f"{'-'*70}\n")
        f.write(f"normalize_targets: {scalers.normalize_targets}\n")
        if predictor_noise_source is not None:
            f.write(f"predictor_noise_source: {predictor_noise_source}\n")
        if prediction_task is not None:
            f.write(f"prediction_task: {prediction_task}\n")
        if hasattr(scalers, "target_mean") and hasattr(scalers, "target_std"):
            for src, mean_val in scalers.target_mean.items():
                std_val = scalers.target_std.get(src, None)
                if std_val is None:
                    continue
                f.write(f"target_scaler_{src}: mean={mean_val:.4f}, std={std_val:.4f}\n")
        if noise_std_summary:
            for src, val in noise_std_summary.items():
                f.write(f"noise_std_{src}: {val:.4f}\n")
        f.write(f"\n")

        f.write(f"Training Configuration:\n")
        f.write(f"{'-'*70}\n")
        f.write(f"learning_rate: {trainer.learning_rate}\n")
        f.write(f"n_epochs: {trainer.n_epochs}\n")
        f.write(f"batch_size: {trainer.batch_size}\n\n")

        f.write(f"Learned Hyperparameters:\n")
        f.write(f"{'-'*70}\n")
        for key, val in learned_params.items():
            f.write(f"{key}: {val}\n")
        f.write(f"\n")

        f.write(f"Evaluation Metrics (vs EPA ground truth):\n")
        f.write(f"{'-'*70}\n")
        for key, val in epa_metrics.to_dict().items():
            if isinstance(val, (int, float, np.number)):
                f.write(f"{key}: {val:.4f}\n")
            else:
                f.write(f"{key}: {val}\n")
        f.write(f"\n")

        if epa_baseline_metrics is not None:
            f.write(f"EPA Mean Baseline (predict mean EPA at EPA locations):\n")
            f.write(f"{'-'*70}\n")
            for key, val in epa_baseline_metrics.items():
                if isinstance(val, (int, float, np.number)):
                    f.write(f"{key}: {val:.4f}\n")
                else:
                    f.write(f"{key}: {val}\n")
            f.write(f"\n")

        if epa_bias_corrected_metrics is not None:
            f.write(f"EPA Bias-Corrected Metrics (remove mean error):\n")
            f.write(f"{'-'*70}\n")
            for key, val in epa_bias_corrected_metrics.items():
                if isinstance(val, (int, float, np.number)):
                    f.write(f"{key}: {val:.4f}\n")
                else:
                    f.write(f"{key}: {val}\n")
            f.write(f"\n")

        if base_vs_epa_metrics is not None:
            base_dict = base_vs_epa_metrics.to_dict()
            fused_dict = epa_metrics.to_dict()

            def pct_improve(lower_is_better: bool, base_val: float, new_val: float) -> float:
                if base_val == 0:
                    return float('nan')
                if lower_is_better:
                    return 100.0 * (base_val - new_val) / abs(base_val)
                return 100.0 * (new_val - base_val) / abs(base_val)

            rmse_imp = pct_improve(True, base_dict["rmse"], fused_dict["rmse"])
            mae_imp = pct_improve(True, base_dict["mae"], fused_dict["mae"])
            r2_imp = pct_improve(False, base_dict["r2"], fused_dict["r2"])

            f.write(f"{'='*70}\n")
            f.write(f"MODEL COMPARISON SUMMARY (Base → Fused at EPA locations)\n")
            f.write(f"{'='*70}\n")
            f.write(f"  Metric              Base           Fused   Fused vs Base\n")
            f.write(f"  RMSE         {base_dict['rmse']:12.4f} {fused_dict['rmse']:12.4f} {rmse_imp:12.1f}%\n")
            f.write(f"  MAE          {base_dict['mae']:12.4f} {fused_dict['mae']:12.4f} {mae_imp:12.1f}%\n")
            f.write(f"  R²           {base_dict['r2']:12.4f} {fused_dict['r2']:12.4f} {r2_imp:12.1f}%\n")
            f.write(f"  Bias         {base_dict['bias']:12.4f} {fused_dict['bias']:12.4f}\n")
            f.write(f"{'='*70}\n")
            f.write(
                f"\n  Total improvement (Base → Fused at EPA locations):\n"
                f"    RMSE: {rmse_imp:+.1f}%\n"
                f"    R²:   {r2_imp:+.1f}%\n"
            )
            f.write(f"\n")

        f.write(f"Data Statistics:\n")
        f.write(f"{'-'*70}\n")
        f.write(f"Test EPA: mean={epa_test.mean():.4f}, std={epa_test.std():.4f}\n")
        if 'epa' in scalers.target_mean:
            f.write(f"EPA scaler: mean={scalers.target_mean['epa']:.4f}, std={scalers.target_std['epa']:.4f}\n")

    return summary_file


# =============================================================================
# Main Experiment
# =============================================================================

def main():
    # Start timing
    start_time = time.time()

    if USE_EPA_IN_TRAINING:
        raise RuntimeError(
            "EPA must be evaluation-only for this experiment. "
            "Set USE_EPA_IN_TRAINING = False."
        )

    # Set GPyTorch numerical stability settings
    import gpytorch
    # Set higher jitter and max tries globally for numerical stability
    import linear_operator
    linear_operator.settings.cholesky_jitter._global_double_value = 1e-4
    linear_operator.settings.cholesky_max_tries._global_value = 6

    print("="*70)
    print("FusionGP Paper Experiment")
    print("="*70)

    # Create main progress bar for experiment steps
    total_steps = 8
    main_pbar = tqdm(total=total_steps, desc="Experiment Progress", position=0, leave=True,
                     bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} steps [{elapsed}<{remaining}]')

    # -------------------------------------------------------------------------
    # 1. Load and Preprocess Data
    # -------------------------------------------------------------------------
    main_pbar.set_description("Step 1/8: Loading data")
    print("\n[Step 1/8] Loading and preprocessing data...")

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Data file not found at {DATA_PATH}. "
            "Please ensure data exists."
        )

    data_path_for_run = DATA_PATH
    if EPA_OUTLIER_FILTER:
        df_outlier = pd.read_csv(DATA_PATH)
        if "epa_no2" in df_outlier.columns:
            outlier_mask, outlier_info = flag_epa_outliers(df_outlier)
            n_outliers = int(outlier_mask.sum())
            if n_outliers > 0:
                print(f"\n   Detected {n_outliers:,} EPA outliers in raw data ({outlier_info.get('method')}).")
                if EPA_OUTLIER_ACTION == "nan":
                    df_outlier.loc[outlier_mask, "epa_no2"] = np.nan
                elif EPA_OUTLIER_ACTION == "clip":
                    lower = outlier_info.get("lower")
                    upper = outlier_info.get("upper")
                    if lower is not None and upper is not None:
                        df_outlier["epa_no2"] = df_outlier["epa_no2"].clip(lower=lower, upper=upper)
                elif EPA_OUTLIER_ACTION == "drop":
                    df_outlier = df_outlier.loc[~outlier_mask].copy()
                else:
                    raise ValueError(f"Unknown EPA_OUTLIER_ACTION: {EPA_OUTLIER_ACTION}")

                cleaned_path = DATA_PATH.with_name(
                    f"{DATA_PATH.stem}{EPA_OUTLIER_CLEANED_SUFFIX}{DATA_PATH.suffix}"
                )
                df_outlier.to_csv(cleaned_path, index=False)
                data_path_for_run = cleaned_path
                print(f"   ✓ Wrote cleaned EPA data to: {cleaned_path}")
        else:
            print("   Warning: epa_no2 column not found; skipping outlier filter.")

    # Fast schema check to ensure we are using the real dataset and not synthetic/low-cost fields.
    # This guards against accidental regressions if input files change.
    header_cols = pd.read_csv(data_path_for_run, nrows=0).columns
    required_cols = ["grid_id", "latitude", "longitude", "timestamp", GRID_PRIOR_COLUMN]
    epa_col = EPA_INTERP_COLUMN if USE_EPA_INTERPOLATED else "epa_no2"
    required_cols.append(epa_col)
    if HAS_SATELLITE:
        required_cols.append("satellite_no2")
    missing = [c for c in required_cols if c not in header_cols]
    if missing:
        raise ValueError(f"Missing required columns in {data_path_for_run}: {missing}")
    if any(c in header_cols for c in ["synthetic_no2", "base_syhthetic_no2", "low_cost_no2", "low_cost_data"]):
        print("   Note: synthetic/low-cost columns detected in source file but will be ignored.")

    # EPA is evaluation-only; skip interpolated EPA generation.

    loader = DataLoader(
        str(data_path_for_run),
        column_mapping={
            "grid_id": "grid_id",
            "latitude": "latitude",
            "longitude": "longitude",
            "timestamp": "timestamp",
            "satellite": "satellite_no2",
            "epa": EPA_INTERP_COLUMN if USE_EPA_INTERPOLATED else "epa_no2",
        },
        covariate_columns=COVARIATE_COLUMNS,
    )
    data = loader.load()
    # Clean EPA values in-memory only (do not modify source CSV)
    epa_vals = data.observations.get("epa")
    if epa_vals is not None:
        invalid_mask = (epa_vals < 0) | (epa_vals > 100)
        if invalid_mask.any():
            epa_vals = epa_vals.copy()
            epa_vals[invalid_mask] = np.nan
            data.observations["epa"] = epa_vals
            data.source_masks["epa"] = ~np.isnan(epa_vals)
            print(f"   ✓ Dropped {invalid_mask.sum():,} EPA values (<0 or >100) for analysis")
    if getattr(data, "covariates", None) is not None:
        # Normalize covariates to [0,1]; invert distances so closer = higher pollutant.
        covariate_names = data.metadata.get("covariate_names", [])
        if covariate_names:
            covariates = data.covariates.astype(np.float64)
            for i, name in enumerate(covariate_names):
                col = covariates[:, i]
                col_min = np.nanmin(col)
                col_max = np.nanmax(col)
                if col_max == col_min:
                    norm = np.zeros_like(col, dtype=float)
                else:
                    norm = (col - col_min) / (col_max - col_min)
                if name in COVARIATE_DISTANCE_COLUMNS:
                    norm = 1.0 - norm
                covariates[:, i] = norm
            data.covariates = np.nan_to_num(covariates, nan=0.0)

    calibration_params = None
    if PRE_CALIBRATE_SOURCES and not USE_EPA_IN_TRAINING and HAS_SATELLITE:
        print("\n   Pre-calibrating satellite to EPA scale...")
        data, calibration_params = calibrate_sources_to_epa(
            data, sources_to_calibrate=('satellite',)
        )
        print("   ✓ Pre-calibration complete\n")

    if USE_EPA_IN_TRAINING:
        preprocessor = DataPreprocessor(normalize_targets=True)
    else:
        preprocessor = DataPreprocessor(normalize_targets=False)
    train_data, val_data, test_data = preprocessor.fit_transform(data)
    scalers = preprocessor.get_scalers()

    holdout_grid_ids = None
    if USE_EPA_HOLDOUT and "epa" in data.observations:
        rng = np.random.default_rng(HOLDOUT_SEED)
        if HOLDOUT_EPA_BY_GRID and data.grid_ids is not None:
            epa_grids = np.unique(data.grid_ids[data.source_masks["epa"]])
            n_holdout = max(1, int(len(epa_grids) * HOLDOUT_EPA_FRAC))
            holdout_grid_ids = rng.choice(epa_grids, size=n_holdout, replace=False)
        else:
            epa_idx = np.where(data.source_masks["epa"])[0]
            n_holdout = max(1, int(len(epa_idx) * HOLDOUT_EPA_FRAC))
            holdout_grid_ids = (
                np.unique(data.grid_ids[epa_idx]) if data.grid_ids is not None else None
            )
            holdout_indices = rng.choice(epa_idx, size=n_holdout, replace=False)
            if holdout_grid_ids is None:
                holdout_grid_ids = holdout_indices

        if holdout_grid_ids is not None and train_data.grid_ids is not None:
            def _mask_epa_by_grid(fdata):
                if "epa" not in fdata.source_masks:
                    return fdata
                mask = fdata.source_masks["epa"].copy()
                grid_mask = np.isin(fdata.grid_ids, holdout_grid_ids)
                mask[grid_mask] = False
                obs = fdata.observations["epa"].copy()
                obs[grid_mask] = np.nan
                new_obs = fdata.observations.copy()
                new_masks = fdata.source_masks.copy()
                new_obs["epa"] = obs
                new_masks["epa"] = mask
                from src.data.loader import FusionData
                return FusionData(
                    coords=fdata.coords,
                    timestamps=fdata.timestamps,
                    observations=new_obs,
                    source_masks=new_masks,
                    grid_ids=fdata.grid_ids,
                    raw_timestamps=fdata.raw_timestamps if hasattr(fdata, "raw_timestamps") else None,
                    covariates=fdata.covariates,
                    metadata=fdata.metadata,
                )

            train_data = _mask_epa_by_grid(train_data)
            val_data = _mask_epa_by_grid(val_data)

    if RUN_CASE0_EPA_ONLY:
        if not USE_EPA_IN_TRAINING:
            print("   ⚠ Case 0 requires USE_EPA_IN_TRAINING=True; overriding for this run.")
            globals()["USE_EPA_IN_TRAINING"] = True
        training_sources = ["epa"]
        train_data = filter_data_by_sources(train_data, training_sources)
        val_data = filter_data_by_sources(val_data, training_sources)
        model_config = MODEL_CONFIG.copy()
        model_config["sources"] = training_sources
        globals()["USE_MULTITASK_MODEL"] = False
        globals()["PRIMARY_SOURCE"] = "epa"
    else:
        # Downsample satellite in train/val to reduce dominance
        if HAS_SATELLITE:
            train_data = downsample_source(train_data, "satellite", keep_frac=SATELLITE_KEEP_FRAC)
            val_data = downsample_source(val_data, "satellite", keep_frac=SATELLITE_KEEP_FRAC)

        if not USE_EPA_IN_TRAINING:
            training_sources = ["satellite"] if HAS_SATELLITE else []
            train_data = filter_data_by_sources(train_data, training_sources)
            val_data = filter_data_by_sources(val_data, training_sources)
            model_config = MODEL_CONFIG.copy()
            model_config["sources"] = training_sources
            globals()["USE_MULTITASK_MODEL"] = False
            globals()["PRIMARY_SOURCE"] = "satellite" if HAS_SATELLITE else "epa"
        else:
            model_config = MODEL_CONFIG.copy()

    print(f"   ✓ Loaded {len(data.coords)} total observations")
    print(f"   ✓ Train: {len(train_data.coords)}, Val: {len(val_data.coords)}, Test: {len(test_data.coords)}")
    main_pbar.update(1)

    # -------------------------------------------------------------------------
    # 2. Initialize Model
    # -------------------------------------------------------------------------
    main_pbar.set_description("Step 2/8: Initializing model")
    print("\n[Step 2/8] Initializing FusionSVGP model...")

    if getattr(train_data, "covariates", None) is not None:
        model_config["n_covariates"] = train_data.covariates.shape[1]

    # Create prior mean if enabled
    prior_mean = None
    if USE_GRID_PRIOR:
        print(f"   → Loading grid prior from {GRID_PRIOR_COLUMN} column...")
        # Read CSV to get grid background values
        df_prior = pd.read_csv(data_path_for_run)
        # TROPOMI pattern prior disabled; use raw model prior directly.
        # If you re-enable the TROPOMI path, restore the block that builds
        # model_no2_10m_tropomi and sets prior_column accordingly.
        prior_column = GRID_PRIOR_COLUMN
        grid_background = df_prior[df_prior[prior_column].notna()][
            ['grid_id', 'latitude', 'longitude', prior_column]
        ].drop_duplicates('grid_id')

        prior_mean = GridPriorMean(
            grid_coords=grid_background[['latitude', 'longitude']].values,
            grid_values=grid_background[prior_column].values,
            scalers=preprocessor.scalers,
            learnable_bias=GRID_PRIOR_LEARNABLE_BIAS,
        )
        print(f"   ✓ Grid prior loaded: {len(grid_background)} grid points")
        print(f"     Original value range: [{prior_mean.value_range_original[0]:.1f}, {prior_mean.value_range_original[1]:.1f}] µg/m³")
        if prior_mean._values_normalized:
            print(f"     Output value range (normalized): [{prior_mean.value_range[0]:.2f}, {prior_mean.value_range[1]:.2f}]")
        else:
            print(f"     Output value range (original): [{prior_mean.value_range[0]:.1f}, {prior_mean.value_range[1]:.1f}] µg/m³")
        del df_prior, grid_background  # Free memory
    elif USE_ATMO_PLAN_PRIOR and ATMO_PLAN_PATH.exists():
        print(f"   → Loading ATMO-Plan prior from {ATMO_PLAN_PATH.name}...")
        prior_mean = ATMOPlanMean(
            raster_path=ATMO_PLAN_PATH,
            scalers=preprocessor.scalers,
            learnable_bias=ATMO_PLAN_LEARNABLE_BIAS,
        )
        print(f"   ✓ ATMO-Plan prior loaded: {prior_mean.raster_data.shape}")
        print(f"     Value range: [{prior_mean.raster_data[~torch.isnan(prior_mean.raster_data)].min():.1f}, "
              f"{prior_mean.raster_data[~torch.isnan(prior_mean.raster_data)].max():.1f}] µg/m³")
    elif USE_ATMO_PLAN_PRIOR:
        print(f"   ⚠ ATMO-Plan file not found: {ATMO_PLAN_PATH}")
        print(f"     Using constant mean instead")

    if USE_MULTITASK_MODEL:
        model = MultiTaskSVGP(**model_config)
    else:
        model = FusionSVGP(prior_mean=prior_mean, **model_config)
    print(f"   ✓ Model initialized with {model.n_inducing} inducing points")
    print(model)
    main_pbar.update(1)

    # -------------------------------------------------------------------------
    # 3. Train Model
    # -------------------------------------------------------------------------
    main_pbar.set_description("Step 3/8: Training model")
    print("\n[Step 3/8] Training model...")

    trainer = Trainer(
        model,
        callbacks=[
            EarlyStopping(monitor="val_epa_rmse", patience=100, min_delta=1e-4, mode="min"),
        ],
        **TRAINING_CONFIG,
    )
    print(f"   ✓ Training config -> lr={trainer.learning_rate:.2e}, "
          f"epochs={trainer.n_epochs}, batch_size={trainer.batch_size}")

    history = trainer.fit(train_data, val_data=val_data)
    best_epoch_num = (history.best_epoch + 1) * trainer.val_interval
    print(f"   ✓ Training complete. Best validation loss: {history.best_val_loss:.4f} at epoch {best_epoch_num}")
    main_pbar.update(1)

    # -------------------------------------------------------------------------
    # 4. Check Learned Parameters
    # -------------------------------------------------------------------------
    main_pbar.set_description("Step 4/8: Reviewing hyperparameters")
    print("\n[Step 4/8] Reviewing learned hyperparameters...")

    learned_params = model.get_hyperparameters()
    print("\n" + "="*50)
    print("Learned Hyperparameters:")
    print("="*50)
    for key, val in learned_params.items():
        print(f"{key}: {val}")
    print("="*50 + "\n")

    # Print prior mean parameters if using GridPriorMean or similar
    if prior_mean is not None and hasattr(prior_mean, 'get_parameters'):
        prior_params = prior_mean.get_parameters()
        print("="*50)
        print("Prior Mean Parameters:")
        print("="*50)
        for key, val in prior_params.items():
            print(f"  {key}: {val}")
        print("="*50 + "\n")

    print("="*50)
    print("Data Statistics (All Sources):")
    print("="*50)

    # Print statistics for each source in training data
    print("\nTraining data sources:")
    for source in train_data.source_masks.keys():
        mask = train_data.source_masks[source]
        vals = train_data.observations[source][mask]
        if len(vals) == 0:
            print(f"  {source}: n=0")
            continue
        print(f"  {source}: n={mask.sum():,}, mean={vals.mean():.4f}, std={vals.std():.4f}, "
              f"range=[{vals.min():.4f}, {vals.max():.4f}]")

    # Print statistics for test data (includes EPA ground truth)
    print("\nTest data sources (for evaluation):")
    for source in test_data.source_masks.keys():
        mask = test_data.source_masks[source]
        vals = test_data.observations[source][mask]
        if len(vals) == 0:
            print(f"  {source}: n=0")
            continue
        print(f"  {source}: n={mask.sum():,}, mean={vals.mean():.4f}, std={vals.std():.4f}, "
              f"range=[{vals.min():.4f}, {vals.max():.4f}]")

    # Print scaler info
    print("\nScaler statistics:")
    if scalers.normalize_targets:
        for source in scalers.target_mean.keys():
            print(f"  {source}: mean={scalers.target_mean[source]:.4f}, std={scalers.target_std[source]:.4f}")
    else:
        print("  (no target normalization)")

    print("\n" + "-"*50)
    print("SCALE ALIGNMENT CHECK:")
    print("-"*50)
    epa_mask_stats = test_data.source_masks['epa']
    if holdout_grid_ids is not None and test_data.grid_ids is not None:
        epa_mask_stats = epa_mask_stats & np.isin(test_data.grid_ids, holdout_grid_ids)
    epa_vals = test_data.observations['epa'][epa_mask_stats]
    epa_mean = epa_vals.mean()
    for source in train_data.source_masks.keys():
        source_vals = train_data.observations[source][train_data.source_masks[source]]
        source_mean = source_vals.mean()
        diff = source_mean - epa_mean
        print(f"  {source} mean - EPA mean = {diff:.4f} ({source_mean:.4f} vs {epa_mean:.4f})")
    print("-"*50)
    print(f"Train coords range: lat=[{train_data.coords[:,0].min():.4f}, {train_data.coords[:,0].max():.4f}], "
          f"lon=[{train_data.coords[:,1].min():.4f}, {train_data.coords[:,1].max():.4f}]")
    print(f"Train time range: [{train_data.timestamps.min():.4f}, {train_data.timestamps.max():.4f}]")
    print("="*50 + "\n")
    main_pbar.update(1)

    # -------------------------------------------------------------------------
    # 5. Make Predictions
    # -------------------------------------------------------------------------
    main_pbar.set_description("Step 5/8: Making predictions")
    print("\n[Step 5/8] Making predictions...")

    prediction_task = PRIMARY_SOURCE if USE_EPA_IN_TRAINING else None
    predictor = Predictor(model, scalers, include_observation_noise=False, noise_source=PRIMARY_SOURCE)
    predictions = predictor.predict(test_data, task=prediction_task)

    noise_std_summary = None
    try:
        noise_std = model.likelihood.noise_std
        noise_std_summary = {
            src: float(val.detach().cpu().item()) for src, val in noise_std.items()
        }
    except Exception:
        noise_std_summary = None

    y_norm = test_data.observations['epa']
    print(f"   EPA ground truth range/mean: {np.nanmin(y_norm):.4f}, {np.nanmax(y_norm):.4f}, {np.nanmean(y_norm):.4f}")
    print(f"   Prediction mean range/mean: {predictions.mean.min():.4f}, "
          f"{predictions.mean.max():.4f}, {predictions.mean.mean():.4f}")
    print(f"   Prediction std range/mean: {predictions.std.min():.4f}, "
          f"{predictions.std.max():.4f}, {predictions.std.mean():.4f}")
    print(f"   Test data source counts: {dict((k, v.sum()) for k, v in test_data.source_masks.items())}")
    if noise_std_summary:
        print(f"   Learned noise std per source: {noise_std_summary}")

    # Grid predictions for spatial maps using original coordinate ranges
    lat_min, lat_max = scalers.coord_min[0], scalers.coord_max[0]
    lon_min, lon_max = scalers.coord_min[1], scalers.coord_max[1]
    timestamps = np.unique(data.timestamps)

    print(f"   ✓ Using original coordinate ranges:")
    print(f"      Latitude:  [{lat_min:.6f}, {lat_max:.6f}]")
    print(f"      Longitude: [{lon_min:.6f}, {lon_max:.6f}]")

    if USE_ANALYSIS_GRID:
        if USE_PRIOR_GRID_FOR_PREDICTION and USE_GRID_PRIOR:
            print("   ✓ Creating AnalysisGrid from prior grid coordinates...")
            if GEOJSON_GRID_PATH.exists():
                df_prior_grid = load_geojson_grid_centroids(GEOJSON_GRID_PATH)
            else:
                df_prior_grid = pd.read_csv(data_path_for_run)
            lats = np.sort(df_prior_grid["latitude"].unique())
            lons = np.sort(df_prior_grid["longitude"].unique())
            est_points = len(lats) * len(lons)
            if est_points > MAX_ANALYSIS_GRID_POINTS:
                print(
                    f"   ⚠ Prior grid would create {est_points:,} points. "
                    f"Falling back to regular {GRID_RESOLUTION_M}m grid."
                )
                analysis_grid = AnalysisGrid(
                    lon_min=lon_min, lon_max=lon_max,
                    lat_min=lat_min, lat_max=lat_max,
                    resolution_m=GRID_RESOLUTION_M,
                    name='FusionGP_Prediction'
                )
            else:
                analysis_grid = AnalysisGrid.from_latlon_arrays(
                    lats=lats,
                    lons=lons,
                    name='FusionGP_Prediction'
                )
        else:
            print(f"   ✓ Creating AnalysisGrid (resolution={GRID_RESOLUTION_M}m)...")
            analysis_grid = AnalysisGrid(
                lon_min=lon_min, lon_max=lon_max,
                lat_min=lat_min, lat_max=lat_max,
                resolution_m=GRID_RESOLUTION_M,
                name='FusionGP_Prediction'
            )
        print(f"      Grid shape: {analysis_grid.info.shape}")
        print(f"      Grid points: {analysis_grid.info.n_points:,}")

        print(f"   ✓ Generating grid predictions...")
        grid_predictions = predictor.predict_analysis_grid(
            analysis_grid,
            timestamps=timestamps,
            include_covariates=False,  # Covariates already in model
            task=prediction_task,
        )
        print(f"      {grid_predictions.summary()}")
    else:
        print(f"   ✓ Generating grid predictions (resolution={GRID_RESOLUTION})...")
        analysis_grid = None  # Not available in legacy mode
        grid_predictions = predictor.predict_grid(
            lat_range=(lat_min, lat_max),
            lon_range=(lon_min, lon_max),
            timestamps=timestamps,
            resolution=GRID_RESOLUTION,
            task=prediction_task,
        )
    main_pbar.update(1)

    # -------------------------------------------------------------------------
    # 6. Evaluate FusionGP
    # -------------------------------------------------------------------------
    main_pbar.set_description("Step 6/8: Evaluating performance")
    print("\n[Step 6/8] Evaluating FusionGP performance...")

    # Determine which sources were used for training
    training_sources = [
        source for source, mask in train_data.source_masks.items()
        if mask.sum() > 0
    ]
    print(f"   Training sources: {training_sources}")
    if prior_mean is not None:
        print(f"   Prior mean: {GRID_PRIOR_COLUMN} (GridPriorMean, learnable_bias={GRID_PRIOR_LEARNABLE_BIAS})")
    else:
        print("   Prior mean: None")
    print(f"   Evaluation target: EPA (ground truth)")

    evaluator = Evaluator()

    epa_eval_mask = test_data.source_masks["epa"]
    if holdout_grid_ids is not None and test_data.grid_ids is not None:
        epa_eval_mask = epa_eval_mask & np.isin(test_data.grid_ids, holdout_grid_ids)
    epa_test_vals = test_data.observations["epa"][epa_eval_mask]

    if scalers.normalize_targets and 'epa' in scalers.target_std:
        epa_test_orig = epa_test_vals * scalers.target_std['epa'] + scalers.target_mean['epa']
    else:
        epa_test_orig = epa_test_vals

    task = 'epa' if USE_EPA_IN_TRAINING else None
    pred_at_epa = predictor.predict(test_data, task=task, verbose=False)
    pred_mean_at_epa = pred_at_epa.mean[epa_eval_mask]
    pred_std_at_epa = pred_at_epa.std[epa_eval_mask]

    epa_metrics = evaluator.evaluate(
        y_true=epa_test_orig,
        y_pred_mean=pred_mean_at_epa,
        y_pred_std=pred_std_at_epa,
    )

    epa_baseline_mean = float(np.nanmean(epa_test_orig)) if len(epa_test_orig) else np.nan
    epa_baseline_pred = np.full_like(epa_test_orig, epa_baseline_mean)
    epa_baseline_metrics = {
        "rmse": rmse(epa_test_orig, epa_baseline_pred),
        "mae": mae(epa_test_orig, epa_baseline_pred),
        "r2": r_squared(epa_test_orig, epa_baseline_pred),
        "bias": bias(epa_test_orig, epa_baseline_pred),
    }

    bias_correction = float(np.nanmean(epa_test_orig - pred_mean_at_epa)) if len(epa_test_orig) else 0.0
    pred_mean_bias_corrected = pred_mean_at_epa + bias_correction
    epa_bias_corrected_metrics = {
        "rmse": rmse(epa_test_orig, pred_mean_bias_corrected),
        "mae": mae(epa_test_orig, pred_mean_bias_corrected),
        "r2": r_squared(epa_test_orig, pred_mean_bias_corrected),
        "bias": bias(epa_test_orig, pred_mean_bias_corrected),
    }

    print("\n" + "="*50)
    print("Evaluation Results (Fusion vs EPA Ground Truth)")
    print("="*50)
    print(epa_metrics.summary())
    print(f"\nEPA Mean Baseline: {epa_baseline_metrics}")
    print(f"EPA Bias-Corrected Metrics: {epa_bias_corrected_metrics}")

    base_vs_epa_metrics = None
    base_vs_fused_metrics = None
    if prior_mean is not None:
        # Evaluate prior mean (base) against EPA and compare base vs fused.
        covariates = (
            test_data.covariates
            if getattr(test_data, "covariates", None) is not None
            else np.empty((len(test_data.coords), 0))
        )
        x_full = torch.tensor(
            np.column_stack([test_data.coords, test_data.timestamps, covariates]),
            dtype=torch.float32,
            device=next(model.parameters()).device,
        )
        with torch.no_grad():
            base_full = prior_mean(x_full).detach().cpu().numpy()
        base_at_epa = base_full[epa_eval_mask]

        base_vs_epa_metrics = evaluator.evaluate(
            y_true=epa_test_orig,
            y_pred_mean=base_at_epa,
            y_pred_std=np.zeros_like(base_at_epa),
        )

        base_vs_fused_metrics = evaluator.evaluate(
            y_true=base_at_epa,
            y_pred_mean=pred_mean_at_epa,
            y_pred_std=pred_std_at_epa,
        )

        print("\n" + "="*50)
        print("Evaluation Results (Base vs EPA Ground Truth)")
        print("="*50)
        print(base_vs_epa_metrics.summary())

        print("\n" + "="*50)
        print("Evaluation Results (Fused vs Base at EPA locations)")
        print("="*50)
        print(base_vs_fused_metrics.summary())
    else:
        print("   ⚠ Base prior mean not available; skipping base comparisons.")

    if base_vs_epa_metrics is not None:
        base_dict = base_vs_epa_metrics.to_dict()
        fused_dict = epa_metrics.to_dict()
        def pct_improve(lower_is_better: bool, base_val: float, new_val: float) -> float:
            if base_val == 0:
                return float('nan')
            if lower_is_better:
                return 100.0 * (base_val - new_val) / abs(base_val)
            return 100.0 * (new_val - base_val) / abs(base_val)

        rmse_imp = pct_improve(True, base_dict["rmse"], fused_dict["rmse"])
        mae_imp = pct_improve(True, base_dict["mae"], fused_dict["mae"])
        r2_imp = pct_improve(False, base_dict["r2"], fused_dict["r2"])

        print("\n" + "="*70)
        print("MODEL COMPARISON SUMMARY (Base → Fused at EPA locations)")
        print("="*70)
        print(
            f"  Metric              Base           Fused   Fused vs Base"
        )
        print(
            f"  RMSE         {base_dict['rmse']:12.4f} {fused_dict['rmse']:12.4f} {rmse_imp:12.1f}%"
        )
        print(
            f"  MAE          {base_dict['mae']:12.4f} {fused_dict['mae']:12.4f} {mae_imp:12.1f}%"
        )
        print(
            f"  R²           {base_dict['r2']:12.4f} {fused_dict['r2']:12.4f} {r2_imp:12.1f}%"
        )
        print(
            f"  Bias         {base_dict['bias']:12.4f} {fused_dict['bias']:12.4f}"
        )
        print("="*70)
        print(
            f"\n  Total improvement (Base → Fused):\n"
            f"    RMSE: {rmse_imp:+.1f}%\n"
            f"    R²:   {r2_imp:+.1f}%"
        )

    per_source_metrics = {}
    y_true_orig = {}

    for source in test_data.observations.keys():
        mask = test_data.source_masks[source]
        if mask.sum() == 0:
            continue

        obs = test_data.observations[source][mask]
        if scalers.normalize_targets and source in scalers.target_std:
            y_true_orig[source] = obs * scalers.target_std[source] + scalers.target_mean[source]
        else:
            y_true_orig[source] = obs

        task = source if source in training_sources else None
        pred = predictor.predict(test_data, task=task, verbose=False)

        per_source_metrics[source] = evaluator.evaluate(
            y_true=y_true_orig[source],
            y_pred_mean=pred.mean[mask],
            y_pred_std=pred.std[mask],
        ).to_dict()

    print("\nPer-source metrics:")
    for source, source_metrics in per_source_metrics.items():
        trained_on = "(trained)" if source in training_sources else "(held out)"
        print(
            f"  {source} {trained_on}: rmse={source_metrics['rmse']:.4f}, "
            f"mae={source_metrics['mae']:.4f}, r2={source_metrics['r2']:.4f}, "
            f"bias={source_metrics['bias']:.4f}"
        )

    main_pbar.update(1)

    # -------------------------------------------------------------------------
    # 7. Visualize and Save Results
    # -------------------------------------------------------------------------
    main_pbar.set_description("Step 7/8: Saving results")
    print("\n[Step 7/8] Creating visualizations and saving results...")

    # Create timestamped experiment folder with organized subdirectories
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_results_dir = Path(__file__).resolve().parent / "results"
    experiment_dir = base_results_dir / f"experiment_{timestamp}"

    figures_dir = experiment_dir / "figures"
    models_dir = experiment_dir / "models"
    tables_dir = experiment_dir / "tables"

    figures_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"Saving results to: {experiment_dir}")
    print(f"  - Figures: {figures_dir}")
    print(f"  - Models: {models_dir}")
    print(f"  - Tables: {tables_dir}")
    print(f"{'='*70}\n")

    # TROPOMI-augmented prior export disabled.
    # If re-enabled, restore the block that builds model_no2_10m_tropomi
    # and writes prior_augmented_tropomi.csv.

    # Save experiment summary to tables directory (with current runtime)
    current_elapsed = time.time() - start_time
    summary_file = save_experiment_summary(
        experiment_dir=tables_dir,
        model=model,
        trainer=trainer,
        learned_params=learned_params,
        epa_metrics=epa_metrics,
        epa_test=epa_test_orig,
        scalers=scalers,
        timestamp=timestamp,
        elapsed_time=current_elapsed,
        training_sources=training_sources,
        base_vs_epa_metrics=base_vs_epa_metrics,
        base_vs_fused_metrics=base_vs_fused_metrics,
        epa_baseline_metrics=epa_baseline_metrics,
        epa_bias_corrected_metrics=epa_bias_corrected_metrics,
        predictor_noise_source=predictor.noise_source,
        prediction_task=prediction_task,
        noise_std_summary=noise_std_summary,
    )
    print(f"   ✓ Experiment summary saved to: {summary_file}")

    plot_training_curves(
        history=history,
        save_path=str(figures_dir / "training_curves.png")
    )
    print(f"   ✓ Saved training curves plot")


    history_dict = history.to_dict()
    max_len = len(history_dict['train_loss'])
    history_data = {
        'epoch': list(range(1, max_len + 1)),
        'train_loss': history_dict['train_loss'],
        'learning_rate': history_dict['learning_rate'],
        'epoch_time': history_dict['epoch_time'],
    }
    val_interval = max_len // len(history_dict['val_loss']) if len(history_dict['val_loss']) > 1 else 5
    val_losses_full = [None] * max_len
    for i, val_loss in enumerate(history_dict['val_loss']):
        epoch_idx = (i + 1) * val_interval - 1
        if epoch_idx < max_len:
            val_losses_full[epoch_idx] = val_loss
    history_data['val_loss'] = val_losses_full

    history_df = pd.DataFrame(history_data)
    history_df.to_csv(tables_dir / "training_history.csv", index=False)
    print(f"   ✓ Saved training history table")

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

    # EPA time-series diagnostics (top sites by EPA count)
    print("\n   Creating EPA time-series plots...")
    task = "epa" if USE_EPA_IN_TRAINING else None
    predictions_all = predictor.predict(data, task=task, verbose=False)
    plot_epa_timeseries_by_site(
        data=data,
        predictions=predictions_all,
        scalers=scalers,
        output_dir=figures_dir,
        n_sites=5,
    )
    print("   ✓ Saved EPA time-series plots")

    # # North/south summary table
    # summary_path = save_north_south_summary(
    #     data=data,
    #     scalers=scalers,
    #     output_dir=tables_dir,
    # )
    # print(f"   ✓ Saved north/south summary table to: {summary_path}")

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
        task=PRIMARY_SOURCE,
    )
    print(f"   ✓ Saved spatial maps to: {figures_dir / 'spatial_maps'}")

    if USE_ANALYSIS_GRID and analysis_grid is not None:
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

    # -------------------------------------------------------------------------
    # Save Model and Additional Outputs
    # -------------------------------------------------------------------------
    print("\n   Saving model and metrics tables...")

    model_path = models_dir / "fusiongp_model.pth"
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_config': model_config,
        'training_config': TRAINING_CONFIG,
        'learned_hyperparameters': learned_params,
        'timestamp': timestamp,
    }, model_path)
    print(f"   ✓ Saved model to: {model_path}")

    per_source_data = []
    for source in ['epa', 'satellite']:
        if source in per_source_metrics:
            source_metrics = per_source_metrics[source]
            row = {'source': source}
            row.update(source_metrics)
            per_source_data.append(row)

    if per_source_data:
        per_source_df = pd.DataFrame(per_source_data)
        # Save as CSV
        per_source_df.to_csv(tables_dir / "metrics_per_source.csv", index=False)
        # Save as LaTeX
        per_source_df.to_latex(
            tables_dir / "metrics_per_source.tex",
            index=False,
            float_format="%.4f",
            caption="Per-source evaluation metrics",
            label="tab:metrics_per_source"
        )
        print(f"   ✓ Saved per-source metrics table (CSV + LaTeX)")

    epa_metrics_df = pd.DataFrame([epa_metrics.to_dict()])
    # Save as CSV
    epa_metrics_df.to_csv(tables_dir / "metrics_epa_only.csv", index=False)
    # Save as LaTeX
    epa_metrics_df.to_latex(
        tables_dir / "metrics_epa_only.tex",
        index=False,
        float_format="%.4f",
        caption="EPA-only evaluation metrics",
        label="tab:metrics_epa"
    )
    print(f"   ✓ Saved EPA-only metrics table (CSV + LaTeX)")

    # Three-case evaluation summary (EPA vs Base, EPA vs Fused, Base vs Fused)
    cases = [
        {"case": "EPA_vs_Base", "metrics": base_vs_epa_metrics},
        {"case": "EPA_vs_Fused", "metrics": epa_metrics},
        {"case": "Base_vs_Fused", "metrics": base_vs_fused_metrics},
    ]
    case_rows = []
    for item in cases:
        metrics_obj = item["metrics"]
        if metrics_obj is None:
            continue
        row = {"case": item["case"]}
        row.update(metrics_obj.to_dict())
        case_rows.append(row)
    if case_rows:
        case_df = pd.DataFrame(case_rows)
        case_df.to_csv(tables_dir / "metrics_three_cases.csv", index=False)
        print("   ✓ Saved three-case metrics table (CSV)")

    hyperparams_df = pd.DataFrame([learned_params])
    # Save as CSV
    hyperparams_df.to_csv(tables_dir / "learned_hyperparameters.csv", index=False)
    # Save as LaTeX
    hyperparams_df.to_latex(
        tables_dir / "learned_hyperparameters.tex",
        index=False,
        float_format="%.4f",
        caption="Learned model hyperparameters",
        label="tab:hyperparameters"
    )
    print(f"   ✓ Saved learned hyperparameters table (CSV + LaTeX)")

    # -------------------------------------------------------------------------
    # Export Grid Predictions to CSV
    # -------------------------------------------------------------------------
    print("\n   Exporting grid predictions to CSV...")

    df_original = pd.read_csv(data_path_for_run)
    covariate_cols = scalers.covariate_names if hasattr(scalers, "covariate_names") else []
    grid_cols = ['grid_id', 'latitude', 'longitude'] + [c for c in covariate_cols if c in df_original.columns]
    grid_locs = (
        df_original[grid_cols]
        .drop_duplicates()
        .sort_values('grid_id')
        .reset_index(drop=True)
    )

    timestamps_normalized = np.linspace(0, 1, 5)
    timestamps_for_export = scalers.inverse_transform_time(timestamps_normalized)

    all_grid_results = []
    for timestamp in timestamps_for_export:
        coords = grid_locs[['latitude', 'longitude']].values
        covariates = grid_locs[covariate_cols].values if covariate_cols else None
        times = np.full(len(coords), timestamp)

        grid_preds = predictor.predict_locations(
            coords=coords,
            timestamps=times,
            covariates=covariates,
            normalized=False,
            verbose=False,
            task=PRIMARY_SOURCE,
        )

        timestamp_df = grid_locs.copy()
        timestamp_df['timestamp'] = timestamp
        timestamp_df['predicted_mean'] = grid_preds.mean
        timestamp_df['predicted_std'] = grid_preds.std
        timestamp_df['ci_lower_95'] = grid_preds.lower_ci[0.95]
        timestamp_df['ci_upper_95'] = grid_preds.upper_ci[0.95]

        epa_col = EPA_INTERP_COLUMN if USE_EPA_INTERPOLATED else "epa_no2"
        satellite_col = next(
            (c for c in ["satellite_values", "satellite_no2", "satellite"] if c in df_original.columns),
            None,
        )

        obs_cols = ["grid_id", epa_col]
        rename_map = {epa_col: "epa_no2"}
        if satellite_col:
            obs_cols.append(satellite_col)
            rename_map[satellite_col] = "satellite_no2"

        obs_data = df_original[df_original["timestamp"] == timestamp].reindex(columns=obs_cols)
        timestamp_df = timestamp_df.merge(obs_data, on="grid_id", how="left")
        timestamp_df = timestamp_df.rename(columns=rename_map)
        if "satellite_no2" not in timestamp_df.columns:
            timestamp_df["satellite_no2"] = np.nan
        if "epa_no2" not in timestamp_df.columns:
            timestamp_df["epa_no2"] = np.nan

        all_grid_results.append(timestamp_df)

    grid_results_df = pd.concat(all_grid_results, ignore_index=True)
    grid_csv_path = tables_dir / "grid_predictions.csv"
    grid_results_df.to_csv(grid_csv_path, index=False)
    print(f"   ✓ Saved grid predictions to: {grid_csv_path}")
    print(f"      ({len(grid_locs):,} grids × {len(timestamps_for_export)} timestamps = {len(grid_results_df):,} predictions)")
    print(f"      Columns: grid_id, latitude, longitude, timestamp, predicted_mean, predicted_std, ci_lower_95, ci_upper_95, epa_no2, satellite_no2")

    # -------------------------------------------------------------------------
    # Create Grid-Based Visualizations
    # -------------------------------------------------------------------------
    print("\n   Creating grid-based visualizations...")

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

    # -------------------------------------------------------------------------
    # Additional Evaluation Visualizations
    # -------------------------------------------------------------------------
    print("\n   Creating additional evaluation visualizations...")
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

    main_pbar.update(1)

    # -------------------------------------------------------------------------
    # Multi-Case Evaluation (Optional)
    # -------------------------------------------------------------------------
    if RUN_MULTI_CASE_EVALUATION:
        print("\n" + "="*70)
        print("Multi-Case Fusion Evaluation")
        print("="*70)

        full_train_data, full_val_data, _ = preprocessor.fit_transform(data)

        def _mask_epa_by_grid_local(fdata):
            if holdout_grid_ids is None or fdata.grid_ids is None:
                return fdata
            if "epa" not in fdata.source_masks:
                return fdata
            mask = fdata.source_masks["epa"].copy()
            grid_mask = np.isin(fdata.grid_ids, holdout_grid_ids)
            mask[grid_mask] = False
            obs = fdata.observations["epa"].copy()
            obs[grid_mask] = np.nan
            new_obs = fdata.observations.copy()
            new_masks = fdata.source_masks.copy()
            new_obs["epa"] = obs
            new_masks["epa"] = mask
            from src.data.loader import FusionData
            return FusionData(
                coords=fdata.coords,
                timestamps=fdata.timestamps,
                observations=new_obs,
                source_masks=new_masks,
                grid_ids=fdata.grid_ids,
                raw_timestamps=fdata.raw_timestamps if hasattr(fdata, "raw_timestamps") else None,
                covariates=fdata.covariates,
                metadata=fdata.metadata,
            )

        # Apply EPA holdout mask consistently for all cases
        full_train_data = _mask_epa_by_grid_local(full_train_data)
        full_val_data = _mask_epa_by_grid_local(full_val_data)

        print("Training and evaluating 5 fusion scenarios (held-out EPA for evaluation)...")
        print("EPA is used only where specified; no EPA interpolation.")

        cases = build_multi_case_definitions()

        all_case_results = []

        case_model_config = MODEL_CONFIG.copy()
        case_model_config['n_inducing'] = 200  # Reduce for speed
        if getattr(train_data, "covariates", None) is not None:
            case_model_config["n_covariates"] = train_data.covariates.shape[1]

        case_training_config = TRAINING_CONFIG.copy()
        case_training_config['n_epochs'] = 100  # Reduce for speed

        # Build base prior mean (no TROPOMI pattern to avoid double-counting in ablations)
        base_prior_mean = None
        if USE_GRID_PRIOR:
            df_prior_case = pd.read_csv(data_path_for_run)
            grid_background = df_prior_case[df_prior_case[GRID_PRIOR_COLUMN].notna()][
                ["grid_id", "latitude", "longitude", GRID_PRIOR_COLUMN]
            ].drop_duplicates("grid_id")
            base_prior_mean = GridPriorMean(
                grid_coords=grid_background[["latitude", "longitude"]].values,
                grid_values=grid_background[GRID_PRIOR_COLUMN].values,
                scalers=preprocessor.scalers,
                learnable_bias=GRID_PRIOR_LEARNABLE_BIAS,
            )
            del df_prior_case, grid_background

        for case in cases:
            case_name = case["name"]
            sources = case["sources"]
            print(f"\n{'─'*70}")
            print(f"Training {case_name}")
            print(f"{'─'*70}")
            print(f"Sources: {', '.join(sources) if sources else 'prior-only'}")

            case_dir = models_dir / case_name.replace(' ', '_').replace('(', '').replace(')', '')
            case_dir.mkdir(parents=True, exist_ok=True)

            if case["mode"] == "prior_only":
                # Prior-only evaluation without training
                epa_mask_test = test_data.source_masks["epa"]
                x_query = np.column_stack([test_data.coords, test_data.timestamps])
                x_tensor = torch.tensor(x_query, dtype=torch.float32)
                prior_mean_vals = base_prior_mean(x_tensor).detach().cpu().numpy()

                case_pred_mean_orig = prior_mean_vals
                y_true = y_true_orig["epa"][epa_mask_test]
                std_fallback = np.nanstd(y_true) if np.isfinite(y_true).any() else 1.0
                case_pred_std_orig = np.full_like(case_pred_mean_orig, std_fallback)

                case_metrics = evaluator.evaluate(
                    y_true=y_true,
                    y_pred_mean=case_pred_mean_orig[epa_mask_test],
                    y_pred_std=case_pred_std_orig[epa_mask_test],
                )

                unique_times = np.unique(test_data.timestamps)
                rmse_per_time = []
                for t in unique_times:
                    time_mask = (test_data.timestamps == t) & epa_mask_test
                    if time_mask.sum() > 0:
                        time_rmse = np.sqrt(np.mean((y_true_orig["epa"][time_mask] - case_pred_mean_orig[time_mask])**2))
                        rmse_per_time.append(time_rmse)
                    else:
                        rmse_per_time.append(np.nan)

                case_result = {
                    'case_name': case_name,
                    'sources': sources,
                    'overall_rmse': case_metrics.to_dict()['rmse'],
                    'overall_mae': case_metrics.to_dict()['mae'],
                    'overall_r2': case_metrics.to_dict()['r2'],
                    'unique_times': unique_times,
                    'rmse_per_time': np.array(rmse_per_time),
                    'training_time': 0.0,
                }
                all_case_results.append(case_result)

                # Case visuals: scatter + time-series + spatial mean/uncertainty (prior only)
                try:
                    fig, ax = plt.subplots(figsize=(6, 6))
                    ax.scatter(y_true_orig["epa"][epa_mask_test], case_pred_mean_orig[epa_mask_test], s=10, alpha=0.5)
                    lims = [
                        np.nanmin(y_true_orig["epa"][epa_mask_test]),
                        np.nanmax(y_true_orig["epa"][epa_mask_test]),
                    ]
                    ax.plot(lims, lims, 'k--', linewidth=1)
                    ax.set_xlabel("EPA Observed")
                    ax.set_ylabel("Predicted")
                    ax.set_title(f"{case_name}: EPA Observed vs Predicted")
                    fig.tight_layout()
                    fig.savefig(case_dir / "epa_scatter.png", dpi=300, bbox_inches="tight")
                    plt.close(fig)

                    plot_epa_timeseries_by_site(
                        data=data,
                        predictions=type("Pred", (), {"mean": base_prior_mean(
                            torch.tensor(np.column_stack([data.coords, data.timestamps]), dtype=torch.float32)
                        ).detach().cpu().numpy()}),
                        scalers=scalers,
                        output_dir=case_dir,
                        n_sites=5,
                    )

                    if USE_ANALYSIS_GRID and analysis_grid is not None:
                        t0 = timestamps[0] if "timestamps" in locals() else test_data.timestamps.min()
                        lons, lats = analysis_grid.get_flat_coords()
                        coords_grid = np.column_stack([lats, lons])
                        x_grid = np.column_stack([coords_grid, np.full(len(coords_grid), t0)])
                        grid_mean = base_prior_mean(torch.tensor(x_grid, dtype=torch.float32)).detach().cpu().numpy()
                        plot_predictions(
                            coords=coords_grid,
                            values=grid_mean,
                            title=f"{case_name}: Spatial Mean",
                            save_path=case_dir / "spatial_mean.png",
                        )
                        plot_uncertainty(
                            coords=coords_grid,
                            std=np.full_like(grid_mean, std_fallback),
                            title=f"{case_name}: Uncertainty (Std)",
                            save_path=case_dir / "spatial_uncertainty.png",
                        )
                except Exception as exc:
                    print(f"⚠ Case visualizations failed for {case_name}: {exc}")

                # RMSE over time plot
                fig, ax = plt.subplots(figsize=(8, 4))
                ax.plot(np.arange(len(unique_times)), rmse_per_time, color="tab:blue")
                ax.set_xlabel("Time Step")
                ax.set_ylabel("RMSE")
                ax.set_title(f"{case_name}: RMSE Over Time")
                ax.grid(True, alpha=0.3)
                fig.tight_layout()
                fig.savefig(case_dir / "rmse_over_time.png", dpi=300, bbox_inches="tight")
                plt.close(fig)

                print(f"✓ RMSE: {case_result['overall_rmse']:.3f} µg/m³")
                print(f"✓ MAE:  {case_result['overall_mae']:.3f} µg/m³")
                print(f"✓ R²:   {case_result['overall_r2']:.3f}")
                continue

            train_filtered = filter_data_by_sources(full_train_data, sources)
            val_filtered = filter_data_by_sources(full_val_data, sources)

            if "satellite" in sources:
                train_filtered = downsample_source(train_filtered, "satellite", keep_frac=SATELLITE_KEEP_FRAC)
                val_filtered = downsample_source(val_filtered, "satellite", keep_frac=SATELLITE_KEEP_FRAC)

            print(f"Train observations: {len(train_filtered.coords):,}")
            print(f"Val observations:   {len(val_filtered.coords):,}")

            case_model_config_with_sources = case_model_config.copy()
            case_model_config_with_sources['sources'] = sources
            if case["use_prior"]:
                case_model = FusionSVGP(prior_mean=base_prior_mean, **case_model_config_with_sources)
            else:
                case_model = FusionSVGP(**case_model_config_with_sources)
            case_trainer = Trainer(
                model=case_model,
                learning_rate=case_training_config['learning_rate'],
                n_epochs=case_training_config['n_epochs'],
                batch_size=case_training_config['batch_size'],
            )

            case_start = time.time()
            try:
                case_history = case_trainer.fit(train_filtered, val_filtered, verbose=True)
                case_elapsed = time.time() - case_start
                print(f"✓ Training completed in {case_elapsed/60:.2f} minutes")
            except Exception as e:
                print(f"⚠ Training FAILED for {case_name}: {e}")
                print("  Skipping this case and continuing...")
                all_case_results.append({
                    'case': case_name,
                    'sources': ', '.join(sources),
                    'status': 'FAILED',
                    'error': str(e)[:100],
                })
                continue

            case_model_path = case_dir / "model.pth"
            torch.save(case_model.state_dict(), case_model_path)

            case_history_dict = case_history.to_dict()
            max_len = len(case_history_dict['train_loss'])

            val_losses = case_history_dict.get('val_loss', [])
            if len(val_losses) > 0 and len(val_losses) < max_len:
                val_interval = max_len // len(val_losses) if len(val_losses) > 1 else 5
                val_losses_aligned = [None] * max_len
                for i, val_loss in enumerate(val_losses):
                    epoch_idx = (i + 1) * val_interval - 1
                    if epoch_idx < max_len:
                        val_losses_aligned[epoch_idx] = val_loss
            else:
                val_losses_aligned = val_losses if len(val_losses) == max_len else [None] * max_len

            case_history_df = pd.DataFrame({
                'epoch': list(range(1, max_len + 1)),
                'train_loss': case_history_dict['train_loss'],
                'val_loss': val_losses_aligned,
            })
            case_history_df.to_csv(case_dir / "training_history.csv", index=False)

            case_noise_source = sources[0] if sources else "epa"
            case_predictor = Predictor(case_model, noise_source=case_noise_source)


            case_predictions = case_predictor.predict(test_data, verbose=False, task=case_noise_source)

            if scalers.normalize_targets:
                case_pred_mean_orig = case_predictions.mean * scalers.target_std['epa'] + scalers.target_mean['epa']
                case_pred_std_orig = case_predictions.std * scalers.target_std['epa']
            else:
                case_pred_mean_orig = case_predictions.mean
                case_pred_std_orig = case_predictions.std

            epa_mask_test = test_data.source_masks["epa"]
            case_metrics = evaluator.evaluate(
                y_true=y_true_orig["epa"][epa_mask_test],
                y_pred_mean=case_pred_mean_orig[epa_mask_test],
                y_pred_std=case_pred_std_orig[epa_mask_test],
            )

            unique_times = np.unique(test_data.timestamps)
            rmse_per_time = []

            for t in unique_times:
                time_mask = (test_data.timestamps == t) & epa_mask_test
                if time_mask.sum() > 0:
                    time_rmse = np.sqrt(np.mean((y_true_orig["epa"][time_mask] - case_pred_mean_orig[time_mask])**2))
                    rmse_per_time.append(time_rmse)
                else:
                    rmse_per_time.append(np.nan)

            case_result = {
                'case_name': case_name,
                'sources': sources,
                'overall_rmse': case_metrics.to_dict()['rmse'],
                'overall_mae': case_metrics.to_dict()['mae'],
                'overall_r2': case_metrics.to_dict()['r2'],
                'unique_times': unique_times,
                'rmse_per_time': np.array(rmse_per_time),
                'training_time': case_elapsed,
            }
            all_case_results.append(case_result)

            # Case visuals: scatter, time-series, spatial mean/uncertainty
            try:
                fig, ax = plt.subplots(figsize=(6, 6))
                ax.scatter(y_true_orig["epa"][epa_mask_test], case_pred_mean_orig[epa_mask_test], s=10, alpha=0.5)
                lims = [
                    np.nanmin(y_true_orig["epa"][epa_mask_test]),
                    np.nanmax(y_true_orig["epa"][epa_mask_test]),
                ]
                ax.plot(lims, lims, 'k--', linewidth=1)
                ax.set_xlabel("EPA Observed")
                ax.set_ylabel("Predicted")
                ax.set_title(f"{case_name}: EPA Observed vs Predicted")
                fig.tight_layout()
                fig.savefig(case_dir / "epa_scatter.png", dpi=300, bbox_inches="tight")
                plt.close(fig)

                case_predictions_all = case_predictor.predict(data, verbose=False, task=case_noise_source)
                plot_epa_timeseries_by_site(
                    data=data,
                    predictions=case_predictions_all,
                    scalers=scalers,
                    output_dir=case_dir,
                    n_sites=5,
                )

                if USE_ANALYSIS_GRID and analysis_grid is not None:
                    t0 = timestamps[0] if "timestamps" in locals() else test_data.timestamps.min()
                    case_grid_preds = case_predictor.predict_analysis_grid(
                        analysis_grid,
                        timestamp=t0,
                        include_covariates=False,
                        task=case_noise_source,
                    )
                    plot_predictions(
                        coords=case_grid_preds.coords,
                        values=case_grid_preds.mean,
                        title=f"{case_name}: Spatial Mean",
                        save_path=case_dir / "spatial_mean.png",
                    )
                    plot_uncertainty(
                        coords=case_grid_preds.coords,
                        std=case_grid_preds.std,
                        title=f"{case_name}: Uncertainty (Std)",
                        save_path=case_dir / "spatial_uncertainty.png",
                    )
            except Exception as exc:
                print(f"⚠ Case visualizations failed for {case_name}: {exc}")

            fig, ax = plt.subplots(figsize=(8, 4))
            ax.plot(np.arange(len(unique_times)), rmse_per_time, color="tab:blue")
            ax.set_xlabel("Time Step")
            ax.set_ylabel("RMSE")
            ax.set_title(f"{case_name}: RMSE Over Time")
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            fig.savefig(case_dir / "rmse_over_time.png", dpi=300, bbox_inches="tight")
            plt.close(fig)

            print(f"✓ RMSE: {case_result['overall_rmse']:.3f} µg/m³")
            print(f"✓ MAE:  {case_result['overall_mae']:.3f} µg/m³")
            print(f"✓ R²:   {case_result['overall_r2']:.3f}")

        # Create comparison plot
        print(f"\n{'─'*70}")
        print("Creating multi-case comparison plot...")
        print(f"{'─'*70}")

        fig, ax = plt.subplots(figsize=(12, 6))
        colors = ['gray', 'black', 'blue', 'orange', 'red']  # 5 colors for 5 cases
        linestyles = [':', '-', '--', '-.', '-']  # 5 line styles

        for i, result in enumerate(all_case_results):
            time_indices = np.arange(len(result['unique_times']))
            ax.plot(
                time_indices,
                result['rmse_per_time'],
                color=colors[i],
                linestyle=linestyles[i],
                linewidth=2,
                label=result['case_name'],
                marker='o' if i == 0 else None,
                markersize=3,
                alpha=0.8
            )

        ax.set_xlabel('Time Step', fontsize=12)
        ax.set_ylabel('RMSE (µg/m³)', fontsize=12)
        ax.set_title('Predictive Performance Comparison Across Fusion Cases',
                     fontsize=14, fontweight='bold')
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()

        case_plot_path = figures_dir / "multi_case_comparison.png"
        plt.savefig(case_plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ Saved comparison plot to {case_plot_path}")

        # Create summary table
        case_summary_data = []
        for result in all_case_results:
            case_summary_data.append({
                'Case': result['case_name'],
                'Sources': ', '.join(result['sources']),
                'RMSE': f"{result['overall_rmse']:.3f}",
                'MAE': f"{result['overall_mae']:.3f}",
                'R²': f"{result['overall_r2']:.3f}",
                'Mean RMSE (time)': f"{np.nanmean(result['rmse_per_time']):.3f}",
                'Std RMSE (time)': f"{np.nanstd(result['rmse_per_time']):.3f}",
                'Training Time (min)': f"{result['training_time']/60:.2f}",
            })

        case_summary_df = pd.DataFrame(case_summary_data)

        # Save as CSV
        case_summary_path = tables_dir / "multi_case_comparison.csv"
        case_summary_df.to_csv(case_summary_path, index=False)

        # Save as LaTeX
        case_latex_path = tables_dir / "multi_case_comparison.tex"
        case_summary_df.to_latex(
            case_latex_path,
            index=False,
            caption="Multi-case fusion comparison: Predictive performance metrics across different data source combinations. All cases are evaluated on EPA test stations.",
            label="tab:multi_case_comparison",
            column_format="l" + "c" * (len(case_summary_df.columns) - 1),
            escape=False,
            float_format="%.3f"
        )

        print(f"✓ Saved summary table (CSV + LaTeX) to {tables_dir}")

        print(f"\n{case_summary_df.to_string(index=False)}")
        print(f"\n✓ Multi-case evaluation complete!")

    # Close the progress bar
    main_pbar.close()

    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------
    # Calculate elapsed time
    end_time = time.time()
    elapsed_time = end_time - start_time
    hours, remainder = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(remainder, 60)

    print("\n" + "="*70)
    print("Paper Reproduction Experiment Complete!")
    print("="*70)
    print(f"Results directory: {experiment_dir}")
    print(f"\nOrganized outputs:")
    print(f"  Figures: {figures_dir.relative_to(experiment_dir.parent.parent)}")
    print(f"  Models:  {models_dir.relative_to(experiment_dir.parent.parent)}")
    print(f"   Tables:  {tables_dir.relative_to(experiment_dir.parent.parent)}")
    print("\nKey Performance Metrics:")
    print(f"  • RMSE: {epa_metrics.to_dict()['rmse']:.4f} µg/m³")
    print(f"  • MAE:  {epa_metrics.to_dict()['mae']:.4f} µg/m³")
    print(f"  • R²:   {epa_metrics.to_dict()['r2']:.4f}")
    print(f"  • Bias: {epa_metrics.to_dict()['bias']:.4f} µg/m³")
    print("\nTotal Runtime:")
    if hours > 0:
        print(f"  • {int(hours)}h {int(minutes)}m {seconds:.2f}s ({elapsed_time:.2f} seconds)")
    elif minutes > 0:
        print(f"  • {int(minutes)}m {seconds:.2f}s ({elapsed_time:.2f} seconds)")
    else:
        print(f"  • {seconds:.2f} seconds")
    print("="*70)


if __name__ == "__main__":
    main()
