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
import json
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from datetime import datetime
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data import DataLoader, DataPreprocessor
from src.models import FusionSVGP, GridPriorMean
from src.training import Trainer
from src.inference import Predictor
from src.evaluation import Evaluator, rmse, mse, mae, bias


# =============================================================================
# Configuration
# =============================================================================

# Fast run toggle (smoke-test). Set False for full experiment.
FAST_MODE = False

# Use the vetted real dataset (no synthetic/low-cost inputs) to avoid data leakage.
USE_DATA_DIR = Path("/media/gabriel-oduori/SERVER/dev_space/FusionGP/data/model_data")
DATA_PATH = USE_DATA_DIR / "FusionData_daily_merged.csv"
USE_INDIVIDUAL_SOURCES = True  # Use separate source files and merge by grid_id + day

TRAFFIC_PATH = DATA_PATH.with_name("traffic_timeseries.csv")
LUR_PATH = DATA_PATH.with_name("lur_predictions.csv")
ID_MAP_PATH = DATA_PATH.with_name("id_mappings.json")
WIND_SECTOR_PATH = USE_DATA_DIR / "wind_sector_features_era5land_2023-06_daily.csv"

# Overpass window: hours (inclusive) over which EPA and traffic readings are
# averaged before entering the model, matching the TROPOMI overpass period
# (~11:00-13:00 local). Without this, EPA/traffic were averaged over the full
# 24h day while satellite only ever has midday retrievals, comparing a
# whole-day EPA mean against a midday-only satellite mean. Same convention as
# gam_ssm_lur (recommended by thesis examiners there).
OVERPASS_WINDOW_START = 11
OVERPASS_WINDOW_END = 14

# Include traffic (time-varying) as the covariate driving NO2 changes.
WIND_SECTOR_COLUMNS = [
    f"wind_sector_{i}_freq" for i in range(8)
] + [
    f"wind_sector_{i}_mean_speed" for i in range(8)
]
COVARIATE_COLUMNS = ["traffic_volume"] + WIND_SECTOR_COLUMNS
COVARIATE_DISTANCE_COLUMNS = []

MODEL_CONFIG = {
    "n_inducing": 310,
    "spatial_kernel_type": "matern32",
    "temporal_kernel_type": "matern32",  # Smoother than exponential for day-to-day interpolation
    "learn_inducing_locations": True,
    "learn_kernel_hyperparams": True,
    "learn_noise": True,
    "learn_calibration": True,  # Learn satellite→latent linear transform (different units)
    "initial_lengthscales": {"spatial_x": 0.15, "spatial_y": 0.15, "temporal": 0.2},
    "initial_noise": {"epa": 0.3, "satellite": 2.0},
    "sources": ["epa", "satellite"],
}

TRAINING_CONFIG = {
    "learning_rate": 0.005,  # Lower LR for stable convergence with small EPA dataset
    "n_epochs": 150,  # More epochs needed with lower LR and sparse data
    "batch_size": 4096,  # train_loader iterates the full fused dataset (~480k rows), not EPA alone;
    # 256 gave ~1875 batches/epoch and made each epoch take minutes on CPU
    "val_interval": 5,
    "gradient_clip": 1.0,
}

USE_EPA_HOLDOUT = True  # Enable EPA holdout when EPA is in training
HOLDOUT_EPA_FRAC = 0.2
HOLDOUT_EPA_BY_GRID = True
HOLDOUT_SEED = 42

USE_EPA_IN_TRAINING = True  # EPA in training with holdout for evaluation
PRIMARY_SOURCE = "epa"
SATELLITE_KEEP_FRAC = 0.5
HAS_SATELLITE = True
RUN_CASE0_EPA_ONLY = False

# Spatio-temporal cross-validation (EPA grid x time blocks)
USE_SPATIOTEMPORAL_CV = True
CV_GRID_FOLDS = 5
CV_TIME_FOLDS = 4
CV_MAX_FOLDS = 3  # Set to an int to limit total folds for quick runs
CV_TRAIN_RATIO = 0.85
CV_VAL_RATIO = 0.15
CV_RANDOM_SEED = 42

# Fast-mode overrides for quick sanity checks
if FAST_MODE:
    USE_SPATIOTEMPORAL_CV = False
    MODEL_CONFIG["n_inducing"] = 50
    TRAINING_CONFIG["n_epochs"] = 10
    SATELLITE_KEEP_FRAC = 0.1

AVAILABLE_SOURCES = ["epa"] + (["satellite"] if HAS_SATELLITE else [])

EPA_INTERP_COLUMN = "epa_interpolated"
USE_EPA_INTERPOLATED = False
PRE_CALIBRATE_SOURCES = False  # EPA evaluation-only; no calibration
EPA_OUTLIER_FILTER = True
EPA_OUTLIER_METHOD = "iqr"  # "iqr" or "zscore"
EPA_OUTLIER_IQR_MULT = 3.0

# Baseline paths (not merged into dataset)
ATMO_PLAN_PATH = USE_DATA_DIR / "atmos_plan_model_no2.csv"
EPA_OUTLIER_Z_THRESH = 4.0
EPA_OUTLIER_ACTION = "nan"  # "nan", "clip", or "drop"
EPA_OUTLIER_CLEANED_SUFFIX = "_epa_cleaned"

# Prior mean configuration
# Option 1: Use ATMO-Plan as GP prior (corr=0.445 vs LUR corr=0.301 at EPA sites)
USE_GRID_PRIOR = True
ATMO_PLAN_AS_PRIOR = True   # True = use ATMO-Plan; False = use LUR (predicted_no2)
ATMO_PLAN_PRIOR_COLUMN = "model_no2_10m"  # Column in atmos_plan_model_no2.csv
GRID_PRIOR_COLUMN = "predicted_no2"  # Fallback LUR column (used if ATMO_PLAN_AS_PRIOR=False)
GRID_PRIOR_LEARNABLE_BIAS = True
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


def mask_epa_by_grid(data, grid_ids):
    """
    Return a new FusionData with EPA masked to NaN for specified grid IDs.
    """
    if "epa" not in data.source_masks or data.grid_ids is None:
        return data

    from src.data.loader import FusionData

    grid_mask = np.isin(data.grid_ids, grid_ids)
    mask = data.source_masks["epa"].copy()
    mask[grid_mask] = False

    obs = data.observations["epa"].copy()
    obs[grid_mask] = np.nan

    new_obs = data.observations.copy()
    new_masks = data.source_masks.copy()
    new_obs["epa"] = obs
    new_masks["epa"] = mask

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


def subset_fusion_data(data, mask):
    """
    Return a new FusionData with rows selected by mask.
    """
    from src.data.loader import FusionData

    return FusionData(
        coords=data.coords[mask],
        timestamps=data.timestamps[mask],
        observations={k: v[mask] for k, v in data.observations.items()},
        source_masks={k: v[mask] for k, v in data.source_masks.items()},
        grid_ids=data.grid_ids[mask] if data.grid_ids is not None else None,
        raw_timestamps=data.raw_timestamps[mask] if hasattr(data, "raw_timestamps") and data.raw_timestamps is not None else None,
        covariates=data.covariates[mask] if getattr(data, "covariates", None) is not None else None,
        metadata=data.metadata if hasattr(data, "metadata") else {}
    )


def mask_epa_by_grid_time(data, grid_ids, dates):
    """
    Mask EPA observations by grid_id and date (daily).
    """
    if "epa" not in data.source_masks or data.grid_ids is None:
        return data

    from src.data.loader import FusionData

    dates_series = pd.to_datetime(data.raw_timestamps).floor("D")
    grid_mask = np.isin(data.grid_ids, grid_ids)
    time_mask = dates_series.isin(dates)
    mask = data.source_masks["epa"].copy()
    mask = mask & ~(grid_mask & time_mask)

    obs = data.observations["epa"].copy()
    obs[grid_mask & time_mask] = np.nan

    new_obs = data.observations.copy()
    new_masks = data.source_masks.copy()
    new_obs["epa"] = obs
    new_masks["epa"] = mask

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


def build_spatiotemporal_folds(data, n_grid_folds, n_time_folds, seed=42):
    """
    Create spatio-temporal folds based on EPA grid_ids and daily time blocks.
    """
    if "epa" not in data.source_masks or data.grid_ids is None:
        raise ValueError("EPA data with grid_ids is required for spatio-temporal CV.")

    epa_mask = data.source_masks["epa"]
    epa_grids = np.unique(data.grid_ids[epa_mask])
    epa_dates = pd.to_datetime(data.raw_timestamps[epa_mask]).floor("D")
    unique_dates = np.array(sorted(epa_dates.unique()))

    rng = np.random.default_rng(seed)
    rng.shuffle(epa_grids)
    grid_folds = np.array_split(epa_grids, n_grid_folds)

    # Contiguous time blocks
    time_folds = np.array_split(unique_dates, n_time_folds)

    folds = []
    for gi, gfold in enumerate(grid_folds):
        for ti, tfold in enumerate(time_folds):
            folds.append({
                "grid_fold": gi,
                "time_fold": ti,
                "grid_ids": np.array(gfold),
                "dates": np.array(tfold),
            })
    return folds


def split_train_val(data, train_ratio, val_ratio, seed=42):
    """
    Random split of FusionData into train/val using indices.
    """
    if train_ratio + val_ratio <= 0:
        raise ValueError("train_ratio and val_ratio must be positive.")
    total = train_ratio + val_ratio
    train_ratio = train_ratio / total

    n = len(data.coords)
    rng = np.random.default_rng(seed)
    indices = np.arange(n)
    rng.shuffle(indices)

    n_train = max(1, int(n * train_ratio))
    train_idx = indices[:n_train]
    val_idx = indices[n_train:]
    if len(val_idx) == 0:
        val_idx = train_idx[:1]

    train_mask = np.zeros(n, dtype=bool)
    val_mask = np.zeros(n, dtype=bool)
    train_mask[train_idx] = True
    val_mask[val_idx] = True

    return subset_fusion_data(data, train_mask), subset_fusion_data(data, val_mask)




def _normalize_grid_id_series(series):
    if series is None:
        return series
    # Grid IDs are string keys like "24_37"
    s = series.astype("string").str.strip()
    s = s.where(series.notna())
    return s


def export_uq_dataset(
    data,
    output_path: Path,
    scalers,
    covariate_columns=None,
    lur_source_column: str = "predicted_no2",
    data_path_for_lur: Path | None = None,
    add_lur_as_source: bool = False,
):
    """
    Export FusionData into a long-form CSV for external UQ.

    Columns: latitude, longitude, timestamp, value, source, grid_id, [covariates...], [lur_no2]
    Optionally appends LUR rows as a separate source.
    """
    rows = []
    covariate_columns = covariate_columns or []

    for source, mask in data.source_masks.items():
        if mask.sum() == 0:
            continue

        vals = data.observations[source].copy()
        if scalers is not None and getattr(scalers, "normalize_targets", False) and source in scalers.target_std:
            vals = vals * scalers.target_std[source] + scalers.target_mean[source]

        coords = data.coords[mask]
        timestamps = data.timestamps[mask]
        grid_ids = data.grid_ids[mask] if data.grid_ids is not None else [float("nan")] * len(timestamps)

        df = pd.DataFrame({
            "latitude": coords[:, 0],
            "longitude": coords[:, 1],
            "timestamp": timestamps,
            "grid_id": grid_ids,
            "value": vals[mask],
            "source": source,
        })

        if getattr(data, "covariates", None) is not None and len(covariate_columns) == data.covariates.shape[1]:
            cov = data.covariates[mask]
            for i, col in enumerate(covariate_columns):
                df[col] = cov[:, i]

        rows.append(df)

    if not rows:
        return

    out = pd.concat(rows, ignore_index=True)

    if data_path_for_lur is not None and "grid_id" in out.columns:
        try:
            df_lur = pd.read_csv(data_path_for_lur, usecols=["grid_id", lur_source_column])
            df_lur = df_lur.dropna(subset=["grid_id"]).drop_duplicates("grid_id")
            out = out.merge(
                df_lur.rename(columns={lur_source_column: "lur_no2"}),
                on="grid_id",
                how="left",
            )
        except Exception:
            out["lur_no2"] = float("nan")

        if add_lur_as_source and "lur_no2" in out.columns:
            lur_rows = out[out["lur_no2"].notna()].copy()
            lur_rows["value"] = lur_rows["lur_no2"]
            lur_rows["source"] = "lur"
            out = pd.concat([out, lur_rows], ignore_index=True)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_path, index=False)


def build_daily_dataset_from_sources(data_dir: Path) -> Path:
    """
    Build a daily-mean dataset by merging individual source files by grid_id + date.
    """
    required_files = {
        "epa": data_dir / "epa_timeseries.csv",
        "satellite": data_dir / "satellite_retreavals.csv",
        "traffic": data_dir / "traffic_timeseries.csv",
        "lur": data_dir / "lur_predictions.csv",
        "grids": data_dir / "grids_coordinates.csv",
        "wind": data_dir / "wind_sector_features_era5land_2023-06_daily.csv",
    }
    for name, path in required_files.items():
        if name == "wind":
            continue
        if not path.exists():
            raise FileNotFoundError(f"Missing {name} source file: {path}")

    grids = pd.read_csv(required_files["grids"])
    grids["grid_id"] = _normalize_grid_id_series(grids["grid_id"])
    grids = grids.dropna(subset=["grid_id"])

    # EPA overpass-window mean: restrict to the same hours TROPOMI overpasses
    # occur in (OVERPASS_WINDOW_START-OVERPASS_WINDOW_END, inclusive) so EPA
    # and satellite represent the same time-of-day before fusion.
    epa = pd.read_csv(required_files["epa"])
    epa["grid_id"] = _normalize_grid_id_series(epa["grid_id"])
    epa["timestamp_utc"] = pd.to_datetime(epa["timestamp_utc"], errors="coerce")
    epa = epa.dropna(subset=["grid_id", "timestamp_utc"])
    epa["date"] = epa["timestamp_utc"].dt.date
    epa["hour"] = epa["timestamp_utc"].dt.hour
    epa = epa[(epa["hour"] >= OVERPASS_WINDOW_START) & (epa["hour"] <= OVERPASS_WINDOW_END)]
    epa_daily = epa.groupby(["grid_id", "date"], as_index=False).agg({"epa_no2": "mean"})

    # Satellite daily mean (use ug/m3)
    sat = pd.read_csv(required_files["satellite"])
    sat["grid_id"] = _normalize_grid_id_series(sat["grid_id"])
    sat["timestamp"] = pd.to_datetime(sat["timestamp"], errors="coerce")
    sat = sat.dropna(subset=["grid_id", "timestamp"])
    sat["date"] = sat["timestamp"].dt.date
    sat_daily = (
        sat.groupby(["grid_id", "date"], as_index=False)
        .agg({"tropomi_no2_ug_m3": "mean"})
        .rename(columns={"tropomi_no2_ug_m3": "satellite_no2"})
    )

    # Traffic overpass-window mean: same window as EPA, for consistency.
    traffic = pd.read_csv(required_files["traffic"])
    traffic["grid_id"] = _normalize_grid_id_series(traffic["grid_id"])
    traffic["traffic_end_time"] = pd.to_datetime(traffic["traffic_end_time"], errors="coerce")
    traffic = traffic.dropna(subset=["grid_id", "traffic_end_time"])
    traffic["date"] = traffic["traffic_end_time"].dt.date
    traffic["hour"] = traffic["traffic_end_time"].dt.hour
    traffic = traffic[(traffic["hour"] >= OVERPASS_WINDOW_START) & (traffic["hour"] <= OVERPASS_WINDOW_END)]
    traffic_daily = (
        traffic.groupby(["grid_id", "date"], as_index=False)
        .agg({"traffic_volume": "mean"})
    )

    # Wind sector features (daily; time-varying)
    wind_path = required_files["wind"]
    if not wind_path.exists() and WIND_SECTOR_PATH.exists():
        wind_path = WIND_SECTOR_PATH
    wind_daily = None
    if wind_path.exists():
        wind = pd.read_csv(wind_path)
        wind["grid_id"] = _normalize_grid_id_series(wind["grid_id"])
        wind["date"] = pd.to_datetime(wind["date"], errors="coerce").dt.date
        wind = wind.dropna(subset=["grid_id", "date"])
        wind_cols = ["grid_id", "date"] + [c for c in wind.columns if c.startswith("wind_sector_")]
        wind_daily = wind[wind_cols]

    # LUR prior (static)
    lur = pd.read_csv(required_files["lur"])
    lur["grid_id"] = _normalize_grid_id_series(lur["grid_id"])
    lur = lur.dropna(subset=["grid_id"])
    lur = lur[["grid_id", "predicted_no2", "pred_std", "ci_lower_95", "ci_upper_95"]]

    # Build full grid x date base using the union of all source date ranges so that
    # days with EPA or traffic observations but no satellite retrieval (e.g. cloud cover)
    # are still included in the merged dataset.
    all_dates = set(epa_daily["date"].unique()) | set(sat_daily["date"].unique()) | set(traffic_daily["date"].unique())
    if not all_dates:
        raise ValueError("No dates found across any source; cannot build full daily grid.")
    date_index = pd.Index(sorted(all_dates), name="date")
    grid_index = pd.Index(grids["grid_id"].unique(), name="grid_id")
    full_index = pd.MultiIndex.from_product([grid_index, date_index]).to_frame(index=False)

    df = full_index.merge(grids, on="grid_id", how="left")
    # Enforce grid_id -> lat/lon mapping
    df = df.dropna(subset=["latitude", "longitude"])
    # Merge in this order: satellite + LUR prior, then wind + traffic, then EPA
    df = df.merge(sat_daily, on=["grid_id", "date"], how="left")
    df = df.merge(lur, on="grid_id", how="left")
    if wind_daily is not None:
        df = df.merge(wind_daily, on=["grid_id", "date"], how="left")
    df = df.merge(traffic_daily, on=["grid_id", "date"], how="left")
    df["traffic_volume"] = df["traffic_volume"].fillna(0.0)
    df = df.merge(epa_daily, on=["grid_id", "date"], how="left")

    df["timestamp"] = pd.to_datetime(df["date"]).astype(str)
    df = df.drop(columns=["date"])

    out_path = data_dir / "FusionData_daily_merged.csv"
    df.to_csv(out_path, index=False)
    return out_path




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









def save_experiment_summary(experiment_dir, model, trainer, learned_params,
                            epa_metrics, epa_test, scalers, timestamp, elapsed_time=None,
                            training_sources=None, base_vs_epa_metrics=None,
                            base_vs_fused_metrics=None, epa_baseline_metrics=None,
                            epa_bias_corrected_metrics=None,
                            predictor_noise_source=None,
                            prediction_task=None,
                            noise_std_summary=None,
                            lur_baseline_metrics=None,
                            atmo_baseline_metrics=None,
                            eval_counts=None,
                            calibration_metrics=None):
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
        if eval_counts and eval_counts.get("epa_holdout_used", False):
            f.write(f"Evaluation target: EPA (held out from training)\n")
        else:
            f.write(f"Evaluation target: EPA (not held out)\n")
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

        if lur_baseline_metrics is not None or atmo_baseline_metrics is not None:
            def _write_compact_metrics(label, metrics_obj):
                if metrics_obj is None:
                    return
                metrics_dict = metrics_obj.to_dict()
                f.write(f"{label}\n")
                f.write(f"  RMSE: {metrics_dict.get('rmse', np.nan):.4f}\n")
                f.write(f"  MSE:  {metrics_dict.get('mse', np.nan):.4f}\n")
                f.write(f"  MAE:  {metrics_dict.get('mae', np.nan):.4f}\n")
                f.write(f"  Bias: {metrics_dict.get('bias', np.nan):.4f}\n")

            f.write(f"FusionGP vs EPA (headline metrics):\n")
            f.write(f"{'-'*70}\n")
            _write_compact_metrics("FusionGP", epa_metrics)
            _write_compact_metrics("LUR baseline", lur_baseline_metrics)
            _write_compact_metrics("ATMO-Plan baseline", atmo_baseline_metrics)
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

        if lur_baseline_metrics is not None:
            f.write(f"LUR Baseline vs EPA Metrics:\n")
            f.write(f"{'-'*70}\n")
            for key, val in lur_baseline_metrics.to_dict().items():
                if isinstance(val, (int, float, np.number)):
                    f.write(f"{key}: {val:.4f}\n")
                else:
                    f.write(f"{key}: {val}\n")
            f.write(f"\n")

        if atmo_baseline_metrics is not None:
            f.write(f"ATMO-Plan Baseline vs EPA Metrics:\n")
            f.write(f"{'-'*70}\n")
            for key, val in atmo_baseline_metrics.to_dict().items():
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

            f.write(f"{'='*70}\n")
            f.write(f"MODEL COMPARISON SUMMARY (Deviation vs Prior at EPA locations)\n")
            f.write(f"{'='*70}\n")
            f.write(f"  Metric              Base           Fused   Fused vs Base\n")
            f.write(f"  RMSE         {base_dict['rmse']:12.4f} {fused_dict['rmse']:12.4f} {rmse_imp:12.1f}%\n")
            f.write(f"  MAE          {base_dict['mae']:12.4f} {fused_dict['mae']:12.4f} {mae_imp:12.1f}%\n")
            f.write(f"  Bias         {base_dict['bias']:12.4f} {fused_dict['bias']:12.4f}\n")
            f.write(f"{'='*70}\n")
            f.write(
                f"\n  Total change (Prior → Fused at EPA locations):\n"
                f"    RMSE: {rmse_imp:+.1f}%\n"
            )
            f.write(f"\n")

        if calibration_metrics is not None:
            f.write(f"Uncertainty Calibration (EPA holdout only):\n")
            f.write(f"{'-'*70}\n")
            for key, val in calibration_metrics.items():
                if isinstance(val, (int, float, np.number)):
                    f.write(f"{key}: {val:.4f}\n")
                else:
                    f.write(f"{key}: {val}\n")
            f.write(f"\n")

        if eval_counts is not None:
            f.write(f"Evaluation Counts:\n")
            f.write(f"{'-'*70}\n")
            for key, val in eval_counts.items():
                f.write(f"{key}: {val}\n")
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

def run_spatiotemporal_cv(data_full, data_path_for_run, start_time):
    """
    Run spatio-temporal CV with EPA grid x time block holdout.
    """
    print("\n" + "="*70)
    print("Spatio-Temporal CV (EPA grid x time blocks)")
    print("="*70)

    folds = build_spatiotemporal_folds(
        data_full,
        n_grid_folds=CV_GRID_FOLDS,
        n_time_folds=CV_TIME_FOLDS,
        seed=CV_RANDOM_SEED,
    )
    if CV_MAX_FOLDS is not None:
        folds = folds[:CV_MAX_FOLDS]

    results_rows = []
    fold_num = 0

    for fold in folds:
        fold_num += 1
        grid_ids = fold["grid_ids"]
        dates = fold["dates"]

        # Build holdout mask on EPA observations
        dates_series = pd.to_datetime(data_full.raw_timestamps).floor("D")
        epa_mask = data_full.source_masks["epa"]
        holdout_mask = epa_mask & np.isin(data_full.grid_ids, grid_ids) & dates_series.isin(dates)
        holdout_count = int(holdout_mask.sum())
        if holdout_count == 0:
            continue

        print(f"\n[CV Fold {fold_num}/{len(folds)}] grid_fold={fold['grid_fold']}, "
              f"time_fold={fold['time_fold']}, holdout={holdout_count}")

        # Mask EPA holdout for training only
        data_train_masked = mask_epa_by_grid_time(data_full, grid_ids, dates)

        # Fit scalers on training data (no holdout EPA)
        preprocessor = DataPreprocessor(normalize_targets=True)
        preprocessor.fit(data_train_masked)
        scalers = preprocessor.get_scalers()

        data_train_all = preprocessor.transform(data_train_masked)
        data_test_full = preprocessor.transform(data_full)

        # Train/val split on all available observations
        train_data, val_data = split_train_val(
            data_train_all, CV_TRAIN_RATIO, CV_VAL_RATIO, seed=CV_RANDOM_SEED + fold_num
        )

        # Downsample satellite in train/val to reduce dominance
        if HAS_SATELLITE:
            train_data = downsample_source(train_data, "satellite", keep_frac=SATELLITE_KEEP_FRAC)
            val_data = downsample_source(val_data, "satellite", keep_frac=SATELLITE_KEEP_FRAC)

        model_config = MODEL_CONFIG.copy()
        if getattr(train_data, "covariates", None) is not None:
            model_config["n_covariates"] = train_data.covariates.shape[1]

        # Prior mean
        prior_mean = None
        if USE_GRID_PRIOR:
            if ATMO_PLAN_AS_PRIOR and ATMO_PLAN_PATH.exists():
                df_prior = pd.read_csv(ATMO_PLAN_PATH)
                prior_column = ATMO_PLAN_PRIOR_COLUMN
            else:
                df_prior = pd.read_csv(data_path_for_run)
                prior_column = GRID_PRIOR_COLUMN
            grid_background = df_prior[df_prior[prior_column].notna()][
                ['latitude', 'longitude', prior_column]
            ].drop_duplicates(['latitude', 'longitude'])
            prior_mean = GridPriorMean(
                grid_coords=grid_background[['latitude', 'longitude']].values,
                grid_values=grid_background[prior_column].values,
                scalers=preprocessor.scalers,
                learnable_bias=GRID_PRIOR_LEARNABLE_BIAS,
            )
            del df_prior, grid_background

        model = FusionSVGP(prior_mean=prior_mean, **model_config)
        trainer = Trainer(model, **TRAINING_CONFIG)
        history = trainer.fit(train_data, val_data=val_data)
        _ = history

        prediction_task = PRIMARY_SOURCE if USE_EPA_IN_TRAINING else None
        predictor = Predictor(model, scalers, include_observation_noise=False, noise_source=PRIMARY_SOURCE)
        predictions = predictor.predict(data_test_full, task=prediction_task)

        evaluator = Evaluator()

        epa_test_vals = data_test_full.observations["epa"][holdout_mask]
        epa_test_orig = (
            epa_test_vals * scalers.target_std["epa"] + scalers.target_mean["epa"]
            if scalers.normalize_targets and "epa" in scalers.target_std
            else epa_test_vals
        )
        pred_mean_at_epa = (
            predictions.mean[holdout_mask] * scalers.target_std["epa"] + scalers.target_mean["epa"]
            if scalers.normalize_targets and "epa" in scalers.target_std
            else predictions.mean[holdout_mask]
        )
        pred_std_at_epa = (
            predictions.std[holdout_mask] * scalers.target_std["epa"]
            if scalers.normalize_targets and "epa" in scalers.target_std
            else predictions.std[holdout_mask]
        )

        epa_metrics = evaluator.evaluate(
            y_true=epa_test_orig,
            y_pred_mean=pred_mean_at_epa,
            y_pred_std=pred_std_at_epa,
        )
        metrics_dict = epa_metrics.to_dict()
        metrics_dict.update({
            "fold": fold_num,
            "grid_fold": fold["grid_fold"],
            "time_fold": fold["time_fold"],
            "holdout_count": holdout_count,
        })
        results_rows.append(metrics_dict)

    if not results_rows:
        print("No CV folds produced any EPA holdout observations. Check data coverage.")
        return

    # Save CV metrics
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_results_dir = Path(__file__).resolve().parent / "results"
    experiment_dir = base_results_dir / f"cv_experiment_{timestamp}"
    tables_dir = experiment_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    cv_df = pd.DataFrame(results_rows)
    cv_df.to_csv(tables_dir / "cv_metrics_epa.csv", index=False)

    summary_path = tables_dir / "cv_summary.txt"
    with open(summary_path, "w") as f:
        f.write("Spatio-Temporal CV Summary (EPA holdout)\n")
        f.write("="*70 + "\n")
        f.write(f"Folds: {len(results_rows)} (grid_folds={CV_GRID_FOLDS}, time_folds={CV_TIME_FOLDS})\n")
        f.write(f"Holdout per fold: mean={cv_df['holdout_count'].mean():.1f}, "
                f"min={cv_df['holdout_count'].min()}, max={cv_df['holdout_count'].max()}\n")
        for metric in ["rmse", "mae", "bias", "r2"]:
            if metric in cv_df.columns:
                f.write(f"{metric.upper()} mean={cv_df[metric].mean():.4f}, std={cv_df[metric].std():.4f}\n")

    elapsed = time.time() - start_time
    print("\n" + "="*70)
    print("Spatio-Temporal CV Complete")
    print("="*70)
    print(f"Results directory: {experiment_dir}")
    print(f"CV metrics: {tables_dir / 'cv_metrics_epa.csv'}")
    print(f"Summary: {summary_path}")
    print(f"Elapsed time: {elapsed:.2f}s")


def main():
    # Start timing
    start_time = time.time()

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

    if USE_INDIVIDUAL_SOURCES:
        if not USE_DATA_DIR.exists():
            raise FileNotFoundError(
                f"Data directory not found at {USE_DATA_DIR}. "
                "Please ensure data exists."
            )
        data_path_for_run = build_daily_dataset_from_sources(USE_DATA_DIR)
    else:
        if not DATA_PATH.exists():
            raise FileNotFoundError(
                f"Data file not found at {DATA_PATH}. "
                "Please ensure data exists."
            )
        data_path_for_run = DATA_PATH

    if EPA_OUTLIER_FILTER:
        df_outlier = pd.read_csv(data_path_for_run)
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

                cleaned_path = Path(data_path_for_run).with_name(
                    f"{Path(data_path_for_run).stem}{EPA_OUTLIER_CLEANED_SUFFIX}{Path(data_path_for_run).suffix}"
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
    if "traffic" in MODEL_CONFIG.get("sources", []):
        required_cols.append("traffic_volume")
    missing = [c for c in required_cols if c not in header_cols]
    # Allow missing LUR prior/traffic here; they will be merged from external files below.
    if not USE_INDIVIDUAL_SOURCES:
        if GRID_PRIOR_COLUMN in missing and LUR_PATH.exists():
            missing = [c for c in missing if c != GRID_PRIOR_COLUMN]
        if "traffic_volume" in missing and TRAFFIC_PATH.exists():
            missing = [c for c in missing if c != "traffic_volume"]
    if missing:
        raise ValueError(f"Missing required columns in {data_path_for_run}: {missing}")
    if any(c in header_cols for c in ["synthetic_no2", "base_syhthetic_no2", "low_cost_no2", "low_cost_data"]):
        print("   Note: synthetic/low-cost columns detected in source file but will be ignored.")

    if not USE_INDIVIDUAL_SOURCES:
        # Aggregate to daily means before loading into FusionData.
        # This keeps the model time index at daily granularity instead of hourly.
        df_raw = pd.read_csv(data_path_for_run)
        # Drop synthetic columns before aggregation to prevent leakage.
        synth_cols = [c for c in df_raw.columns if "synthetic" in c.lower()]
        # Explicitly drop known unwanted columns.
        drop_cols = synth_cols + [c for c in ["base_syhthetic_no2", "epa_interpolated", "current_rating", "model_no_100m"] if c in df_raw.columns]
        if drop_cols:
            df_raw = df_raw.drop(columns=drop_cols)
            print(f"   ✓ Dropped columns: {drop_cols}")

        # Normalize grid_id to integer indices for consistent merges with LUR/traffic.
        if "grid_id" in df_raw.columns and df_raw["grid_id"].dtype == object and ID_MAP_PATH.exists():
            with ID_MAP_PATH.open("r") as fh:
                id_maps = json.load(fh)
            grid_id_map = id_maps.get("grid_id_map", {})
            df_raw["grid_id"] = df_raw["grid_id"].astype(str).map(grid_id_map)
            before = len(df_raw)
            df_raw = df_raw.dropna(subset=["grid_id"]).copy()
            df_raw["grid_id"] = df_raw["grid_id"].astype(int)
            dropped = before - len(df_raw)
            if dropped > 0:
                print(f"   ⚠ Dropped {dropped:,} rows with unmapped grid_id")
        df_raw["timestamp"] = pd.to_datetime(df_raw["timestamp"], errors="coerce")
        df_raw = df_raw.dropna(subset=["timestamp"]).copy()
        df_raw["date"] = df_raw["timestamp"].dt.floor("D")

        # Aggregate all numeric columns by daily mean per grid cell.
        numeric_cols = df_raw.select_dtypes(include=[np.number]).columns.tolist()
        # Keep grid_id out of aggregation; it is the grouping key.
        numeric_cols = [c for c in numeric_cols if c != "grid_id"]
        agg_dict = {c: "mean" for c in numeric_cols}
        for c in ["latitude", "longitude"]:
            if c in df_raw.columns:
                agg_dict[c] = "first"

        df_daily = df_raw.groupby(["grid_id", "date"]).agg(agg_dict).reset_index()

        # Merge daily mean traffic volumes (grid cells without traffic -> 0.0).
        if TRAFFIC_PATH.exists() and ID_MAP_PATH.exists():
            try:
                with ID_MAP_PATH.open("r") as fh:
                    id_maps = json.load(fh)
                grid_id_map = id_maps.get("grid_id_map", {})

                traffic_df = pd.read_csv(
                    TRAFFIC_PATH,
                    usecols=["grid_id", "traffic_end_time", "traffic_volume"],
                )
                traffic_df["grid_id"] = traffic_df["grid_id"].astype(str).map(grid_id_map)
                traffic_df = traffic_df.dropna(subset=["grid_id"]).copy()
                traffic_df["grid_id"] = traffic_df["grid_id"].astype(int)

                traffic_df["traffic_end_time"] = pd.to_datetime(
                    traffic_df["traffic_end_time"], errors="coerce"
                )
                traffic_df = traffic_df.dropna(subset=["traffic_end_time"]).copy()
                traffic_df["date"] = traffic_df["traffic_end_time"].dt.floor("D")

                traffic_daily = (
                    traffic_df.groupby(["grid_id", "date"], as_index=False)[["traffic_volume"]]
                    .mean()
                )
                df_daily = df_daily.merge(traffic_daily, on=["grid_id", "date"], how="left")
                df_daily["traffic_volume"] = df_daily["traffic_volume"].fillna(0.0)
                print(f"   ✓ Merged daily traffic volumes from {TRAFFIC_PATH}")
            except Exception as exc:
                df_daily["traffic_volume"] = 0.0
                print(f"   ⚠ Traffic merge failed ({exc}); using zeros for traffic_volume")
        else:
            df_daily["traffic_volume"] = 0.0
            if not TRAFFIC_PATH.exists():
                print(f"   ⚠ Traffic file not found at {TRAFFIC_PATH}; using zeros for traffic_volume")
            if not ID_MAP_PATH.exists():
                print(f"   ⚠ ID map not found at {ID_MAP_PATH}; using zeros for traffic_volume")

        # Merge LUR prior (static per grid_id).
        if LUR_PATH.exists() and ID_MAP_PATH.exists():
            try:
                with ID_MAP_PATH.open("r") as fh:
                    id_maps = json.load(fh)
                grid_id_map = id_maps.get("grid_id_map", {})

                lur_df = pd.read_csv(LUR_PATH, usecols=["grid_id", "predicted_no2"])
                lur_df["grid_id"] = lur_df["grid_id"].astype(str).map(grid_id_map)
                lur_df = lur_df.dropna(subset=["grid_id"]).copy()
                lur_df["grid_id"] = lur_df["grid_id"].astype(int)

                df_daily = df_daily.merge(lur_df, on="grid_id", how="left")
                print(f"   ✓ Merged LUR prior from {LUR_PATH}")
            except Exception as exc:
                df_daily["predicted_no2"] = np.nan
                print(f"   ⚠ LUR merge failed ({exc}); predicted_no2 set to NaN")
        else:
            df_daily["predicted_no2"] = np.nan
            if not LUR_PATH.exists():
                print(f"   ⚠ LUR file not found at {LUR_PATH}; predicted_no2 set to NaN")
            if not ID_MAP_PATH.exists():
                print(f"   ⚠ ID map not found at {ID_MAP_PATH}; predicted_no2 set to NaN")

        # Merge wind sector features (daily) if available.
        wind_path = WIND_SECTOR_PATH
        if wind_path.exists() and ID_MAP_PATH.exists():
            try:
                with ID_MAP_PATH.open("r") as fh:
                    id_maps = json.load(fh)
                grid_id_map = id_maps.get("grid_id_map", {})

                wind_df = pd.read_csv(wind_path)
                wind_df["grid_id"] = wind_df["grid_id"].astype(str).map(grid_id_map)
                wind_df = wind_df.dropna(subset=["grid_id"]).copy()
                wind_df["grid_id"] = wind_df["grid_id"].astype(int)
                wind_df["date"] = pd.to_datetime(wind_df["date"], errors="coerce").dt.floor("D")
                wind_df = wind_df.dropna(subset=["date"]).copy()
                wind_cols = ["grid_id", "date"] + [c for c in wind_df.columns if c.startswith("wind_sector_")]
                wind_df = wind_df[wind_cols]

                df_daily = df_daily.merge(wind_df, on=["grid_id", "date"], how="left")
                print(f"   ✓ Merged wind sector features from {wind_path}")
            except Exception as exc:
                print(f"   ⚠ Wind sector merge failed ({exc}); wind columns set to NaN")
                for col in WIND_SECTOR_COLUMNS:
                    if col not in df_daily.columns:
                        df_daily[col] = np.nan
        else:
            if not wind_path.exists():
                print(f"   ⚠ Wind sector file not found at {wind_path}; wind columns set to NaN")
            if not ID_MAP_PATH.exists():
                print(f"   ⚠ ID map not found at {ID_MAP_PATH}; wind columns set to NaN")
            for col in WIND_SECTOR_COLUMNS:
                if col not in df_daily.columns:
                    df_daily[col] = np.nan
        # Convert daily bucket back to timestamp string for the DataLoader.
        df_daily["timestamp"] = pd.to_datetime(df_daily["date"]).astype(str)
        df_daily = df_daily.drop(columns=["date"])

        daily_path = Path(data_path_for_run).with_name(f"{Path(data_path_for_run).stem}_daily.csv")
        df_daily.to_csv(daily_path, index=False)
        data_path_for_run = daily_path
        del df_raw, df_daily

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

    if USE_SPATIOTEMPORAL_CV:
        main_pbar.close()
        run_spatiotemporal_cv(data, data_path_for_run, start_time)
        return

    def to_original_scale(source, values):
        if scalers.normalize_targets and source in scalers.target_std:
            return values * scalers.target_std[source] + scalers.target_mean[source]
        return values

    def to_original_std(source, values):
        if scalers.normalize_targets and source in scalers.target_std:
            return values * scalers.target_std[source]
        return values

    # Normalize targets per source to ~N(0,1) so EPA and satellite are on the same scale.
    preprocessor = DataPreprocessor(normalize_targets=True)
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
            train_data = mask_epa_by_grid(train_data, holdout_grid_ids)
            val_data = mask_epa_by_grid(val_data, holdout_grid_ids)

    train_data_full = train_data
    val_data_full = val_data
    test_data_full = test_data

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
        if ATMO_PLAN_AS_PRIOR and ATMO_PLAN_PATH.exists():
            print(f"   → Loading ATMO-Plan prior from {ATMO_PLAN_PATH.name} ({ATMO_PLAN_PRIOR_COLUMN})...")
            df_prior = pd.read_csv(ATMO_PLAN_PATH)
            prior_column = ATMO_PLAN_PRIOR_COLUMN
        else:
            print(f"   → Loading LUR prior from {GRID_PRIOR_COLUMN} column...")
            df_prior = pd.read_csv(data_path_for_run)
            prior_column = GRID_PRIOR_COLUMN

        grid_background = df_prior[df_prior[prior_column].notna()][
            ['latitude', 'longitude', prior_column]
        ].drop_duplicates(['latitude', 'longitude'])

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
    model = FusionSVGP(prior_mean=prior_mean, **model_config)
    print(f"   ✓ Model initialized with {model.n_inducing} inducing points")
    print(model)
    main_pbar.update(1)

    # -------------------------------------------------------------------------
    # 3. Train Model
    # -------------------------------------------------------------------------
    main_pbar.set_description("Step 3/8: Training model")
    print("\n[Step 3/8] Training model...")

    trainer = Trainer(model, **TRAINING_CONFIG)
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
    predictions_by_task = {}
    def get_predictions(task):
        if task not in predictions_by_task:
            predictions_by_task[task] = predictor.predict(test_data, task=task)
        return predictions_by_task[task]

    predictions = get_predictions(prediction_task)

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

    grid_predictions = None
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
    eval_counts = {
        "epa_eval_count": int(epa_eval_mask.sum()),
        "epa_total_count": int(test_data.source_masks["epa"].sum()),
        "epa_holdout_used": bool(holdout_grid_ids is not None),
        "holdout_grid_count": int(len(holdout_grid_ids)) if holdout_grid_ids is not None else 0,
    }

    epa_test_orig = to_original_scale("epa", epa_test_vals)

    pred_mean_at_epa = to_original_scale("epa", predictions.mean[epa_eval_mask])
    pred_std_at_epa = to_original_std("epa", predictions.std[epa_eval_mask])

    epa_metrics = evaluator.evaluate(
        y_true=epa_test_orig,
        y_pred_mean=pred_mean_at_epa,
        y_pred_std=pred_std_at_epa,
    )
    # Debug: check EPA variance to explain R² behavior.
    print(f"   EPA eval count={len(epa_test_orig)}, mean={np.nanmean(epa_test_orig):.4f}, "
          f"std={np.nanstd(epa_test_orig):.6f}, var={np.nanvar(epa_test_orig):.6f}")

    epa_baseline_mean = float(np.nanmean(epa_test_orig)) if len(epa_test_orig) else np.nan
    epa_baseline_pred = np.full_like(epa_test_orig, epa_baseline_mean)
    epa_baseline_metrics = {
        "rmse": rmse(epa_test_orig, epa_baseline_pred),
        "mse": mse(epa_test_orig, epa_baseline_pred),
        "mae": mae(epa_test_orig, epa_baseline_pred),
        "bias": bias(epa_test_orig, epa_baseline_pred),
    }

    bias_correction = float(np.nanmean(epa_test_orig - pred_mean_at_epa)) if len(epa_test_orig) else 0.0
    pred_mean_bias_corrected = pred_mean_at_epa + bias_correction
    epa_bias_corrected_metrics = {
        "rmse": rmse(epa_test_orig, pred_mean_bias_corrected),
        "mse": mse(epa_test_orig, pred_mean_bias_corrected),
        "mae": mae(epa_test_orig, pred_mean_bias_corrected),
        "bias": bias(epa_test_orig, pred_mean_bias_corrected),
    }

    print("\n" + "="*50)
    print("Evaluation Results (Fusion vs EPA Ground Truth)")
    print("="*50)
    print(epa_metrics.summary())
    print(f"\nEPA Mean Baseline: {epa_baseline_metrics}")
    print(f"EPA Bias-Corrected Metrics: {epa_bias_corrected_metrics}")

    # Uncertainty calibration (EPA holdout only)
    calibration_metrics = None
    if len(epa_test_orig):
        valid_std = pred_std_at_epa > 0
        if np.any(valid_std):
            resid = epa_test_orig[valid_std] - pred_mean_at_epa[valid_std]
            z = resid / pred_std_at_epa[valid_std]
            calibration_metrics = {
                "n_eval": int(len(resid)),
                "coverage_1sigma": float(np.mean(np.abs(z) <= 1.0)),
                "coverage_2sigma": float(np.mean(np.abs(z) <= 2.0)),
                "mean_abs_z": float(np.mean(np.abs(z))),
            }
            print("\nUncertainty calibration (EPA holdout):")
            print(f"  coverage_1sigma: {calibration_metrics['coverage_1sigma']:.4f}")
            print(f"  coverage_2sigma: {calibration_metrics['coverage_2sigma']:.4f}")
            print(f"  mean_abs_z: {calibration_metrics['mean_abs_z']:.4f}")

    base_vs_epa_metrics = None
    base_vs_fused_metrics = None
    lur_baseline_metrics = None
    atmo_baseline_metrics = None
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
        print("Evaluation Results (Deviation vs Base at EPA locations)")
        print("="*50)
        print(base_vs_fused_metrics.summary())
    else:
        print("   ⚠ Base prior mean not available; skipping base comparisons.")

    # LUR baseline (predicted_no2) vs EPA at EPA locations
    try:
        df_baselines = pd.read_csv(data_path_for_run, usecols=["grid_id", "predicted_no2"])
        if test_data.grid_ids is not None:
            baseline_map = df_baselines.dropna(subset=["grid_id"]).drop_duplicates("grid_id").set_index("grid_id")
            lur_at_epa = pd.Series(test_data.grid_ids[epa_eval_mask]).map(baseline_map["predicted_no2"]).to_numpy()
            lur_baseline_metrics = evaluator.evaluate(
                y_true=epa_test_orig,
                y_pred_mean=lur_at_epa,
                y_pred_std=np.zeros_like(lur_at_epa),
            )
            print("\n" + "="*50)
            print("Evaluation Results (LUR Baseline vs EPA Ground Truth)")
            print("="*50)
            print(lur_baseline_metrics.summary())
    except Exception as exc:
        print(f"   ⚠ LUR baseline evaluation skipped ({exc})")

    # ATMO-Plan baseline vs EPA at EPA locations (not merged into dataset)
    if ATMO_PLAN_PATH.exists() and test_data.grid_ids is not None:
        try:
            df_atmo = pd.read_csv(ATMO_PLAN_PATH, usecols=["grid_id", "model_no2_10m"])
            df_atmo = df_atmo.dropna(subset=["grid_id"]).drop_duplicates("grid_id").set_index("grid_id")
            atmo_at_epa = pd.Series(test_data.grid_ids[epa_eval_mask]).map(df_atmo["model_no2_10m"]).to_numpy()
            atmo_baseline_metrics = evaluator.evaluate(
                y_true=epa_test_orig,
                y_pred_mean=atmo_at_epa,
                y_pred_std=np.zeros_like(atmo_at_epa),
            )
            print("\n" + "="*50)
            print("Evaluation Results (ATMO-Plan Baseline vs EPA Ground Truth)")
            print("="*50)
            print(atmo_baseline_metrics.summary())
        except Exception as exc:
            print(f"   ⚠ ATMO-Plan baseline evaluation skipped ({exc})")

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

        print("\n" + "="*70)
        print("MODEL COMPARISON SUMMARY (Deviation vs Prior at EPA locations)")
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
            f"  Bias         {base_dict['bias']:12.4f} {fused_dict['bias']:12.4f}"
        )
        print("="*70)
        print(
            f"\n  Total change (Prior → Fused):\n"
            f"    RMSE: {rmse_imp:+.1f}%\n"
        )

    per_source_metrics = {}
    y_true_orig = {}

    for source in test_data.observations.keys():
        mask = test_data.source_masks[source]
        if mask.sum() == 0:
            continue

        obs = test_data.observations[source][mask]
        y_true_orig[source] = to_original_scale(source, obs)

        task = source if source in training_sources else None
        pred = get_predictions(task)
        per_source_metrics[source] = evaluator.evaluate(
            y_true=y_true_orig[source],
            y_pred_mean=to_original_scale(source, pred.mean[mask]),
            y_pred_std=to_original_std(source, pred.std[mask]),
        ).to_dict()

    print("\nPer-source metrics:")
    for source, source_metrics in per_source_metrics.items():
        trained_on = "(trained)" if source in training_sources else "(held out)"
        print(
            f"  {source} {trained_on}: rmse={source_metrics['rmse']:.4f}, "
            f"mae={source_metrics['mae']:.4f}, "
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

    uq_dir = experiment_dir / "uq"
    uq_dir.mkdir(parents=True, exist_ok=True)

    # Export long-form datasets for downstream UQ
    try:
        export_uq_dataset(
            train_data,
            uq_dir / "uq_train.csv",
            scalers,
            covariate_columns=COVARIATE_COLUMNS,
            data_path_for_lur=data_path_for_run,
            add_lur_as_source=True,
        )
        if val_data is not None:
            export_uq_dataset(
                val_data,
                uq_dir / "uq_val.csv",
                scalers,
                covariate_columns=COVARIATE_COLUMNS,
                data_path_for_lur=data_path_for_run,
                add_lur_as_source=True,
            )
        export_uq_dataset(
            test_data,
            uq_dir / "uq_test.csv",
            scalers,
            covariate_columns=COVARIATE_COLUMNS,
            data_path_for_lur=data_path_for_run,
            add_lur_as_source=True,
        )
        print(f"   ✓ Saved UQ export datasets to: {uq_dir}")
    except Exception as e:
        print(f"   ⚠ Failed to export UQ datasets: {e}")

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
        lur_baseline_metrics=lur_baseline_metrics,
        atmo_baseline_metrics=atmo_baseline_metrics,
        eval_counts=eval_counts,
        calibration_metrics=calibration_metrics,
    )
    print(f"   ✓ Experiment summary saved to: {summary_file}")

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

    # Save prediction results to CSV for external plotting
    # 1. EPA evaluation predictions (observed vs all models)
    epa_pred_df = pd.DataFrame({
        'latitude': predictions.coords[epa_eval_mask, 0],
        'longitude': predictions.coords[epa_eval_mask, 1],
        'timestamp': predictions.timestamps[epa_eval_mask],
        'grid_id': test_data.grid_ids[epa_eval_mask] if test_data.grid_ids is not None else np.nan,
        'observed_epa_no2': epa_test_orig,
        'fusiongp_no2': pred_mean_at_epa,
        'fusiongp_std': pred_std_at_epa,
        'residual': epa_test_orig - pred_mean_at_epa,
    })
    # Merge LUR predictions by grid_id
    try:
        df_lur = pd.read_csv(data_path_for_run, usecols=["grid_id", "predicted_no2"])
        df_lur = df_lur.dropna(subset=["grid_id"]).drop_duplicates("grid_id")
        epa_pred_df = epa_pred_df.merge(
            df_lur.rename(columns={"predicted_no2": "lur_no2"}),
            on="grid_id", how="left",
        )
    except Exception:
        epa_pred_df["lur_no2"] = np.nan
    # Merge ATMO-Plan predictions by grid_id
    try:
        if ATMO_PLAN_PATH.exists():
            df_atmo = pd.read_csv(ATMO_PLAN_PATH, usecols=["grid_id", "model_no2_10m"])
            df_atmo = df_atmo.dropna(subset=["grid_id"]).drop_duplicates("grid_id")
            epa_pred_df = epa_pred_df.merge(
                df_atmo.rename(columns={"model_no2_10m": "atmo_plan_no2"}),
                on="grid_id", how="left",
            )
        else:
            epa_pred_df["atmo_plan_no2"] = np.nan
    except Exception:
        epa_pred_df["atmo_plan_no2"] = np.nan
    epa_pred_df.to_csv(tables_dir / "predictions_epa_eval.csv", index=False)
    print(f"   ✓ Saved EPA evaluation predictions ({len(epa_pred_df)} rows)")

    # 2. Full test set predictions (all sources, all locations, all models)
    prediction_output_source = prediction_task if prediction_task is not None else "epa"
    pred_mean_full = to_original_scale(prediction_output_source, predictions.mean)
    pred_std_full = to_original_std(prediction_output_source, predictions.std)
    pred_lower_full = to_original_scale(prediction_output_source, predictions.lower_ci[0.95])
    pred_upper_full = to_original_scale(prediction_output_source, predictions.upper_ci[0.95])
    full_pred_df = pd.DataFrame({
        'latitude': predictions.coords[:, 0],
        'longitude': predictions.coords[:, 1],
        'timestamp': predictions.timestamps,
        'grid_id': test_data.grid_ids if test_data.grid_ids is not None else np.nan,
        'fusiongp_no2': pred_mean_full,
        'fusiongp_std': pred_std_full,
        'fusiongp_lower_95': pred_lower_full,
        'fusiongp_upper_95': pred_upper_full,
    })
    # Add observed values per source
    for source in test_data.observations:
        mask = test_data.source_masks[source]
        vals = test_data.observations[source].copy()
        if scalers.normalize_targets and source in scalers.target_std:
            vals = vals * scalers.target_std[source] + scalers.target_mean[source]
        vals[~mask] = np.nan
        full_pred_df[f'observed_{source}'] = vals
    # Merge LUR and ATMO-Plan baselines
    try:
        df_lur = pd.read_csv(data_path_for_run, usecols=["grid_id", "predicted_no2"])
        df_lur = df_lur.dropna(subset=["grid_id"]).drop_duplicates("grid_id")
        full_pred_df = full_pred_df.merge(
            df_lur.rename(columns={"predicted_no2": "lur_no2"}),
            on="grid_id", how="left",
        )
    except Exception:
        full_pred_df["lur_no2"] = np.nan
    try:
        if ATMO_PLAN_PATH.exists():
            df_atmo = pd.read_csv(ATMO_PLAN_PATH, usecols=["grid_id", "model_no2_10m"])
            df_atmo = df_atmo.dropna(subset=["grid_id"]).drop_duplicates("grid_id")
            full_pred_df = full_pred_df.merge(
                df_atmo.rename(columns={"model_no2_10m": "atmo_plan_no2"}),
                on="grid_id", how="left",
            )
        else:
            full_pred_df["atmo_plan_no2"] = np.nan
    except Exception:
        full_pred_df["atmo_plan_no2"] = np.nan
    full_pred_df.to_csv(tables_dir / "predictions_full_test.csv", index=False)
    print(f"   ✓ Saved full test predictions ({len(full_pred_df)} rows)")

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

    # Generate all figures
    from plot_results import generate_all_figures
    generate_all_figures(
        figures_dir=figures_dir, history=history,
        epa_test_orig=epa_test_orig, pred_mean_at_epa=pred_mean_at_epa,
        pred_std_at_epa=pred_std_at_epa, epa_metrics=epa_metrics,
        epa_eval_mask=epa_eval_mask, test_data=test_data,
        train_data=train_data, scalers=scalers, predictor=predictor,
        per_source_metrics=per_source_metrics,
        training_sources=training_sources,
        learned_params=learned_params,
        base_vs_epa_metrics=base_vs_epa_metrics,
        data_path=data_path_for_run,
        atmo_plan_path=ATMO_PLAN_PATH,
    )

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
