"""Data loading, merging, and source-manipulation helpers."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import config


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


def _normalize_grid_id_series(series):
    if series is None:
        return series
    # Grid IDs are string keys like "24_37"
    s = series.astype("string").str.strip()
    s = s.where(series.notna())
    return s


def _infer_overpass_window(hours: pd.Series, pad_hours: int = 1) -> tuple:
    """
    Infer the satellite overpass hour window from the actual hour distribution
    of its retrieval timestamps, rather than assuming a fixed UTC window.

    A sun-synchronous satellite's overpass UTC hour depends on the site's
    longitude (and varies slightly day to day), so a hardcoded window from one
    deployment (e.g. Dublin) would silently misalign on a different site or
    timestamp timezone. We use the 1st-99th percentile of observed retrieval
    hours (trimming rare stray timestamps) plus a small pad for robustness,
    matching the "window mean over single-hour matching" rationale used here.
    """
    hours = pd.to_numeric(hours, errors="coerce").dropna()
    if hours.empty:
        raise ValueError("No satellite timestamps available to infer overpass window.")
    lo = int(hours.quantile(0.01))
    hi = int(hours.quantile(0.99))
    start = max(0, lo - pad_hours)
    end = min(23, hi + pad_hours)
    return start, end


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

    # Satellite daily mean: use the raw TROPOMI column density (mol/m^2), not
    # tropomi_no2_ug_m3. The ug/m3 column applies a surface-conversion equation
    # we can't verify; gam_ssm_lur instead feeds the raw mol/m^2 retrieval
    # straight into the model and lets calibration (learned here via
    # learn_calibration / sat_slope/sat_intercept) absorb the unit/bias gap to
    # EPA, rather than baking in an unverified conversion upstream.
    sat = pd.read_csv(required_files["satellite"])
    sat["grid_id"] = _normalize_grid_id_series(sat["grid_id"])
    sat["timestamp"] = pd.to_datetime(sat["timestamp"], errors="coerce")
    sat = sat.dropna(subset=["grid_id", "timestamp", "tropomi_no2"])
    sat = sat[sat["tropomi_no2"] > 0]
    sat["date"] = sat["timestamp"].dt.date
    sat["hour"] = sat["timestamp"].dt.hour

    if config.OVERPASS_WINDOW_OVERRIDE is not None:
        overpass_start, overpass_end = config.OVERPASS_WINDOW_OVERRIDE
    else:
        overpass_start, overpass_end = _infer_overpass_window(
            sat["hour"], pad_hours=config.OVERPASS_WINDOW_PAD_HOURS
        )
    print(f"   Overpass window (inferred from satellite retrievals): "
          f"{overpass_start:02d}:00-{overpass_end:02d}:00")

    sat = sat[(sat["hour"] >= overpass_start) & (sat["hour"] <= overpass_end)]
    sat_daily = (
        sat.groupby(["grid_id", "date"], as_index=False)
        .agg({"tropomi_no2": "mean"})
        .rename(columns={"tropomi_no2": "satellite_no2"})
    )

    # EPA overpass-window mean: restrict to the same hours the satellite
    # overpasses occur in (inferred above) so EPA and satellite represent the
    # same time-of-day before fusion.
    epa = pd.read_csv(required_files["epa"])
    epa["grid_id"] = _normalize_grid_id_series(epa["grid_id"])
    epa["timestamp_utc"] = pd.to_datetime(epa["timestamp_utc"], errors="coerce")
    epa = epa.dropna(subset=["grid_id", "timestamp_utc"])
    epa["date"] = epa["timestamp_utc"].dt.date
    epa["hour"] = epa["timestamp_utc"].dt.hour
    epa = epa[(epa["hour"] >= overpass_start) & (epa["hour"] <= overpass_end)]
    epa_daily = epa.groupby(["grid_id", "date"], as_index=False).agg({"epa_no2": "mean"})

    # Traffic overpass-window mean: same window as EPA, for consistency.
    traffic = pd.read_csv(required_files["traffic"])
    traffic["grid_id"] = _normalize_grid_id_series(traffic["grid_id"])
    traffic["traffic_end_time"] = pd.to_datetime(traffic["traffic_end_time"], errors="coerce")
    traffic = traffic.dropna(subset=["grid_id", "traffic_end_time"])
    traffic["date"] = traffic["traffic_end_time"].dt.date
    traffic["hour"] = traffic["traffic_end_time"].dt.hour
    traffic = traffic[(traffic["hour"] >= overpass_start) & (traffic["hour"] <= overpass_end)]
    traffic_daily = (
        traffic.groupby(["grid_id", "date"], as_index=False)
        .agg({"traffic_volume": "mean"})
    )

    # Wind sector features (daily; time-varying)
    wind_path = required_files["wind"]
    if not wind_path.exists() and config.WIND_SECTOR_PATH.exists():
        wind_path = config.WIND_SECTOR_PATH
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
    method=config.EPA_OUTLIER_METHOD,
    iqr_mult=config.EPA_OUTLIER_IQR_MULT,
    z_thresh=config.EPA_OUTLIER_Z_THRESH,
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
