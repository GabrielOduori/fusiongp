"""Prediction generation: pointwise EPA-test predictions and grid/analysis-grid surfaces."""

import logging

import numpy as np
import pandas as pd

from src.data import AnalysisGrid

from . import config
from .data import load_geojson_grid_centroids

logger = logging.getLogger(__name__)


def run_predictions(
    model,
    predictor,
    data,
    test_data,
    scalers,
    prediction_task,
    data_path_for_run,
):
    """
    Generate pointwise test predictions plus a spatial grid of predictions
    for mapping (Step 5 of the original pipeline).

    Returns a dict: predictions, analysis_grid, grid_predictions,
    noise_std_summary, lat_min, lat_max, lon_min, lon_max, timestamps.
    """
    predictions = predictor.predict(test_data, task=prediction_task)

    noise_std_summary = None
    try:
        noise_std = model.likelihood.noise_std
        noise_std_summary = {
            src: float(val.detach().cpu().item()) for src, val in noise_std.items()
        }
    except (AttributeError, TypeError, ValueError) as exc:
        logger.debug("Could not summarize learned noise std: %s", exc)
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

    if config.USE_ANALYSIS_GRID:
        if config.USE_PRIOR_GRID_FOR_PREDICTION and config.USE_GRID_PRIOR:
            print("   ✓ Creating AnalysisGrid from prior grid coordinates...")
            if config.GEOJSON_GRID_PATH.exists():
                df_prior_grid = load_geojson_grid_centroids(config.GEOJSON_GRID_PATH)
            else:
                df_prior_grid = pd.read_csv(data_path_for_run)
            lats = np.sort(df_prior_grid["latitude"].unique())
            lons = np.sort(df_prior_grid["longitude"].unique())
            est_points = len(lats) * len(lons)
            if est_points > config.MAX_ANALYSIS_GRID_POINTS:
                print(
                    f"   ⚠ Prior grid would create {est_points:,} points. "
                    f"Falling back to regular {config.GRID_RESOLUTION_M}m grid."
                )
                analysis_grid = AnalysisGrid(
                    lon_min=lon_min, lon_max=lon_max,
                    lat_min=lat_min, lat_max=lat_max,
                    resolution_m=config.GRID_RESOLUTION_M,
                    name='FusionGP_Prediction'
                )
            else:
                analysis_grid = AnalysisGrid.from_latlon_arrays(
                    lats=lats,
                    lons=lons,
                    name='FusionGP_Prediction'
                )
        else:
            print(f"   ✓ Creating AnalysisGrid (resolution={config.GRID_RESOLUTION_M}m)...")
            analysis_grid = AnalysisGrid(
                lon_min=lon_min, lon_max=lon_max,
                lat_min=lat_min, lat_max=lat_max,
                resolution_m=config.GRID_RESOLUTION_M,
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
        print(f"   ✓ Generating grid predictions (resolution={config.GRID_RESOLUTION})...")
        analysis_grid = None  # Not available in legacy mode
        grid_predictions = predictor.predict_grid(
            lat_range=(lat_min, lat_max),
            lon_range=(lon_min, lon_max),
            timestamps=timestamps,
            resolution=config.GRID_RESOLUTION,
            task=prediction_task,
        )

    return {
        "predictions": predictions,
        "analysis_grid": analysis_grid,
        "grid_predictions": grid_predictions,
        "noise_std_summary": noise_std_summary,
        "lat_min": lat_min,
        "lat_max": lat_max,
        "lon_min": lon_min,
        "lon_max": lon_max,
        "timestamps": timestamps,
    }
