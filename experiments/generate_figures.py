"""
Figure generation for FusionGP outputs.

Reads all saved CSVs from a completed pipeline run and generates every figure.
Run this independently of the pipeline to regenerate or tweak plots without
re-training.

Figures produced:
  Spatial maps (publication quality):
    - Per-day GPKF mean + uncertainty side-by-side with OSM basemap
    - Static LUR and ATMO-Plan baseline maps
    - SVGP fusion prediction and uncertainty maps
    - Gaussian smoothing, EPA station overlay, percentile-clipped colourmaps

  Diagnostics:
    - Training loss curves (train / val)
    - SVGP calibration and residual analysis
    - GPKF calibration diagnostic
    - Spatial scatter: predicted vs observed station means (all 4 models)

  Time series:
    - Domain-averaged NO₂ over time (GPKF vs Kalman smoother, mean ± 1σ)
    - Daily EPA observed vs FusionSVGP predicted

Usage:
    python experiments/generate_figures.py [run_dir]

If run_dir is omitted, the most-recent outputs/demo_run_* is used.
"""

import sys
import os
import glob
from pathlib import Path
from scipy.stats import norm as scipy_norm

from dotenv import load_dotenv
load_dotenv()  # loads .env from cwd or any parent directory

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.colors import BoundaryNorm, ListedColormap, Normalize
from matplotlib.lines import Line2D
from scipy.ndimage import gaussian_filter
import contextily as ctx
import geopandas as gpd
from shapely.geometry import Point

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
GRID_ROWS = 180
GRID_COLS = 127
SIGMA = float(os.getenv("FUSIONGP_MAP_SIGMA", "0.6"))  # Gaussian smoothing kernel (grid cells)
ALPHA = float(os.getenv("FUSIONGP_MAP_ALPHA", "0.42"))  # heatmap opacity over basemap
CMAP_STD  = "Purples"   # for uncertainty
DPI = 200

# Fixed NO2 classes copied from the gam_ssm_lur publication maps. This keeps
# maps comparable across runs and makes implausible model ranges obvious.
NO2_BOUNDS = [0, 5, 8, 10, 12, 15, 18, 22, 28, 40]
NO2_COLOURS = [
    "#ffffcc", "#c7e9b4", "#7fcdbb", "#41b6c4", "#1d91c0",
    "#225ea8", "#253494", "#081d58", "#4d004b",
]
CMAP_MEAN = ListedColormap(NO2_COLOURS, name="no2_atmos")
NORM_MEAN = BoundaryNorm(NO2_BOUNDS, CMAP_MEAN.N)
CMAP_MEAN.set_under("#f0f0f0")
CMAP_MEAN.set_over("#2b0029")
CMAP_MEAN.set_bad((0, 0, 0, 0))

# Stadia Maps API key — loaded from .env (STADIA_API_KEY=...)
# Falls back to CartoDB.Positron if not set
STADIA_API_KEY = os.getenv("STADIA_API_KEY", "")
USE_BASEMAP = os.getenv("FUSIONGP_BASEMAP", "1").lower() not in {"0", "false", "no"}


# ---------------------------------------------------------------------------
# Helper: locate run directory
# ---------------------------------------------------------------------------
def find_run_dir(arg=None):
    if arg:
        return Path(arg)
    candidates = sorted(
        glob.glob("outputs/demo_run_*"),
        key=os.path.getmtime
    )
    if not candidates:
        raise FileNotFoundError("No outputs/demo_run_* directories found.")
    return Path(candidates[-1])


# ---------------------------------------------------------------------------
# Helper: reshape flat CSV to 2-D grid using grid_id col (row_col)
# ---------------------------------------------------------------------------
def csv_to_grid(df, col, rows=GRID_ROWS, cols=GRID_COLS):
    """Return (rows, cols) array; NaN where grid_id absent."""
    ids = df["grid_id"].str.split("_", expand=True).astype(int)
    grid = np.full((rows, cols), np.nan)
    grid[ids[0].values, ids[1].values] = df[col].values
    return grid


def valid_value_mask_from_csv(csv_path, value_col="value", rows=GRID_ROWS, cols=GRID_COLS):
    """Return True where the source grid has an actual finite value."""
    df = pd.read_csv(csv_path)
    if "grid_id" not in df.columns or value_col not in df.columns:
        raise ValueError(f"{csv_path} must contain grid_id and {value_col}")

    ids = df["grid_id"].str.split("_", expand=True).astype(int)
    valid = pd.to_numeric(df[value_col], errors="coerce").notna().to_numpy()
    mask = np.zeros((rows, cols), dtype=bool)
    mask[ids[0].values, ids[1].values] = valid
    return mask


# ---------------------------------------------------------------------------
# Helper: lat/lon arrays from one CSV
# ---------------------------------------------------------------------------
def grid_coords_arrays(df, rows=GRID_ROWS, cols=GRID_COLS):
    ids = df["grid_id"].str.split("_", expand=True).astype(int)
    lat_grid = np.full((rows, cols), np.nan)
    lon_grid = np.full((rows, cols), np.nan)
    lat_grid[ids[0].values, ids[1].values] = df["latitude"].values
    lon_grid[ids[0].values, ids[1].values] = df["longitude"].values
    return lat_grid, lon_grid


# ---------------------------------------------------------------------------
# Helper: convert 2-D lat/lon grids to Web Mercator extent
# ---------------------------------------------------------------------------
def latlon_grid_to_3857_extent(lat_grid, lon_grid):
    """
    Returns (xmin, xmax, ymin, ymax) in EPSG:3857 for use with imshow extent.

    The CSV coordinates are grid-cell centres. imshow's extent is interpreted as
    image edges, so using the centre min/max directly shifts the raster by about
    half a cell relative to the basemap. Pad by half the median projected cell
    spacing to align cells with roads and coastlines.
    """
    lat_flat = lat_grid.ravel()
    lon_flat = lon_grid.ravel()
    mask = ~np.isnan(lat_flat)
    pts = gpd.GeoDataFrame(
        {"lat": lat_flat[mask], "lon": lon_flat[mask]},
        geometry=[Point(lo, la) for lo, la in
                  zip(lon_flat[mask], lat_flat[mask])],
        crs="EPSG:4326",
    ).to_crs("EPSG:3857")
    x = pts.geometry.x.values
    y = pts.geometry.y.values

    dx = dy = 0.0
    if lat_grid.shape[0] > 1:
        row_x = []
        for i in range(lat_grid.shape[0]):
            valid = ~np.isnan(lat_grid[i, :]) & ~np.isnan(lon_grid[i, :])
            if valid.any():
                row_pts = gpd.GeoDataFrame(
                    geometry=[Point(lo, la) for lo, la in zip(lon_grid[i, valid], lat_grid[i, valid])],
                    crs="EPSG:4326",
                ).to_crs("EPSG:3857")
                row_x.append(float(row_pts.geometry.x.mean()))
        if len(row_x) > 1:
            dx = float(np.nanmedian(np.abs(np.diff(np.sort(row_x)))))

    if lat_grid.shape[1] > 1:
        col_y = []
        for j in range(lat_grid.shape[1]):
            valid = ~np.isnan(lat_grid[:, j]) & ~np.isnan(lon_grid[:, j])
            if valid.any():
                col_pts = gpd.GeoDataFrame(
                    geometry=[Point(lo, la) for lo, la in zip(lon_grid[valid, j], lat_grid[valid, j])],
                    crs="EPSG:4326",
                ).to_crs("EPSG:3857")
                col_y.append(float(col_pts.geometry.y.mean()))
        if len(col_y) > 1:
            dy = float(np.nanmedian(np.abs(np.diff(np.sort(col_y)))))

    return x.min() - dx / 2, x.max() + dx / 2, y.min() - dy / 2, y.max() + dy / 2


# ---------------------------------------------------------------------------
# Build station GeoDataFrames (EPA only — satellite is raster, not point obs)
# ---------------------------------------------------------------------------
def load_epa_stations(predictions_csv):
    df = pd.read_csv(predictions_csv)
    epa = df[df["is_epa"] == True][["latitude", "longitude"]].drop_duplicates()
    gdf = gpd.GeoDataFrame(
        epa,
        geometry=[Point(lo, la) for lo, la in
                  zip(epa["longitude"], epa["latitude"])],
        crs="EPSG:4326",
    ).to_crs("EPSG:3857")
    return gdf


# ---------------------------------------------------------------------------
# Single-panel plot (used inside combined figure)
# ---------------------------------------------------------------------------
def render_panel(ax, grid_smooth, title, cmap, xmin, xmax, ymin, ymax,
                 epa_gdf, unit_label, vmin=None, vmax=None,
                 image_extent=None):
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)

    # Basemap
    if USE_BASEMAP:
        try:
            if STADIA_API_KEY:
                source = f"https://tiles.stadiamaps.com/tiles/stamen_toner_lite/{{z}}/{{x}}/{{y}}.png?api_key={STADIA_API_KEY}"
            else:
                source = ctx.providers.CartoDB.Positron
            ctx.add_basemap(ax, source=source, crs="EPSG:3857", zoom="auto", timeout=8)
        except (OSError, RuntimeError, ValueError) as exc:
            print(f"Skipping basemap for {title}: {exc}")

    if cmap is CMAP_MEAN:
        norm = NORM_MEAN
        vmin = vmax = None
        extend = "both"
    else:
        # Percentile-clipped colour scale for uncertainty maps.
        if vmin is None:
            vmin = float(np.nanpercentile(grid_smooth, 2))
        if vmax is None:
            vmax = float(np.nanpercentile(grid_smooth, 98))
        norm = Normalize(vmin=vmin, vmax=vmax)
        extend = "neither"

    # grid_id is row_col, but in this Dublin grid row indexes longitude
    # (west-east) and col indexes latitude (south-north). imshow expects
    # array rows on y and columns on x, so transpose before drawing.
    im = ax.imshow(
        grid_smooth.T,
        extent=image_extent or (xmin, xmax, ymin, ymax),
        origin="lower",
        cmap=cmap,
        norm=norm,
        alpha=ALPHA,
        zorder=2,
    )

    # EPA station markers
    ax.scatter(
        epa_gdf.geometry.x, epa_gdf.geometry.y,
        s=60, c="white", edgecolors="black", linewidths=1.2,
        zorder=5, label="EPA station",
    )

    # Colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.75, pad=0.02, extend=extend)
    cbar.set_label(unit_label, fontsize=9)  # NO₂ µg/m³
    cbar.ax.tick_params(labelsize=8)

    ax.set_title(title, fontsize=11, weight="bold", pad=6)

    # Geographic tick labels (convert 3857 back to lon/lat)
    _add_geo_ticks(ax, xmin, xmax, ymin, ymax)


def warn_if_implausible_map_values(model_name, frames):
    mean_vals = []
    std_vals = []
    for df in frames.values():
        if "mean_ug_m3" in df.columns:
            mean_vals.append(pd.to_numeric(df["mean_ug_m3"], errors="coerce").to_numpy())
        if "std_ug_m3" in df.columns:
            std_vals.append(pd.to_numeric(df["std_ug_m3"], errors="coerce").to_numpy())
    if not mean_vals:
        return

    means = np.concatenate(mean_vals)
    means = means[np.isfinite(means)]
    if len(means) == 0:
        return
    p2, p50, p98 = np.nanpercentile(means, [2, 50, 98])
    if p2 < -5 or p98 > 80:
        print(
            f"WARNING: {model_name} mean map range looks implausible "
            f"(p2={p2:.1f}, median={p50:.1f}, p98={p98:.1f} µg/m³). "
            "Check model calibration before using these figures."
        )

    if std_vals:
        stds = np.concatenate(std_vals)
        stds = stds[np.isfinite(stds)]
        if len(stds) > 0:
            s50, s98 = np.nanpercentile(stds, [50, 98])
            if s50 > 50 or s98 > 100:
                print(
                    f"WARNING: {model_name} uncertainty is very large "
                    f"(median std={s50:.1f}, p98 std={s98:.1f} µg/m³)."
                )


def _add_geo_ticks(ax, xmin, xmax, ymin, ymax, n=5):
    xticks = np.linspace(xmin, xmax, n)
    yticks = np.linspace(ymin, ymax, n)
    mid_y = (ymin + ymax) / 2.0
    mid_x = (xmin + xmax) / 2.0
    lon_labels = (
        gpd.GeoSeries(
            [Point(x, mid_y) for x in xticks], crs="EPSG:3857"
        ).to_crs("EPSG:4326").x
    )
    lat_labels = (
        gpd.GeoSeries(
            [Point(mid_x, y) for y in yticks], crs="EPSG:3857"
        ).to_crs("EPSG:4326").y
    )
    ax.set_xticks(xticks)
    ax.set_xticklabels([f"{v:.3f}°" for v in lon_labels], fontsize=8)
    ax.set_xlabel("Longitude", fontsize=9)
    ax.set_yticks(yticks)
    ax.set_yticklabels([f"{v:.3f}°" for v in lat_labels], fontsize=8)
    ax.set_ylabel("Latitude", fontsize=9)


# ---------------------------------------------------------------------------
# Main: generate all daily combined figures for one model
# ---------------------------------------------------------------------------
def process_model(csv_files, model_name, out_dir, epa_gdf,
                  xmin, xmax, ymin, ymax, lat_grid, lon_grid,
                  image_extent=None, support_mask=None):
    out_dir.mkdir(parents=True, exist_ok=True)

    # Pre-compute global colour limits across all days (consistent scale)
    all_means, all_stds = [], []
    frames = {}
    for p in sorted(csv_files):
        df = pd.read_csv(p)
        if "grid_id" not in df.columns:
            # Kalman CSVs may lack grid_id; build it from sorted (lat, lon)
            df = _attach_grid_id(df, lat_grid, lon_grid)
        frames[p] = df
        mean_grid = csv_to_grid(df, "mean_ug_m3")
        std_grid = csv_to_grid(df, "std_ug_m3")
        if support_mask is not None:
            mean_grid = np.where(support_mask, mean_grid, np.nan)
            std_grid = np.where(support_mask, std_grid, np.nan)
        all_means.append(mean_grid)
        all_stds.append(std_grid)

    warn_if_implausible_map_values(model_name, frames)

    all_means_arr = np.stack(all_means)
    all_stds_arr  = np.stack(all_stds)
    mean_vmin = float(np.nanpercentile(all_means_arr, 2))
    mean_vmax = float(np.nanpercentile(all_means_arr, 98))
    std_vmin  = float(np.nanpercentile(all_stds_arr,  2))
    std_vmax  = float(np.nanpercentile(all_stds_arr,  98))

    legend_elements = [
        Line2D([0], [0], marker="o", color="black", markerfacecolor="white",
               markersize=7, linestyle="", label="EPA station"),
    ]

    for idx, p in enumerate(sorted(csv_files)):
        df = frames[p]
        day_val = df["day"].iloc[0]
        n_obs   = int(df["n_obs"].iloc[0]) if "n_obs" in df.columns else -1
        day_label = idx
        if p.stem.startswith("fusion_grid_day_"):
            try:
                day_label = int(p.stem.rsplit("_", 1)[1])
            except ValueError:
                day_label = int(round(day_val)) + 1

        mean_raw = csv_to_grid(df, "mean_ug_m3")
        std_raw  = csv_to_grid(df, "std_ug_m3")
        if support_mask is not None:
            mean_raw = np.where(support_mask, mean_raw, np.nan)
            std_raw = np.where(support_mask, std_raw, np.nan)

        mean_smooth = gaussian_filter(
            np.where(np.isnan(mean_raw), np.nanmedian(mean_raw), mean_raw),
            sigma=SIGMA,
        )
        std_smooth = gaussian_filter(
            np.where(np.isnan(std_raw), np.nanmedian(std_raw), std_raw),
            sigma=SIGMA,
        )
        if support_mask is not None:
            mean_smooth = np.where(support_mask, mean_smooth, np.nan)
            std_smooth = np.where(support_mask, std_smooth, np.nan)

        obs_str = f"  ({n_obs} obs)" if n_obs >= 0 else ""
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        fig.suptitle(
            f"{model_name} — Day {day_label:02d} (t={day_val:.0f}){obs_str}",
            fontsize=13, weight="bold", y=1.01,
        )

        render_panel(
            axes[0], mean_smooth,
            title="Mean NO₂ (µg/m³)",
            cmap=CMAP_MEAN,
            xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax,
            epa_gdf=epa_gdf,
            unit_label="NO₂ (µg/m³)",
            vmin=mean_vmin, vmax=mean_vmax,
            image_extent=image_extent,
        )
        render_panel(
            axes[1], std_smooth,
            title="Uncertainty — Std (µg/m³)",
            cmap=CMAP_STD,
            xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax,
            epa_gdf=epa_gdf,
            unit_label="NO₂ std (µg/m³)",
            vmin=std_vmin, vmax=std_vmax,
            image_extent=image_extent,
        )

        axes[0].legend(
            handles=legend_elements, loc="upper left",
            frameon=True, facecolor="white", framealpha=0.8, fontsize=9,
        )

        plt.tight_layout()
        out_path = out_dir / f"{model_name.lower().replace(' ', '_')}_day_{idx:02d}.png"
        fig.savefig(out_path, dpi=DPI, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {out_path.name}")


# ---------------------------------------------------------------------------
# Static single-map plot (deterministic LUR and ATMO-Plan background)
# ---------------------------------------------------------------------------
def process_static_baselines(lur_csv, atmo_csv, out_dir, epa_gdf,
                              xmin, xmax, ymin, ymax, image_extent=None):
    out_dir.mkdir(parents=True, exist_ok=True)

    lur  = pd.read_csv(lur_csv)
    atmo = pd.read_csv(atmo_csv)

    lur_grid  = csv_to_grid(lur,  "value")
    atmo_grid = csv_to_grid(atmo, "value")

    lur_smooth  = gaussian_filter(
        np.where(np.isnan(lur_grid),  np.nanmedian(lur_grid),  lur_grid),
        sigma=SIGMA,
    )
    atmo_smooth = gaussian_filter(
        np.where(np.isnan(atmo_grid), np.nanmedian(atmo_grid), atmo_grid),
        sigma=SIGMA,
    )

    # Shared colour limits across both baselines
    all_vals = np.concatenate([lur_smooth.ravel(), atmo_smooth.ravel()])
    vmin = float(np.nanpercentile(all_vals, 2))
    vmax = float(np.nanpercentile(all_vals, 98))

    legend_elements = [
        Line2D([0], [0], marker="o", color="black", markerfacecolor="white",
               markersize=7, linestyle="", label="EPA station"),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Deterministic Baseline Fields — Time-averaged NO₂ (µg/m³)",
                 fontsize=13, weight="bold", y=1.01)

    render_panel(axes[0], lur_smooth,  title="Deterministic LUR baseline",
                 cmap=CMAP_MEAN, xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax,
                 epa_gdf=epa_gdf, unit_label="NO₂ (µg/m³)", vmin=vmin, vmax=vmax,
                 image_extent=image_extent)
    render_panel(axes[1], atmo_smooth, title="ATMO-Plan background surface",
                 cmap=CMAP_MEAN, xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax,
                 epa_gdf=epa_gdf, unit_label="NO₂ (µg/m³)", vmin=vmin, vmax=vmax,
                 image_extent=image_extent)

    axes[0].legend(handles=legend_elements, loc="upper left",
                   frameon=True, facecolor="white", framealpha=0.8, fontsize=9)

    plt.tight_layout()
    out_path = out_dir / "baselines_lur_atmo.png"
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path.name}")


# ---------------------------------------------------------------------------
# SVGP fusion maps (sparse per-day; fill grid with NaN then smooth)
# ---------------------------------------------------------------------------
def process_svgp_maps(pred_csv, unc_csv, out_dir, epa_gdf,
                      xmin, xmax, ymin, ymax, image_extent=None):
    out_dir.mkdir(parents=True, exist_ok=True)

    pred_df = pd.read_csv(pred_csv)
    unc_df  = pd.read_csv(unc_csv)

    unique_days = sorted(pred_df["timestamp"].unique())

    # Global colour limits across all days
    all_means, all_stds = [], []
    for day in unique_days:
        pm = csv_to_grid(pred_df[pred_df.timestamp == day].rename(
                columns={"value": "mean_ug_m3"}), "mean_ug_m3")
        pu = csv_to_grid(unc_df[unc_df.timestamp == day].rename(
                columns={"value": "std_ug_m3"}), "std_ug_m3")
        all_means.append(pm)
        all_stds.append(pu)

    mean_vmin = float(np.nanpercentile(np.stack(all_means), 2))
    mean_vmax = float(np.nanpercentile(np.stack(all_means), 98))
    std_vmin  = float(np.nanpercentile(np.stack(all_stds),  2))
    std_vmax  = float(np.nanpercentile(np.stack(all_stds),  98))

    legend_elements = [
        Line2D([0], [0], marker="o", color="black", markerfacecolor="white",
               markersize=7, linestyle="", label="EPA station"),
    ]

    for idx, day in enumerate(unique_days):
        mean_raw = all_means[idx]
        std_raw  = all_stds[idx]

        # Sparse grid: fill NaN with spatial median before smoothing
        mean_fill = np.where(np.isnan(mean_raw), np.nanmedian(mean_raw), mean_raw)
        std_fill  = np.where(np.isnan(std_raw),  np.nanmedian(std_raw),  std_raw)

        mean_smooth = gaussian_filter(mean_fill, sigma=SIGMA)
        std_smooth  = gaussian_filter(std_fill,  sigma=SIGMA)

        n_obs = int((~np.isnan(mean_raw)).sum())
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        fig.suptitle(
            f"SVGP Fusion — Day {idx:02d} (t={day:.0f})  ({n_obs} obs)",
            fontsize=13, weight="bold", y=1.01,
        )

        render_panel(axes[0], mean_smooth, title="Mean NO₂ (µg/m³)",
                     cmap=CMAP_MEAN, xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax,
                     epa_gdf=epa_gdf, unit_label="NO₂ (µg/m³)",
                     vmin=mean_vmin, vmax=mean_vmax,
                     image_extent=image_extent)
        render_panel(axes[1], std_smooth, title="Uncertainty — Std (µg/m³)",
                     cmap=CMAP_STD, xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax,
                     epa_gdf=epa_gdf, unit_label="NO₂ std (µg/m³)",
                     vmin=std_vmin, vmax=std_vmax,
                     image_extent=image_extent)

        axes[0].legend(handles=legend_elements, loc="upper left",
                       frameon=True, facecolor="white", framealpha=0.8, fontsize=9)

        plt.tight_layout()
        out_path = out_dir / f"svgp_fusion_day_{idx:02d}.png"
        fig.savefig(out_path, dpi=DPI, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {out_path.name}")


# ---------------------------------------------------------------------------
# Attach grid_id to Kalman CSVs (which lack it) by matching sorted order
# ---------------------------------------------------------------------------
def _attach_grid_id(df, lat_grid, lon_grid):
    """Match Kalman rows to grid_id by nearest lat/lon match."""
    # Build reference table from day-00 GPKF (grid_id → lat, lon)
    ref_lat = lat_grid.ravel()
    ref_lon = lon_grid.ravel()
    # Build grid_id strings: row_col
    rows_idx, cols_idx = np.unravel_index(
        np.arange(GRID_ROWS * GRID_COLS), (GRID_ROWS, GRID_COLS)
    )
    ref_ids = np.array([f"{r}_{c}" for r, c in zip(rows_idx, cols_idx)])

    # For each Kalman row find nearest grid cell
    obs_lat = df["latitude"].values
    obs_lon = df["longitude"].values
    # Use rounded match (grid spacing ~0.001°)
    ref_lat_r = np.round(ref_lat, 4)
    ref_lon_r = np.round(ref_lon, 4)
    obs_lat_r = np.round(obs_lat, 4)
    obs_lon_r = np.round(obs_lon, 4)

    from scipy.spatial import cKDTree
    tree = cKDTree(np.column_stack([ref_lat_r, ref_lon_r]))
    _, inds = tree.query(np.column_stack([obs_lat_r, obs_lon_r]))
    df = df.copy()
    df["grid_id"] = ref_ids[inds]
    return df


# ---------------------------------------------------------------------------
# Time series plots
# ---------------------------------------------------------------------------

def plot_timeseries(run_dir: Path, pub_dir: Path):
    """
    Generate two time series figures from saved pipeline outputs:

    1. Domain-averaged NO₂ over time — GPKF and Kalman smoother mean ± 1σ
       on the same axes, showing how the spatial field evolves day by day.

    2. Daily EPA observed vs SVGP predicted — aggregated to daily means,
       showing how well the fusion model tracks ground-truth EPA values.

    Saved to pub_dir/timeseries_domain_average.png and
    pub_dir/timeseries_epa_vs_svgp.png.
    """
    gpkf_dir   = run_dir / "gpkf_maps"
    kalman_dir = run_dir / "kalman_maps"
    preds_csv  = run_dir / "predictions.csv"
    pub_dir.mkdir(parents=True, exist_ok=True)

    # --- Panel 1: domain-averaged NO₂ over time ---
    gpkf_csvs   = sorted(gpkf_dir.glob("gpkf_day_*.csv"))
    kalman_csvs = sorted(kalman_dir.glob("kalman_day_*.csv")) if kalman_dir.exists() else []

    def _domain_stats(csvs, mean_col="mean_ug_m3", std_col="std_ug_m3"):
        days, means, stds = [], [], []
        for i, p in enumerate(csvs):
            df = pd.read_csv(p)
            days.append(i)
            means.append(df[mean_col].mean())
            stds.append(df[std_col].mean())
        return np.array(days), np.array(means), np.array(stds)

    fig1, ax1 = plt.subplots(figsize=(11, 4))

    if gpkf_csvs:
        gdays, gmeans, gstds = _domain_stats(gpkf_csvs)
        ax1.plot(gdays, gmeans, color="steelblue", linewidth=1.8, label="GPKF")
        ax1.fill_between(gdays, gmeans - gstds, gmeans + gstds,
                         alpha=0.20, color="steelblue")

    if kalman_csvs:
        kdays, kmeans, kstds = _domain_stats(kalman_csvs)
        ax1.plot(kdays, kmeans, color="darkorange", linewidth=1.8,
                 linestyle="--", label="Kalman smoother")
        ax1.fill_between(kdays, kmeans - kstds, kmeans + kstds,
                         alpha=0.15, color="darkorange")

    ax1.set_xlabel("Day")
    ax1.set_ylabel("NO₂ (µg/m³)")
    ax1.set_title("Domain-averaged NO₂ over time (mean ± 1σ across grid cells)")
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    fig1.tight_layout()
    out1 = pub_dir / "timeseries_domain_average.png"
    fig1.savefig(out1, dpi=DPI, bbox_inches="tight")
    plt.close(fig1)
    print(f"Saved: {out1}")

    # --- Panel 2: daily EPA observed vs SVGP predicted ---
    if not preds_csv.exists():
        print(f"Skipping EPA time series: {preds_csv} not found")
        return

    df = pd.read_csv(preds_csv)
    epa_df = df[df["is_epa"] == True].copy()
    if epa_df.empty:
        print("Skipping EPA time series: no EPA rows in predictions.csv")
        return

    # Aggregate to daily mean
    daily = (
        epa_df.groupby("timestamp")[["epa_true", "mean"]]
        .mean()
        .reset_index()
        .sort_values("timestamp")
    )
    daily["day_idx"] = np.arange(len(daily))

    fig2, ax2 = plt.subplots(figsize=(11, 4))
    ax2.plot(daily["day_idx"], daily["epa_true"], "ko-",
             markersize=5, linewidth=1.4, label="EPA observed (daily mean)")
    ax2.plot(daily["day_idx"], daily["mean"], "s--",
             color="steelblue", markersize=5, linewidth=1.4,
             label="SVGP predicted (daily mean)")
    ax2.set_xticks(daily["day_idx"])
    ax2.set_xticklabels(
        [str(t)[:10] for t in daily["timestamp"]],
        rotation=45, ha="right", fontsize=8
    )
    ax2.set_ylabel("NO₂ (µg/m³)")
    ax2.set_title("Daily EPA observations vs FusionSVGP predictions")
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    fig2.tight_layout()
    out2 = pub_dir / "timeseries_epa_vs_svgp.png"
    fig2.savefig(out2, dpi=DPI, bbox_inches="tight")
    plt.close(fig2)
    print(f"Saved: {out2}")


# ---------------------------------------------------------------------------
# Diagnostic plots (training, SVGP, GPKF)
# ---------------------------------------------------------------------------

def plot_training_history(run_dir: Path, pub_dir: Path):
    """
    Plot training and validation loss curves from training_history.csv.

    Saved to pub_dir/training_history.png.
    """
    csv = run_dir / "training_history.csv"
    if not csv.exists():
        print(f"Skipping training history: {csv} not found")
        return

    df = pd.read_csv(csv)
    pub_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(df["epoch"], df["train_loss"], color="steelblue", linewidth=1.8, label="Train loss")
    if "val_loss" in df.columns and df["val_loss"].notna().any():
        ax.plot(df["epoch"], df["val_loss"], color="darkorange", linewidth=1.8,
                linestyle="--", label="Val loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("ELBO loss")
    ax.set_title("SVGP training history")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out = pub_dir / "training_history.png"
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


def plot_svgp_diagnostics(run_dir: Path, pub_dir: Path):
    """
    Calibration scatter and residual histogram for SVGP predictions vs EPA.

    Reads predictions.csv (columns: mean, std, epa_true, is_epa).
    Saved to pub_dir/svgp_diagnostics.png.
    """
    csv = run_dir / "predictions.csv"
    if not csv.exists():
        print(f"Skipping SVGP diagnostics: {csv} not found")
        return

    df = pd.read_csv(csv)
    epa = df[(df["is_epa"] == True) & df["epa_true"].notna() & df["mean"].notna()]
    if epa.empty:
        print("Skipping SVGP diagnostics: no valid EPA rows in predictions.csv")
        return

    pub_dir.mkdir(parents=True, exist_ok=True)
    y_true = epa["epa_true"].values
    y_pred = epa["mean"].values
    residuals = y_true - y_pred

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("SVGP diagnostics — EPA test-set predictions", fontsize=13, weight="bold")

    # Calibration scatter
    ax = axes[0]
    lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
    ax.scatter(y_true, y_pred, s=20, alpha=0.5, color="steelblue", edgecolors="none")
    ax.plot(lims, lims, "k--", linewidth=1.2, label="1:1 line")
    ax.set_xlabel("EPA observed (µg/m³)")
    ax.set_ylabel("SVGP predicted (µg/m³)")
    ax.set_title("Predicted vs observed")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Residual histogram
    ax = axes[1]
    ax.hist(residuals, bins=30, color="steelblue", edgecolor="white", alpha=0.8)
    ax.axvline(0, color="k", linewidth=1.2, linestyle="--")
    ax.axvline(residuals.mean(), color="darkorange", linewidth=1.4,
               linestyle="-", label=f"Mean = {residuals.mean():.2f}")
    ax.set_xlabel("Residual: observed − predicted (µg/m³)")
    ax.set_ylabel("Count")
    ax.set_title("Residual distribution")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    out = pub_dir / "svgp_diagnostics.png"
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


def plot_gpkf_diagnostics(run_dir: Path, pub_dir: Path):
    """
    Calibration scatter and normalised-error histogram for GPKF predictions vs EPA.

    Reads gpkf_epa_predictions.csv (columns: epa_true, gpkf_pred, gpkf_pred_std).
    Saved to pub_dir/gpkf_diagnostics.png.
    """
    csv = run_dir / "gpkf_epa_predictions.csv"
    if not csv.exists():
        print(f"Skipping GPKF diagnostics: {csv} not found")
        return

    df = pd.read_csv(csv)
    valid = df["epa_true"].notna() & df["gpkf_pred"].notna() & df["gpkf_pred_std"].notna()
    df = df[valid]
    if df.empty:
        print("Skipping GPKF diagnostics: no valid rows in gpkf_epa_predictions.csv")
        return

    pub_dir.mkdir(parents=True, exist_ok=True)
    y_true = df["epa_true"].values
    y_pred = df["gpkf_pred"].values
    y_std  = df["gpkf_pred_std"].values
    z_scores = (y_true - y_pred) / np.maximum(y_std, 1e-6)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("GPKF diagnostics — EPA test-set predictions", fontsize=13, weight="bold")

    # Calibration scatter with ±1σ error bars
    ax = axes[0]
    lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
    ax.errorbar(y_true, y_pred, yerr=y_std, fmt="o", markersize=4,
                alpha=0.5, color="steelblue", ecolor="lightsteelblue",
                elinewidth=0.8, capsize=0)
    ax.plot(lims, lims, "k--", linewidth=1.2, label="1:1 line")
    ax.set_xlabel("EPA observed (µg/m³)")
    ax.set_ylabel("GPKF predicted (µg/m³)")
    ax.set_title("Predicted vs observed (±1σ)")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Normalised-error histogram (should be ~N(0,1) if well-calibrated)
    ax = axes[1]
    ax.hist(z_scores, bins=30, color="steelblue", edgecolor="white", alpha=0.8,
            density=True, label="z-scores")
    # Overlay standard normal
    zx = np.linspace(-4, 4, 200)
    ax.plot(zx, np.exp(-0.5 * zx**2) / np.sqrt(2 * np.pi),
            "k--", linewidth=1.4, label="N(0,1)")
    ax.axvline(0, color="grey", linewidth=0.8)
    ax.set_xlabel("Normalised error (z-score)")
    ax.set_ylabel("Density")
    ax.set_title("Calibration: normalised error distribution")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    out = pub_dir / "gpkf_diagnostics.png"
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


# ---------------------------------------------------------------------------
# Coverage reliability diagram + RMSE comparison bar chart
# ---------------------------------------------------------------------------

def plot_pit_histograms(run_dir: Path, pub_dir: Path):
    """
    Probability Integral Transform (PIT) histograms for FusionGP and GPKF.

    PIT = Φ((y_true − μ_pred) / σ_pred).  If the predictive distribution is
    perfectly calibrated, PIT values are uniformly distributed on [0, 1].
    - Peaked in the centre → over-dispersed (intervals too wide)
    - U-shaped → over-confident (intervals too narrow)
    - Skewed → systematic bias

    Panels: SVGP (left) and GPKF (right), each with a uniform reference line.
    Saved to pub_dir/pit_histograms.png.
    """
    pred_csv = run_dir / "predictions.csv"
    gpkf_csv = run_dir / "gpkf_epa_predictions.csv"

    pub_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("PIT histograms — calibration check", fontsize=13, weight="bold")

    n_bins = 10
    uniform_height = 1.0 / n_bins  # expected bar height for uniform distribution

    # --- SVGP ---
    ax = axes[0]
    if pred_csv.exists():
        df = pd.read_csv(pred_csv)
        epa = df[(df["is_epa"] == True) & df["epa_true"].notna()
                 & df["mean"].notna() & df["std"].notna()]
        if not epa.empty:
            pit = scipy_norm.cdf(
                (epa["epa_true"].values - epa["mean"].values)
                / np.maximum(epa["std"].values, 1e-6)
            )
            ax.hist(pit, bins=n_bins, range=(0, 1), density=True,
                    color="steelblue", edgecolor="white", alpha=0.85)
            ax.axhline(1.0, color="k", linestyle="--", linewidth=1.2,
                       label="Uniform (ideal)")
            n_svgp = len(pit)
        else:
            ax.text(0.5, 0.5, "No EPA predictions found",
                    ha="center", va="center", transform=ax.transAxes)
            n_svgp = 0
    else:
        ax.text(0.5, 0.5, f"{pred_csv.name} not found",
                ha="center", va="center", transform=ax.transAxes)
        n_svgp = 0

    ax.set_xlabel("PIT value", fontsize=11)
    ax.set_ylabel("Density", fontsize=11)
    ax.set_title(f"FusionGP (SVGP)  [n={n_svgp}]", fontsize=11)
    ax.set_xlim(0, 1)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # --- GPKF ---
    ax = axes[1]
    if gpkf_csv.exists():
        df = pd.read_csv(gpkf_csv)
        valid = df["epa_true"].notna() & df["gpkf_pred"].notna() & df["gpkf_pred_std"].notna()
        df = df[valid]
        if not df.empty:
            pit = scipy_norm.cdf(
                (df["epa_true"].values - df["gpkf_pred"].values)
                / np.maximum(df["gpkf_pred_std"].values, 1e-6)
            )
            ax.hist(pit, bins=n_bins, range=(0, 1), density=True,
                    color="darkorange", edgecolor="white", alpha=0.85)
            ax.axhline(1.0, color="k", linestyle="--", linewidth=1.2,
                       label="Uniform (ideal)")
            n_gpkf = len(pit)
        else:
            ax.text(0.5, 0.5, "No valid GPKF rows found",
                    ha="center", va="center", transform=ax.transAxes)
            n_gpkf = 0
    else:
        ax.text(0.5, 0.5, f"{gpkf_csv.name} not found",
                ha="center", va="center", transform=ax.transAxes)
        n_gpkf = 0

    ax.set_xlabel("PIT value", fontsize=11)
    ax.set_ylabel("Density", fontsize=11)
    ax.set_title(f"GPKF (sequential)  [n={n_gpkf}]", fontsize=11)
    ax.set_xlim(0, 1)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    out = pub_dir / "pit_histograms.png"
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


def plot_coverage_reliability(run_dir: Path, pub_dir: Path):
    """
    Coverage reliability diagram: observed coverage vs nominal level for
    FusionGP (SVGP) and GPKF.

    A well-calibrated model sits on the diagonal. Points below the diagonal
    indicate over-confidence (intervals too narrow); above = under-confidence.

    Reads final_results_table.csv.
    Saved to pub_dir/coverage_reliability.png.
    """
    csv = run_dir / "final_results_table.csv"
    if not csv.exists():
        print(f"Skipping coverage reliability: {csv} not found")
        return

    df = pd.read_csv(csv)
    pub_dir.mkdir(parents=True, exist_ok=True)

    levels   = [0.50, 0.80, 0.90, 0.95]
    col_map  = {
        0.50: "coverage_50",
        0.80: "coverage_80",
        0.90: "coverage_90",
        0.95: "coverage_95",
    }

    fig, ax = plt.subplots(figsize=(6, 6))

    colours = {"FusionGP": "steelblue", "GPKF": "darkorange"}
    markers = {"FusionGP": "o",         "GPKF": "s"}

    for _, row in df.iterrows():
        model = row["model"]
        if model not in colours:
            continue
        observed = [row[col_map[l]] for l in levels]
        ax.plot(
            levels, observed,
            marker=markers[model], color=colours[model],
            linewidth=1.8, markersize=8, label=model,
        )
        # Annotate each point with the coverage value
        for lv, ov in zip(levels, observed):
            ax.annotate(
                f"{ov:.2f}",
                xy=(lv, ov), xytext=(4, 4),
                textcoords="offset points", fontsize=7.5,
                color=colours[model],
            )

    # Perfect-calibration diagonal
    ax.plot([0, 1], [0, 1], "k--", linewidth=1.2, label="Perfect calibration")

    ax.set_xlim(0.4, 1.0)
    ax.set_ylim(0.3, 1.0)
    ax.set_xlabel("Nominal coverage level", fontsize=11)
    ax.set_ylabel("Observed coverage", fontsize=11)
    ax.set_title("Coverage reliability diagram", fontsize=12, weight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    out = pub_dir / "coverage_reliability.png"
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


def plot_scatter_spatial_station_means(run_dir: Path, pub_dir: Path):
    """
    Spatial correlation scatter plots: predicted vs observed mean NO₂ at the
    9 EPA test stations, aggregated over the study period.

    For static baselines (LUR, ATMO-Plan) this is the natural comparison —
    they carry no temporal information.  For dynamic models (FusionGP, GPKF)
    the time-mean is used so all four panels share the same x-axis.

    Reads: predictions.csv, gpkf_epa_predictions.csv, plus repo-local
           LUR and ATMO baseline CSVs.
    Saved to pub_dir/scatter_spatial_station_means.png.
    """
    from scipy import stats as scipy_stats

    def first_existing(*paths: str) -> Path:
        for path in paths:
            p = Path(path)
            if p.exists():
                return p
        return Path(paths[0])

    def atmo_value_column(df: pd.DataFrame) -> str:
        for col in ("model_no2_10m", "model_no2", "no2", "predicted_no2"):
            if col in df.columns:
                return col
        raise ValueError(
            "ATMO-Plan file must contain one of: "
            "model_no2_10m, model_no2, no2, predicted_no2"
        )

    pred_csv = run_dir / "predictions.csv"
    gpkf_csv = run_dir / "gpkf_epa_predictions.csv"
    lur_csv  = first_existing(
        "data/lur_predictions.csv",
        "data_for_notebook/lur_predictions.csv",
        "experiment_data_used/lur_predictions.csv",
        "data_for_analysis/lur_predictions.csv",
    )
    atmo_csv = first_existing(
        "data/atmos_plan_model_no2.csv",
        "data_for_notebook/atmos_plan_model_no2.csv",
        "experiment_data_used/atmos_plan_model_no2.csv",
        "data_for_analysis/atmos_plan_model_no2.csv",
    )

    for p in (pred_csv, gpkf_csv, lur_csv, atmo_csv):
        if not p.exists():
            print(f"Skipping spatial scatter: {p} not found")
            return

    pub_dir.mkdir(parents=True, exist_ok=True)

    def rc(df):
        df = df.copy()
        df["latitude"]  = df["latitude"].round(6)
        df["longitude"] = df["longitude"].round(6)
        return df

    gpkf     = rc(pd.read_csv(gpkf_csv))
    svgp_epa = pd.read_csv(pred_csv)
    svgp_epa = rc(svgp_epa[svgp_epa["epa_true"].notna()].copy())
    lur      = rc(pd.read_csv(lur_csv))
    atmo     = rc(pd.read_csv(atmo_csv))
    atmo_col = atmo_value_column(atmo)
    atmo = atmo.rename(columns={atmo_col: "model_no2"})

    gpkf["day"]     = (pd.to_datetime(gpkf["timestamp"]) - pd.Timestamp("2023-06-01")).dt.days
    svgp_epa["day"] = svgp_epa["timestamp"].astype(int)

    merged = gpkf.merge(
        svgp_epa[["latitude", "longitude", "day", "mean"]].rename(columns={"mean": "svgp_pred"}),
        on=["latitude", "longitude", "day"], how="left",
    )
    merged = merged.merge(lur[["latitude", "longitude", "predicted_no2"]],
                          on=["latitude", "longitude"], how="left")
    merged = merged.merge(atmo[["latitude", "longitude", "model_no2"]],
                          on=["latitude", "longitude"], how="left")

    sm = merged.groupby(["latitude", "longitude"])[
        ["epa_true", "svgp_pred", "gpkf_pred", "predicted_no2", "model_no2"]
    ].mean().reset_index()

    y_s = sm["epa_true"].values
    n_stations = int(sm[["latitude", "longitude"]].drop_duplicates().shape[0])
    models = [
        ("LUR",       "predicted_no2", "#9B59B6", "static"),
        ("ATMO-Plan", "model_no2",     "#E67E22", "static (annual)"),
        ("FusionGP",  "svgp_pred",     "#2980B9", "dynamic (mean)"),
        ("GPKF",      "gpkf_pred",     "#27AE60", "dynamic (mean)"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(10, 9))
    axes = axes.flatten()
    lim = (-5, 60)

    for ax, (name, col, colour, kind) in zip(axes, models):
        yhat = sm[col].values
        mask = ~np.isnan(yhat)
        slope, intercept, r, _, _ = scipy_stats.linregress(y_s[mask], yhat[mask])
        r2   = r ** 2
        rmse = np.sqrt(np.mean((y_s[mask] - yhat[mask]) ** 2))

        ax.scatter(y_s[mask], yhat[mask], color=colour, alpha=0.8, s=80,
                   edgecolors="white", linewidths=0.5, zorder=3)
        ax.plot(lim, lim, "k--", lw=1, alpha=0.5)
        xfit = np.array(lim)
        ax.plot(xfit, slope * xfit + intercept, color=colour, lw=1.8, alpha=0.85)
        ax.set_xlim(lim); ax.set_ylim(lim)
        ax.set_xlabel("EPA mean NO$_2$ (µg/m³)", fontsize=10)
        ax.set_ylabel("Predicted mean NO$_2$ (µg/m³)", fontsize=10)
        ax.set_title(f"{name}  [{kind}]", fontsize=11, fontweight="bold", color=colour)
        ax.text(0.05, 0.95,
                f"$R^2$ = {r2:.3f}\nRMSE = {rmse:.2f} µg/m³\n$n$ = {mask.sum()} stations",
                transform=ax.transAxes, fontsize=9, va="top",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                          alpha=0.85, edgecolor="#cccccc"))
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.suptitle(
        "Spatial Correlation: Predicted vs Observed Mean NO$_2$\n"
        f"(Station-level means, June 2023, $n={n_stations}$ stations)",
        fontsize=13, fontweight="bold", y=1.01,
    )
    fig.tight_layout()
    out = pub_dir / "scatter_spatial_station_means.png"
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


def plot_rmse_comparison(run_dir: Path, pub_dir: Path):
    """
    Bar chart comparing RMSE across all four models:
    LUR, ATMO-Plan, FusionGP (SVGP), and GPKF.

    Also shows correlation (r) as a secondary marker so the reader can see
    both accuracy and pattern agreement in one figure.

    Reads final_results_table.csv.
    Saved to pub_dir/rmse_comparison.png.
    """
    csv = run_dir / "final_results_table.csv"
    if not csv.exists():
        print(f"Skipping RMSE comparison: {csv} not found")
        return

    df = pd.read_csv(csv)
    pub_dir.mkdir(parents=True, exist_ok=True)

    # Desired order and display labels
    order  = ["LUR", "ATMO-Plan", "FusionGP", "GPKF"]
    labels = ["LUR\n(physical)", "ATMO-Plan\n(numerical)", "FusionGP\n(SVGP)", "GPKF\n(sequential)"]
    colors = ["#b0c4de", "#f4a460", "#4682b4", "#2e8b57"]

    df_plot = df.set_index("model").reindex(order)

    fig, ax1 = plt.subplots(figsize=(8, 5))

    x = np.arange(len(order))
    bars = ax1.bar(x, df_plot["rmse"], color=colors, edgecolor="white",
                   linewidth=0.8, zorder=3)

    # Annotate bars with RMSE value
    for bar, val in zip(bars, df_plot["rmse"]):
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.15,
            f"{val:.2f}",
            ha="center", va="bottom", fontsize=9, fontweight="bold",
        )

    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=10)
    ax1.set_ylabel("RMSE (µg/m³)", fontsize=11)
    ax1.set_ylim(0, df_plot["rmse"].max() * 1.25)
    ax1.set_title("Model comparison — RMSE at EPA test locations", fontsize=12, weight="bold")
    ax1.grid(axis="y", alpha=0.3, zorder=0)
    ax1.set_axisbelow(True)

    # Secondary axis: Pearson r
    if "corr" in df_plot.columns and df_plot["corr"].notna().any():
        ax2 = ax1.twinx()
        ax2.plot(x, df_plot["corr"], "D--", color="crimson", markersize=8,
                 linewidth=1.6, label="Pearson r", zorder=5)
        ax2.set_ylabel("Pearson r (correlation)", fontsize=11, color="crimson")
        ax2.tick_params(axis="y", colors="crimson")
        ax2.set_ylim(0, 1.0)
        ax2.legend(loc="upper right", fontsize=9)

    fig.tight_layout()
    out = pub_dir / "rmse_comparison.png"
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main():
    run_dir = find_run_dir(sys.argv[1] if len(sys.argv) > 1 else None)
    print(f"Run directory: {run_dir}")

    gpkf_dir   = run_dir / "gpkf_maps"
    kalman_dir = run_dir / "kalman_maps"
    preds_csv  = run_dir / "predictions.csv"
    pub_dir    = run_dir / "publication_maps"
    map_data_dir = run_dir / "map_data"

    if not gpkf_dir.exists():
        raise FileNotFoundError(f"No gpkf_maps/ in {run_dir}")

    # Load reference grid geometry from day-00 GPKF
    day00 = pd.read_csv(sorted(gpkf_dir.glob("gpkf_day_*.csv"))[0])
    lat_grid, lon_grid = grid_coords_arrays(day00)

    # Compute Web Mercator bounding box
    print("Computing Web Mercator extent ...")
    xmin, xmax, ymin, ymax = latlon_grid_to_3857_extent(lat_grid, lon_grid)
    image_extent = (xmin, xmax, ymin, ymax)
    # 5% padding
    dx, dy = (xmax - xmin) * 0.05, (ymax - ymin) * 0.05
    xmin -= dx; xmax += dx; ymin -= dy; ymax += dy

    # EPA stations
    epa_gdf = None
    if preds_csv.exists():
        try:
            epa_gdf = load_epa_stations(preds_csv)
            print(f"Loaded {len(epa_gdf)} EPA station locations")
        except Exception as e:
            print(f"Warning: could not load EPA stations: {e}")
    if epa_gdf is None:
        epa_gdf = gpd.GeoDataFrame(geometry=[], crs="EPSG:3857")

    atmo_support_mask = None
    atmo_csv = map_data_dir / "atmo_plan_baseline.csv"
    if atmo_csv.exists():
        try:
            atmo_support_mask = valid_value_mask_from_csv(atmo_csv, "value")
            masked = int((~atmo_support_mask).sum())
            total = int(atmo_support_mask.size)
            print(
                f"ATMO support mask: hiding {masked}/{total} cells "
                f"({masked / total:.1%}) in GPKF maps"
            )
        except Exception as exc:
            print(f"Warning: could not build ATMO support mask: {exc}")

    # GPKF maps
    gpkf_csvs = sorted(gpkf_dir.glob("gpkf_day_*.csv"))
    print(f"\nGenerating GPKF publication maps ({len(gpkf_csvs)} days) ...")
    process_model(
        gpkf_csvs, "GP-Kalman Filter",
        pub_dir / "gpkf",
        epa_gdf, xmin, xmax, ymin, ymax,
        lat_grid, lon_grid,
        image_extent=image_extent,
        support_mask=atmo_support_mask,
    )

    # Kalman smoother maps skipped — smoother uses SVGP batch posteriors which
    # already integrate all 29 days, so per-day variation is <1 µg/m³ and maps
    # are visually identical from day 5 onwards. Use GPKF maps instead.

    # Static baselines (LUR + Atmo Plan)
    lur_csv  = map_data_dir / "lur_baseline.csv"
    if lur_csv.exists() and atmo_csv.exists():
        print("\nGenerating static baseline maps ...")
        process_static_baselines(
            lur_csv, atmo_csv, pub_dir / "baselines",
            epa_gdf, xmin, xmax, ymin, ymax,
            image_extent=image_extent,
        )

    # SVGP fusion maps. Prefer dense full-grid posterior exports from
    # run_demo_pipeline.py; fall back to legacy sparse test-set prediction CSVs.
    dense_svgp_csvs = sorted(map_data_dir.glob("fusion_grid_day_*.csv"))
    if dense_svgp_csvs:
        print(f"\nGenerating dense SVGP fusion maps ({len(dense_svgp_csvs)} selected days) ...")
        process_model(
            dense_svgp_csvs, "SVGP Fusion",
            pub_dir / "svgp",
            epa_gdf, xmin, xmax, ymin, ymax,
            lat_grid, lon_grid,
            image_extent=image_extent,
        )
        pred_csv = unc_csv = None
    else:
        pred_csv = map_data_dir / "fusion_predictions.csv"
        unc_csv  = map_data_dir / "fusion_uncertainty.csv"

    if pred_csv is not None and pred_csv.exists() and unc_csv.exists():
        print(f"\nGenerating SVGP fusion maps ...")
        process_svgp_maps(
            pred_csv, unc_csv, pub_dir / "svgp",
            epa_gdf, xmin, xmax, ymin, ymax,
            image_extent=image_extent,
        )

    # Time series
    print("\nGenerating time series plots ...")
    plot_timeseries(run_dir, pub_dir / "timeseries")

    # Diagnostic plots
    diag_dir = pub_dir / "diagnostics"
    print("\nGenerating diagnostic plots ...")
    plot_training_history(run_dir, diag_dir)
    plot_svgp_diagnostics(run_dir, diag_dir)
    plot_gpkf_diagnostics(run_dir, diag_dir)

    # Summary comparison figures
    print("\nGenerating summary comparison figures ...")
    plot_pit_histograms(run_dir, pub_dir / "diagnostics")
    plot_coverage_reliability(run_dir, pub_dir / "diagnostics")
    plot_rmse_comparison(run_dir, pub_dir / "diagnostics")
    plot_scatter_spatial_station_means(run_dir, pub_dir / "diagnostics")

    print(f"\nAll figures saved to: {pub_dir}")


if __name__ == "__main__":
    main()
