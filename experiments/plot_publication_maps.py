"""
Publication-quality map generation for FusionGP outputs.

Reads GPKF and Kalman smoother daily CSVs and generates enhanced figures:
  - Gaussian smoothing (removes blocky pixel artefacts)
  - Combined mean + std side-by-side figure per day
  - EPA station overlay
  - Percentile-clipped colourmaps
  - OpenStreetMap basemap via contextily

Usage:
    python experiments/plot_publication_maps.py [run_dir]

If run_dir is omitted, the most-recent outputs/demo_run_* is used.
"""

import sys
import os
import glob
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()  # loads .env from cwd or any parent directory

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.colors import Normalize
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
SIGMA = 2.0          # Gaussian smoothing kernel (grid cells)
ALPHA = 0.55         # heatmap transparency over basemap
CMAP_MEAN = "RdYlGn_r"  # for mean NO₂ — green=low, red=high
CMAP_STD  = "Purples"   # for uncertainty
DPI = 200

# Stadia Maps API key — loaded from .env (STADIA_API_KEY=...)
# Falls back to CartoDB.Positron if not set
STADIA_API_KEY = os.getenv("STADIA_API_KEY", "")


# ---------------------------------------------------------------------------
# Helper: locate run directory
# ---------------------------------------------------------------------------
def find_run_dir(arg=None):
    if arg:
        return Path(arg)
    candidates = sorted(
        glob.glob("outputs/demo_run_*"),
        key=lambda p: Path(p).name
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
    Also returns (x_3857, y_3857) flat arrays for the grid cell centres.
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
    return x.min(), x.max(), y.min(), y.max()


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
                 epa_gdf, unit_label, vmin=None, vmax=None):
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)

    # Basemap
    try:
        if STADIA_API_KEY:
            source = f"https://tiles.stadiamaps.com/tiles/stamen_toner_lite/{{z}}/{{x}}/{{y}}.png?api_key={STADIA_API_KEY}"
        else:
            source = ctx.providers.CartoDB.Positron
        ctx.add_basemap(ax, source=source, crs="EPSG:3857", zoom="auto")
    except Exception:
        pass  # offline / tile fetch failure — continue without basemap

    # Percentile-clipped colour scale
    if vmin is None:
        vmin = float(np.nanpercentile(grid_smooth, 2))
    if vmax is None:
        vmax = float(np.nanpercentile(grid_smooth, 98))
    norm = Normalize(vmin=vmin, vmax=vmax)

    im = ax.imshow(
        grid_smooth,
        extent=(xmin, xmax, ymin, ymax),
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
    cbar = plt.colorbar(im, ax=ax, shrink=0.75, pad=0.02)
    cbar.set_label(unit_label, fontsize=9)  # NO₂ µg/m³
    cbar.ax.tick_params(labelsize=8)

    ax.set_title(title, fontsize=11, weight="bold", pad=6)

    # Geographic tick labels (convert 3857 back to lon/lat)
    _add_geo_ticks(ax, xmin, xmax, ymin, ymax)


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
                  xmin, xmax, ymin, ymax, lat_grid, lon_grid):
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
        all_means.append(csv_to_grid(df, "mean_ug_m3"))
        all_stds.append(csv_to_grid(df, "std_ug_m3"))

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

        mean_raw = csv_to_grid(df, "mean_ug_m3")
        std_raw  = csv_to_grid(df, "std_ug_m3")

        mean_smooth = gaussian_filter(
            np.where(np.isnan(mean_raw), np.nanmedian(mean_raw), mean_raw),
            sigma=SIGMA,
        )
        std_smooth = gaussian_filter(
            np.where(np.isnan(std_raw), np.nanmedian(std_raw), std_raw),
            sigma=SIGMA,
        )

        obs_str = f"  ({n_obs} obs)" if n_obs >= 0 else ""
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        fig.suptitle(
            f"{model_name} — Day {idx:02d} (t={day_val:.0f}){obs_str}",
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
        )
        render_panel(
            axes[1], std_smooth,
            title="Uncertainty — Std (µg/m³)",
            cmap=CMAP_STD,
            xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax,
            epa_gdf=epa_gdf,
            unit_label="NO₂ std (µg/m³)",
            vmin=std_vmin, vmax=std_vmax,
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
# Static single-map plot (LUR / Atmo Plan baselines)
# ---------------------------------------------------------------------------
def process_static_baselines(lur_csv, atmo_csv, out_dir, epa_gdf,
                              xmin, xmax, ymin, ymax):
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
    fig.suptitle("Static Baselines — Time-averaged NO₂ (µg/m³)",
                 fontsize=13, weight="bold", y=1.01)

    render_panel(axes[0], lur_smooth,  title="LUR Baseline",
                 cmap=CMAP_MEAN, xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax,
                 epa_gdf=epa_gdf, unit_label="NO₂ (µg/m³)", vmin=vmin, vmax=vmax)
    render_panel(axes[1], atmo_smooth, title="Atmo Plan Baseline",
                 cmap=CMAP_MEAN, xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax,
                 epa_gdf=epa_gdf, unit_label="NO₂ (µg/m³)", vmin=vmin, vmax=vmax)

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
                      xmin, xmax, ymin, ymax):
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
                     vmin=mean_vmin, vmax=mean_vmax)
        render_panel(axes[1], std_smooth, title="Uncertainty — Std (µg/m³)",
                     cmap=CMAP_STD, xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax,
                     epa_gdf=epa_gdf, unit_label="NO₂ std (µg/m³)",
                     vmin=std_vmin, vmax=std_vmax)

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
# Entry point
# ---------------------------------------------------------------------------
def main():
    run_dir = find_run_dir(sys.argv[1] if len(sys.argv) > 1 else None)
    print(f"Run directory: {run_dir}")

    gpkf_dir   = run_dir / "gpkf_maps"
    kalman_dir = run_dir / "kalman_maps"
    preds_csv  = run_dir / "predictions.csv"
    pub_dir    = run_dir / "publication_maps"

    if not gpkf_dir.exists():
        raise FileNotFoundError(f"No gpkf_maps/ in {run_dir}")

    # Load reference grid geometry from day-00 GPKF
    day00 = pd.read_csv(sorted(gpkf_dir.glob("gpkf_day_*.csv"))[0])
    lat_grid, lon_grid = grid_coords_arrays(day00)

    # Compute Web Mercator bounding box
    print("Computing Web Mercator extent ...")
    xmin, xmax, ymin, ymax = latlon_grid_to_3857_extent(lat_grid, lon_grid)
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

    # GPKF maps
    gpkf_csvs = sorted(gpkf_dir.glob("gpkf_day_*.csv"))
    print(f"\nGenerating GPKF publication maps ({len(gpkf_csvs)} days) ...")
    process_model(
        gpkf_csvs, "GP-Kalman Filter",
        pub_dir / "gpkf",
        epa_gdf, xmin, xmax, ymin, ymax,
        lat_grid, lon_grid,
    )

    # Kalman smoother maps skipped — smoother uses SVGP batch posteriors which
    # already integrate all 29 days, so per-day variation is <1 µg/m³ and maps
    # are visually identical from day 5 onwards. Use GPKF maps instead.

    # Static baselines (LUR + Atmo Plan)
    map_data_dir = run_dir / "map_data"
    lur_csv  = map_data_dir / "lur_baseline.csv"
    atmo_csv = map_data_dir / "atmo_plan_baseline.csv"
    if lur_csv.exists() and atmo_csv.exists():
        print("\nGenerating static baseline maps ...")
        process_static_baselines(
            lur_csv, atmo_csv, pub_dir / "baselines",
            epa_gdf, xmin, xmax, ymin, ymax,
        )

    # SVGP fusion maps
    pred_csv = map_data_dir / "fusion_predictions.csv"
    unc_csv  = map_data_dir / "fusion_uncertainty.csv"
    if pred_csv.exists() and unc_csv.exists():
        print(f"\nGenerating SVGP fusion maps ...")
        process_svgp_maps(
            pred_csv, unc_csv, pub_dir / "svgp",
            epa_gdf, xmin, xmax, ymin, ymax,
        )

    print(f"\nAll publication maps saved to: {pub_dir}")


if __name__ == "__main__":
    main()
