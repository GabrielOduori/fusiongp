# generate_leaflet_maps.py
# Produces publication-quality PNG maps.
# - If the CSV has grid_id: joins to grid.geojson → choropleth (filled polygons)
# - Otherwise: coloured scatter dots on basemap
#
# Reads map_data/*.csv from a run directory and saves PNGs to map_images/.

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import contextily as ctx
import geopandas as gpd
import warnings

warnings.filterwarnings("ignore")

GRID_GEOJSON = Path("data/grid/grid.geojson")

TITLES = {
    "lur_baseline":               "LUR Baseline – NO₂ (µg/m³)",
    "atmo_plan_baseline":         "ATMO-Plan Baseline – NO₂ (µg/m³)",
    "fusion_predictions":         "FusionGP – Predicted NO₂ (µg/m³)",
    "fusion_uncertainty":         "FusionGP – Prediction Uncertainty (µg/m³)",
    "fusion_predictions_day_07":  "FusionGP – Day 7 NO₂ (µg/m³)",
    "fusion_predictions_day_14":  "FusionGP – Day 14 NO₂ (µg/m³)",
    "fusion_predictions_day_21":  "FusionGP – Day 21 NO₂ (µg/m³)",
    "fusion_predictions_day_28":  "FusionGP – Day 28 NO₂ (µg/m³)",
}

# Load grid once
print("Loading grid geometry …")
GRID_GDF = gpd.read_file(GRID_GEOJSON).to_crs("EPSG:3857")
print(f"  {len(GRID_GDF):,} grid cells loaded.")


def _assign_grid_ids_by_location(df):
    """Spatial join: find which grid cell each (lat, lon) point falls in."""
    points = gpd.GeoDataFrame(
        df[["value"]].copy(),
        geometry=gpd.points_from_xy(df["longitude"], df["latitude"]),
        crs="EPSG:4326",
    ).to_crs(GRID_GDF.crs)
    joined = gpd.sjoin(points, GRID_GDF[["grid_id", "geometry"]], how="left", predicate="within")
    df = df.copy()
    df["grid_id"] = joined["grid_id"].values
    return df


def make_choropleth(df, title, out_path, cmap="viridis"):
    if "grid_id" not in df.columns or df["grid_id"].isna().all():
        print(f"  No grid_id — assigning via spatial join …")
        df = _assign_grid_ids_by_location(df)
    merged = GRID_GDF.merge(df[["grid_id", "value"]].dropna(), on="grid_id", how="inner")
    if merged.empty:
        print(f"  No matching grid_ids — skipping {out_path.name}")
        return

    values = merged["value"].values
    vmin = float(np.nanpercentile(values, 2))
    vmax = float(np.nanpercentile(values, 98))

    x0, y0, x1, y1 = merged.total_bounds
    fig, ax = plt.subplots(figsize=(10, 9))
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    ctx.add_basemap(ax, source=ctx.providers.CartoDB.Positron, zoom=12, crs="EPSG:3857")
    merged.plot(column="value", ax=ax, cmap=cmap, vmin=vmin, vmax=vmax,
                linewidth=0, edgecolor="none", alpha=0.7)
    ax.set_axis_off()

    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize
    sm = ScalarMappable(cmap=cmap, norm=Normalize(vmin=vmin, vmax=vmax))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("NO₂ (µg/m³)", fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold", pad=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved choropleth: {out_path}")


def make_dots(df, title, out_path, cmap="viridis", dot_size=8, alpha=0.85):
    import pyproj
    transformer = pyproj.Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
    x, y = transformer.transform(df["longitude"].values, df["latitude"].values)
    values = df["value"].values
    vmin = float(np.nanpercentile(values, 2))
    vmax = float(np.nanpercentile(values, 98))

    fig, ax = plt.subplots(figsize=(10, 9))
    sc = ax.scatter(x, y, c=values, cmap=cmap, vmin=vmin, vmax=vmax,
                    s=dot_size, alpha=alpha, linewidths=0)
    ctx.add_basemap(ax, source=ctx.providers.CartoDB.Positron, zoom=12)
    ax.set_axis_off()
    cbar = fig.colorbar(sc, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("NO₂ (µg/m³)", fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold", pad=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved dots: {out_path}")


run_dir = Path("outputs/demo_run_20260223_085327_n300 copy")   # <-- change this
map_data = run_dir / "map_data"
gpkf_dir = run_dir / "gpkf_maps"
out_dir = run_dir / "map_images"
out_dir.mkdir(exist_ok=True)

# --- Static baseline maps (LUR, ATMO-Plan, FusionGP test-station predictions) ---
for csv in sorted(map_data.glob("*.csv")):
    df = pd.read_csv(csv).dropna(subset=["latitude", "longitude", "value"])
    if df.empty:
        print(f"Skipping {csv.name} — no valid rows")
        continue
    title = TITLES.get(csv.stem, csv.stem.replace("_", " ").title())
    make_choropleth(df, title, out_dir / f"{csv.stem}.png")

# --- GPKF sequential fusion: one map per day (mean NO2 + uncertainty) ---
if gpkf_dir.exists():
    gpkf_out = out_dir / "gpkf"
    gpkf_out.mkdir(exist_ok=True)
    gpkf_csvs = sorted(gpkf_dir.glob("gpkf_day_*.csv"))
    print(f"\nProcessing {len(gpkf_csvs)} GPKF day maps …")
    for csv in gpkf_csvs:
        df = pd.read_csv(csv).dropna(subset=["latitude", "longitude", "mean_ug_m3"])
        if df.empty:
            continue
        day_num = int(csv.stem.split("_")[-1])
        day_label = f"Day {day_num + 1}"

        # Mean NO2 map — clip to physical range
        df["mean_ug_m3"] = df["mean_ug_m3"].clip(lower=0)
        df_mean = df.rename(columns={"mean_ug_m3": "value"})
        make_choropleth(df_mean, f"FusionGP (GPKF) – NO₂ {day_label} (µg/m³)",
                        gpkf_out / f"{csv.stem}_mean.png")

        # Uncertainty map
        df_std = df.rename(columns={"std_ug_m3": "value"})
        make_choropleth(df_std, f"FusionGP (GPKF) – Uncertainty {day_label} (µg/m³)",
                        gpkf_out / f"{csv.stem}_std.png", cmap="plasma")
