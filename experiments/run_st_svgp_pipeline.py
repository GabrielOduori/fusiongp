"""
ST-SVGP pipeline scaffold (fusion-ready).

This script mirrors the SVGP demo pipeline but routes training to
STSVGPTrainer (CVI natural-gradient + filtering/smoothing).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from datetime import datetime

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data import DataLoader, DataPreprocessor
from src.models import STSVGPModel, STSVGPConfig, MultiSourceLikelihood
from src.training import STSVGPTrainer
from src.visualization import plot_predictions

try:
    import folium
    from branca.colormap import linear as _linear_colormap
except ImportError:
    folium = None
    _linear_colormap = None
from src.evaluation import rmse, mae, bias, mse
from src.evaluation.metrics import mape
from experiments.reproduce_paper_daily import build_daily_dataset_from_sources


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run ST-SVGP pipeline (scaffold)")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--n-spatial-inducing", type=int, default=100)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--out-dir", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_dir = args.data_dir
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    np.random.seed(42)
    torch.manual_seed(42)

    merged_path = build_daily_dataset_from_sources(data_dir)

    loader = DataLoader(
        merged_path,
        column_mapping={
            "grid_id": "grid_id",
            "latitude": "latitude",
            "longitude": "longitude",
            "timestamp": "timestamp",
            "satellite": "satellite_no2",
            "epa": "epa_no2",
        },
    )
    data = loader.load()

    preprocessor = DataPreprocessor(
        normalize_coords=True,
        normalize_time=True,
        normalize_targets=True,
    )
    train_data, _, test_data = preprocessor.fit_transform(
        data,
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        split_strategy="random",
        random_seed=42,
    )

    likelihood = MultiSourceLikelihood(sources=["epa", "satellite"])
    config = STSVGPConfig(n_spatial_inducing=args.n_spatial_inducing)
    model = STSVGPModel(config=config, likelihood=likelihood)

    trainer = STSVGPTrainer(model=model, n_epochs=args.epochs, lr=args.lr)
    # Create timestamped output directory (like demo pipeline)
    if args.out_dir is None:
        run_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = Path("outputs") / f"st_svgp_run_{run_tag}"
    else:
        out_dir = args.out_dir

    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    leaflet_dir = out_dir / "leaflet_maps"
    leaflet_dir.mkdir(parents=True, exist_ok=True)

    history = trainer.fit(train_data)
    if history.elbo:
        print(f"ST-SVGP last ELBO proxy: {history.elbo[-1]:.4f}")

    # Predict on full grid (normalized coords from preprocessor)
    grid_df = (
        data.metadata.get("grid_df")
        if isinstance(data.metadata.get("grid_df"), np.ndarray)
        else None
    )
    if grid_df is None:
        # Build grid from unique lat/lon in the data
        coords = data.coords
        unique_idx = np.unique(data.grid_ids, return_index=True)[1]
        grid_coords = coords[unique_idx]
        grid_ids = data.grid_ids[unique_idx]
    else:
        grid_coords = grid_df
        grid_ids = np.full(len(grid_coords), np.nan)

    # Normalize grid coords using preprocessor scalers
    grid_coords_norm = (
        (grid_coords - preprocessor.scalers.coord_min)
        / preprocessor.scalers.coord_scale
    )
    grid_coords_norm_t = torch.tensor(grid_coords_norm, dtype=model.Z_s.dtype)

    means, vars_, days = trainer.predict(grid_coords_norm_t)

    # Save per-day CSV + maps (matching demo pipeline style)
    # out_dir already created above
    maps_dir = out_dir / "st_svgp_maps"
    maps_dir.mkdir(parents=True, exist_ok=True)

    # Inverse-transform to µg/m³
    t_mean = preprocessor.scalers.target_mean.get("epa", 0.0)
    t_std = preprocessor.scalers.target_std.get("epa", 1.0)

    def _save_leaflet(path: Path, coords: np.ndarray, values: np.ndarray, title: str):
        if folium is None or _linear_colormap is None:
            return
        if len(coords) == 0:
            return
        center = [float(np.nanmean(coords[:, 0])), float(np.nanmean(coords[:, 1]))]
        m = folium.Map(location=center, zoom_start=11, tiles="cartodbpositron")
        finite = values[np.isfinite(values)]
        if len(finite) == 0:
            return
        vmin = float(finite.min())
        vmax = float(finite.max())
        if vmax <= vmin:
            vmax = vmin + 1.0
        cmap = _linear_colormap.viridis.scale(vmin, vmax)
        cmap.caption = title
        cmap.add_to(m)
        for (lat, lon), val in zip(coords, values):
            if not np.isfinite(val):
                continue
            color = cmap(val)
            folium.CircleMarker(
                location=[float(lat), float(lon)],
                radius=2,
                color=color,
                fill=True,
                fill_opacity=0.7,
                opacity=0.7,
                weight=0,
            ).add_to(m)
        m.save(path)

    n_days = means.shape[1]
    for d_idx in range(n_days):
        mean_d = means[:, d_idx].detach().cpu().numpy()
        std_d = np.sqrt(vars_[:, d_idx].detach().cpu().numpy())
        mean_d = mean_d * t_std + t_mean
        std_d = std_d * t_std
        day_val = float(days[d_idx].detach().cpu().item())

        csv_path = maps_dir / f"st_svgp_day_{d_idx:02d}.csv"
        import pandas as pd
        df_out = pd.DataFrame({
            "grid_id": grid_ids,
            "latitude": grid_coords[:, 0],
            "longitude": grid_coords[:, 1],
            "mean_ug_m3": mean_d,
            "stdev_ug_m3": std_d,
            "day": np.full(len(mean_d), day_val),
        })
        df_out.to_csv(csv_path, index=False)

        # PNG map
        fig = plot_predictions(
            grid_coords,
            mean_d,
            title=f"ST-SVGP Predictions (Day {d_idx:02d})",
        )
        fig.savefig(fig_dir / f"st_svgp_day_{d_idx:02d}.png", dpi=200, bbox_inches="tight")
        import matplotlib.pyplot as plt
        plt.close(fig)

        # Leaflet map
        _save_leaflet(
            leaflet_dir / f"st_svgp_day_{d_idx:02d}.html",
            grid_coords,
            mean_d,
            f"ST-SVGP Predictions (Day {d_idx:02d})",
        )

    # Metrics vs EPA at observation locations (test split)
    test_mask = test_data.source_masks.get("epa", None)
    if test_mask is not None and test_mask.any():
        # Predict at EPA test coords for each day, then align by timestamp
        epa_coords_norm = test_data.coords[test_mask]
        epa_coords_norm_t = torch.tensor(epa_coords_norm, dtype=model.Z_s.dtype)

        # Map each EPA obs to a day index
        unique_days = days.detach().cpu().numpy()
        obs_days = test_data.timestamps[test_mask]

        mean_by_day, var_by_day, _ = trainer.predict(epa_coords_norm_t)
        mean_by_day = mean_by_day.detach().cpu().numpy()

        y_true = test_data.observations["epa"][test_mask]
        y_pred = np.full_like(y_true, np.nan, dtype=float)
        for i, t_val in enumerate(obs_days):
            day_idx = int(np.argmin(np.abs(unique_days - t_val)))
            y_pred[i] = mean_by_day[i, day_idx]

        valid = ~np.isnan(y_true) & ~np.isnan(y_pred)
        if valid.any():
            # Convert to original units (µg/m³)
            t_mean = preprocessor.scalers.target_mean.get("epa", 0.0)
            t_std = preprocessor.scalers.target_std.get("epa", 1.0)
            y_true_u = y_true[valid] * t_std + t_mean
            y_pred_u = y_pred[valid] * t_std + t_mean

            # Build baselines (LUR, ATMO-Plan) at EPA test locations
            import pandas as pd
            merged_df = pd.read_csv(merged_path)
            if "grid_id" in merged_df.columns and "predicted_no2" in merged_df.columns:
                lur_map = (
                    merged_df[["grid_id", "predicted_no2"]]
                    .dropna(subset=["grid_id", "predicted_no2"])
                    .drop_duplicates("grid_id")
                    .set_index("grid_id")["predicted_no2"]
                )
                test_grid_ids = pd.Series(test_data.grid_ids[test_mask])
                lur_pred = test_grid_ids.map(lur_map).to_numpy(dtype=float)
            else:
                lur_pred = np.full(np.sum(test_mask), np.nan, dtype=float)

            atmo_pred = None
            atmo_path = data_dir / "atmos_plan_model_no2.csv"
            if atmo_path.exists():
                atmo_df = pd.read_csv(atmo_path)
                if "grid_id" in atmo_df.columns and "model_no2" in atmo_df.columns:
                    atmo_map = (
                        atmo_df[["grid_id", "model_no2"]]
                        .dropna(subset=["grid_id", "model_no2"])
                        .drop_duplicates("grid_id")
                        .set_index("grid_id")["model_no2"]
                    )
                    test_grid_ids = pd.Series(test_data.grid_ids[test_mask])
                    atmo_pred = test_grid_ids.map(atmo_map).to_numpy(dtype=float)

            def point_metrics(y_true_vals, y_pred_vals):
                mask = ~(np.isnan(y_true_vals) | np.isnan(y_pred_vals))
                if mask.sum() == 0:
                    return {"rmse": np.nan, "mse": np.nan, "mae": np.nan,
                            "bias": np.nan, "mape": np.nan, "corr": np.nan}
                yt = y_true_vals[mask]
                yp = y_pred_vals[mask]
                corr = np.corrcoef(yt, yp)[0, 1] if len(yt) > 1 else np.nan
                return {
                    "rmse": rmse(yt, yp),
                    "mse": mse(yt, yp),
                    "mae": mae(yt, yp),
                    "bias": bias(yt, yp),
                    "mape": mape(yt, yp),
                    "corr": corr,
                }

            metrics_st = point_metrics(y_true_u, y_pred_u)
            metrics_lur = point_metrics(y_true_u, lur_pred)
            metrics_rows = [
                {"model": "ST-SVGP", **metrics_st},
                {"model": "LUR", **metrics_lur},
            ]
            if atmo_pred is not None:
                metrics_atmo = point_metrics(y_true_u, atmo_pred)
                metrics_rows.append({"model": "ATMO-Plan", **metrics_atmo})

            # Save baseline comparison table (matching SVGP/GPKF format)
            comp_lines = []
            comp_lines.append(
                f"BASELINE COMPARISONS (µg/m³, EPA mean={y_true_u.mean():.1f}, std={y_true_u.std():.1f})"
            )
            comp_lines.append("  Source                    RMSE(µg/m³) MAE(µg/m³)    Corr")
            for row in metrics_rows:
                comp_lines.append(
                    f"  {row['model']:<24} {row['rmse']:>10.2f} {row['mae']:>11.2f} "
                    f"{row['corr']:>8.4f}"
                )
            comp_text = "\\n".join(comp_lines)
            comp_path = out_dir / "baseline_comparison_table.txt"
            comp_path.write_text(comp_text)
            print(f"Baseline comparison table saved to {comp_path}")

            # Save metrics_baselines.csv
            import pandas as pd
            metrics_table = pd.DataFrame(metrics_rows)
            metrics_table_path = out_dir / "metrics_baselines.csv"
            metrics_table.to_csv(metrics_table_path, index=False)
            print(f"Baseline metrics table saved to {metrics_table_path}")

            # Final results table (CSV + Markdown)
            final_table = metrics_table.copy()
            final_table_path = out_dir / "final_results_table.csv"
            final_table.to_csv(final_table_path, index=False)
            headers = list(final_table.columns)
            rows = final_table.values.tolist()
            md_lines = []
            md_lines.append("| " + " | ".join(headers) + " |")
            md_lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
            for row in rows:
                md_lines.append("| " + " | ".join(str(v) for v in row) + " |")
            final_md_path = out_dir / "final_results_table.md"
            final_md_path.write_text("\n".join(md_lines))

            # Model comparison summary (ST-SVGP vs baselines)
            def format_comparison(title, base_metrics, fused_metrics):
                def pct_improve(base, fused):
                    if np.isnan(base) or base == 0:
                        return "n/a"
                    return f"{(1.0 - fused / base) * 100:>6.1f}%"

                lines = []
                lines.append("=" * 70)
                lines.append(title)
                lines.append("=" * 70)
                lines.append("  Metric              Base           Fused   Fused vs Base")
                lines.append(f"  RMSE          {base_metrics['rmse']:>10.4f}  {fused_metrics['rmse']:>10.4f}   {pct_improve(base_metrics['rmse'], fused_metrics['rmse'])}")
                lines.append(f"  MAE           {base_metrics['mae']:>10.4f}  {fused_metrics['mae']:>10.4f}   {pct_improve(base_metrics['mae'], fused_metrics['mae'])}")
                lines.append(f"  Bias          {base_metrics['bias']:>10.4f}  {fused_metrics['bias']:>10.4f}")
                lines.append("=" * 70)
                return "\n".join(lines)

            metrics_st = metrics_rows[0]
            metrics_lur = metrics_rows[1]
            summary_lines = []
            summary_lines.append(format_comparison(
                "MODEL COMPARISON SUMMARY (LUR → ST-SVGP at EPA locations)",
                metrics_lur,
                metrics_st,
            ))
            if len(metrics_rows) > 2:
                metrics_atmo = metrics_rows[2]
                summary_lines.append(format_comparison(
                    "MODEL COMPARISON SUMMARY (ATMO-Plan → ST-SVGP at EPA locations)",
                    metrics_atmo,
                    metrics_st,
                ))

            summary_text = "\n\n".join(summary_lines)
            summary_path = out_dir / "model_comparison_summary.txt"
            summary_path.write_text(summary_text)

            # Final report (Markdown)
            report_lines = []
            report_lines.append("# Model Validation Report")
            report_lines.append("")
            report_lines.append("## Summary Table (EPA Test Locations)")
            report_lines.append("")
            report_lines.extend(md_lines)
            report_lines.append("")
            report_lines.append("## Improvement Summary")
            report_lines.append("")
            report_lines.append(summary_text)
            report_path = out_dir / "final_results_report.md"
            report_path.write_text("\n".join(report_lines))


if __name__ == "__main__":
    main()
