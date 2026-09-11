"""Summarize leave-one-station-out FusionGP results.

This script is intentionally platform-neutral so paper reviewers can reproduce
the reported tables without shell-specific loops.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_STATIONS = ["10_94", "115_50", "121_61", "23_44", "52_41", "61_52", "76_51", "80_26", "88_52"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize LOSO result tables.")
    parser.add_argument("--outputs-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--run-prefix", default="loso_nobias")
    parser.add_argument("--stations", nargs="+", default=DEFAULT_STATIONS)
    parser.add_argument("--out-prefix", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_prefix = args.out_prefix or args.run_prefix

    rows = []
    for station in args.stations:
        table_path = args.outputs_dir / f"demo_run_{args.run_prefix}_{station}" / "final_results_table.csv"
        if not table_path.exists():
            raise FileNotFoundError(f"Missing LOSO result table: {table_path}")
        table = pd.read_csv(table_path)
        for _, row in table.iterrows():
            record = {"station": station}
            record.update(row.to_dict())
            rows.append(record)

    results = pd.DataFrame(rows)
    model_summary = (
        results.groupby("model")
        .agg(
            n=("rmse", "count"),
            rmse_mean=("rmse", "mean"),
            rmse_median=("rmse", "median"),
            mae_mean=("mae", "mean"),
            mae_median=("mae", "median"),
            bias_mean=("bias", "mean"),
            abs_bias_mean=("bias", lambda values: float(np.mean(np.abs(values)))),
            corr_mean=("corr", "mean"),
            crps_mean=("crps", "mean"),
            nll_mean=("nll", "mean"),
            cov90_mean=("coverage_90", "mean"),
        )
        .reset_index()
    )

    rmse_wide = results.pivot(index="station", columns="model", values="rmse")
    station_summary = pd.DataFrame({"station": args.stations})
    station_summary["rmse_fusion_minus_atmo"] = [
        rmse_wide.loc[station, "FusionGP"] - rmse_wide.loc[station, "ATMO-Plan"]
        for station in args.stations
    ]
    station_summary["rmse_fusion_minus_lur"] = [
        rmse_wide.loc[station, "FusionGP"] - rmse_wide.loc[station, "LUR"]
        for station in args.stations
    ]
    station_summary["rmse_winner"] = [rmse_wide.loc[station].idxmin() for station in args.stations]

    models = ["FusionGP", "LUR", "ATMO-Plan", "GPKF"]
    pooled = {model: {"se": 0.0, "ae": 0.0, "err": 0.0, "n": 0} for model in models}
    for station in args.stations:
        pred_path = args.outputs_dir / f"demo_run_{args.run_prefix}_{station}" / "predictions.csv"
        table_path = args.outputs_dir / f"demo_run_{args.run_prefix}_{station}" / "final_results_table.csv"
        if not pred_path.exists():
            raise FileNotFoundError(f"Missing LOSO predictions: {pred_path}")
        n_epa = int(pd.read_csv(pred_path, usecols=["is_epa"])["is_epa"].sum())
        table = pd.read_csv(table_path).set_index("model")
        for model in models:
            row = table.loc[model]
            pooled[model]["se"] += float(row["mse"] * n_epa)
            pooled[model]["ae"] += float(row["mae"] * n_epa)
            pooled[model]["err"] += float(row["bias"] * n_epa)
            pooled[model]["n"] += n_epa

    pooled_summary = pd.DataFrame(
        [
            {
                "model": model,
                "n": values["n"],
                "pooled_rmse": (values["se"] / values["n"]) ** 0.5,
                "pooled_mae": values["ae"] / values["n"],
                "pooled_bias": values["err"] / values["n"],
            }
            for model, values in pooled.items()
        ]
    ).sort_values("pooled_rmse")

    args.outputs_dir.mkdir(parents=True, exist_ok=True)
    model_path = args.outputs_dir / f"{out_prefix}_model_summary.csv"
    station_path = args.outputs_dir / f"{out_prefix}_station_summary.csv"
    pooled_path = args.outputs_dir / f"{out_prefix}_pooled_summary.csv"
    model_summary.to_csv(model_path, index=False)
    station_summary.to_csv(station_path, index=False)
    pooled_summary.to_csv(pooled_path, index=False)

    print(f"Wrote {model_path}")
    print(f"Wrote {station_path}")
    print(f"Wrote {pooled_path}")
    print("\nModel summary:")
    print(model_summary.sort_values("rmse_mean").to_string(index=False))
    print("\nPooled summary:")
    print(pooled_summary.to_string(index=False))


if __name__ == "__main__":
    main()
