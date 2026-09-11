"""
Fast regression-based fusion (Malings-style) for thesis results.

Pipeline:
1) Build TROPOMI pattern prior over calibration window (restricted).
2) Combine with model prior: model_no2_10m + pattern.
3) Fit OLS to EPA on calibration subset.
4) Evaluate on EPA holdout.

Outputs saved to experiments/results/regression_experiment_YYYYMMDD_HHMMSS/.
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.evaluation import rmse, mae, r_squared, bias


# =============================================================================
# Configuration
# =============================================================================

# Use the vetted real dataset (no synthetic/low-cost inputs) to avoid data leakage.
DATA_PATH = Path("/media/gabriel-oduori/SERVER/dev_space/FusionGP/data_test/FusionData.csv")

GRID_PRIOR_COLUMN = "model_no_100m"
TROPOMI_COLUMN = "satellite_no2"
TROPOMI_CALIBRATION_DAYS = 14
TROPOMI_RESTRICTED = True
AUGMENTED_PRIOR_COLUMN = "model_no2_10m_tropomi"

EPA_COLUMN = "epa_no2"
HOLDOUT_EPA_FRAC = 0.2
HOLDOUT_EPA_BY_GRID = True
HOLDOUT_SEED = 42

EXPERIMENT_NAME = "regression_experiment"
RESULTS_BASE = Path("/media/gabriel-oduori/SERVER/dev_space/FusionGP/experiments/results")


# =============================================================================
# Helpers
# =============================================================================

def build_tropomi_pattern_prior(
    df: pd.DataFrame,
    model_col: str,
    tropomi_col: str,
    window_days: int = 7,
    restricted: bool = True,
    output_col: str = "model_no2_10m_tropomi",
) -> pd.DataFrame:
    """
    Build a TROPOMI-augmented prior using a typical-pattern approach.
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
    mod_mean = calib.groupby("grid_id", as_index=False)[model_col].mean()
    merged = sat_mean.merge(mod_mean, on="grid_id", how="inner")

    valid = merged[tropomi_col].notna() & merged[model_col].notna()
    x = merged.loc[valid, tropomi_col].values
    y = merged.loc[valid, model_col].values

    if len(x) < 10:
        raise ValueError("Not enough colocated satellite/model points for pattern regression")

    A = np.vstack([x, np.ones_like(x)]).T
    slope, intercept = np.linalg.lstsq(A, y, rcond=None)[0]
    merged["pattern"] = slope * merged[tropomi_col] + intercept - merged[model_col]

    pattern_map = merged[["grid_id", "pattern"]]
    df = df.merge(pattern_map, on="grid_id", how="left")
    df["pattern"] = df["pattern"].fillna(0.0)
    df[output_col] = df[model_col] + df["pattern"]
    df.drop(columns=["pattern"], inplace=True)

    return df


def make_experiment_dir() -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = RESULTS_BASE / f"{EXPERIMENT_NAME}_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "tables").mkdir(parents=True, exist_ok=True)
    return out_dir


def evaluate_metrics(y_true, y_pred):
    return {
        "rmse": rmse(y_true, y_pred),
        "mae": mae(y_true, y_pred),
        "r2": r_squared(y_true, y_pred),
        "bias": bias(y_true, y_pred),
    }


# =============================================================================
# Main
# =============================================================================

def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Data file not found: {DATA_PATH}")

    out_dir = make_experiment_dir()
    tables_dir = out_dir / "tables"

    df = pd.read_csv(DATA_PATH, low_memory=False)
    # Guard against schema drift and ensure required fields exist for regression baselines.
    required_cols = ["timestamp", "latitude", "longitude", GRID_PRIOR_COLUMN, TROPOMI_COLUMN, EPA_COLUMN]
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns in {DATA_PATH}: {missing_cols}")
    if any(c in df.columns for c in ["synthetic_no2", "base_syhthetic_no2", "low_cost_no2", "low_cost_data"]):
        print("Note: synthetic/low-cost columns detected in source file but will be ignored.")
    if "grid_id" not in df.columns:
        df["grid_id"] = np.arange(len(df))

    # Build prior: model + TROPOMI pattern
    df = build_tropomi_pattern_prior(
        df,
        model_col=GRID_PRIOR_COLUMN,
        tropomi_col=TROPOMI_COLUMN,
        window_days=TROPOMI_CALIBRATION_DAYS,
        restricted=TROPOMI_RESTRICTED,
        output_col=AUGMENTED_PRIOR_COLUMN,
    )
    # Fallback to model prior where augmented prior is missing
    df["prior_used"] = df[AUGMENTED_PRIOR_COLUMN]
    df.loc[df["prior_used"].isna(), "prior_used"] = df[GRID_PRIOR_COLUMN]

    # EPA holdout split (require prior + EPA)
    epa_mask = df[EPA_COLUMN].notna()
    prior_mask = df["prior_used"].notna()
    df_epa = df.loc[epa_mask & prior_mask].copy()
    if df_epa.empty:
        raise ValueError("No rows with both EPA and prior values available.")

    rng = np.random.default_rng(HOLDOUT_SEED)
    if HOLDOUT_EPA_BY_GRID and "grid_id" in df_epa.columns:
        grids = df_epa["grid_id"].unique()
        n_holdout = max(1, int(len(grids) * HOLDOUT_EPA_FRAC))
        holdout_grids = rng.choice(grids, size=n_holdout, replace=False)
        holdout_mask = df_epa["grid_id"].isin(holdout_grids)
    else:
        n_holdout = max(1, int(len(df_epa) * HOLDOUT_EPA_FRAC))
        holdout_idx = rng.choice(df_epa.index.to_numpy(), size=n_holdout, replace=False)
        holdout_mask = df_epa.index.isin(holdout_idx)

    df_train = df_epa.loc[~holdout_mask].copy()
    df_test = df_epa.loc[holdout_mask].copy()

    # OLS regression: EPA ~ prior
    x_train = df_train["prior_used"].values.reshape(-1, 1)
    y_train = df_train[EPA_COLUMN].values
    finite_mask = np.isfinite(x_train[:, 0]) & np.isfinite(y_train)
    x_train = x_train[finite_mask]
    y_train = y_train[finite_mask]
    if len(x_train) < 10:
        raise ValueError("Not enough finite EPA training points for OLS calibration")
    A = np.hstack([x_train, np.ones((len(x_train), 1))])
    coef, intercept = np.linalg.lstsq(A, y_train, rcond=None)[0]

    # Predictions
    x_test = df_test["prior_used"].values
    y_test = df_test[EPA_COLUMN].values
    finite_test = np.isfinite(x_test) & np.isfinite(y_test)
    x_test = x_test[finite_test]
    y_test = y_test[finite_test]
    if len(y_test) == 0:
        # fallback: evaluate on all EPA points with valid prior
        df_test = df_epa.copy()
        x_test = df_test["prior_used"].values
        y_test = df_test[EPA_COLUMN].values
        finite_test = np.isfinite(x_test) & np.isfinite(y_test)
        x_test = x_test[finite_test]
        y_test = y_test[finite_test]
        if len(y_test) == 0:
            raise ValueError("No valid EPA points available after filtering.")
    y_pred = coef * x_test + intercept
    base_pred = x_test

    # Proxy LUR baseline (multivariate linear regression using available fields)
    lur_features = ["latitude", "longitude", "prior_used", TROPOMI_COLUMN]
    for col in lur_features:
        if col not in df_epa.columns:
            df_epa[col] = np.nan
    df_epa_lur = df_epa[lur_features + [EPA_COLUMN]].copy()
    for col in lur_features:
        if df_epa_lur[col].isna().any():
            df_epa_lur[col] = df_epa_lur[col].fillna(df_epa_lur[col].mean())

    df_train_lur = df_epa_lur.loc[df_train.index.intersection(df_epa_lur.index)]
    df_test_lur = df_epa_lur.loc[df_test.index.intersection(df_epa_lur.index)]

    X_train_lur = df_train_lur[lur_features].values
    y_train_lur = df_train_lur[EPA_COLUMN].values
    X_test_lur = df_test_lur[lur_features].values
    y_test_lur = df_test_lur[EPA_COLUMN].values
    if len(y_test_lur) != len(y_test):
        X_test_lur = X_test_lur[: len(y_test)]
        y_test_lur = y_test_lur[: len(y_test)]
    A_lur = np.hstack([X_train_lur, np.ones((len(X_train_lur), 1))])
    coef_lur = np.linalg.lstsq(A_lur, y_train_lur, rcond=None)[0]
    X_test_lur_aug = np.hstack([X_test_lur, np.ones((len(X_test_lur), 1))])
    lur_pred = X_test_lur_aug @ coef_lur

    metrics = evaluate_metrics(y_test, y_pred)
    base_metrics = evaluate_metrics(y_test, base_pred)
    lur_metrics = evaluate_metrics(y_test_lur, lur_pred)
    baseline_mean = np.mean(y_test)
    baseline_pred = np.full_like(y_test, baseline_mean)
    baseline_metrics = evaluate_metrics(y_test, baseline_pred)

    # Save metrics table
    metrics_df = pd.DataFrame([
        {"case": "EPA_mean_baseline", **baseline_metrics},
        {"case": "Base_prior", **base_metrics},
        {"case": "OLS_calibrated", **metrics},
        {"case": "Proxy_LUR_linear", **lur_metrics},
    ])
    metrics_df.to_csv(tables_dir / "metrics_regression.csv", index=False)

    # Write summary
    summary_path = tables_dir / "experiment_summary.txt"
    with open(summary_path, "w") as f:
        f.write("Regression Fusion Experiment (Malings-style)\n")
        f.write("=" * 70 + "\n")
        f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Data: {DATA_PATH}\n\n")

        f.write("Methodology:\n")
        f.write("-" * 70 + "\n")
        f.write(f"Prior: {GRID_PRIOR_COLUMN} + TROPOMI pattern ({TROPOMI_CALIBRATION_DAYS} days, "
                f"{'restricted' if TROPOMI_RESTRICTED else 'full'})\n")
        f.write("Calibration: OLS regression to EPA on calibration subset\n")
        f.write("Evaluation: EPA holdout\n\n")

        f.write("OLS Parameters:\n")
        f.write("-" * 70 + "\n")
        f.write(f"coef: {coef:.6f}\n")
        f.write(f"intercept: {intercept:.6f}\n\n")

        f.write("Metrics (EPA holdout):\n")
        f.write("-" * 70 + "\n")
        for name, m in [("EPA mean baseline", baseline_metrics),
                        ("Base prior", base_metrics),
                        ("OLS calibrated", metrics),
                        ("Proxy LUR (linear)", lur_metrics)]:
            f.write(f"{name}:\n")
            for k, v in m.items():
                f.write(f"  {k}: {v:.4f}\n")
        f.write("\n")

        f.write("Data Statistics:\n")
        f.write("-" * 70 + "\n")
        f.write(f"EPA test mean={y_test.mean():.4f}, std={y_test.std():.4f}\n")

    print(f"Saved results to: {out_dir}")
    print(f"Summary: {summary_path}")
    print(f"Metrics: {tables_dir / 'metrics_regression.csv'}")


if __name__ == "__main__":
    main()
