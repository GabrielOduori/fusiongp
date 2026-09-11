"""Saving experiment summaries, metrics tables, model checkpoints, and grid-prediction exports."""

from datetime import datetime

import numpy as np
import pandas as pd
import torch

from . import config


def save_experiment_summary(experiment_dir, model, trainer, learned_params,
                            epa_metrics, epa_test, scalers, timestamp, elapsed_time=None,
                            training_sources=None, base_vs_epa_metrics=None,
                            base_vs_fused_metrics=None, epa_baseline_metrics=None,
                            epa_bias_corrected_metrics=None,
                            predictor_noise_source=None,
                            prediction_task=None,
                            noise_std_summary=None,
                            epa_r2=None,
                            base_r2=None):
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
        training_sources = config.AVAILABLE_SOURCES

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
        if config.USE_GRID_PRIOR:
            f.write(f"Prior mean: {config.GRID_PRIOR_COLUMN} (GridPriorMean, learnable_bias={config.GRID_PRIOR_LEARNABLE_BIAS})\n")
        else:
            f.write("Prior mean: None\n")
        if 'epa' not in training_sources:
            f.write(f"NOTE: EPA data was NOT used in training to avoid data leakage.\n")
            if config.HAS_SATELLITE:
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
        f.write(f"learn_kernel_hyperparams: {config.MODEL_CONFIG['learn_kernel_hyperparams']}\n")
        f.write(f"learn_noise: {config.MODEL_CONFIG['learn_noise']}\n")
        f.write(f"learn_calibration: {config.MODEL_CONFIG['learn_calibration']}\n\n")

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
            # MetricsResult.to_dict() doesn't include r2; use the values computed by the caller.
            base_dict["r2"] = base_r2 if base_r2 is not None else float('nan')
            fused_dict["r2"] = epa_r2 if epa_r2 is not None else float('nan')

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


def save_training_history(history, tables_dir):
    """Save the per-epoch training/validation history to CSV."""
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


def save_model_checkpoint(model, model_config, training_config, learned_params, timestamp, models_dir):
    """Save the trained model state dict + config to a .pth checkpoint."""
    model_path = models_dir / "fusiongp_model.pth"
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_config': model_config,
        'training_config': training_config,
        'learned_hyperparameters': learned_params,
        'timestamp': timestamp,
    }, model_path)
    print(f"   ✓ Saved model to: {model_path}")


def save_metrics_tables(per_source_metrics, epa_metrics, base_vs_epa_metrics, base_vs_fused_metrics, learned_params, tables_dir):
    """Save per-source metrics, EPA-only metrics, three-case comparison, and hyperparameters tables."""
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


def export_grid_predictions(data_path_for_run, scalers, predictor, tables_dir, primary_source):
    """Export per-grid-cell predictions across 5 timestamps to CSV.

    Returns (grid_results_df, timestamps_for_export, grid_locs).
    """
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
            task=primary_source,
        )

        timestamp_df = grid_locs.copy()
        timestamp_df['timestamp'] = timestamp
        timestamp_df['predicted_mean'] = grid_preds.mean
        timestamp_df['predicted_std'] = grid_preds.std
        timestamp_df['ci_lower_95'] = grid_preds.lower_ci[0.95]
        timestamp_df['ci_upper_95'] = grid_preds.upper_ci[0.95]

        epa_col = config.EPA_INTERP_COLUMN if config.USE_EPA_INTERPOLATED else "epa_no2"
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

    return grid_results_df, timestamps_for_export, grid_locs
