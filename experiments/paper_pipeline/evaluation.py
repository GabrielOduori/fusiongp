"""Evaluation of FusionGP predictions against EPA ground truth."""

import numpy as np
import torch

from src.evaluation import Evaluator, rmse, mae, r_squared, bias


def _get_original_coords(data, scalers):
    """Return coordinates in original scale, handling preprocessed inputs."""
    if isinstance(getattr(data, "metadata", None), dict) and data.metadata.get("preprocessed"):
        return scalers.inverse_transform_coords(data.coords)
    return data.coords


def evaluate_against_epa(
    model,
    predictor,
    train_data,
    test_data,
    scalers,
    holdout_grid_ids,
    prior_mean,
    use_epa_in_training,
    grid_prior_column,
    grid_prior_learnable_bias,
):
    """
    Evaluate the fused model and the prior-mean baseline against held-out EPA
    ground truth (Step 6 of the original pipeline).

    Returns a dict with every metrics object/array needed by reporting and
    plotting downstream: epa_metrics, epa_baseline_metrics,
    epa_bias_corrected_metrics, base_vs_epa_metrics, base_vs_fused_metrics,
    base_at_epa, per_source_metrics, y_true_orig, epa_eval_mask,
    epa_test_orig, pred_mean_at_epa, pred_std_at_epa, training_sources.
    """
    # Determine which sources were used for training
    training_sources = [
        source for source, mask in train_data.source_masks.items()
        if mask.sum() > 0
    ]
    print(f"   Training sources: {training_sources}")
    if prior_mean is not None:
        print(f"   Prior mean: {grid_prior_column} (GridPriorMean, learnable_bias={grid_prior_learnable_bias})")
    else:
        print("   Prior mean: None")
    print(f"   Evaluation target: EPA (ground truth)")

    evaluator = Evaluator()

    epa_eval_mask = test_data.source_masks["epa"]
    if holdout_grid_ids is not None and test_data.grid_ids is not None:
        epa_eval_mask = epa_eval_mask & np.isin(test_data.grid_ids, holdout_grid_ids)
    epa_test_vals = test_data.observations["epa"][epa_eval_mask]

    if scalers.normalize_targets and 'epa' in scalers.target_std:
        epa_test_orig = epa_test_vals * scalers.target_std['epa'] + scalers.target_mean['epa']
    else:
        epa_test_orig = epa_test_vals

    task = 'epa' if use_epa_in_training else None
    pred_at_epa = predictor.predict(test_data, task=task, verbose=False)
    pred_mean_at_epa = pred_at_epa.mean[epa_eval_mask]
    pred_std_at_epa = pred_at_epa.std[epa_eval_mask]

    epa_metrics = evaluator.evaluate(
        y_true=epa_test_orig,
        y_pred_mean=pred_mean_at_epa,
        y_pred_std=pred_std_at_epa,
    )

    epa_baseline_mean = float(np.nanmean(epa_test_orig)) if len(epa_test_orig) else np.nan
    epa_baseline_pred = np.full_like(epa_test_orig, epa_baseline_mean)
    epa_baseline_metrics = {
        "rmse": rmse(epa_test_orig, epa_baseline_pred),
        "mae": mae(epa_test_orig, epa_baseline_pred),
        "r2": r_squared(epa_test_orig, epa_baseline_pred),
        "bias": bias(epa_test_orig, epa_baseline_pred),
    }

    bias_correction = float(np.nanmean(epa_test_orig - pred_mean_at_epa)) if len(epa_test_orig) else 0.0
    pred_mean_bias_corrected = pred_mean_at_epa + bias_correction
    epa_bias_corrected_metrics = {
        "rmse": rmse(epa_test_orig, pred_mean_bias_corrected),
        "mae": mae(epa_test_orig, pred_mean_bias_corrected),
        "r2": r_squared(epa_test_orig, pred_mean_bias_corrected),
        "bias": bias(epa_test_orig, pred_mean_bias_corrected),
    }

    print("\n" + "="*50)
    print("Evaluation Results (Fusion vs EPA Ground Truth)")
    print("="*50)
    print(epa_metrics.summary())
    print(f"\nEPA Mean Baseline: {epa_baseline_metrics}")
    print(f"EPA Bias-Corrected Metrics: {epa_bias_corrected_metrics}")

    base_vs_epa_metrics = None
    base_vs_fused_metrics = None
    base_at_epa = None
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
        # GridPriorMean normalizes its output to the EPA target scale
        # (target_source='epa') but never inverse-transforms it back; without
        # this, base_full stays in ~N(0,1) units while epa_test_orig is in
        # original µg/m³, making every downstream metric meaningless.
        if scalers.normalize_targets and 'epa' in scalers.target_std:
            base_full = base_full * scalers.target_std['epa'] + scalers.target_mean['epa']
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
        print("Evaluation Results (Fused vs Base at EPA locations)")
        print("="*50)
        print(base_vs_fused_metrics.summary())
    else:
        print("   ⚠ Base prior mean not available; skipping base comparisons.")

    if base_vs_epa_metrics is not None:
        base_dict = base_vs_epa_metrics.to_dict()
        fused_dict = epa_metrics.to_dict()
        # MetricsResult.to_dict() doesn't include r2; compute it directly here.
        base_dict["r2"] = r_squared(epa_test_orig, base_at_epa)
        fused_dict["r2"] = r_squared(epa_test_orig, pred_mean_at_epa)

        def pct_improve(lower_is_better: bool, base_val: float, new_val: float) -> float:
            if base_val == 0:
                return float('nan')
            if lower_is_better:
                return 100.0 * (base_val - new_val) / abs(base_val)
            return 100.0 * (new_val - base_val) / abs(base_val)

        rmse_imp = pct_improve(True, base_dict["rmse"], fused_dict["rmse"])
        mae_imp = pct_improve(True, base_dict["mae"], fused_dict["mae"])
        r2_imp = pct_improve(False, base_dict["r2"], fused_dict["r2"])

        print("\n" + "="*70)
        print("MODEL COMPARISON SUMMARY (Base → Fused at EPA locations)")
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
            f"  R²           {base_dict['r2']:12.4f} {fused_dict['r2']:12.4f} {r2_imp:12.1f}%"
        )
        print(
            f"  Bias         {base_dict['bias']:12.4f} {fused_dict['bias']:12.4f}"
        )
        print("="*70)
        print(
            f"\n  Total improvement (Base → Fused):\n"
            f"    RMSE: {rmse_imp:+.1f}%\n"
            f"    R²:   {r2_imp:+.1f}%"
        )

    per_source_metrics = {}
    y_true_orig = {}

    for source in test_data.observations.keys():
        mask = test_data.source_masks[source]
        if mask.sum() == 0:
            continue

        obs = test_data.observations[source][mask]
        if scalers.normalize_targets and source in scalers.target_std:
            y_true_orig[source] = obs * scalers.target_std[source] + scalers.target_mean[source]
        else:
            y_true_orig[source] = obs

        source_task = source if source in training_sources else None
        pred = predictor.predict(test_data, task=source_task, verbose=False)

        per_source_metrics[source] = evaluator.evaluate(
            y_true=y_true_orig[source],
            y_pred_mean=pred.mean[mask],
            y_pred_std=pred.std[mask],
        ).to_dict()
        per_source_metrics[source]["r2"] = r_squared(y_true_orig[source], pred.mean[mask])

    print("\nPer-source metrics:")
    for source, source_metrics in per_source_metrics.items():
        trained_on = "(trained)" if source in training_sources else "(held out)"
        print(
            f"  {source} {trained_on}: rmse={source_metrics['rmse']:.4f}, "
            f"mae={source_metrics['mae']:.4f}, r2={source_metrics['r2']:.4f}, "
            f"bias={source_metrics['bias']:.4f}"
        )

    return {
        "training_sources": training_sources,
        "epa_eval_mask": epa_eval_mask,
        "epa_test_orig": epa_test_orig,
        "pred_mean_at_epa": pred_mean_at_epa,
        "pred_std_at_epa": pred_std_at_epa,
        "epa_metrics": epa_metrics,
        "epa_baseline_metrics": epa_baseline_metrics,
        "epa_bias_corrected_metrics": epa_bias_corrected_metrics,
        "base_vs_epa_metrics": base_vs_epa_metrics,
        "base_vs_fused_metrics": base_vs_fused_metrics,
        "base_at_epa": base_at_epa,
        "per_source_metrics": per_source_metrics,
        "y_true_orig": y_true_orig,
    }
