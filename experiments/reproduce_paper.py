"""
Reproduce Paper Results

This script reproduces the main experimental results from the FusionGP paper.
It performs multi-source NO₂ fusion using EPA monitors, low-cost sensors, and
satellite retrievals with full evaluation and visualization.

Key Features:
- High-capacity SVGP model (800 inducing points)
- Fixed hyperparameters for reproducibility
- IDW baseline comparison
- Comprehensive evaluation metrics per source
- Enhanced spatial maps with contours
- Timestamped experiment outputs

Usage:
    python reproduce_paper.py

Outputs:
    results/experiment_YYYYMMDD_HHMMSS/
    ├── experiment_summary.txt        # Configuration and metrics
    ├── predictions.png               # Mean predictions
    ├── uncertainty.png               # Uncertainty estimates
    ├── uncertainty_ci.png            # Confidence intervals
    ├── predictions_surface.png       # Gridded mean surface
    ├── uncertainty_surface.png       # Gridded uncertainty surface
    └── spatial_maps/                 # Time-series spatial maps
        ├── mean_t0.png
        ├── std_t0.png
        └── ...

Expected Runtime: ~5-10 minutes (CPU with 300 epochs)
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
from sklearn.neighbors import NearestNeighbors

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "fusiongp"))

from fusiongp.data import DataLoader, DataPreprocessor
from fusiongp.models import FusionSVGP
from fusiongp.training import Trainer
from fusiongp.inference import Predictor
from fusiongp.evaluation import Evaluator, rmse, mae, r_squared, bias
from fusiongp.visualization import plot_predictions, plot_uncertainty, plot_confidence_intervals
from fusiongp.visualization.spatial_maps import create_spatial_maps


# =============================================================================
# Configuration
# =============================================================================

# Data path (modify if needed)
DATA_PATH = Path(__file__).parent.parent / "fusiongp/notebooks/synthetic_no2_data.csv"

# Model hyperparameters (paper configuration)
MODEL_CONFIG = {
    "n_inducing": 800,  # High capacity for capturing local variations
    "kernel_type": "matern32",
    "learn_inducing_locations": True,
    "learn_kernel_hyperparams": False,  # Fixed for reproducibility
    "learn_noise": False,
    "learn_calibration": False,
    "initial_lengthscales": {
        "spatial_x": 0.1,
        "spatial_y": 0.1,
        "temporal": 0.1,
    },
    "initial_noise": {
        "epa": 0.8,      # Tighter for reference measurements
        "low_cost": 1.5,  # Higher for biased sensors
        "satellite": 1.2,  # Moderate for satellite
    },
}

# Training hyperparameters
TRAINING_CONFIG = {
    "learning_rate": 0.0005,
    "n_epochs": 300,
    "batch_size": 1024,
}

# Grid prediction resolution
GRID_RESOLUTION = 0.01

# IDW baseline parameters
IDW_K_NEIGHBORS = 8
IDW_POWER = 2.0


# =============================================================================
# Helper Functions
# =============================================================================

def idw_predict(train_coords, train_vals, query_coords, k=8, power=2.0, eps=1e-12):
    """
    Inverse Distance Weighting baseline interpolation.

    Args:
        train_coords: Training coordinates (N, D)
        train_vals: Training values (N,)
        query_coords: Query coordinates (M, D)
        k: Number of neighbors
        power: IDW power parameter
        eps: Small epsilon for numerical stability

    Returns:
        predictions: Interpolated values (M,)
    """
    if len(train_coords) == 0:
        raise ValueError("No training points for IDW")

    k = min(k, len(train_coords))
    nn = NearestNeighbors(n_neighbors=k)
    nn.fit(train_coords)
    dists, idxs = nn.kneighbors(query_coords)

    preds = np.empty(len(query_coords))
    for i in range(len(query_coords)):
        if (dists[i] < eps).any():
            # Exact match
            preds[i] = train_vals[idxs[i][dists[i] < eps][0]]
            continue
        weights = 1.0 / np.power(dists[i] + eps, power)
        w_sum = weights.sum()
        preds[i] = np.dot(weights, train_vals[idxs[i]]) / w_sum if w_sum > 0 else np.nan

    return preds


def save_experiment_summary(experiment_dir, model, trainer, learned_params,
                            epa_metrics, epa_train, epa_test, scalers, timestamp):
    """Save experiment configuration and results to text file."""
    summary_file = experiment_dir / "experiment_summary.txt"

    with open(summary_file, 'w') as f:
        f.write(f"FusionGP Paper Reproduction Experiment\n")
        f.write(f"{'='*70}\n")
        f.write(f"Timestamp: {timestamp}\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        f.write(f"Model Configuration:\n")
        f.write(f"{'-'*70}\n")
        f.write(f"n_inducing: {model.n_inducing}\n")
        f.write(f"kernel_type: {model.kernel_type}\n")
        f.write(f"learn_inducing_locations: {model.learn_inducing_locations}\n")
        f.write(f"learn_kernel_hyperparams: {MODEL_CONFIG['learn_kernel_hyperparams']}\n")
        f.write(f"learn_noise: {MODEL_CONFIG['learn_noise']}\n")
        f.write(f"learn_calibration: {MODEL_CONFIG['learn_calibration']}\n\n")

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

        f.write(f"Evaluation Metrics (EPA only):\n")
        f.write(f"{'-'*70}\n")
        for key, val in epa_metrics.to_dict().items():
            if isinstance(val, (int, float, np.number)):
                f.write(f"{key}: {val:.4f}\n")
            else:
                f.write(f"{key}: {val}\n")
        f.write(f"\n")

        f.write(f"Data Statistics:\n")
        f.write(f"{'-'*70}\n")
        f.write(f"Train EPA (normalized): mean={epa_train.mean():.4f}, std={epa_train.std():.4f}\n")
        f.write(f"Test EPA (normalized): mean={epa_test.mean():.4f}, std={epa_test.std():.4f}\n")
        f.write(f"Target scaler: mean={scalers.target_mean['epa']:.4f}, std={scalers.target_std['epa']:.4f}\n")

    return summary_file


# =============================================================================
# Main Experiment
# =============================================================================

def main():
    print("="*70)
    print("FusionGP Paper Reproduction Experiment")
    print("="*70)

    # -------------------------------------------------------------------------
    # 1. Load and Preprocess Data
    # -------------------------------------------------------------------------
    print("\n[Step 1/8] Loading and preprocessing data...")

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Data file not found at {DATA_PATH}. "
            "Please ensure synthetic_no2_data.csv exists."
        )

    loader = DataLoader(str(DATA_PATH))
    data = loader.load()

    preprocessor = DataPreprocessor(normalize_targets=True)
    train_data, val_data, test_data = preprocessor.fit_transform(data)
    scalers = preprocessor.get_scalers()

    print(f"   ✓ Loaded {len(data.coords)} total observations")
    print(f"   ✓ Train: {len(train_data.coords)}, Val: {len(val_data.coords)}, Test: {len(test_data.coords)}")

    # -------------------------------------------------------------------------
    # 2. Initialize Model
    # -------------------------------------------------------------------------
    print("\n[Step 2/8] Initializing FusionSVGP model...")

    model = FusionSVGP(**MODEL_CONFIG)
    print(f"   ✓ Model initialized with {model.n_inducing} inducing points")
    print(model)

    # -------------------------------------------------------------------------
    # 3. Train Model
    # -------------------------------------------------------------------------
    print("\n[Step 3/8] Training model...")

    trainer = Trainer(model, **TRAINING_CONFIG)
    print(f"   ✓ Training config -> lr={trainer.learning_rate:.2e}, "
          f"epochs={trainer.n_epochs}, batch_size={trainer.batch_size}")

    trainer.fit(train_data, val_data=val_data)

    # -------------------------------------------------------------------------
    # 4. Check Learned Parameters
    # -------------------------------------------------------------------------
    print("\n[Step 4/8] Reviewing learned hyperparameters...")

    learned_params = model.get_hyperparameters()
    print("\n" + "="*50)
    print("Learned Hyperparameters:")
    print("="*50)
    for key, val in learned_params.items():
        print(f"{key}: {val}")
    print("="*50 + "\n")

    # Data statistics
    print("="*50)
    print("Data Statistics:")
    print("="*50)
    epa_train = train_data.observations['epa'][train_data.source_masks['epa']]
    epa_test = test_data.observations['epa'][test_data.source_masks['epa']]
    print(f"Train EPA (normalized): mean={epa_train.mean():.4f}, std={epa_train.std():.4f}, "
          f"range=[{epa_train.min():.4f}, {epa_train.max():.4f}]")
    print(f"Test EPA (normalized): mean={epa_test.mean():.4f}, std={epa_test.std():.4f}, "
          f"range=[{epa_test.min():.4f}, {epa_test.max():.4f}]")
    print(f"Target scaler: mean={scalers.target_mean['epa']:.4f}, std={scalers.target_std['epa']:.4f}")
    print(f"Train coords range: lat=[{train_data.coords[:,0].min():.4f}, {train_data.coords[:,0].max():.4f}], "
          f"lon=[{train_data.coords[:,1].min():.4f}, {train_data.coords[:,1].max():.4f}]")
    print(f"Train time range: [{train_data.timestamps.min():.4f}, {train_data.timestamps.max():.4f}]")
    print("="*50 + "\n")

    # -------------------------------------------------------------------------
    # 5. Make Predictions
    # -------------------------------------------------------------------------
    print("\n[Step 5/8] Making predictions...")

    # Create predictor WITH observation noise
    predictor = Predictor(model, scalers, include_observation_noise=True, noise_source='epa')

    # Point predictions on test set
    predictions = predictor.predict(test_data)

    # Diagnostics
    y_norm = test_data.observations['epa']
    print(f"   y_norm range/mean: {np.nanmin(y_norm):.4f}, {np.nanmax(y_norm):.4f}, {np.nanmean(y_norm):.4f}")
    print(f"   pred mean range/mean (orig scale): {predictions.mean.min():.4f}, "
          f"{predictions.mean.max():.4f}, {predictions.mean.mean():.4f}")
    print(f"   pred std range/mean (orig scale): {predictions.std.min():.4f}, "
          f"{predictions.std.max():.4f}, {predictions.std.mean():.4f}")
    print(f"   mask counts: {k: v.sum() for k, v in test_data.source_masks.items()}")

    # Grid predictions for spatial maps
    lat_min, lat_max = data.coords[:, 0].min(), data.coords[:, 0].max()
    lon_min, lon_max = data.coords[:, 1].min(), data.coords[:, 1].max()
    timestamps = np.unique(data.timestamps)

    print(f"   ✓ Generating grid predictions (resolution={GRID_RESOLUTION})...")
    grid_predictions = predictor.predict_grid(
        lat_range=(lat_min, lat_max),
        lon_range=(lon_min, lon_max),
        timestamps=timestamps,
        resolution=GRID_RESOLUTION,
    )

    # -------------------------------------------------------------------------
    # 6. IDW Baseline Comparison
    # -------------------------------------------------------------------------
    print("\n[Step 6/8] Computing IDW baseline...")

    epa_train_mask = train_data.source_masks['epa']
    epa_test_mask = test_data.source_masks['epa']
    epa_train_coords = train_data.coords[epa_train_mask]
    epa_train_vals = train_data.observations['epa'][epa_train_mask]
    epa_test_coords = test_data.coords[epa_test_mask]
    epa_test_vals = test_data.observations['epa'][epa_test_mask]

    if len(epa_train_coords) and len(epa_test_coords):
        # IDW on normalized values
        idw_preds_norm = idw_predict(epa_train_coords, epa_train_vals,
                                     epa_test_coords, k=IDW_K_NEIGHBORS, power=IDW_POWER)

        # Denormalize for evaluation
        if scalers.normalize_targets and 'epa' in scalers.target_std:
            scale = scalers.target_std['epa']
            offset = scalers.target_mean['epa']
            epa_test_vals_orig = epa_test_vals * scale + offset
            idw_preds = idw_preds_norm * scale + offset
        else:
            epa_test_vals_orig = epa_test_vals
            idw_preds = idw_preds_norm

        baseline_metrics = {
            'rmse': rmse(epa_test_vals_orig, idw_preds),
            'mae': mae(epa_test_vals_orig, idw_preds),
            'r2': r_squared(epa_test_vals_orig, idw_preds),
            'bias': bias(epa_test_vals_orig, idw_preds),
        }
        print(f"   ✓ IDW baseline (EPA only): {baseline_metrics}")
    else:
        print("   ⚠ Skipping IDW baseline: no EPA points in train/test split.")

    # -------------------------------------------------------------------------
    # 7. Evaluate FusionGP
    # -------------------------------------------------------------------------
    print("\n[Step 7/8] Evaluating FusionGP performance...")

    # Denormalize targets
    if scalers.normalize_targets:
        y_true_orig = {
            src: obs * scalers.target_std[src] + scalers.target_mean[src]
            for src, obs in test_data.observations.items()
        }
    else:
        y_true_orig = test_data.observations

    evaluator = Evaluator()

    # Per-source metrics
    metrics = evaluator.evaluate_per_source(
        y_true=y_true_orig,
        y_pred_mean=predictions.mean,
        y_pred_std=predictions.std,
        source_masks=test_data.source_masks,
    )
    print("\n" + metrics.summary())

    # EPA-only metrics (primary evaluation)
    epa_mask = test_data.source_masks["epa"]
    epa_metrics = evaluator.evaluate(
        y_true=y_true_orig["epa"][epa_mask],
        y_pred_mean=predictions.mean[epa_mask],
        y_pred_std=predictions.std[epa_mask],
    )
    print("\n   EPA-only metrics:")
    for key, val in epa_metrics.to_dict().items():
        if isinstance(val, (int, float, np.number)):
            print(f"      {key}: {val:.4f}")

    # -------------------------------------------------------------------------
    # 8. Visualize and Save Results
    # -------------------------------------------------------------------------
    print("\n[Step 8/8] Creating visualizations and saving results...")

    # Create timestamped experiment folder
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_results_dir = Path(__file__).resolve().parent / "results"
    experiment_dir = base_results_dir / f"experiment_{timestamp}"
    experiment_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"Saving results to: {experiment_dir}")
    print(f"{'='*70}\n")

    # Save experiment summary
    summary_file = save_experiment_summary(
        experiment_dir, model, trainer, learned_params,
        epa_metrics, epa_train, epa_test, scalers, timestamp
    )
    print(f"   ✓ Experiment summary saved to: {summary_file}")

    # Basic prediction plots
    plot_predictions(
        coords=grid_predictions.coords,
        values=grid_predictions.mean,
        save_path=str(experiment_dir / "predictions.png"),
    )
    print(f"   ✓ Saved predictions plot")

    plot_uncertainty(
        coords=grid_predictions.coords,
        std=grid_predictions.std,
        save_path=str(experiment_dir / "uncertainty.png"),
    )
    print(f"   ✓ Saved uncertainty plot")

    # EPA confidence interval plot
    epa_mask = test_data.source_masks["epa"]
    if epa_mask.any():
        y_epa = y_true_orig["epa"][epa_mask]
        x_idx = np.arange(len(y_epa))
        fig_ci = plot_confidence_intervals(
            x=x_idx,
            mean=predictions.mean[epa_mask],
            std=predictions.std[epa_mask],
            y_true=y_epa,
            confidence_levels=[0.5, 0.8, 0.9, 0.95],
            xlabel="EPA test index",
            ylabel="NO₂ (ppb)",
            title="EPA Predictions with Confidence Intervals",
            save_path=str(experiment_dir / "uncertainty_ci.png"),
        )
        plt.close(fig_ci)
        print(f"   ✓ Saved confidence intervals plot")

    # Enhanced spatial maps with contours
    print("\n   Creating enhanced spatial maps...")
    create_spatial_maps(
        predictor=predictor,
        scalers=scalers,
        lat_range=(lat_min, lat_max),
        lon_range=(lon_min, lon_max),
        n_times=4,
        nx=100,
        ny=100,
        output_dir=str(experiment_dir / 'spatial_maps'),
        cmap_mean='RdYlBu_r',
        cmap_std='plasma',
    )
    print(f"   ✓ Saved spatial maps to: {experiment_dir / 'spatial_maps'}")

    # Gridded surface plots (first timestamp)
    ts0 = np.unique(grid_predictions.timestamps)[0]
    mask = grid_predictions.timestamps == ts0
    lat_unique = np.unique(grid_predictions.coords[mask][:, 0])
    lon_unique = np.unique(grid_predictions.coords[mask][:, 1])

    mean_grid = grid_predictions.mean[mask].reshape(len(lat_unique), len(lon_unique))
    std_grid = grid_predictions.std[mask].reshape(len(lat_unique), len(lon_unique))

    # Mean surface
    fig, ax = plt.subplots(figsize=(8, 6))
    pcm = ax.pcolormesh(lon_unique, lat_unique, mean_grid, shading="auto", cmap="RdYlBu_r")
    fig.colorbar(pcm, ax=ax, label="Mean NO₂ (ppb)")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title(f"Gridded Mean Predictions (t={ts0:.2f})")
    fig.savefig(experiment_dir / "predictions_surface.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"   ✓ Saved mean surface plot")

    # Uncertainty surface
    fig, ax = plt.subplots(figsize=(8, 6))
    pcm = ax.pcolormesh(lon_unique, lat_unique, std_grid, shading="auto", cmap="YlOrRd")
    fig.colorbar(pcm, ax=ax, label="Std Dev (ppb)")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title(f"Gridded Uncertainty (t={ts0:.2f})")
    fig.savefig(experiment_dir / "uncertainty_surface.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"   ✓ Saved uncertainty surface plot")

    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------
    print("\n" + "="*70)
    print("Paper Reproduction Experiment Complete!")
    print("="*70)
    print(f"Results directory: {experiment_dir}")
    print("\nKey Performance Metrics:")
    print(f"  • RMSE: {epa_metrics.to_dict()['rmse']:.4f} ppb")
    print(f"  • MAE:  {epa_metrics.to_dict()['mae']:.4f} ppb")
    print(f"  • R²:   {epa_metrics.to_dict()['r_squared']:.4f}")
    print(f"  • Bias: {epa_metrics.to_dict()['bias']:.4f} ppb")
    print("="*70)


if __name__ == "__main__":
    main()
