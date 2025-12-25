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
    ├── figures/
    │   ├── training_curves.png       # Loss curves and convergence
    │   ├── predictions.png           # Mean predictions
    │   ├── uncertainty.png           # Uncertainty estimates
    │   ├── uncertainty_ci.png        # Confidence intervals
    │   ├── predictions_surface.png   # Gridded mean surface
    │   ├── uncertainty_surface.png   # Gridded uncertainty surface
    │   └── spatial_maps/             # Time-series spatial maps
    │       ├── mean_t0.png
    │       ├── std_t0.png
    │       └── ...
    ├── models/
    │   └── fusiongp_model.pth        # Trained model checkpoint
    └── tables/
        ├── experiment_summary.txt    # Configuration and metrics
        ├── training_history.csv      # Training loss per epoch
        ├── metrics_per_source.csv    # Per-source metrics
        ├── metrics_epa_only.csv      # EPA-only metrics
        └── learned_hyperparameters.csv

Expected Runtime: ~5-10 minutes (CPU with 300 epochs)
"""

import sys
import time
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data import DataLoader, DataPreprocessor
from src.models import FusionSVGP
from src.training import Trainer
from src.inference import Predictor
from src.evaluation import Evaluator, rmse, mae, r_squared, bias
from src.visualization import plot_predictions, plot_uncertainty, plot_confidence_intervals
from src.visualization.spatial_maps import create_spatial_maps


# =============================================================================
# Configuration
# =============================================================================

# Data path (modify if needed)
# DATA_PATH = Path(__file__).parent.parent / "data/test_data.csv"
DATA_PATH = Path(__file__).parent.parent / "data" / "dublin_realistic_no2.csv"

# Model hyperparameters (paper configuration)
MODEL_CONFIG = {
    "n_inducing": 800,  # High capacity for capturing local variations
    "kernel_type": "matern32",
    "learn_inducing_locations": True,
    "learn_kernel_hyperparams": True,  # Enable learning for better fit
    "learn_noise": True,  # Learn noise levels from data
    "learn_calibration": True,  # Learn calibration for low-cost sensors
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
    "learning_rate": 0.01,  # Higher learning rate for faster convergence
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


def plot_training_curves(history, save_path):
    """
    Plot training and validation loss curves.

    Args:
        history: TrainingHistory object from trainer
        save_path: Path to save the plot
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Training and Validation Loss
    epochs_train = np.arange(1, len(history.train_loss) + 1)
    ax1.plot(epochs_train, history.train_loss, 'b-', linewidth=2, label='Training Loss', alpha=0.8)

    if history.val_loss:
        # Validation is computed every val_interval epochs
        val_interval = len(history.train_loss) // len(history.val_loss) if len(history.val_loss) > 1 else 5
        epochs_val = np.arange(val_interval, len(history.train_loss) + 1, val_interval)[:len(history.val_loss)]
        ax1.plot(epochs_val, history.val_loss, 'r-', linewidth=2, label='Validation Loss', alpha=0.8, marker='o')

    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss (Negative ELBO)', fontsize=12)
    ax1.set_title('Training Progress', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    # Highlight best epoch if validation exists
    if history.val_loss:
        best_epoch = history.best_epoch
        best_loss = history.best_val_loss
        ax1.axvline(x=epochs_val[best_epoch], color='green', linestyle='--', linewidth=1.5, alpha=0.7, label='Best Model')
        ax1.scatter([epochs_val[best_epoch]], [best_loss], color='green', s=100, zorder=5, marker='*')
        ax1.legend(fontsize=10)

    # Plot 2: Learning Rate Schedule
    ax2.plot(epochs_train, history.learning_rates, 'g-', linewidth=2, alpha=0.8)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Learning Rate', fontsize=12)
    ax2.set_title('Learning Rate Schedule', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.set_yscale('log')

    plt.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

    return fig


def save_experiment_summary(experiment_dir, model, trainer, learned_params,
                            epa_metrics, epa_train, epa_test, scalers, timestamp, elapsed_time=None):
    """Save experiment configuration and results to text file."""
    summary_file = experiment_dir / "experiment_summary.txt"

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
    # Start timing
    start_time = time.time()

    print("="*70)
    print("FusionGP Paper Reproduction Experiment")
    print("="*70)

    # Create main progress bar for experiment steps
    total_steps = 8
    main_pbar = tqdm(total=total_steps, desc="Experiment Progress", position=0, leave=True,
                     bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} steps [{elapsed}<{remaining}]')

    # -------------------------------------------------------------------------
    # 1. Load and Preprocess Data
    # -------------------------------------------------------------------------
    main_pbar.set_description("Step 1/8: Loading data")
    print("\n[Step 1/8] Loading and preprocessing data...")

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Data file not found at {DATA_PATH}. "
            "Please ensure data exists."
        )

    loader = DataLoader(
        str(DATA_PATH),
        column_mapping={
            "grid_id": "grid_id",
            "latitude": "latitude",
            "longitude": "longitude",
            "timestamp": "timestamp",
            "satellite": "satellite_values",
            "low_cost": "low_cost_data",
            "epa": "epa_no2",
        },
    )
    data = loader.load()

    preprocessor = DataPreprocessor(normalize_targets=True)
    train_data, val_data, test_data = preprocessor.fit_transform(data)
    scalers = preprocessor.get_scalers()

    print(f"   ✓ Loaded {len(data.coords)} total observations")
    print(f"   ✓ Train: {len(train_data.coords)}, Val: {len(val_data.coords)}, Test: {len(test_data.coords)}")
    main_pbar.update(1)

    # -------------------------------------------------------------------------
    # 2. Initialize Model
    # -------------------------------------------------------------------------
    main_pbar.set_description("Step 2/8: Initializing model")
    print("\n[Step 2/8] Initializing FusionSVGP model...")

    model = FusionSVGP(**MODEL_CONFIG)
    print(f"   ✓ Model initialized with {model.n_inducing} inducing points")
    print(model)
    main_pbar.update(1)

    # -------------------------------------------------------------------------
    # 3. Train Model
    # -------------------------------------------------------------------------
    main_pbar.set_description("Step 3/8: Training model")
    print("\n[Step 3/8] Training model...")

    trainer = Trainer(model, **TRAINING_CONFIG)
    print(f"   ✓ Training config -> lr={trainer.learning_rate:.2e}, "
          f"epochs={trainer.n_epochs}, batch_size={trainer.batch_size}")

    history = trainer.fit(train_data, val_data=val_data)
    print(f"   ✓ Training complete. Best validation loss: {history.best_val_loss:.4f} at epoch {history.best_epoch + 1}")
    main_pbar.update(1)

    # -------------------------------------------------------------------------
    # 4. Check Learned Parameters
    # -------------------------------------------------------------------------
    main_pbar.set_description("Step 4/8: Reviewing hyperparameters")
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
    main_pbar.update(1)

    # -------------------------------------------------------------------------
    # 5. Make Predictions
    # -------------------------------------------------------------------------
    main_pbar.set_description("Step 5/8: Making predictions")
    print("\n[Step 5/8] Making predictions...")

    # Create predictor WITHOUT observation noise for evaluation
    # We want to evaluate the latent function, not noisy observations
    predictor = Predictor(model, scalers, include_observation_noise=False)

    # Point predictions on test set
    predictions = predictor.predict(test_data)

    # Diagnostics
    y_norm = test_data.observations['epa']
    print(f"   y_norm range/mean: {np.nanmin(y_norm):.4f}, {np.nanmax(y_norm):.4f}, {np.nanmean(y_norm):.4f}")
    print(f"   pred mean range/mean (orig scale): {predictions.mean.min():.4f}, "
          f"{predictions.mean.max():.4f}, {predictions.mean.mean():.4f}")
    print(f"   pred std range/mean (orig scale): {predictions.std.min():.4f}, "
          f"{predictions.std.max():.4f}, {predictions.std.mean():.4f}")
    print(f"   mask counts: {dict((k, v.sum()) for k, v in test_data.source_masks.items())}")

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
    main_pbar.update(1)

    # -------------------------------------------------------------------------
    # 6. IDW Baseline Comparison
    # -------------------------------------------------------------------------
    main_pbar.set_description("Step 6/8: Computing IDW baseline")
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
    main_pbar.update(1)

    # -------------------------------------------------------------------------
    # 7. Evaluate FusionGP
    # -------------------------------------------------------------------------
    main_pbar.set_description("Step 7/8: Evaluating performance")
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
    main_pbar.update(1)

    # -------------------------------------------------------------------------
    # 8. Visualize and Save Results
    # -------------------------------------------------------------------------
    main_pbar.set_description("Step 8/8: Saving results")
    print("\n[Step 8/8] Creating visualizations and saving results...")

    # Create timestamped experiment folder with organized subdirectories
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_results_dir = Path(__file__).resolve().parent / "results"
    experiment_dir = base_results_dir / f"experiment_{timestamp}"

    # Subdirectories for organized output
    figures_dir = experiment_dir / "figures"
    models_dir = experiment_dir / "models"
    tables_dir = experiment_dir / "tables"

    figures_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"Saving results to: {experiment_dir}")
    print(f"  - Figures: {figures_dir}")
    print(f"  - Models: {models_dir}")
    print(f"  - Tables: {tables_dir}")
    print(f"{'='*70}\n")

    # Save experiment summary to tables directory (with current runtime)
    current_elapsed = time.time() - start_time
    summary_file = save_experiment_summary(
        tables_dir, model, trainer, learned_params,
        epa_metrics, epa_train, epa_test, scalers, timestamp, current_elapsed
    )
    print(f"   ✓ Experiment summary saved to: {summary_file}")

    # Plot training curves - save to figures directory
    plot_training_curves(
        history=history,
        save_path=str(figures_dir / "training_curves.png")
    )
    print(f"   ✓ Saved training curves plot")

    # Save training history as CSV - save to tables directory
    history_dict = history.to_dict()
    # Create DataFrame with aligned data (validation is computed every val_interval)
    max_len = len(history_dict['train_loss'])
    history_data = {
        'epoch': list(range(1, max_len + 1)),
        'train_loss': history_dict['train_loss'],
        'learning_rate': history_dict['learning_rate'],
        'epoch_time': history_dict['epoch_time'],
    }
    # Add validation loss (sparse, only at validation epochs)
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

    # Basic prediction plots - save to figures directory
    plot_predictions(
        coords=grid_predictions.coords,
        values=grid_predictions.mean,
        save_path=str(figures_dir / "predictions.png"),
    )
    print(f"   ✓ Saved predictions plot")

    plot_uncertainty(
        coords=grid_predictions.coords,
        std=grid_predictions.std,
        save_path=str(figures_dir / "uncertainty.png"),
    )
    print(f"   ✓ Saved uncertainty plot")

    # EPA confidence interval plot - save to figures directory
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
            ylabel="NO₂ (µg/m³)",
            title="EPA Predictions with Confidence Intervals",
            save_path=str(figures_dir / "uncertainty_ci.png"),
        )
        plt.close(fig_ci)
        print(f"   ✓ Saved confidence intervals plot")

    # Enhanced spatial maps with contours - save to figures directory
    print("\n   Creating enhanced spatial maps...")
    create_spatial_maps(
        predictor=predictor,
        scalers=scalers,
        lat_range=(lat_min, lat_max),
        lon_range=(lon_min, lon_max),
        n_times=4,
        nx=100,
        ny=100,
        output_dir=str(figures_dir / 'spatial_maps'),
        cmap_mean='RdYlBu_r',
        cmap_std='plasma',
    )
    print(f"   ✓ Saved spatial maps to: {figures_dir / 'spatial_maps'}")

    # Gridded surface plots (first timestamp)
    ts0 = np.unique(grid_predictions.timestamps)[0]
    mask = grid_predictions.timestamps == ts0
    lat_unique = np.unique(grid_predictions.coords[mask][:, 0])
    lon_unique = np.unique(grid_predictions.coords[mask][:, 1])

    mean_grid = grid_predictions.mean[mask].reshape(len(lat_unique), len(lon_unique))
    std_grid = grid_predictions.std[mask].reshape(len(lat_unique), len(lon_unique))

    # Mean surface - save to figures directory
    fig, ax = plt.subplots(figsize=(8, 6))
    pcm = ax.pcolormesh(lon_unique, lat_unique, mean_grid, shading="auto", cmap="RdYlBu_r")
    fig.colorbar(pcm, ax=ax, label="Mean NO₂ (µg/m³)")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title(f"Gridded Mean Predictions (t={ts0:.2f})")
    fig.savefig(figures_dir / "predictions_surface.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"   ✓ Saved mean surface plot")

    # Uncertainty surface - save to figures directory
    fig, ax = plt.subplots(figsize=(8, 6))
    pcm = ax.pcolormesh(lon_unique, lat_unique, std_grid, shading="auto", cmap="YlOrRd")
    fig.colorbar(pcm, ax=ax, label="Std Dev (µg/m³)")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title(f"Gridded Uncertainty (t={ts0:.2f})")
    fig.savefig(figures_dir / "uncertainty_surface.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"   ✓ Saved uncertainty surface plot")

    # -------------------------------------------------------------------------
    # Save Model and Additional Outputs
    # -------------------------------------------------------------------------
    print("\n   Saving model and metrics tables...")

    # Save trained model to models directory
    model_path = models_dir / "fusiongp_model.pth"
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_config': MODEL_CONFIG,
        'training_config': TRAINING_CONFIG,
        'learned_hyperparameters': learned_params,
        'timestamp': timestamp,
    }, model_path)
    print(f"   ✓ Saved model to: {model_path}")

    # Save metrics as CSV table
    metrics_dict = metrics.to_dict()

    # Per-source metrics table
    per_source_data = []
    for source in ['epa', 'low_cost', 'satellite']:
        if source in metrics_dict:
            source_metrics = metrics_dict[source]
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

    # EPA-only metrics table
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

    # Save hyperparameters as CSV and LaTeX
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
    main_pbar.update(1)

    # Close the progress bar
    main_pbar.close()

    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------
    # Calculate elapsed time
    end_time = time.time()
    elapsed_time = end_time - start_time
    hours, remainder = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(remainder, 60)

    print("\n" + "="*70)
    print("Paper Reproduction Experiment Complete!")
    print("="*70)
    print(f"Results directory: {experiment_dir}")
    print(f"\nOrganized outputs:")
    print(f"  Figures: {figures_dir.relative_to(experiment_dir.parent.parent)}")
    print(f"  Models:  {models_dir.relative_to(experiment_dir.parent.parent)}")
    print(f"   Tables:  {tables_dir.relative_to(experiment_dir.parent.parent)}")
    print("\nKey Performance Metrics:")
    print(f"  • RMSE: {epa_metrics.to_dict()['rmse']:.4f} µg/m³")
    print(f"  • MAE:  {epa_metrics.to_dict()['mae']:.4f} µg/m³")
    print(f"  • R²:   {epa_metrics.to_dict()['r2']:.4f}")
    print(f"  • Bias: {epa_metrics.to_dict()['bias']:.4f} µg/m³")
    print("\nTotal Runtime:")
    if hours > 0:
        print(f"  • {int(hours)}h {int(minutes)}m {seconds:.2f}s ({elapsed_time:.2f} seconds)")
    elif minutes > 0:
        print(f"  • {int(minutes)}m {seconds:.2f}s ({elapsed_time:.2f} seconds)")
    else:
        print(f"  • {seconds:.2f} seconds")
    print("="*70)


if __name__ == "__main__":
    main()
