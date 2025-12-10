"""
Basic Usage Example for FusionGP

This script demonstrates the basic workflow for training and evaluating
the FusionGP model on synthetic NO₂ data from multiple sources.

Workflow:
1. Load synthetic multi-source NO₂ data
2. Preprocess and split data (train/val/test)
3. Initialize FusionSVGP model
4. Train the model
5. Make predictions
6. Evaluate performance
7. Visualize results
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Add parent directory to path to import src
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data import DataLoader, DataPreprocessor
from src.models import FusionSVGP
from src.training import Trainer, EarlyStopping, ModelCheckpoint
from src.inference import Predictor
from src.evaluation import Evaluator


def main():
    print("="*70)
    print("FusionGP Basic Usage Example")
    print("="*70)

    # -------------------------------------------------------------------------
    # 1. Load Data
    # -------------------------------------------------------------------------
    print("\n[1/7] Loading synthetic NO₂ data...")

    data_path = Path(__file__).parent.parent / "notebooks/synthetic_no2_data.csv"

    if not data_path.exists():
        raise FileNotFoundError(
            f"Synthetic data not found at {data_path}. "
            "Please ensure the data file exists."
        )

    loader = DataLoader(str(data_path))
    data = loader.load()

    print(f"   ✓ Loaded {len(data.coords)} observations")
    print(f"   ✓ Sources: {list(data.source_masks.keys())}")
    print(f"   ✓ Spatial extent: lat=[{data.coords[:,0].min():.2f}, {data.coords[:,0].max():.2f}], "
          f"lon=[{data.coords[:,1].min():.2f}, {data.coords[:,1].max():.2f}]")
    print(f"   ✓ Temporal extent: {len(np.unique(data.timestamps))} unique timestamps")

    # -------------------------------------------------------------------------
    # 2. Preprocess Data
    # -------------------------------------------------------------------------
    print("\n[2/7] Preprocessing data (normalization + train/val/test split)...")

    preprocessor = DataPreprocessor(
        normalize_targets=True,
        normalize_coords=True,
        normalize_time=True
    )

    train_data, val_data, test_data = preprocessor.fit_transform(
        data,
        train_ratio=0.6,
        val_ratio=0.2,
        test_ratio=0.2
    )
    scalers = preprocessor.get_scalers()

    print(f"   ✓ Train: {len(train_data.coords)} samples")
    print(f"   ✓ Val:   {len(val_data.coords)} samples")
    print(f"   ✓ Test:  {len(test_data.coords)} samples")

    # -------------------------------------------------------------------------
    # 3. Initialize Model
    # -------------------------------------------------------------------------
    print("\n[3/7] Initializing FusionSVGP model...")

    model = FusionSVGP(
        n_inducing=300,  # Smaller for faster training in demo
        kernel_type="matern32",
        learn_inducing_locations=True,
        learn_kernel_hyperparams=True,
        learn_noise=True,
        learn_calibration=True,
        initial_lengthscales={
            "spatial_x": 0.15,
            "spatial_y": 0.15,
            "temporal": 0.2,
        },
        initial_noise={
            "epa": 0.1,
            "low_cost": 0.3,
            "satellite": 0.2,
        },
    )

    print(f"   ✓ Model initialized with {model.n_inducing} inducing points")
    print(f"   ✓ Kernel: {model.kernel_type}")
    print(f"   ✓ Learnable parameters: inducing locations, lengthscales, noise, calibration")

    # -------------------------------------------------------------------------
    # 4. Train Model
    # -------------------------------------------------------------------------
    print("\n[4/7] Training model...")

    # Setup callbacks
    callbacks = [
        EarlyStopping(patience=20, monitor='val_loss', mode='min'),
        ModelCheckpoint(save_dir="experiments/checkpoints", monitor='val_loss', mode='min')
    ]

    trainer = Trainer(
        model,
        learning_rate=0.01,
        n_epochs=100,  # Reduced for faster demo
        batch_size=512,
        callbacks=callbacks
    )

    print(f"   ✓ Training config: lr={trainer.learning_rate}, "
          f"epochs={trainer.n_epochs}, batch_size={trainer.batch_size}")

    history = trainer.fit(train_data, val_data=val_data, verbose=True)

    print(f"   ✓ Training completed in {len(history.train_loss)} epochs")
    print(f"   ✓ Best validation loss: {history.best_val_loss:.4f}")

    # -------------------------------------------------------------------------
    # 5. Make Predictions
    # -------------------------------------------------------------------------
    print("\n[5/7] Making predictions on test set...")

    predictor = Predictor(
        model,
        scalers,
        include_observation_noise=True,
        noise_source='epa'
    )

    predictions = predictor.predict(test_data)

    print(f"   ✓ Predicted {len(predictions.mean)} test points")
    print(f"   ✓ Mean prediction: {predictions.mean.mean():.2f} ± {predictions.std.mean():.2f}")

    # -------------------------------------------------------------------------
    # 6. Evaluate Performance
    # -------------------------------------------------------------------------
    print("\n[6/7] Evaluating model performance...")

    evaluator = Evaluator()

    # Get EPA observations for evaluation (ground truth)
    epa_mask = test_data.source_masks['epa']
    y_true = test_data.observations['epa'][epa_mask]
    y_pred = predictions.mean[epa_mask]
    y_std = predictions.std[epa_mask]

    metrics = evaluator.evaluate(
        y_true=y_true,
        y_pred_mean=y_pred,
        y_pred_std=y_std
    )

    print("\n   Metrics (on EPA test observations):")
    print(f"   {'Metric':<20} {'Value':>10}")
    print("   " + "-"*32)
    for metric_name, metric_value in metrics.to_dict().items():
        if isinstance(metric_value, (int, float, np.number)):
            print(f"   {metric_name:<20} {metric_value:>10.4f}")

    # -------------------------------------------------------------------------
    # 7. Visualize Results
    # -------------------------------------------------------------------------
    print("\n[7/7] Creating visualizations...")

    # Create output directory
    output_dir = Path("experiments/outputs")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Plot 1: Training history
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(history.train_loss, label='Train Loss', alpha=0.7)
    axes[0].plot(history.val_loss, label='Val Loss', alpha=0.7)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Negative ELBO')
    axes[0].set_title('Training History')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Plot 2: Predictions vs True
    axes[1].scatter(y_true, y_pred, alpha=0.5, s=20)
    axes[1].plot([y_true.min(), y_true.max()],
                 [y_true.min(), y_true.max()],
                 'r--', label='Perfect prediction')
    axes[1].set_xlabel('True NO₂ (ppb)')
    axes[1].set_ylabel('Predicted NO₂ (ppb)')
    axes[1].set_title(f'Predictions vs Truth (R²={metrics.to_dict()["r2"]:.3f})')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'basic_usage_results.png', dpi=150, bbox_inches='tight')
    print(f"   ✓ Saved plot to {output_dir / 'basic_usage_results.png'}")

    # Plot 3: Uncertainty calibration
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    errors = np.abs(y_pred - y_true)
    ax.scatter(y_std, errors, alpha=0.5, s=20)
    ax.plot([0, y_std.max()], [0, y_std.max()], 'r--',
            label='Perfect calibration')
    ax.set_xlabel('Predicted Std (ppb)')
    ax.set_ylabel('Absolute Error (ppb)')
    ax.set_title('Uncertainty Calibration')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'uncertainty_calibration.png', dpi=150, bbox_inches='tight')
    print(f"   ✓ Saved plot to {output_dir / 'uncertainty_calibration.png'}")

    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------
    print("\n" + "="*70)
    print("Experiment Complete!")
    print("="*70)
    print(f"Model checkpoint: experiments/checkpoints/best_model.pt")
    print(f"Outputs: {output_dir}")
    print("\nKey Results:")
    metrics_dict = metrics.to_dict()
    print(f"  • RMSE: {metrics_dict['rmse']:.4f} ppb")
    print(f"  • MAE:  {metrics_dict['mae']:.4f} ppb")
    print(f"  • R²:   {metrics_dict['r2']:.4f}")
    print(f"  • CRPS: {metrics_dict.get('crps', 'N/A')}")
    print("="*70)


if __name__ == "__main__":
    main()
