"""
Quick test to verify the performance fixes work.

This runs a lightweight version of the experiment with the fixes applied.
"""

import sys
import numpy as np
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data import DataLoader, DataPreprocessor
from src.models import FusionSVGP
from src.training import Trainer
from src.inference import Predictor
from src.evaluation import rmse, mae, r_squared, bias

print("="*70)
print("Testing FusionGP Performance Fixes")
print("="*70)

# Load data
DATA_PATH = Path(__file__).parent.parent / "data/test_data.csv"

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

print(f"\nData loaded: {len(train_data.coords)} train, {len(test_data.coords)} test")

# Create model WITH FIXES
print("\nCreating model with fixes...")
model = FusionSVGP(
    n_inducing=200,  # Smaller for quick test
    kernel_type='matern32',
    learn_inducing_locations=True,
    learn_kernel_hyperparams=True,  # ✓ FIX 1: Enable learning
    learn_noise=True,                # ✓ FIX 2: Enable learning
    learn_calibration=True,          # ✓ FIX 3: Enable learning
    initial_lengthscales={
        "spatial_x": 0.05,  # Smaller for less over-smoothing
        "spatial_y": 0.05,
        "temporal": 0.05,
    },
    initial_noise={
        "epa": 0.5,        # Start below old bound
        "low_cost": 0.8,   # Expect higher noise than EPA
        "satellite": 0.6,  # Medium noise
    },
)

print("✓ Model created with hyperparameter learning ENABLED")

# Train
print("\nTraining model...")
trainer = Trainer(
    model,
    learning_rate=0.01,  # ✓ FIX 4: Higher learning rate
    n_epochs=100,  # Fewer epochs for quick test
    batch_size=1024,
)

history = trainer.fit(train_data, val_data=val_data, verbose=True)

print(f"\nTraining complete:")
print(f"  Best val loss: {history.best_val_loss:.4f} at epoch {history.best_epoch + 1}")

# Predict WITHOUT observation noise (FIX 5)
print("\nMaking predictions...")
predictor = Predictor(model, scalers, include_observation_noise=False)  # ✓ FIX 5
predictions = predictor.predict(test_data)

# Evaluate
epa_mask = test_data.source_masks["epa"]
if scalers.normalize_targets:
    y_true = test_data.observations['epa'][epa_mask] * scalers.target_std['epa'] + scalers.target_mean['epa']
else:
    y_true = test_data.observations['epa'][epa_mask]

y_pred = predictions.mean[epa_mask]

# Compute metrics
test_rmse = rmse(y_true, y_pred)
test_mae = mae(y_true, y_pred)
test_r2 = r_squared(y_true, y_pred)
test_bias = bias(y_true, y_pred)

# Baseline
baseline_rmse = rmse(y_true, np.full_like(y_true, y_true.mean()))

print("\n" + "="*70)
print("RESULTS")
print("="*70)

print(f"\nBaseline (constant mean):")
print(f"  RMSE: {baseline_rmse:.4f} µg/m³")
print(f"  R²:   0.0000 (by definition)")

print(f"\nModel (with fixes):")
print(f"  RMSE: {test_rmse:.4f} µg/m³")
print(f"  MAE:  {test_mae:.4f} µg/m³")
print(f"  R²:   {test_r2:.4f}")
print(f"  Bias: {test_bias:.4f} µg/m³")

# Check success criteria
print("\n" + "="*70)
print("VERIFICATION")
print("="*70)

success = True

if test_r2 > 0.0:
    print("✓ R² is POSITIVE (model beats baseline)")
else:
    print("✗ R² is NEGATIVE (model worse than baseline)")
    success = False

if test_rmse < baseline_rmse:
    print("✓ RMSE is better than baseline")
else:
    print("✗ RMSE is worse than baseline")
    success = False

# Check learned hyperparameters
learned = model.get_hyperparameters()
print(f"\nLearned hyperparameters:")
print(f"  Spatial lengthscale: {learned['spatial_lengthscale']}")
print(f"  Temporal lengthscale: {learned['temporal_lengthscale']}")
print(f"  EPA noise: {learned['noise_std_epa']:.4f}")
print(f"  Low-cost noise: {learned['noise_std_low_cost']:.4f}")
print(f"  Satellite noise: {learned['noise_std_satellite']:.4f}")
print(f"  Calibration slope: {learned['lc_slope']:.4f}")
print(f"  Calibration intercept: {learned['lc_intercept']:.4f}")

# Check if calibration was learned
if abs(learned['lc_slope'] - 1.0) > 0.01 or abs(learned['lc_intercept']) > 0.01:
    print("✓ Calibration parameters LEARNED (different from initial)")
else:
    print("✗ Calibration parameters NOT learned (still at initial values)")
    success = False

print("\n" + "="*70)
if success:
    print("SUCCESS: Fixes are working!")
else:
    print("ISSUES DETECTED: Further investigation needed")
print("="*70)
