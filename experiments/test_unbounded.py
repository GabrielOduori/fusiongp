"""
Test with UNBOUNDED parameters - let the model learn freely.

This will reveal what the model actually wants to do with the data sources.
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
print("Testing FusionGP with UNBOUNDED Parameters")
print("="*70)

# Load data
DATA_PATH = Path(__file__).parent.parent / "data/dublin_realistic_no2.csv"

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

# Create model with ALL BOUNDS REMOVED
print("\nCreating model with UNBOUNDED parameters...")
model = FusionSVGP(
    n_inducing=200,
    kernel_type='matern32',
    learn_inducing_locations=True,
    learn_kernel_hyperparams=True,
    learn_noise=True,
    learn_calibration=True,
    initial_lengthscales={
        "spatial_x": 0.05,
        "spatial_y": 0.05,
        "temporal": 0.05,
    },
    initial_noise={
        "epa": 0.5,
        "low_cost": 0.8,
        "satellite": 0.6,
    },
)

print("✓ Model created with NO BOUNDS on noise or calibration")

# Train
print("\nTraining model...")
trainer = Trainer(
    model,
    learning_rate=0.01,
    n_epochs=100,
    batch_size=1024,
)

history = trainer.fit(train_data, val_data=val_data, verbose=True)

print(f"\nTraining complete:")
print(f"  Best val loss: {history.best_val_loss:.4f} at epoch {history.best_epoch + 1}")

# Predict
print("\nMaking predictions...")
predictor = Predictor(model, scalers, include_observation_noise=False)
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

print(f"\nModel (UNBOUNDED):")
print(f"  RMSE: {test_rmse:.4f} µg/m³")
print(f"  MAE:  {test_mae:.4f} µg/m³")
print(f"  R²:   {test_r2:.4f}")
print(f"  Bias: {test_bias:.4f} µg/m³")

# Improvement
improvement = (baseline_rmse - test_rmse) / baseline_rmse * 100
print(f"  Improvement over baseline: {improvement:.1f}%")

print("\n" + "="*70)
print("LEARNED HYPERPARAMETERS (UNBOUNDED)")
print("="*70)

learned = model.get_hyperparameters()
print(f"\nKernel parameters:")
print(f"  Spatial lengthscale: {learned['spatial_lengthscale']}")
print(f"  Temporal lengthscale: {learned['temporal_lengthscale']}")
print(f"  Output scale: {learned['outputscale']:.4f}")

print(f"\nNoise parameters (NO BOUNDS):")
print(f"  EPA noise:       {learned['noise_std_epa']:.6f}")
print(f"  Low-cost noise:  {learned['noise_std_low_cost']:.6f}")
print(f"  Satellite noise: {learned['noise_std_satellite']:.6f}")

print(f"\nCalibration parameters (NO BOUNDS):")
print(f"  LC slope:     {learned['lc_slope']:.6f}")
print(f"  LC intercept: {learned['lc_intercept']:.6f}")

print("\n" + "="*70)
print("INTERPRETATION")
print("="*70)

# Analyze what the model learned
if learned['noise_std_satellite'] < 0.001:
    print("⚠ Satellite noise → 0: Model wants to TRUST satellite data completely")
elif learned['noise_std_satellite'] > 10:
    print("⚠ Satellite noise very high: Model wants to IGNORE satellite data")
else:
    print(f"✓ Satellite noise reasonable: {learned['noise_std_satellite']:.3f}")

if learned['lc_slope'] < 0.001:
    print("⚠ LC slope → 0: Model wants to SUPPRESS low-cost data completely")
elif learned['lc_slope'] > 10:
    print(f"⚠ LC slope very high ({learned['lc_slope']:.2f}): Model amplifying low-cost data")
else:
    print(f"✓ LC slope reasonable: {learned['lc_slope']:.3f}")

if test_r2 > 0.0:
    print(f"\n✓ SUCCESS: R² = {test_r2:.4f} (POSITIVE!)")
else:
    print(f"\n✗ STILL NEGATIVE R²: {test_r2:.4f}")
    print("   → Model fundamentally unable to beat baseline")
    print("   → Suggests data fusion may not be helpful, or data quality issues")

print("\n" + "="*70)
