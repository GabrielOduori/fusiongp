"""
Diagnostic script to investigate model performance issues.

This script examines:
1. Prediction distributions
2. Training data statistics
3. Model hyperparameters
4. ELBO components
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

# Load data
DATA_PATH = Path(__file__).parent.parent / "data/test_data.csv"

print("="*70)
print("FusionGP Diagnostic Analysis")
print("="*70)

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

print(f"\n1. DATA STATISTICS")
print("-" * 70)
print(f"Total observations: {len(data.coords)}")
print(f"Train: {len(train_data.coords)}, Val: {len(val_data.coords)}, Test: {len(test_data.coords)}")

# EPA statistics
epa_train = train_data.observations['epa'][train_data.source_masks['epa']]
epa_test = test_data.observations['epa'][test_data.source_masks['epa']]

print(f"\nEPA Train (normalized):")
print(f"  Mean: {epa_train.mean():.4f}, Std: {epa_train.std():.4f}")
print(f"  Range: [{epa_train.min():.4f}, {epa_train.max():.4f}]")

print(f"\nEPA Test (normalized):")
print(f"  Mean: {epa_test.mean():.4f}, Std: {epa_test.std():.4f}")
print(f"  Range: [{epa_test.min():.4f}, {epa_test.max():.4f}]")

print(f"\nScaler parameters:")
print(f"  EPA mean: {scalers.target_mean['epa']:.4f}")
print(f"  EPA std: {scalers.target_std['epa']:.4f}")

# Create simple model
print(f"\n2. SIMPLE MODEL TEST")
print("-" * 70)

model = FusionSVGP(
    n_inducing=100,  # Small for quick test
    kernel_type='matern32',
    learn_inducing_locations=True,
    learn_kernel_hyperparams=True,  # Enable learning
    learn_noise=True,
    learn_calibration=False,
    initial_lengthscales={
        "spatial_x": 0.1,
        "spatial_y": 0.1,
        "temporal": 0.1,
    },
    initial_noise={
        "epa": 0.5,
        "low_cost": 1.0,
        "satellite": 0.8,
    },
)

print(f"Model created: {model.n_inducing} inducing points")

# Train for a few epochs
trainer = Trainer(
    model,
    learning_rate=0.01,  # Higher learning rate
    n_epochs=50,
    batch_size=512,
)

print(f"\nTraining for {trainer.n_epochs} epochs...")
history = trainer.fit(train_data, val_data=val_data, verbose=True)

print(f"\nTraining complete:")
print(f"  Final train loss: {history.train_loss[-1]:.4f}")
print(f"  Final val loss: {history.val_loss[-1]:.4f}")
print(f"  Best val loss: {history.best_val_loss:.4f} at epoch {history.best_epoch + 1}")

# Make predictions
print(f"\n3. PREDICTION ANALYSIS")
print("-" * 70)

# Test with noise
predictor_with_noise = Predictor(model, scalers, include_observation_noise=True, noise_source='epa')
pred_with_noise = predictor_with_noise.predict(test_data)

# Test without noise
predictor_no_noise = Predictor(model, scalers, include_observation_noise=False)
pred_no_noise = predictor_no_noise.predict(test_data)

print(f"\nPredictions WITH observation noise:")
print(f"  Mean range: [{pred_with_noise.mean.min():.4f}, {pred_with_noise.mean.max():.4f}]")
print(f"  Mean avg: {pred_with_noise.mean.mean():.4f}")
print(f"  Std range: [{pred_with_noise.std.min():.4f}, {pred_with_noise.std.max():.4f}]")
print(f"  Std avg: {pred_with_noise.std.mean():.4f}")

print(f"\nPredictions WITHOUT observation noise:")
print(f"  Mean range: [{pred_no_noise.mean.min():.4f}, {pred_no_noise.mean.max():.4f}]")
print(f"  Mean avg: {pred_no_noise.mean.mean():.4f}")
print(f"  Std range: [{pred_no_noise.std.min():.4f}, {pred_no_noise.std.max():.4f}]")
print(f"  Std avg: {pred_no_noise.std.mean():.4f}")

# True values
epa_mask = test_data.source_masks["epa"]
if scalers.normalize_targets:
    y_true = test_data.observations['epa'][epa_mask] * scalers.target_std['epa'] + scalers.target_mean['epa']
else:
    y_true = test_data.observations['epa'][epa_mask]

print(f"\nTrue EPA values:")
print(f"  Range: [{y_true.min():.4f}, {y_true.max():.4f}]")
print(f"  Mean: {y_true.mean():.4f}, Std: {y_true.std():.4f}")

# Compute metrics manually
from src.evaluation import rmse, mae, r_squared, bias

print(f"\n4. METRICS COMPARISON")
print("-" * 70)

y_pred_with = pred_with_noise.mean[epa_mask]
y_pred_no = pred_no_noise.mean[epa_mask]

print(f"\nMETRICS WITH observation noise:")
print(f"  RMSE: {rmse(y_true, y_pred_with):.4f}")
print(f"  MAE:  {mae(y_true, y_pred_with):.4f}")
print(f"  R²:   {r_squared(y_true, y_pred_with):.4f}")
print(f"  Bias: {bias(y_true, y_pred_with):.4f}")

print(f"\nMETRICS WITHOUT observation noise:")
print(f"  RMSE: {rmse(y_true, y_pred_no):.4f}")
print(f"  MAE:  {mae(y_true, y_pred_no):.4f}")
print(f"  R²:   {r_squared(y_true, y_pred_no):.4f}")
print(f"  Bias: {bias(y_true, y_pred_no):.4f}")

# Baseline: predict mean
y_mean = y_true.mean()
print(f"\nBASELINE (constant mean = {y_mean:.4f}):")
print(f"  RMSE: {rmse(y_true, np.full_like(y_true, y_mean)):.4f}")
print(f"  MAE:  {mae(y_true, np.full_like(y_true, y_mean)):.4f}")
print(f"  R²:   0.0000 (by definition)")
print(f"  Bias: 0.0000 (by definition)")

print(f"\n5. LEARNED HYPERPARAMETERS")
print("-" * 70)
learned_params = model.get_hyperparameters()
for key, val in learned_params.items():
    print(f"  {key}: {val}")

print("\n" + "="*70)
print("Diagnostic Complete")
print("="*70)
