"""
Deep diagnostic to investigate why model parameters hit bounds.

This script examines:
1. Data characteristics and distribution
2. Initial vs learned parameters
3. Gradient flow and optimization behavior
4. ELBO component analysis
5. Prediction quality vs parameter values
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

print("="*80)
print("DEEP DIAGNOSTIC: Understanding Parameter Convergence Issues")
print("="*80)

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

print("\n" + "="*80)
print("1. DATA CHARACTERISTICS")
print("="*80)

# Analyze each source
for source in ['epa', 'low_cost', 'satellite']:
    mask_train = train_data.source_masks[source]
    mask_test = test_data.source_masks[source]

    if mask_train.any():
        obs_train = train_data.observations[source][mask_train]
        print(f"\n{source.upper()} - Training (normalized):")
        print(f"  Count: {mask_train.sum()}")
        print(f"  Mean: {obs_train.mean():.4f}, Std: {obs_train.std():.4f}")
        print(f"  Range: [{obs_train.min():.4f}, {obs_train.max():.4f}]")
        print(f"  Percentiles [10, 50, 90]: {np.percentile(obs_train, [10, 50, 90])}")

    if mask_test.any():
        obs_test = test_data.observations[source][mask_test]
        print(f"{source.upper()} - Test (normalized):")
        print(f"  Count: {mask_test.sum()}")
        print(f"  Mean: {obs_test.mean():.4f}, Std: {obs_test.std():.4f}")

# Check if low-cost is biased relative to EPA
if train_data.source_masks['epa'].any() and train_data.source_masks['low_cost'].any():
    epa_vals = train_data.observations['epa'][train_data.source_masks['epa']]
    lc_vals = train_data.observations['low_cost'][train_data.source_masks['low_cost']]

    print(f"\nEPA vs Low-Cost Comparison:")
    print(f"  EPA mean: {epa_vals.mean():.4f}")
    print(f"  Low-cost mean: {lc_vals.mean():.4f}")
    print(f"  Difference: {lc_vals.mean() - epa_vals.mean():.4f}")
    print(f"  → Suggests needed calibration: slope≈{epa_vals.std()/lc_vals.std():.3f}, intercept≈{epa_vals.mean() - lc_vals.mean():.3f}")

print(f"\nCoordinate Statistics:")
coords = train_data.coords
print(f"  Shape: {coords.shape}")
print(f"  Lat range: [{coords[:, 0].min():.4f}, {coords[:, 0].max():.4f}]")
print(f"  Lon range: [{coords[:, 1].min():.4f}, {coords[:, 1].max():.4f}]")
if coords.shape[1] > 2:
    print(f"  Time range: [{coords[:, 2].min():.4f}, {coords[:, 2].max():.4f}]")
print(f"  Spatial extent: {coords[:, 0].max() - coords[:, 0].min():.4f} × {coords[:, 1].max() - coords[:, 1].min():.4f}")

print("\n" + "="*80)
print("2. NOISE LEVEL ESTIMATION FROM DATA")
print("="*80)

# Estimate noise from nearby observations
def estimate_noise_from_neighbors(coords, obs, mask, max_dist=0.05):
    """Estimate noise by looking at variance in nearby observations."""
    valid_coords = coords[mask]
    valid_obs = obs[mask]

    if len(valid_obs) < 10:
        return None

    # Find pairs of nearby points
    diffs = []
    for i in range(min(100, len(valid_obs))):
        # Compute distances to this point
        dist = np.sqrt(((valid_coords - valid_coords[i])**2).sum(axis=1))
        nearby = (dist < max_dist) & (dist > 0)

        if nearby.sum() > 0:
            # Variance among nearby points
            nearby_obs = valid_obs[nearby]
            diffs.extend(nearby_obs - valid_obs[i])

    if len(diffs) > 5:
        # Assuming differences are mostly noise: Var[y_i - y_j] ≈ 2σ²
        return np.std(diffs) / np.sqrt(2)
    return None

for source in ['epa', 'low_cost', 'satellite']:
    mask = train_data.source_masks[source]
    if mask.any():
        noise_est = estimate_noise_from_neighbors(
            train_data.coords,
            train_data.observations[source],
            mask
        )
        if noise_est:
            print(f"{source.upper()}: Estimated noise std ≈ {noise_est:.4f} (in normalized space)")

print("\n" + "="*80)
print("3. PARAMETER EVOLUTION DURING TRAINING")
print("="*80)

# Create model with logging
model = FusionSVGP(
    n_inducing=100,
    kernel_type='matern32',
    learn_inducing_locations=True,
    learn_kernel_hyperparams=True,
    learn_noise=True,
    learn_calibration=True,
    initial_lengthscales={
        "spatial_x": 0.1,
        "spatial_y": 0.1,
        "temporal": 0.1,
    },
    initial_noise={
        "epa": 0.8,
        "low_cost": 1.5,
        "satellite": 1.2,
    },
)

print("\nInitial hyperparameters:")
init_params = model.get_hyperparameters()
for key, val in init_params.items():
    print(f"  {key}: {val}")

# Custom training loop with parameter tracking
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
n_epochs = 50
batch_size = 512

# Create dataset
from src.data.loader import FusionDataset
from torch.utils.data import DataLoader as TorchDataLoader

train_dataset = FusionDataset(train_data)
train_loader = TorchDataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
)

# Track parameters
param_history = {
    'epoch': [],
    'train_loss': [],
    'spatial_ls': [],
    'temporal_ls': [],
    'noise_epa': [],
    'noise_lc': [],
    'noise_sat': [],
    'lc_slope': [],
    'lc_intercept': [],
}

print(f"\nTraining for {n_epochs} epochs...")
print("Epoch | Loss    | Spatial LS | Temporal LS | EPA Noise | LC Noise | LC Slope")
print("-" * 80)

for epoch in range(n_epochs):
    model.train()
    epoch_loss = 0.0
    n_batches = 0

    for batch in train_loader:
        optimizer.zero_grad()

        # Unpack batch
        coords, timestamps, observations, masks, indices = batch

        # Combine coords and timestamps
        x = torch.cat([coords, timestamps.unsqueeze(-1)], dim=-1)

        # Compute ELBO
        loss = -model.elbo(x, observations, masks)

        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        n_batches += 1

    avg_loss = epoch_loss / n_batches

    # Get current parameters
    params = model.get_hyperparameters()

    # Store history
    param_history['epoch'].append(epoch)
    param_history['train_loss'].append(avg_loss)
    param_history['spatial_ls'].append(params['spatial_lengthscale'][0, 0])
    param_history['temporal_ls'].append(params['temporal_lengthscale'][0, 0])
    param_history['noise_epa'].append(params['noise_std_epa'])
    param_history['noise_lc'].append(params['noise_std_low_cost'])
    param_history['noise_sat'].append(params['noise_std_satellite'])
    param_history['lc_slope'].append(params['lc_slope'])
    param_history['lc_intercept'].append(params['lc_intercept'])

    if epoch % 10 == 0 or epoch == n_epochs - 1:
        print(f"{epoch:5d} | {avg_loss:7.2f} | {params['spatial_lengthscale'][0,0]:10.4f} | "
              f"{params['temporal_lengthscale'][0,0]:11.4f} | {params['noise_std_epa']:9.4f} | "
              f"{params['noise_std_low_cost']:8.4f} | {params['lc_slope']:8.4f}")

print("\n" + "="*80)
print("4. PARAMETER CONVERGENCE ANALYSIS")
print("="*80)

print("\nParameter changes:")
for key in ['spatial_ls', 'temporal_ls', 'noise_epa', 'noise_lc', 'noise_sat', 'lc_slope', 'lc_intercept']:
    initial = param_history[key][0]
    final = param_history[key][-1]
    change = final - initial
    pct_change = 100 * change / initial if abs(initial) > 1e-6 else 0
    print(f"  {key:15s}: {initial:8.4f} → {final:8.4f} (Δ={change:+7.4f}, {pct_change:+6.1f}%)")

# Check if hitting bounds
print("\nBound analysis:")
if param_history['noise_epa'][-1] <= 1.01:
    print("  ⚠ EPA noise at/near lower bound (1.0)")
if param_history['noise_lc'][-1] <= 1.01:
    print("  ⚠ Low-cost noise at/near lower bound (1.0)")
if param_history['noise_sat'][-1] <= 1.01:
    print("  ⚠ Satellite noise at/near lower bound (1.0)")
if param_history['lc_slope'][-1] <= 0.11:
    print("  ⚠ LC slope at/near lower bound (0.1)")
if param_history['lc_slope'][-1] >= 9.9:
    print("  ⚠ LC slope at/near upper bound (10.0)")

print("\n" + "="*80)
print("5. GRADIENT ANALYSIS")
print("="*80)

# Check gradients at current parameters
model.eval()
optimizer.zero_grad()

# Use a batch
batch = next(iter(train_loader))
coords, timestamps, observations, masks, indices = batch
x = torch.cat([coords, timestamps.unsqueeze(-1)], dim=-1)
loss = -model.elbo(x, observations, masks)
loss.backward()

print("\nGradients (indicates which way parameters want to move):")
print(f"  Raw noise EPA grad: {model.likelihood.raw_noise['epa'].grad}")
print(f"  Raw noise LC grad: {model.likelihood.raw_noise['low_cost'].grad}")
print(f"  Raw noise SAT grad: {model.likelihood.raw_noise['satellite'].grad}")
print(f"  Raw LC slope grad: {model.likelihood.raw_lc_slope.grad}")
print(f"  Raw LC intercept grad: {model.likelihood.raw_lc_intercept.grad}")

# Note: negative gradient means parameter wants to decrease
print("\nInterpretation:")
if model.likelihood.raw_noise['epa'].grad and model.likelihood.raw_noise['epa'].grad < 0:
    print("  → EPA noise wants to DECREASE (but may be at bound)")
if model.likelihood.raw_lc_slope.grad and model.likelihood.raw_lc_slope.grad < 0:
    print("  → LC slope wants to DECREASE (but may be at bound)")

print("\n" + "="*80)
print("6. PREDICTION QUALITY")
print("="*80)

model.eval()
predictor = Predictor(model, scalers, include_observation_noise=False)
predictions = predictor.predict(test_data)

epa_mask = test_data.source_masks["epa"]
if scalers.normalize_targets:
    y_true = test_data.observations['epa'][epa_mask] * scalers.target_std['epa'] + scalers.target_mean['epa']
else:
    y_true = test_data.observations['epa'][epa_mask]

y_pred = predictions.mean[epa_mask]

print(f"\nTest set performance:")
print(f"  RMSE: {rmse(y_true, y_pred):.4f} µg/m³")
print(f"  MAE:  {mae(y_true, y_pred):.4f} µg/m³")
print(f"  R²:   {r_squared(y_true, y_pred):.4f}")
print(f"  Bias: {bias(y_true, y_pred):.4f} µg/m³")

baseline_rmse = rmse(y_true, np.full_like(y_true, y_true.mean()))
print(f"\nBaseline RMSE: {baseline_rmse:.4f} µg/m³")
print(f"Improvement over baseline: {(baseline_rmse - rmse(y_true, y_pred)) / baseline_rmse * 100:.1f}%")

print("\n" + "="*80)
print("7. RECOMMENDATIONS")
print("="*80)

recommendations = []

# Check noise bounds
if any(param_history['noise_epa'][-1] <= 1.01 or
       param_history['noise_lc'][-1] <= 1.01 or
       param_history['noise_sat'][-1] <= 1.01 for _ in [0]):
    recommendations.append(
        "CRITICAL: Reduce noise lower bound from 1.0 to 0.01\n"
        "  → Current bound is too restrictive for normalized data\n"
        "  → Change likelihoods.py:115 to noise_bounds=(0.01, 100.0)"
    )

# Check calibration
if param_history['lc_slope'][-1] <= 0.11:
    recommendations.append(
        "WARNING: LC slope hitting lower bound (0.1)\n"
        "  → Model wants slope < 0.1, consider lowering bound to 0.01\n"
        "  → OR investigate if low-cost data is fundamentally misaligned"
    )

# Check lengthscales
if param_history['spatial_ls'][-1] > 0.5:
    recommendations.append(
        "INFO: High spatial lengthscale causing over-smoothing\n"
        "  → Try smaller initial lengthscales (0.01-0.05)\n"
        "  → Or ensure spatial coordinates are properly normalized"
    )

# Check performance
if r_squared(y_true, y_pred) < 0:
    recommendations.append(
        "CRITICAL: Negative R² indicates severe underfitting\n"
        "  → Primary cause likely the restrictive noise bounds\n"
        "  → Fix noise bounds first, then reassess"
    )

if recommendations:
    for i, rec in enumerate(recommendations, 1):
        print(f"\n{i}. {rec}")
else:
    print("\n✓ No major issues detected!")

print("\n" + "="*80)
print("DIAGNOSTIC COMPLETE")
print("="*80)
