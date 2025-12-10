
import numpy as np

from fusiongp.data import DataLoader, DataPreprocessor
from fusiongp.models import FusionSVGP
from fusiongp.training import Trainer
from fusiongp.inference import Predictor
from fusiongp.evaluation import Evaluator, rmse, mae, r_squared, bias
from fusiongp.visualization import plot_predictions, plot_uncertainty, plot_confidence_intervals
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors

# 1. Load and preprocess data
loader = DataLoader("/media/gabriel-oduori/SERVER/dev_space/fusionGP/data/test_data.csv")
data = loader.load()

preprocessor = DataPreprocessor(normalize_targets=True)
train_data, val_data, test_data = preprocessor.fit_transform(data)
scalers = preprocessor.get_scalers()

# 2. Initialize model with proper noise settings
# In normalized space (target std=1.0), we need reasonable noise floors
# to prevent overconfidence and achieve good calibration
model = FusionSVGP(
    n_inducing=800,  # Increased from 500 for better local variation capture
    kernel_type="matern32",
    learn_inducing_locations=True,
    # Disable hyperparameter optimization to speed things up
    learn_kernel_hyperparams=False,
    learn_noise=False,
    learn_calibration=False,
    # Set smaller initial spatial lengthscales for normalized [0,1] coords
    # Aim for ~0.1-0.2 to capture local structure
    initial_lengthscales={
        "spatial_x": 0.1,
        "spatial_y": 0.1,
        "temporal": 0.1,
    },
    # Initial noise (in normalized space, std=1.0)
    # Higher values for better calibration with high capacity model
    initial_noise={
        "epa": 0.8,      # Slightly tighter to sharpen EPA mean
        "low_cost": 1.5,  # More noisy
        "satellite": 1.2,  # Moderate noise
    },
)

print(model)

# 3. Train
trainer = Trainer(model, 
                  learning_rate=0.0005,  
                  n_epochs=300)
print(f"Training config -> lr={trainer.learning_rate:.2e}, epochs={trainer.n_epochs}, batch_size={trainer.batch_size}")
trainer.fit(train_data, 
            val_data=val_data)

# 4. Predict
# Check learned noise values
learned_params = model.get_hyperparameters()
print("\n" + "="*50)
print("Learned Hyperparameters:")
print("="*50)
for key, val in learned_params.items():
    print(f"{key}: {val}")
print("="*50 + "\n")

# Check data statistics
print("="*50)
print("Data Statistics:")
print("="*50)
epa_train = train_data.observations['epa'][train_data.source_masks['epa']]
epa_test = test_data.observations['epa'][test_data.source_masks['epa']]
print(f"Train EPA (normalized): mean={epa_train.mean():.4f}, std={epa_train.std():.4f}, range=[{epa_train.min():.4f}, {epa_train.max():.4f}]")
print(f"Test EPA (normalized): mean={epa_test.mean():.4f}, std={epa_test.std():.4f}, range=[{epa_test.min():.4f}, {epa_test.max():.4f}]")
print(f"Target scaler: mean={scalers.target_mean['epa']:.4f}, std={scalers.target_std['epa']:.4f}")
print(f"Train coords range: lat=[{train_data.coords[:,0].min():.4f}, {train_data.coords[:,0].max():.4f}], lon=[{train_data.coords[:,1].min():.4f}, {train_data.coords[:,1].max():.4f}]")
print(f"Train time range: [{train_data.timestamps.min():.4f}, {train_data.timestamps.max():.4f}]")
print("="*50 + "\n")

# Create predictor WITH observation noise (default behavior)
predictor = Predictor(model, scalers, include_observation_noise=True, noise_source='epa')

# Point predictions on test set (for metrics)
predictions = predictor.predict(test_data)

# Quick diagnostics on scales (normalized targets)
y_norm = test_data.observations['epa']
print("y_norm range/mean:", np.nanmin(y_norm), np.nanmax(y_norm), np.nanmean(y_norm))
print("pred mean range/mean (orig scale):", predictions.mean.min(), predictions.mean.max(), predictions.mean.mean())
print("pred std range/mean (orig scale):", predictions.std.min(), predictions.std.max(), predictions.std.mean())
print("mask counts:", {k: v.sum() for k, v in test_data.source_masks.items()})

# Grid predictions for map outputs
lat_min, lat_max = data.coords[:, 0].min(), data.coords[:, 0].max()
lon_min, lon_max = data.coords[:, 1].min(), data.coords[:, 1].max()
timestamps = np.unique(data.timestamps)  # use all available times; subset if needed
grid_predictions = predictor.predict_grid(
    lat_range=(lat_min, lat_max),
    lon_range=(lon_min, lon_max),
    timestamps=timestamps,
    resolution=0.01,  # adjust for finer/coarser grids
)

# Simple EPA-only IDW baseline for comparison
def idw_predict(train_coords, train_vals, query_coords, k=8, power=2.0, eps=1e-12):
    if len(train_coords) == 0:
        raise ValueError("No training points for IDW")
    k = min(k, len(train_coords))
    nn = NearestNeighbors(n_neighbors=k)
    nn.fit(train_coords)
    dists, idxs = nn.kneighbors(query_coords)
    
    preds = np.empty(len(query_coords))
    for i in range(len(query_coords)):
        if (dists[i] < eps).any():
            # Exact match, take the first matching neighbor's value
            preds[i] = train_vals[idxs[i][dists[i] < eps][0]]
            continue
        weights = 1.0 / np.power(dists[i] + eps, power)
        w_sum = weights.sum()
        preds[i] = np.dot(weights, train_vals[idxs[i]]) / w_sum if w_sum > 0 else np.nan
    return preds

epa_train_mask = train_data.source_masks['epa']
epa_test_mask = test_data.source_masks['epa']
epa_train_coords = train_data.coords[epa_train_mask]
epa_train_vals = train_data.observations['epa'][epa_train_mask]
epa_test_coords = test_data.coords[epa_test_mask]
epa_test_vals = test_data.observations['epa'][epa_test_mask]

if len(epa_train_coords) and len(epa_test_coords):
    # IDW on normalized values
    idw_preds_norm = idw_predict(epa_train_coords, epa_train_vals, epa_test_coords, k=8, power=2.0)

    # Denormalize EPA targets/preds for baseline/eval
    if scalers.normalize_targets and 'epa' in scalers.target_std:
        scale = scalers.target_std['epa']
        offset = scalers.target_mean['epa']
        epa_train_vals_orig = epa_train_vals * scale + offset
        epa_test_vals_orig = epa_test_vals * scale + offset
        idw_preds = idw_preds_norm * scale + offset
    else:
        epa_train_vals_orig = epa_train_vals
        epa_test_vals_orig = epa_test_vals
        idw_preds = idw_preds_norm
    baseline_metrics = {
        'rmse': rmse(epa_test_vals_orig, idw_preds),
        'mae': mae(epa_test_vals_orig, idw_preds),
        'r2': r_squared(epa_test_vals_orig, idw_preds),
        'bias': bias(epa_test_vals_orig, idw_preds),
    }
    print("IDW baseline (EPA only):", baseline_metrics)
else:
    print("Skipping IDW baseline: no EPA points in train/test split.")


# 5. Evaluate (denormalize targets if needed)
if scalers.normalize_targets:
    y_true_orig = {
        src: obs * scalers.target_std[src] + scalers.target_mean[src]
        for src, obs in test_data.observations.items()
    }
else:
    y_true_orig = test_data.observations

evaluator = Evaluator()
metrics = evaluator.evaluate_per_source(
    y_true=y_true_orig,
    y_pred_mean=predictions.mean,
    y_pred_std=predictions.std,
    source_masks=test_data.source_masks,
)
print(metrics.summary())

# EPA-only metrics to see fusion impact
epa_mask = test_data.source_masks["epa"]
epa_metrics = evaluator.evaluate(
    y_true=y_true_orig["epa"][epa_mask],
    y_pred_mean=predictions.mean[epa_mask],
    y_pred_std=predictions.std[epa_mask],
)
print("EPA-only metrics:", epa_metrics.to_dict())


# 6. Visualize
from pathlib import Path
from datetime import datetime
from fusiongp.visualization.spatial_maps import create_spatial_maps

# Create timestamped experiment folder
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
base_results_dir = Path(__file__).resolve().parent / "results"
experiment_dir = base_results_dir / f"experiment_{timestamp}"
experiment_dir.mkdir(parents=True, exist_ok=True)

print(f"\n{'='*60}")
print(f"Saving results to: {experiment_dir}")
print(f"{'='*60}\n")

# Save experiment summary
summary_file = experiment_dir / "experiment_summary.txt"
with open(summary_file, 'w') as f:
    f.write(f"FusionGP Experiment Summary\n")
    f.write(f"{'='*60}\n")
    f.write(f"Timestamp: {timestamp}\n")
    f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

    f.write(f"Model Configuration:\n")
    f.write(f"{'-'*60}\n")
    f.write(f"n_inducing: {model.n_inducing}\n")
    f.write(f"kernel_type: {model.kernel_type}\n")
    f.write(f"learning_rate: {trainer.learning_rate}\n")
    f.write(f"n_epochs: {trainer.n_epochs}\n")
    f.write(f"batch_size: {trainer.batch_size}\n\n")

    f.write(f"Learned Hyperparameters:\n")
    f.write(f"{'-'*60}\n")
    for key, val in learned_params.items():
        f.write(f"{key}: {val}\n")
    f.write(f"\n")

    f.write(f"Evaluation Metrics (EPA only):\n")
    f.write(f"{'-'*60}\n")
    for key, val in epa_metrics.to_dict().items():
        if isinstance(val, (int, float, np.number)):
            f.write(f"{key}: {val:.4f}\n")
        else:
            f.write(f"{key}: {val}\n")
    f.write(f"\n")

    f.write(f"Data Statistics:\n")
    f.write(f"{'-'*60}\n")
    f.write(f"Train EPA (normalized): mean={epa_train.mean():.4f}, std={epa_train.std():.4f}\n")
    f.write(f"Test EPA (normalized): mean={epa_test.mean():.4f}, std={epa_test.std():.4f}\n")
    f.write(f"Target scaler: mean={scalers.target_mean['epa']:.4f}, std={scalers.target_std['epa']:.4f}\n")

print(f"Experiment summary saved to: {summary_file}")

# Original simple plots
plot_predictions(
    coords=grid_predictions.coords,
    values=grid_predictions.mean,
    save_path=str(experiment_dir / "predictions.png"),
)
plot_uncertainty(
    coords=grid_predictions.coords,
    std=grid_predictions.std,
    save_path=str(experiment_dir / "uncertainty.png"),
)

# EPA uncertainty band plot (mean with confidence intervals over test points)
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
        ylabel="NO₂ (original units)",
        title="EPA Predictions with Confidence Intervals",
        save_path=str(experiment_dir / "uncertainty_ci.png"),
    )
    plt.close(fig_ci)

# Enhanced spatial maps with contours
print("\nCreating enhanced spatial maps...")
create_spatial_maps(
    predictor=predictor,
    scalers=scalers,
    lat_range=(lat_min, lat_max),
    lon_range=(lon_min, lon_max),
    n_times=4,  # Create 4 maps at different times
    nx=100,     # Grid resolution
    ny=100,
    output_dir=str(experiment_dir / 'spatial_maps'),
    cmap_mean='RdYlBu_r',
    cmap_std='plasma',
)

# Smooth gridded maps (first timestamp)
ts0 = np.unique(grid_predictions.timestamps)[0]
mask = grid_predictions.timestamps == ts0
lat_unique = np.unique(grid_predictions.coords[mask][:, 0])
lon_unique = np.unique(grid_predictions.coords[mask][:, 1])

mean_grid = grid_predictions.mean[mask].reshape(len(lat_unique), len(lon_unique))
std_grid = grid_predictions.std[mask].reshape(len(lat_unique), len(lon_unique))

fig, ax = plt.subplots(figsize=(8, 6))
pcm = ax.pcolormesh(lon_unique, lat_unique, mean_grid, shading="auto", cmap="RdYlBu_r")
fig.colorbar(pcm, ax=ax, label="Mean")
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")
ax.set_title(f"Gridded Mean (timestamp={ts0})")
fig.savefig(experiment_dir / "predictions_surface.png", dpi=300, bbox_inches="tight")
plt.close(fig)

fig, ax = plt.subplots(figsize=(8, 6))
pcm = ax.pcolormesh(lon_unique, lat_unique, std_grid, shading="auto", cmap="YlOrRd")
fig.colorbar(pcm, ax=ax, label="Std Dev")
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")
ax.set_title(f"Gridded Uncertainty (timestamp={ts0})")
fig.savefig(experiment_dir / "uncertainty_surface.png", dpi=300, bbox_inches="tight")
plt.close(fig)
