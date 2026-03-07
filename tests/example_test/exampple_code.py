import time
import torch
import gpytorch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from tqdm import tqdm

# ==========================================
# PART 1: GENERATE SYNTHETIC DUBLIN NO2 DATA
# Using real Dublin coordinates
# ==========================================
start_time = time.perf_counter()
run_timestamp = time.strftime("%Y%m%d_%H%M%S")
print("Generating Synthetic Dublin NO2...")

# 1. Define Dublin Study Area (Real Coordinates)
# Bounding box covering Greater Dublin Area
LON_MIN, LON_MAX = -6.45, -6.10  # West to East (~28 km)
LAT_MIN, LAT_MAX = 53.25, 53.45  # South to North (~22 km)

# Calculate center from bounding box
CENTER_LON = (LON_MIN + LON_MAX) / 2  # -6.275
CENTER_LAT = (LAT_MIN + LAT_MAX) / 2  # 53.35

print(f"Study area center: ({CENTER_LON:.4f}, {CENTER_LAT:.4f})")

# 2. Create Prediction Grid (~10m resolution)
# Use separate lat/lon degree steps to account for latitude-dependent longitude scale.
target_resolution_m = 10.0
meters_per_degree_lat = 111_320.0
meters_per_degree_lon = 111_320.0 * np.cos(np.deg2rad(CENTER_LAT))
lat_res = target_resolution_m / meters_per_degree_lat
lon_res = target_resolution_m / meters_per_degree_lon
lons = np.arange(LON_MIN, LON_MAX, lon_res)
lats = np.arange(LAT_MIN, LAT_MAX, lat_res)
N_lon, N_lat = len(lons), len(lats)

print(f"Grid dimensions: {N_lon} x {N_lat} = {N_lon * N_lat} points")
print(f"Resolution: ~{target_resolution_m:.0f}m")
print(f"Extent: {LON_MIN:.2f} to {LON_MAX:.2f} E, {LAT_MIN:.2f} to {LAT_MAX:.2f} N")

# Create meshgrid
lon_grid, lat_grid = np.meshgrid(lons, lats)

# 3. Create Ground Truth NO2 Features (Synthetic but geographically realistic)
# Normalize coordinates for feature calculation (centered on study area)
lon_norm = (lon_grid - CENTER_LON) / (LON_MAX - LON_MIN)
lat_norm = (lat_grid - CENTER_LAT) / (LAT_MAX - LAT_MIN)

# A. M50 Ring Road (High NO2 from traffic) - circular ring around Dublin
radius = np.sqrt(lon_norm**2 + lat_norm**2)
m50_ring = np.exp(-(radius - 0.25)**2 / 0.005) * 12.0  # Peak ~12 ppb on M50

# B. City Center (NO2 hotspot from urban activity)
city_center = np.exp(-(lon_norm**2 + lat_norm**2) / 0.02) * 15.0  # Peak ~15 ppb in center

# C. Background NO2 (Gradient West-to-East, cleaner Atlantic air from west)
background = ((lon_grid - LON_MIN) / (LON_MAX - LON_MIN)) * 8.0 + 5.0  # 5-13 ppb gradient

# D. Port/Industrial area (Dublin Port - northeast)
port_lon_norm = (lon_grid - (-6.15)) / (LON_MAX - LON_MIN)
port_lat_norm = (lat_grid - 53.35) / (LAT_MAX - LAT_MIN)
port_hotspot = np.exp(-(port_lon_norm**2 + port_lat_norm**2) / 0.01) * 8.0

# E. Ground Truth NO2 (What we want to predict) in mg/m^3
ground_truth = background + city_center + m50_ring + port_hotspot
# Add randomness (weather/atmospheric variability)
ground_truth += np.random.normal(0, 1.5, ground_truth.shape)
ground_truth = np.clip(ground_truth, 0, None)  # NO2 can't be negative

print(f"Ground truth NO2 range: {ground_truth.min():.1f} - {ground_truth.max():.1f} mg/m^3")
print(f"Time so far: {time.perf_counter() - start_time:.1f}s")

# 4. Create The "Input" Datasets (Covariates)
# Input A: ADMS NO2 Model (Smoothed & Biased - Misses sharp traffic gradients)
adms_feature = gaussian_filter(ground_truth, sigma=5.0) * 0.85

# Input B: REAL Traffic Volume Data (from Dublin traffic counters)
# Load traffic data and interpolate to grid using IDW
TRAFFIC_CSV = "/media/gabriel-oduori/SERVER/dev_space/data-tools/traffic/dublin_stations_traffic_timeseries.csv"

print(f"\nLoading traffic data from: {TRAFFIC_CSV}")
traffic_df = pd.read_csv(TRAFFIC_CSV)
traffic_df = traffic_df[traffic_df['traffic_volume'].notna()].copy()

# Filter to study area
traffic_df = traffic_df[
    (traffic_df['longitude'] >= LON_MIN) & (traffic_df['longitude'] <= LON_MAX) &
    (traffic_df['latitude'] >= LAT_MIN) & (traffic_df['latitude'] <= LAT_MAX)
].copy()

# Aggregate: mean traffic volume per site
traffic_sites = traffic_df.groupby('site_id').agg({
    'longitude': 'first',
    'latitude': 'first',
    'site_description_lower': 'first',
    'traffic_volume': 'mean'
}).reset_index()

print(f"Found {len(traffic_sites)} traffic sites in study area")
print(f"Traffic volume range: {traffic_sites['traffic_volume'].min():.0f} - {traffic_sites['traffic_volume'].max():.0f} veh/hr")

# Interpolate traffic to grid using IDW (Inverse Distance Weighting)
def idw_interpolate(grid_lon, grid_lat, site_lons, site_lats, site_values, power=2):
    """Interpolate point values to a grid using IDW."""
    result = np.zeros_like(grid_lon)

    for i in range(grid_lon.shape[0]):
        for j in range(grid_lon.shape[1]):
            # Distance from grid point to all sites
            dist = np.sqrt((site_lons - grid_lon[i, j])**2 + (site_lats - grid_lat[i, j])**2)

            # Handle exact matches (avoid division by zero)
            if np.any(dist < 1e-10):
                result[i, j] = site_values[dist < 1e-10][0]
            else:
                weights = 1.0 / (dist ** power)
                result[i, j] = np.sum(weights * site_values) / np.sum(weights)

    return result

print("Interpolating traffic to grid (IDW)...")
traffic_start = time.perf_counter()
traffic_feature = idw_interpolate(
    lon_grid, lat_grid,
    traffic_sites['longitude'].values,
    traffic_sites['latitude'].values,
    traffic_sites['traffic_volume'].values,
    power=2
)
print(f"Traffic interpolation time: {time.perf_counter() - traffic_start:.1f}s")
print(f"Traffic feature range: {traffic_feature.min():.0f} - {traffic_feature.max():.0f} veh/hr")

# Normalize traffic feature for model input (scale to similar range as other features)
traffic_feature_norm = traffic_feature / traffic_feature.max()  # 0-1 range

# Input C: LUR Model Predictions
LUR_CSV = "/media/gabriel-oduori/SERVER/dev_space/data-tools/lur_results/lur_predictions.csv"

print(f"\nLoading LUR predictions from: {LUR_CSV}")
lur_df = pd.read_csv(LUR_CSV)

# Filter to study area
lur_df = lur_df[
    (lur_df['longitude'] >= LON_MIN) & (lur_df['longitude'] <= LON_MAX) &
    (lur_df['latitude'] >= LAT_MIN) & (lur_df['latitude'] <= LAT_MAX)
].copy()

print(f"Found {len(lur_df)} LUR grid points in study area")
print(f"LUR predicted NO2 range: {lur_df['predicted_no2'].min():.1f} to {lur_df['predicted_no2'].max():.1f} mg/m^3")

# Clip extreme values to reasonable range (0-100 mg/m^3)
lur_df['predicted_no2_clipped'] = lur_df['predicted_no2'].clip(0, 100)
print(f"LUR clipped range: {lur_df['predicted_no2_clipped'].min():.1f} to {lur_df['predicted_no2_clipped'].max():.1f} mg/m^3")

# Interpolate LUR to our prediction grid using IDW
print("Interpolating LUR to prediction grid (IDW)...")
lur_start = time.perf_counter()
lur_feature = idw_interpolate(
    lon_grid, lat_grid,
    lur_df['longitude'].values,
    lur_df['latitude'].values,
    lur_df['predicted_no2_clipped'].values,
    power=2
)
print(f"LUR interpolation time: {time.perf_counter() - lur_start:.1f}s")
print(f"LUR feature range: {lur_feature.min():.1f} - {lur_feature.max():.1f} mg/m^3")

# Normalize LUR for model input
lur_feature_norm = lur_feature / lur_feature.max()  # 0-1 range

# 5. Load EPA NO2 Sensors with REAL NO2 measurements
# Load REAL EPA Ireland station coordinates and NO2 timeseries from CSV

EPA_TIMESERIES_CSV = "/media/gabriel-oduori/SERVER/dev_space/data-tools/airquality/dublin_stations_epa_timeseries.csv"

# Choose whether to use real NO2 or synthetic ground truth values
USE_REAL_NO2 = True  # Set to False to use synthetic values from ground_truth

print(f"\nLoading EPA NO2 data from: {EPA_TIMESERIES_CSV}")
epa_df = pd.read_csv(EPA_TIMESERIES_CSV)

# Filter to rows with valid NO2 readings
epa_df = epa_df[epa_df['epa_no2'].notna()].copy()

# Filter stations within our study area
epa_df = epa_df[
    (epa_df['longitude'] >= LON_MIN) & (epa_df['longitude'] <= LON_MAX) &
    (epa_df['latitude'] >= LAT_MIN) & (epa_df['latitude'] <= LAT_MAX)
].copy()

# Aggregate: compute mean NO2 per station (or pick a specific timestamp)
# Option 1: Use mean NO2 (time-averaged spatial map)
# Option 2: Use a specific hour (snapshot)

AGGREGATION_METHOD = "mean"  # "mean" or "snapshot"
SNAPSHOT_TIME = "2023-06-15 12:00:00"  # Used if AGGREGATION_METHOD = "snapshot"

if AGGREGATION_METHOD == "mean":
    print("Using MEAN NO2 values (time-averaged)")
    station_data = epa_df.groupby('station_id').agg({
        'longitude': 'first',
        'latitude': 'first',
        'location': 'first',
        'epa_no2': 'mean'
    }).reset_index()
else:
    print(f"Using SNAPSHOT at {SNAPSHOT_TIME}")
    station_data = epa_df[epa_df['timestamp_utc'] == SNAPSHOT_TIME].copy()

# Extract coordinates and values
sensor_lons = station_data['longitude'].values
sensor_lats = station_data['latitude'].values
station_names = station_data['location'].tolist()
station_ids = station_data['station_id'].tolist()
real_no2_vals = station_data['epa_no2'].values
n_sensors = len(sensor_lons)

print(f"\nFound {n_sensors} EPA stations with NO2 data:")
for i, (name, lon, lat, no2) in enumerate(zip(station_names, sensor_lons, sensor_lats, real_no2_vals)):
    print(f"  {i+1}. {name}: {no2:.1f} mg/m^3 ({lon:.4f}, {lat:.4f})")

# Find nearest grid cell for each sensor
sensor_lon_idx = np.array([np.argmin(np.abs(lons - lon)) for lon in sensor_lons])
sensor_lat_idx = np.array([np.argmin(np.abs(lats - lat)) for lat in sensor_lats])

# Choose training target: real NO2 or synthetic ground truth
if USE_REAL_NO2:
    train_y_vals = real_no2_vals
    print(f"\n>>> Using REAL EPA NO2 measurements for training <<<")
else:
    train_y_vals = ground_truth[sensor_lat_idx, sensor_lon_idx]
    print(f"\n>>> Using SYNTHETIC ground truth for training <<<")

train_coords = np.column_stack((sensor_lons, sensor_lats))

# Extract covariate values at sensor locations
train_adms = adms_feature[sensor_lat_idx, sensor_lon_idx]
train_traffic = traffic_feature_norm[sensor_lat_idx, sensor_lon_idx]
train_lur = lur_feature_norm[sensor_lat_idx, sensor_lon_idx]

print(f"\nTraining data summary:")
print(f"  NO2 range: {train_y_vals.min():.1f} - {train_y_vals.max():.1f} mg/m^3")
print(f"  NO2 mean:  {train_y_vals.mean():.1f} mg/m^3")

# ==========================================
# PART 2: DATA PREPARATION (TENSORIZING)
# ==========================================

# Normalize coordinates for GP (important for lengthscale learning)
lon_mean, lon_std = train_coords[:, 0].mean(), train_coords[:, 0].std()
lat_mean, lat_std = train_coords[:, 1].mean(), train_coords[:, 1].std()

train_coords_norm = np.column_stack([
    (train_coords[:, 0] - lon_mean) / lon_std,
    (train_coords[:, 1] - lat_mean) / lat_std
])

# Stack inputs: [lon_norm, lat_norm, adms_val, traffic_val, lur_val]
train_x_tensor = torch.tensor(
    np.column_stack((train_coords_norm, train_adms, train_traffic, train_lur)),
    dtype=torch.float32
)
train_y_tensor = torch.tensor(train_y_vals, dtype=torch.float32)

# Create the Prediction Grid (Test Data)
flat_coords = np.column_stack((lon_grid.ravel(), lat_grid.ravel()))
flat_coords_norm = np.column_stack([
    (flat_coords[:, 0] - lon_mean) / lon_std,
    (flat_coords[:, 1] - lat_mean) / lat_std
])
flat_adms = adms_feature.ravel()
flat_traffic = traffic_feature_norm.ravel()
flat_lur = lur_feature_norm.ravel()

test_x_tensor = torch.tensor(
    np.column_stack((flat_coords_norm, flat_adms, flat_traffic, flat_lur)),
    dtype=torch.float32
)

print(f"\nTraining on {len(train_y_tensor)} sensor points.")
print(f"Predicting on {len(test_x_tensor)} grid pixels.")

# ==========================================
# PART 3: BUILD THE FUSION MODEL
# ==========================================

# Custom Mean Function: Use LUR as prior mean
# GP learns residuals (corrections) to LUR predictions
class LURMean(gpytorch.means.Mean):
    """Mean function that returns LUR predictions at input locations."""
    def __init__(self, lur_values_tensor):
        super().__init__()
        # Store LUR values for training points
        self.register_buffer('lur_values', lur_values_tensor)
        # Learn a scaling factor for LUR
        self.lur_scale = torch.nn.Parameter(torch.tensor(1.0))
        self.lur_bias = torch.nn.Parameter(torch.tensor(0.0))

    def forward(self, x):
        # x has shape (n, 5) where dim 4 is LUR (normalized)
        # Extract LUR dimension and scale back to original units
        lur_norm = x[:, 4]  # Normalized LUR (0-1)
        # Return scaled LUR as mean
        return self.lur_scale * lur_norm * lur_feature.max() + self.lur_bias


class FusionGPModel(gpytorch.models.ExactGP):
    """Original model with constant mean + additive kernel."""
    def __init__(self, train_x, train_y, likelihood):
        super(FusionGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()

        # KERNEL ARCHITECTURE (The "Secret Sauce")
        # ----------------------------------------
        # Kernel 1: Spatial RBF (Active on dims 0,1 -> lon, lat)
        # Handles the "Residuals" (spatial patterns not explained by covariates)
        self.spatial_kernel = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(active_dims=[0, 1])
        )

        # Kernel 2: Linear Feature Kernel (Active on dims 2,3,4 -> ADMS, Traffic, LUR)
        # Handles the "Physics" (Land Use Regression)
        # Learns: NO2 = a*ADMS + b*Traffic + c*LUR
        self.feature_kernel = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.LinearKernel(active_dims=[2, 3, 4])
        )

        # Combine them (Additive Model)
        self.covar_module = self.spatial_kernel + self.feature_kernel

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class LURResidualGPModel(gpytorch.models.ExactGP):
    """Improved model: LUR as mean function, GP learns spatial residuals."""
    def __init__(self, train_x, train_y, likelihood):
        super(LURResidualGPModel, self).__init__(train_x, train_y, likelihood)

        # Mean = LUR prediction (learns scale and bias correction)
        self.mean_module = LURMean(train_x[:, 4])

        # Kernel: Spatial RBF only (learns residual spatial patterns)
        # Simpler kernel works better with sparse data
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(active_dims=[0, 1])
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


# Choose model type
USE_LUR_MEAN = True  # Set to False to use original constant-mean model

if USE_LUR_MEAN:
    print("\n>>> Using LUR-as-mean GP model (learning residuals) <<<")
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    model = LURResidualGPModel(train_x_tensor, train_y_tensor, likelihood)
else:
    print("\n>>> Using original constant-mean GP model <<<")
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    model = FusionGPModel(train_x_tensor, train_y_tensor, likelihood)

# ==========================================
# PART 4: TRAINING LOOP
# ==========================================
print("\nTraining GP Model (Optimizing Hyperparameters)...")
train_start = time.perf_counter()
model.train()
likelihood.train()
optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

pbar = tqdm(range(100), desc="Training GP")
for i in pbar:
    optimizer.zero_grad()
    output = model(train_x_tensor)
    loss = -mll(output, train_y_tensor)
    loss.backward()
    optimizer.step()
    pbar.set_postfix(loss=f"{loss.item():.3f}")
print(f"Training time: {time.perf_counter() - train_start:.1f}s")

# ==========================================
# PART 5: PREDICTION & PLOTTING
# ==========================================
print("\nPredicting on full grid...")
predict_start = time.perf_counter()
model.eval()
likelihood.eval()

with torch.no_grad(), gpytorch.settings.fast_pred_var():
    # Make prediction on the full grid
    observed_pred = likelihood(model(test_x_tensor))

    # Reshape back to grid dimensions
    pred_mean = observed_pred.mean.numpy().reshape(N_lat, N_lon)
    pred_var = observed_pred.variance.numpy().reshape(N_lat, N_lon)
    pred_std = np.sqrt(pred_var)
print(f"Prediction time: {time.perf_counter() - predict_start:.1f}s")

# ==========================================
# PART 6: VISUALIZATION
# ==========================================
print("\nGenerating visualization...")
if AGGREGATION_METHOD == "snapshot":
    map_timestamp = f"Timestamp: {SNAPSHOT_TIME} UTC"
else:
    map_timestamp = "Timestamp: time-averaged"

fig, axs = plt.subplots(4, 2, figsize=(12, 20))
fig.suptitle("Dublin NO2 Fusion: EPA + Traffic + LUR", fontsize=14, fontweight='bold')
fig.text(0.5, 0.01, map_timestamp, ha='center', fontsize=9)

# Common extent for all maps
extent = [LON_MIN, LON_MAX, LAT_MIN, LAT_MAX]
vmin, vmax = 0, 40  # NO2 scale in mg/m^3

# 1. Traffic Volume (Input covariate)
im0 = axs[0, 0].imshow(traffic_feature, cmap='YlOrRd', extent=extent, origin='lower', aspect='auto')
axs[0, 0].scatter(traffic_sites['longitude'], traffic_sites['latitude'],
                   c='blue', s=5, alpha=0.3, label=f'Traffic sites (n={len(traffic_sites)})')
axs[0, 0].set_title("1. Traffic Volume (Real Data)")
axs[0, 0].set_xlabel("Longitude")
axs[0, 0].set_ylabel("Latitude")
axs[0, 0].legend(loc='upper right', fontsize=8)
plt.colorbar(im0, ax=axs[0, 0], label='Vehicles/hr', shrink=0.8)

# 2. ADMS Input (Synthetic baseline)
im1 = axs[0, 1].imshow(adms_feature, cmap='jet', vmin=vmin, vmax=vmax,
                        extent=extent, origin='lower', aspect='auto')
axs[0, 1].set_title("2. ADMS NO2 (Synthetic Baseline)")
axs[0, 1].set_xlabel("Longitude")
axs[0, 1].set_ylabel("Latitude")
plt.colorbar(im1, ax=axs[0, 1], label='NO2 (mg/m^3)', shrink=0.8)

# 3. LUR Predictions (Input covariate)
im2 = axs[1, 0].imshow(lur_feature, cmap='jet', vmin=vmin, vmax=vmax,
                        extent=extent, origin='lower', aspect='auto')
axs[1, 0].set_title("3. LUR Predictions (Real Data)")
axs[1, 0].set_xlabel("Longitude")
axs[1, 0].set_ylabel("Latitude")
plt.colorbar(im2, ax=axs[1, 0], label='NO2 (mg/m^3)', shrink=0.8)

# 4. EPA Sensors (Training points)
im3 = axs[1, 1].imshow(np.zeros_like(ground_truth), cmap='gray', extent=extent, origin='lower', aspect='auto', alpha=0.3)
scatter = axs[1, 1].scatter(sensor_lons, sensor_lats, c=train_y_vals, cmap='jet',
                             vmin=vmin, vmax=vmax, s=150, edgecolors='black', zorder=5)
for i, name in enumerate(station_names):
    axs[1, 1].annotate(name.split(',')[0][:15], (sensor_lons[i], sensor_lats[i]),
                        fontsize=6, ha='center', va='bottom', xytext=(0, 5),
                        textcoords='offset points')
axs[1, 1].set_title(f"4. EPA Sensors (n={n_sensors}, Real NO2)")
axs[1, 1].set_xlabel("Longitude")
axs[1, 1].set_ylabel("Latitude")
axs[1, 1].set_xlim(LON_MIN, LON_MAX)
axs[1, 1].set_ylim(LAT_MIN, LAT_MAX)
plt.colorbar(scatter, ax=axs[1, 1], label='NO2 (mg/m^3)', shrink=0.8)

# 5. GP Fused Prediction
im4 = axs[2, 0].imshow(pred_mean, cmap='jet', vmin=vmin, vmax=vmax,
                        extent=extent, origin='lower', aspect='auto')
axs[2, 0].scatter(sensor_lons, sensor_lats, c='white', edgecolors='black',
                   s=30, zorder=5, alpha=0.7)
axs[2, 0].set_title("5. Fused NO2 Prediction (GP)")
axs[2, 0].set_xlabel("Longitude")
axs[2, 0].set_ylabel("Latitude")
plt.colorbar(im4, ax=axs[2, 0], label='NO2 (mg/m^3)', shrink=0.8)

# 6. Uncertainty (Standard Deviation)
im5 = axs[2, 1].imshow(pred_std, cmap='magma', extent=extent, origin='lower', aspect='auto')
axs[2, 1].scatter(sensor_lons, sensor_lats, c='cyan', edgecolors='black',
                   s=30, zorder=5, alpha=0.7)
axs[2, 1].set_title("6. Prediction Uncertainty (Std Dev)")
axs[2, 1].set_xlabel("Longitude")
axs[2, 1].set_ylabel("Latitude")
plt.colorbar(im5, ax=axs[2, 1], label='Std Dev (mg/m^3)', shrink=0.8)

# 7. Residuals at sensor locations (if using real NO2)
if USE_REAL_NO2:
    pred_at_sensors = pred_mean[sensor_lat_idx, sensor_lon_idx]
    residuals_at_sensors = pred_at_sensors - train_y_vals

    im6 = axs[3, 0].imshow(np.zeros_like(ground_truth), cmap='gray', extent=extent, origin='lower', aspect='auto', alpha=0.3)
    scatter2 = axs[3, 0].scatter(sensor_lons, sensor_lats, c=residuals_at_sensors, cmap='RdBu_r',
                                  vmin=-15, vmax=15, s=150, edgecolors='black', zorder=5)
    axs[3, 0].set_title("7. Prediction Errors at Sensors")
    axs[3, 0].set_xlabel("Longitude")
    axs[3, 0].set_ylabel("Latitude")
    axs[3, 0].set_xlim(LON_MIN, LON_MAX)
    axs[3, 0].set_ylim(LAT_MIN, LAT_MAX)
    plt.colorbar(scatter2, ax=axs[3, 0], label='Error (mg/m^3)', shrink=0.8)
else:
    # Compare to synthetic ground truth
    residuals_grid = pred_mean - ground_truth
    im6 = axs[3, 0].imshow(residuals_grid, cmap='RdBu_r', vmin=-10, vmax=10,
                            extent=extent, origin='lower', aspect='auto')
    axs[3, 0].set_title("7. Residuals (Pred - Truth)")
    axs[3, 0].set_xlabel("Longitude")
    axs[3, 0].set_ylabel("Latitude")
    plt.colorbar(im6, ax=axs[3, 0], label='Error (mg/m^3)', shrink=0.8)

# 8. GP vs LUR comparison
im7 = axs[3, 1].imshow(pred_mean - lur_feature, cmap='RdBu_r', vmin=-15, vmax=15,
                        extent=extent, origin='lower', aspect='auto')
axs[3, 1].set_title("8. GP Fusion - LUR (Improvement)")
axs[3, 1].set_xlabel("Longitude")
axs[3, 1].set_ylabel("Latitude")
plt.colorbar(im7, ax=axs[3, 1], label='Difference (mg/m^3)', shrink=0.8)

plt.tight_layout()
fusion_map_path = f"dublin_no2_fusion_test_{run_timestamp}.png"
plt.savefig(fusion_map_path, dpi=150, bbox_inches='tight')
print(f"Saved: {fusion_map_path}")

# Standalone uncertainty plot
plt.figure(figsize=(7, 6))
im_unc = plt.imshow(pred_std, cmap='magma', extent=extent, origin='lower', aspect='auto')
plt.scatter(sensor_lons, sensor_lats, c='cyan', edgecolors='black',
            s=25, alpha=0.7, zorder=5)
plt.title("Prediction Uncertainty (Std Dev)")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.colorbar(im_unc, label='Std Dev (mg/m^3)', shrink=0.8)
plt.figtext(0.5, 0.01, map_timestamp, ha='center', fontsize=9)
plt.tight_layout()
uncertainty_map_path = f"dublin_no2_uncertainty_{run_timestamp}.png"
plt.savefig(uncertainty_map_path, dpi=150, bbox_inches='tight')
print(f"Saved: {uncertainty_map_path}")
plt.show()
print(f"Total runtime: {time.perf_counter() - start_time:.1f}s")

# ==========================================
# PART 7: EVALUATION METRICS
# ==========================================
print("\n" + "="*50)
print("EVALUATION METRICS")
print("="*50)

if USE_REAL_NO2:
    # Cross-validation: compare predictions at sensor locations to actual values
    # Get predicted values at sensor locations
    pred_at_sensors = pred_mean[sensor_lat_idx, sensor_lon_idx]

    residuals = pred_at_sensors - train_y_vals
    rmse = np.sqrt(np.mean(residuals**2))
    mae = np.mean(np.abs(residuals))
    bias = np.mean(residuals)
    correlation = np.corrcoef(pred_at_sensors, train_y_vals)[0, 1]
    r_squared = correlation**2

    print("(Training fit - predictions at sensor locations vs observed)")
    print(f"RMSE:        {rmse:.2f} mg/m^3")
    print(f"MAE:         {mae:.2f} mg/m^3")
    print(f"Bias:        {bias:.2f} ppb")
    print(f"Correlation: {correlation:.3f}")
    print(f"R²:          {r_squared:.3f}")

    print(f"\nPer-station comparison:")
    print(f"{'Station':<35} {'Observed':>10} {'Predicted':>10} {'Error':>10}")
    print("-" * 70)
    for i, name in enumerate(station_names):
        obs = train_y_vals[i]
        pred = pred_at_sensors[i]
        err = pred - obs
        print(f"{name:<35} {obs:>10.1f} {pred:>10.1f} {err:>+10.1f}")
else:
    # Compare to synthetic ground truth across entire grid
    residuals = pred_mean - ground_truth
    rmse = np.sqrt(np.mean(residuals**2))
    mae = np.mean(np.abs(residuals))
    bias = np.mean(residuals)
    correlation = np.corrcoef(pred_mean.ravel(), ground_truth.ravel())[0, 1]
    r_squared = correlation**2

    print("(vs Synthetic Ground Truth - full grid)")
    print(f"RMSE:        {rmse:.2f} mg/m^3")
    print(f"MAE:         {mae:.2f} mg/m^3")
    print(f"Bias:        {bias:.2f} ppb")
    print(f"Correlation: {correlation:.3f}")
    print(f"R²:          {r_squared:.3f}")
    print(f"\nMean predicted NO2: {pred_mean.mean():.1f} mg/m^3")
    print(f"Mean ground truth:  {ground_truth.mean():.1f} mg/m^3")

# ==========================================
# PART 8: LEAVE-ONE-OUT CROSS-VALIDATION
# ==========================================
print("\n" + "="*50)
print("LEAVE-ONE-OUT CROSS-VALIDATION (LOOCV)")
print("="*50)
print("Training N models, each holding out one station...")

loocv_predictions = np.zeros(n_sensors)
loocv_uncertainties = np.zeros(n_sensors)
lur_at_sensors = lur_feature[sensor_lat_idx, sensor_lon_idx]

for i in tqdm(range(n_sensors), desc="LOOCV"):
    # Create train/test split: leave out station i
    mask = np.ones(n_sensors, dtype=bool)
    mask[i] = False

    # Training data (N-1 stations)
    loo_train_x = train_x_tensor[mask]
    loo_train_y = train_y_tensor[mask]

    # Test point (held-out station)
    loo_test_x = train_x_tensor[i:i+1]

    # Build and train model (use same model type as main training)
    loo_likelihood = gpytorch.likelihoods.GaussianLikelihood()
    if USE_LUR_MEAN:
        loo_model = LURResidualGPModel(loo_train_x, loo_train_y, loo_likelihood)
    else:
        loo_model = FusionGPModel(loo_train_x, loo_train_y, loo_likelihood)

    loo_model.train()
    loo_likelihood.train()
    loo_optimizer = torch.optim.Adam(loo_model.parameters(), lr=0.1)
    loo_mll = gpytorch.mlls.ExactMarginalLogLikelihood(loo_likelihood, loo_model)

    # Train (fewer iterations for speed)
    for _ in range(50):
        loo_optimizer.zero_grad()
        output = loo_model(loo_train_x)
        loss = -loo_mll(output, loo_train_y)
        loss.backward()
        loo_optimizer.step()

    # Predict at held-out station
    loo_model.eval()
    loo_likelihood.eval()
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        loo_pred = loo_likelihood(loo_model(loo_test_x))
        loocv_predictions[i] = loo_pred.mean.numpy()[0]
        loocv_uncertainties[i] = np.sqrt(loo_pred.variance.numpy()[0])

# LOOCV Metrics for GP Fusion
loocv_residuals = loocv_predictions - train_y_vals
loocv_rmse = np.sqrt(np.mean(loocv_residuals**2))
loocv_mae = np.mean(np.abs(loocv_residuals))
loocv_bias = np.mean(loocv_residuals)
loocv_corr = np.corrcoef(loocv_predictions, train_y_vals)[0, 1]
loocv_r2 = loocv_corr**2

# LUR Metrics (baseline comparison)
lur_residuals = lur_at_sensors - train_y_vals
lur_rmse = np.sqrt(np.mean(lur_residuals**2))
lur_mae = np.mean(np.abs(lur_residuals))
lur_bias = np.mean(lur_residuals)
lur_corr = np.corrcoef(lur_at_sensors, train_y_vals)[0, 1]
lur_r2 = lur_corr**2

print("\n" + "-"*70)
print(f"{'Metric':<15} {'GP Fusion (LOOCV)':>20} {'LUR (Direct)':>20} {'Improvement':>15}")
print("-"*70)
print(f"{'RMSE':<15} {loocv_rmse:>17.2f} mg/m^3 {lur_rmse:>17.2f} mg/m^3 {(lur_rmse - loocv_rmse):>+12.2f}")
print(f"{'MAE':<15} {loocv_mae:>17.2f} mg/m^3 {lur_mae:>17.2f} mg/m^3 {(lur_mae - loocv_mae):>+12.2f}")
print(f"{'Bias':<15} {loocv_bias:>17.2f} mg/m^3 {lur_bias:>17.2f} mg/m^3 {abs(lur_bias) - abs(loocv_bias):>+12.2f}")
print(f"{'Correlation':<15} {loocv_corr:>20.3f} {lur_corr:>20.3f} {(loocv_corr - lur_corr):>+15.3f}")
print(f"{'R²':<15} {loocv_r2:>20.3f} {lur_r2:>20.3f} {(loocv_r2 - lur_r2):>+15.3f}")
print("-"*70)

print(f"\nPer-station LOOCV results:")
print(f"{'Station':<35} {'Observed':>10} {'GP Pred':>10} {'GP Err':>10} {'LUR':>10} {'LUR Err':>10}")
print("-"*90)
for i, name in enumerate(station_names):
    obs = train_y_vals[i]
    gp_pred = loocv_predictions[i]
    gp_err = gp_pred - obs
    lur_val = lur_at_sensors[i]
    lur_err = lur_val - obs
    print(f"{name:<35} {obs:>10.1f} {gp_pred:>10.1f} {gp_err:>+10.1f} {lur_val:>10.1f} {lur_err:>+10.1f}")

# Summary
print("\n" + "="*50)
print("SUMMARY")
print("="*50)
model_name = "GP+LUR (residual learning)" if USE_LUR_MEAN else "GP Fusion (constant mean)"
print(f"Model: {model_name}")
print(f"Training points: {n_sensors} EPA stations")

if loocv_rmse < lur_rmse:
    improvement_pct = (lur_rmse - loocv_rmse) / lur_rmse * 100
    print(f"\n{model_name} OUTPERFORMS standalone LUR by {improvement_pct:.1f}% (LOOCV RMSE)")
    print(f"  LUR RMSE:      {lur_rmse:.2f} ug/m^3")
    print(f"  GP+LUR RMSE:   {loocv_rmse:.2f} ug/m^3")
else:
    degradation_pct = (loocv_rmse - lur_rmse) / lur_rmse * 100
    print(f"\nStandalone LUR outperforms {model_name} by {degradation_pct:.1f}% (LOOCV RMSE)")
    print(f"  LUR RMSE:      {lur_rmse:.2f} ug/m^3")
    print(f"  GP RMSE:       {loocv_rmse:.2f} ug/m^3")
    print("\nPossible improvements:")
    print("  - Add more training data (low-cost sensors, passive samplers)")
    print("  - Try different kernel (Matern, periodic)")
    print("  - Increase training iterations")

print(f"\nNote: LOOCV provides unbiased estimate of generalization performance.")
print(f"      Training fit metrics are overly optimistic.")
