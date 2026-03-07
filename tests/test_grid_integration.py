"""
Integration test: AnalysisGrid with a simple GP model.

This test demonstrates the full workflow without needing real data files.
Uses a simple GPyTorch ExactGP to verify the grid integration works.

Run with: python tests/test_grid_integration.py
"""

import numpy as np
import torch
import gpytorch
import sys
sys.path.insert(0, '/media/gabriel-oduori/SERVER/dev_space/FusionGP')

from src.data.grid import AnalysisGrid, dublin_grid


# =============================================================================
# Simple GP Model (similar to example test code)
# =============================================================================

class SimpleGP(gpytorch.models.ExactGP):
    """Simple 2D spatial GP for testing."""

    def __init__(self, train_x, train_y, likelihood):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(ard_num_dims=2)
        )

    def forward(self, x):
        mean = self.mean_module(x)
        covar = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean, covar)


# =============================================================================
# Tests
# =============================================================================

def test_grid_with_simple_gp():
    """Test AnalysisGrid workflow with a simple GP model."""
    print("\n" + "=" * 60)
    print("Integration Test: AnalysisGrid + Simple GP")
    print("=" * 60)

    # 1. Create a small analysis grid
    print("\n1. Creating analysis grid...")
    grid = AnalysisGrid(
        lon_min=-6.35, lon_max=-6.20,
        lat_min=53.30, lat_max=53.40,
        resolution_m=500,  # 500m resolution
        name='Dublin_Test'
    )
    print(f"   Grid: {grid.info.shape} = {grid.info.n_points} points")

    # 2. Create synthetic sensor data (like EPA stations)
    print("\n2. Creating synthetic sensor data...")
    np.random.seed(42)

    # 5 synthetic sensors
    sensor_lons = np.array([-6.32, -6.28, -6.25, -6.22, -6.30])
    sensor_lats = np.array([53.32, 53.35, 53.38, 53.33, 53.37])

    # Ground truth: spatial pattern (higher in center)
    def true_field(lon, lat):
        center_lon, center_lat = -6.275, 53.35
        dist = np.sqrt((lon - center_lon)**2 + (lat - center_lat)**2)
        return 30 - 50 * dist + np.random.normal(0, 1, size=lon.shape)

    sensor_values = true_field(sensor_lons, sensor_lats)

    print(f"   Sensors: {len(sensor_lons)}")
    print(f"   Value range: [{sensor_values.min():.1f}, {sensor_values.max():.1f}]")

    # 3. Add synthetic covariate (e.g., traffic)
    print("\n3. Adding traffic covariate...")
    traffic_lons = np.random.uniform(-6.35, -6.20, 20)
    traffic_lats = np.random.uniform(53.30, 53.40, 20)
    traffic_vals = np.random.uniform(50, 500, 20)

    grid.add_covariate_points('traffic', traffic_lons, traffic_lats, traffic_vals)
    print(f"   Traffic points: {len(traffic_vals)}")
    print(f"   Interpolated to grid: {grid.covariates['traffic'].shape}")

    # 4. Normalize coordinates for GP
    print("\n4. Preparing training data...")
    lon_mean, lon_std = sensor_lons.mean(), sensor_lons.std()
    lat_mean, lat_std = sensor_lats.mean(), sensor_lats.std()

    train_x = torch.tensor(np.column_stack([
        (sensor_lons - lon_mean) / lon_std,
        (sensor_lats - lat_mean) / lat_std,
    ]), dtype=torch.float32)

    train_y = torch.tensor(sensor_values, dtype=torch.float32)

    print(f"   Train X shape: {train_x.shape}")
    print(f"   Train Y shape: {train_y.shape}")

    # 5. Train simple GP
    print("\n5. Training GP model...")
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    model = SimpleGP(train_x, train_y, likelihood)

    model.train()
    likelihood.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    for i in range(50):
        optimizer.zero_grad()
        output = model(train_x)
        loss = -mll(output, train_y)
        loss.backward()
        optimizer.step()

    print(f"   Final loss: {loss.item():.3f}")

    # 6. Get prediction input from grid
    print("\n6. Getting prediction input from grid...")
    flat_lons, flat_lats = grid.get_flat_coords()

    # Normalize using same stats
    test_x = torch.tensor(np.column_stack([
        (flat_lons - lon_mean) / lon_std,
        (flat_lats - lat_mean) / lat_std,
    ]), dtype=torch.float32)

    print(f"   Test X shape: {test_x.shape}")

    # 7. Make predictions
    print("\n7. Making predictions...")
    model.eval()
    likelihood.eval()

    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        pred_dist = likelihood(model(test_x))
        pred_mean = pred_dist.mean.numpy()
        pred_std = np.sqrt(pred_dist.variance.numpy())

    print(f"   Predictions: {len(pred_mean)}")
    print(f"   Mean range: [{pred_mean.min():.1f}, {pred_mean.max():.1f}]")
    print(f"   Std range: [{pred_std.min():.2f}, {pred_std.max():.2f}]")

    # 8. Reshape to grid
    print("\n8. Reshaping to grid...")
    mean_grid = grid.reshape_to_grid(pred_mean)
    std_grid = grid.reshape_to_grid(pred_std)

    print(f"   Mean grid shape: {mean_grid.shape}")
    print(f"   Expected shape: {grid.info.shape}")

    assert mean_grid.shape == grid.info.shape
    assert std_grid.shape == grid.info.shape

    # 9. Test plotting (non-interactive)
    print("\n9. Testing visualization...")
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 3, figsize=(15, 4))

        # Traffic covariate
        grid.plot_covariate('traffic', ax=axes[0], cmap='YlOrRd',
                           title='Traffic (Covariate)')

        # Prediction mean
        grid.plot(mean_grid, ax=axes[1], cmap='jet',
                 title='GP Prediction', colorbar_label='NO2')
        axes[1].scatter(sensor_lons, sensor_lats, c='white',
                       edgecolors='black', s=50, zorder=5)

        # Uncertainty
        grid.plot(std_grid, ax=axes[2], cmap='magma',
                 title='Uncertainty (Std)', colorbar_label='Std')
        axes[2].scatter(sensor_lons, sensor_lats, c='cyan',
                       edgecolors='black', s=50, zorder=5)

        plt.tight_layout()
        plt.savefig('tests/test_grid_integration_output.png', dpi=100)
        plt.close()

        print("   Saved: tests/test_grid_integration_output.png")
        print("   [PASS]")
    except Exception as e:
        print(f"   Plotting skipped: {e}")

    # 10. Test DataFrame export
    print("\n10. Testing exports...")
    df = grid.to_dataframe()
    df['prediction'] = pred_mean
    df['uncertainty'] = pred_std

    print(f"   DataFrame: {df.shape}")
    print(f"   Columns: {list(df.columns)}")

    print("\n" + "=" * 60)
    print("Integration test PASSED!")
    print("=" * 60)

    return grid, mean_grid, std_grid


def test_grid_predictions_class():
    """Test the GridPredictions dataclass directly."""
    print("\n" + "=" * 60)
    print("Test: GridPredictions Class")
    print("=" * 60)

    from src.inference.predictor import GridPredictions

    # Create small grid
    grid = AnalysisGrid(
        lon_min=0, lon_max=1,
        lat_min=0, lat_max=1,
        resolution_m=50000,  # 50km
    )

    n_points = grid.info.n_points

    # Create fake predictions
    mean = np.random.uniform(20, 40, n_points)
    std = np.random.uniform(1, 5, n_points)

    # Create GridPredictions
    preds = GridPredictions(
        mean=mean,
        std=std,
        lower_ci={0.95: mean - 1.96 * std},
        upper_ci={0.95: mean + 1.96 * std},
        grid=grid,
        n_timestamps=1,
        timestamps=np.array([0.0]),
    )

    print(f"\n{preds.summary()}")

    # Test properties
    print(f"\nmean_grid shape: {preds.mean_grid.shape}")
    print(f"std_grid shape: {preds.std_grid.shape}")

    assert preds.mean_grid.shape == grid.info.shape
    assert preds.std_grid.shape == grid.info.shape

    # Test plotting
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        fig = preds.plot_with_uncertainty()
        plt.savefig('tests/test_grid_predictions_output.png', dpi=100)
        plt.close()
        print("\nSaved: tests/test_grid_predictions_output.png")
    except Exception as e:
        print(f"\nPlotting skipped: {e}")

    print("\n[PASS]")


def test_multi_timestamp():
    """Test spatio-temporal predictions."""
    print("\n" + "=" * 60)
    print("Test: Multi-Timestamp Predictions")
    print("=" * 60)

    from src.inference.predictor import GridPredictions

    grid = AnalysisGrid(
        lon_min=0, lon_max=1,
        lat_min=0, lat_max=1,
        resolution_m=100000,  # 100km for small grid
    )

    n_points = grid.info.n_points
    n_times = 3
    timestamps = np.array([0.0, 0.5, 1.0])

    # Create fake spatio-temporal predictions
    mean = np.random.uniform(20, 40, n_points * n_times)
    std = np.random.uniform(1, 5, n_points * n_times)

    preds = GridPredictions(
        mean=mean,
        std=std,
        lower_ci={0.95: mean - 1.96 * std},
        upper_ci={0.95: mean + 1.96 * std},
        grid=grid,
        n_timestamps=n_times,
        timestamps=timestamps,
    )

    print(f"\nGrid shape: {grid.info.shape}")
    print(f"N timestamps: {n_times}")
    print(f"Mean array shape: {mean.shape}")
    print(f"Mean grid shape: {preds.mean_grid.shape}")
    print(f"Expected: ({n_times}, {grid.info.n_lat}, {grid.info.n_lon})")

    assert preds.mean_grid.shape == (n_times, grid.info.n_lat, grid.info.n_lon)

    print("\n[PASS]")


if __name__ == '__main__':
    test_grid_with_simple_gp()
    test_grid_predictions_class()
    test_multi_timestamp()

    print("\n" + "=" * 60)
    print("All integration tests passed!")
    print("=" * 60)
