"""
Test AnalysisGrid functionality without running full FusionGP pipeline.

Run with: python -m pytest tests/test_analysis_grid.py -v
Or directly: python tests/test_analysis_grid.py
"""

import numpy as np
import sys
sys.path.insert(0, '/media/gabriel-oduori/SERVER/dev_space/FusionGP')

from src.data.grid import AnalysisGrid, GridInfo, dublin_grid, meters_per_degree_lon


def test_grid_creation():
    """Test basic grid creation."""
    print("\n=== Test: Grid Creation ===")

    grid = AnalysisGrid(
        lon_min=-6.45, lon_max=-6.10,
        lat_min=53.25, lat_max=53.45,
        resolution_m=500.0,  # 500m for faster test
        name='Dublin_Test'
    )

    print(f"Grid created: {grid.name}")
    print(f"  Shape: {grid.info.shape}")
    print(f"  N points: {grid.info.n_points:,}")
    print(f"  Resolution: {grid.info.resolution_m}m")
    print(f"  Extent: {grid.info.extent}")

    # Verify shape
    assert grid.info.n_lat > 0
    assert grid.info.n_lon > 0
    assert grid.lon_grid.shape == grid.info.shape
    assert grid.lat_grid.shape == grid.info.shape

    print("  [PASS]")
    return grid


def test_dublin_convenience():
    """Test dublin_grid convenience function."""
    print("\n=== Test: Dublin Convenience Function ===")

    grid = dublin_grid(resolution_m=1000)  # 1km for speed

    print(f"Dublin grid: {grid.info.shape}")
    assert grid.name == 'Dublin'
    assert grid.info.lon_min == -6.45
    assert grid.info.lat_min == 53.25

    print("  [PASS]")
    return grid


def test_aspect_ratio():
    """Test that aspect ratio correction works."""
    print("\n=== Test: Aspect Ratio Correction ===")

    grid = AnalysisGrid(
        lon_min=0.0, lon_max=1.0,
        lat_min=53.0, lat_max=54.0,  # ~53.5° latitude
        resolution_m=10000,  # 10km
    )

    # At ~53.5° latitude, lon degrees are smaller than lat degrees
    # So we should have MORE lon cells than lat cells for same degree extent
    print(f"  Lat cells: {grid.info.n_lat}")
    print(f"  Lon cells: {grid.info.n_lon}")
    print(f"  Lat res (deg): {grid.info.lat_res:.6f}")
    print(f"  Lon res (deg): {grid.info.lon_res:.6f}")

    # Lon resolution in degrees should be larger than lat (to get same meters)
    assert grid.info.lon_res > grid.info.lat_res, "Aspect ratio correction not working"

    print("  [PASS]")


def test_covariate_interpolation():
    """Test IDW interpolation of covariates."""
    print("\n=== Test: Covariate Interpolation (IDW) ===")

    grid = AnalysisGrid(
        lon_min=0.0, lon_max=1.0,
        lat_min=0.0, lat_max=1.0,
        resolution_m=50000,  # 50km for small test grid
    )

    # Create synthetic point data
    np.random.seed(42)
    n_points = 20
    point_lons = np.random.uniform(0.1, 0.9, n_points)
    point_lats = np.random.uniform(0.1, 0.9, n_points)
    point_vals = np.random.uniform(10, 50, n_points)

    print(f"  Grid shape: {grid.info.shape}")
    print(f"  Input points: {n_points}")
    print(f"  Value range: [{point_vals.min():.1f}, {point_vals.max():.1f}]")

    # Add covariate
    cov_grid = grid.add_covariate_points(
        'test_covariate',
        point_lons, point_lats, point_vals,
        interpolation='idw',
        idw_power=2.0
    )

    print(f"  Interpolated range: [{cov_grid.min():.1f}, {cov_grid.max():.1f}]")

    # Verify
    assert 'test_covariate' in grid.covariates
    assert cov_grid.shape == grid.info.shape
    # IDW should produce values within original range (approximately)
    assert cov_grid.min() >= point_vals.min() - 1
    assert cov_grid.max() <= point_vals.max() + 1

    print("  [PASS]")
    return grid


def test_prediction_input():
    """Test getting prediction input array."""
    print("\n=== Test: Prediction Input Generation ===")

    grid = AnalysisGrid(
        lon_min=-6.3, lon_max=-6.2,
        lat_min=53.3, lat_max=53.4,
        resolution_m=1000,  # 1km
    )

    # Add a covariate
    grid.add_covariate_points(
        'traffic',
        np.array([-6.25, -6.28]),
        np.array([53.35, 53.32]),
        np.array([100, 200]),
    )

    # Get prediction input (single timestamp)
    X = grid.get_prediction_input(timestamp=0.5)

    print(f"  Grid shape: {grid.info.shape}")
    print(f"  N points: {grid.info.n_points}")
    print(f"  X shape: {X.shape}")
    print(f"  X columns: [lat, lon, time, traffic]")

    # Verify shape: (n_points, 3 + n_covariates)
    assert X.shape[0] == grid.info.n_points
    assert X.shape[1] == 4  # lat, lon, time, traffic

    # Verify time column
    assert np.allclose(X[:, 2], 0.5)

    print("  [PASS]")


def test_prediction_input_multi_timestamp():
    """Test prediction input with multiple timestamps."""
    print("\n=== Test: Multi-Timestamp Prediction Input ===")

    grid = AnalysisGrid(
        lon_min=0.0, lon_max=0.1,
        lat_min=0.0, lat_max=0.1,
        resolution_m=5000,
    )

    timestamps = np.array([0.0, 0.5, 1.0])
    X = grid.get_prediction_input(timestamps=timestamps)

    print(f"  Grid points: {grid.info.n_points}")
    print(f"  Timestamps: {len(timestamps)}")
    print(f"  X shape: {X.shape}")

    # Should have n_points * n_timestamps rows
    assert X.shape[0] == grid.info.n_points * len(timestamps)

    print("  [PASS]")


def test_reshape_to_grid():
    """Test reshaping flat arrays back to grid."""
    print("\n=== Test: Reshape to Grid ===")

    grid = AnalysisGrid(
        lon_min=0.0, lon_max=1.0,
        lat_min=0.0, lat_max=1.0,
        resolution_m=50000,
    )

    # Simulate flat predictions
    n_points = grid.info.n_points
    flat_preds = np.arange(n_points, dtype=float)

    # Reshape
    grid_preds = grid.reshape_to_grid(flat_preds)

    print(f"  Flat shape: {flat_preds.shape}")
    print(f"  Grid shape: {grid_preds.shape}")
    print(f"  Expected: {grid.info.shape}")

    assert grid_preds.shape == grid.info.shape

    # Test multi-timestamp reshape
    n_times = 3
    flat_multi = np.arange(n_points * n_times, dtype=float)
    grid_multi = grid.reshape_to_grid(flat_multi, n_timestamps=n_times)

    print(f"  Multi-time shape: {grid_multi.shape}")
    assert grid_multi.shape == (n_times, *grid.info.shape)

    print("  [PASS]")


def test_dataframe_export():
    """Test DataFrame export."""
    print("\n=== Test: DataFrame Export ===")

    grid = AnalysisGrid(
        lon_min=0.0, lon_max=0.5,
        lat_min=0.0, lat_max=0.5,
        resolution_m=50000,
    )

    grid.add_covariate_points(
        'test_cov',
        np.array([0.25]),
        np.array([0.25]),
        np.array([100.0]),
    )

    df = grid.to_dataframe()

    print(f"  DataFrame shape: {df.shape}")
    print(f"  Columns: {list(df.columns)}")

    assert len(df) == grid.info.n_points
    assert 'longitude' in df.columns
    assert 'latitude' in df.columns
    assert 'test_cov' in df.columns

    print("  [PASS]")


def test_from_center():
    """Test grid creation from center point."""
    print("\n=== Test: From Center ===")

    grid = AnalysisGrid.from_center(
        center_lon=-6.275,
        center_lat=53.35,
        width_km=10,
        height_km=10,
        resolution_m=1000,
    )

    print(f"  Center: {grid.info.center}")
    print(f"  Shape: {grid.info.shape}")

    # Verify center is approximately correct
    actual_center = grid.info.center
    assert abs(actual_center[0] - (-6.275)) < 0.01
    assert abs(actual_center[1] - 53.35) < 0.01

    print("  [PASS]")


def test_plotting(show=False):
    """Test plotting functionality."""
    print("\n=== Test: Plotting ===")

    grid = AnalysisGrid(
        lon_min=0.0, lon_max=1.0,
        lat_min=0.0, lat_max=1.0,
        resolution_m=20000,
    )

    # Create synthetic data with spatial pattern
    data = np.sin(grid.lon_grid * np.pi) * np.cos(grid.lat_grid * np.pi) * 20 + 30

    print(f"  Data shape: {data.shape}")
    print(f"  Data range: [{data.min():.1f}, {data.max():.1f}]")

    try:
        import matplotlib
        if not show:
            matplotlib.use('Agg')  # Non-interactive backend
        import matplotlib.pyplot as plt

        ax = grid.plot(
            data,
            cmap='jet',
            vmin=10,
            vmax=50,
            title='Test Plot',
            colorbar_label='Test Units',
        )

        if show:
            plt.show()
        else:
            plt.close()

        print("  [PASS]")
    except ImportError:
        print("  [SKIP] matplotlib not available")


def run_all_tests():
    """Run all tests."""
    print("=" * 60)
    print("AnalysisGrid Test Suite")
    print("=" * 60)

    test_grid_creation()
    test_dublin_convenience()
    test_aspect_ratio()
    test_covariate_interpolation()
    test_prediction_input()
    test_prediction_input_multi_timestamp()
    test_reshape_to_grid()
    test_dataframe_export()
    test_from_center()
    test_plotting(show=False)

    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)


if __name__ == '__main__':
    run_all_tests()
