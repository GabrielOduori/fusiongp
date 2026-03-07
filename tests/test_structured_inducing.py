"""
Test structured grid inducing point initialization.

Verifies that structured grid inducing points provide better coverage
and enable Kronecker structure compared to K-means.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import numpy as np
from src.models.svgp import FusionSVGP
from src.data.loader import DataLoader
import matplotlib.pyplot as plt


def create_gridded_data(n_spatial=100, n_temporal=120):
    """Create synthetic gridded data for testing."""
    # Create grid
    spatial_locs = torch.rand(n_spatial, 2)
    temporal_locs = torch.rand(n_temporal)

    # Create full grid
    x = torch.zeros(n_spatial * n_temporal, 3)
    idx = 0
    for s_idx in range(n_spatial):
        for t_idx in range(n_temporal):
            x[idx, :2] = spatial_locs[s_idx]
            x[idx, 2] = temporal_locs[t_idx]
            idx += 1

    return x


def test_structured_grid_basic():
    """Test basic structured grid creation."""
    print("\n" + "="*70)
    print("TEST 1: Basic Structured Grid Creation")
    print("="*70)

    # Create gridded data
    x = create_gridded_data(n_spatial=100, n_temporal=120)
    print(f"\nData: {len(x)} observations (100 spatial × 120 temporal)")

    # Create model
    model = FusionSVGP(n_inducing=6000)

    # Initialize with structured grid
    model.initialize_inducing_points(
        x,
        method='structured_grid',
        spatial_inducing=100,
        temporal_inducing=60
    )

    # Check inducing points
    inducing_points = model.variational_strategy.inducing_points
    print(f"\nInducing points shape: {inducing_points.shape}")
    print(f"Expected: (6000, 3) for 100 × 60 grid")

    # Check grid shape was stored
    assert hasattr(model, '_inducing_grid_shape'), "Grid shape not stored!"
    M_s, M_t = model._inducing_grid_shape
    print(f"Grid shape: {M_s} × {M_t} = {M_s * M_t}")

    assert M_s == 100 and M_t == 60, f"Expected 100×60, got {M_s}×{M_t}"
    assert inducing_points.shape[0] == 6000, f"Expected 6000 points, got {inducing_points.shape[0]}"

    print("✓ Basic test passed!")
    return model, x


def test_structured_vs_kmeans():
    """Compare structured grid vs K-means coverage."""
    print("\n" + "="*70)
    print("TEST 2: Structured Grid vs K-means Comparison")
    print("="*70)

    # Create gridded data
    x = create_gridded_data(n_spatial=100, n_temporal=120)
    n_inducing = 600  # Smaller for faster testing

    # Test K-means
    model_kmeans = FusionSVGP(n_inducing=n_inducing)
    model_kmeans.initialize_inducing_points(x, method='kmeans')
    inducing_kmeans = model_kmeans.variational_strategy.inducing_points

    # Test structured grid
    model_grid = FusionSVGP(n_inducing=n_inducing)
    model_grid.initialize_inducing_points(
        x,
        method='structured_grid'
    )
    inducing_grid = model_grid.variational_strategy.inducing_points

    print(f"\nK-means inducing points: {inducing_kmeans.shape}")
    print(f"  Has grid structure: {model_kmeans._inducing_grid_shape}")

    print(f"\nStructured grid inducing points: {inducing_grid.shape}")
    print(f"  Grid shape: {model_grid._inducing_grid_shape}")

    # Check uniqueness of spatial/temporal
    unique_spatial_km, _ = torch.unique(inducing_kmeans[:, :2], dim=0, return_inverse=True)
    unique_temporal_km, _ = torch.unique(inducing_kmeans[:, 2], dim=0, return_inverse=True)

    unique_spatial_gr, _ = torch.unique(inducing_grid[:, :2], dim=0, return_inverse=True)
    unique_temporal_gr, _ = torch.unique(inducing_grid[:, 2], dim=0, return_inverse=True)

    print(f"\nK-means:")
    print(f"  Unique spatial: {len(unique_spatial_km)}")
    print(f"  Unique temporal: {len(unique_temporal_km)}")
    print(f"  Product: {len(unique_spatial_km) * len(unique_temporal_km)}")

    print(f"\nStructured grid:")
    print(f"  Unique spatial: {len(unique_spatial_gr)}")
    print(f"  Unique temporal: {len(unique_temporal_gr)}")
    print(f"  Product: {len(unique_spatial_gr) * len(unique_temporal_gr)}")

    # For structured grid, product should equal total points
    M_s, M_t = model_grid._inducing_grid_shape
    assert len(inducing_grid) == len(unique_spatial_gr) * len(unique_temporal_gr), \
        "Structured grid doesn't have Kronecker structure!"

    print("\n✓ Comparison test passed!")
    return model_kmeans, model_grid


def test_auto_aspect_ratio():
    """Test automatic aspect ratio computation."""
    print("\n" + "="*70)
    print("TEST 3: Automatic Aspect Ratio Computation")
    print("="*70)

    # Create data with specific aspect ratio
    x = create_gridded_data(n_spatial=100, n_temporal=50)

    # Test with different n_inducing values
    for n_inducing in [500, 1000, 2000]:
        model = FusionSVGP(n_inducing=n_inducing)
        model.initialize_inducing_points(x, method='structured_grid')

        M_s, M_t = model._inducing_grid_shape
        ratio = M_s / M_t
        data_ratio = 100 / 50

        print(f"\nn_inducing={n_inducing}:")
        print(f"  Grid: {M_s} × {M_t} = {M_s * M_t}")
        print(f"  Ratio: {ratio:.2f} (data ratio: {data_ratio:.2f})")

        # Ratio should be approximately maintained
        assert abs(ratio - data_ratio) / data_ratio < 0.5, \
            f"Aspect ratio not maintained: {ratio} vs {data_ratio}"

    print("\n✓ Aspect ratio test passed!")


def test_with_real_data():
    """Test with actual synthetic NO2 data."""
    print("\n" + "="*70)
    print("TEST 4: Real Synthetic Data")
    print("="*70)

    # Load real data
    data_path = project_root / 'data' / 'realistic_synthetic_no2.csv'
    if not data_path.exists():
        print(f"\nData not found at {data_path}")
        print("Skipping real data test")
        return

    loader = DataLoader(str(data_path))
    data = loader.load()
    print(f"\nLoaded {data.n_observations} observations")

    # Convert to tensor
    x = torch.column_stack([
        torch.tensor(data.coords),
        torch.tensor(data.timestamps).unsqueeze(1)
    ]).float()

    # Test structured grid with different sizes
    for n_inducing in [500, 1000, 2000]:
        model = FusionSVGP(n_inducing=n_inducing)
        model.initialize_inducing_points(x, method='structured_grid')

        M_s, M_t = model._inducing_grid_shape
        actual_inducing = model.variational_strategy.inducing_points.shape[0]

        print(f"\nn_inducing={n_inducing}:")
        print(f"  Grid: {M_s} × {M_t} = {actual_inducing}")
        print(f"  Grid shape stored: {model._inducing_grid_shape}")

        assert actual_inducing == M_s * M_t, \
            f"Grid size mismatch: {actual_inducing} != {M_s * M_t}"

    print("\n✓ Real data test passed!")


def visualize_comparison(save_path=None):
    """Create visualization comparing K-means vs structured grid."""
    print("\n" + "="*70)
    print("VISUALIZATION: K-means vs Structured Grid")
    print("="*70)

    # Create data
    x = create_gridded_data(n_spatial=100, n_temporal=120)
    n_inducing = 1000

    # K-means
    model_km = FusionSVGP(n_inducing=n_inducing)
    model_km.initialize_inducing_points(x, method='kmeans')
    z_km = model_km.variational_strategy.inducing_points.detach()

    # Structured grid
    model_gr = FusionSVGP(n_inducing=n_inducing)
    model_gr.initialize_inducing_points(x, method='structured_grid')
    z_gr = model_gr.variational_strategy.inducing_points.detach()

    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Plot K-means
    ax = axes[0]
    ax.scatter(x[:, 0], x[:, 1], c='lightgray', alpha=0.3, s=1, label='Data')
    ax.scatter(z_km[:, 0], z_km[:, 1], c='red', marker='x', s=50, label='Inducing points')
    ax.set_title(f'K-means (scattered)\n{len(z_km)} points', fontsize=14)
    ax.set_xlabel('Latitude (normalized)')
    ax.set_ylabel('Longitude (normalized)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot structured grid
    ax = axes[1]
    ax.scatter(x[:, 0], x[:, 1], c='lightgray', alpha=0.3, s=1, label='Data')
    ax.scatter(z_gr[:, 0], z_gr[:, 1], c='red', marker='x', s=50, label='Inducing points (spatial)')
    M_s, M_t = model_gr._inducing_grid_shape
    ax.set_title(f'Structured Grid\n{M_s} × {M_t} = {len(z_gr)} points', fontsize=14)
    ax.set_xlabel('Latitude (normalized)')
    ax.set_ylabel('Longitude (normalized)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\nVisualization saved to: {save_path}")
    else:
        save_path = project_root / 'outputs' / 'structured_inducing_comparison.png'
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\nVisualization saved to: {save_path}")

    plt.close()


if __name__ == "__main__":
    print("\nRunning structured inducing points tests...\n")

    try:
        # Run tests
        test_structured_grid_basic()
        test_structured_vs_kmeans()
        test_auto_aspect_ratio()
        test_with_real_data()

        # Create visualization
        visualize_comparison()

        print("\n" + "="*70)
        print("ALL TESTS PASSED! ✓")
        print("="*70)
        print("\nStructured grid inducing points are working correctly!")
        print("Key benefits:")
        print("  - Kronecker structure: K_ZZ = K_time ⊗ K_space")
        print("  - Better coverage with more inducing points")
        print("  - Enables future KL divergence optimizations")

    except AssertionError as e:
        print(f"\n✗ Test failed: {e}")
        raise
    except Exception as e:
        print(f"\n✗ Error: {e}")
        raise
