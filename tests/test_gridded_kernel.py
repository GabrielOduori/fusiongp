"""
Unit tests for gridded kernel evaluation optimization.

Tests verify that the gridded kernel evaluation produces identical results
to the element-wise approach while being more efficient.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pytest
import torch
import numpy as np
from src.models.kernels import SpatioTemporalKernel


class TestGriddedKernel:
    """Test suite for gridded kernel evaluation."""

    def test_gridded_vs_elementwise_small(self):
        """Test that gridded and element-wise kernels produce identical results on small data."""
        # Create small gridded dataset
        n_spatial = 10
        n_temporal = 12

        # Create grid
        spatial_locs = torch.rand(n_spatial, 2)
        temporal_locs = torch.rand(n_temporal)

        # Create full grid (all combinations)
        x = torch.zeros(n_spatial * n_temporal, 3)
        idx = 0
        for s_idx in range(n_spatial):
            for t_idx in range(n_temporal):
                x[idx, :2] = spatial_locs[s_idx]
                x[idx, 2] = temporal_locs[t_idx]
                idx += 1

        # Create kernel
        kernel = SpatioTemporalKernel(
            spatial_kernel_type='matern32',
            temporal_kernel_type='matern32',
        )

        # Compute with gridded optimization
        K_gridded = kernel.forward(x, x, use_gridded=True)

        # Compute with element-wise (standard)
        K_elementwise = kernel.forward(x, x, use_gridded=False)

        # Convert to dense if lazy
        if hasattr(K_gridded, 'to_dense'):
            K_gridded = K_gridded.to_dense()
        if hasattr(K_elementwise, 'to_dense'):
            K_elementwise = K_elementwise.to_dense()

        # Check they're identical
        max_diff = (K_gridded - K_elementwise).abs().max().item()
        print(f"\nSmall data test:")
        print(f"  Grid size: {n_spatial} × {n_temporal} = {n_spatial * n_temporal}")
        print(f"  Max difference: {max_diff:.2e}")

        assert max_diff < 1e-4, f"Gridded and element-wise differ by {max_diff}"

    def test_gridded_vs_elementwise_realistic(self):
        """Test on realistic grid size similar to actual data."""
        # Create realistic gridded dataset (smaller than full 8695×53)
        n_spatial = 100
        n_temporal = 50

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

        # Create kernel
        kernel = SpatioTemporalKernel(
            spatial_kernel_type='matern32',
            temporal_kernel_type='matern32',
        )

        # Compute with gridded optimization
        K_gridded = kernel.forward(x, x, use_gridded=True)

        # Compute with element-wise (standard)
        K_elementwise = kernel.forward(x, x, use_gridded=False)

        # Convert to dense if lazy
        if hasattr(K_gridded, 'to_dense'):
            K_gridded = K_gridded.to_dense()
        if hasattr(K_elementwise, 'to_dense'):
            K_elementwise = K_elementwise.to_dense()

        # Check they're identical (allow small numerical error)
        max_diff = (K_gridded - K_elementwise).abs().max().item()
        mean_diff = (K_gridded - K_elementwise).abs().mean().item()

        print(f"\nRealistic data test:")
        print(f"  Grid size: {n_spatial} × {n_temporal} = {n_spatial * n_temporal}")
        print(f"  Max difference: {max_diff:.2e}")
        print(f"  Mean difference: {mean_diff:.2e}")

        # Tolerances: max 1e-4 (good enough for practical use), mean should be much smaller
        assert max_diff < 1e-4, f"Gridded and element-wise differ by {max_diff}"
        assert mean_diff < 1e-6, f"Mean difference too large: {mean_diff}"

    def test_sparse_grid_fallback(self):
        """Test that sparse grids fall back to element-wise."""
        # Create sparse "grid" (low coverage)
        n_spatial = 100
        n_temporal = 50

        # Only sample 20% of grid
        n_samples = int(0.2 * n_spatial * n_temporal)

        x = torch.rand(n_samples, 3)

        # Create kernel
        kernel = SpatioTemporalKernel(
            spatial_kernel_type='matern32',
            temporal_kernel_type='matern32',
        )

        # Both should work and produce same result (gridded falls back to elementwise)
        K_gridded = kernel.forward(x, x, use_gridded=True)
        K_elementwise = kernel.forward(x, x, use_gridded=False)

        # Convert to dense if lazy
        if hasattr(K_gridded, 'to_dense'):
            K_gridded = K_gridded.to_dense()
        if hasattr(K_elementwise, 'to_dense'):
            K_elementwise = K_elementwise.to_dense()

        max_diff = (K_gridded - K_elementwise).abs().max().item()

        print(f"\nSparse grid test (fallback):")
        print(f"  Samples: {n_samples} (20% coverage)")
        print(f"  Max difference: {max_diff:.2e}")

        assert max_diff < 1e-4

    def test_cross_covariance_uses_elementwise(self):
        """Test that cross-covariance (x1 != x2) uses element-wise."""
        # Create two different datasets
        n1 = 50
        n2 = 30

        x1 = torch.rand(n1, 3)
        x2 = torch.rand(n2, 3)

        kernel = SpatioTemporalKernel(
            spatial_kernel_type='matern32',
            temporal_kernel_type='matern32',
        )

        # Cross-covariance should use element-wise even with use_gridded=True
        K_cross = kernel.forward(x1, x2, use_gridded=True)

        # Convert to dense if lazy
        if hasattr(K_cross, 'to_dense'):
            K_cross = K_cross.to_dense()

        # Check shape
        assert K_cross.shape == (n1, n2), f"Expected shape ({n1}, {n2}), got {K_cross.shape}"

        print(f"\nCross-covariance test:")
        print(f"  x1 shape: {x1.shape}")
        print(f"  x2 shape: {x2.shape}")
        print(f"  K shape: {K_cross.shape}")

    def test_diagonal_mode(self):
        """Test diagonal mode still works."""
        n = 100
        x = torch.rand(n, 3)

        kernel = SpatioTemporalKernel(
            spatial_kernel_type='matern32',
            temporal_kernel_type='matern32',
        )

        # Diagonal with gridded
        K_diag_gridded = kernel.forward(x, x, diag=True, use_gridded=True)

        # Diagonal without gridded
        K_diag_elementwise = kernel.forward(x, x, diag=True, use_gridded=False)

        # Check they're identical
        max_diff = (K_diag_gridded - K_diag_elementwise).abs().max().item()

        print(f"\nDiagonal mode test:")
        print(f"  Shape: {K_diag_gridded.shape}")
        print(f"  Max difference: {max_diff:.2e}")

        assert max_diff < 1e-4


def test_performance_improvement():
    """Benchmark performance improvement of gridded vs element-wise."""
    import time

    # Create moderately sized grid
    n_spatial = 100
    n_temporal = 120

    # Create grid
    spatial_locs = torch.rand(n_spatial, 2)
    temporal_locs = torch.rand(n_temporal)

    x = torch.zeros(n_spatial * n_temporal, 3)
    idx = 0
    for s_idx in range(n_spatial):
        for t_idx in range(n_temporal):
            x[idx, :2] = spatial_locs[s_idx]
            x[idx, 2] = temporal_locs[t_idx]
            idx += 1

    print(f"\n{'='*70}")
    print("PERFORMANCE BENCHMARK")
    print(f"{'='*70}")
    print(f"\nGrid size: {n_spatial} × {n_temporal} = {len(x):,} observations")

    kernel = SpatioTemporalKernel(
        spatial_kernel_type='matern32',
        temporal_kernel_type='matern32',
    )

    # Warm up
    _ = kernel.forward(x, x, use_gridded=True)
    _ = kernel.forward(x, x, use_gridded=False)

    # Benchmark gridded
    n_trials = 5
    times_gridded = []
    for _ in range(n_trials):
        start = time.time()
        K = kernel.forward(x, x, use_gridded=True)
        if hasattr(K, 'to_dense'):
            K = K.to_dense()
        times_gridded.append(time.time() - start)

    # Benchmark element-wise
    times_elementwise = []
    for _ in range(n_trials):
        start = time.time()
        K = kernel.forward(x, x, use_gridded=False)
        if hasattr(K, 'to_dense'):
            K = K.to_dense()
        times_elementwise.append(time.time() - start)

    avg_gridded = np.mean(times_gridded)
    avg_elementwise = np.mean(times_elementwise)
    speedup = avg_elementwise / avg_gridded

    print(f"\n1. Gridded optimization:")
    print(f"   Average time: {avg_gridded*1000:.2f} ms")

    print(f"\n2. Element-wise (baseline):")
    print(f"   Average time: {avg_elementwise*1000:.2f} ms")

    print(f"\n3. Performance:")
    print(f"   Speedup: {speedup:.2f}×")

    # Memory estimation
    mem_gridded = (n_spatial**2 + n_temporal**2) * 4 / 1e6  # MB
    mem_elementwise = (n_spatial * n_temporal)**2 * 4 / 1e6  # MB
    mem_reduction = mem_elementwise / mem_gridded

    print(f"   Memory reduction: {mem_reduction:.2f}×")
    print(f"   (Gridded: {mem_gridded:.1f} MB, Element-wise: {mem_elementwise:.1f} MB)")

    print(f"\n{'='*70}\n")

    assert speedup > 1.0, f"Expected speedup, got {speedup:.2f}×"


if __name__ == "__main__":
    # Run tests
    print("Running gridded kernel tests...\n")

    test_suite = TestGriddedKernel()

    try:
        test_suite.test_gridded_vs_elementwise_small()
        print("✓ Small data test passed")

        test_suite.test_gridded_vs_elementwise_realistic()
        print("✓ Realistic data test passed")

        test_suite.test_sparse_grid_fallback()
        print("✓ Sparse grid fallback test passed")

        test_suite.test_cross_covariance_uses_elementwise()
        print("✓ Cross-covariance test passed")

        test_suite.test_diagonal_mode()
        print("✓ Diagonal mode test passed")

        test_performance_improvement()
        print("✓ Performance benchmark completed")

        print("\n" + "="*70)
        print("ALL TESTS PASSED! ✓")
        print("="*70)

    except AssertionError as e:
        print(f"\n✗ Test failed: {e}")
        raise
