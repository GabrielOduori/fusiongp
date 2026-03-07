"""
Performance profiling script to identify bottlenecks in FusionGP training.

This script helps diagnose why training takes 10 hours instead of 2 hours.
"""

import time
import torch
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models import FusionSVGP
from src.models.kernels import SpatioTemporalKernel

def profile_kernel_computation():
    """Profile the kernel computation with different grid sizes."""
    print("="*70)
    print("PROFILING KERNEL COMPUTATION")
    print("="*70)

    test_configs = [
        (100, 50, 5000, "Small grid"),
        (300, 100, 30000, "Medium grid"),
        (680, 100, 68000, "Large grid (close to your data)"),
    ]

    kernel = SpatioTemporalKernel(
        spatial_kernel_type='matern32',
        temporal_kernel_type='matern32',
        spatial_ard=True
    )

    for n_spatial, n_temporal, n_total, label in test_configs:
        # Create test data (regular grid)
        spatial_locs = np.random.rand(n_spatial, 2).astype(np.float32)
        temporal_locs = np.random.rand(n_temporal).astype(np.float32)

        x_grid = []
        for t in temporal_locs:
            for s in spatial_locs:
                x_grid.append([s[0], s[1], t])
        x = torch.from_numpy(np.array(x_grid[:n_total], dtype=np.float32))

        # Warmup
        _ = kernel(x, x)

        # Profile gridded kernel
        start = time.perf_counter()
        K_gridded = kernel.forward(x, x, use_gridded=True)
        gridded_time = time.perf_counter() - start

        # Profile element-wise kernel
        start = time.perf_counter()
        K_elementwise = kernel.forward(x, x, use_gridded=False)
        elementwise_time = time.perf_counter() - start

        print(f"\n{label} ({n_spatial}×{n_temporal} = {n_total} obs):")
        print(f"  Gridded:     {gridded_time:.3f}s")
        print(f"  Element-wise: {elementwise_time:.3f}s")
        print(f"  Speedup:     {elementwise_time/gridded_time:.2f}x")
        print(f"  Memory:      {n_total**2 * 4 / 1e9:.2f} GB (full) vs {(n_spatial**2 + n_temporal**2) * 4 / 1e6:.2f} MB (gridded)")


def profile_inducing_points():
    """Profile impact of number of inducing points."""
    print("\n" + "="*70)
    print("PROFILING INDUCING POINTS")
    print("="*70)

    n_data = 10000  # Fixed data size
    x_train = torch.randn(n_data, 3)
    y_train = torch.randn(n_data)
    source_mask = torch.zeros(n_data, 3)
    source_mask[:, 0] = 1  # All EPA

    inducing_configs = [200, 400, 800, 1200]

    for n_inducing in inducing_configs:
        print(f"\n{n_inducing} inducing points:")

        model = FusionSVGP(n_inducing=n_inducing, kernel_type='matern32')
        model.initialize_inducing_points(x_train)

        # Profile single ELBO computation
        start = time.perf_counter()
        loss = -model(x_train).log_prob(y_train).sum()
        loss_time = time.perf_counter() - start

        # Profile gradient computation
        start = time.perf_counter()
        loss.backward()
        grad_time = time.perf_counter() - start

        total_time = loss_time + grad_time

        print(f"  Forward:  {loss_time:.3f}s")
        print(f"  Backward: {grad_time:.3f}s")
        print(f"  Total:    {total_time:.3f}s")
        print(f"  Per epoch (300 epochs): {total_time * 300:.1f}s = {total_time * 300 / 60:.1f} min")


def estimate_full_training_time():
    """Estimate training time for the full dataset."""
    print("\n" + "="*70)
    print("ESTIMATING FULL TRAINING TIME")
    print("="*70)

    # Your configuration
    n_data = 468900  # Your actual data size
    n_inducing = 800
    n_epochs = 300
    batch_size = 1024

    print(f"\nConfiguration:")
    print(f"  Data points: {n_data:,}")
    print(f"  Inducing points: {n_inducing}")
    print(f"  Epochs: {n_epochs}")
    print(f"  Batch size: {batch_size}")

    # Estimate based on smaller sample
    sample_size = min(10000, n_data)
    x_sample = torch.randn(sample_size, 3)
    y_sample = torch.randn(sample_size)

    model = FusionSVGP(n_inducing=n_inducing, kernel_type='matern32')
    model.initialize_inducing_points(x_sample)

    # Time one batch
    print(f"\nProfiling batch of {batch_size} samples...")
    x_batch = x_sample[:batch_size]
    y_batch = y_sample[:batch_size]

    start = time.perf_counter()
    loss = -model(x_batch).log_prob(y_batch).sum()
    loss.backward()
    batch_time = time.perf_counter() - start

    # Estimate full training time
    batches_per_epoch = np.ceil(n_data / batch_size)
    time_per_epoch = batch_time * batches_per_epoch
    total_time = time_per_epoch * n_epochs

    print(f"\nEstimates:")
    print(f"  Time per batch: {batch_time:.3f}s")
    print(f"  Batches per epoch: {batches_per_epoch:.0f}")
    print(f"  Time per epoch: {time_per_epoch:.1f}s = {time_per_epoch/60:.1f} min")
    print(f"  Total training time: {total_time:.0f}s = {total_time/3600:.1f} hours")

    # Check if this matches your experience
    if total_time / 3600 > 8:
        print(f"\n⚠️  WARNING: Estimated {total_time/3600:.1f} hours matches your 10-hour runtime!")
        print(f"   This suggests the performance regression is due to:")
        print(f"   1. Very large dataset (468K observations)")
        print(f"   2. High number of inducing points (800)")
        print(f"   3. Many epochs (300)")
    else:
        print(f"\n✓ Estimated time seems reasonable")


def check_gridded_optimization():
    """Check if gridded optimization is actually being used."""
    print("\n" + "="*70)
    print("CHECKING GRIDDED OPTIMIZATION")
    print("="*70)

    # Create a grid
    n_spatial, n_temporal = 100, 50
    spatial_locs = np.random.rand(n_spatial, 2).astype(np.float32)
    temporal_locs = np.random.rand(n_temporal).astype(np.float32)

    x_grid = []
    for t in temporal_locs:
        for s in spatial_locs:
            x_grid.append([s[0], s[1], t])
    x = torch.from_numpy(np.array(x_grid, dtype=np.float32))

    kernel = SpatioTemporalKernel()

    # Check if it detects grid structure
    spatial_coords = x[:, :2]
    temporal_coords = x[:, 2]

    unique_spatial, spatial_inv = torch.unique(
        spatial_coords, dim=0, return_inverse=True
    )
    unique_temporal, temporal_inv = torch.unique(
        temporal_coords, dim=0, return_inverse=True
    )

    n_s = len(unique_spatial)
    n_t = len(unique_temporal)
    expected_grid_size = n_s * n_t
    actual_size = len(x)
    coverage = actual_size / expected_grid_size if expected_grid_size > 0 else 0

    print(f"\nGrid structure detection:")
    print(f"  Unique spatial: {n_s}")
    print(f"  Unique temporal: {n_t}")
    print(f"  Expected grid size: {expected_grid_size}")
    print(f"  Actual size: {actual_size}")
    print(f"  Coverage: {coverage:.2%}")
    print(f"  Will use gridded: {'YES' if coverage >= 0.3 else 'NO'}")

    # Time both methods
    start = time.perf_counter()
    K1 = kernel.forward(x, x, use_gridded=True)
    gridded_time = time.perf_counter() - start

    start = time.perf_counter()
    K2 = kernel.forward(x, x, use_gridded=False)
    elementwise_time = time.perf_counter() - start

    print(f"\nPerformance:")
    print(f"  Gridded: {gridded_time:.3f}s")
    print(f"  Element-wise: {elementwise_time:.3f}s")
    print(f"  Speedup: {elementwise_time/gridded_time:.2f}x")


def main():
    print("\n")
    print("█" * 70)
    print("  FusionGP Performance Profiling")
    print("█" * 70)
    print("\nThis script profiles your codebase to identify performance bottlenecks.")
    print("Running on CPU - this may take a few minutes...\n")

    # Run profiling tests
    profile_kernel_computation()
    profile_inducing_points()
    estimate_full_training_time()
    check_gridded_optimization()

    print("\n" + "="*70)
    print("PROFILING COMPLETE")
    print("="*70)
    print("\nRecommendations will be based on the results above.")


if __name__ == "__main__":
    main()
