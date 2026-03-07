"""
Test script to analyze grid structure and Kronecker product feasibility.

This script:
1. Loads your actual data
2. Analyzes the spatial/temporal grid structure
3. Verifies if Kronecker optimization is possible
4. Benchmarks kernel evaluation strategies
5. Generates diagnostic plots

Usage:
    python tests/test_grid_structure.py
    python tests/test_grid_structure.py --data-path /path/to/data.csv
"""

import argparse
import sys
from pathlib import Path
import time

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data.loader import DataLoader
from src.data.preprocessor import DataPreprocessor
from src.models.kernels import SpatioTemporalKernel


def analyze_grid_structure(coords, timestamps, verbose=True):
    """
    Analyze if data is on a regular spatio-temporal grid.

    Parameters
    ----------
    coords : torch.Tensor or np.ndarray
        Spatial coordinates, shape (N, 2)
    timestamps : torch.Tensor or np.ndarray
        Temporal values, shape (N,)
    verbose : bool
        Print detailed analysis

    Returns
    -------
    dict
        Analysis results with keys:
        - n_spatial: Number of unique spatial locations
        - n_temporal: Number of unique temporal points
        - n_obs: Total observations
        - expected_full_grid: N_spatial × N_temporal
        - coverage: Fraction of grid filled
        - is_gridded: Boolean indicating if structured
        - spatial_repeats: Distribution of repeats per location
        - temporal_repeats: Distribution of repeats per time
    """
    if isinstance(coords, np.ndarray):
        coords = torch.tensor(coords)
    if isinstance(timestamps, np.ndarray):
        timestamps = torch.tensor(timestamps)

    # Unique spatial locations (centroids)
    unique_spatial, spatial_inverse = torch.unique(coords, dim=0, return_inverse=True)
    n_spatial = len(unique_spatial)

    # Unique temporal points
    unique_temporal, temporal_inverse = torch.unique(timestamps, return_inverse=True)
    n_temporal = len(unique_temporal)

    # Total observations
    n_obs = len(coords)

    # Expected full grid
    expected_full_grid = n_spatial * n_temporal
    coverage = n_obs / expected_full_grid

    # Analyze spatial repeats (how many times each location appears)
    spatial_counts = torch.bincount(spatial_inverse)
    avg_spatial_reps = spatial_counts.float().mean().item()

    # Analyze temporal repeats (how many times each time appears)
    temporal_counts = torch.bincount(temporal_inverse)
    avg_temporal_reps = temporal_counts.float().mean().item()

    # Check if truly gridded (most locations appear ~same number of times)
    spatial_std = spatial_counts.float().std().item()
    temporal_std = temporal_counts.float().std().item()

    # Heuristic: gridded if coverage > 50% and low variance in repeats
    is_gridded = (coverage > 0.5) or (spatial_std / avg_spatial_reps < 0.3)

    results = {
        'n_spatial': n_spatial,
        'n_temporal': n_temporal,
        'n_obs': n_obs,
        'expected_full_grid': expected_full_grid,
        'coverage': coverage,
        'is_gridded': is_gridded,
        'avg_spatial_reps': avg_spatial_reps,
        'avg_temporal_reps': avg_temporal_reps,
        'spatial_std': spatial_std,
        'temporal_std': temporal_std,
        'unique_spatial': unique_spatial,
        'unique_temporal': unique_temporal,
        'spatial_inverse': spatial_inverse,
        'temporal_inverse': temporal_inverse,
    }

    if verbose:
        print("=" * 70)
        print("GRID STRUCTURE ANALYSIS")
        print("=" * 70)
        print(f"\nSpatial Structure:")
        print(f"  Unique spatial locations (centroids): {n_spatial:,}")
        print(f"  Average observations per centroid: {avg_spatial_reps:.1f}")
        print(f"  Std dev of repeats: {spatial_std:.2f}")
        print(f"  Min/Max repeats: {spatial_counts.min().item()}/{spatial_counts.max().item()}")

        print(f"\nTemporal Structure:")
        print(f"  Unique temporal points: {n_temporal:,}")
        print(f"  Average observations per time: {avg_temporal_reps:.1f}")
        print(f"  Std dev of repeats: {temporal_std:.2f}")
        print(f"  Min/Max repeats: {temporal_counts.min().item()}/{temporal_counts.max().item()}")

        print(f"\nGrid Properties:")
        print(f"  Potential grid size: {n_spatial:,} × {n_temporal:,} = {expected_full_grid:,}")
        print(f"  Actual observations: {n_obs:,}")
        print(f"  Grid coverage: {coverage:.1%}")
        print(f"  Is gridded: {'✓ YES' if is_gridded else '✗ NO (scattered)'}")

        print(f"\nKronecker Structure:")
        if is_gridded:
            print(f"  ✓ Can exploit Kronecker structure!")
            print(f"  ✓ K_space size: ({n_spatial}, {n_spatial})")
            print(f"  ✓ K_time size: ({n_temporal}, {n_temporal})")
            storage_reduction = (expected_full_grid**2) / (n_spatial**2 + n_temporal**2)
            print(f"  ✓ Storage reduction: {storage_reduction:.1f}×")
        else:
            print(f"  ⚠ Data is scattered, limited Kronecker benefits")

        print("=" * 70)

    return results


def benchmark_kernel_evaluation(coords, timestamps, n_trials=10):
    """
    Benchmark different kernel evaluation strategies.

    Compares:
    1. Element-wise (current implementation)
    2. Gridded with unique extraction (proposed)
    3. Full Kronecker (if applicable)

    Parameters
    ----------
    coords : torch.Tensor
        Spatial coordinates
    timestamps : torch.Tensor
        Temporal values
    n_trials : int
        Number of benchmark trials

    Returns
    -------
    dict
        Timing results
    """
    print("\n" + "=" * 70)
    print("KERNEL EVALUATION BENCHMARK")
    print("=" * 70)

    # Create kernel
    kernel = SpatioTemporalKernel(
        spatial_kernel_type='matern32',
        temporal_kernel_type='matern32',
    )

    # Combine coords + time
    x = torch.cat([coords, timestamps.unsqueeze(-1)], dim=-1).float()
    n = len(x)

    print(f"\nBenchmarking on {n:,} observations...")

    # Strategy 1: Element-wise (current)
    print("\n1. Element-wise multiplication (current implementation):")
    times_elementwise = []
    for _ in range(n_trials):
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        start = time.time()

        spatial_covar = kernel.spatial_kernel(x, x)
        temporal_covar = kernel.temporal_kernel(x, x)
        if hasattr(spatial_covar, 'to_dense'):
            spatial_covar = spatial_covar.to_dense()
        if hasattr(temporal_covar, 'to_dense'):
            temporal_covar = temporal_covar.to_dense()
        K_elementwise = spatial_covar * temporal_covar

        torch.cuda.synchronize() if torch.cuda.is_available() else None
        times_elementwise.append(time.time() - start)

    avg_elementwise = np.mean(times_elementwise)
    print(f"   Average time: {avg_elementwise*1000:.2f} ms")

    # Strategy 2: Gridded with unique extraction
    print("\n2. Gridded with unique extraction (proposed):")
    unique_spatial, spatial_inv = torch.unique(x[:, :2], dim=0, return_inverse=True)
    unique_temporal, temporal_inv = torch.unique(x[:, 2], dim=0, return_inverse=True)

    n_s = len(unique_spatial)
    n_t = len(unique_temporal)
    print(f"   Unique spatial: {n_s}, Unique temporal: {n_t}")

    # Create padded inputs for kernels (they expect full 3D inputs due to active_dims)
    unique_spatial_padded = torch.cat([unique_spatial, torch.zeros(n_s, 1)], dim=-1)
    unique_temporal_padded = torch.cat([torch.zeros(n_t, 2), unique_temporal.unsqueeze(-1)], dim=-1)

    times_gridded = []
    for _ in range(n_trials):
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        start = time.time()

        # Compute on unique values
        K_s = kernel.spatial_kernel(unique_spatial_padded, unique_spatial_padded)
        K_t = kernel.temporal_kernel(unique_temporal_padded, unique_temporal_padded)
        if hasattr(K_s, 'to_dense'):
            K_s = K_s.to_dense()
        if hasattr(K_t, 'to_dense'):
            K_t = K_t.to_dense()

        # Index to get full covariance
        K_gridded = K_s[spatial_inv][:, spatial_inv] * K_t[temporal_inv][:, temporal_inv]

        torch.cuda.synchronize() if torch.cuda.is_available() else None
        times_gridded.append(time.time() - start)

    avg_gridded = np.mean(times_gridded)
    speedup = avg_elementwise / avg_gridded
    print(f"   Average time: {avg_gridded*1000:.2f} ms")
    print(f"   Speedup: {speedup:.2f}×")

    # Verify results are the same
    diff = (K_elementwise - K_gridded).abs().max().item()
    print(f"   Max difference: {diff:.2e} (should be ~0)")

    # Memory comparison
    mem_elementwise = n**2 * 4 / 1e6  # 4 bytes per float32, convert to MB
    mem_gridded = (n_s**2 + n_t**2 + n) * 4 / 1e6
    print(f"\n3. Memory usage:")
    print(f"   Element-wise: {mem_elementwise:.1f} MB (full {n}×{n} matrices)")
    print(f"   Gridded: {mem_gridded:.1f} MB ({n_s}×{n_s} + {n_t}×{n_t} + indexing)")
    print(f"   Memory reduction: {mem_elementwise/mem_gridded:.2f}×")

    print("=" * 70)

    return {
        'n_obs': n,
        'n_spatial': n_s,
        'n_temporal': n_t,
        'time_elementwise_ms': avg_elementwise * 1000,
        'time_gridded_ms': avg_gridded * 1000,
        'speedup': speedup,
        'mem_elementwise_mb': mem_elementwise,
        'mem_gridded_mb': mem_gridded,
        'mem_reduction': mem_elementwise / mem_gridded,
    }


def visualize_grid_structure(coords, timestamps, grid_analysis, save_path=None):
    """
    Create visualizations of the grid structure.

    Parameters
    ----------
    coords : torch.Tensor
        Spatial coordinates
    timestamps : torch.Tensor
        Temporal values
    grid_analysis : dict
        Results from analyze_grid_structure
    save_path : str, optional
        Path to save figure
    """
    if isinstance(coords, torch.Tensor):
        coords = coords.numpy()
    if isinstance(timestamps, torch.Tensor):
        timestamps = timestamps.numpy()

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Grid Structure Analysis', fontsize=16, fontweight='bold')

    # 1. Spatial distribution
    ax = axes[0, 0]
    scatter = ax.scatter(coords[:, 1], coords[:, 0], alpha=0.3, s=10)
    unique_spatial = grid_analysis['unique_spatial'].numpy()
    ax.scatter(unique_spatial[:, 1], unique_spatial[:, 0],
              color='red', s=100, marker='x', linewidths=2,
              label='Unique centroids')
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_title(f'Spatial Distribution\n({grid_analysis["n_spatial"]} unique centroids)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2. Temporal distribution
    ax = axes[0, 1]
    unique_temporal = grid_analysis['unique_temporal'].numpy()
    ax.hist(timestamps, bins=50, alpha=0.6, label='All observations')
    ax.scatter(unique_temporal, np.zeros_like(unique_temporal),
              color='red', s=100, marker='|', linewidths=2,
              label='Unique times')
    ax.set_xlabel('Time (normalized)')
    ax.set_ylabel('Count')
    ax.set_title(f'Temporal Distribution\n({grid_analysis["n_temporal"]} unique times)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 3. Coverage heatmap
    ax = axes[0, 2]
    spatial_inv = grid_analysis['spatial_inverse'].numpy()
    temporal_inv = grid_analysis['temporal_inverse'].numpy()

    # Create sparse coverage matrix
    coverage_matrix = np.zeros((grid_analysis['n_spatial'], grid_analysis['n_temporal']))
    for s_idx, t_idx in zip(spatial_inv, temporal_inv):
        coverage_matrix[s_idx, t_idx] = 1

    im = ax.imshow(coverage_matrix, aspect='auto', cmap='RdYlGn',
                   interpolation='nearest', vmin=0, vmax=1)
    ax.set_xlabel('Time index')
    ax.set_ylabel('Spatial index')
    ax.set_title(f'Coverage Matrix\n({grid_analysis["coverage"]:.1%} filled)')
    plt.colorbar(im, ax=ax, label='Observed')

    # 4. Spatial repeats distribution
    ax = axes[1, 0]
    spatial_counts = torch.bincount(grid_analysis['spatial_inverse']).numpy()
    ax.hist(spatial_counts, bins=30, edgecolor='black', alpha=0.7)
    ax.axvline(spatial_counts.mean(), color='red', linestyle='--',
              label=f'Mean: {spatial_counts.mean():.1f}')
    ax.set_xlabel('Observations per centroid')
    ax.set_ylabel('Number of centroids')
    ax.set_title('Spatial Repeats Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 5. Temporal repeats distribution
    ax = axes[1, 1]
    temporal_counts = torch.bincount(grid_analysis['temporal_inverse']).numpy()
    ax.hist(temporal_counts, bins=30, edgecolor='black', alpha=0.7)
    ax.axvline(temporal_counts.mean(), color='red', linestyle='--',
              label=f'Mean: {temporal_counts.mean():.1f}')
    ax.set_xlabel('Observations per time point')
    ax.set_ylabel('Number of time points')
    ax.set_title('Temporal Repeats Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 6. Summary statistics
    ax = axes[1, 2]
    ax.axis('off')

    stats_text = f"""
    Grid Structure Summary
    {'='*30}

    Spatial:
      • Unique centroids: {grid_analysis['n_spatial']:,}
      • Avg obs/centroid: {grid_analysis['avg_spatial_reps']:.1f}
      • Std dev: {grid_analysis['spatial_std']:.2f}

    Temporal:
      • Unique times: {grid_analysis['n_temporal']:,}
      • Avg obs/time: {grid_analysis['avg_temporal_reps']:.1f}
      • Std dev: {grid_analysis['temporal_std']:.2f}

    Grid:
      • Potential size: {grid_analysis['expected_full_grid']:,}
      • Actual obs: {grid_analysis['n_obs']:,}
      • Coverage: {grid_analysis['coverage']:.1%}
      • Is gridded: {'✓ YES' if grid_analysis['is_gridded'] else '✗ NO'}

    Kronecker Benefits:
      {'✓ Can exploit structure!' if grid_analysis['is_gridded'] else '⚠ Limited benefits'}
    """

    ax.text(0.1, 0.5, stats_text, fontsize=10, family='monospace',
           verticalalignment='center', transform=ax.transAxes)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\nVisualization saved to: {save_path}")

    plt.show()


def analyze_inducing_point_strategies(coords, timestamps, grid_analysis):
    """
    Compare different inducing point initialization strategies.

    Parameters
    ----------
    coords : torch.Tensor
        Spatial coordinates
    timestamps : torch.Tensor
        Temporal values
    grid_analysis : dict
        Results from analyze_grid_structure
    """
    print("\n" + "=" * 70)
    print("INDUCING POINT STRATEGIES")
    print("=" * 70)

    from sklearn.cluster import KMeans

    # Strategy 1: K-means on all data (current)
    x = torch.cat([coords, timestamps.unsqueeze(-1)], dim=-1).numpy()
    n_inducing = 500

    print(f"\n1. K-means on all {len(x):,} observations (current):")
    start = time.time()
    kmeans_all = KMeans(n_clusters=n_inducing, random_state=42, n_init=10)
    kmeans_all.fit(x)
    time_kmeans_all = time.time() - start
    print(f"   Time: {time_kmeans_all:.2f} seconds")
    print(f"   Inducing points: {n_inducing}")

    # Strategy 2: Structured grid (spatial K-means × temporal uniform)
    if grid_analysis['is_gridded']:
        print(f"\n2. Structured grid (proposed):")

        unique_spatial = grid_analysis['unique_spatial'].numpy()
        unique_temporal = grid_analysis['unique_temporal'].numpy()

        # Spatial: K-means on unique centroids
        M_s = 100  # Spatial inducing points
        M_t = 60   # Temporal inducing points

        start = time.time()
        if M_s < len(unique_spatial):
            kmeans_spatial = KMeans(n_clusters=M_s, random_state=42, n_init=10)
            kmeans_spatial.fit(unique_spatial)
            spatial_inducing = kmeans_spatial.cluster_centers_
        else:
            spatial_inducing = unique_spatial
            M_s = len(unique_spatial)

        # Temporal: uniform sampling
        if M_t < len(unique_temporal):
            temporal_indices = np.linspace(0, len(unique_temporal)-1, M_t).astype(int)
            temporal_inducing = unique_temporal[temporal_indices]
        else:
            temporal_inducing = unique_temporal
            M_t = len(unique_temporal)

        # Cartesian product
        spatial_grid = np.repeat(spatial_inducing, M_t, axis=0)
        temporal_grid = np.tile(temporal_inducing, M_s)
        inducing_structured = np.column_stack([spatial_grid, temporal_grid[:, None]])

        time_structured = time.time() - start

        print(f"   Time: {time_structured:.2f} seconds")
        print(f"   Inducing points: {len(inducing_structured)} ({M_s} × {M_t})")
        print(f"   Speedup: {time_kmeans_all/time_structured:.2f}×")
        print(f"   ✓ Has Kronecker structure: K_ZZ = K_time ⊗ K_space")

        # Visualize comparison
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # K-means all
        ax = axes[0]
        ax.scatter(coords[:, 1], coords[:, 0], alpha=0.1, s=5, label='Data')
        ax.scatter(kmeans_all.cluster_centers_[:, 1],
                  kmeans_all.cluster_centers_[:, 0],
                  color='red', s=50, marker='x', label='Inducing points')
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        ax.set_title(f'K-means (scattered)\n{n_inducing} points')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Structured
        ax = axes[1]
        ax.scatter(coords[:, 1], coords[:, 0], alpha=0.1, s=5, label='Data')
        ax.scatter(inducing_structured[:, 1], inducing_structured[:, 0],
                  color='red', s=50, marker='x', label='Inducing points (spatial)')
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        ax.set_title(f'Structured Grid\n{M_s}×{M_t} = {len(inducing_structured)} points')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(project_root / 'outputs' / 'inducing_points_comparison.png',
                   dpi=150, bbox_inches='tight')
        print(f"\n   Visualization saved to: outputs/inducing_points_comparison.png")
        plt.show()
    else:
        print("\n2. Structured grid: Not applicable (data not gridded)")

    print("=" * 70)


def main(args):
    """Main test function."""
    print("\n" + "="*70)
    print("FUSIONGP GRID STRUCTURE TEST")
    print("="*70)

    # Create output directory
    output_dir = project_root / 'outputs'
    output_dir.mkdir(exist_ok=True)

    # Load data
    print(f"\nLoading data from: {args.data_path}")
    loader = DataLoader(args.data_path)
    data = loader.load()

    print(f"\nLoaded {data.n_observations:,} observations")
    print(f"Sources: {data.sources}")

    # Preprocess (normalize)
    preprocessor = DataPreprocessor(
        normalize_coords=True,
        normalize_time=True,
        normalize_targets=False
    )
    preprocessor.fit(data)
    data_normalized = preprocessor.transform(data)

    # Convert to tensors
    coords = torch.tensor(data_normalized.coords, dtype=torch.float32)
    timestamps = torch.tensor(data_normalized.timestamps, dtype=torch.float32)

    # 1. Analyze grid structure
    print("\n" + "="*70)
    print("STEP 1: ANALYZING GRID STRUCTURE")
    print("="*70)
    grid_analysis = analyze_grid_structure(coords, timestamps, verbose=True)

    # 2. Benchmark kernel evaluation (skip for large datasets to avoid OOM)
    n_obs = len(coords)
    skip_benchmark = args.skip_benchmark or (n_obs > 50000 and not args.benchmark)

    if skip_benchmark:
        print("\n" + "="*70)
        print("STEP 2: SKIPPING BENCHMARK (dataset too large)")
        print("="*70)
        print(f"\nDataset has {n_obs:,} observations.")
        print("Benchmarking would require ~{:.1f} GB of memory.".format(n_obs**2 * 4 / 1e9))
        print("Use --benchmark flag to force benchmarking (may cause OOM).")
        benchmark_results = None
    elif args.benchmark:
        print("\n" + "="*70)
        print("STEP 2: BENCHMARKING KERNEL EVALUATION")
        print("="*70)
        benchmark_results = benchmark_kernel_evaluation(coords, timestamps, n_trials=args.n_trials)

    # 3. Visualize grid structure
    if args.visualize:
        print("\n" + "="*70)
        print("STEP 3: VISUALIZING GRID STRUCTURE")
        print("="*70)
        save_path = output_dir / 'grid_structure_analysis.png'
        visualize_grid_structure(coords, timestamps, grid_analysis, save_path=save_path)

    # 4. Test inducing point strategies
    if args.test_inducing:
        print("\n" + "="*70)
        print("STEP 4: TESTING INDUCING POINT STRATEGIES")
        print("="*70)
        test_inducing_point_strategies(coords, timestamps, grid_analysis)

    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"\n✓ Data analyzed: {data.n_observations:,} observations")
    print(f"✓ Grid structure: {grid_analysis['n_spatial']} spatial × {grid_analysis['n_temporal']} temporal")
    print(f"✓ Coverage: {grid_analysis['coverage']:.1%}")
    print(f"✓ Gridded: {'YES - Can exploit Kronecker!' if grid_analysis['is_gridded'] else 'NO - Scattered data'}")

    if benchmark_results is not None:
        print(f"\n✓ Kernel evaluation speedup: {benchmark_results['speedup']:.2f}×")
        print(f"✓ Memory reduction: {benchmark_results['mem_reduction']:.2f}×")

    print("\n" + "="*70)
    print("Test complete! Check outputs/ directory for visualizations.")
    print("="*70 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test FusionGP grid structure')

    # Data path
    default_data = project_root / 'data' / 'realistic_synthetic_no2.csv'
    parser.add_argument('--data-path', type=str, default=str(default_data),
                       help='Path to data CSV file')

    # Test options
    parser.add_argument('--benchmark', action='store_true', default=False,
                       help='Run kernel evaluation benchmark')
    parser.add_argument('--skip-benchmark', action='store_true', default=False,
                       help='Skip kernel evaluation benchmark')
    parser.add_argument('--visualize', action='store_true', default=True,
                       help='Generate visualization plots')
    parser.add_argument('--test-inducing', action='store_true', default=False,
                       help='Test inducing point strategies')
    parser.add_argument('--n-trials', type=int, default=10,
                       help='Number of benchmark trials')

    args = parser.parse_args()

    # Run tests
    main(args)
