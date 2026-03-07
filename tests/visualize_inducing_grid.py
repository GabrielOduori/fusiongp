"""
Better visualization of structured inducing points showing the Kronecker structure.
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import numpy as np
import matplotlib.pyplot as plt
from src.models.svgp import FusionSVGP


def create_gridded_data(n_spatial=100, n_temporal=120):
    """Create synthetic gridded data."""
    spatial_locs = torch.rand(n_spatial, 2)
    temporal_locs = torch.rand(n_temporal)

    x = torch.zeros(n_spatial * n_temporal, 3)
    idx = 0
    for s_idx in range(n_spatial):
        for t_idx in range(n_temporal):
            x[idx, :2] = spatial_locs[s_idx]
            x[idx, 2] = temporal_locs[t_idx]
            idx += 1

    return x


def visualize_kronecker_structure():
    """Visualize the Kronecker structure of structured inducing points."""
    print("Creating visualization of Kronecker-structured inducing points...")

    # Create data
    x = create_gridded_data(n_spatial=100, n_temporal=120)

    # Create model with structured grid
    model = FusionSVGP(n_inducing=600)
    model.initialize_inducing_points(x, method='structured_grid')

    z = model.variational_strategy.inducing_points.detach()
    M_s, M_t = model._inducing_grid_shape

    print(f"\nStructured grid: {M_s} spatial × {M_t} temporal = {len(z)} total inducing points")

    # Extract unique spatial and temporal
    unique_spatial_z = torch.unique(z[:, :2], dim=0)
    unique_temporal_z = torch.unique(z[:, 2])

    print(f"Unique spatial inducing points: {len(unique_spatial_z)} (should be {M_s})")
    print(f"Unique temporal inducing points: {len(unique_temporal_z)} (should be {M_t})")
    print(f"Kronecker product: {len(unique_spatial_z)} × {len(unique_temporal_z)} = {len(unique_spatial_z) * len(unique_temporal_z)}")
    print(f"Total inducing points: {len(z)}")

    # Create comprehensive visualization
    fig = plt.figure(figsize=(18, 12))

    # 1. Spatial distribution (all inducing points projected to lat/lon)
    ax1 = plt.subplot(2, 3, 1)
    ax1.scatter(x[:, 0], x[:, 1], c='lightgray', alpha=0.3, s=1, label='Data')
    ax1.scatter(z[:, 0], z[:, 1], c='red', marker='x', s=30, alpha=0.5, label=f'Inducing ({len(z)} total)')
    ax1.set_title(f'All {len(z)} Inducing Points\nProjected to Spatial Plane', fontsize=12)
    ax1.set_xlabel('Latitude (normalized)')
    ax1.set_ylabel('Longitude (normalized)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. Unique spatial locations
    ax2 = plt.subplot(2, 3, 2)
    ax2.scatter(x[:, 0], x[:, 1], c='lightgray', alpha=0.3, s=1, label='Data')
    ax2.scatter(unique_spatial_z[:, 0], unique_spatial_z[:, 1],
                c='red', marker='x', s=80, linewidths=2, label=f'Unique spatial ({M_s})')
    ax2.set_title(f'{M_s} Unique Spatial Locations\n(each used {M_t} times)', fontsize=12)
    ax2.set_xlabel('Latitude (normalized)')
    ax2.set_ylabel('Longitude (normalized)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 3. Temporal distribution
    ax3 = plt.subplot(2, 3, 3)
    ax3.scatter(x[:, 2], np.zeros_like(x[:, 2]), c='lightgray', alpha=0.3, s=1, label='Data')
    ax3.scatter(unique_temporal_z, np.zeros_like(unique_temporal_z),
                c='red', marker='x', s=80, linewidths=2, label=f'Unique temporal ({M_t})')
    ax3.set_title(f'{M_t} Unique Temporal Points\n(each used {M_s} times)', fontsize=12)
    ax3.set_xlabel('Time (normalized)')
    ax3.set_yticks([])
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis='x')
    ax3.set_ylim(-0.1, 0.1)

    # 4. 3D view of inducing points
    ax4 = fig.add_subplot(2, 3, 4, projection='3d')
    ax4.scatter(x[:, 0], x[:, 1], x[:, 2], c='lightgray', alpha=0.1, s=0.5, label='Data')
    ax4.scatter(z[:, 0], z[:, 1], z[:, 2], c='red', marker='x', s=20, label='Inducing')
    ax4.set_xlabel('Latitude')
    ax4.set_ylabel('Longitude')
    ax4.set_zlabel('Time')
    ax4.set_title(f'3D Spatio-Temporal Grid\n{M_s} × {M_t} = {len(z)} points', fontsize=12)
    ax4.legend()

    # 5. Heatmap showing grid structure
    ax5 = plt.subplot(2, 3, 5)

    # Create matrix showing which (spatial, temporal) combinations exist
    grid_matrix = np.zeros((M_s, M_t))
    for i in range(len(z)):
        # Find which spatial index this point corresponds to
        s_idx = ((unique_spatial_z[:, :2] - z[i, :2].unsqueeze(0)).abs().sum(1) < 1e-6).nonzero()[0].item()
        # Find which temporal index
        t_idx = ((unique_temporal_z - z[i, 2]).abs() < 1e-6).nonzero()[0].item()
        grid_matrix[s_idx, t_idx] = 1

    im = ax5.imshow(grid_matrix, cmap='RdYlGn', aspect='auto', interpolation='nearest')
    ax5.set_title(f'Kronecker Structure Verification\n(All cells should be 1)', fontsize=12)
    ax5.set_xlabel(f'Temporal Index (0-{M_t-1})')
    ax5.set_ylabel(f'Spatial Index (0-{M_s-1})')
    plt.colorbar(im, ax=ax5, label='Point exists')

    # 6. Summary text
    ax6 = plt.subplot(2, 3, 6)
    ax6.axis('off')

    summary_text = f"""
    KRONECKER STRUCTURE SUMMARY
    ══════════════════════════════

    Grid Dimensions:
      • M_s (spatial): {M_s}
      • M_t (temporal): {M_t}
      • Total: M_s × M_t = {M_s * M_t}

    Inducing Points:
      • Total created: {len(z)}
      • Unique spatial: {len(unique_spatial_z)}
      • Unique temporal: {len(unique_temporal_z)}

    Verification:
      ✓ M_s × M_t = {len(z)} ✓
      ✓ {len(unique_spatial_z)} unique spatial locations
      ✓ {len(unique_temporal_z)} unique temporal points
      ✓ Kronecker structure: K_ZZ = K_time ⊗ K_space

    What This Means:
      • Each of {M_s} spatial locations appears
        exactly {M_t} times (once per time point)
      • Each of {M_t} temporal points appears
        exactly {M_s} times (once per spatial location)
      • This creates a FULL grid in 3D space

    Benefits:
      • Enables Kronecker KL divergence
      • O(M³) → O(M_s³ + M_t³) complexity
      • {M_s * M_t}³ → {M_s}³ + {M_t}³ operations
      • Speedup: {(M_s * M_t)**3 / (M_s**3 + M_t**3):.0f}×
    """

    ax6.text(0.1, 0.5, summary_text, fontsize=10, family='monospace',
             verticalalignment='center', transform=ax6.transAxes)

    plt.tight_layout()

    # Save
    output_path = project_root / 'outputs' / 'kronecker_structure_explained.png'
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Visualization saved to: {output_path}")

    plt.close()

    # Verify grid matrix is all ones
    if np.all(grid_matrix == 1):
        print("✓ VERIFIED: Perfect Kronecker structure - all (spatial, temporal) pairs present!")
    else:
        missing = np.sum(grid_matrix == 0)
        print(f"⚠ WARNING: {missing} missing pairs in grid")

    return z, M_s, M_t


if __name__ == "__main__":
    print("\n" + "="*70)
    print("KRONECKER STRUCTURE VISUALIZATION")
    print("="*70)

    z, M_s, M_t = visualize_kronecker_structure()

    print("\n" + "="*70)
    print("EXPLANATION")
    print("="*70)
    print(f"""
The structured grid creates {len(z)} inducing points from:
  • {M_s} unique spatial locations (lat, lon pairs)
  • {M_t} unique temporal points

These combine via Kronecker product to give:
  {M_s} × {M_t} = {len(z)} total 3D inducing points

When you view ONLY the spatial projection (lat, lon):
  - You see {M_s} unique spatial locations
  - But each location appears {M_t} times in the full 3D grid
  - This is CORRECT for Kronecker structure!

The visualization shows:
  1. All {len(z)} points projected to 2D (looks like {M_s} points with overlap)
  2. The {M_s} unique spatial locations clearly
  3. The {M_t} unique temporal points
  4. The full 3D grid structure
  5. Heatmap confirming all (s,t) pairs exist
  6. Summary of the Kronecker structure

Check: outputs/kronecker_structure_explained.png
    """)
