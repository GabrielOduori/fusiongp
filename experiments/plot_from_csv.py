"""
Simple script to plot grid predictions from CSV.

Usage:
    python plot_from_csv.py
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Load predictions
csv_path = Path(__file__).parent.parent / "outputs" / "grid_predictions.csv"

if not csv_path.exists():
    print(f"Error: {csv_path} not found!")
    print("Run save_grid_predictions.py first.")
    exit(1)

print(f"Loading predictions from {csv_path}...")
df = pd.read_csv(csv_path)

print(f"Loaded {len(df):,} predictions")
print(f"Unique grids: {df.grid_id.nunique():,}")
print(f"Unique timestamps: {df.timestamp.nunique()}")

# Get unique timestamps
timestamps = sorted(df.timestamp.unique())

print(f"\nCreating plots for {len(timestamps)} timestamps...")

# Create figure with subplots
fig, axes = plt.subplots(2, len(timestamps), figsize=(4*len(timestamps), 8))
if len(timestamps) == 1:
    axes = axes.reshape(2, 1)

for i, t in enumerate(timestamps):
    # Filter data for this timestamp
    data_t = df[df.timestamp == t]

    # Plot 1: Predictions (scatter - grid centroids)
    ax1 = axes[0, i]
    scatter1 = ax1.scatter(
        data_t.longitude,
        data_t.latitude,
        c=data_t.predicted_mean,
        s=5,  # Larger points to show grid cells
        marker='s',  # Square markers to represent grid cells
        cmap='RdYlBu_r',
        vmin=df.predicted_mean.min(),
        vmax=df.predicted_mean.max(),
        alpha=0.9,
        edgecolors='none'
    )
    ax1.set_xlabel('Longitude')
    ax1.set_ylabel('Latitude')
    ax1.set_title(f'Predictions at t={t:.1f} days\n(Grid Centroids)')
    ax1.set_aspect('equal')
    plt.colorbar(scatter1, ax=ax1, label='NO₂ (µg/m³)', fraction=0.046)

    # Plot 2: Uncertainty (scatter - grid centroids)
    ax2 = axes[1, i]
    scatter2 = ax2.scatter(
        data_t.longitude,
        data_t.latitude,
        c=data_t.predicted_std,
        s=5,  # Larger points
        marker='s',  # Square markers
        cmap='viridis',
        vmin=df.predicted_std.min(),
        vmax=df.predicted_std.max(),
        alpha=0.9,
        edgecolors='none'
    )
    ax2.set_xlabel('Longitude')
    ax2.set_ylabel('Latitude')
    ax2.set_title(f'Uncertainty at t={t:.1f} days\n(Grid Centroids)')
    ax2.set_aspect('equal')
    plt.colorbar(scatter2, ax=ax2, label='Std (µg/m³)', fraction=0.046)

plt.tight_layout()

# Save
output_path = Path(__file__).parent / "test_outputs" / "grid_predictions_plot.png"
output_path.parent.mkdir(exist_ok=True)
plt.savefig(output_path, dpi=200, bbox_inches='tight')

print(f"\n✓ Saved to: {output_path}")

# Also create individual high-res plots for each timestamp
print("\nCreating individual high-resolution plots...")
for t in timestamps:
    data_t = df[df.timestamp == t]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    # Predictions (grid centroids as squares)
    scatter1 = ax1.scatter(
        data_t.longitude,
        data_t.latitude,
        c=data_t.predicted_mean,
        s=10,  # Larger for visibility
        marker='s',  # Square markers for grid cells
        cmap='RdYlBu_r',
        alpha=0.95,
        edgecolors='none'
    )
    ax1.set_xlabel('Longitude', fontsize=12)
    ax1.set_ylabel('Latitude', fontsize=12)
    ax1.set_title(f'NO₂ Predictions at t={t:.1f} days (Grid Centroids)', fontsize=14, fontweight='bold')
    ax1.set_aspect('equal')
    ax1.grid(True, alpha=0.2)
    cbar1 = plt.colorbar(scatter1, ax=ax1, label='NO₂ (µg/m³)')
    cbar1.ax.tick_params(labelsize=10)

    # Uncertainty (grid centroids as squares)
    scatter2 = ax2.scatter(
        data_t.longitude,
        data_t.latitude,
        c=data_t.predicted_std,
        s=10,  # Larger for visibility
        marker='s',  # Square markers for grid cells
        cmap='viridis',
        alpha=0.95,
        edgecolors='none'
    )
    ax2.set_xlabel('Longitude', fontsize=12)
    ax2.set_ylabel('Latitude', fontsize=12)
    ax2.set_title(f'Uncertainty (Std) at t={t:.1f} days (Grid Centroids)', fontsize=14, fontweight='bold')
    ax2.set_aspect('equal')
    ax2.grid(True, alpha=0.2)
    cbar2 = plt.colorbar(scatter2, ax=ax2, label='Std (µg/m³)')
    cbar2.ax.tick_params(labelsize=10)

    plt.tight_layout()

    output_path_t = Path(__file__).parent / "test_outputs" / f"predictions_t{t:.1f}.png"
    plt.savefig(output_path_t, dpi=250, bbox_inches='tight')
    plt.close()

    print(f"  ✓ Saved: {output_path_t.name}")

print("\n" + "="*70)
print("DONE!")
print("="*70)
