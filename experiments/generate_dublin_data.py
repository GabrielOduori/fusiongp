"""
Generate REALISTIC synthetic multi-source NO₂ data for Dublin study area.

Uses the actual grid locations and timestamps from test_data.csv to create
realistic observations where all sources measure the same underlying NO₂ field
with appropriate biases and noise.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

np.random.seed(42)

def generate_true_no2_field(df):
    """
    Generate a realistic spatio-temporal NO₂ field for Dublin.

    Components:
    - Spatial pattern (city center hotspot, coastal areas)
    - Temporal pattern (daily cycle, day-of-week, seasonal)
    - Random variation (weather effects)
    """

    print("Generating true NO₂ field...")

    # Get unique locations and times
    lats = df['latitude'].values
    lons = df['longitude'].values
    timestamps = pd.to_datetime(df['timestamp'])

    # Dublin city center (approximate)
    city_center_lat = 53.3498
    city_center_lon = -6.2603

    # Calculate distance from city center (in degrees, roughly)
    dist_from_center = np.sqrt((lats - city_center_lat)**2 + (lons - city_center_lon)**2)

    # Spatial component: Urban hotspot pattern
    # Higher NO₂ in city center, decreases with distance
    base_concentration = 25.0  # Base urban background
    hotspot_strength = 20.0
    spatial_pattern = base_concentration + hotspot_strength * np.exp(-dist_from_center**2 / 0.005)

    # Add some random spatial variation (local sources, neighborhoods)
    spatial_pattern += np.random.randn(len(lats)) * 3.0

    # Temporal modulation for each observation
    true_field = np.zeros(len(df))

    for i in tqdm(range(len(df)), desc="Computing temporal patterns"):
        ts = timestamps.iloc[i]
        hour = ts.hour
        day_of_week = ts.dayofweek
        month = ts.month

        # Daily cycle (higher during rush hours)
        if 7 <= hour <= 9 or 17 <= hour <= 19:  # Morning/evening rush
            temporal_factor = 1.5
        elif 22 <= hour or hour <= 5:  # Night
            temporal_factor = 0.6
        else:  # Daytime
            temporal_factor = 1.0

        # Weekend effect (lower on weekends)
        weekend_factor = 0.7 if day_of_week >= 5 else 1.0

        # Seasonal variation (higher in winter due to heating)
        if month in [12, 1, 2]:  # Winter
            seasonal_factor = 1.3
        elif month in [6, 7, 8]:  # Summer
            seasonal_factor = 0.8
        else:  # Spring/Fall
            seasonal_factor = 1.0

        # Combine all factors
        true_field[i] = spatial_pattern[i] * temporal_factor * weekend_factor * seasonal_factor

        # Add temporal noise (weather, wind, etc.)
        true_field[i] += np.random.randn() * 2.0

    # Ensure non-negative
    true_field = np.maximum(true_field, 0.1)

    return true_field


def add_multi_source_observations(df, true_no2):
    """
    Add realistic EPA, low-cost sensor, and satellite observations.

    Each source has:
    - Different coverage patterns
    - Different noise characteristics
    - Different biases (for low-cost sensors)
    """

    print("\nAdding multi-source observations...")

    n_obs = len(df)

    # Initialize columns
    epa_no2 = np.full(n_obs, np.nan)
    low_cost_data = np.full(n_obs, np.nan)
    satellite_values = np.full(n_obs, np.nan)

    # Get unique grid locations
    unique_grids = df['grid_id'].unique()
    n_grids = len(unique_grids)

    # Determine which grids have EPA monitors (sparse, ~5%)
    n_epa_sites = max(1, int(n_grids * 0.05))
    epa_grid_ids = np.random.choice(unique_grids, size=n_epa_sites, replace=False)
    epa_grids = set(epa_grid_ids)

    print(f"  EPA sites: {n_epa_sites} out of {n_grids} grids ({100*n_epa_sites/n_grids:.1f}%)")

    # Determine which grids have low-cost sensors (dense, ~60%)
    n_lc_sites = max(1, int(n_grids * 0.60))
    lc_grid_ids = np.random.choice(unique_grids, size=n_lc_sites, replace=False)
    lc_grids = set(lc_grid_ids)

    print(f"  Low-cost sensor sites: {n_lc_sites} out of {n_grids} grids ({100*n_lc_sites/n_grids:.1f}%)")

    # Generate observations
    for i in tqdm(range(n_obs), desc="Generating observations"):
        grid_id = df.iloc[i]['grid_id']
        true_value = true_no2[i]

        # EPA observations: Sparse, low noise, unbiased
        if grid_id in epa_grids:
            epa_noise = 1.5  # ±1.5 µg/m³
            epa_no2[i] = true_value + np.random.randn() * epa_noise

        # Low-cost sensors: Dense, higher noise, biased
        if grid_id in lc_grids:
            lc_noise = 4.0  # ±4.0 µg/m³
            # Realistic calibration error
            # Each sensor has slightly different bias
            grid_idx = np.where(unique_grids == grid_id)[0][0]
            np.random.seed(grid_idx)  # Consistent bias per grid
            lc_slope = 1.15 + np.random.randn() * 0.05  # ~1.15 ± 0.05
            lc_intercept = 3.0 + np.random.randn() * 1.0  # ~3.0 ± 1.0
            np.random.seed(None)  # Reset seed

            low_cost_data[i] = lc_slope * true_value + lc_intercept + np.random.randn() * lc_noise

        # Satellite observations: Moderate coverage (~40%, cloud gaps), moderate noise
        if np.random.rand() < 0.40:
            sat_noise = 3.0  # ±3.0 µg/m³
            satellite_values[i] = true_value + np.random.randn() * sat_noise

    # Add to dataframe
    df['true_no2'] = true_no2
    df['epa_no2'] = epa_no2
    df['low_cost_data'] = low_cost_data
    df['satellite_values'] = satellite_values

    # Print coverage statistics
    print(f"\nCoverage Statistics:")
    print(f"  EPA: {np.sum(~np.isnan(epa_no2)):,} / {n_obs:,} ({100*np.sum(~np.isnan(epa_no2))/n_obs:.1f}%)")
    print(f"  Low-cost: {np.sum(~np.isnan(low_cost_data)):,} / {n_obs:,} ({100*np.sum(~np.isnan(low_cost_data))/n_obs:.1f}%)")
    print(f"  Satellite: {np.sum(~np.isnan(satellite_values)):,} / {n_obs:,} ({100*np.sum(~np.isnan(satellite_values))/n_obs:.1f}%)")

    return df


def main():
    print("="*70)
    print("Generating Realistic Synthetic NO₂ Data for Dublin")
    print("="*70)

    # Load existing grid structure
    input_path = Path(__file__).parent.parent / "data" / "test_data.csv"

    print(f"\nLoading grid structure from: {input_path}")
    df_original = pd.read_csv(input_path)

    print(f"  Original data shape: {df_original.shape}")
    print(f"  Unique grids: {df_original['grid_id'].nunique()}")
    print(f"  Unique timestamps: {df_original['timestamp'].nunique()}")

    # Keep only the grid structure
    df = df_original[['grid_id', 'latitude', 'longitude', 'timestamp']].copy()

    print(f"\n  Total observations: {len(df):,}")
    print(f"  Lat range: [{df['latitude'].min():.4f}, {df['latitude'].max():.4f}]")
    print(f"  Lon range: [{df['longitude'].min():.4f}, {df['longitude'].max():.4f}]")
    print(f"  Time range: {df['timestamp'].min()} to {df['timestamp'].max()}")

    # Generate true NO₂ field
    true_no2 = generate_true_no2_field(df)

    print(f"\n  True NO₂ range: [{true_no2.min():.1f}, {true_no2.max():.1f}] µg/m³")
    print(f"  True NO₂ mean: {true_no2.mean():.1f} µg/m³")
    print(f"  True NO₂ std: {true_no2.std():.1f} µg/m³")

    # Add multi-source observations
    df = add_multi_source_observations(df, true_no2)

    # Save to CSV
    output_path = Path(__file__).parent.parent / "data" / "dublin_realistic_no2.csv"
    df.to_csv(output_path, index=False)

    print(f"\n{'='*70}")
    print(f"✓ Realistic Dublin NO₂ data saved to: {output_path}")
    print(f"  Total records: {len(df):,}")
    print(f"  File size: {output_path.stat().st_size / 1024 / 1024:.1f} MB")

    # Show sample
    print(f"\nSample data (first 10 rows):")
    print(df.head(10)[['grid_id', 'timestamp', 'true_no2', 'epa_no2', 'low_cost_data', 'satellite_values']])

    print(f"\n{'='*70}")


if __name__ == "__main__":
    main()
