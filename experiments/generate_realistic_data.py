"""
Generate REALISTIC synthetic multi-source NO₂ data for FusionGP testing.

This creates data where:
1. All sources observe the same underlying NO₂ field
2. Each source has realistic bias and noise characteristics
3. Coverage patterns are realistic (sparse EPA, dense low-cost, patchy satellite)
"""

import numpy as np
import pandas as pd
from pathlib import Path

np.random.seed(42)

def generate_true_no2_field(lats, lons, times):
    """
    Generate a realistic spatio-temporal NO₂ field.

    Components:
    - Spatial pattern (urban hotspots, highways)
    - Temporal pattern (daily cycle, day-of-week)
    - Random variation
    """
    n_locs = len(lats)
    n_times = len(times)

    # Spatial component (static urban pattern)
    # Create 2-3 urban hotspots
    hotspot1 = np.exp(-((lats - 40.75)**2 + (lons - (-74.00))**2) / 0.001)  # Downtown
    hotspot2 = np.exp(-((lats - 40.73)**2 + (lons - (-74.02))**2) / 0.002)  # Midtown
    spatial_pattern = 15 + 20 * hotspot1 + 15 * hotspot2

    # Temporal component (varies over time)
    true_field = np.zeros((n_locs, n_times))

    for t in range(n_times):
        hour = times[t].hour
        day_of_week = times[t].dayofweek

        # Daily cycle (higher during rush hours)
        if 7 <= hour <= 9 or 17 <= hour <= 19:  # Rush hours
            temporal_factor = 1.4
        elif 22 <= hour or hour <= 5:  # Night
            temporal_factor = 0.6
        else:
            temporal_factor = 1.0

        # Weekend effect (lower on weekends)
        weekend_factor = 0.7 if day_of_week >= 5 else 1.0

        # Combine spatial and temporal
        true_field[:, t] = spatial_pattern * temporal_factor * weekend_factor

        # Add smooth random variation (weather effects, etc.)
        true_field[:, t] += np.random.randn(n_locs) * 2.0

    return true_field


def generate_realistic_multi_source_data(
    n_locations=100,
    n_days=30,
    hours_per_day=4,  # Sample 4 times per day
    lat_range=(40.70, 40.80),
    lon_range=(-74.05, -73.95)
):
    """
    Generate realistic multi-source NO₂ observations.

    Returns DataFrame with columns:
    - grid_id, latitude, longitude, timestamp
    - true_no2 (ground truth, not observed in practice)
    - epa_no2, low_cost_data, satellite_values
    """

    print("Generating realistic synthetic NO₂ data...")

    # Create spatial locations
    n_lat = int(np.sqrt(n_locations))
    n_lon = int(np.sqrt(n_locations))
    lats = np.linspace(lat_range[0], lat_range[1], n_lat)
    lons = np.linspace(lon_range[0], lon_range[1], n_lon)

    # Create grid of all location combinations
    loc_lats = np.repeat(lats, n_lon)
    loc_lons = np.tile(lons, n_lat)

    # Create timestamps (multiple times per day)
    timestamps = []
    for day in range(n_days):
        base_date = pd.Timestamp('2024-01-01') + pd.Timedelta(days=day)
        for hour_idx in range(hours_per_day):
            hour = [6, 12, 18, 22][hour_idx]  # 6am, noon, 6pm, 10pm
            timestamps.append(base_date + pd.Timedelta(hours=hour))

    timestamps = pd.DatetimeIndex(timestamps)
    n_times = len(timestamps)

    print(f"  Locations: {len(loc_lats)}")
    print(f"  Timestamps: {n_times}")
    print(f"  Total potential observations: {len(loc_lats) * n_times}")

    # Generate TRUE underlying NO₂ field
    true_no2 = generate_true_no2_field(loc_lats, loc_lons, timestamps)

    # Create observations with realistic characteristics
    records = []

    for loc_idx in range(len(loc_lats)):
        lat = loc_lats[loc_idx]
        lon = loc_lons[loc_idx]
        grid_id = loc_idx

        for t_idx, timestamp in enumerate(timestamps):
            true_value = true_no2[loc_idx, t_idx]

            # EPA observations: SPARSE, LOW NOISE, unbiased
            # Only ~5% of locations have EPA monitors
            # They report hourly (so present for all times if location has monitor)
            has_epa = (grid_id % 20 == 0)  # ~5% coverage
            if has_epa:
                epa_noise = 1.5  # Low noise (±1.5 µg/m³)
                epa_no2 = true_value + np.random.randn() * epa_noise
            else:
                epa_no2 = np.nan

            # Low-cost sensors: DENSE, HIGHER NOISE, BIASED
            # ~60% of locations have sensors
            # Characteristic bias: multiplicative + additive
            has_low_cost = (np.random.rand() < 0.6)
            if has_low_cost:
                lc_noise = 4.0  # Higher noise
                # Calibration error: y = slope * x + intercept + noise
                # Realistic values: slope ~1.2 (over-estimates), intercept ~3 µg/m³
                lc_slope = 1.15 + np.random.randn() * 0.05  # Slight variation per sensor
                lc_intercept = 3.0 + np.random.randn() * 1.0
                low_cost_data = lc_slope * true_value + lc_intercept + np.random.randn() * lc_noise
            else:
                low_cost_data = np.nan

            # Satellite observations: MODERATE COVERAGE, MODERATE NOISE
            # ~40% coverage (cloud gaps, QA filtering)
            # Moderate noise but unbiased
            has_satellite = (np.random.rand() < 0.4)
            if has_satellite:
                sat_noise = 3.0
                satellite_values = true_value + np.random.randn() * sat_noise
            else:
                satellite_values = np.nan

            records.append({
                'grid_id': grid_id,
                'latitude': lat,
                'longitude': lon,
                'timestamp': timestamp,
                'true_no2': true_value,  # Ground truth (not available in practice)
                'epa_no2': epa_no2,
                'low_cost_data': low_cost_data,
                'satellite_values': satellite_values,
            })

    df = pd.DataFrame(records)

    # Print coverage statistics
    print(f"\nCoverage Statistics:")
    for col in ['epa_no2', 'low_cost_data', 'satellite_values']:
        n_valid = df[col].notna().sum()
        print(f"  {col}: {n_valid:,} valid ({100*n_valid/len(df):.1f}%)")

    # Print some sample statistics
    print(f"\nTrue NO₂ range: [{df['true_no2'].min():.1f}, {df['true_no2'].max():.1f}] µg/m³")
    print(f"True NO₂ mean: {df['true_no2'].mean():.1f} µg/m³")

    return df


if __name__ == "__main__":
    # Generate realistic data
    df = generate_realistic_multi_source_data(
        n_locations=100,  # 10x10 grid
        n_days=30,
        hours_per_day=4,  # 4 observations per day
    )

    # Save to CSV
    output_path = Path(__file__).parent.parent / "data" / "realistic_synthetic_no2.csv"
    output_path.parent.mkdir(exist_ok=True)
    df.to_csv(output_path, index=False)

    print(f"\n✓ Realistic synthetic data saved to: {output_path}")
    print(f"  Total records: {len(df):,}")

    # Show sample
    print("\nSample data (first 10 rows):")
    print(df.head(10)[['grid_id', 'timestamp', 'true_no2', 'epa_no2', 'low_cost_data', 'satellite_values']])
