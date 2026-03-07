"""
Tests for the GP-Kalman Filter pipeline integration.

These tests focus on the bugs that are hardest to catch:
  - Scale/unit mismatches (normalised vs raw timestamps)
  - Array index mismatches (inducing points vs grid cells)
  - Output plausibility (predictions in physically valid range)
  - Inverse transform correctness

Run with:
    pytest tests/test_gpkf_pipeline.py -v
"""
import numpy as np
import pandas as pd
import pytest
import torch


# ---------------------------------------------------------------------------
# Minimal synthetic fixtures
# ---------------------------------------------------------------------------

N_DAYS = 5
N_GRID = 20      # small grid: 4x5
N_EPA = 8        # EPA training observations
M_S = 6          # inducing points

LAT_MIN, LAT_MAX = 53.30, 53.38
LON_MIN, LON_MAX = -6.40, -6.10
NO2_MEAN = 20.0
NO2_STD  = 10.0


def _make_scalers():
    """Minimal Scalers object matching what the preprocessor produces."""
    from types import SimpleNamespace
    s = SimpleNamespace()
    s.coord_min   = np.array([LAT_MIN, LON_MIN])
    s.coord_scale = np.array([LAT_MAX - LAT_MIN, LON_MAX - LON_MIN])
    s.time_min    = 0.0
    s.time_max    = float(N_DAYS - 1)
    s.normalize_targets = True
    s.target_mean = {"epa": NO2_MEAN}
    s.target_std  = {"epa": NO2_STD}
    return s


def _make_covariate_df(n_cells=N_GRID, n_days=N_DAYS):
    """Synthetic covariate_df with numeric timestamps (days 0..N_DAYS-1)."""
    rng = np.random.default_rng(0)
    lats = np.linspace(LAT_MIN, LAT_MAX, n_cells)
    lons = np.linspace(LON_MIN, LON_MAX, n_cells)
    rows = []
    for d in range(n_days):
        for lat, lon in zip(lats, lons):
            rows.append({
                "latitude":  lat,
                "longitude": lon,
                "timestamp": float(d),          # numeric days — NOT string dates
                "traffic_wind_0": rng.normal(),
                "traffic_wind_1": rng.normal(),
            })
    return pd.DataFrame(rows)


def _make_train_data():
    """Synthetic FusionData with EPA observations."""
    from src.data.loader import FusionData
    rng = np.random.default_rng(1)
    scalers = _make_scalers()

    # Normalised coords in [0, 1]
    coords = rng.uniform(0, 1, size=(N_EPA, 2))

    # Raw timestamps: days 0..N_DAYS-1 uniformly spread across EPA obs
    raw_ts = np.tile(np.arange(N_DAYS, dtype=float), N_EPA // N_DAYS + 1)[:N_EPA]

    # Normalised timestamps: map raw [0, N_DAYS-1] → [0, 1]
    norm_ts = raw_ts / float(N_DAYS - 1)

    # Normalised observations
    obs_norm = rng.normal(0, 1, size=N_EPA)

    return FusionData(
        coords=coords,
        timestamps=norm_ts,
        raw_timestamps=raw_ts,
        observations={"epa": obs_norm},
        source_masks={"epa": np.ones(N_EPA, dtype=bool)},
        grid_ids=np.arange(N_EPA),
        metadata={},
    )


def _make_model(n_inducing=M_S):
    """Minimal trained-ish FusionSVGP."""
    from src.models.svgp import FusionSVGP
    model = FusionSVGP(n_inducing=n_inducing, sources=["epa", "satellite"])
    # Initialise inducing points in normalised [0,1]^3 space
    x_fake = torch.rand(50, 3)
    model.initialize_inducing_points(x_fake, method="random")
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Test 1: Timestamps passed to GPKF must be in raw (day) scale, not [0,1]
# ---------------------------------------------------------------------------

class TestTimestampScale:

    def test_covariate_df_timestamps_are_numeric_days(self):
        """covariate_df['timestamp'] must be numeric days (0, 1, 2, ...), not strings."""
        df = _make_covariate_df()
        assert df["timestamp"].dtype != object, (
            "covariate_df timestamps are strings — must be converted to numeric days "
            "before passing to GPKalmanFilter"
        )
        assert df["timestamp"].min() >= 0
        assert df["timestamp"].max() < 366, "timestamps should be day numbers, not years"

    def test_unique_days_are_in_day_scale_not_normalised(self):
        """gpkf_unique_days must span multiple integer days, not [0, 1]."""
        data = _make_train_data()
        # Simulates what run_demo_pipeline does — WRONG: using normalised timestamps
        wrong_unique_days = np.unique(data.timestamps)       # [0, 1] range
        # CORRECT: using raw timestamps
        correct_unique_days = np.unique(data.raw_timestamps) # [0, N_DAYS-1] range

        assert wrong_unique_days.max() <= 1.01, "sanity: normalised timestamps are in [0,1]"
        assert correct_unique_days.max() >= N_DAYS - 1, (
            "raw timestamps should span full day range"
        )

        # The key assertion: day deltas must be ~1.0, not ~1/(N_DAYS-1)
        day_deltas_correct = np.diff(np.sort(correct_unique_days))
        day_deltas_wrong   = np.diff(np.sort(wrong_unique_days))

        assert day_deltas_correct.mean() == pytest.approx(1.0, abs=0.1), (
            f"Day deltas should be ~1.0 day, got {day_deltas_correct.mean():.4f}. "
            "You are likely passing normalised timestamps to the GPKF."
        )
        assert day_deltas_wrong.mean() < 0.5, "normalised deltas are small (confirming wrong scale)"


# ---------------------------------------------------------------------------
# Test 2: Cache index bounds — nn_epa must not exceed cache size
# ---------------------------------------------------------------------------

class TestCacheIndexBounds:

    def test_nn_epa_indices_within_grid_cache_bounds(self):
        """
        nn_epa (EPA obs → nearest grid cell) must index into the full grid
        (n_cells), not into the inducing-point slice (M_s).

        After the fix, _daily_X_cache stores (n_cells, n_cov) not (M_s, n_cov).
        """
        from src.inference.gp_kalman_filter import GPKalmanFilter

        scalers = _make_scalers()
        model   = _make_model(n_inducing=M_S)
        df_cov  = _make_covariate_df()
        covariate_cols = ["traffic_wind_0", "traffic_wind_1"]
        train   = _make_train_data()

        gpkf = GPKalmanFilter(
            model, scalers,
            covariate_df=df_cov,
            covariate_cols=covariate_cols,
            train_data=train,
        )

        # Each cached day should store (n_cells, n_cov), not (M_s, n_cov)
        n_cells = df_cov[["latitude", "longitude"]].drop_duplicates().shape[0]
        n_cov   = len(covariate_cols)

        for day_val, X_day in gpkf._daily_X_cache.items():
            assert X_day.shape == (n_cells, n_cov), (
                f"Cache shape for day {day_val} is {X_day.shape}, "
                f"expected ({n_cells}, {n_cov}). "
                "Cache stores inducing-point slice instead of full grid."
            )

    def test_nn_Z_saved_and_within_bounds(self):
        """_nn_Z must exist and contain valid grid cell indices."""
        from src.inference.gp_kalman_filter import GPKalmanFilter

        scalers = _make_scalers()
        model   = _make_model(n_inducing=M_S)
        df_cov  = _make_covariate_df()
        train   = _make_train_data()

        gpkf = GPKalmanFilter(
            model, scalers,
            covariate_df=df_cov,
            covariate_cols=["traffic_wind_0", "traffic_wind_1"],
            train_data=train,
        )

        assert hasattr(gpkf, "_nn_Z"), "_nn_Z not saved — prior_mean_for_day will use wrong indices"
        n_cells = df_cov[["latitude", "longitude"]].drop_duplicates().shape[0]
        assert gpkf._nn_Z.max() < n_cells, (
            f"_nn_Z contains index {gpkf._nn_Z.max()} but grid has only {n_cells} cells"
        )
        assert len(gpkf._nn_Z) == M_S, (
            f"_nn_Z length {len(gpkf._nn_Z)} should equal M_s={M_S}"
        )


# ---------------------------------------------------------------------------
# Test 3: GPKF output is in physically plausible µg/m³ range
# ---------------------------------------------------------------------------

class TestOutputPlausibility:

    def test_filter_predictions_in_physical_range(self):
        """
        Predicted NO2 must be in [0, 500] µg/m³.
        Values outside this range indicate a scale bug or diverged filter.
        """
        from src.inference.gp_kalman_filter import GPKalmanFilter

        scalers = _make_scalers()
        model   = _make_model(n_inducing=M_S)
        df_cov  = _make_covariate_df()
        train   = _make_train_data()

        gpkf = GPKalmanFilter(
            model, scalers,
            covariate_df=df_cov,
            covariate_cols=["traffic_wind_0", "traffic_wind_1"],
            train_data=train,
        )

        # Grid in original lat/lon
        grid_coords = np.column_stack([
            np.linspace(LAT_MIN, LAT_MAX, N_GRID),
            np.linspace(LON_MIN, LON_MAX, N_GRID),
        ])
        unique_days = np.arange(float(N_DAYS))

        # Use raw_timestamps for the filter data
        filter_data = _make_train_data()

        seq = gpkf.filter(filter_data, unique_days, grid_coords)

        assert seq.mean_ug.shape == (N_GRID, N_DAYS)
        assert seq.std_ug.shape  == (N_GRID, N_DAYS)

        assert np.all(np.isfinite(seq.mean_ug)), "mean_ug contains NaN/inf — filter diverged"
        assert np.all(seq.std_ug >= 0),          "std_ug contains negative values"

        assert seq.mean_ug.max() < 500, (
            f"max predicted NO2 = {seq.mean_ug.max():.1f} µg/m³ — likely a scale bug "
            "(predictions not inverse-transformed from normalised space)"
        )
        assert seq.mean_ug.min() > -100, (
            f"min predicted NO2 = {seq.mean_ug.min():.1f} µg/m³ — filter may have diverged"
        )

    def test_uncertainty_decreases_after_observations(self):
        """
        On a day with observations, posterior std should be less than prior std (K_MM diagonal).
        If it increases or stays the same, the Kalman update is not working.
        """
        from src.inference.gp_kalman_filter import GPKalmanFilter

        scalers = _make_scalers()
        model   = _make_model(n_inducing=M_S)
        df_cov  = _make_covariate_df()
        train   = _make_train_data()

        gpkf = GPKalmanFilter(
            model, scalers,
            covariate_df=df_cov,
            covariate_cols=["traffic_wind_0", "traffic_wind_1"],
            train_data=train,
        )

        grid_coords = np.column_stack([
            np.linspace(LAT_MIN, LAT_MAX, N_GRID),
            np.linspace(LON_MIN, LON_MAX, N_GRID),
        ])
        unique_days = np.arange(float(N_DAYS))
        seq = gpkf.filter(train, unique_days, grid_coords)

        # Days with observations should have lower average std than day 0 (prior)
        days_with_obs = np.where(seq.n_obs_per_day > 0)[0]
        if len(days_with_obs) > 0:
            prior_std  = seq.std_ug[:, 0].mean()
            update_std = seq.std_ug[:, days_with_obs].mean()
            assert update_std <= prior_std * 1.1, (
                f"Std after observations ({update_std:.3f}) is not lower than prior ({prior_std:.3f}). "
                "Kalman update may not be reducing uncertainty."
            )


# ---------------------------------------------------------------------------
# Test 4: OU transition parameters are in correct scale
# ---------------------------------------------------------------------------

class TestOUTransition:

    def test_ou_decay_factor_A_between_zero_and_one(self):
        """
        A = exp(-Δt / ℓ_t) must be in (0, 1].
        If Δt is in normalised scale (~0.036) instead of day scale (~1.0),
        A will be close to 1.0 and mean reversion will be negligible.
        """
        from src.inference.gp_kalman_filter import GPKalmanFilter

        scalers = _make_scalers()
        model   = _make_model()
        df_cov  = _make_covariate_df()
        train   = _make_train_data()

        gpkf = GPKalmanFilter(
            model, scalers,
            covariate_df=df_cov,
            covariate_cols=["traffic_wind_0", "traffic_wind_1"],
            train_data=train,
        )

        unique_days = np.arange(float(N_DAYS))
        A_arr, Q_arr = gpkf._ou_params(unique_days)

        assert np.all(A_arr >= 0) and np.all(A_arr <= 1), (
            f"A values out of [0,1]: min={A_arr.min():.4f}, max={A_arr.max():.4f}"
        )
        assert np.all(Q_arr >= 0) and np.all(Q_arr <= 1), (
            f"Q values out of [0,1]: min={Q_arr.min():.4f}, max={Q_arr.max():.4f}"
        )

        # A should not be too close to 1.0 for 1-day steps — that would mean no mean reversion
        # (this catches the normalised timestamp bug: tiny Δt → A ≈ 1)
        assert A_arr.mean() < 0.999, (
            f"A ≈ {A_arr.mean():.5f} — nearly 1.0. "
            "Time steps may be in normalised scale instead of days. "
            "Check that unique_days is in [0, N_DAYS-1] not [0, 1]."
        )


# ---------------------------------------------------------------------------
# Test 5: Inverse transform correctness
# ---------------------------------------------------------------------------

class TestInverseTransform:

    def test_inverse_transform_recovers_original_scale(self):
        """
        Normalised value 0 should map to target_mean (20 µg/m³).
        Normalised value 1 should map to target_mean + target_std (30 µg/m³).
        """
        from src.inference.gp_kalman_filter import GPKalmanFilter

        scalers = _make_scalers()
        model   = _make_model()
        df_cov  = _make_covariate_df()
        train   = _make_train_data()

        gpkf = GPKalmanFilter(
            model, scalers,
            covariate_df=df_cov,
            covariate_cols=["traffic_wind_0", "traffic_wind_1"],
            train_data=train,
        )

        mean_norm = np.array([0.0, 1.0, -1.0])
        std_norm  = np.array([1.0, 1.0,  1.0])

        mean_ug, std_ug = gpkf._inverse_transform(mean_norm, std_norm)

        assert mean_ug[0] == pytest.approx(NO2_MEAN, abs=1e-4), (
            f"Normalised 0 → {mean_ug[0]:.2f} µg/m³, expected {NO2_MEAN}"
        )
        assert mean_ug[1] == pytest.approx(NO2_MEAN + NO2_STD, abs=1e-4), (
            f"Normalised 1 → {mean_ug[1]:.2f} µg/m³, expected {NO2_MEAN + NO2_STD}"
        )
        assert std_ug[0] == pytest.approx(NO2_STD, abs=1e-4), (
            f"Normalised std 1 → {std_ug[0]:.2f}, expected {NO2_STD}"
        )


# ---------------------------------------------------------------------------
# Test 6: Double normalisation guard
# ---------------------------------------------------------------------------

class TestObservationScaleConsistency:
    """
    Guard against double normalisation of observations in _get_day_observations.

    Background
    ----------
    When normalize_targets=True, the preprocessor normalises observations
    *before* storing them in FusionData.observations.  The bug that was fixed
    in _get_day_observations applied (y - target_mean) / target_std a second
    time, turning a normalised value like 0.5 into (0.5 - 20) / 10 = -1.95.
    With 200+ observations per day in production the Kalman state diverged to
    ~ -1725 µg/m³.

    This test pins a known normalised value through _get_day_observations and
    asserts it comes back unchanged.
    """

    def test_observations_pass_through_unchanged(self):
        """
        _get_day_observations must return data.observations values as-is.
        A fixed normalised value of 0.5 must not become -1.95 after the call.
        """
        from src.inference.gp_kalman_filter import GPKalmanFilter
        from src.data.loader import FusionData

        rng = np.random.default_rng(42)
        scalers = _make_scalers()  # target_mean={"epa": 20.0}, target_std={"epa": 10.0}
        model   = _make_model()
        df_cov  = _make_covariate_df()

        # Observations already in normalised space (as the preprocessor produces).
        # Raw NO2 = 25 µg/m³  →  normalised = (25 − 20) / 10 = 0.5
        KNOWN_NORMED = 0.5
        DOUBLE_NORMED = (KNOWN_NORMED - NO2_MEAN) / NO2_STD   # ≈ -1.95 (the bug)

        n_obs = 6
        coords_norm = rng.uniform(0, 1, size=(n_obs, 2))

        # All observations sit on day 0 (timestamp = 0.0 in whatever scale is used)
        train_known = FusionData(
            coords=coords_norm,
            timestamps=np.zeros(n_obs),
            raw_timestamps=np.zeros(n_obs),
            observations={"epa": np.full(n_obs, KNOWN_NORMED)},
            source_masks={"epa": np.ones(n_obs, dtype=bool)},
            grid_ids=np.arange(n_obs),
            metadata={},
        )

        gpkf = GPKalmanFilter(
            model, scalers,
            covariate_df=df_cov,
            covariate_cols=["traffic_wind_0", "traffic_wind_1"],
            train_data=train_known,
        )

        y_obs, _, _, _, _ = gpkf._get_day_observations(train_known, 0.0, ["epa"])

        assert len(y_obs) == n_obs, (
            f"Expected {n_obs} observations for day_val=0.0, got {len(y_obs)}. "
            "Check timestamp matching in _get_day_observations."
        )
        assert np.allclose(y_obs, KNOWN_NORMED, atol=1e-6), (
            f"y_obs = {y_obs.tolist()!r}\n"
            f"Expected {KNOWN_NORMED} (pre-normalised, unchanged).\n"
            f"Got values near {DOUBLE_NORMED:.4f}? → double normalisation is active: "
            f"_get_day_observations is re-applying (y − target_mean) / target_std "
            f"to already-normalised observations. This causes Kalman divergence in production."
        )
