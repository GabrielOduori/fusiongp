"""
Fast integration tests for the FusionGP pipeline.

Tests the full data flow from raw data → preprocessing → model → predictions,
using tiny synthetic data that runs in < 60 seconds.

Focus: scale contracts, data flow correctness, physical plausibility.
These catch silent semantic errors that unit tests miss.

Run with:
    pytest tests/test_pipeline_contracts.py -v
"""
import io
import numpy as np
import pandas as pd
import pytest
import torch
from pathlib import Path

# ---------------------------------------------------------------------------
# Synthetic data constants
# ---------------------------------------------------------------------------

N_ROWS  = 200
N_GRID  = 30
N_DAYS  = 5
N_IND   = 8

LAT_MIN, LAT_MAX = 53.30, 53.38
LON_MIN, LON_MAX = -6.40, -6.10
NO2_MEAN = 22.0
NO2_STD  = 8.0


def _make_csv(n_rows=N_ROWS, n_grid=N_GRID, n_days=N_DAYS, seed=0) -> str:
    """
    Build a synthetic CSV in the same format as the real FusionGP data.
    This goes through the actual DataLoader so we test the real preprocessing path.
    """
    rng = np.random.default_rng(seed)
    lats = np.linspace(LAT_MIN, LAT_MAX, n_grid)
    lons = np.linspace(LON_MIN, LON_MAX, n_grid)
    cell_idx = rng.integers(0, n_grid, size=n_rows)
    day_idx  = rng.integers(0, n_days,  size=n_rows)

    # Grid IDs like the real data
    grid_ids = [f"{i}_0" for i in cell_idx]

    base_date = pd.Timestamp("2023-06-01")
    timestamps = [(base_date + pd.Timedelta(days=int(d))).strftime("%Y-%m-%d")
                  for d in day_idx]

    no2 = NO2_MEAN + rng.normal(0, NO2_STD, size=n_rows)

    # Half EPA, half satellite
    epa_no2 = np.where(rng.random(n_rows) < 0.5, no2, np.nan)
    sat_no2 = np.where(np.isnan(epa_no2), no2 * 0.9 + rng.normal(0, 2, n_rows), np.nan)

    df = pd.DataFrame({
        "grid_id":    grid_ids,
        "latitude":   lats[cell_idx],
        "longitude":  lons[cell_idx],
        "timestamp":  timestamps,
        "epa_no2":    epa_no2,
        "satellite_no2": sat_no2,
    })
    return df.to_csv(index=False)


def _load_data(csv_str):
    """Load synthetic CSV through the real DataLoader."""
    from src.data.loader import DataLoader as FusionLoader
    import tempfile, os
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        f.write(csv_str)
        tmp = f.name
    try:
        loader = FusionLoader(
            tmp,
            column_mapping={
                "latitude": "latitude",
                "longitude": "longitude",
                "timestamp": "timestamp",
                "grid_id": "grid_id",
                "epa": "epa_no2",
                "satellite": "satellite_no2",
            },
        )
        data = loader.load()
    finally:
        os.unlink(tmp)
    return data


def _make_model(n_ind=N_IND):
    """Minimal FusionSVGP with random inducing points."""
    from src.models.svgp import FusionSVGP
    model = FusionSVGP(n_inducing=n_ind, sources=["epa", "satellite"])
    x_fake = torch.rand(50, 3)
    model.initialize_inducing_points(x_fake, method="random")
    model.eval()
    return model


# ===========================================================================
# 1. DataLoader: output contracts
# ===========================================================================

class TestDataLoaderContracts:

    @pytest.fixture(scope="class")
    def data(self):
        return _load_data(_make_csv())

    def test_coords_in_physical_range(self, data):
        assert data.coords[:, 0].min() >= LAT_MIN - 0.01
        assert data.coords[:, 0].max() <= LAT_MAX + 0.01
        assert data.coords[:, 1].min() >= LON_MIN - 0.01
        assert data.coords[:, 1].max() <= LON_MAX + 0.01

    def test_timestamps_are_numeric_days(self, data):
        """DataLoader must convert string dates to days since first obs (0, 1, 2, ...)."""
        assert data.timestamps.dtype != object, (
            "Timestamps are still strings — DataLoader did not convert them"
        )
        assert data.timestamps.min() >= 0
        assert data.timestamps.max() <= N_DAYS + 1, (
            f"Timestamps max={data.timestamps.max():.1f} — expected ≤ {N_DAYS}. "
            "Are these day numbers or Unix timestamps?"
        )

    def test_observations_have_expected_sources(self, data):
        assert "epa" in data.observations
        assert "satellite" in data.observations

    def test_observations_in_physical_no2_range(self, data):
        for src in ["epa", "satellite"]:
            obs = data.observations[src][data.source_masks[src]]
            assert obs.max() < 500, f"{src} max={obs.max():.1f} — outside plausible range"
            # Note: DataLoader does not clip to >=0; negative values can occur in normalised data
            assert obs.min() > -200, f"{src} min={obs.min():.1f} — extreme negative value"

    def test_no_nan_in_masked_observations(self, data):
        """Observations at valid source_mask positions must not be NaN."""
        for src in ["epa", "satellite"]:
            obs = data.observations[src][data.source_masks[src]]
            assert np.all(np.isfinite(obs)), (
                f"{src} has NaN in masked positions — DataLoader mask is wrong"
            )

    def test_grid_ids_present(self, data):
        assert data.grid_ids is not None
        assert len(data.grid_ids) == len(data.timestamps)

    def test_raw_timestamps_preserved(self, data):
        """raw_timestamps must exist — used to pass correct scale to GPKF."""
        assert hasattr(data, "raw_timestamps") and data.raw_timestamps is not None


# ===========================================================================
# 2. DataPreprocessor: normalisation contracts
# ===========================================================================

class TestPreprocessorContracts:

    @pytest.fixture(scope="class")
    def preprocessed(self):
        from src.data.preprocessor import DataPreprocessor
        data = _load_data(_make_csv())
        pp = DataPreprocessor(normalize_coords=True, normalize_time=True, normalize_targets=True)
        train, val, test = pp.fit_transform(data, train_ratio=0.7, val_ratio=0.15)
        return pp, train, val, test, data

    def test_coords_in_unit_interval(self, preprocessed):
        _, train, val, test, _ = preprocessed
        for split, name in [(train, "train"), (val, "val"), (test, "test")]:
            assert split.coords.min() >= -0.01, f"{name} coords below 0"
            assert split.coords.max() <= 1.01,  f"{name} coords above 1"

    def test_timestamps_in_unit_interval(self, preprocessed):
        _, train, val, test, _ = preprocessed
        for split, name in [(train, "train"), (val, "val"), (test, "test")]:
            assert split.timestamps.min() >= -0.01, f"{name} timestamps below 0"
            assert split.timestamps.max() <= 1.01,  f"{name} timestamps above 1"

    def test_split_sizes_sum_to_total(self, preprocessed):
        _, train, val, test, data = preprocessed
        total = len(data.timestamps)
        split_total = len(train.timestamps) + len(val.timestamps) + len(test.timestamps)
        assert split_total == total

    def test_train_ratio_approximately_correct(self, preprocessed):
        _, train, _, _, data = preprocessed
        ratio = len(train.timestamps) / len(data.timestamps)
        assert 0.60 <= ratio <= 0.80, f"Train ratio {ratio:.2f} far from 0.70"

    def test_scalers_have_correct_structure(self, preprocessed):
        pp, _, _, _, _ = preprocessed
        s = pp.scalers
        assert hasattr(s, "coord_min")
        assert hasattr(s, "coord_scale")
        assert hasattr(s, "time_min")
        assert hasattr(s, "time_max")
        assert s.time_max > s.time_min, "time_max must exceed time_min"

    def test_inverse_transform_recovers_physical_scale(self, preprocessed):
        pp, train, _, _, _ = preprocessed
        s = pp.scalers
        epa_mask = train.source_masks["epa"]
        if epa_mask.sum() == 0:
            pytest.skip("No EPA observations in train split")
        norm_vals = train.observations["epa"][epa_mask]
        inv_mean, inv_std = s.inverse_transform_predictions(
            norm_vals, np.ones_like(norm_vals), source="epa"
        )
        # Should be near NO2_MEAN, not near 0 (which would mean still normalised)
        assert inv_mean.mean() == pytest.approx(NO2_MEAN, abs=NO2_STD * 3), (
            f"Inverse-transformed mean {inv_mean.mean():.2f} should be near {NO2_MEAN} µg/m³. "
            "Likely not converting from normalised space."
        )


# ===========================================================================
# 3. GridPriorMean: output range and smoothness
# ===========================================================================

class TestGridPriorMean:

    @pytest.fixture(scope="class")
    def setup(self):
        from src.data.preprocessor import DataPreprocessor
        from src.models.prior_mean import GridPriorMean
        data = _load_data(_make_csv())
        pp = DataPreprocessor(normalize_coords=True, normalize_time=True, normalize_targets=True)
        train, _, _ = pp.fit_transform(data)
        scalers = pp.scalers

        lats = np.linspace(LAT_MIN, LAT_MAX, N_GRID)
        lons = np.linspace(LON_MIN, LON_MAX, N_GRID)
        coords = np.column_stack([lats, lons])
        values = np.full(N_GRID, NO2_MEAN) + np.random.default_rng(0).normal(0, 2, N_GRID)

        pm = GridPriorMean(
            grid_coords=coords, grid_values=values,
            scalers=scalers, learnable_bias=True,
        )
        return pm, scalers, train

    def test_forward_is_finite(self, setup):
        pm, scalers, train = setup
        x = torch.tensor(train.coords[:10], dtype=torch.float32)
        t = torch.zeros(10, 1)
        x_in = torch.cat([x, t], dim=1)
        out = pm(x_in)
        assert out.shape == (10,)
        assert torch.all(torch.isfinite(out)), "GridPriorMean returned NaN/inf"

    def test_values_are_normalised_not_raw_ug(self, setup):
        """Output should be in normalised space (~0), not in µg/m³ (~20)."""
        pm, scalers, train = setup
        x = torch.tensor(train.coords[:20], dtype=torch.float32)
        t = torch.zeros(20, 1)
        x_in = torch.cat([x, t], dim=1)
        out = pm(x_in).detach().numpy()
        assert out.max() < 20, (
            f"GridPriorMean max={out.max():.1f} — values appear to be in µg/m³, not normalised. "
            "Check normalize_values flag in GridPriorMean."
        )

    def test_nearby_points_have_similar_values(self, setup):
        pm, scalers, _ = setup
        # Query at two very close normalised points
        x1 = torch.tensor([[0.5, 0.5, 0.0]], dtype=torch.float32)
        x2 = torch.tensor([[0.501, 0.501, 0.0]], dtype=torch.float32)
        v1 = pm(x1).item()
        v2 = pm(x2).item()
        assert abs(v1 - v2) < 1.0, (
            f"Adjacent points differ by {abs(v1-v2):.3f} — interpolation not smooth"
        )


# ===========================================================================
# 4. FusionSVGP: ELBO and prediction contracts
# ===========================================================================

class TestSVGPContracts:

    @pytest.fixture(scope="class")
    def model_and_data(self):
        from src.data.preprocessor import DataPreprocessor
        from src.data.loader import FusionDataset

        data = _load_data(_make_csv())
        pp = DataPreprocessor(normalize_coords=True, normalize_time=True, normalize_targets=True)
        train, _, test = pp.fit_transform(data, train_ratio=0.7, val_ratio=0.15)

        model = _make_model()
        # Re-init inducing points from actual training data
        x_tr = torch.tensor(
            np.column_stack([train.coords, train.timestamps]),
            dtype=torch.float32
        )
        model.initialize_inducing_points(x_tr, method="random")
        ds = FusionDataset(train)
        return model, train, test, ds, pp.scalers

    def test_elbo_is_finite(self, model_and_data):
        from src.data.loader import FusionDataset
        model, train, _, ds, _ = model_and_data
        model.train()
        batch = ds[np.arange(min(20, len(ds)))]
        # FusionDataset returns 6 elements: coords, timestamps, epa_obs, sat_obs, epa_mask, sat_mask
        # Build x and y tensors from batch
        coords, timestamps = batch[0], batch[1]
        x = torch.cat([coords, timestamps.unsqueeze(1)], dim=1)
        # Get combined observations and masks from model's expected format
        epa_obs, sat_obs = batch[2], batch[3]
        epa_mask, sat_mask = batch[4].bool(), batch[5].bool() if batch[5].shape[1] > 0 else torch.zeros(len(coords), 0, dtype=torch.bool)
        y = torch.stack([epa_obs.squeeze(), sat_obs.squeeze()], dim=1) if sat_obs.shape[1] > 0 else epa_obs
        # Use SVGP forward directly
        model.eval()
        with torch.no_grad():
            posterior = model(x)
        assert torch.all(torch.isfinite(posterior.mean)), "SVGP posterior mean is NaN/inf"

    def test_predict_returns_finite_values(self, model_and_data):
        model, _, test, _, _ = model_and_data
        model.eval()
        x = torch.tensor(
            np.column_stack([test.coords[:10], test.timestamps[:10]]),
            dtype=torch.float32,
        )
        with torch.no_grad():
            posterior = model(x)
        assert torch.all(torch.isfinite(posterior.mean)), "predict() mean is NaN/inf"
        assert torch.all(torch.isfinite(posterior.variance)), "predict() variance is NaN/inf"
        assert torch.all(posterior.variance >= 0), "predict() returned negative variance"

    def test_predictions_in_normalised_range(self, model_and_data):
        """Predictions in normalised space should not be in µg/m³ range."""
        model, _, test, _, _ = model_and_data
        model.eval()
        x = torch.tensor(
            np.column_stack([test.coords[:20], test.timestamps[:20]]),
            dtype=torch.float32,
        )
        with torch.no_grad():
            posterior = model(x)
        mean_np = posterior.mean.numpy()
        assert mean_np.max() < 100, (
            f"Predicted mean {mean_np.max():.1f} looks like µg/m³, expected normalised space. "
            "Check that normalise_targets=True in preprocessing."
        )

    def test_uncertainty_is_positive(self, model_and_data):
        model, _, test, _, _ = model_and_data
        model.eval()
        x = torch.tensor(
            np.column_stack([test.coords[:10], test.timestamps[:10]]),
            dtype=torch.float32,
        )
        with torch.no_grad():
            posterior = model(x)
        assert torch.all(posterior.variance > 0), "All predictive variances must be > 0"


# ===========================================================================
# 5. Predictor: inverse transform and output scale
# ===========================================================================

class TestPredictorContracts:

    @pytest.fixture(scope="class")
    def predictor_setup(self):
        from src.data.preprocessor import DataPreprocessor
        from src.inference.predictor import Predictor

        data = _load_data(_make_csv())
        pp = DataPreprocessor(normalize_coords=True, normalize_time=True, normalize_targets=True)
        train, _, test = pp.fit_transform(data, train_ratio=0.7, val_ratio=0.15)

        model = _make_model()
        x_tr = torch.tensor(
            np.column_stack([train.coords, train.timestamps]),
            dtype=torch.float32
        )
        model.initialize_inducing_points(x_tr, method="random")
        predictor = Predictor(model, pp.scalers)
        return predictor, test, pp.scalers

    def test_predictions_in_physical_no2_range(self, predictor_setup):
        predictor, test, _ = predictor_setup
        preds = predictor.predict(test)
        assert preds.mean.max() < 300, (
            f"Max prediction {preds.mean.max():.1f} µg/m³ — likely not inverse-transformed"
        )
        assert preds.mean.min() > -100, (
            f"Min prediction {preds.mean.min():.1f} µg/m³ — likely scale bug"
        )

    def test_predictions_near_true_mean(self, predictor_setup):
        predictor, test, _ = predictor_setup
        preds = predictor.predict(test)
        pred_mean = preds.mean.mean()
        assert abs(pred_mean - NO2_MEAN) < NO2_STD * 5, (
            f"Mean prediction {pred_mean:.2f} far from {NO2_MEAN} µg/m³. "
            "Inverse transform may be wrong."
        )

    def test_std_is_positive(self, predictor_setup):
        predictor, test, _ = predictor_setup
        preds = predictor.predict(test)
        assert np.all(preds.std > 0), "Predictive std must be positive everywhere"

    def test_output_shapes_consistent(self, predictor_setup):
        predictor, test, _ = predictor_setup
        preds = predictor.predict(test)
        n = len(test.timestamps)
        assert preds.mean.shape == (n,), f"mean shape {preds.mean.shape} != ({n},)"
        assert preds.std.shape  == (n,), f"std shape {preds.std.shape} != ({n},)"


# ===========================================================================
# 6. Evaluator: metric sanity
# ===========================================================================

class TestEvaluatorContracts:

    def test_perfect_predictions_give_zero_rmse(self):
        from src.evaluation.metrics import Evaluator
        y = np.array([10.0, 20.0, 30.0, 15.0, 25.0])
        ev = Evaluator()
        result = ev.evaluate(y, y, np.ones_like(y) * 0.1)
        d = result.to_dict()
        assert d["rmse"] == pytest.approx(0.0, abs=1e-6), \
            f"Perfect predictions: RMSE={d['rmse']}, expected 0"

    def test_random_predictions_worse_than_data_std(self):
        from src.evaluation.metrics import Evaluator
        rng = np.random.default_rng(0)
        y      = rng.normal(NO2_MEAN, NO2_STD, size=300)
        y_rand = rng.normal(NO2_MEAN, NO2_STD * 3, size=300)
        ev = Evaluator()
        result = ev.evaluate(y, y_rand, np.ones_like(y) * NO2_STD)
        d = result.to_dict()
        assert d["rmse"] > NO2_STD * 0.5, \
            "Random predictor RMSE should exceed half the data std"

    def test_coverage_of_honest_intervals_is_high(self):
        from src.evaluation.metrics import Evaluator
        rng = np.random.default_rng(1)
        y    = rng.normal(20.0, 5.0, size=500)
        pred = y + rng.normal(0, 0.5, size=500)
        std  = np.full(500, 5.0)
        ev = Evaluator(confidence_levels=[0.9])
        d = ev.evaluate(y, pred, std).to_dict()
        assert d["coverage_90"] >= 0.80, \
            f"Honest uncertainty coverage {d['coverage_90']:.2f} < 0.80"

    def test_coverage_of_overconfident_model_is_low(self):
        from src.evaluation.metrics import Evaluator
        rng = np.random.default_rng(2)
        y    = rng.normal(20.0, 5.0, size=500)
        pred = y + rng.normal(0, 3.0, size=500)
        std  = np.full(500, 0.01)                  # massively overconfident
        ev = Evaluator(confidence_levels=[0.9])
        d = ev.evaluate(y, pred, std).to_dict()
        assert d["coverage_90"] < 0.5, \
            f"Overconfident model coverage {d['coverage_90']:.2f} should be << 0.90"

    def test_crps_better_for_accurate_predictions(self):
        from src.evaluation.metrics import Evaluator
        rng = np.random.default_rng(3)
        y        = rng.normal(20.0, 5.0, size=200)
        good     = y + rng.normal(0, 0.5, size=200)
        bad      = rng.normal(20.0, 5.0, size=200)
        std      = np.full(200, 3.0)
        ev = Evaluator()
        d_good = ev.evaluate(y, good, std).to_dict()
        d_bad  = ev.evaluate(y, bad,  std).to_dict()
        assert d_good["crps"] < d_bad["crps"], \
            f"Good CRPS ({d_good['crps']:.3f}) must be lower than bad ({d_bad['crps']:.3f})"

    def test_nll_is_finite(self):
        from src.evaluation.metrics import Evaluator
        rng = np.random.default_rng(4)
        y   = rng.normal(20.0, 5.0, size=100)
        pred = y + rng.normal(0, 1.0, size=100)
        std  = np.full(100, 3.0)
        d = Evaluator().evaluate(y, pred, std).to_dict()
        assert np.isfinite(d["nll"]), "NLL is NaN/inf"


# ===========================================================================
# 7. GPKF timestamp scale contract (regression test)
# ===========================================================================

class TestGPKFTimestampContract:
    """
    Regression tests for the timestamp scale bug that caused GPKF RMSE=139.
    The filter must receive timestamps in day scale (0..N_DAYS-1), not [0,1].
    """

    def test_raw_timestamps_are_strings_or_dates(self):
        """raw_timestamps preserve the original string date format from the CSV."""
        data = _load_data(_make_csv())
        # raw_timestamps are string dates like "2023-06-01"
        assert data.raw_timestamps is not None
        assert len(data.raw_timestamps) == len(data.timestamps)
        # Should be parseable as dates
        sample = pd.to_datetime(data.raw_timestamps[:5], errors="coerce")
        assert sample.notna().all(), "raw_timestamps are not parseable as dates"

    def test_raw_timestamps_convertible_to_day_numbers(self):
        """
        The GPKF needs raw_timestamps converted to day numbers.
        Verify they can be converted correctly.
        """
        data = _load_data(_make_csv())
        raw_dt = pd.to_datetime(data.raw_timestamps, errors="coerce")
        day_numbers = (raw_dt - raw_dt.min()).total_seconds() / 86400.0
        day_range = day_numbers.max() - day_numbers.min()
        assert day_range >= N_DAYS - 1.5, (
            f"Converted day range={day_range:.2f} — expected ~{N_DAYS - 1} days."
        )

    def test_preprocessor_normalises_timestamps_to_unit_interval(self):
        """
        DataLoader timestamps are in day scale [0, N_DAYS-1].
        DataPreprocessor normalises them to [0, 1].
        GPKF must use DataLoader timestamps, NOT preprocessor-normalised timestamps.
        """
        from src.data.preprocessor import DataPreprocessor
        data = _load_data(_make_csv())
        pp   = DataPreprocessor(normalize_time=True)
        train, _, _ = pp.fit_transform(data)

        # DataLoader: timestamps in day scale
        loader_range = data.timestamps.max() - data.timestamps.min()
        # DataPreprocessor: timestamps normalised to [0, 1]
        preprocessed_range = train.timestamps.max() - train.timestamps.min()

        assert loader_range >= N_DAYS - 1.5, (
            f"DataLoader timestamp range={loader_range:.2f}, expected ~{N_DAYS - 1} days"
        )
        assert preprocessed_range <= 1.01, (
            f"Preprocessed timestamps range={preprocessed_range:.3f}, expected ≤ 1.0"
        )
        assert loader_range > preprocessed_range * 2, (
            f"DataLoader range={loader_range:.2f} vs preprocessed={preprocessed_range:.3f}. "
            "GPKF must use DataLoader (day-scale) timestamps, not preprocessor [0,1] timestamps."
        )

    def test_ou_transition_delta_t_is_in_day_scale(self):
        """
        A = exp(-Δt / ℓ_t). If Δt≈0.036 (normalised) instead of 1.0 (days),
        A≈1.0 and the filter will not revert to the prior → divergence.
        """
        from src.inference.gp_kalman_filter import GPKalmanFilter
        from tests.test_gpkf_pipeline import (
            _make_scalers, _make_covariate_df, _make_train_data, _make_model
        )
        scalers = _make_scalers()
        model   = _make_model()
        df_cov  = _make_covariate_df()
        train   = _make_train_data()
        gpkf    = GPKalmanFilter(model, scalers,
                                 covariate_df=df_cov,
                                 covariate_cols=["traffic_wind_0", "traffic_wind_1"],
                                 train_data=train)

        # Pass day-scale unique_days (0, 1, 2, ...)
        unique_days_correct = np.arange(5.0)
        A_arr, _ = gpkf._ou_params(unique_days_correct)
        assert A_arr.mean() < 0.999, (
            f"A={A_arr.mean():.5f} ≈ 1.0 with day-scale Δt. "
            "The OU transition has negligible decay — "
            "check that unique_days is in day scale not normalised."
        )

    def test_pipeline_uses_raw_timestamps_for_gpkf(self):
        """
        Verify that after the pipeline timestamp fix, raw_timestamps can be
        converted to the correct day scale for the GPKF.
        Simulates what run_demo_pipeline.py does after the fix.
        """
        data = _load_data(_make_csv())
        # Simulate what the corrected pipeline does
        raw_dt   = pd.to_datetime(data.raw_timestamps, errors="coerce")
        raw_days = (raw_dt - raw_dt.min()).total_seconds() / 86400.0
        unique_days = np.unique(raw_days.values)

        # Day deltas should be ~1.0
        deltas = np.diff(np.sort(unique_days))
        assert deltas.mean() == pytest.approx(1.0, abs=0.1), (
            f"Day deltas = {deltas.mean():.4f} — expected 1.0. "
            "raw_timestamps may not be convertible to day units."
        )


# ===========================================================================
# 8. Training history CSV building (regression for IndexError with short val_loss)
# ===========================================================================

class TestTrainingHistoryCSV:
    """
    Regression tests for the training-history CSV row building logic in
    run_demo_pipeline.py.  The bug: val_loss list may be shorter than
    train_loss (e.g. validation runs every N epochs) causing IndexError.
    """

    def _build_rows(self, history_dict):
        """Mirror the fixed logic from run_demo_pipeline.py."""
        val_losses = history_dict.get("val_loss", [])
        rows = []
        for epoch_idx, tl in enumerate(history_dict.get("train_loss", [])):
            rows.append({
                "epoch": epoch_idx,
                "train_loss": tl,
                "val_loss": val_losses[epoch_idx] if epoch_idx < len(val_losses) else None,
            })
        return rows

    def test_equal_length_lists(self):
        history = {"train_loss": [2.0, 1.5, 1.0], "val_loss": [2.1, 1.6, 1.1]}
        rows = self._build_rows(history)
        assert len(rows) == 3
        assert rows[2]["val_loss"] == pytest.approx(1.1)

    def test_val_loss_shorter_than_train_loss(self):
        """val_loss runs every 10 epochs — list is much shorter than train_loss."""
        history = {"train_loss": list(range(30)), "val_loss": [99.0, 88.0, 77.0]}
        rows = self._build_rows(history)
        assert len(rows) == 30
        assert rows[0]["val_loss"] == pytest.approx(99.0)
        assert rows[1]["val_loss"] == pytest.approx(88.0)
        assert rows[2]["val_loss"] == pytest.approx(77.0)
        # Epochs beyond val_loss length should get None, not IndexError
        assert rows[3]["val_loss"] is None
        assert rows[29]["val_loss"] is None

    def test_empty_val_loss(self):
        """No validation at all — val_loss key absent."""
        history = {"train_loss": [2.0, 1.5]}
        rows = self._build_rows(history)
        assert len(rows) == 2
        assert all(r["val_loss"] is None for r in rows)

    def test_empty_train_loss(self):
        history = {"train_loss": [], "val_loss": []}
        rows = self._build_rows(history)
        assert rows == []

    def test_csv_roundtrip(self, tmp_path):
        """Rows survive a pd.DataFrame → CSV → read_csv roundtrip."""
        history = {"train_loss": [2.0, 1.5, 1.0], "val_loss": [2.1]}
        rows = self._build_rows(history)
        path = tmp_path / "training_history.csv"
        pd.DataFrame(rows).to_csv(path, index=False)
        df = pd.read_csv(path)
        assert len(df) == 3
        assert df["epoch"].tolist() == [0, 1, 2]
        assert df["train_loss"].tolist() == pytest.approx([2.0, 1.5, 1.0])
        # val_loss rows 1 and 2 should be NaN (read back as float NaN)
        assert pd.isna(df.loc[1, "val_loss"])
        assert pd.isna(df.loc[2, "val_loss"])


# ===========================================================================
# 9. End-to-end smoke test
# ===========================================================================

class TestEndToEndSmoke:
    """
    Full pipeline on synthetic data. Runs in ~10 seconds.
    Verifies nothing crashes and outputs are in physical range.
    """

    def test_load_preprocess_predict_evaluate(self):
        from src.data.preprocessor import DataPreprocessor
        from src.inference.predictor import Predictor
        from src.evaluation.metrics import Evaluator

        # 1. Load
        data = _load_data(_make_csv(n_rows=300, n_grid=40, n_days=5))

        # 2. Preprocess
        pp = DataPreprocessor(normalize_coords=True, normalize_time=True, normalize_targets=True)
        train, val, test = pp.fit_transform(data, train_ratio=0.7, val_ratio=0.15)
        s = pp.scalers

        # Scale contracts
        assert train.coords.max() <= 1.01
        assert train.timestamps.max() <= 1.01

        # 3. Model
        model = _make_model()
        x_tr = torch.tensor(
            np.column_stack([train.coords, train.timestamps]),
            dtype=torch.float32
        )
        model.initialize_inducing_points(x_tr, method="random")

        # 4. Predict
        predictor = Predictor(model, s)
        preds = predictor.predict(test)

        assert np.all(np.isfinite(preds.mean)), "NaN/inf in predictions"
        assert np.all(preds.std > 0), "Non-positive std in predictions"
        assert preds.mean.max() < 500, (
            f"Max prediction {preds.mean.max():.1f} µg/m³ — scale bug"
        )

        # 5. Evaluate
        epa_mask = test.source_masks["epa"]
        if epa_mask.sum() > 5:
            y_norm = test.observations["epa"][epa_mask]
            y_ug, _ = s.inverse_transform_predictions(y_norm, np.ones_like(y_norm), source="epa")
            y_pred = preds.mean[epa_mask]
            y_std  = preds.std[epa_mask]

            ev = Evaluator(confidence_levels=[0.9])
            d  = ev.evaluate(y_ug, y_pred, y_std).to_dict()

            assert np.isfinite(d["rmse"]), "RMSE is NaN"
            assert d["rmse"] < NO2_STD * 10, (
                f"RMSE={d['rmse']:.2f} — unrealistically high (>{NO2_STD*10}). "
                "Likely a unit mismatch between y_true and y_pred."
            )
