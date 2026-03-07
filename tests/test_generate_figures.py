"""
Tests for experiments/generate_figures.py.

Tests use temporary directories with synthetic CSVs so no real pipeline output
is needed.  Figures are written to tmp dirs and assertions check that the files
exist and are non-empty.
"""

import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# We need matplotlib to avoid display errors in headless CI
import matplotlib
matplotlib.use("Agg")

from experiments.generate_figures import (
    csv_to_grid,
    plot_training_history,
    plot_svgp_diagnostics,
    plot_gpkf_diagnostics,
    plot_timeseries,
    find_run_dir,
    GRID_ROWS,
    GRID_COLS,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_grid_id(n=20):
    """Return n grid_id strings like '0_0', '0_1', ..."""
    rows = np.arange(n) % GRID_ROWS
    cols = np.arange(n) % GRID_COLS
    return [f"{r}_{c}" for r, c in zip(rows, cols)]


@pytest.fixture()
def run_dir(tmp_path):
    """Minimal run directory with all required CSVs."""
    n = 20
    gids = _make_grid_id(n)
    lats = np.linspace(53.3, 53.4, n)
    lons = np.linspace(-6.4, -6.3, n)

    # training_history.csv
    pd.DataFrame({
        "epoch": np.arange(10),
        "train_loss": np.linspace(2.0, 0.5, 10),
        "val_loss":   np.linspace(2.2, 0.6, 10),
    }).to_csv(tmp_path / "training_history.csv", index=False)

    # predictions.csv  (SVGP output — include EPA rows)
    pd.DataFrame({
        "latitude":  lats,
        "longitude": lons,
        "timestamp": np.tile([0.0, 1.0], n // 2),
        "mean":      np.random.uniform(10, 30, n),
        "std":       np.random.uniform(1, 5, n),
        "is_epa":    [True] * (n // 2) + [False] * (n // 2),
        "epa_true":  np.concatenate([np.random.uniform(10, 30, n // 2),
                                     np.full(n // 2, np.nan)]),
        "grid_id":   gids,
    }).to_csv(tmp_path / "predictions.csv", index=False)

    # gpkf_epa_predictions.csv
    pd.DataFrame({
        "latitude":     lats[:10],
        "longitude":    lons[:10],
        "timestamp":    np.zeros(10),
        "epa_true":     np.random.uniform(10, 30, 10),
        "gpkf_pred":    np.random.uniform(10, 30, 10),
        "gpkf_pred_std": np.random.uniform(1, 5, 10),
    }).to_csv(tmp_path / "gpkf_epa_predictions.csv", index=False)

    # gpkf_maps/  (2 days)
    gpkf_dir = tmp_path / "gpkf_maps"
    gpkf_dir.mkdir()
    for day in range(2):
        pd.DataFrame({
            "grid_id":    gids,
            "latitude":   lats,
            "longitude":  lons,
            "mean_ug_m3": np.random.uniform(10, 30, n),
            "std_ug_m3":  np.random.uniform(1, 5, n),
            "n_obs":      [5] * n,
            "day":        [float(day)] * n,
        }).to_csv(gpkf_dir / f"gpkf_day_{day:02d}.csv", index=False)

    return tmp_path


# ---------------------------------------------------------------------------
# Unit: csv_to_grid
# ---------------------------------------------------------------------------

class TestCsvToGrid:
    def test_basic_shape(self):
        n = 10
        gids = _make_grid_id(n)
        df = pd.DataFrame({"grid_id": gids, "value": np.ones(n)})
        grid = csv_to_grid(df, "value")
        assert grid.shape == (GRID_ROWS, GRID_COLS)

    def test_values_placed_correctly(self):
        df = pd.DataFrame({"grid_id": ["0_0", "1_2"], "value": [42.0, 7.0]})
        grid = csv_to_grid(df, "value")
        assert grid[0, 0] == pytest.approx(42.0)
        assert grid[1, 2] == pytest.approx(7.0)
        assert np.isnan(grid[0, 1])

    def test_nan_where_absent(self):
        df = pd.DataFrame({"grid_id": ["0_0"], "value": [1.0]})
        grid = csv_to_grid(df, "value")
        # All other cells should be NaN
        assert np.isnan(grid[5, 5])


# ---------------------------------------------------------------------------
# Integration: plot functions produce output files
# ---------------------------------------------------------------------------

class TestPlotTrainingHistory:
    def test_creates_png(self, run_dir):
        out = run_dir / "diag"
        plot_training_history(run_dir, out)
        assert (out / "training_history.png").exists()
        assert (out / "training_history.png").stat().st_size > 0

    def test_missing_csv_does_not_raise(self, tmp_path):
        out = tmp_path / "diag"
        # No training_history.csv in tmp_path
        plot_training_history(tmp_path, out)
        assert not (out / "training_history.png").exists()


class TestPlotSvgpDiagnostics:
    def test_creates_png(self, run_dir):
        out = run_dir / "diag"
        plot_svgp_diagnostics(run_dir, out)
        assert (out / "svgp_diagnostics.png").exists()
        assert (out / "svgp_diagnostics.png").stat().st_size > 0

    def test_missing_csv_does_not_raise(self, tmp_path):
        out = tmp_path / "diag"
        plot_svgp_diagnostics(tmp_path, out)
        assert not (out / "svgp_diagnostics.png").exists()

    def test_no_epa_rows_does_not_raise(self, tmp_path):
        pd.DataFrame({
            "latitude": [53.3], "longitude": [-6.3],
            "timestamp": [0.0], "mean": [15.0], "std": [2.0],
            "is_epa": [False], "epa_true": [np.nan],
        }).to_csv(tmp_path / "predictions.csv", index=False)
        out = tmp_path / "diag"
        plot_svgp_diagnostics(tmp_path, out)
        assert not (out / "svgp_diagnostics.png").exists()


class TestPlotGpkfDiagnostics:
    def test_creates_png(self, run_dir):
        out = run_dir / "diag"
        plot_gpkf_diagnostics(run_dir, out)
        assert (out / "gpkf_diagnostics.png").exists()
        assert (out / "gpkf_diagnostics.png").stat().st_size > 0

    def test_missing_csv_does_not_raise(self, tmp_path):
        out = tmp_path / "diag"
        plot_gpkf_diagnostics(tmp_path, out)
        assert not (out / "gpkf_diagnostics.png").exists()

    def test_all_nan_does_not_raise(self, tmp_path):
        pd.DataFrame({
            "epa_true": [np.nan], "gpkf_pred": [np.nan], "gpkf_pred_std": [np.nan],
        }).to_csv(tmp_path / "gpkf_epa_predictions.csv", index=False)
        out = tmp_path / "diag"
        plot_gpkf_diagnostics(tmp_path, out)
        assert not (out / "gpkf_diagnostics.png").exists()


class TestPlotTimeseries:
    def test_domain_average_png(self, run_dir):
        out = run_dir / "ts"
        plot_timeseries(run_dir, out)
        assert (out / "timeseries_domain_average.png").exists()
        assert (out / "timeseries_domain_average.png").stat().st_size > 0

    def test_epa_vs_svgp_png(self, run_dir):
        out = run_dir / "ts"
        plot_timeseries(run_dir, out)
        assert (out / "timeseries_epa_vs_svgp.png").exists()
        assert (out / "timeseries_epa_vs_svgp.png").stat().st_size > 0

    def test_missing_gpkf_dir_does_not_raise(self, tmp_path):
        # No gpkf_maps/ — both panels should be skipped gracefully
        pd.DataFrame({
            "latitude": [53.3], "longitude": [-6.3], "timestamp": [0.0],
            "mean": [15.0], "std": [2.0], "is_epa": [True], "epa_true": [15.0],
        }).to_csv(tmp_path / "predictions.csv", index=False)
        out = tmp_path / "ts"
        plot_timeseries(tmp_path, out)  # should not raise


class TestFindRunDir:
    def test_explicit_path(self, tmp_path):
        assert find_run_dir(str(tmp_path)) == tmp_path

    def test_no_candidates_raises(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        with pytest.raises(FileNotFoundError):
            find_run_dir(None)
