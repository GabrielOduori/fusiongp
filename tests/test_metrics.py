"""
Tests for evaluation metrics.

Run with: pytest tests/test_metrics.py -v
"""

import numpy as np
import pytest

from src.evaluation.metrics import (
    rmse, mae, r_squared, bias, mape,
    negative_log_likelihood, crps, dawid_sebastiani_score,
    calibration_error, coverage, sharpness,
    pit_values, pit_deviation, interval_score,
    Evaluator, compute_metrics,
)


class TestPointMetrics:
    """Tests for point prediction metrics."""
    
    def test_rmse_perfect(self):
        """Test RMSE with perfect predictions."""
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.0, 2.0, 3.0])
        
        assert rmse(y_true, y_pred) == 0.0
    
    def test_rmse_known(self):
        """Test RMSE with known values."""
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([2.0, 3.0, 4.0])  # All off by 1
        
        assert rmse(y_true, y_pred) == 1.0
    
    def test_mae_perfect(self):
        """Test MAE with perfect predictions."""
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.0, 2.0, 3.0])
        
        assert mae(y_true, y_pred) == 0.0
    
    def test_mae_known(self):
        """Test MAE with known values."""
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([0.0, 3.0, 5.0])  # Errors: 1, 1, 2
        
        assert mae(y_true, y_pred) == pytest.approx(4/3)
    
    def test_r_squared_perfect(self):
        """Test R² with perfect predictions."""
        y_true = np.array([1.0, 2.0, 3.0, 4.0])
        y_pred = np.array([1.0, 2.0, 3.0, 4.0])
        
        assert r_squared(y_true, y_pred) == pytest.approx(1.0)
    
    def test_r_squared_mean(self):
        """Test R² when predicting the mean."""
        y_true = np.array([1.0, 2.0, 3.0, 4.0])
        y_pred = np.array([2.5, 2.5, 2.5, 2.5])  # Mean of y_true
        
        assert r_squared(y_true, y_pred) == pytest.approx(0.0)
    
    def test_bias_positive(self):
        """Test positive bias (over-prediction)."""
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([2.0, 3.0, 4.0])  # All +1
        
        assert bias(y_true, y_pred) == pytest.approx(1.0)
    
    def test_bias_negative(self):
        """Test negative bias (under-prediction)."""
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([0.0, 1.0, 2.0])  # All -1
        
        assert bias(y_true, y_pred) == pytest.approx(-1.0)


class TestProbabilisticMetrics:
    """Tests for probabilistic metrics."""
    
    def test_nll_low_uncertainty(self):
        """Test NLL increases with lower uncertainty (if predictions wrong)."""
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred_mean = np.array([2.0, 3.0, 4.0])  # Off by 1
        
        nll_low = negative_log_likelihood(y_true, y_pred_mean, np.array([0.1, 0.1, 0.1]))
        nll_high = negative_log_likelihood(y_true, y_pred_mean, np.array([1.0, 1.0, 1.0]))
        
        assert nll_low > nll_high
    
    def test_nll_perfect(self):
        """Test NLL with perfect predictions and appropriate uncertainty."""
        y_true = np.array([0.0, 0.0, 0.0])
        y_pred_mean = np.array([0.0, 0.0, 0.0])
        y_pred_std = np.array([1.0, 1.0, 1.0])
        
        # NLL for N(0,1) at y=0 is 0.5*log(2*pi) ≈ 0.919
        nll = negative_log_likelihood(y_true, y_pred_mean, y_pred_std)
        assert nll == pytest.approx(0.5 * np.log(2 * np.pi), rel=0.01)
    
    def test_crps_perfect(self):
        """Test CRPS approaches 0 for perfect predictions."""
        y_true = np.array([0.0, 0.0, 0.0])
        y_pred_mean = np.array([0.0, 0.0, 0.0])
        y_pred_std = np.array([0.01, 0.01, 0.01])  # Very confident
        
        crps_value = crps(y_true, y_pred_mean, y_pred_std)
        assert crps_value < 0.1  # Should be very small
    
    def test_crps_positive(self):
        """Test CRPS is always positive."""
        np.random.seed(42)
        y_true = np.random.randn(100)
        y_pred_mean = np.random.randn(100)
        y_pred_std = np.abs(np.random.randn(100)) + 0.1
        
        assert crps(y_true, y_pred_mean, y_pred_std) > 0
    
    def test_dss_finite(self):
        """Test DSS returns finite values."""
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred_mean = np.array([1.1, 2.1, 3.1])
        y_pred_std = np.array([0.5, 0.5, 0.5])
        
        dss = dawid_sebastiani_score(y_true, y_pred_mean, y_pred_std)
        assert np.isfinite(dss)
    
    def test_sharpness(self):
        """Test sharpness computation."""
        y_pred_std = np.array([1.0, 2.0, 3.0])
        
        assert sharpness(y_pred_std) == pytest.approx(2.0)


class TestCalibrationMetrics:
    """Tests for calibration metrics."""
    
    def test_coverage_perfect(self):
        """Test coverage with well-calibrated predictions."""
        np.random.seed(42)
        n = 10000
        
        # Generate from known distribution
        y_pred_mean = np.zeros(n)
        y_pred_std = np.ones(n)
        y_true = np.random.randn(n)  # From N(0,1)
        
        cov = coverage(y_true, y_pred_mean, y_pred_std, [0.5, 0.9, 0.95])
        
        # Should be close to nominal levels
        assert cov[0.5] == pytest.approx(0.5, abs=0.02)
        assert cov[0.9] == pytest.approx(0.9, abs=0.02)
        assert cov[0.95] == pytest.approx(0.95, abs=0.02)
    
    def test_calibration_error_perfect(self):
        """Test calibration error is low for well-calibrated model."""
        np.random.seed(42)
        n = 10000
        
        y_pred_mean = np.zeros(n)
        y_pred_std = np.ones(n)
        y_true = np.random.randn(n)
        
        errors = calibration_error(y_true, y_pred_mean, y_pred_std, [0.5, 0.9])
        
        assert errors[0.5] < 0.03
        assert errors[0.9] < 0.03
    
    def test_pit_uniform(self):
        """Test PIT values are approximately uniform for well-calibrated model."""
        np.random.seed(42)
        n = 10000
        
        y_pred_mean = np.zeros(n)
        y_pred_std = np.ones(n)
        y_true = np.random.randn(n)
        
        pit = pit_values(y_true, y_pred_mean, y_pred_std)
        
        # Should be roughly uniform - check mean ≈ 0.5
        assert pit.mean() == pytest.approx(0.5, abs=0.02)
    
    def test_pit_deviation_low(self):
        """Test PIT deviation is low for well-calibrated model."""
        np.random.seed(42)
        n = 10000
        
        y_pred_mean = np.zeros(n)
        y_pred_std = np.ones(n)
        y_true = np.random.randn(n)
        
        ks_stat = pit_deviation(y_true, y_pred_mean, y_pred_std)
        
        assert ks_stat < 0.05  # Small KS statistic


class TestIntervalScore:
    """Tests for interval score."""
    
    def test_interval_score_in_interval(self):
        """Test interval score when all observations in interval."""
        y_true = np.array([0.0, 0.0, 0.0])
        lower = np.array([-1.0, -1.0, -1.0])
        upper = np.array([1.0, 1.0, 1.0])
        
        score = interval_score(y_true, lower, upper, alpha=0.1)
        
        # Should just be the width (2.0) with no penalties
        assert score == pytest.approx(2.0)
    
    def test_interval_score_penalty(self):
        """Test interval score with observations outside interval."""
        y_true = np.array([5.0])  # Way above interval
        lower = np.array([0.0])
        upper = np.array([1.0])
        
        score = interval_score(y_true, lower, upper, alpha=0.1)
        
        # Should be much larger than width due to penalty
        assert score > 10  # Width + large penalty


class TestEvaluator:
    """Tests for Evaluator class."""
    
    def test_evaluate_basic(self):
        """Test basic evaluation."""
        evaluator = Evaluator()
        
        np.random.seed(42)
        n = 100
        y_true = np.random.randn(n)
        y_pred_mean = y_true + 0.1 * np.random.randn(n)
        y_pred_std = 0.5 * np.ones(n)
        
        result = evaluator.evaluate(y_true, y_pred_mean, y_pred_std)
        
        assert 'rmse' in result.point_metrics
        assert 'nll' in result.probabilistic_metrics
        assert 'coverage' in result.calibration_metrics
    
    def test_evaluate_with_nan(self):
        """Test evaluation handles NaN values."""
        evaluator = Evaluator()
        
        y_true = np.array([1.0, np.nan, 3.0, 4.0])
        y_pred_mean = np.array([1.1, 2.1, np.nan, 4.1])
        y_pred_std = np.array([0.5, 0.5, 0.5, 0.5])
        
        result = evaluator.evaluate(y_true, y_pred_mean, y_pred_std)
        
        # Should still produce valid results
        assert np.isfinite(result.point_metrics['rmse'])
    
    def test_to_dict(self):
        """Test conversion to dictionary."""
        evaluator = Evaluator()
        
        np.random.seed(42)
        n = 100
        y_true = np.random.randn(n)
        y_pred_mean = y_true + 0.1 * np.random.randn(n)
        y_pred_std = 0.5 * np.ones(n)
        
        result = evaluator.evaluate(y_true, y_pred_mean, y_pred_std)
        metrics_dict = result.to_dict()
        
        assert isinstance(metrics_dict, dict)
        assert 'rmse' in metrics_dict
        assert 'nll' in metrics_dict


class TestComputeMetrics:
    """Tests for compute_metrics convenience function."""
    
    def test_compute_metrics_basic(self):
        """Test compute_metrics returns dictionary."""
        np.random.seed(42)
        n = 100
        y_true = np.random.randn(n)
        y_pred_mean = y_true + 0.1 * np.random.randn(n)
        y_pred_std = 0.5 * np.ones(n)
        
        metrics = compute_metrics(y_true, y_pred_mean, y_pred_std)
        
        assert isinstance(metrics, dict)
        assert len(metrics) > 5  # Should have multiple metrics


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
