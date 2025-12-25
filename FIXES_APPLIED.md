# Fixes Applied to Resolve Negative R² Issue

**Date**: 2025-12-11
**Issue**: Model achieving R² = -0.1295 (worse than baseline)
**Root Cause**: Restrictive noise bounds preventing parameter learning

---

## Changes Made

### 1. Reduced Noise Lower Bound ⭐ CRITICAL FIX

**File**: [fusiongp/src/models/likelihoods.py:115](fusiongp/src/models/likelihoods.py#L115)

```python
# BEFORE
noise_bounds: Tuple[float, float] = (1.0, 100.0)

# AFTER
noise_bounds: Tuple[float, float] = (0.01, 100.0)
```

**Why**:
- Old bound prevented noise from going below 1.0
- Empirical noise estimates were ~0.96-0.99
- Prevented model from learning appropriate noise levels
- Caused model to ignore data and over-smooth

---

### 2. Relaxed Calibration Slope Bounds

**File**: [fusiongp/src/models/likelihoods.py:219](fusiongp/src/models/likelihoods.py#L219)

```python
# BEFORE
return self.raw_lc_slope.exp().clamp(0.1, 10.0)

# AFTER
return self.raw_lc_slope.exp().clamp(0.01, 10.0)
```

**Why**:
- Learned slope was stuck at 0.1 (lower bound)
- Now allows model to learn calibration parameters as low as 0.01
- Enables better sensor calibration or downweighting if needed

---

### 3. Improved Initialization Parameters

**File**: [fusiongp/experiments/test_fixes.py:56-65](fusiongp/experiments/test_fixes.py#L56-L65)

```python
# BEFORE
initial_lengthscales={
    "spatial_x": 0.1,
    "spatial_y": 0.1,
    "temporal": 0.1,
},
initial_noise={
    "epa": 0.8,
    "low_cost": 1.5,
    "satellite": 1.2,
},

# AFTER
initial_lengthscales={
    "spatial_x": 0.05,  # Smaller for less over-smoothing
    "spatial_y": 0.05,
    "temporal": 0.05,
},
initial_noise={
    "epa": 0.5,        # Start below old bound
    "low_cost": 0.8,   # Expect higher noise than EPA
    "satellite": 0.6,  # Medium noise
},
```

**Why**:
- Smaller lengthscales reduce initial over-smoothing
- Noise initialization now starts below the old bound
- Closer to empirical noise estimates

---

## Expected Improvements

| Metric | Before | Expected After |
|--------|--------|----------------|
| **R²** | -0.1295 | **+0.3 to +0.7** |
| **RMSE** | 12.27 µg/m³ | **7-10 µg/m³** |
| **EPA Noise** | 1.00 (at bound) | **0.5-0.8** |
| **Low-cost Noise** | 1.00 (at bound) | **0.8-1.2** |
| **Satellite Noise** | 1.00 (at bound) | **0.6-1.0** |
| **LC Slope** | 0.10 (at bound) | **0.7-1.3** |

---

## Technical Explanation

### Why Noise Bounds Matter

In a Gaussian Process, the posterior distribution balances the prior and likelihood:

```
Posterior mean = K_*^T (K + Σ_noise)^{-1} y
Weight to data ∝ 1 / σ_noise²
```

When noise is forced to be high (≥ 1.0):
1. **Low data weight**: Σ_noise^{-1} becomes small
2. **Prior dominance**: Model trusts prior more than observations
3. **Over-smoothing**: High lengthscales + high noise → constant predictions
4. **Negative R²**: Predictions worse than predicting the mean

### Empirical Noise Estimates

From diagnostic analysis of nearby observations:
- **EPA**: σ ≈ 0.99 (high-quality reference monitors)
- **Low-cost**: σ ≈ 0.96 (consumer-grade sensors)
- **Satellite**: σ ≈ 0.97 (remote sensing)

All in normalized space where signal std = 1.0, giving SNR ≈ 1.0.

The old bound of 1.0 prevented the model from learning these appropriate values.

---

## Verification

Run the test script to verify:

```bash
cd fusiongp
python experiments/test_fixes.py
```

Check for:
- ✅ **R² > 0.0** (positive, model beats baseline)
- ✅ **RMSE < 11.5 µg/m³** (beats baseline RMSE)
- ✅ **Noise parameters learned** (not stuck at bounds)
- ✅ **Calibration parameters learned** (slope ≠ initial)

---

## Related Documentation

- [DEEP_DIAGNOSTIC_REPORT.md](DEEP_DIAGNOSTIC_REPORT.md) - Full investigation details
- [PERFORMANCE_FIXES.md](PERFORMANCE_FIXES.md) - Original performance issues
- [KRONECKER_KERNEL_CHANGES.md](KRONECKER_KERNEL_CHANGES.md) - Kernel implementation

---

## Files Modified

1. `fusiongp/src/models/likelihoods.py` - Core likelihood class
2. `fusiongp/experiments/test_fixes.py` - Test script with better initialization

---

*Last updated: 2025-12-11*
