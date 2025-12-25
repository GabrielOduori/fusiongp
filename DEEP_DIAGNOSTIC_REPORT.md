# Deep Diagnostic Report: FusionGP Parameter Convergence Issues

**Date**: 2025-12-11
**Issue**: Model achieving negative R² (-0.1295), parameters hitting bounds

---

## Executive Summary

The model is **severely underfitting** due to **inappropriate noise bounds** that prevent learning. Key finding: **The noise lower bound of 1.0 is too restrictive for normalized data**, forcing all noise parameters to their minimum values and preventing proper multi-source fusion.

---

## 1. Data Characteristics (From Diagnostic)

### Normalized Target Statistics
All data sources are properly normalized with:
- **Mean ≈ 0.0**
- **Standard deviation ≈ 1.0**

| Source | Count (train) | Mean | Std | Range |
|--------|---------------|------|-----|-------|
| EPA | 328,229 | 0.0000 | 1.0000 | [-2.32, 1.73] |
| Low-cost | 328,229 | 0.0000 | 1.0000 | [-1.66, 1.66] |
| Satellite | 328,229 | -0.0000 | 1.0000 | [-3.01, 7.42] |

### Spatial Extent
- Coordinates normalized to [0, 1] × [0, 1]
- Full spatial coverage
- **Missing temporal dimension in test data** (only 2D coords, not 3D)

### Estimated Noise Levels (From Nearby Observations)
Using spatial neighbors to estimate observation noise in normalized space:

| Source | Estimated Noise σ |
|--------|-------------------|
| EPA | **0.99** |
| Low-cost | **0.96** |
| Satellite | **0.97** |

**Critical Finding**: All sources have noise levels around σ ≈ 1.0 in normalized space, which is similar to the signal standard deviation (also 1.0). This means:
- Signal-to-noise ratio ≈ 1.0 (moderate noise)
- The current noise lower bound of 1.0 prevents the model from learning below this level
- Model cannot learn appropriate noise levels

---

## 2. Root Cause Analysis

### Issue #1: Noise Bound Too Restrictive ⚠️ **CRITICAL**

**Location**: [likelihoods.py:115](fusiongp/src/models/likelihoods.py#L115)

```python
noise_bounds: Tuple[float, float] = (1.0, 100.0)
```

**Problem**:
1. All learned noise values are exactly 1.0 (at the lower bound)
2. Empirical noise estimates are ~0.96-0.99
3. Model wants to learn noise < 1.0 but cannot
4. This prevents proper likelihood estimation and data fusion

**Impact**:
- Forces model to assume 100% noise floor relative to normalized signal
- Cannot distinguish between reliable (EPA) and less reliable (low-cost) sources
- Likelihood becomes uninformative
- Model defaults to over-smoothed prior, ignoring data

**Evidence**:
```
Learned hyperparameters:
  EPA noise: 1.0000        ← AT BOUND
  Low-cost noise: 1.0000   ← AT BOUND
  Satellite noise: 1.0000  ← AT BOUND
```

---

### Issue #2: Calibration Slope at Lower Bound ⚠️ **MAJOR**

**Location**: [likelihoods.py:219](fusiongp/src/models/likelihoods.py#L219)

```python
return self.raw_lc_slope.exp().clamp(0.1, 10.0)
```

**Problem**:
- Learned slope = 0.1 (exactly at lower bound)
- Model is suppressing low-cost sensor contributions as much as possible
- Suggests either: (a) bound too restrictive, or (b) low-cost data fundamentally misaligned

**Evidence**:
```
Calibration slope: 0.1000  ← AT BOUND
```

**Interpretation**:
Given that EPA and low-cost have similar means (both 0.0) and similar std (both 1.0), we'd expect slope ≈ 1.0. The fact that the model learns slope = 0.1 suggests:
1. The bound prevents it from learning lower values
2. OR the low-cost data is so noisy/misaligned that the model wants to ignore it

---

### Issue #3: Over-Smoothing from High Lengthscales

**Learned spatial lengthscale**: 0.7

**Problem**:
- In normalized coordinates [0, 1], lengthscale of 0.7 means points up to 70% of the domain apart are highly correlated
- This causes severe over-smoothing
- Model predicts nearly constant values across space

**Impact**:
- Loses spatial variation
- Cannot capture local pollution patterns
- Contributes to poor R²

---

## 3. Why Negative R²?

**R² = -0.1295** means the model performs **13% worse** than simply predicting the mean.

**Mechanism**:
1. **Noise bounds force σ_obs = 1.0 minimum**
2. **High noise → low data weighting in posterior**
3. **Model trusts prior (constant mean) more than data**
4. **High lengthscales → further over-smoothing**
5. **Result: Nearly constant predictions that ignore spatial/temporal variation**

**Formula**:
```
Posterior variance = [K_prior^{-1} + Σ_noise^{-1}]^{-1}
```

When σ_noise is large (≥ 1.0):
- Σ_noise^{-1} is small
- Data has little influence on posterior
- Model defaults to smoothed prior

---

## 4. Why Parameters Hit Bounds

### Noise Parameters
The optimization wants to:
1. Increase data likelihood by fitting observations better
2. Reduce noise to increase data weight in posterior
3. **BUT**: Clamped at 1.0, preventing further reduction

### Calibration Slope
The optimization finds that:
1. Low-cost data is inconsistent with EPA/satellite
2. Wants to downweight low-cost by reducing slope
3. **BUT**: Clamped at 0.1, so settles for minimal contribution

---

## 5. Recommended Fixes

### Fix #1: Reduce Noise Lower Bound ⭐ **HIGHEST PRIORITY**

**Change** [likelihoods.py:115](fusiongp/src/models/likelihoods.py#L115):

```python
# OLD
noise_bounds: Tuple[float, float] = (1.0, 100.0)

# NEW
noise_bounds: Tuple[float, float] = (0.01, 100.0)
```

**Rationale**:
- Allows noise to go as low as 0.01 (1% of signal std)
- Given empirical estimates around 0.96-0.99, model should learn appropriate values
- Still prevents numerical issues (noise > 0.01)

**Expected Impact**:
- Noise will learn values around 0.5-1.0
- Data will have more influence on posterior
- R² should become positive

---

### Fix #2: Relax Calibration Bounds

**Change** [likelihoods.py:219](fusiongp/src/models/likelihoods.py#L219):

```python
# OLD
return self.raw_lc_slope.exp().clamp(0.1, 10.0)

# NEW
return self.raw_lc_slope.exp().clamp(0.01, 10.0)
```

**Rationale**:
- If low-cost truly needs slope < 0.1, allow it
- Model will naturally regularize via ELBO
- Can diagnose if low-cost data is useful

---

### Fix #3: Better Lengthscale Initialization

**Change test script initialization**:

```python
initial_lengthscales={
    "spatial_x": 0.05,  # Smaller for less smoothing
    "spatial_y": 0.05,
    "temporal": 0.05,
}
```

**Rationale**:
- Start with tighter spatial correlation
- Let model expand if needed (easier than contracting)
- In [0,1] normalized space, 0.05 = 5% of domain

---

### Fix #4: Better Noise Initialization

**Change test script**:

```python
initial_noise={
    "epa": 0.5,        # Lower starting point
    "low_cost": 0.8,   # Expect higher noise
    "satellite": 0.6,  # Medium noise
}
```

**Rationale**:
- Start closer to empirical estimates
- Prevents starting at bound
- Allows gradient descent to refine

---

## 6. Alternative Hypothesis: Data Quality Issues

If fixes #1-4 don't resolve the issue, consider:

### Hypothesis A: Low-Cost Data is Corrupted
**Evidence**:
- Slope wants to go to 0.1 (suppress low-cost)
- Low-cost has similar statistics to EPA (suspicious if it's supposed to be biased)

**Test**: Train model with only EPA + satellite (exclude low-cost)

### Hypothesis B: Temporal Information Missing
**Evidence**:
- Coordinates are 2D (lat, lon) but model expects 3D (lat, lon, time)
- Missing temporal correlation

**Test**: Verify timestamp data is included in FusionDataset

### Hypothesis C: Too Few Inducing Points
**Evidence**:
- Using only 100-200 inducing points
- May be insufficient for complex spatial field

**Test**: Increase to 500-800 inducing points

---

## 7. Validation Plan

After applying Fix #1 (noise bounds):

1. **Run test_fixes.py** and check:
   - ✓ R² > 0.0 (should be positive)
   - ✓ RMSE < baseline (should beat 11.55)
   - ✓ Noise values in range [0.3, 1.5]
   - ✓ Calibration slope in range [0.5, 2.0]

2. **Inspect learned parameters**:
   ```python
   params = model.get_hyperparameters()
   print(params)
   ```
   - EPA noise should be lowest (~0.5-0.8)
   - Satellite noise medium (~0.6-1.0)
   - Low-cost noise highest (~0.8-1.5)

3. **Plot predictions vs truth**:
   - Should show correlation
   - Residuals should be normally distributed

---

## 8. Implementation Priority

1. **IMMEDIATE**: Fix noise bounds (likelihoods.py:115)
2. **HIGH**: Fix calibration bounds (likelihoods.py:219)
3. **MEDIUM**: Better initialization (test scripts)
4. **LOW**: Investigate data quality issues (if still underperforming)

---

## 9. Code Changes Required

### File: `fusiongp/src/models/likelihoods.py`

```python
# Line 115
noise_bounds: Tuple[float, float] = (0.01, 100.0),  # Changed from (1.0, 100.0)

# Line 219
return self.raw_lc_slope.exp().clamp(0.01, 10.0)  # Changed from (0.1, 10.0)
```

### File: `fusiongp/experiments/test_fixes.py`

```python
# Lines 56-65 (better initialization)
initial_lengthscales={
    "spatial_x": 0.05,
    "spatial_y": 0.05,
    "temporal": 0.05,
},
initial_noise={
    "epa": 0.5,
    "low_cost": 0.8,
    "satellite": 0.6,
},
```

---

## 10. Expected Results After Fixes

| Metric | Before | After (Expected) |
|--------|--------|------------------|
| R² | -0.1295 | +0.3 to +0.7 |
| RMSE | 12.27 µg/m³ | 7-10 µg/m³ |
| EPA Noise | 1.00 (bound) | 0.5-0.8 |
| LC Noise | 1.00 (bound) | 0.8-1.2 |
| Satellite Noise | 1.00 (bound) | 0.6-1.0 |
| LC Slope | 0.10 (bound) | 0.7-1.3 |

---

## References

1. **Noise estimation**: Based on variogram analysis of nearby observations
2. **Normalized data**: Signal and noise both in units of original std
3. **GP posterior**: Balances prior and likelihood based on noise levels
4. **Bounds**: Should allow physically plausible values, not constrain optimization

---

*End of Report*
