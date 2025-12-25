# FusionGP Performance Issues and Fixes

## Problem Summary

The FusionGP model was producing poor performance metrics:
- **RMSE**: 11.78 µg/m³
- **MAE**: 10.14 µg/m³
- **R²**: **-0.041** (negative R² indicates worse than baseline!)
- **Bias**: -0.10 µg/m³

A negative R² score means the model performs worse than simply predicting the mean value for all test points.

## Root Causes Identified

### 1. **Observation Noise in Evaluation** (Critical Issue)
**File**: `fusiongp/experiments/reproduce_paper.py:372`

**Problem**:
```python
predictor = Predictor(model, scalers, include_observation_noise=True, noise_source='epa')
```

**Impact**: While this doesn't affect the mean predictions, it inflates the uncertainty estimates. However, the diagnostic revealed this was NOT the main issue.

**Fix Applied**:
```python
predictor = Predictor(model, scalers, include_observation_noise=False)
```

**Rationale**: For evaluation, we want to measure the model's ability to predict the **latent NO₂ concentration field**, not noisy observations. The latent function represents the true underlying signal.

---

### 2. **Disabled Hyperparameter Learning** (Major Issue)
**File**: `fusiongp/experiments/reproduce_paper.py:79-81`

**Problem**:
```python
"learn_kernel_hyperparams": False,  # Fixed for reproducibility
"learn_noise": False,
"learn_calibration": False,
```

**Impact**:
- Fixed lengthscales prevent the model from adapting spatial/temporal correlations to the data
- Fixed noise levels may be inappropriate for the actual noise characteristics
- Prevents calibration of low-cost sensors (critical for multi-source fusion!)

**Diagnostic Evidence**: The diagnostic showed noise parameters stuck at 1.0 and calibration parameters at identity (slope=1.0, intercept=0.0), indicating no learning occurred.

**Fix Applied**:
```python
"learn_kernel_hyperparams": True,  # Enable learning for better fit
"learn_noise": True,  # Learn noise levels from data
"learn_calibration": True,  # Learn calibration for low-cost sensors
```

**Rationale**: The model MUST learn these parameters from data to:
1. Discover appropriate spatial/temporal correlation scales
2. Estimate source-specific noise levels
3. Calibrate biased low-cost sensor measurements

---

### 3. **Low Learning Rate** (Performance Issue)
**File**: `fusiongp/experiments/reproduce_paper.py:96`

**Problem**:
```python
"learning_rate": 0.0005,  # Too low
```

**Impact**: With 300 epochs, a learning rate of 0.0005 may not provide sufficient optimization steps for convergence, especially with 800 inducing points and many parameters to learn.

**Fix Applied**:
```python
"learning_rate": 0.01,  # Higher learning rate for faster convergence
```

**Rationale**: A learning rate of 0.01 with Adam optimizer is standard for GP models and should enable better convergence within 300 epochs.

---

### 4. **Model Underfitting** (Fundamental Issue)
**Evidence from Diagnostic**:
```
BASELINE (constant mean = 32.08):
  RMSE: 11.55
  R²:   0.0000 (by definition)

Model predictions:
  RMSE: 11.86
  R²:   -0.0542
```

**Problem**: The model's predictions have **higher error** than simply predicting the mean, indicating severe underfitting.

**Possible Causes**:
1. Insufficient inducing points (100 in diagnostic test)
2. Poor inducing point initialization
3. Inappropriate hyperparameter initialization
4. Training not converging
5. KL divergence term dominating the ELBO

**Fix Applied**: Enabled hyperparameter learning and increased learning rate.

---

## Changes Made

### Modified Files

1. **`fusiongp/experiments/reproduce_paper.py`**
   - Line 372: Changed `include_observation_noise=True` → `False`
   - Line 79: Changed `learn_kernel_hyperparams: False` → `True`
   - Line 80: Changed `learn_noise: False` → `True`
   - Line 81: Changed `learn_calibration: False` → `True`
   - Line 96: Changed `learning_rate: 0.0005` → `0.01`

2. **`fusiongp/src/models/svgp.py`**
   - Lines 353-402: Enhanced ELBO documentation to clarify mini-batch scaling
   - Added `n_data` parameter (optional) for future scaling flexibility

### New Files Created

1. **`fusiongp/experiments/diagnose_model.py`**
   - Diagnostic script to investigate model performance
   - Compares predictions with/without observation noise
   - Computes baseline metrics for comparison
   - Reports learned hyperparameters

---

## Expected Improvements

After these fixes, we expect:

1. **Positive R² score** - Model should outperform mean baseline
2. **Lower RMSE/MAE** - Better predictions through learned parameters
3. **Learned Hyperparameters** will show:
   - Spatial lengthscales adapted to data correlation structure
   - Appropriate noise levels for each source
   - Calibration parameters for low-cost sensors (slope ≠ 1.0, intercept ≠ 0.0)

---

## How to Verify Fixes

Run the experiment again:

```bash
cd fusiongp
python experiments/reproduce_paper.py
```

Check the results for:
- **R² > 0.0** (should be positive, ideally > 0.5 for good fit)
- **RMSE < 11.5** (should beat baseline RMSE of 11.55)
- **Learned hyperparameters** different from initial values

Run diagnostic for comparison:

```bash
python experiments/diagnose_model.py
```

---

## Technical Background

### Why Observation Noise Matters

In Gaussian Processes, we model:
- **Latent function**: `f(x) ~ GP(m(x), k(x,x'))`  - The true underlying signal
- **Observations**: `y = f(x) + ε`, where `ε ~ N(0, σ²)` - Noisy measurements

For **prediction**:
- **With noise** (`include_observation_noise=True`): Predicts `y*` including observation noise
  - Variance: `Var[y*] = Var[f*] + σ²`
  - Use case: Simulating new observations

- **Without noise** (`include_observation_noise=False`): Predicts latent function `f*`
  - Variance: `Var[f*]` only
  - Use case: **Evaluation** - measuring model's ability to recover true signal

For evaluation metrics (RMSE, R², etc.), we want to measure how well the model predicts the **true latent field**, not how well it predicts noisy observations.

### Why Hyperparameter Learning Matters

Fixed hyperparameters assume we know:
1. **Spatial correlation scale** - How far apart do measurements need to be before they become uncorrelated?
2. **Temporal correlation scale** - How quickly does pollution change over time?
3. **Noise levels** - How reliable is each data source?
4. **Sensor calibration** - What is the bias/gain of low-cost sensors vs. reference monitors?

In reality, we **don't know** these values a priori. The model must learn them from data through maximum likelihood (ELBO maximization).

### Multi-Source Fusion

FusionGP combines three data sources:
- **EPA monitors**: High accuracy, sparse coverage
- **Low-cost sensors**: Lower accuracy, dense coverage, **biased** (need calibration!)
- **Satellite**: Moderate accuracy, complete spatial coverage, coarse resolution

The key innovation is learning **source-specific likelihoods**:
- Different noise levels per source
- Calibration parameters (slope, intercept) for low-cost sensors
- Single shared latent GP represents true NO₂ field

Disabling `learn_calibration` prevents the model from correcting systematic biases in low-cost sensors, severely limiting fusion performance.

---

## Next Steps

1. **Re-run experiment** with fixes applied
2. **Compare metrics** with baseline and previous results
3. **Inspect learned hyperparameters** to ensure they're reasonable:
   - Spatial lengthscales: Should be ~0.01-1.0 (depending on coordinate normalization)
   - Temporal lengthscales: Should be ~0.01-0.5
   - Noise std: EPA < Satellite < Low-cost
   - Calibration: Slope ≈ 0.8-1.2, intercept ≠ 0

4. **If still underperforming**, investigate:
   - Increase number of inducing points (currently 800)
   - Try different kernel types (RBF, Matérn-5/2)
   - Adjust initial lengthscales
   - Check for numerical stability issues
   - Verify data preprocessing (normalization)

---

## References

1. Hensman, J., Fusi, N., & Lawrence, N. D. (2013). Gaussian processes for big data. UAI.
2. Titsias, M. (2009). Variational learning of inducing variables in sparse Gaussian processes. AISTATS.
3. GPyTorch documentation: https://gpytorch.ai/

---

*Last updated: 2025-12-11*
