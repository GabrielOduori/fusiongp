# FusionGP Investigation Summary

**Date**: 2025-12-11
**Issue**: Multi-source sensor data fusion model achieving negative R² (worse than baseline)

---

## Problem Statement

FusionGP model for air quality prediction was producing:
- **R² = -0.1295** (negative = worse than predicting mean)
- **RMSE = 12.27 µg/m³** (vs baseline 11.55 µg/m³)
- Hyperparameters stuck at initialization or bounds

**Data sources being fused:**
- EPA monitors (gold standard, sparse, high accuracy)
- Low-cost sensors (dense, lower accuracy, potentially biased)
- Satellite data (complete spatial coverage, moderate accuracy)

---

## Investigation Timeline

### Stage 1: Initial Diagnosis

**Found**: Parameters not learning at all
- Noise stuck at 1.0
- Calibration at identity (slope=1.0, intercept=0.0)
- Lengthscales not updating properly

**Root cause**: Hyperparameter learning was DISABLED in experiment config
- `learn_kernel_hyperparams: False`
- `learn_noise: False`
- `learn_calibration: False`

**Fix Applied**: Enable all learning flags
**Result**: Parameters started learning BUT still hitting bounds

---

### Stage 2: Noise Bound Issue

**Found**: All noise parameters stuck at exactly 1.0
- This was the LOWER bound: `noise_bounds = (1.0, 100.0)`
- Model gradient indicated it wanted to go LOWER
- Calibration slope stuck at 0.1 (also hitting lower bound)

**Analysis**:
- Data is normalized (mean=0, std=1)
- Empirical noise estimates from spatial neighbors: ~0.96-0.99
- Lower bound of 1.0 prevents learning appropriate noise levels
- High noise → model ignores data → over-smooths → negative R²

**Fix Applied**: Reduce noise lower bound to 0.01
**Result**: Noise parameters learned (EPA=0.60, Sat=0.01) BUT R² still negative (-0.055)

---

### Stage 3: Bounded vs Unbounded

**Observation**: Even with looser bounds, parameters hitting limits:
- Satellite noise = 0.01 (new lower bound)
- Calibration slope = 0.01 (new lower bound)

**Interpretation**: Model wants to COMPLETELY IGNORE low-cost and satellite data

**Hypothesis**: Either:
1. Bounds still too restrictive
2. Data sources fundamentally misaligned (don't correlate with EPA)
3. Model architecture/initialization issues

**Current Experiment**: Remove ALL bounds
- Let model learn ANY values (noise can → 0 or → ∞)
- See what it naturally converges to
- Will reveal if data fusion is fundamentally viable

---

## Key Files Modified

### 1. `fusiongp/src/models/likelihoods.py`
**Changes**:
- Line 115: `noise_bounds=(1.0, 100.0)` → `(0.01, 100.0)` → NO CLAMPING
- Line 219: Calibration slope clamp `(0.1, 10.0)` → `(0.01, 10.0)` → NO CLAMPING
- Line 224: Calibration intercept clamp `(-50, 50)` → NO CLAMPING

### 2. `fusiongp/experiments/test_fixes.py`
**Changes**:
- Enabled hyperparameter learning
- Better initialization (smaller lengthscales, lower noise)

### 3. New Files Created
- `deep_diagnostic.py` - Comprehensive model diagnostics
- `test_unbounded.py` - Test with no parameter bounds
- `DEEP_DIAGNOSTIC_REPORT.md` - Full investigation details
- `FIXES_APPLIED.md` - Summary of applied fixes
- `UNBOUNDED_EXPERIMENT.md` - Current experiment documentation

---

## Current Status

**Experiment Running**: Unbounded parameter test
- Training with NO bounds on any parameters
- Will reveal natural parameter values model wants to learn
- Expected completion: ~20 minutes

**Best validation loss so far**: 4.0790 (improving!)

---

## Possible Outcomes

### ✅ Best Case: R² becomes positive
→ Bounds were the issue
→ Set appropriate bounds based on learned values
→ Multi-source fusion working

### ⚠️ Medium Case: R² still negative but parameters stabilize
→ Model learns reasonable values but still can't beat baseline
→ Suggests fundamental limitations:
  - Not enough spatial/temporal coverage from EPA alone
  - Low-cost/satellite don't add predictive value
  - Need different model architecture

### ❌ Worst Case: Parameters explode or collapse
→ Numerical instability without bounds
→ Need some regularization
→ Try wider (but not unbounded) constraints

---

## Next Steps (Pending Results)

1. **If unbounded succeeds** → Set sensible bounds around learned values
2. **If still fails** → Investigate data correlation between sources
3. **Alternative**: Train EPA-only model to establish baseline GP performance

---

## Technical Insights

### Why Noise Bounds Matter in Normalized Data

In normalized space where σ_signal = 1.0:
- Noise σ = 0.5 means SNR = 2.0 (good)
- Noise σ = 1.0 means SNR = 1.0 (50% noise)
- Noise σ = 2.0 means SNR = 0.5 (mostly noise)

Lower bound of 1.0 forced SNR ≤ 1.0 for all sources, making data barely useful.

### GP Posterior Weight

```
Weight to data ∝ 1/σ_noise²
```

When σ_noise = 1.0:
- Data weight = 1.0
- Prior weight = 1.0
- Equal influence → heavy smoothing

When σ_noise = 0.5:
- Data weight = 4.0
- Model trusts data 4x more than prior
- Less smoothing → better fit

---

*Last updated: 2025-12-11 (experiment in progress)*
