# Unbounded Parameter Experiment

**Date**: 2025-12-11
**Goal**: Remove ALL parameter bounds to see what the model actually wants to learn

---

## Motivation

Previous experiments showed parameters hitting bounds:
- Bounded (0.01, 100): Satellite noise → 0.01, Calibration slope → 0.01
- This suggested model wanted to ignore non-EPA data sources

**Hypothesis**: Bounds are preventing the model from finding optimal parameters for data fusion.

---

## Changes Made

### File: `fusiongp/src/models/likelihoods.py`

**Removed ALL clamping:**

1. **Noise std** (Line 197):
   ```python
   # OLD
   noise = self.raw_noise[source].exp().clamp(self.noise_bounds[0], self.noise_bounds[1])

   # NEW
   noise = self.raw_noise[source].exp()  # NO CLAMPING
   ```

2. **LC slope** (Line 216):
   ```python
   # OLD
   return self.raw_lc_slope.exp().clamp(0.01, 10.0)

   # NEW
   return self.raw_lc_slope.exp()  # NO CLAMPING
   ```

3. **LC intercept** (Line 221):
   ```python
   # OLD
   return self.raw_lc_intercept.clamp(-50.0, 50.0)

   # NEW
   return self.raw_lc_intercept  # NO CLAMPING
   ```

---

## What This Reveals

By removing bounds, we can see:

1. **What noise levels does the model naturally learn?**
   - If satellite noise → 0: Satellite data is perfect/reliable
   - If satellite noise → ∞: Satellite data is useless
   - If noise ~0.5-2.0: Data source is useful with reasonable uncertainty

2. **How does the model want to calibrate low-cost sensors?**
   - Slope < 0.1: Model wants to strongly suppress low-cost data
   - Slope ~1.0: Low-cost data aligns well with EPA
   - Slope > 2.0: Low-cost data needs amplification

3. **Can the model achieve positive R² when unconstrained?**
   - If YES: Bounds were the problem → tune bounds appropriately
   - If NO: Data fusion fundamentally not working → investigate data quality

---

## Expected Outcomes

### Scenario A: Unbounded helps (R² becomes positive)
→ Previous bounds were too restrictive
→ Tune bounds based on learned values
→ Data fusion IS working

### Scenario B: Still negative R² but different parameters
→ Model learns extreme values (noise → 0 or → ∞)
→ Suggests fundamental data quality/alignment issues
→ May need to examine individual data sources

### Scenario C: Numerical instability
→ Parameters explode/collapse
→ Some regularization needed
→ Try softer constraints (wider bounds)

---

## Running

```bash
cd fusiongp
python experiments/test_unbounded.py
```

---

*Experiment in progress...*
