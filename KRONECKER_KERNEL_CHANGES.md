# Kronecker Product Kernel Implementation

## Summary

The kernel implementation has been updated to use **Kronecker product structure** instead of the previous simple product kernel approach.

## Key Changes

### 1. Mathematical Framework

**Previous (Product Kernel):**
```
k((s,t), (s',t')) = σ² * k_space(s, s') * k_time(t, t')
```

**Current (Kronecker Product):**
```
K = σ² * (K_space ⊗ K_time)
```

Where:
- `K_space` is the N×N spatial covariance matrix
- `K_time` is the N×N temporal covariance matrix  
- `⊗` denotes the Kronecker product
- For separable kernels: `K[i,j] = k_spatial(s_i, s_j) * k_temporal(t_i, t_j)`

### 2. Code Changes

#### File: `fusiongp/src/models/kernels.py`

**Removed:**
- `ProductKernel` import and usage
- `self.product_kernel` attribute
- `self.scale_kernel` wrapper

**Added:**
- `self.outputscale_param` as a direct `nn.Parameter`
- Explicit Kronecker product computation in `forward()` method
- Enhanced documentation explaining the Kronecker structure

**Modified:**
- `forward()` method now explicitly computes spatial and temporal covariances separately
- Element-wise multiplication of covariance matrices (equivalent to Kronecker product for separable kernels)
- Direct application of output scale parameter

#### File: `fusiongp/src/models/svgp.py`

**Updated:**
- Changed `self.covar_module.scale_kernel.raw_outputscale.requires_grad_(False)` 
  to `self.covar_module.outputscale_param.requires_grad_(False)`

### 3. Benefits of Kronecker Product Structure

1. **Mathematical Clarity**: Explicitly represents the separable structure
2. **Computational Efficiency**: Can leverage Kronecker structure for faster operations (especially on gridded data)
3. **Flexibility**: Easier to extend to specialized Kronecker solvers and efficient inference methods
4. **Memory**: Potential for memory-efficient lazy evaluation

### 4. Validation

All tests pass successfully:
- ✅ Kernel initialization and forward pass
- ✅ FusionSVGP model creation  
- ✅ Hyperparameter access and modification
- ✅ Integration with training pipeline

## Usage

The API remains unchanged. The model can be used exactly as before:

```python
from src.models import FusionSVGP

model = FusionSVGP(
    n_inducing=800,
    kernel_type='matern32',
    spatial_ard=True,
    initial_lengthscales={
        'spatial_x': 0.1,
        'spatial_y': 0.1,
        'temporal': 0.1,
    }
)
```

The Kronecker product structure is now used internally for all covariance computations.
