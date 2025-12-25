"""
Spatio-temporal kernel definitions for FusionGP.

This module provides kernel (covariance function) implementations for
modeling spatial and temporal correlations in the latent NO₂ field.

The main kernel is a separable product of spatial and temporal components:
    k((s,t), (s',t')) = k_space(s, s') * k_time(t, t')

This separability assumption is common in environmental modeling and allows
efficient computation while capturing distinct spatial and temporal patterns.

References
----------
.. [1] Williams, C. K., & Rasmussen, C. E. (2006). 
       Gaussian processes for machine learning. MIT Press.
.. [2] Hamelijnck, O., et al. (2021). Spatio-temporal variational 
       Gaussian processes. NeurIPS.

Example
-------
>>> kernel = SpatioTemporalKernel(
...     spatial_kernel_type='matern32',
...     temporal_kernel_type='matern32',
...     spatial_ard=True
... )
>>> # Evaluate covariance
>>> K = kernel(x1, x2)
"""

from __future__ import annotations

import logging
from typing import Optional, Literal

import torch
import torch.nn as nn
import gpytorch
from gpytorch.kernels import (
    Kernel,
    MaternKernel,
    RBFKernel,
    ScaleKernel,
)

logger = logging.getLogger(__name__)


# Type alias for kernel types
KernelType = Literal["matern12", "matern32", "matern52", "rbf"]


def create_base_kernel(
    kernel_type: KernelType,
    active_dims: Optional[torch.Tensor] = None,
    ard_num_dims: Optional[int] = None,
) -> Kernel:
    """
    Create a base kernel of the specified type.
    
    Parameters
    ----------
    kernel_type : str
        Type of kernel: 'matern12', 'matern32', 'matern52', or 'rbf'.
    active_dims : torch.Tensor, optional
        Indices of dimensions this kernel operates on.
    ard_num_dims : int, optional
        Number of ARD dimensions (for separate lengthscales per dimension).
        
    Returns
    -------
    Kernel
        GPyTorch kernel instance.
        
    Raises
    ------
    ValueError
        If kernel_type is not recognized.
        
    Example
    -------
    >>> kernel = create_base_kernel('matern32', active_dims=torch.tensor([0, 1]))
    """
    kernel_map = {
        'matern12': lambda: MaternKernel(
            nu=0.5, 
            active_dims=active_dims,
            ard_num_dims=ard_num_dims
        ),
        'matern32': lambda: MaternKernel(
            nu=1.5, 
            active_dims=active_dims,
            ard_num_dims=ard_num_dims
        ),
        'matern52': lambda: MaternKernel(
            nu=2.5, 
            active_dims=active_dims,
            ard_num_dims=ard_num_dims
        ),
        'rbf': lambda: RBFKernel(
            active_dims=active_dims,
            ard_num_dims=ard_num_dims
        ),
    }
    
    if kernel_type not in kernel_map:
        raise ValueError(
            f"Unknown kernel type: {kernel_type}. "
            f"Must be one of: {list(kernel_map.keys())}"
        )
    
    return kernel_map[kernel_type]()


class SpatioTemporalKernel(Kernel):
    """
    Separable spatio-temporal kernel using Kronecker product structure.

    This kernel computes covariance using the Kronecker product of spatial
    and temporal covariance matrices, exploiting the separability assumption:

        K = σ² * (K_space ⊗ K_time)

    where:
    - K_space is the spatial covariance matrix
    - K_time is the temporal covariance matrix
    - ⊗ denotes the Kronecker product
    - σ² is the output variance (scaling)

    The Kronecker product structure enables efficient computations and is
    particularly useful for gridded spatio-temporal data.
    
    Parameters
    ----------
    spatial_kernel_type : str, default='matern32'
        Type of spatial kernel: 'matern12', 'matern32', 'matern52', 'rbf'.
    temporal_kernel_type : str, default='matern32'
        Type of temporal kernel.
    spatial_ard : bool, default=True
        Whether to use ARD (separate lengthscales for lat/lon).
    initial_spatial_lengthscale : float or tuple, default=0.1
        Initial spatial lengthscale(s). If ARD, can be tuple (lat_ls, lon_ls).
    initial_temporal_lengthscale : float, default=0.1
        Initial temporal lengthscale.
    initial_outputscale : float, default=1.0
        Initial output variance.
        
    Attributes
    ----------
    spatial_kernel : Kernel
        The spatial component kernel (2D Matérn or RBF).
    temporal_kernel : Kernel
        The temporal component kernel (1D Matérn or RBF).
    outputscale_param : nn.Parameter
        Output variance parameter applied to the Kronecker product.
        
    Example
    -------
    >>> kernel = SpatioTemporalKernel(
    ...     spatial_kernel_type='matern32',
    ...     temporal_kernel_type='matern32',
    ...     spatial_ard=True,
    ...     initial_spatial_lengthscale=(0.05, 0.05),
    ...     initial_temporal_lengthscale=0.1
    ... )
    >>>
    >>> # Input shape: (N, 3) with columns [lat, lon, time]
    >>> x = torch.randn(100, 3)
    >>> # Computes Kronecker product: K = σ² * (K_space ⊗ K_time)
    >>> K = kernel(x, x)
    >>> print(K.shape)
    torch.Size([100, 100])
    """
    
    # Input dimension: [lat, lon, time]
    SPATIAL_DIMS = torch.tensor([0, 1])
    TEMPORAL_DIMS = torch.tensor([2])
    
    def __init__(
        self,
        spatial_kernel_type: KernelType = 'matern32',
        temporal_kernel_type: KernelType = 'matern32',
        spatial_ard: bool = True,
        initial_spatial_lengthscale: float | tuple = 0.1,
        initial_temporal_lengthscale: float = 0.1,
        initial_outputscale: float = 1.0,
        **kwargs,
    ):
        """
        Initialize the SpatioTemporalKernel.
        
        Parameters
        ----------
        spatial_kernel_type : str
            Type of spatial kernel.
        temporal_kernel_type : str
            Type of temporal kernel.
        spatial_ard : bool
            Use ARD for spatial dimensions.
        initial_spatial_lengthscale : float or tuple
            Initial spatial lengthscale(s).
        initial_temporal_lengthscale : float
            Initial temporal lengthscale.
        initial_outputscale : float
            Initial output scale.
        **kwargs
            Additional arguments passed to base Kernel class.
        """
        super().__init__(**kwargs)
        
        self.spatial_ard = spatial_ard
        
        # Create spatial kernel with lengthscale constraints
        ard_dims = 2 if spatial_ard else None
        self.spatial_kernel = create_base_kernel(
            spatial_kernel_type,
            active_dims=self.SPATIAL_DIMS,
            ard_num_dims=ard_dims,
        )
        # Constrain spatial lengthscales to [0.05, 0.7] for normalized [0,1] space
        # This prevents over-smoothing while allowing sufficient smoothing for calibration
        self.spatial_kernel.register_constraint(
            "raw_lengthscale",
            gpytorch.constraints.Interval(0.05, 0.7)
        )

        # Create temporal kernel with constraints
        self.temporal_kernel = create_base_kernel(
            temporal_kernel_type,
            active_dims=self.TEMPORAL_DIMS,
            ard_num_dims=None,
        )
        # Constrain temporal lengthscale to [0.05, 0.7]
        self.temporal_kernel.register_constraint(
            "raw_lengthscale",
            gpytorch.constraints.Interval(0.05, 0.7)
        )

        # Output scaling parameter (applied to full Kronecker product)
        self.outputscale_param = nn.Parameter(torch.tensor(1.0))
        
        # Initialize lengthscales
        self._initialize_lengthscales(
            initial_spatial_lengthscale,
            initial_temporal_lengthscale,
            initial_outputscale,
        )
        
        logger.info(
            f"Created SpatioTemporalKernel: "
            f"spatial={spatial_kernel_type}, temporal={temporal_kernel_type}, "
            f"spatial_ard={spatial_ard}"
        )
    
    def _initialize_lengthscales(
        self,
        spatial_ls: float | tuple,
        temporal_ls: float,
        outputscale: float,
    ) -> None:
        """
        Initialize kernel hyperparameters.
        
        Parameters
        ----------
        spatial_ls : float or tuple
            Spatial lengthscale(s).
        temporal_ls : float
            Temporal lengthscale.
        outputscale : float
            Output scale.
        """
        # Spatial lengthscale(s)
        if isinstance(spatial_ls, (tuple, list)):
            spatial_ls_tensor = torch.tensor(spatial_ls, dtype=torch.float32)
        else:
            if self.spatial_ard:
                spatial_ls_tensor = torch.tensor([spatial_ls, spatial_ls], dtype=torch.float32)
            else:
                spatial_ls_tensor = torch.tensor(spatial_ls, dtype=torch.float32)
        
        self.spatial_kernel.lengthscale = spatial_ls_tensor
        
        # Temporal lengthscale
        self.temporal_kernel.lengthscale = torch.tensor(temporal_ls, dtype=torch.float32)

        # Output scale
        self.outputscale_param.data = torch.tensor(outputscale, dtype=torch.float32)
        
        logger.debug(
            f"Initialized lengthscales: "
            f"spatial={self.spatial_kernel.lengthscale}, "
            f"temporal={self.temporal_kernel.lengthscale}, "
            f"outputscale={self.outputscale_param}"
        )
    
    def forward(
        self,
        x1: torch.Tensor,
        x2: torch.Tensor,
        diag: bool = False,
        **params,
    ) -> torch.Tensor:
        """
        Compute the covariance matrix using Kronecker product structure.

        For inputs with shape (N, 3) where columns are [lat, lon, time],
        this computes: K = σ² * (K_space ⊗ K_time)

        Parameters
        ----------
        x1 : torch.Tensor
            First input, shape (N, 3) with columns [lat, lon, time].
        x2 : torch.Tensor
            Second input, shape (M, 3).
        diag : bool, default=False
            If True, return only diagonal elements (element-wise product).
        **params
            Additional parameters.

        Returns
        -------
        torch.Tensor or gpytorch.lazy.LazyTensor
            Covariance matrix of shape (N, M), or (N,) if diag=True.
        """
        if diag:
            # For diagonal: just element-wise product of spatial and temporal kernels
            spatial_diag = self.spatial_kernel(x1, x2, diag=True, **params)
            temporal_diag = self.temporal_kernel(x1, x2, diag=True, **params)
            return self.outputscale_param * spatial_diag * temporal_diag

        # Compute spatial and temporal covariances separately
        spatial_covar = self.spatial_kernel(x1, x2, diag=False, **params)
        temporal_covar = self.temporal_kernel(x1, x2, diag=False, **params)

        # Kronecker product structure: K = K_space ⊗ K_time
        # For a separable kernel, the full covariance is:
        #   K[i,j] = k_spatial(s_i, s_j) * k_temporal(t_i, t_j)
        # This is mathematically equivalent to the Kronecker product when
        # data is structured on a space-time grid, and reduces to
        # element-wise multiplication for general scattered observations.
        if hasattr(spatial_covar, 'to_dense'):
            spatial_covar = spatial_covar.to_dense()
        if hasattr(temporal_covar, 'to_dense'):
            temporal_covar = temporal_covar.to_dense()

        # Compute Kronecker product via element-wise multiplication
        covar = spatial_covar * temporal_covar

        # Apply output scale
        return self.outputscale_param * covar
    
    @property
    def spatial_lengthscale(self) -> torch.Tensor:
        """Get spatial lengthscale(s)."""
        return self.spatial_kernel.lengthscale
    
    @property
    def temporal_lengthscale(self) -> torch.Tensor:
        """Get temporal lengthscale."""
        return self.temporal_kernel.lengthscale
    
    @property
    def outputscale(self) -> torch.Tensor:
        """Get output scale."""
        return self.outputscale_param
    
    def get_hyperparameters(self) -> dict:
        """
        Get all kernel hyperparameters as a dictionary.
        
        Returns
        -------
        dict
            Dictionary with keys: 'spatial_lengthscale', 'temporal_lengthscale',
            'outputscale'.
        """
        return {
            'spatial_lengthscale': self.spatial_lengthscale.detach().cpu().numpy(),
            'temporal_lengthscale': self.temporal_lengthscale.detach().cpu().numpy(),
            'outputscale': self.outputscale.detach().cpu().numpy(),
        }
    
    def __repr__(self) -> str:
        """String representation."""
        return (
            f"SpatioTemporalKernel(\n"
            f"  spatial={type(self.spatial_kernel).__name__}(ls={self.spatial_lengthscale.data}),\n"
            f"  temporal={type(self.temporal_kernel).__name__}(ls={self.temporal_lengthscale.data}),\n"
            f"  outputscale={self.outputscale.data}\n"
            f")"
        )
