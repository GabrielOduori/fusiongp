"""
Sparse Variational Gaussian Process model for FusionGP.

This module provides the main FusionSVGP class that implements the core
SVGP model for multi-source NO₂ fusion. The model uses inducing points
for scalability and variational inference for tractable posterior approximation.

The model follows the formulation of Hensman et al. (2013):
- A set of M inducing points Z in (x, y, t) space
- Inducing values u = f(Z)
- Variational posterior q(u) = N(m, S)
- Predictions via q(f*) = ∫ p(f*|u) q(u) du

References
----------
.. [1] Hensman, J., Fusi, N., & Lawrence, N. D. (2013). 
       Gaussian processes for big data. UAI.
.. [2] Titsias, M. (2009). Variational learning of inducing variables 
       in sparse Gaussian processes. AISTATS.

Example
-------
>>> model = FusionSVGP(n_inducing=500)
>>> model.initialize_inducing_points(train_x)
>>> 
>>> # Training
>>> optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
>>> for epoch in range(100):
...     loss = -model.elbo(train_x, train_y, source_masks)
...     loss.backward()
...     optimizer.step()
>>> 
>>> # Prediction
>>> pred_mean, pred_var = model.predict(test_x)
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import gpytorch
from gpytorch.models import ApproximateGP
from gpytorch.variational import (
    CholeskyVariationalDistribution,
    VariationalStrategy,
)
from gpytorch.means import ConstantMean
from gpytorch.distributions import MultivariateNormal
from sklearn.cluster import KMeans

from src.models.kernels import SpatioTemporalKernel
from src.models.likelihoods import MultiSourceLikelihood

logger = logging.getLogger(__name__)


class FusionSVGP(ApproximateGP):
    """
    Sparse Variational Gaussian Process for multi-source NO₂ fusion.
    
    This model represents the latent NO₂ concentration field as a single
    Gaussian Process, with observations from multiple sources (EPA, low-cost,
    satellite) entering through source-specific likelihoods.
    
    The SVGP approximation uses inducing points for scalability:
    - Complexity: O(NM²) instead of O(N³)
    - Memory: O(M²) instead of O(N²)
    
    Parameters
    ----------
    n_inducing : int, default=500
        Number of inducing points.
    kernel_type : str, default='matern32'
        Type of kernel for both spatial and temporal components.
    spatial_ard : bool, default=True
        Use ARD (separate lengthscales) for spatial dimensions.
    learn_inducing_locations : bool, default=True
        Whether to optimize inducing point locations.
    initial_lengthscales : Dict[str, float], optional
        Initial lengthscales for kernel.
    initial_noise : Dict[str, float], optional
        Initial noise std for each source.
    learn_kernel_hyperparams : bool, default=True
        Whether to optimize kernel lengthscales/outputscale.
    learn_noise : bool, default=True
        Whether to optimize source-specific noise levels.
    learn_calibration : bool, default=True
        Whether to optimize low-cost sensor calibration parameters.
    sources : List[str], optional
        List of source names.
        
    Attributes
    ----------
    n_inducing : int
        Number of inducing points.
    mean_module : ConstantMean
        Mean function.
    covar_module : SpatioTemporalKernel
        Covariance function.
    likelihood : MultiSourceLikelihood
        Multi-source likelihood.
    variational_strategy : VariationalStrategy
        SVGP variational strategy.
        
    Example
    -------
    >>> # Create model
    >>> model = FusionSVGP(
    ...     n_inducing=500,
    ...     kernel_type='matern32',
    ...     initial_noise={'epa': 1.0, 'low_cost': 5.0, 'satellite': 3.0}
    ... )
    >>> 
    >>> # Initialize inducing points from training data
    >>> model.initialize_inducing_points(train_x)
    >>> 
    >>> # Forward pass gives posterior at inputs
    >>> posterior = model(train_x)
    """
    
    def __init__(
        self,
        n_inducing: int = 500,
        kernel_type: str = 'matern32',
        spatial_ard: bool = True,
        learn_inducing_locations: bool = True,
        initial_lengthscales: Optional[Dict[str, float]] = None,
        initial_noise: Optional[Dict[str, float]] = None,
        learn_kernel_hyperparams: bool = True,
        learn_noise: bool = True,
        learn_calibration: bool = True,
        sources: Optional[List[str]] = None,
    ):
        """
        Initialize the FusionSVGP model.
        
        Parameters
        ----------
        n_inducing : int
            Number of inducing points.
        kernel_type : str
            Kernel type ('matern32', 'matern52', 'rbf').
        spatial_ard : bool
            Use ARD for spatial dimensions.
        learn_inducing_locations : bool
            Optimize inducing locations.
        initial_lengthscales : Dict[str, float]
            Initial kernel lengthscales.
        initial_noise : Dict[str, float]
            Initial noise std per source.
        learn_kernel_hyperparams : bool
            Optimize kernel lengthscale/outputscale.
        learn_noise : bool
            Optimize source-specific noise.
        learn_calibration : bool
            Optimize low-cost calibration parameters.
        sources : List[str]
            Source names.
        """
        self.n_inducing = n_inducing
        self.kernel_type = kernel_type
        self.learn_inducing_locations = learn_inducing_locations
        
        # Default sources
        self.sources = sources if sources else ['epa', 'low_cost', 'satellite']
        
        # Default lengthscales
        ls_init = initial_lengthscales or {}
        spatial_ls = (
            ls_init.get('spatial_x', 0.1),
            ls_init.get('spatial_y', 0.1)
        )
        temporal_ls = ls_init.get('temporal', 0.1)
        
        # Initialize inducing points (will be properly set later)
        # Placeholder: uniform grid in [0, 1]^3
        inducing_points = torch.rand(n_inducing, 3)
        
        # Create variational distribution
        variational_distribution = CholeskyVariationalDistribution(n_inducing)
        
        # Create variational strategy
        variational_strategy = VariationalStrategy(
            self,
            inducing_points,
            variational_distribution,
            learn_inducing_locations=learn_inducing_locations,
        )
        
        # Initialize parent class
        super().__init__(variational_strategy)
        
        # Mean function
        self.mean_module = ConstantMean()
        
        # Covariance function
        self.covar_module = SpatioTemporalKernel(
            spatial_kernel_type=kernel_type,
            temporal_kernel_type=kernel_type,
            spatial_ard=spatial_ard,
            initial_spatial_lengthscale=spatial_ls,
            initial_temporal_lengthscale=temporal_ls,
        )
        if not learn_kernel_hyperparams:
            # Freeze kernel hyperparameters to skip costly hyperparameter optimization
            self.covar_module.spatial_kernel.raw_lengthscale.requires_grad_(False)
            self.covar_module.temporal_kernel.raw_lengthscale.requires_grad_(False)
            self.covar_module.scale_kernel.raw_outputscale.requires_grad_(False)
        
        # Likelihood
        self.likelihood = MultiSourceLikelihood(
            sources=self.sources,
            initial_noise=initial_noise,
            learn_noise=learn_noise,
            learn_calibration=learn_calibration,
        )
        
        logger.info(
            f"Created FusionSVGP: n_inducing={n_inducing}, "
            f"kernel={kernel_type}, sources={self.sources}"
        )
    
    def forward(self, x: torch.Tensor) -> MultivariateNormal:
        """
        Compute the GP prior/posterior at input locations.
        
        Parameters
        ----------
        x : torch.Tensor
            Input locations, shape (N, 3) with columns [lat, lon, time].
            
        Returns
        -------
        MultivariateNormal
            GP distribution at input locations.
        """
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)
    
    def initialize_inducing_points(
        self,
        x: torch.Tensor,
        method: str = 'kmeans',
        random_state: int = 42,
    ) -> None:
        """
        Initialize inducing point locations from training data.
        
        Parameters
        ----------
        x : torch.Tensor
            Training input locations, shape (N, 3).
        method : str, default='kmeans'
            Initialization method: 'kmeans', 'random', or 'grid'.
        random_state : int, default=42
            Random seed for reproducibility.
            
        Notes
        -----
        K-means initialization places inducing points at cluster centers,
        ensuring good coverage of the data distribution. This is recommended
        for most applications.
        """
        n_data = x.shape[0]
        n_inducing = min(self.n_inducing, n_data)
        
        logger.info(f"Initializing {n_inducing} inducing points using {method}")
        
        if method == 'kmeans':
            # Use K-means clustering
            x_np = x.detach().cpu().numpy()
            kmeans = KMeans(
                n_clusters=n_inducing,
                random_state=random_state,
                n_init=10,
            )
            kmeans.fit(x_np)
            inducing_points = torch.tensor(
                kmeans.cluster_centers_,
                dtype=x.dtype,
                device=x.device,
            )
            
        elif method == 'random':
            # Random subset of training points
            torch.manual_seed(random_state)
            indices = torch.randperm(n_data)[:n_inducing]
            inducing_points = x[indices].clone()
            
        elif method == 'grid':
            # Regular grid in normalized space
            n_spatial = int(n_inducing ** (2/3))
            n_temporal = n_inducing // n_spatial
            
            # Create grid
            lat_grid = torch.linspace(x[:, 0].min(), x[:, 0].max(), int(n_spatial**0.5))
            lon_grid = torch.linspace(x[:, 1].min(), x[:, 1].max(), int(n_spatial**0.5))
            time_grid = torch.linspace(x[:, 2].min(), x[:, 2].max(), n_temporal)
            
            # Meshgrid
            lat, lon, time = torch.meshgrid(lat_grid, lon_grid, time_grid, indexing='ij')
            inducing_points = torch.stack([
                lat.flatten(),
                lon.flatten(),
                time.flatten()
            ], dim=1)[:n_inducing]
            
            inducing_points = inducing_points.to(x.dtype).to(x.device)
            
        else:
            raise ValueError(f"Unknown initialization method: {method}")
        
        # Set inducing points
        self.variational_strategy.inducing_points.data = inducing_points
        
        logger.info(
            f"Inducing points initialized: shape={inducing_points.shape}, "
            f"bounds=[{inducing_points.min(0).values.tolist()}, "
            f"{inducing_points.max(0).values.tolist()}]"
        )
    
    def elbo(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        source_masks: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute the Evidence Lower Bound (ELBO).
        
        The ELBO is:
            L = E_q(f)[log p(y|f)] - KL(q(u) || p(u))
        
        This is the objective maximized during training.
        
        Parameters
        ----------
        x : torch.Tensor
            Input locations, shape (N, 3).
        y : torch.Tensor
            Observations, shape (N, n_sources).
        source_masks : torch.Tensor
            Valid observation masks, shape (N, n_sources).
            
        Returns
        -------
        torch.Tensor
            ELBO value (scalar).
        """
        # Get variational posterior at x
        variational_dist = self.variational_strategy(x)
        
        # Expected log likelihood
        expected_log_lik = self.likelihood.expected_log_prob(
            y, variational_dist, source_masks
        )
        
        # KL divergence
        kl_divergence = self.variational_strategy.kl_divergence()
        
        # ELBO = E[log p(y|f)] - KL
        elbo = expected_log_lik - kl_divergence
        
        return elbo
    
    def predict(
        self,
        x: torch.Tensor,
        include_noise: bool = False,
        source: str = 'epa',
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Make predictions at new locations.
        
        Parameters
        ----------
        x : torch.Tensor
            Prediction locations, shape (N, 3).
        include_noise : bool, default=False
            Whether to include observation noise in variance.
        source : str, default='epa'
            Source for noise variance (if include_noise=True).
            
        Returns
        -------
        mean : torch.Tensor
            Predictive mean, shape (N,).
        variance : torch.Tensor
            Predictive variance, shape (N,).
        """
        self.eval()
        with torch.no_grad():
            # Get posterior
            posterior = self(x)
            mean = posterior.mean
            variance = posterior.variance
            
            if include_noise:
                noise_var = self.likelihood.noise_variance[source]
                variance = variance + noise_var
        
        return mean, variance
    
    def predict_with_samples(
        self,
        x: torch.Tensor,
        n_samples: int = 100,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Make predictions with posterior samples for uncertainty.
        
        Parameters
        ----------
        x : torch.Tensor
            Prediction locations, shape (N, 3).
        n_samples : int, default=100
            Number of posterior samples.
            
        Returns
        -------
        mean : torch.Tensor
            Predictive mean, shape (N,).
        std : torch.Tensor
            Predictive standard deviation, shape (N,).
        samples : torch.Tensor
            Posterior samples, shape (n_samples, N).
        """
        self.eval()
        with torch.no_grad():
            posterior = self(x)
            samples = posterior.rsample(torch.Size([n_samples]))
            mean = samples.mean(dim=0)
            std = samples.std(dim=0)
        
        return mean, std, samples
    
    def get_inducing_points(self) -> torch.Tensor:
        """
        Get current inducing point locations.
        
        Returns
        -------
        torch.Tensor
            Inducing points, shape (M, 3).
        """
        return self.variational_strategy.inducing_points.detach()
    
    def get_hyperparameters(self) -> Dict[str, any]:
        """
        Get all model hyperparameters.
        
        Returns
        -------
        Dict[str, any]
            Dictionary of hyperparameters.
        """
        kernel_params = self.covar_module.get_hyperparameters()
        likelihood_params = self.likelihood.get_parameters()
        
        return {
            'mean': self.mean_module.constant.detach().cpu().numpy(),
            **kernel_params,
            **{k: v.cpu().numpy() for k, v in likelihood_params.items()},
            'n_inducing': self.n_inducing,
        }
    
    def __repr__(self) -> str:
        """String representation."""
        return (
            f"FusionSVGP(\n"
            f"  n_inducing={self.n_inducing},\n"
            f"  mean={self.mean_module},\n"
            f"  kernel={self.covar_module},\n"
            f"  likelihood={self.likelihood}\n"
            f")"
        )
