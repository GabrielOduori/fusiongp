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

import numpy as np
import torch
import torch.nn as nn
import gpytorch
from gpytorch.models import ApproximateGP
from gpytorch.variational import (
    CholeskyVariationalDistribution,
    VariationalStrategy,
)
from gpytorch.means import ConstantMean, Mean
from gpytorch.distributions import MultivariateNormal
from sklearn.cluster import KMeans

from src.models.kernels import SpatioTemporalKernel
from src.models.likelihoods import MultiSourceLikelihood
from src.models.prior_mean import RasterMean, ATMOPlanMean

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
    prior_mean : gpytorch.means.Mean, optional
        Custom prior mean function (e.g., ATMOPlanMean for physics-based prior).
        If None, uses ConstantMean. When provided, the GP learns residuals
        from this background field.
        
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
        spatial_kernel_type: Optional[str] = None,
        temporal_kernel_type: Optional[str] = None,
        spatial_ard: bool = True,
        learn_inducing_locations: bool = True,
        n_covariates: int = 0,
        initial_lengthscales: Optional[Dict[str, float]] = None,
        initial_noise: Optional[Dict[str, float]] = None,
        noise_bounds: tuple[float, float] = (0.01, 100.0),
        variational_jitter: float = 1e-4,
        learn_kernel_hyperparams: bool = True,
        learn_noise: bool = True,
        learn_noise_sources: Optional[List[str]] = None,
        learn_calibration: bool = True,
        sources: Optional[List[str]] = None,
        prior_mean: Optional[Mean] = None,
    ):
        """
        Initialize the FusionSVGP model.
        
        Parameters
        ----------
        n_inducing : int
            Number of inducing points.
        kernel_type : str
            Kernel type for both spatial and temporal ('matern32', 'matern52', 'rbf').
            Overridden by spatial_kernel_type and temporal_kernel_type if provided.
        spatial_kernel_type : str, optional
            Kernel type for spatial component only. If None, uses kernel_type.
        temporal_kernel_type : str, optional
            Kernel type for temporal component only. If None, uses kernel_type.
            Use 'exponential' for rapid temporal decay (environmental modeling).
        spatial_ard : bool
            Use ARD for spatial dimensions.
        learn_inducing_locations : bool
            Optimize inducing locations.
        n_covariates : int
            Number of additional covariates appended to inputs.
        initial_lengthscales : Dict[str, float]
            Initial kernel lengthscales.
        initial_noise : Dict[str, float]
            Initial noise std per source.
        noise_bounds : tuple[float, float]
            Lower and upper bounds for learned likelihood noise std.
        variational_jitter : float
            Diagonal jitter used by the variational strategy Cholesky factor.
        learn_kernel_hyperparams : bool
            Optimize kernel lengthscale/outputscale.
        learn_noise : bool
            Optimize source-specific noise.
        learn_noise_sources : List[str], optional
            Subset of sources to learn noise for.
        learn_calibration : bool
            Optimize low-cost calibration parameters.
        sources : List[str]
            Source names.
        prior_mean : gpytorch.means.Mean, optional
            Custom prior mean function. Use ATMOPlanMean to incorporate
            ATMO-Plan background as the GP prior.
        """
        self.n_inducing = n_inducing
        self.kernel_type = kernel_type
        self.learn_inducing_locations = learn_inducing_locations
        self.n_covariates = n_covariates
        self.variational_jitter = float(variational_jitter)

        # Allow separate spatial and temporal kernel types
        self.spatial_kernel_type = spatial_kernel_type or kernel_type
        self.temporal_kernel_type = temporal_kernel_type or kernel_type
        
        # Default sources
        self.sources = sources if sources else ['epa', 'satellite']
        
        # Default lengthscales
        ls_init = initial_lengthscales or {}
        spatial_ls = (
            ls_init.get('spatial_x', 0.1),
            ls_init.get('spatial_y', 0.1)
        )
        temporal_ls = ls_init.get('temporal', 0.1)
        
        # Initialize inducing points (will be properly set later)
        # Placeholder: uniform grid in [0, 1]^(3 + n_covariates)
        inducing_points = torch.rand(n_inducing, 3 + self.n_covariates)
        
        # Create variational distribution
        variational_distribution = CholeskyVariationalDistribution(n_inducing)
        
        # Create variational strategy
        variational_strategy = VariationalStrategy(
            self,
            inducing_points,
            variational_distribution,
            learn_inducing_locations=learn_inducing_locations,
            jitter_val=self.variational_jitter,
        )
        
        # Initialize parent class
        super().__init__(variational_strategy)

        # Mean function
        # If prior_mean provided (e.g., ATMO-Plan), use it; otherwise constant mean
        if prior_mean is not None:
            self.mean_module = prior_mean
            self._has_prior_mean = True
            logger.info(f"Using custom prior mean: {type(prior_mean).__name__}")
        else:
            self.mean_module = ConstantMean()
            self._has_prior_mean = False
        
        # Covariance function
        covariate_dims = list(range(3, 3 + self.n_covariates)) if self.n_covariates > 0 else None
        self.covar_module = SpatioTemporalKernel(
            spatial_kernel_type=self.spatial_kernel_type,
            temporal_kernel_type=self.temporal_kernel_type,
            spatial_ard=spatial_ard,
            initial_spatial_lengthscale=spatial_ls,
            initial_temporal_lengthscale=temporal_ls,
            covariate_dims=covariate_dims,
        )
        if not learn_kernel_hyperparams:
            # Freeze kernel hyperparameters to skip costly hyperparameter optimization
            self.covar_module.spatial_kernel.raw_lengthscale.requires_grad_(False)
            self.covar_module.temporal_kernel.raw_lengthscale.requires_grad_(False)
            self.covar_module.outputscale_param.requires_grad_(False)
        
        # Likelihood
        self.likelihood = MultiSourceLikelihood(
            sources=self.sources,
            initial_noise=initial_noise,
            noise_bounds=noise_bounds,
            learn_noise=learn_noise,
            learn_noise_sources=learn_noise_sources,
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
            Input locations, shape (N, 3 + K) with columns
            [lat, lon, time, covariates...].
            
        Returns
        -------
        MultivariateNormal
            GP distribution at input locations.
        """
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)

    def __call__(self, *args, **kwargs):
        """Evaluate the model with the configured Cholesky jitter."""
        with gpytorch.settings.cholesky_jitter(
            float_value=self.variational_jitter,
            double_value=self.variational_jitter,
        ):
            return super().__call__(*args, **kwargs)
    
    def initialize_inducing_points(
        self,
        x: torch.Tensor,
        method: str = 'kmeans',
        random_state: int = 42,
        spatial_inducing: Optional[int] = None,
        temporal_inducing: Optional[int] = None,
    ) -> None:
        """
        Initialize inducing point locations from training data.

        Parameters
        ----------
        x : torch.Tensor
            Training input locations, shape (N, 3).
        method : str, default='kmeans'
            Initialization method:
            - 'kmeans': K-means clustering (default, good for scattered data)
            - 'random': Random subset of training points
            - 'grid': Regular grid (simple but not data-adaptive)
            - 'structured_grid': Kronecker-structured grid from data (recommended for gridded data)
        random_state : int, default=42
            Random seed for reproducibility.
        spatial_inducing : int, optional
            Number of spatial inducing points (for structured_grid method).
            If None, auto-computed to maintain aspect ratio.
        temporal_inducing : int, optional
            Number of temporal inducing points (for structured_grid method).
            If None, auto-computed to maintain aspect ratio.

        Notes
        -----
        - **K-means**: Good for scattered data, places points at cluster centers
        - **Structured grid**: Exploits Kronecker structure for gridded data,
          enables future optimizations in KL divergence computation.
          Creates M_s × M_t grid where M = M_s × M_t = n_inducing.
        """
        n_data = x.shape[0]
        n_inducing = min(self.n_inducing, n_data)

        if n_inducing != self.n_inducing:
            logger.warning(
                f"Requested {self.n_inducing} inducing points, but only {n_data} data points available. "
                f"Using {n_inducing} inducing points instead."
            )

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
            self._inducing_grid_shape = None  # Not structured

        elif method == 'random':
            # Random subset of training points
            torch.manual_seed(random_state)
            indices = torch.randperm(n_data)[:n_inducing]
            inducing_points = x[indices].clone()
            self._inducing_grid_shape = None  # Not structured

        elif method == 'structured_grid':
            # Kronecker-structured grid from actual data points
            inducing_points, grid_shape = self._create_structured_grid(
                x, n_inducing, spatial_inducing, temporal_inducing
            )
            self._inducing_grid_shape = grid_shape
            # Update n_inducing to actual grid size (may differ slightly from requested)
            n_inducing = inducing_points.shape[0]
            logger.info(
                f"Created structured grid: {grid_shape[0]} spatial × {grid_shape[1]} temporal "
                f"= {grid_shape[0] * grid_shape[1]} inducing points"
            )

        elif method == 'grid':
            # Simple regular grid in normalized space (less data-adaptive)
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

            if self.n_covariates > 0:
                cov_mean = x[:, 3:].mean(dim=0)
                covars = cov_mean.repeat(inducing_points.shape[0], 1)
                inducing_points = torch.cat([inducing_points, covars], dim=1)

            inducing_points = inducing_points.to(x.dtype).to(x.device)
            self._inducing_grid_shape = None  # Not Kronecker-structured

        else:
            raise ValueError(f"Unknown initialization method: {method}")
        
        # If the number of inducing points changed, we need to reinitialize the variational strategy
        if n_inducing != self.variational_strategy.inducing_points.shape[0]:
            logger.info(
                f"Reinitializing variational strategy with {n_inducing} inducing points "
                f"(was {self.variational_strategy.inducing_points.shape[0]})"
            )

            # Create new variational distribution with correct size
            variational_distribution = CholeskyVariationalDistribution(n_inducing)

            # Create new variational strategy
            variational_strategy = VariationalStrategy(
                self,
                inducing_points,
                variational_distribution,
                learn_inducing_locations=self.learn_inducing_locations,
                jitter_val=self.variational_jitter,
            )

            # Replace the old variational strategy
            self.variational_strategy = variational_strategy
        else:
            # Just update inducing points
            self.variational_strategy.inducing_points.data = inducing_points

        logger.info(
            f"Inducing points initialized: shape={inducing_points.shape}, "
            f"bounds=[{inducing_points.min(0).values.tolist()}, "
            f"{inducing_points.max(0).values.tolist()}]"
        )

    def _create_structured_grid(
        self,
        x: torch.Tensor,
        n_inducing: int,
        spatial_inducing: Optional[int] = None,
        temporal_inducing: Optional[int] = None,
    ) -> tuple[torch.Tensor, tuple[int, int]]:
        """
        Create Kronecker-structured inducing points from gridded data.

        For data on N_s × N_t grid, this creates M_s × M_t structured
        inducing points by sampling uniformly from unique spatial/temporal values.

        This enables future Kronecker-aware variational inference optimizations
        where K_ZZ = K_time ⊗ K_space.

        Parameters
        ----------
        x : torch.Tensor
            Training data, shape (N, 3).
        n_inducing : int
            Target number of inducing points (M = M_s × M_t).
        spatial_inducing : int, optional
            Number of spatial inducing points M_s.
            If None, auto-computed to maintain data aspect ratio.
        temporal_inducing : int, optional
            Number of temporal inducing points M_t.
            If None, auto-computed to maintain data aspect ratio.

        Returns
        -------
        inducing_points : torch.Tensor
            Inducing points, shape (M_s × M_t, 3).
        grid_shape : tuple[int, int]
            Grid dimensions (M_s, M_t).
        """
        # Extract unique spatial and temporal locations from data
        unique_spatial, _ = torch.unique(x[:, :2], dim=0, return_inverse=True)
        unique_temporal, _ = torch.unique(x[:, 2], dim=0, return_inverse=True)

        n_s = len(unique_spatial)
        n_t = len(unique_temporal)

        logger.info(
            f"Data has {n_s} unique spatial locations and {n_t} unique temporal points"
        )

        # Determine grid dimensions
        if spatial_inducing is not None and temporal_inducing is not None:
            M_s = min(spatial_inducing, n_s)
            M_t = min(temporal_inducing, n_t)
        elif spatial_inducing is not None:
            M_s = min(spatial_inducing, n_s)
            M_t = min(n_inducing // M_s, n_t)
        elif temporal_inducing is not None:
            M_t = min(temporal_inducing, n_t)
            M_s = min(n_inducing // M_t, n_s)
        else:
            # Auto-compute to maintain aspect ratio N_s / N_t
            ratio = n_s / n_t
            M_t = int(np.sqrt(n_inducing / ratio))
            M_t = max(M_t, 1)
            M_s = n_inducing // M_t
            M_s = max(M_s, 1)

            # Adjust to get closer to target n_inducing
            while M_s * M_t < n_inducing and M_s < n_s:
                M_s += 1
            while M_s * M_t < n_inducing and M_t < n_t:
                M_t += 1

            # Clamp to data limits
            M_s = min(M_s, n_s)
            M_t = min(M_t, n_t)

        # Sample uniformly from unique spatial and temporal points
        if M_s < n_s:
            # Uniformly spaced indices
            spatial_indices = torch.linspace(0, n_s - 1, M_s).long()
            inducing_spatial = unique_spatial[spatial_indices]
        else:
            inducing_spatial = unique_spatial

        if M_t < n_t:
            temporal_indices = torch.linspace(0, n_t - 1, M_t).long()
            inducing_temporal = unique_temporal[temporal_indices]
        else:
            inducing_temporal = unique_temporal

        # Create grid: all combinations of (spatial, temporal)
        # This is the Kronecker structure
        n_inducing_actual = M_s * M_t

        inducing_points = torch.zeros(
            n_inducing_actual,
            3 + self.n_covariates,
            dtype=x.dtype,
            device=x.device,
        )
        covariate_mean = None
        if self.n_covariates > 0:
            covariate_mean = x[:, 3:].mean(dim=0)

        idx = 0
        for s_idx in range(M_s):
            for t_idx in range(M_t):
                inducing_points[idx, :2] = inducing_spatial[s_idx]
                inducing_points[idx, 2] = inducing_temporal[t_idx]
                if covariate_mean is not None:
                    inducing_points[idx, 3:] = covariate_mean
                idx += 1

        return inducing_points, (M_s, M_t)

    def elbo(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        source_masks: torch.Tensor,
        n_data: int = None,
    ) -> torch.Tensor:
        """
        Compute the Evidence Lower Bound (ELBO).

        The ELBO is:
            L = E_q(f)[log p(y|f)] - KL(q(u) || p(u))

        For mini-batch training, the KL term should be scaled by the batch size.

        This is the objective maximized during training.

        Parameters
        ----------
        x : torch.Tensor
            Input locations, shape (N, 3).
        y : torch.Tensor
            Observations, shape (N, n_sources).
        source_masks : torch.Tensor
            Valid observation masks, shape (N, n_sources).
        n_data : int, optional
            Total number of data points (for mini-batch scaling).
            If None, uses batch size.

        Returns
        -------
        torch.Tensor
            ELBO value (scalar).
        """
        # Get variational posterior at x
        with gpytorch.settings.cholesky_jitter(
            float_value=self.variational_jitter,
            double_value=self.variational_jitter,
        ):
            variational_dist = self.variational_strategy(x)

        # Expected log likelihood
        expected_log_lik = self.likelihood.expected_log_prob(
            y, variational_dist, source_masks
        )
        if n_data is not None:
            # Scale mini-batch likelihood to approximate full-data objective.
            expected_log_lik = expected_log_lik * (n_data / x.shape[0])

        # KL divergence (should be independent of batch size)
        kl_divergence = self.variational_strategy.kl_divergence()

        # ELBO = E[log p(y|f)] - KL
        # For minibatch training, the likelihood is scaled; KL remains unscaled.
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
        
        mean_params = {}
        if hasattr(self.mean_module, "constant"):
            mean_params["mean"] = self.mean_module.constant.detach().cpu().numpy()

        return {
            **mean_params,
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
