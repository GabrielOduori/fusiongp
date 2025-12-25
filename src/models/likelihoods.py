"""
Multi-source likelihood for FusionGP.

This module provides the MultiSourceLikelihood class that implements
heteroscedastic Gaussian likelihoods with source-specific noise variances
and calibration parameters for low-cost sensors.

The likelihood model is:
    EPA:       y = f(s,t) + ε,         ε ~ N(0, σ²_EPA)
    Low-cost:  y = a*f(s,t) + b + ε,   ε ~ N(0, σ²_LC)
    Satellite: y = f(s,t) + ε,         ε ~ N(0, σ²_SAT)

where a and b are learned calibration parameters for low-cost sensors.

References
----------
.. [1] Hensman, J., Fusi, N., & Lawrence, N. D. (2013). 
       Gaussian processes for big data. UAI.
.. [2] Malings, C., et al. (2017). Surface heat assessment for developed 
       environments. Urban Climate.

Example
-------
>>> likelihood = MultiSourceLikelihood(
...     sources=['epa', 'low_cost', 'satellite'],
...     initial_noise={'epa': 1.0, 'low_cost': 5.0, 'satellite': 3.0}
... )
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import gpytorch
from gpytorch.likelihoods import Likelihood
from gpytorch.distributions import MultivariateNormal

logger = logging.getLogger(__name__)


class MultiSourceLikelihood(Likelihood):
    """
    Heteroscedastic Gaussian likelihood for multi-source observations.
    
    This likelihood handles observations from multiple sources (EPA, low-cost
    sensors, satellite) with:
    - Source-specific noise variances
    - Linear calibration for low-cost sensors (y = a*f + b)
    - Masking for missing observations
    
    The noise variances and calibration parameters can be learned during
    training via gradient descent.
    
    Parameters
    ----------
    sources : List[str], default=['epa', 'low_cost', 'satellite']
        List of source names.
    initial_noise : Dict[str, float], optional
        Initial noise standard deviations per source.
        Default: {'epa': 1.0, 'low_cost': 5.0, 'satellite': 3.0}
    learn_noise : bool, default=True
        Whether to learn noise parameters.
    noise_bounds : Tuple[float, float], default=(0.01, 100.0)
        Bounds for noise standard deviations.
    initial_calibration : Dict[str, float], optional
        Initial calibration parameters for low-cost sensors.
        Default: {'slope': 1.0, 'intercept': 0.0}
    learn_calibration : bool, default=True
        Whether to learn calibration parameters.
        
    Attributes
    ----------
    sources : List[str]
        Source names.
    source_to_idx : Dict[str, int]
        Mapping from source name to index.
    raw_noise : nn.ParameterDict
        Raw (unconstrained) noise parameters.
    lc_slope : nn.Parameter
        Low-cost sensor calibration slope.
    lc_intercept : nn.Parameter
        Low-cost sensor calibration intercept.
        
    Example
    -------
    >>> likelihood = MultiSourceLikelihood()
    >>> 
    >>> # During training, compute log probability
    >>> f_dist = model(train_x)  # GP posterior
    >>> log_prob = likelihood.log_marginal(train_y, f_dist, source_masks)
    >>> 
    >>> # During prediction
    >>> pred_dist = likelihood(f_dist)
    """
    
    DEFAULT_NOISE = {
        'epa': 1.0,
        'low_cost': 5.0,
        'satellite': 3.0,
    }
    
    DEFAULT_CALIBRATION = {
        'slope': 1.0,
        'intercept': 0.0,
    }
    
    def __init__(
        self,
        sources: List[str] = None,
        initial_noise: Dict[str, float] = None,
        learn_noise: bool = True,
        noise_bounds: Tuple[float, float] = (0.01, 100.0),  # Allow learning low noise for normalized data
        initial_calibration: Dict[str, float] = None,
        learn_calibration: bool = True,
    ):
        """
        Initialize the MultiSourceLikelihood.
        
        Parameters
        ----------
        sources : List[str]
            Source names.
        initial_noise : Dict[str, float]
            Initial noise std per source.
        learn_noise : bool
            Learn noise parameters.
        noise_bounds : Tuple[float, float]
            Noise std bounds.
        initial_calibration : Dict[str, float]
            Initial calibration params.
        learn_calibration : bool
            Learn calibration parameters.
        """
        super().__init__()
        
        # Set sources
        self.sources = sources if sources else ['epa', 'low_cost', 'satellite']
        self.source_to_idx = {s: i for i, s in enumerate(self.sources)}
        
        # Merge defaults with provided values
        noise_init = {**self.DEFAULT_NOISE}
        if initial_noise:
            noise_init.update(initial_noise)
        
        calib_init = {**self.DEFAULT_CALIBRATION}
        if initial_calibration:
            calib_init.update(initial_calibration)
        
        # Store bounds
        self.noise_bounds = noise_bounds
        self.learn_noise = learn_noise
        self.learn_calibration = learn_calibration
        
        # Create noise parameters (stored in log space for positivity)
        self.raw_noise = nn.ParameterDict()
        for source in self.sources:
            init_val = noise_init.get(source, 1.0)
            # Store as log(noise_std) for unconstrained optimization
            raw_val = torch.tensor(init_val).log()
            self.raw_noise[source] = nn.Parameter(
                raw_val,
                requires_grad=learn_noise
            )
        
        # Calibration parameters for low-cost sensors
        # y_lc = slope * f + intercept
        self.raw_lc_slope = nn.Parameter(
            torch.tensor(calib_init['slope']).log(),  # log for positivity
            requires_grad=learn_calibration
        )
        self.raw_lc_intercept = nn.Parameter(
            torch.tensor(calib_init['intercept']),
            requires_grad=learn_calibration
        )
        
        logger.info(
            f"Created MultiSourceLikelihood: sources={self.sources}, "
            f"learn_noise={learn_noise}, learn_calibration={learn_calibration}"
        )
    
    @property
    def noise_std(self) -> Dict[str, torch.Tensor]:
        """
        Get noise standard deviations for each source.

        Returns
        -------
        Dict[str, torch.Tensor]
            Noise std per source (exp for positivity, no bounds).
        """
        result = {}
        for source in self.sources:
            # exp for positivity, NO CLAMPING - let model learn freely
            noise = self.raw_noise[source].exp()
            result[source] = noise
        return result
    
    @property
    def noise_variance(self) -> Dict[str, torch.Tensor]:
        """
        Get noise variances for each source.
        
        Returns
        -------
        Dict[str, torch.Tensor]
            Noise variance per source.
        """
        return {s: n**2 for s, n in self.noise_std.items()}
    
    @property
    def lc_slope(self) -> torch.Tensor:
        """Get low-cost sensor calibration slope (exp for positivity, no bounds)."""
        return self.raw_lc_slope.exp()  # NO CLAMPING
    
    @property
    def lc_intercept(self) -> torch.Tensor:
        """Get low-cost sensor calibration intercept (no bounds)."""
        return self.raw_lc_intercept  # NO CLAMPING
    
    def transform_latent(
        self,
        f: torch.Tensor,
        source: str,
    ) -> torch.Tensor:
        """
        Transform latent function values for a specific source.
        
        Applies the link function h_q(f) for each source:
        - EPA: h(f) = f
        - Low-cost: h(f) = a*f + b
        - Satellite: h(f) = f
        
        Parameters
        ----------
        f : torch.Tensor
            Latent function values.
        source : str
            Source name.
            
        Returns
        -------
        torch.Tensor
            Transformed values.
        """
        if source == 'low_cost':
            return self.lc_slope * f + self.lc_intercept
        else:
            return f
    
    def inverse_transform_latent(
        self,
        y: torch.Tensor,
        source: str,
    ) -> torch.Tensor:
        """
        Inverse transform observations to latent space.
        
        Used for initializing or interpreting observations in latent space.
        
        Parameters
        ----------
        y : torch.Tensor
            Observed values.
        source : str
            Source name.
            
        Returns
        -------
        torch.Tensor
            Latent space values.
        """
        if source == 'low_cost':
            return (y - self.lc_intercept) / self.lc_slope
        else:
            return y
    
    def forward(
        self,
        function_dist: MultivariateNormal,
        source: str = 'epa',
        **kwargs,
    ) -> MultivariateNormal:
        """
        Add observation noise to the GP posterior.
        
        Parameters
        ----------
        function_dist : MultivariateNormal
            GP posterior distribution.
        source : str
            Source for noise variance.
        **kwargs
            Additional arguments.
            
        Returns
        -------
        MultivariateNormal
            Posterior predictive distribution including observation noise.
        """
        mean = function_dist.mean
        covar = function_dist.covariance_matrix
        
        # Transform mean for source
        transformed_mean = self.transform_latent(mean, source)
        
        # Transform covariance for low-cost (scale by slope^2)
        if source == 'low_cost':
            scale = self.lc_slope ** 2
            transformed_covar = covar * scale
        else:
            transformed_covar = covar
        
        # Add observation noise
        noise_var = self.noise_variance[source]
        noise_covar = torch.eye(
            transformed_covar.shape[-1],
            device=transformed_covar.device,
            dtype=transformed_covar.dtype
        ) * noise_var
        
        return MultivariateNormal(
            transformed_mean,
            transformed_covar + noise_covar
        )
    
    def log_marginal(
        self,
        observations: torch.Tensor,
        function_dist: MultivariateNormal,
        source_masks: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute log marginal likelihood for multi-source observations.
        
        This is the main function used during training to compute the
        expected log likelihood term in the ELBO.
        
        Parameters
        ----------
        observations : torch.Tensor
            Observation values, shape (N, n_sources).
        function_dist : MultivariateNormal
            GP posterior at observation locations.
        source_masks : torch.Tensor
            Boolean masks for valid observations, shape (N, n_sources).
            
        Returns
        -------
        torch.Tensor
            Log marginal likelihood (scalar).
        """
        f_mean = function_dist.mean
        f_var = function_dist.variance
        
        total_log_prob = torch.tensor(0.0, device=f_mean.device, dtype=f_mean.dtype)
        
        for source_idx, source in enumerate(self.sources):
            mask = source_masks[:, source_idx]
            if not mask.any():
                continue
            
            # Get observations for this source
            y = observations[mask, source_idx]
            f_m = f_mean[mask]
            f_v = f_var[mask]
            
            # Transform latent to observation space
            obs_mean = self.transform_latent(f_m, source)
            
            # Transform variance for low-cost
            if source == 'low_cost':
                obs_var = (self.lc_slope ** 2) * f_v
            else:
                obs_var = f_v
            
            # Add noise variance
            noise_var = self.noise_variance[source]
            total_var = obs_var + noise_var
            
            # Gaussian log probability
            log_prob = -0.5 * (
                torch.log(2 * torch.pi * total_var) +
                (y - obs_mean) ** 2 / total_var
            )
            
            total_log_prob = total_log_prob + log_prob.sum()
        
        return total_log_prob
    
    def expected_log_prob(
        self,
        observations: torch.Tensor,
        function_dist: MultivariateNormal,
        source_masks: torch.Tensor,
    ) -> torch.Tensor:
        """
        Alias for log_marginal for compatibility with GPyTorch.
        
        Parameters
        ----------
        observations : torch.Tensor
            Observation values.
        function_dist : MultivariateNormal
            GP posterior.
        source_masks : torch.Tensor
            Valid observation masks.
            
        Returns
        -------
        torch.Tensor
            Expected log probability.
        """
        return self.log_marginal(observations, function_dist, source_masks)
    
    def get_parameters(self) -> Dict[str, torch.Tensor]:
        """
        Get all likelihood parameters as a dictionary.
        
        Returns
        -------
        Dict[str, torch.Tensor]
            Dictionary of parameters.
        """
        params = {
            f'noise_std_{s}': self.noise_std[s].detach() 
            for s in self.sources
        }
        params['lc_slope'] = self.lc_slope.detach()
        params['lc_intercept'] = self.lc_intercept.detach()
        return params
    
    def __repr__(self) -> str:
        """String representation."""
        noise_str = ", ".join(
            f"{s}={self.noise_std[s].item():.3f}" 
            for s in self.sources
        )
        return (
            f"MultiSourceLikelihood(\n"
            f"  noise_std: {noise_str}\n"
            f"  lc_calibration: slope={self.lc_slope.item():.3f}, "
            f"intercept={self.lc_intercept.item():.3f}\n"
            f")"
        )
