"""
Prediction utilities for FusionGP.

This module provides the Predictor class for making predictions on
new locations with uncertainty quantification.

The predictor handles:
- Batched prediction for memory efficiency
- Uncertainty quantification via predictive variance
- Inverse transformation to original scale
- Gridded prediction outputs

References
----------
.. [1] Williams, C. K., & Rasmussen, C. E. (2006). 
       Gaussian processes for machine learning. MIT Press.

Example
-------
>>> predictor = Predictor(model, scalers)
>>> predictions = predictor.predict_grid(test_data)
>>> print(predictions.keys())
dict_keys(['mean', 'std', 'lower_95', 'upper_95', 'coords', 'timestamps'])
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from tqdm import tqdm

from src.data.loader import FusionData
from src.data.preprocessor import Scalers
from src.models.svgp import FusionSVGP

logger = logging.getLogger(__name__)


@dataclass
class Predictions:
    """
    Container for prediction results.
    
    Attributes
    ----------
    mean : np.ndarray
        Predictive mean, shape (N,) or (N, T).
    std : np.ndarray
        Predictive standard deviation.
    lower_ci : Dict[float, np.ndarray]
        Lower confidence bounds at various levels.
    upper_ci : Dict[float, np.ndarray]
        Upper confidence bounds at various levels.
    coords : np.ndarray
        Prediction coordinates (original scale).
    timestamps : np.ndarray
        Prediction timestamps (original scale).
    grid_ids : np.ndarray
        Grid cell IDs.
    """
    mean: np.ndarray
    std: np.ndarray
    lower_ci: Dict[float, np.ndarray]
    upper_ci: Dict[float, np.ndarray]
    coords: np.ndarray
    timestamps: np.ndarray
    grid_ids: Optional[np.ndarray] = None
    
    def to_dataframe(self) -> 'pd.DataFrame':
        """
        Convert predictions to pandas DataFrame.
        
        Returns
        -------
        pd.DataFrame
            DataFrame with predictions and coordinates.
        """
        import pandas as pd
        
        data = {
            'latitude': self.coords[:, 0],
            'longitude': self.coords[:, 1],
            'timestamp': self.timestamps,
            'mean': self.mean,
            'std': self.std,
        }
        
        if self.grid_ids is not None:
            data['grid_id'] = self.grid_ids
        
        for level, lower in self.lower_ci.items():
            data[f'lower_{int(level*100)}'] = lower
        for level, upper in self.upper_ci.items():
            data[f'upper_{int(level*100)}'] = upper
        
        return pd.DataFrame(data)
    
    def summary(self) -> str:
        """Generate summary string."""
        return (
            f"Predictions Summary\n"
            f"{'='*40}\n"
            f"N predictions: {len(self.mean):,}\n"
            f"Mean range: [{self.mean.min():.2f}, {self.mean.max():.2f}]\n"
            f"Std range: [{self.std.min():.2f}, {self.std.max():.2f}]\n"
            f"Confidence levels: {list(self.lower_ci.keys())}"
        )


class Predictor:
    """
    Predictor for FusionGP models.
    
    Makes predictions on new locations with uncertainty quantification,
    handling batching for memory efficiency and inverse transformation
    to the original data scale.
    
    Parameters
    ----------
    model : FusionSVGP
        Trained FusionSVGP model.
    scalers : Scalers, optional
        Scalers for inverse transformation.
    batch_size : int, default=2048
        Batch size for prediction.
    device : str, default='cpu'
        Device for prediction.
    confidence_levels : List[float], default=[0.5, 0.9, 0.95]
        Confidence levels for prediction intervals.
        
    Attributes
    ----------
    model : FusionSVGP
        The model.
    scalers : Scalers
        Normalization scalers.
    
    Example
    -------
    >>> predictor = Predictor(model, scalers, batch_size=4096)
    >>> 
    >>> # Predict on test data
    >>> predictions = predictor.predict(test_data)
    >>> 
    >>> # Predict on custom grid
    >>> grid_predictions = predictor.predict_grid(
    ...     lat_range=(40.7, 40.9),
    ...     lon_range=(-74.1, -73.9),
    ...     timestamps=[0.0, 0.5, 1.0]
    ... )
    """
    
    # Z-scores for confidence intervals
    Z_SCORES = {
        0.5: 0.6745,
        0.8: 1.2816,
        0.9: 1.6449,
        0.95: 1.9600,
        0.99: 2.5758,
    }
    
    def __init__(
        self,
        model: FusionSVGP,
        scalers: Optional[Scalers] = None,
        batch_size: int = 2048,
        device: str = 'cpu',
        confidence_levels: List[float] = None,
        include_observation_noise: bool = True,
        noise_source: str = 'epa',
    ):
        """
        Initialize the Predictor.

        Parameters
        ----------
        model : FusionSVGP
            Trained model.
        scalers : Scalers
            Normalization scalers.
        batch_size : int
            Prediction batch size.
        device : str
            Device.
        confidence_levels : List[float]
            Confidence levels.
        include_observation_noise : bool, default=True
            Whether to include observation noise in predictive variance.
            Set to True for prediction at new locations (default).
            Set to False for interpolation at observed locations.
        noise_source : str, default='epa'
            Which source's noise variance to use when include_observation_noise=True.
        """
        self.model = model
        self.scalers = scalers
        self.batch_size = batch_size
        self.device = torch.device(device)
        self.confidence_levels = confidence_levels or [0.5, 0.9, 0.95]
        self.include_observation_noise = include_observation_noise
        self.noise_source = noise_source

        # Move model to device
        self.model = self.model.to(self.device)
        self.model.eval()

        logger.info(
            f"Predictor initialized: batch_size={batch_size}, "
            f"confidence_levels={self.confidence_levels}, "
            f"include_observation_noise={include_observation_noise}, "
            f"noise_source={noise_source}"
        )
    
    def predict(
        self,
        data: FusionData,
        return_samples: bool = False,
        n_samples: int = 100,
        verbose: bool = True,
    ) -> Predictions:
        """
        Make predictions on FusionData.
        
        Parameters
        ----------
        data : FusionData
            Data with prediction locations.
        return_samples : bool, default=False
            Whether to include posterior samples.
        n_samples : int, default=100
            Number of posterior samples.
        verbose : bool, default=True
            Show progress bar.
            
        Returns
        -------
        Predictions
            Prediction results.
        """
        # Prepare input tensor
        x = torch.tensor(
            np.column_stack([data.coords, data.timestamps]),
            dtype=torch.float32,
            device=self.device,
        )
        
        # Make predictions
        mean, std = self._predict_batched(x, verbose=verbose)
        
        # Inverse transform to original scale
        if self.scalers is not None:
            coords_original = self.scalers.inverse_transform_coords(data.coords)
            timestamps_original = self.scalers.inverse_transform_time(data.timestamps)
            mean, std = self.scalers.inverse_transform_predictions(mean, std)
        else:
            coords_original = data.coords
            timestamps_original = data.timestamps
        
        # Compute confidence intervals
        lower_ci, upper_ci = self._compute_confidence_intervals(mean, std)
        
        return Predictions(
            mean=mean,
            std=std,
            lower_ci=lower_ci,
            upper_ci=upper_ci,
            coords=coords_original,
            timestamps=timestamps_original,
            grid_ids=data.grid_ids,
        )
    
    def predict_locations(
        self,
        coords: np.ndarray,
        timestamps: np.ndarray,
        normalized: bool = True,
        verbose: bool = True,
    ) -> Predictions:
        """
        Make predictions at arbitrary locations.
        
        Parameters
        ----------
        coords : np.ndarray
            Spatial coordinates, shape (N, 2).
        timestamps : np.ndarray
            Temporal values, shape (N,).
        normalized : bool, default=True
            Whether inputs are already normalized.
        verbose : bool
            Show progress bar.
            
        Returns
        -------
        Predictions
            Prediction results.
        """
        # Normalize if needed
        if not normalized and self.scalers is not None:
            coords_norm = (coords - self.scalers.coord_min) / (
                self.scalers.coord_max - self.scalers.coord_min
            )
            timestamps_norm = (timestamps - self.scalers.time_min) / (
                self.scalers.time_max - self.scalers.time_min
            )
        else:
            coords_norm = coords
            timestamps_norm = timestamps
        
        # Create input tensor
        x = torch.tensor(
            np.column_stack([coords_norm, timestamps_norm]),
            dtype=torch.float32,
            device=self.device,
        )
        
        # Predict
        mean, std = self._predict_batched(x, verbose=verbose)
        
        # Inverse transform
        if self.scalers is not None:
            if normalized:
                coords_original = self.scalers.inverse_transform_coords(coords)
                timestamps_original = self.scalers.inverse_transform_time(timestamps)
            else:
                coords_original = coords
                timestamps_original = timestamps
            mean, std = self.scalers.inverse_transform_predictions(mean, std)
        else:
            coords_original = coords
            timestamps_original = timestamps
        
        # Confidence intervals
        lower_ci, upper_ci = self._compute_confidence_intervals(mean, std)
        
        return Predictions(
            mean=mean,
            std=std,
            lower_ci=lower_ci,
            upper_ci=upper_ci,
            coords=coords_original,
            timestamps=timestamps_original,
        )
    
    def predict_grid(
        self,
        lat_range: Tuple[float, float],
        lon_range: Tuple[float, float],
        timestamps: Union[np.ndarray, List[float]],
        resolution: float = 0.001,
        normalized: bool = False,
        verbose: bool = True,
    ) -> Predictions:
        """
        Make predictions on a regular grid.
        
        Parameters
        ----------
        lat_range : Tuple[float, float]
            (min_lat, max_lat).
        lon_range : Tuple[float, float]
            (min_lon, max_lon).
        timestamps : array-like
            Timestamps to predict at.
        resolution : float, default=0.001
            Grid resolution in coordinate units.
        normalized : bool, default=False
            Whether inputs are normalized.
        verbose : bool
            Show progress bar.
            
        Returns
        -------
        Predictions
            Gridded predictions.
        """
        # Create grid
        lats = np.arange(lat_range[0], lat_range[1], resolution)
        lons = np.arange(lon_range[0], lon_range[1], resolution)
        times = np.array(timestamps)
        
        # Meshgrid
        lat_grid, lon_grid, time_grid = np.meshgrid(lats, lons, times, indexing='ij')
        
        coords = np.column_stack([lat_grid.ravel(), lon_grid.ravel()])
        timestamps_flat = time_grid.ravel()
        
        logger.info(
            f"Predicting on grid: {len(lats)}x{len(lons)}x{len(times)} = "
            f"{len(coords):,} points"
        )
        
        return self.predict_locations(
            coords=coords,
            timestamps=timestamps_flat,
            normalized=normalized,
            verbose=verbose,
        )
    
    def _predict_batched(
        self,
        x: torch.Tensor,
        verbose: bool = True,
        include_noise: bool = None,
        noise_source: str = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Make predictions in batches.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor, shape (N, 3).
        verbose : bool
            Show progress bar.
        include_noise : bool, optional
            Override default include_observation_noise setting.
        noise_source : str, optional
            Override default noise_source.

        Returns
        -------
        mean : np.ndarray
            Predictive means.
        std : np.ndarray
            Predictive standard deviations.
        """
        # Use instance defaults if not specified
        if include_noise is None:
            include_noise = self.include_observation_noise
        if noise_source is None:
            noise_source = self.noise_source

        n_samples = len(x)
        n_batches = (n_samples + self.batch_size - 1) // self.batch_size

        means = []
        stds = []

        iterator = range(n_batches)
        if verbose:
            iterator = tqdm(iterator, desc="Predicting")

        self.model.eval()
        with torch.no_grad():
            for i in iterator:
                start = i * self.batch_size
                end = min((i + 1) * self.batch_size, n_samples)
                x_batch = x[start:end]

                # Get posterior
                posterior = self.model(x_batch)
                mean = posterior.mean.cpu().numpy()
                var = posterior.variance.cpu().numpy()

                # Add observation noise if requested
                if include_noise:
                    noise_var = self.model.likelihood.noise_variance[noise_source].cpu().numpy()
                    var = var + noise_var

                std = np.sqrt(var)

                means.append(mean)
                stds.append(std)

        return np.concatenate(means), np.concatenate(stds)
    
    def _compute_confidence_intervals(
        self,
        mean: np.ndarray,
        std: np.ndarray,
    ) -> Tuple[Dict[float, np.ndarray], Dict[float, np.ndarray]]:
        """
        Compute confidence intervals.
        
        Parameters
        ----------
        mean : np.ndarray
            Predictive means.
        std : np.ndarray
            Predictive stds.
            
        Returns
        -------
        lower_ci : Dict[float, np.ndarray]
            Lower bounds.
        upper_ci : Dict[float, np.ndarray]
            Upper bounds.
        """
        lower_ci = {}
        upper_ci = {}
        
        for level in self.confidence_levels:
            if level not in self.Z_SCORES:
                # Compute from scipy if not cached
                from scipy import stats
                z = stats.norm.ppf((1 + level) / 2)
            else:
                z = self.Z_SCORES[level]
            
            lower_ci[level] = mean - z * std
            upper_ci[level] = mean + z * std
        
        return lower_ci, upper_ci
    
    def get_latent_posterior(
        self,
        x: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get the full latent posterior distribution.
        
        Parameters
        ----------
        x : torch.Tensor
            Input locations.
            
        Returns
        -------
        mean : torch.Tensor
            Posterior mean.
        covariance : torch.Tensor
            Posterior covariance matrix.
        """
        self.model.eval()
        with torch.no_grad():
            posterior = self.model(x.to(self.device))
            return posterior.mean, posterior.covariance_matrix
