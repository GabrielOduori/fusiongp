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


@dataclass
class GridPredictions:
    """
    Container for gridded prediction results.

    This class wraps predictions made on an AnalysisGrid, providing
    convenient methods for reshaping to 2D grids and visualization.

    Attributes
    ----------
    mean : np.ndarray
        Predictive mean, shape (n_points,) or (n_points * n_timestamps,).
    std : np.ndarray
        Predictive standard deviation.
    lower_ci : Dict[float, np.ndarray]
        Lower confidence bounds at various levels.
    upper_ci : Dict[float, np.ndarray]
        Upper confidence bounds at various levels.
    grid : AnalysisGrid
        The analysis grid used for prediction.
    n_timestamps : int
        Number of timestamps in prediction.
    timestamps : np.ndarray
        Timestamp values.

    Example
    -------
    >>> preds = predictor.predict_analysis_grid(grid, timestamp=0.5)
    >>>
    >>> # Get 2D grid for visualization
    >>> mean_grid = preds.mean_grid  # shape (n_lat, n_lon)
    >>> std_grid = preds.std_grid
    >>>
    >>> # Plot directly
    >>> preds.plot(title='NO2 Prediction')
    >>>
    >>> # Export
    >>> preds.to_geotiff('prediction.tif')
    """
    mean: np.ndarray
    std: np.ndarray
    lower_ci: Dict[float, np.ndarray]
    upper_ci: Dict[float, np.ndarray]
    grid: 'AnalysisGrid'
    n_timestamps: int = 1
    timestamps: Optional[np.ndarray] = None

    @property
    def coords(self) -> np.ndarray:
        """Get flat coordinates array (lat, lon) for compatibility with legacy code."""
        lons, lats = self.grid.get_flat_coords()
        if self.n_timestamps > 1:
            # Tile for multiple timestamps
            lons = np.tile(lons, self.n_timestamps)
            lats = np.tile(lats, self.n_timestamps)
        return np.column_stack([lats, lons])

    @property
    def mean_grid(self) -> np.ndarray:
        """Get mean reshaped to grid, shape (n_lat, n_lon) or (n_t, n_lat, n_lon)."""
        return self.grid.reshape_to_grid(self.mean, self.n_timestamps)

    @property
    def std_grid(self) -> np.ndarray:
        """Get std reshaped to grid."""
        return self.grid.reshape_to_grid(self.std, self.n_timestamps)

    def lower_ci_grid(self, level: float = 0.95) -> np.ndarray:
        """Get lower CI reshaped to grid."""
        return self.grid.reshape_to_grid(self.lower_ci[level], self.n_timestamps)

    def upper_ci_grid(self, level: float = 0.95) -> np.ndarray:
        """Get upper CI reshaped to grid."""
        return self.grid.reshape_to_grid(self.upper_ci[level], self.n_timestamps)

    def plot(
        self,
        variable: str = 'mean',
        timestamp_idx: int = 0,
        ax=None,
        cmap: str = 'jet',
        vmin: Optional[float] = None,
        vmax: Optional[float] = None,
        title: Optional[str] = None,
        colorbar_label: Optional[str] = None,
        **kwargs,
    ):
        """
        Plot gridded predictions.

        Parameters
        ----------
        variable : str, default='mean'
            Which variable to plot: 'mean', 'std', 'lower_95', 'upper_95'.
        timestamp_idx : int, default=0
            Which timestamp to plot (for spatio-temporal predictions).
        ax : matplotlib.axes.Axes, optional
            Axes to plot on.
        cmap : str
            Colormap.
        vmin, vmax : float, optional
            Color scale limits.
        title : str, optional
            Plot title.
        colorbar_label : str, optional
            Colorbar label.
        **kwargs
            Additional arguments passed to grid.plot().

        Returns
        -------
        matplotlib.axes.Axes
            The axes.
        """
        # Get the data
        if variable == 'mean':
            data = self.mean_grid
        elif variable == 'std':
            data = self.std_grid
        elif variable.startswith('lower_'):
            level = float(variable.split('_')[1]) / 100
            data = self.lower_ci_grid(level)
        elif variable.startswith('upper_'):
            level = float(variable.split('_')[1]) / 100
            data = self.upper_ci_grid(level)
        else:
            raise ValueError(f"Unknown variable: {variable}")

        # Handle multi-timestamp
        if data.ndim == 3:
            data = data[timestamp_idx]

        # Default title
        if title is None:
            if self.n_timestamps > 1:
                title = f"{variable.capitalize()} (t={self.timestamps[timestamp_idx]:.2f})"
            else:
                title = f"{variable.capitalize()}"

        return self.grid.plot(
            data,
            ax=ax,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            title=title,
            colorbar_label=colorbar_label,
            **kwargs,
        )

    def plot_with_uncertainty(
        self,
        timestamp_idx: int = 0,
        figsize: Tuple[int, int] = (14, 5),
        vmin: Optional[float] = None,
        vmax: Optional[float] = None,
        std_vmax: Optional[float] = None,
    ):
        """
        Plot mean and uncertainty side by side.

        Parameters
        ----------
        timestamp_idx : int
            Which timestamp to plot.
        figsize : Tuple[int, int]
            Figure size.
        vmin, vmax : float, optional
            Color scale for mean.
        std_vmax : float, optional
            Max value for uncertainty colorbar.

        Returns
        -------
        matplotlib.figure.Figure
            The figure.
        """
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=figsize)

        # Mean
        self.plot(
            'mean',
            timestamp_idx=timestamp_idx,
            ax=axes[0],
            vmin=vmin,
            vmax=vmax,
            title='Prediction Mean',
            colorbar_label='µg/m³',
        )

        # Uncertainty
        self.plot(
            'std',
            timestamp_idx=timestamp_idx,
            ax=axes[1],
            cmap='magma',
            vmin=0,
            vmax=std_vmax,
            title='Prediction Uncertainty (Std)',
            colorbar_label='µg/m³',
        )

        plt.tight_layout()
        return fig

    def to_xarray(self) -> 'xr.Dataset':
        """
        Convert to xarray Dataset.

        Returns
        -------
        xr.Dataset
            Dataset with mean, std, and confidence intervals.
        """
        import xarray as xr

        coords = {
            'latitude': self.grid.info.lats,
            'longitude': self.grid.info.lons,
        }

        if self.n_timestamps > 1:
            coords['time'] = self.timestamps
            dims = ['time', 'latitude', 'longitude']
        else:
            dims = ['latitude', 'longitude']

        data_vars = {
            'mean': (dims, self.mean_grid),
            'std': (dims, self.std_grid),
        }

        for level in self.lower_ci:
            level_str = f'{int(level * 100)}'
            data_vars[f'lower_{level_str}'] = (dims, self.lower_ci_grid(level))
            data_vars[f'upper_{level_str}'] = (dims, self.upper_ci_grid(level))

        ds = xr.Dataset(data_vars, coords=coords)
        ds.attrs['grid_name'] = self.grid.name
        ds.attrs['resolution_m'] = self.grid.info.resolution_m

        return ds

    def to_geotiff(
        self,
        filepath: str,
        variable: str = 'mean',
        timestamp_idx: int = 0,
    ) -> None:
        """
        Save prediction to GeoTIFF.

        Parameters
        ----------
        filepath : str
            Output file path.
        variable : str
            Variable to save ('mean', 'std').
        timestamp_idx : int
            Timestamp index for spatio-temporal predictions.
        """
        if variable == 'mean':
            data = self.mean_grid
        elif variable == 'std':
            data = self.std_grid
        else:
            raise ValueError(f"Unknown variable: {variable}")

        if data.ndim == 3:
            data = data[timestamp_idx]

        self.grid.save_geotiff(data, filepath)

    def summary(self) -> str:
        """Generate summary string."""
        return (
            f"GridPredictions Summary\n"
            f"{'='*40}\n"
            f"Grid: {self.grid.name}\n"
            f"Shape: {self.grid.info.shape}\n"
            f"Resolution: {self.grid.info.resolution_m:.1f}m\n"
            f"N timestamps: {self.n_timestamps}\n"
            f"Mean range: [{self.mean.min():.2f}, {self.mean.max():.2f}]\n"
            f"Std range: [{self.std.min():.2f}, {self.std.max():.2f}]"
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
        task: Optional[str] = None,
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
        covariates = data.covariates if getattr(data, "covariates", None) is not None else np.empty((len(data.coords), 0))
        x = torch.tensor(
            np.column_stack([data.coords, data.timestamps, covariates]),
            dtype=torch.float32,
            device=self.device,
        )
        
        # Make predictions
        mean, std = self._predict_batched(x, verbose=verbose, task=task)
        
        # Inverse transform to original scale
        if self.scalers is not None:
            coords_original = self.scalers.inverse_transform_coords(data.coords)
            timestamps_original = self.scalers.inverse_transform_time(data.timestamps)
            mean, std = self.scalers.inverse_transform_predictions(
                mean, std, source=task or self.noise_source
            )
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
        covariates: Optional[np.ndarray] = None,
        normalized: bool = True,
        verbose: bool = True,
        task: Optional[str] = None,
    ) -> Predictions:
        """
        Make predictions at arbitrary locations.
        
        Parameters
        ----------
        coords : np.ndarray
            Spatial coordinates, shape (N, 2).
        timestamps : np.ndarray
            Temporal values, shape (N,).
        covariates : np.ndarray, optional
            Additional covariates, shape (N, K).
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
            # Match DataPreprocessor's isotropic scaling.
            coords_norm = (coords - self.scalers.coord_min) / self.scalers.coord_scale
            timestamps_norm = (timestamps - self.scalers.time_min) / (
                self.scalers.time_max - self.scalers.time_min
            )
            if covariates is not None:
                covariates_norm = self.scalers.transform_covariates(covariates)
            elif self.scalers.covariate_mean.size > 0:
                covariates_norm = np.zeros((len(coords), len(self.scalers.covariate_mean)))
            else:
                covariates_norm = np.empty((len(coords), 0))
        else:
            coords_norm = coords
            timestamps_norm = timestamps
            if covariates is not None:
                covariates_norm = covariates
            elif self.scalers is not None and self.scalers.covariate_mean.size > 0:
                covariates_norm = np.zeros((len(coords), len(self.scalers.covariate_mean)))
            else:
                covariates_norm = np.empty((len(coords), 0))
        
        # Create input tensor
        x = torch.tensor(
            np.column_stack([coords_norm, timestamps_norm, covariates_norm]),
            dtype=torch.float32,
            device=self.device,
        )
        
        # Predict
        mean, std = self._predict_batched(x, verbose=verbose, task=task)
        
        # Inverse transform
        if self.scalers is not None:
            if normalized:
                coords_original = self.scalers.inverse_transform_coords(coords)
                timestamps_original = self.scalers.inverse_transform_time(timestamps)
            else:
                coords_original = coords
                timestamps_original = timestamps
            mean, std = self.scalers.inverse_transform_predictions(
                mean, std, source=task or self.noise_source
            )
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
        task: Optional[str] = None,
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
            covariates=None,
            normalized=normalized,
            verbose=verbose,
            task=task,
        )

    def predict_analysis_grid(
        self,
        grid: 'AnalysisGrid',
        timestamp: float = 0.0,
        timestamps: Optional[np.ndarray] = None,
        include_covariates: bool = True,
        verbose: bool = True,
        task: Optional[str] = None,
    ) -> 'GridPredictions':
        """
        Make predictions on an AnalysisGrid.

        This is the recommended method for spatial predictions as it:
        - Uses proper meter-based resolution
        - Handles lat/lon aspect ratio correctly
        - Includes interpolated covariates
        - Returns results that can be directly reshaped to grid

        Parameters
        ----------
        grid : AnalysisGrid
            Analysis grid with covariates.
        timestamp : float, default=0.0
            Single timestamp for prediction.
        timestamps : np.ndarray, optional
            Multiple timestamps (creates spatio-temporal prediction).
        include_covariates : bool, default=True
            Include grid covariates in prediction.
        verbose : bool
            Show progress bar.
        task : str, optional
            Task name for multi-output models.

        Returns
        -------
        GridPredictions
            Predictions with grid metadata for easy visualization.

        Example
        -------
        >>> from src.data import AnalysisGrid, dublin_grid
        >>>
        >>> # Create grid and add covariates
        >>> grid = dublin_grid(resolution_m=100)
        >>> grid.add_covariate_points('traffic', traffic_lons, traffic_lats, traffic_vals)
        >>>
        >>> # Predict
        >>> preds = predictor.predict_analysis_grid(grid, timestamp=0.5)
        >>>
        >>> # Visualize (automatically reshaped)
        >>> grid.plot(preds.mean, title='NO2 Prediction', colorbar_label='µg/m³')
        """
        from src.data.grid import AnalysisGrid

        # Get prediction input from grid
        X = grid.get_prediction_input(
            timestamp=timestamp if timestamps is None else None,
            timestamps=timestamps,
            include_covariates=include_covariates,
            normalize=self.scalers is not None,
            scalers=self.scalers,
        )

        n_timestamps = 1 if timestamps is None else len(timestamps)

        # Pad with zeros if model expects more dimensions (covariates)
        # This handles the case where grid has no covariates but model was trained with them
        if self.scalers is not None and hasattr(self.scalers, 'covariate_mean'):
            n_expected_covariates = len(self.scalers.covariate_mean)
            n_current_dims = X.shape[1]
            n_expected_dims = 3 + n_expected_covariates  # lat, lon, time + covariates

            if n_current_dims < n_expected_dims:
                # Pad with zeros (mean covariates in normalized space = 0)
                n_pad = n_expected_dims - n_current_dims
                padding = np.zeros((X.shape[0], n_pad))
                X = np.column_stack([X, padding])
                logger.info(
                    f"Padded input from {n_current_dims} to {n_expected_dims} dims "
                    f"(added {n_pad} zero covariate columns)"
                )

        # Convert to tensor
        x_tensor = torch.tensor(X, dtype=torch.float32, device=self.device)

        # Make predictions
        mean, std = self._predict_batched(x_tensor, verbose=verbose, task=task)

        # Inverse transform if scalers available
        if self.scalers is not None:
            mean, std = self.scalers.inverse_transform_predictions(
                mean, std, source=task or self.noise_source
            )

        # Compute confidence intervals
        lower_ci, upper_ci = self._compute_confidence_intervals(mean, std)

        return GridPredictions(
            mean=mean,
            std=std,
            lower_ci=lower_ci,
            upper_ci=upper_ci,
            grid=grid,
            n_timestamps=n_timestamps,
            timestamps=timestamps if timestamps is not None else np.array([timestamp]),
        )

    def _predict_batched(
        self,
        x: torch.Tensor,
        verbose: bool = True,
        include_noise: bool = None,
        noise_source: str = None,
        task: Optional[str] = None,
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
        if task is not None:
            noise_source = task

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
                mean = posterior.mean
                var = posterior.variance
                if mean.ndim > 1:
                    if task is None:
                        raise ValueError("task must be specified for multi-output models.")
                    if not hasattr(self.model, "sources"):
                        raise ValueError("model missing sources for task selection.")
                    task_idx = self.model.sources.index(task)
                    mean = mean[:, task_idx]
                    var = var[:, task_idx]
                mean = mean.cpu().numpy()
                var = var.cpu().numpy()

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
