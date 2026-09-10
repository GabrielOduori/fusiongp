"""
Data preprocessing utilities for FusionGP.

This module provides the DataPreprocessor class for normalizing features,
handling missing values, and splitting data into train/validation/test sets.

The preprocessing pipeline follows best practices for Gaussian Process models:
1. Normalize spatial coordinates to [0, 1] range
2. Normalize temporal values to [0, 1] range  
3. Optionally standardize observation values
4. Create reproducible train/val/test splits

References
----------
.. [1] Williams, C. K., & Rasmussen, C. E. (2006). 
       Gaussian processes for machine learning. MIT Press.

Example
-------
>>> preprocessor = DataPreprocessor(normalize_coords=True, normalize_targets=True)
>>> train_data, val_data, test_data = preprocessor.fit_transform(
...     data, 
...     train_ratio=0.7, 
...     val_ratio=0.15
... )
>>> # Later, for new data:
>>> new_data_processed = preprocessor.transform(new_data)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from sklearn.model_selection import train_test_split

from src.data.loader import FusionData

logger = logging.getLogger(__name__)


@dataclass
class Scalers:
    """
    Container for normalization parameters.
    
    Stores the min/max or mean/std values used for normalization,
    enabling inverse transformation of predictions.
    
    Attributes
    ----------
    coord_min : np.ndarray
        Minimum coordinate values, shape (2,).
    coord_max : np.ndarray
        Maximum coordinate values, shape (2,).
    coord_scale : float
        Scaling factor for coordinates (max range across lat/lon).
        Used for isotropic scaling to preserve aspect ratio.
    time_min : float
        Minimum timestamp value.
    time_max : float
        Maximum timestamp value.
    target_mean : Dict[str, float]
        Mean observation value per source.
    target_std : Dict[str, float]
        Standard deviation of observations per source.
    covariate_mean : np.ndarray
        Mean of covariate columns.
    covariate_std : np.ndarray
        Standard deviation of covariate columns.
    covariate_names : List[str]
        Names of covariate columns.
    normalize_targets : bool
        Whether targets were normalized.
    """
    coord_min: np.ndarray
    coord_max: np.ndarray
    coord_scale: float
    time_min: float
    time_max: float
    target_mean: Dict[str, float] = field(default_factory=dict)
    target_std: Dict[str, float] = field(default_factory=dict)
    covariate_mean: np.ndarray = field(default_factory=lambda: np.array([]))
    covariate_std: np.ndarray = field(default_factory=lambda: np.array([]))
    covariate_names: List[str] = field(default_factory=list)
    normalize_targets: bool = False
    
    def inverse_transform_coords(self, coords: np.ndarray) -> np.ndarray:
        """
        Inverse transform normalized coordinates back to original scale.

        Uses isotropic scaling to preserve aspect ratio.

        Parameters
        ----------
        coords : np.ndarray
            Normalized coordinates, shape (..., 2).

        Returns
        -------
        np.ndarray
            Original-scale coordinates.
        """
        return coords * self.coord_scale + self.coord_min
    
    def inverse_transform_time(self, time: np.ndarray) -> np.ndarray:
        """
        Inverse transform normalized timestamps back to original scale.
        
        Parameters
        ----------
        time : np.ndarray
            Normalized timestamps.
            
        Returns
        -------
        np.ndarray
            Original-scale timestamps.
        """
        return time * (self.time_max - self.time_min) + self.time_min
    
    def inverse_transform_predictions(
        self,
        mean: np.ndarray,
        std: np.ndarray,
        source: str = 'epa'
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Inverse transform predicted mean and std to original scale.
        
        Parameters
        ----------
        mean : np.ndarray
            Predicted means in normalized scale.
        std : np.ndarray
            Predicted standard deviations in normalized scale.
        source : str
            Source name for target scaling.
            
        Returns
        -------
        mean_original : np.ndarray
            Means in original scale.
        std_original : np.ndarray
            Standard deviations in original scale.
        """
        if not self.normalize_targets or source not in self.target_std:
            return mean, std
        
        scale = self.target_std[source]
        offset = self.target_mean[source]
        
        return mean * scale + offset, std * scale

    def transform_covariates(self, covariates: np.ndarray) -> np.ndarray:
        """
        Standardize covariates using fitted mean/std.

        Parameters
        ----------
        covariates : np.ndarray
            Covariate matrix, shape (N, K).

        Returns
        -------
        np.ndarray
            Standardized covariates.
        """
        if covariates.size == 0:
            return covariates
        return (covariates - self.covariate_mean) / self.covariate_std


class DataPreprocessor:
    """
    Preprocessor for FusionGP data.
    
    Handles normalization of coordinates and observations, as well as
    train/validation/test splitting with various strategies.
    
    Parameters
    ----------
    normalize_coords : bool, default=True
        Whether to normalize spatial coordinates to [0, 1].
    normalize_time : bool, default=True
        Whether to normalize timestamps to [0, 1].
    normalize_targets : bool, default=False
        Whether to standardize observation values (zero mean, unit variance).
        Note: This is typically not needed for GPs with learned mean.
    
    Attributes
    ----------
    scalers : Optional[Scalers]
        Fitted normalization parameters.
    fitted : bool
        Whether the preprocessor has been fitted.
        
    Example
    -------
    >>> preprocessor = DataPreprocessor()
    >>> train, val, test = preprocessor.fit_transform(data, train_ratio=0.7, val_ratio=0.15)
    >>> print(f"Train: {train.n_observations}, Val: {val.n_observations}, Test: {test.n_observations}")
    """
    
    def __init__(
        self,
        normalize_coords: bool = True,
        normalize_time: bool = True,
        normalize_targets: bool = False,
    ):
        """
        Initialize the DataPreprocessor.
        
        Parameters
        ----------
        normalize_coords : bool
            Normalize spatial coordinates to [0, 1].
        normalize_time : bool
            Normalize timestamps to [0, 1].
        normalize_targets : bool
            Standardize observation values.
        """
        self.normalize_coords = normalize_coords
        self.normalize_time = normalize_time
        self.normalize_targets = normalize_targets
        
        self.scalers: Optional[Scalers] = None
        self.fitted = False
        
        logger.info(
            f"Initialized DataPreprocessor: "
            f"normalize_coords={normalize_coords}, "
            f"normalize_time={normalize_time}, "
            f"normalize_targets={normalize_targets}"
        )
    
    def fit(self, data: FusionData) -> 'DataPreprocessor':
        """
        Fit the preprocessor to the data.
        
        Computes normalization parameters from the input data.
        
        Parameters
        ----------
        data : FusionData
            Data to fit on.
            
        Returns
        -------
        self
            Fitted preprocessor.
        """
        logger.info("Fitting preprocessor...")
        
        # Compute coordinate bounds
        coord_min = data.coords.min(axis=0)
        coord_max = data.coords.max(axis=0)
        
        # Handle case where min == max (single location)
        coord_range = coord_max - coord_min
        coord_range[coord_range == 0] = 1.0
        
        # Compute time bounds
        time_min = float(data.timestamps.min())
        time_max = float(data.timestamps.max())
        if time_max == time_min:
            time_max = time_min + 1.0
        
        # Compute target statistics
        target_mean = {}
        target_std = {}
        
        for source, obs in data.observations.items():
            mask = data.source_masks[source]
            if mask.sum() > 0:
                valid_obs = obs[mask]
                target_mean[source] = float(np.nanmean(valid_obs))
                target_std[source] = float(np.nanstd(valid_obs))
                # Prevent division by zero
                if target_std[source] == 0:
                    target_std[source] = 1.0
            else:
                target_mean[source] = 0.0
                target_std[source] = 1.0

        # Compute isotropic scaling factor (max range across dimensions)
        coord_range = coord_max - coord_min
        coord_scale = float(coord_range.max())

        # Compute covariate statistics if present
        covariate_mean = np.array([])
        covariate_std = np.array([])
        covariate_names = []
        if getattr(data, "covariates", None) is not None:
            covariates = data.covariates
            covariate_mean = np.nanmean(covariates, axis=0)
            covariate_std = np.nanstd(covariates, axis=0)
            covariate_std[covariate_std == 0] = 1.0
            covariate_names = list(
                data.metadata.get("covariate_names", [f"cov_{i}" for i in range(covariates.shape[1])])
            )

        self.scalers = Scalers(
            coord_min=coord_min,
            coord_max=coord_max,
            coord_scale=coord_scale,
            time_min=time_min,
            time_max=time_max,
            target_mean=target_mean,
            target_std=target_std,
            covariate_mean=covariate_mean,
            covariate_std=covariate_std,
            covariate_names=covariate_names,
            normalize_targets=self.normalize_targets,
        )
        
        self.fitted = True
        logger.info(
            f"Preprocessor fitted: "
            f"coord_range={coord_max - coord_min}, "
            f"time_range={time_max - time_min:.2f}"
        )
        
        return self
    
    def transform(self, data: FusionData) -> FusionData:
        """
        Transform data using fitted parameters.
        
        Parameters
        ----------
        data : FusionData
            Data to transform.
            
        Returns
        -------
        FusionData
            Transformed data.
            
        Raises
        ------
        RuntimeError
            If preprocessor has not been fitted.
        """
        if not self.fitted:
            raise RuntimeError("Preprocessor must be fitted before transform")
        
        # Transform coordinates
        if self.normalize_coords:
            # Use isotropic scaling (same scale for lat and lon) to preserve aspect ratio
            coords = (data.coords - self.scalers.coord_min) / self.scalers.coord_scale
        else:
            coords = data.coords.copy()
        
        # Transform timestamps
        if self.normalize_time:
            timestamps = (data.timestamps - self.scalers.time_min) / (
                self.scalers.time_max - self.scalers.time_min
            )
        else:
            timestamps = data.timestamps.copy()
        
        # Transform observations
        observations = {}
        for source, obs in data.observations.items():
            if self.normalize_targets and source in self.scalers.target_mean:
                obs_transformed = (obs - self.scalers.target_mean[source]) / (
                    self.scalers.target_std[source]
                )
            else:
                obs_transformed = obs.copy()
            observations[source] = obs_transformed

        # Transform covariates
        covariates = None
        if getattr(data, "covariates", None) is not None:
            covariates = self.scalers.transform_covariates(data.covariates)
            covariates = np.nan_to_num(covariates, nan=0.0)
        
        # Create transformed FusionData
        transformed = FusionData(
            coords=coords,
            timestamps=timestamps,
            observations=observations,
            source_masks={k: v.copy() for k, v in data.source_masks.items()},
            grid_ids=data.grid_ids.copy(),
            raw_timestamps=data.raw_timestamps.copy(),
            covariates=covariates,
            metadata={**data.metadata, 'preprocessed': True},
        )
        
        return transformed
    
    def fit_transform(
        self,
        data: FusionData,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: Optional[float] = None,
        split_strategy: str = 'random',
        random_seed: int = 42,
        epa_holdout_grid_ids: Optional[List[str]] = None,
    ) -> Tuple[FusionData, FusionData, FusionData]:
        """
        Fit the preprocessor and split data into train/val/test sets.
        
        Parameters
        ----------
        data : FusionData
            Input data.
        train_ratio : float, default=0.7
            Fraction of data for training.
        val_ratio : float, default=0.15
            Fraction of data for validation.
        test_ratio : float, optional
            Fraction of data for testing. If None, computed as 1 - train - val.
        split_strategy : str, default='random'
            Splitting strategy: 'random', 'temporal', 'spatial', or
            'epa_station_holdout'.
        random_seed : int, default=42
            Random seed for reproducibility.
        epa_holdout_grid_ids : list[str], optional
            Explicit EPA station grid IDs to hold out when split_strategy is
            'epa_station_holdout'. If omitted, stations are chosen randomly.
            
        Returns
        -------
        train_data : FusionData
            Training data.
        val_data : FusionData
            Validation data.
        test_data : FusionData
            Test data.
        """
        # Compute test ratio if not provided
        if test_ratio is None:
            test_ratio = 1.0 - train_ratio - val_ratio
        
        # Validate ratios
        total = train_ratio + val_ratio + test_ratio
        if not np.isclose(total, 1.0):
            logger.warning(f"Split ratios sum to {total}, normalizing...")
            train_ratio /= total
            val_ratio /= total
            test_ratio /= total
        
        logger.info(
            f"Splitting data: train={train_ratio:.1%}, "
            f"val={val_ratio:.1%}, test={test_ratio:.1%}, "
            f"strategy={split_strategy}"
        )
        
        # Get split indices
        n = data.n_observations
        indices = np.arange(n)
        
        if split_strategy == 'random':
            train_idx, temp_idx = train_test_split(
                indices, 
                train_size=train_ratio, 
                random_state=random_seed
            )
            val_size = val_ratio / (val_ratio + test_ratio)
            val_idx, test_idx = train_test_split(
                temp_idx, 
                train_size=val_size, 
                random_state=random_seed
            )
            
        elif split_strategy == 'temporal':
            # Sort by timestamp and split sequentially
            sorted_idx = np.argsort(data.timestamps)
            n_train = int(n * train_ratio)
            n_val = int(n * val_ratio)
            
            train_idx = sorted_idx[:n_train]
            val_idx = sorted_idx[n_train:n_train + n_val]
            test_idx = sorted_idx[n_train + n_val:]
            
        elif split_strategy == 'spatial':
            # Split by grid cells
            unique_grids = np.unique(data.grid_ids)
            np.random.seed(random_seed)
            np.random.shuffle(unique_grids)
            
            n_grids = len(unique_grids)
            n_train_grids = int(n_grids * train_ratio)
            n_val_grids = int(n_grids * val_ratio)
            
            train_grids = set(unique_grids[:n_train_grids])
            val_grids = set(unique_grids[n_train_grids:n_train_grids + n_val_grids])
            test_grids = set(unique_grids[n_train_grids + n_val_grids:])
            
            train_idx = indices[np.isin(data.grid_ids, list(train_grids))]
            val_idx = indices[np.isin(data.grid_ids, list(val_grids))]
            test_idx = indices[np.isin(data.grid_ids, list(test_grids))]

        elif split_strategy == 'epa_station_holdout':
            epa_mask = data.source_masks.get('epa')
            if epa_mask is None or not np.any(epa_mask):
                raise ValueError("epa_station_holdout split requires at least one EPA observation")

            epa_grids = np.unique(data.grid_ids[epa_mask])
            if epa_holdout_grid_ids:
                requested = np.array([str(grid_id) for grid_id in epa_holdout_grid_ids])
                missing = sorted(set(requested) - set(map(str, epa_grids)))
                if missing:
                    raise ValueError(
                        "EPA holdout grid IDs are not EPA station grids: "
                        + ", ".join(missing)
                    )
                test_grids = set(requested)
            else:
                rng = np.random.default_rng(random_seed)
                epa_grids = epa_grids.copy()
                rng.shuffle(epa_grids)

                n_test_grids = max(1, int(round(len(epa_grids) * test_ratio)))
                n_test_grids = min(n_test_grids, max(1, len(epa_grids) - 1))
                test_grids = set(epa_grids[:n_test_grids])

            test_grid_mask = np.isin(data.grid_ids, list(test_grids))
            test_idx = indices[test_grid_mask]
            remaining_idx = indices[~test_grid_mask]
            if len(remaining_idx) == 0:
                raise ValueError("epa_station_holdout left no rows for training/validation")

            val_fraction = val_ratio / max(train_ratio + val_ratio, 1e-12)
            train_idx, val_idx = train_test_split(
                remaining_idx,
                test_size=val_fraction,
                random_state=random_seed,
            )
            self.epa_holdout_grid_ids_ = np.array(sorted(test_grids))
            logger.info(
                "EPA station holdout split: held out %d/%d EPA station grid cells: %s",
                len(test_grids),
                len(epa_grids),
                ", ".join(map(str, sorted(test_grids))),
            )
	            
        else:
            raise ValueError(f"Unknown split strategy: {split_strategy}")
        
        # Store split indices so callers can reconstruct raw subsets (e.g. GPKF)
        self.train_indices_ = train_idx
        self.val_indices_ = val_idx
        self.test_indices_ = test_idx

        # Fit on training data only
        train_subset = self._subset_data(data, train_idx)
        self.fit(train_subset)
        
        # Transform all splits
        train_data = self.transform(train_subset)
        val_data = self.transform(self._subset_data(data, val_idx))
        test_data = self.transform(self._subset_data(data, test_idx))
        
        logger.info(
            f"Split complete: "
            f"train={train_data.n_observations:,}, "
            f"val={val_data.n_observations:,}, "
            f"test={test_data.n_observations:,}"
        )
        
        return train_data, val_data, test_data
    
    def _subset_data(self, data: FusionData, indices: np.ndarray) -> FusionData:
        """
        Create a subset of the data using given indices.
        
        Parameters
        ----------
        data : FusionData
            Original data.
        indices : np.ndarray
            Indices to include.
            
        Returns
        -------
        FusionData
            Subset of the data.
        """
        covariates = None
        if getattr(data, "covariates", None) is not None:
            covariates = data.covariates[indices]

        return FusionData(
            coords=data.coords[indices],
            timestamps=data.timestamps[indices],
            observations={k: v[indices] for k, v in data.observations.items()},
            source_masks={k: v[indices] for k, v in data.source_masks.items()},
            grid_ids=data.grid_ids[indices],
            raw_timestamps=data.raw_timestamps[indices],
            covariates=covariates,
            metadata={**data.metadata, 'subset_size': len(indices)},
        )
    
    def get_scalers(self) -> Scalers:
        """
        Get the fitted scalers.
        
        Returns
        -------
        Scalers
            Normalization parameters.
            
        Raises
        ------
        RuntimeError
            If preprocessor has not been fitted.
        """
        if not self.fitted:
            raise RuntimeError("Preprocessor must be fitted first")
        return self.scalers
