"""
Cross-validation utilities for FusionGP.

This module provides spatial and temporal cross-validation strategies
for robust model evaluation in spatio-temporal settings.

Cross-validation strategies:
- Random K-fold: Standard random partitioning
- Spatial: Leave-out spatial blocks
- Temporal: Leave-out time periods
- Spatio-temporal: Combined spatial and temporal blocking

References
----------
.. [1] Roberts, D. R., et al. (2017). Cross-validation strategies for data 
       with temporal, spatial, hierarchical, or phylogenetic structure.
       Ecography.

Example
-------
>>> cv = SpatialCV(n_folds=5)
>>> for train_idx, test_idx in cv.split(data):
...     # Train and evaluate
...     pass
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Generator, List, Tuple

import numpy as np
from sklearn.cluster import KMeans
from sklearn.model_selection import KFold

from fusiongp.data.loader import FusionData

logger = logging.getLogger(__name__)


class CVStrategy(ABC):
    """
    Abstract base class for cross-validation strategies.
    """
    
    @abstractmethod
    def split(
        self,
        data: FusionData,
    ) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        """
        Generate train/test indices for each fold.
        
        Parameters
        ----------
        data : FusionData
            Data to split.
            
        Yields
        ------
        train_idx : np.ndarray
            Training indices.
        test_idx : np.ndarray
            Test indices.
        """
        pass
    
    @abstractmethod
    def get_n_splits(self) -> int:
        """Return the number of splits."""
        pass


class RandomCV(CVStrategy):
    """
    Random K-fold cross-validation.
    
    Standard random partitioning of data points.
    
    Parameters
    ----------
    n_folds : int, default=5
        Number of folds.
    shuffle : bool, default=True
        Whether to shuffle before splitting.
    random_state : int, default=42
        Random seed.
    """
    
    def __init__(
        self,
        n_folds: int = 5,
        shuffle: bool = True,
        random_state: int = 42,
    ):
        self.n_folds = n_folds
        self.shuffle = shuffle
        self.random_state = random_state
        self._kfold = KFold(
            n_splits=n_folds,
            shuffle=shuffle,
            random_state=random_state,
        )
    
    def split(
        self,
        data: FusionData,
    ) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        """Generate random K-fold splits."""
        indices = np.arange(data.n_observations)
        
        for train_idx, test_idx in self._kfold.split(indices):
            yield train_idx, test_idx
    
    def get_n_splits(self) -> int:
        return self.n_folds


class SpatialCV(CVStrategy):
    """
    Spatial block cross-validation.
    
    Groups nearby spatial locations together to test spatial generalization.
    This prevents information leakage from spatially autocorrelated data.
    
    Parameters
    ----------
    n_folds : int, default=5
        Number of spatial folds.
    random_state : int, default=42
        Random seed for clustering.
        
    Notes
    -----
    Uses K-means clustering on spatial coordinates to create spatially
    contiguous folds.
    """
    
    def __init__(
        self,
        n_folds: int = 5,
        random_state: int = 42,
    ):
        self.n_folds = n_folds
        self.random_state = random_state
    
    def split(
        self,
        data: FusionData,
    ) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        """Generate spatial block splits."""
        # Cluster based on unique grid cells
        unique_grids = np.unique(data.grid_ids)
        
        # Get centroid of each grid cell
        grid_coords = {}
        for gid in unique_grids:
            mask = data.grid_ids == gid
            grid_coords[gid] = data.coords[mask].mean(axis=0)
        
        coords_array = np.array([grid_coords[gid] for gid in unique_grids])
        
        # K-means clustering
        kmeans = KMeans(
            n_clusters=self.n_folds,
            random_state=self.random_state,
            n_init=10,
        )
        cluster_labels = kmeans.fit_predict(coords_array)
        
        # Map grid IDs to clusters
        grid_to_cluster = {
            gid: cluster_labels[i] 
            for i, gid in enumerate(unique_grids)
        }
        
        # Generate folds
        indices = np.arange(data.n_observations)
        observation_clusters = np.array([
            grid_to_cluster[gid] for gid in data.grid_ids
        ])
        
        for fold in range(self.n_folds):
            test_mask = observation_clusters == fold
            train_idx = indices[~test_mask]
            test_idx = indices[test_mask]
            yield train_idx, test_idx
    
    def get_n_splits(self) -> int:
        return self.n_folds


class TemporalCV(CVStrategy):
    """
    Temporal block cross-validation.
    
    Splits data by time periods to test temporal generalization.
    This respects the temporal ordering of data.
    
    Parameters
    ----------
    n_folds : int, default=5
        Number of temporal folds.
    strategy : str, default='block'
        Splitting strategy:
        - 'block': Contiguous time blocks
        - 'forward': Expanding window (train on past, test on future)
    """
    
    def __init__(
        self,
        n_folds: int = 5,
        strategy: str = 'block',
    ):
        self.n_folds = n_folds
        self.strategy = strategy
    
    def split(
        self,
        data: FusionData,
    ) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        """Generate temporal block splits."""
        # Sort by timestamp
        sorted_idx = np.argsort(data.timestamps)
        n = len(sorted_idx)
        
        if self.strategy == 'block':
            # Equal-sized time blocks
            fold_size = n // self.n_folds
            
            for fold in range(self.n_folds):
                start = fold * fold_size
                end = (fold + 1) * fold_size if fold < self.n_folds - 1 else n
                
                test_idx = sorted_idx[start:end]
                train_idx = np.concatenate([
                    sorted_idx[:start],
                    sorted_idx[end:]
                ])
                
                yield train_idx, test_idx
                
        elif self.strategy == 'forward':
            # Expanding window: train on past, test on future
            fold_size = n // (self.n_folds + 1)
            
            for fold in range(self.n_folds):
                train_end = (fold + 1) * fold_size
                test_start = train_end
                test_end = (fold + 2) * fold_size if fold < self.n_folds - 1 else n
                
                train_idx = sorted_idx[:train_end]
                test_idx = sorted_idx[test_start:test_end]
                
                yield train_idx, test_idx
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")
    
    def get_n_splits(self) -> int:
        return self.n_folds


class SpatioTemporalCV(CVStrategy):
    """
    Combined spatial and temporal cross-validation.
    
    Creates folds that vary in both space and time for the most
    stringent test of generalization.
    
    Parameters
    ----------
    n_spatial_folds : int, default=3
        Number of spatial folds.
    n_temporal_folds : int, default=3
        Number of temporal folds.
    random_state : int, default=42
        Random seed.
    """
    
    def __init__(
        self,
        n_spatial_folds: int = 3,
        n_temporal_folds: int = 3,
        random_state: int = 42,
    ):
        self.n_spatial_folds = n_spatial_folds
        self.n_temporal_folds = n_temporal_folds
        self.random_state = random_state
    
    def split(
        self,
        data: FusionData,
    ) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        """Generate spatio-temporal block splits."""
        # Create spatial clusters
        unique_grids = np.unique(data.grid_ids)
        grid_coords = {}
        for gid in unique_grids:
            mask = data.grid_ids == gid
            grid_coords[gid] = data.coords[mask].mean(axis=0)
        
        coords_array = np.array([grid_coords[gid] for gid in unique_grids])
        
        kmeans = KMeans(
            n_clusters=self.n_spatial_folds,
            random_state=self.random_state,
            n_init=10,
        )
        spatial_clusters = kmeans.fit_predict(coords_array)
        grid_to_spatial = {
            gid: spatial_clusters[i] 
            for i, gid in enumerate(unique_grids)
        }
        
        # Create temporal clusters
        sorted_times = np.sort(np.unique(data.timestamps))
        time_fold_size = len(sorted_times) // self.n_temporal_folds
        time_to_temporal = {}
        for i, t in enumerate(sorted_times):
            fold = min(i // time_fold_size, self.n_temporal_folds - 1)
            time_to_temporal[t] = fold
        
        # Assign each observation to spatial and temporal cluster
        obs_spatial = np.array([grid_to_spatial[gid] for gid in data.grid_ids])
        obs_temporal = np.array([time_to_temporal[t] for t in data.timestamps])
        
        # Generate all spatial-temporal fold combinations
        indices = np.arange(data.n_observations)
        
        for s_fold in range(self.n_spatial_folds):
            for t_fold in range(self.n_temporal_folds):
                test_mask = (obs_spatial == s_fold) & (obs_temporal == t_fold)
                train_idx = indices[~test_mask]
                test_idx = indices[test_mask]
                
                if len(test_idx) > 0:
                    yield train_idx, test_idx
    
    def get_n_splits(self) -> int:
        return self.n_spatial_folds * self.n_temporal_folds
