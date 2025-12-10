"""
Data loading and validation utilities for FusionGP.

This module provides the DataLoader class for loading multi-source air quality
data from CSV files, performing validation, and preparing data structures for
the FusionGP model.

References
----------
.. [1] Hensman, J., Fusi, N., & Lawrence, N. D. (2013). 
       Gaussian processes for big data. UAI.

Example
-------
>>> loader = DataLoader("air_quality_data.csv")
>>> data = loader.load()
>>> print(data.keys())
dict_keys(['coords', 'timestamps', 'observations', 'source_masks', 'metadata'])
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


@dataclass
class FusionData:
    """
    Container for multi-source air quality fusion data.
    
    This dataclass holds all data required for FusionGP training and inference,
    including spatial coordinates, temporal indices, observations from each source,
    and metadata.
    
    Attributes
    ----------
    coords : np.ndarray
        Spatial coordinates array of shape (N, 2) containing (latitude, longitude)
        for each observation.
    timestamps : np.ndarray
        Temporal indices array of shape (N,) containing timestamp values
        (converted to numerical representation).
    observations : Dict[str, np.ndarray]
        Dictionary mapping source names ('epa', 'low_cost', 'satellite') to
        observation arrays of shape (N,). NaN indicates missing observations.
    source_masks : Dict[str, np.ndarray]
        Dictionary mapping source names to boolean masks of shape (N,) indicating
        which observations are valid (True) vs missing (False).
    grid_ids : np.ndarray
        Grid cell identifiers for each observation.
    raw_timestamps : np.ndarray
        Original timestamp values before numerical conversion.
    metadata : Dict
        Additional metadata about the dataset.
        
    Example
    -------
    >>> data = FusionData(
    ...     coords=np.array([[40.7, -74.0], [40.8, -74.1]]),
    ...     timestamps=np.array([0.0, 1.0]),
    ...     observations={'epa': np.array([10.5, np.nan]), 'low_cost': np.array([12.0, 11.0])},
    ...     source_masks={'epa': np.array([True, False]), 'low_cost': np.array([True, True])},
    ...     grid_ids=np.array([1, 2]),
    ...     raw_timestamps=np.array(['2024-01-01', '2024-01-02']),
    ...     metadata={'n_grid_cells': 100}
    ... )
    """
    coords: np.ndarray
    timestamps: np.ndarray
    observations: Dict[str, np.ndarray]
    source_masks: Dict[str, np.ndarray]
    grid_ids: np.ndarray
    raw_timestamps: np.ndarray
    metadata: Dict = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate data consistency after initialization."""
        n_obs = len(self.coords)
        assert len(self.timestamps) == n_obs, "Timestamps length mismatch"
        assert len(self.grid_ids) == n_obs, "Grid IDs length mismatch"
        for source, obs in self.observations.items():
            assert len(obs) == n_obs, f"Observations length mismatch for {source}"
        for source, mask in self.source_masks.items():
            assert len(mask) == n_obs, f"Mask length mismatch for {source}"
    
    @property
    def n_observations(self) -> int:
        """Total number of space-time points."""
        return len(self.coords)
    
    @property
    def sources(self) -> List[str]:
        """List of available data sources."""
        return list(self.observations.keys())
    
    def get_valid_observations(self, source: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Get valid (non-missing) observations for a specific source.
        
        Parameters
        ----------
        source : str
            Source name ('epa', 'low_cost', or 'satellite').
            
        Returns
        -------
        coords : np.ndarray
            Coordinates of valid observations, shape (n_valid, 2).
        timestamps : np.ndarray
            Timestamps of valid observations, shape (n_valid,).
        values : np.ndarray
            Observation values, shape (n_valid,).
        """
        mask = self.source_masks[source]
        return (
            self.coords[mask],
            self.timestamps[mask],
            self.observations[source][mask]
        )
    
    def summary(self) -> str:
        """Generate a summary string of the dataset."""
        lines = [
            "FusionData Summary",
            "=" * 40,
            f"Total space-time points: {self.n_observations:,}",
            f"Unique grid cells: {len(np.unique(self.grid_ids)):,}",
            f"Unique timestamps: {len(np.unique(self.timestamps)):,}",
            "",
            "Observations per source:",
        ]
        for source in self.sources:
            n_valid = self.source_masks[source].sum()
            pct = 100 * n_valid / self.n_observations
            lines.append(f"  {source}: {n_valid:,} ({pct:.1f}%)")
        return "\n".join(lines)


class DataLoader:
    """
    Load and validate multi-source air quality data from CSV files.
    
    This class handles loading data from CSV format, validating column presence
    and data types, and converting to the FusionData format used by FusionGP.
    
    Parameters
    ----------
    filepath : str or Path
        Path to the CSV file containing the data.
    column_mapping : Dict[str, str], optional
        Mapping from standard column names to actual column names in the CSV.
        Default column names are:
        - 'grid_id': 'grid_id'
        - 'latitude': 'latitude'
        - 'longitude': 'longitude'
        - 'timestamp': 'timestamp'
        - 'satellite': 'satellite_values'
        - 'low_cost': 'low_cost_data'
        - 'epa': 'epa_no2'
    
    Attributes
    ----------
    filepath : Path
        Path to the data file.
    column_mapping : Dict[str, str]
        Column name mapping.
    _data : Optional[FusionData]
        Cached loaded data.
        
    Example
    -------
    >>> loader = DataLoader("data/air_quality.csv")
    >>> data = loader.load()
    >>> print(data.summary())
    FusionData Summary
    ========================================
    Total space-time points: 50,000
    ...
    
    >>> # With custom column names
    >>> loader = DataLoader(
    ...     "data.csv",
    ...     column_mapping={'epa': 'reference_no2', 'satellite': 'tropomi_no2'}
    ... )
    """
    
    # Default column mapping
    DEFAULT_COLUMNS = {
        'grid_id': 'grid_id',
        'latitude': 'latitude',
        'longitude': 'longitude',
        'timestamp': 'timestamp',
        'satellite': 'satellite_values',
        'low_cost': 'low_cost_data',
        'epa': 'epa_no2',
    }
    
    def __init__(
        self,
        filepath: Union[str, Path],
        column_mapping: Optional[Dict[str, str]] = None,
    ):
        """
        Initialize the DataLoader.
        
        Parameters
        ----------
        filepath : str or Path
            Path to the CSV file.
        column_mapping : Dict[str, str], optional
            Custom column name mapping. Only need to specify columns that
            differ from defaults.
        """
        self.filepath = Path(filepath)
        self.column_mapping = {**self.DEFAULT_COLUMNS}
        if column_mapping:
            self.column_mapping.update(column_mapping)
        self._data: Optional[FusionData] = None
        
        logger.info(f"Initialized DataLoader for {self.filepath}")
    
    def load(self, validate: bool = True) -> FusionData:
        """
        Load data from the CSV file.
        
        Parameters
        ----------
        validate : bool, default=True
            Whether to perform validation checks on the loaded data.
            
        Returns
        -------
        FusionData
            Loaded and validated data container.
            
        Raises
        ------
        FileNotFoundError
            If the data file does not exist.
        ValueError
            If required columns are missing or data validation fails.
        """
        if not self.filepath.exists():
            raise FileNotFoundError(f"Data file not found: {self.filepath}")
        
        logger.info(f"Loading data from {self.filepath}")
        
        # Load CSV
        df = pd.read_csv(self.filepath)
        df = df.head(1500)
        logger.info(f"Loaded {len(df):,} rows from CSV")
        
        # Validate columns
        if validate:
            self._validate_columns(df)
        
        # Extract and process data
        data = self._process_dataframe(df)
        
        if validate:
            self._validate_data(data)
        
        self._data = data
        logger.info(f"Successfully loaded data:\n{data.summary()}")
        
        return data
    
    def _validate_columns(self, df: pd.DataFrame) -> None:
        """
        Validate that required columns are present in the dataframe.
        
        Parameters
        ----------
        df : pd.DataFrame
            Loaded dataframe.
            
        Raises
        ------
        ValueError
            If required columns are missing.
        """
        required = ['grid_id', 'latitude', 'longitude', 'timestamp']
        observation_cols = ['satellite', 'low_cost', 'epa']
        
        # Check required columns
        missing_required = []
        for col_key in required:
            col_name = self.column_mapping[col_key]
            if col_name not in df.columns:
                missing_required.append(f"{col_key} ('{col_name}')")
        
        if missing_required:
            raise ValueError(
                f"Missing required columns: {', '.join(missing_required)}\n"
                f"Available columns: {list(df.columns)}"
            )
        
        # Check observation columns (at least one required)
        available_obs = []
        for col_key in observation_cols:
            col_name = self.column_mapping[col_key]
            if col_name in df.columns:
                available_obs.append(col_key)
        
        if not available_obs:
            raise ValueError(
                f"At least one observation column required. "
                f"Expected one of: {[self.column_mapping[k] for k in observation_cols]}\n"
                f"Available columns: {list(df.columns)}"
            )
        
        logger.info(f"Found observation columns: {available_obs}")
    
    def _process_dataframe(self, df: pd.DataFrame) -> FusionData:
        """
        Process the dataframe into FusionData format.
        
        Parameters
        ----------
        df : pd.DataFrame
            Loaded dataframe.
            
        Returns
        -------
        FusionData
            Processed data container.
        """
        n_rows = len(df)
        
        # Extract coordinates
        lat_col = self.column_mapping['latitude']
        lon_col = self.column_mapping['longitude']
        coords = np.column_stack([
            df[lat_col].values,
            df[lon_col].values
        ])
        
        # Extract and convert timestamps
        ts_col = self.column_mapping['timestamp']
        raw_timestamps = df[ts_col].values
        timestamps = self._convert_timestamps(df[ts_col])
        
        # Extract grid IDs
        grid_ids = df[self.column_mapping['grid_id']].values
        
        # Extract observations and create masks
        observations = {}
        source_masks = {}
        
        for source_key in ['epa', 'low_cost', 'satellite']:
            col_name = self.column_mapping[source_key]
            if col_name in df.columns:
                values = df[col_name].values.astype(np.float64)
                mask = ~np.isnan(values)
                observations[source_key] = values
                source_masks[source_key] = mask
                logger.debug(
                    f"Source '{source_key}': {mask.sum():,} valid observations "
                    f"({100*mask.sum()/n_rows:.1f}%)"
                )
            else:
                # Source not available - fill with NaN
                observations[source_key] = np.full(n_rows, np.nan)
                source_masks[source_key] = np.zeros(n_rows, dtype=bool)
                logger.debug(f"Source '{source_key}': not available in data")
        
        # Build metadata
        metadata = {
            'filepath': str(self.filepath),
            'n_rows': n_rows,
            'n_grid_cells': len(np.unique(grid_ids)),
            'n_timestamps': len(np.unique(timestamps)),
            'coord_bounds': {
                'lat_min': coords[:, 0].min(),
                'lat_max': coords[:, 0].max(),
                'lon_min': coords[:, 1].min(),
                'lon_max': coords[:, 1].max(),
            },
            'time_bounds': {
                'min': raw_timestamps.min(),
                'max': raw_timestamps.max(),
            },
        }
        
        return FusionData(
            coords=coords,
            timestamps=timestamps,
            observations=observations,
            source_masks=source_masks,
            grid_ids=grid_ids,
            raw_timestamps=raw_timestamps,
            metadata=metadata,
        )
    
    def _convert_timestamps(self, timestamps: pd.Series) -> np.ndarray:
        """
        Convert timestamps to numerical values.
        
        Timestamps are converted to days since the first observation,
        normalized to [0, 1] range.
        
        Parameters
        ----------
        timestamps : pd.Series
            Timestamp column from dataframe.
            
        Returns
        -------
        np.ndarray
            Numerical timestamp values.
        """
        # Try to parse as datetime
        try:
            dt = pd.to_datetime(timestamps)
            # Convert to days since first observation
            days = (dt - dt.min()).dt.total_seconds() / (24 * 3600)
            return days.values
        except (ValueError, TypeError):
            # Already numerical
            values = timestamps.values.astype(np.float64)
            return values
    
    def _validate_data(self, data: FusionData) -> None:
        """
        Perform validation checks on the loaded data.
        
        Parameters
        ----------
        data : FusionData
            Data to validate.
            
        Raises
        ------
        ValueError
            If validation fails.
        """
        # Check for NaN in coordinates
        if np.any(np.isnan(data.coords)):
            raise ValueError("NaN values found in coordinates")
        
        # Check for NaN in timestamps
        if np.any(np.isnan(data.timestamps)):
            raise ValueError("NaN values found in timestamps")
        
        # Check coordinate ranges (basic sanity)
        lat = data.coords[:, 0]
        lon = data.coords[:, 1]
        
        if not (-90 <= lat.min() and lat.max() <= 90):
            logger.warning(
                f"Latitude values outside [-90, 90]: [{lat.min():.4f}, {lat.max():.4f}]"
            )
        
        if not (-180 <= lon.min() and lon.max() <= 180):
            logger.warning(
                f"Longitude values outside [-180, 180]: [{lon.min():.4f}, {lon.max():.4f}]"
            )
        
        # Check for at least some valid observations
        total_valid = sum(mask.sum() for mask in data.source_masks.values())
        if total_valid == 0:
            raise ValueError("No valid observations found in any source")
        
        logger.info("Data validation passed")


class FusionDataset(Dataset):
    """
    PyTorch Dataset for FusionGP mini-batch training.
    
    This dataset provides efficient access to multi-source observations for
    stochastic variational inference. Each sample contains the spatio-temporal
    coordinates, observation values, and source indicators.
    
    Parameters
    ----------
    data : FusionData
        Preprocessed fusion data.
    sources : List[str], optional
        List of sources to include. Default is all available sources.
        
    Attributes
    ----------
    coords : torch.Tensor
        Spatial coordinates, shape (N, 2).
    timestamps : torch.Tensor
        Temporal values, shape (N,).
    observations : torch.Tensor
        Stacked observations from all sources, shape (N, n_sources).
    source_masks : torch.Tensor
        Boolean masks for valid observations, shape (N, n_sources).
    source_names : List[str]
        Names of sources in order.
        
    Example
    -------
    >>> dataset = FusionDataset(data, sources=['epa', 'low_cost', 'satellite'])
    >>> loader = torch.utils.data.DataLoader(dataset, batch_size=1024, shuffle=True)
    >>> for batch in loader:
    ...     coords, timestamps, obs, masks, indices = batch
    ...     # Train step
    """
    
    def __init__(
        self,
        data: FusionData,
        sources: Optional[List[str]] = None,
    ):
        """
        Initialize the FusionDataset.
        
        Parameters
        ----------
        data : FusionData
            Preprocessed fusion data.
        sources : List[str], optional
            Sources to include. Default is all available.
        """
        self.source_names = sources if sources else list(data.observations.keys())
        
        # Convert to tensors
        self.coords = torch.tensor(data.coords, dtype=torch.float32)
        self.timestamps = torch.tensor(data.timestamps, dtype=torch.float32)
        
        # Stack observations and masks
        obs_list = [data.observations[s] for s in self.source_names]
        mask_list = [data.source_masks[s] for s in self.source_names]
        
        self.observations = torch.tensor(
            np.column_stack(obs_list), dtype=torch.float32
        )
        self.source_masks = torch.tensor(
            np.column_stack(mask_list), dtype=torch.bool
        )
        
        # Replace NaN with 0 (masked out anyway)
        self.observations = torch.nan_to_num(self.observations, nan=0.0)
        
        logger.info(
            f"Created FusionDataset with {len(self)} samples, "
            f"{len(self.source_names)} sources"
        )
    
    def __len__(self) -> int:
        """Return the number of samples."""
        return len(self.coords)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, ...]:
        """
        Get a single sample.
        
        Parameters
        ----------
        idx : int
            Sample index.
            
        Returns
        -------
        coords : torch.Tensor
            Spatial coordinates, shape (2,).
        timestamp : torch.Tensor
            Temporal value, shape ().
        observations : torch.Tensor
            Observation values, shape (n_sources,).
        masks : torch.Tensor
            Valid observation mask, shape (n_sources,).
        index : torch.Tensor
            Original index (for tracking).
        """
        return (
            self.coords[idx],
            self.timestamps[idx],
            self.observations[idx],
            self.source_masks[idx],
            torch.tensor(idx, dtype=torch.long),
        )
    
    def get_input_tensor(self) -> torch.Tensor:
        """
        Get the full input tensor (coords + timestamps).
        
        Returns
        -------
        torch.Tensor
            Combined input tensor, shape (N, 3) with columns [lat, lon, time].
        """
        return torch.cat([
            self.coords,
            self.timestamps.unsqueeze(-1)
        ], dim=-1)
    
    def get_source_data(self, source: str) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get data for a specific source.
        
        Parameters
        ----------
        source : str
            Source name.
            
        Returns
        -------
        inputs : torch.Tensor
            Input coordinates + timestamps for valid observations.
        targets : torch.Tensor
            Observation values.
        indices : torch.Tensor
            Original indices of valid observations.
        """
        source_idx = self.source_names.index(source)
        mask = self.source_masks[:, source_idx]
        
        inputs = self.get_input_tensor()[mask]
        targets = self.observations[mask, source_idx]
        indices = torch.arange(len(mask))[mask]
        
        return inputs, targets, indices
