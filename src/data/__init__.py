"""
Data loading and preprocessing utilities for FusionGP.

This module provides utilities for:
- Loading and validating multi-source air quality data
- Preprocessing and normalization
- Train/validation/test splitting
- Mini-batch generation for stochastic training
- Analysis grid definition and covariate interpolation
- GeoTIFF raster loading and sampling

Classes
-------
DataLoader
    Load and validate input data from CSV files.
DataPreprocessor
    Normalize features, handle missing values, and split data.
FusionDataset
    PyTorch Dataset for mini-batch training.
AnalysisGrid
    Define prediction grids with covariate interpolation.
GridInfo
    Container for grid metadata.

Functions
---------
dublin_grid
    Create default Dublin study area grid.
sample_geotiff_at_points
    Sample GeoTIFF raster values at point locations.
"""

from src.data.loader import DataLoader, FusionDataset
from src.data.preprocessor import DataPreprocessor
from src.data.grid import AnalysisGrid, GridInfo, dublin_grid, sample_geotiff_at_points

__all__ = [
    "DataLoader",
    "DataPreprocessor",
    "FusionDataset",
    "AnalysisGrid",
    "GridInfo",
    "dublin_grid",
    "sample_geotiff_at_points",
]
