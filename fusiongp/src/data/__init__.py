"""
Data loading and preprocessing utilities for FusionGP.

This module provides utilities for:
- Loading and validating multi-source air quality data
- Preprocessing and normalization
- Train/validation/test splitting
- Mini-batch generation for stochastic training

Classes
-------
DataLoader
    Load and validate input data from CSV files.
DataPreprocessor
    Normalize features, handle missing values, and split data.
FusionDataset
    PyTorch Dataset for mini-batch training.
"""

from fusiongp.data.loader import DataLoader, FusionDataset
from fusiongp.data.preprocessor import DataPreprocessor

__all__ = [
    "DataLoader",
    "DataPreprocessor",
    "FusionDataset",
]
