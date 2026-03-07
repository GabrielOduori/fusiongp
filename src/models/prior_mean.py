"""
Prior mean functions for FusionGP.

This module provides spatially-varying mean functions that can incorporate
physics-based model output (e.g., ATMO-Plan) as the GP prior mean.

When using a deterministic model as prior mean, the GP learns:
    f(x) = m(x) + g(x)
where m(x) is the ATMO-Plan value and g(x) ~ GP(0, k) is the residual process.

This approach:
- Leverages physics-based spatial structure from deterministic models
- GP corrects biases and adds uncertainty quantification
- More data-efficient than learning spatial patterns from scratch

Classes
-------
GridPriorMean
    Spatially-varying mean from tabular grid data (CSV, DataFrame).
RasterMean
    Spatially-varying mean from a raster (GeoTIFF) file.
ATMOPlanMean
    Convenience class specifically for ATMO-Plan background maps.

Example
-------
>>> from src.models.prior_mean import ATMOPlanMean
>>>
>>> # Create mean function from ATMO-Plan raster
>>> prior_mean = ATMOPlanMean(
...     raster_path='data/atmo_plan_dublin.tif',
...     coord_scaler=data.scalers,
... )
>>>
>>> # Use in model
>>> model = FusionSVGP(n_inducing=500, prior_mean=prior_mean)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from gpytorch.means import Mean

logger = logging.getLogger(__name__)


class GridPriorMean(Mean):
    """
    Spatially-varying GP mean function from tabular grid data.

    This mean module looks up prior values from a grid of pre-computed model
    values using nearest-neighbor interpolation. Useful when the background
    model output is provided as tabular data (e.g., CSV with grid_id, lat, lon,
    model_value) rather than a raster file.

    The GP learns residuals on top of the prior:
        f(x) = m(x) + g(x)
    where m(x) is the grid prior value and g(x) ~ GP(0, k) is the residual.

    Parameters
    ----------
    grid_coords : np.ndarray
        Grid point coordinates, shape (N, 2) with [lat, lon].
        If `scalers` is provided, these should be original (unnormalized) coords.
        If `scalers` is None, these should already be normalized.
    grid_values : np.ndarray
        Prior mean values at each grid point, shape (N,).
    scalers : object, optional
        Data scalers from DataPreprocessor. If provided, grid coordinates
        will be normalized using scalers.coord_min and scalers.coord_scale.
    learnable_bias : bool, default=True
        Whether to learn a bias correction.
    learnable_scale : bool, default=False
        Whether to learn a scale factor.
    default_value : float, optional
        Value for NaN grid values. If None, uses mean of valid values.

    Example
    -------
    >>> # From a DataFrame with grid background values
    >>> grid_coords = df[['latitude', 'longitude']].values
    >>> grid_values = df['model_no2'].values
    >>> prior_mean = GridPriorMean(grid_coords, grid_values, scalers=preprocessor.scalers)
    >>>
    >>> # Use in model
    >>> model = FusionSVGP(n_inducing=500, prior_mean=prior_mean)
    """

    def __init__(
        self,
        grid_coords: np.ndarray,
        grid_values: np.ndarray,
        scalers=None,
        learnable_bias: bool = True,
        learnable_scale: bool = False,
        default_value: Optional[float] = None,
        normalize_values: bool = None,
        target_source: str = 'epa',
    ):
        super().__init__()

        # Normalize grid coordinates if scalers provided
        if scalers is not None:
            if np.isscalar(scalers.coord_scale):
                coord_scale = np.array([scalers.coord_scale, scalers.coord_scale])
            else:
                coord_scale = np.array([scalers.coord_scale[0], scalers.coord_scale[1]])

            coord_min = np.array([scalers.coord_min[0], scalers.coord_min[1]])
            grid_coords_norm = (grid_coords - coord_min) / coord_scale
        else:
            grid_coords_norm = grid_coords

        # Normalize grid values if targets are normalized
        grid_values_processed = grid_values.copy()
        self._values_normalized = False
        self._target_mean = 0.0
        self._target_std = 1.0

        # Auto-detect: normalize if scalers has normalize_targets=True
        if normalize_values is None and scalers is not None:
            normalize_values = getattr(scalers, 'normalize_targets', False)

        if normalize_values and scalers is not None:
            if hasattr(scalers, 'target_mean') and hasattr(scalers, 'target_std'):
                if target_source in scalers.target_mean and target_source in scalers.target_std:
                    self._target_mean = scalers.target_mean[target_source]
                    self._target_std = scalers.target_std[target_source]
                    grid_values_processed = (grid_values - self._target_mean) / self._target_std
                    self._values_normalized = True
                    logger.info(
                        f"GridPriorMean: Normalizing values using {target_source} stats "
                        f"(mean={self._target_mean:.2f}, std={self._target_std:.2f})"
                    )

        # Store grid data
        self.register_buffer(
            'grid_coords',
            torch.tensor(grid_coords_norm, dtype=torch.float32)
        )
        self.register_buffer(
            'grid_values',
            torch.tensor(grid_values_processed, dtype=torch.float32)
        )

        # Learnable parameters
        if learnable_bias:
            self.bias = nn.Parameter(torch.tensor(0.0))
        else:
            self.register_buffer('bias', torch.tensor(0.0))

        if learnable_scale:
            self.scale = nn.Parameter(torch.tensor(1.0))
        else:
            self.register_buffer('scale', torch.tensor(1.0))

        # Compute grid statistics (using processed values for consistency)
        valid_mask = ~np.isnan(grid_values_processed)
        valid_values = grid_values_processed[valid_mask]
        self.value_range = (float(valid_values.min()), float(valid_values.max()))
        self.default_value = default_value if default_value is not None else float(valid_values.mean())

        # Also store original value range for reporting
        valid_orig = grid_values[~np.isnan(grid_values)]
        self.value_range_original = (float(valid_orig.min()), float(valid_orig.max()))

        norm_status = "(normalized)" if self._values_normalized else "(original scale)"
        logger.info(
            f"GridPriorMean initialized with {len(grid_coords)} grid points\n"
            f"  Original value range: [{self.value_range_original[0]:.2f}, {self.value_range_original[1]:.2f}]\n"
            f"  Output value range {norm_status}: [{self.value_range[0]:.2f}, {self.value_range[1]:.2f}]"
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute prior mean values at input locations via nearest neighbor.

        Parameters
        ----------
        x : torch.Tensor
            Input coordinates, shape (N, D) where D >= 2.
            First two columns are [lat_normalized, lon_normalized].

        Returns
        -------
        torch.Tensor
            Prior mean values at each location, shape (N,).
        """
        # Extract spatial coordinates (detach to avoid gradient issues)
        query_coords = x[:, :2].detach()

        # For large inputs, use batched computation
        n_points = query_coords.shape[0]
        batch_size = 1000

        if n_points <= batch_size:
            # Direct computation for small inputs
            diffs = query_coords.unsqueeze(1) - self.grid_coords.unsqueeze(0)
            distances = torch.sum(diffs ** 2, dim=2)
            nearest_idx = torch.argmin(distances, dim=1)
            values = self.grid_values[nearest_idx]
        else:
            # Batched computation for large inputs
            values = torch.zeros(n_points, device=x.device, dtype=x.dtype)
            for i in range(0, n_points, batch_size):
                end_idx = min(i + batch_size, n_points)
                batch_coords = query_coords[i:end_idx]
                diffs = batch_coords.unsqueeze(1) - self.grid_coords.unsqueeze(0)
                distances = torch.sum(diffs ** 2, dim=2)
                nearest_idx = torch.argmin(distances, dim=1)
                values[i:end_idx] = self.grid_values[nearest_idx]

        # Handle NaN values
        nan_mask = torch.isnan(values)
        if nan_mask.any():
            values = torch.where(
                nan_mask,
                torch.tensor(self.default_value, device=values.device, dtype=values.dtype),
                values
            )

        # Apply learnable scale and bias
        return values * self.scale + self.bias

    def get_parameters(self) -> dict:
        """Get mean function parameters."""
        return {
            'n_grid_points': len(self.grid_coords),
            'value_range_original': getattr(self, 'value_range_original', self.value_range),
            'value_range_output': self.value_range,
            'values_normalized': getattr(self, '_values_normalized', False),
            'bias': self.bias.item() if isinstance(self.bias, torch.Tensor) else self.bias,
            'scale': self.scale.item() if isinstance(self.scale, torch.Tensor) else self.scale,
        }

    def __repr__(self) -> str:
        bias = self.bias.item() if isinstance(self.bias, torch.Tensor) else self.bias
        scale = self.scale.item() if isinstance(self.scale, torch.Tensor) else self.scale
        normalized = getattr(self, '_values_normalized', False)
        orig_range = getattr(self, 'value_range_original', self.value_range)
        return (
            f"GridPriorMean(\n"
            f"  grid_points: {len(self.grid_coords)}\n"
            f"  original_value_range: [{orig_range[0]:.2f}, {orig_range[1]:.2f}]\n"
            f"  output_value_range: [{self.value_range[0]:.2f}, {self.value_range[1]:.2f}]\n"
            f"  values_normalized: {normalized}\n"
            f"  learned_bias: {bias:.4f}\n"
            f"  learned_scale: {scale:.4f}\n"
            f")"
        )


class RasterMean(Mean):
    """
    Spatially-varying GP mean function from a raster file.

    This mean module samples values from a GeoTIFF raster at input
    coordinates, providing a spatially-varying prior mean. The GP
    then learns deviations from this background field.

    Parameters
    ----------
    raster_path : str or Path
        Path to the GeoTIFF raster file.
    coord_min : Tuple[float, float]
        (lat_min, lon_min) used for coordinate normalization.
    coord_scale : Tuple[float, float]
        (lat_scale, lon_scale) used for coordinate normalization.
    band : int, default=1
        Band number to read from raster.
    default_value : float, default=0.0
        Value to use for coordinates outside raster bounds.
    target_scale : float, default=1.0
        Scale factor to apply to raster values (for unit conversion).
    target_offset : float, default=0.0
        Offset to add to raster values (for unit conversion).
    learnable_scale : bool, default=False
        Whether to learn a scale factor during training.
    learnable_offset : bool, default=False
        Whether to learn an offset during training.

    Attributes
    ----------
    raster_data : torch.Tensor
        Loaded raster data.
    raster_bounds : tuple
        (lon_min, lat_min, lon_max, lat_max) in geographic coordinates.
    scale : nn.Parameter or float
        Scale factor for raster values.
    offset : nn.Parameter or float
        Offset for raster values.

    Example
    -------
    >>> mean = RasterMean(
    ...     raster_path='data/background_no2.tif',
    ...     coord_min=(53.25, -6.45),
    ...     coord_scale=(0.20, 0.35),
    ... )
    >>>
    >>> # Get mean at normalized coordinates
    >>> x = torch.tensor([[0.5, 0.5, 0.2]])  # [lat_norm, lon_norm, time]
    >>> m = mean(x)  # Returns ATMO-Plan value at that location
    """

    def __init__(
        self,
        raster_path: Union[str, Path],
        coord_min: Tuple[float, float],
        coord_scale: Tuple[float, float],
        band: int = 1,
        default_value: float = 0.0,
        target_scale: float = 1.0,
        target_offset: float = 0.0,
        learnable_scale: bool = False,
        learnable_offset: bool = False,
    ):
        super().__init__()

        self.raster_path = Path(raster_path)
        self.band = band
        self.default_value = default_value

        # Coordinate transformation parameters
        # FusionGP uses: normalized = (original - min) / scale
        # So: original = normalized * scale + min
        self.register_buffer('coord_min', torch.tensor(coord_min, dtype=torch.float32))
        self.register_buffer('coord_scale', torch.tensor(coord_scale, dtype=torch.float32))

        # Scale and offset for target values
        if learnable_scale:
            self.scale = nn.Parameter(torch.tensor(target_scale))
        else:
            self.register_buffer('scale', torch.tensor(target_scale))

        if learnable_offset:
            self.offset = nn.Parameter(torch.tensor(target_offset))
        else:
            self.register_buffer('offset', torch.tensor(target_offset))

        # Load raster data
        self._load_raster()

        logger.info(
            f"Created RasterMean from {self.raster_path.name}\n"
            f"  Shape: {self.raster_data.shape}\n"
            f"  Bounds: {self.raster_bounds}\n"
            f"  Value range: [{self.raster_data.min():.2f}, {self.raster_data.max():.2f}]"
        )

    def _load_raster(self) -> None:
        """Load raster data and metadata."""
        try:
            import rasterio
            from rasterio.crs import CRS
            from rasterio.warp import transform_bounds
        except ImportError:
            raise ImportError(
                "rasterio is required for RasterMean. "
                "Install with: pip install rasterio"
            )

        if not self.raster_path.exists():
            raise FileNotFoundError(f"Raster file not found: {self.raster_path}")

        with rasterio.open(self.raster_path) as src:
            # Read data
            data = src.read(self.band).astype(np.float32)

            # Handle nodata
            if src.nodata is not None:
                data[data == src.nodata] = np.nan

            # Get bounds in WGS84
            if src.crs and src.crs != CRS.from_epsg(4326):
                bounds = transform_bounds(src.crs, CRS.from_epsg(4326), *src.bounds)
            else:
                bounds = src.bounds

            # Store bounds: (lon_min, lat_min, lon_max, lat_max)
            self.raster_bounds = (bounds[0], bounds[1], bounds[2], bounds[3])

            # Store as tensor (flip to match lat increasing upward)
            # Note: np.flipud creates a view with negative stride, need copy for torch
            self.register_buffer(
                'raster_data',
                torch.tensor(np.flipud(data).copy(), dtype=torch.float32)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute mean values at input locations.

        Parameters
        ----------
        x : torch.Tensor
            Input coordinates, shape (N, D) where D >= 2.
            First two columns are [lat_normalized, lon_normalized].
            Additional columns (e.g., time) are ignored for spatial lookup.

        Returns
        -------
        torch.Tensor
            Mean values at each location, shape (N,).

        Notes
        -----
        Coordinates are detached before raster sampling to prevent unstable
        gradients from the bilinear interpolation (which uses .long() and
        torch.where operations). Gradients still flow through the learnable
        scale and offset parameters.
        """
        # Extract spatial coordinates (assumed: lat, lon in first two columns)
        # Detach to prevent unstable gradients from raster sampling
        # The raster values don't change, so learning to move coordinates
        # for different mean values is not meaningful anyway
        lat_norm = x[:, 0].detach()
        lon_norm = x[:, 1].detach()

        # Convert normalized coordinates back to geographic
        lat = lat_norm * self.coord_scale[0] + self.coord_min[0]
        lon = lon_norm * self.coord_scale[1] + self.coord_min[1]

        # Sample raster at these locations
        values = self._sample_raster(lat, lon)

        # Apply scale and offset (these ARE learnable)
        return values * self.scale + self.offset

    def _sample_raster(
        self,
        lat: torch.Tensor,
        lon: torch.Tensor,
    ) -> torch.Tensor:
        """
        Sample raster values using bilinear interpolation.

        Parameters
        ----------
        lat, lon : torch.Tensor
            Geographic coordinates.

        Returns
        -------
        torch.Tensor
            Sampled values.
        """
        lon_min, lat_min, lon_max, lat_max = self.raster_bounds
        h, w = self.raster_data.shape

        # Normalize to [0, 1] range within raster bounds
        x_norm = (lon - lon_min) / (lon_max - lon_min)
        y_norm = (lat - lat_min) / (lat_max - lat_min)

        # Convert to pixel coordinates
        # Note: raster_data is stored with lat increasing upward after flipud
        px = x_norm * (w - 1)
        py = y_norm * (h - 1)

        # Bilinear interpolation
        px0 = px.long().clamp(0, w - 2)
        py0 = py.long().clamp(0, h - 2)
        px1 = px0 + 1
        py1 = py0 + 1

        # Fractional parts
        fx = (px - px0.float()).clamp(0, 1)
        fy = (py - py0.float()).clamp(0, 1)

        # Sample four corners
        v00 = self.raster_data[py0, px0]
        v01 = self.raster_data[py0, px1]
        v10 = self.raster_data[py1, px0]
        v11 = self.raster_data[py1, px1]

        # Handle NaN values (from nodata)
        def safe_interp(v00, v01, v10, v11, fx, fy):
            # If any corner is NaN, use default value
            mask = torch.isnan(v00) | torch.isnan(v01) | torch.isnan(v10) | torch.isnan(v11)

            # Bilinear interpolation
            val = (v00 * (1 - fx) * (1 - fy) +
                   v01 * fx * (1 - fy) +
                   v10 * (1 - fx) * fy +
                   v11 * fx * fy)

            # Replace NaN results with default
            val = torch.where(mask, torch.tensor(self.default_value, device=val.device), val)
            return val

        values = safe_interp(v00, v01, v10, v11, fx, fy)

        # Handle out-of-bounds coordinates
        out_of_bounds = (
            (lon < lon_min) | (lon > lon_max) |
            (lat < lat_min) | (lat > lat_max)
        )
        values = torch.where(
            out_of_bounds,
            torch.tensor(self.default_value, device=values.device),
            values
        )

        return values

    def get_parameters(self) -> dict:
        """Get mean function parameters."""
        return {
            'raster_path': str(self.raster_path),
            'scale': self.scale.item() if isinstance(self.scale, torch.Tensor) else self.scale,
            'offset': self.offset.item() if isinstance(self.offset, torch.Tensor) else self.offset,
            'raster_bounds': self.raster_bounds,
        }

    def __repr__(self) -> str:
        return (
            f"RasterMean(\n"
            f"  raster: {self.raster_path.name}\n"
            f"  shape: {self.raster_data.shape}\n"
            f"  bounds: {self.raster_bounds}\n"
            f"  scale: {self.scale}, offset: {self.offset}\n"
            f")"
        )


class ATMOPlanMean(RasterMean):
    """
    ATMO-Plan background map as GP prior mean.

    This is a convenience class for using ATMO-Plan deterministic model
    output as the prior mean function. ATMO-Plan provides physics-based
    annual NO₂ concentrations that serve as the background from which
    the GP learns residuals.

    The complete model becomes:
        y = m_atmo(x) + f(x) + ε
    where:
        - m_atmo(x): ATMO-Plan background value at location x
        - f(x) ~ GP(0, k): Residual process (what the GP learns)
        - ε: Observation noise

    Parameters
    ----------
    raster_path : str or Path
        Path to ATMO-Plan GeoTIFF file.
    scalers : object
        Data scalers from DataPreprocessor (must have coord_min, coord_scale).
    band : int, default=1
        Band to read from raster.
    learnable_bias : bool, default=True
        Whether to learn a bias correction for ATMO-Plan.
        Recommended as deterministic models often have systematic biases.
    learnable_scale : bool, default=False
        Whether to learn a scale factor for ATMO-Plan values.

    Example
    -------
    >>> # Load data and get scalers
    >>> preprocessor = DataPreprocessor()
    >>> data = preprocessor.prepare_training_data(df)
    >>>
    >>> # Create ATMO-Plan mean
    >>> prior_mean = ATMOPlanMean(
    ...     raster_path='data/atmo_plan_dublin.tif',
    ...     scalers=preprocessor.scalers,
    ...     learnable_bias=True,
    ... )
    >>>
    >>> # Create model with ATMO-Plan prior
    >>> model = FusionSVGP(n_inducing=500, prior_mean=prior_mean)
    """

    def __init__(
        self,
        raster_path: Union[str, Path],
        scalers,
        band: int = 1,
        learnable_bias: bool = True,
        learnable_scale: bool = False,
    ):
        # Extract coordinate transformation from scalers
        # FusionGP convention: coords are [lat, lon]
        if hasattr(scalers, 'coord_min') and hasattr(scalers, 'coord_scale'):
            coord_min = (scalers.coord_min[0], scalers.coord_min[1])
            # coord_scale may be a single float (isotropic) or array
            if np.isscalar(scalers.coord_scale):
                coord_scale = (scalers.coord_scale, scalers.coord_scale)
            else:
                coord_scale = (scalers.coord_scale[0], scalers.coord_scale[1])
        else:
            raise ValueError(
                "scalers must have 'coord_min' and 'coord_scale' attributes. "
                "Use the scalers from DataPreprocessor."
            )

        # Initialize parent with ATMO-Plan specific defaults
        super().__init__(
            raster_path=raster_path,
            coord_min=coord_min,
            coord_scale=coord_scale,
            band=band,
            default_value=0.0,  # Outside bounds: assume zero contribution
            target_scale=1.0,
            target_offset=0.0,
            learnable_scale=learnable_scale,
            learnable_offset=learnable_bias,  # Bias correction
        )

        self.scalers = scalers

        logger.info(
            f"Created ATMOPlanMean prior\n"
            f"  Raster: {self.raster_path.name}\n"
            f"  Learnable bias: {learnable_bias}\n"
            f"  Learnable scale: {learnable_scale}"
        )

    def __repr__(self) -> str:
        bias = self.offset.item() if isinstance(self.offset, torch.Tensor) else self.offset
        scale = self.scale.item() if isinstance(self.scale, torch.Tensor) else self.scale
        # Handle NaN in value range
        valid_data = self.raster_data[~torch.isnan(self.raster_data)]
        if len(valid_data) > 0:
            vmin, vmax = valid_data.min().item(), valid_data.max().item()
            value_range = f"[{vmin:.2f}, {vmax:.2f}]"
        else:
            value_range = "[no valid data]"
        return (
            f"ATMOPlanMean(\n"
            f"  raster: {self.raster_path.name}\n"
            f"  shape: {tuple(self.raster_data.shape)}\n"
            f"  value range: {value_range}\n"
            f"  learned bias: {bias:.4f}\n"
            f"  learned scale: {scale:.4f}\n"
            f")"
        )
