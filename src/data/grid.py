"""
Analysis grid for FusionGP spatial predictions.

This module provides the AnalysisGrid class for defining prediction grids
with proper geographic handling, covariate interpolation, and visualization.

The grid handles:
- Study area definition with bounding box
- Resolution in meters (geographically meaningful)
- Latitude-dependent longitude scaling (aspect ratio correction)
- Covariate interpolation to grid via IDW
- Grid shape tracking for easy visualization
- Export to various formats (GeoTIFF, NetCDF, CSV)

Example
-------
>>> # Define Dublin study area
>>> grid = AnalysisGrid(
...     lon_min=-6.45, lon_max=-6.10,
...     lat_min=53.25, lat_max=53.45,
...     resolution_m=100.0
... )
>>>
>>> # Add covariates
>>> grid.add_covariate_points('traffic', traffic_lons, traffic_lats, traffic_values)
>>> grid.add_covariate_points('lur', lur_lons, lur_lats, lur_values)
>>>
>>> # Get prediction input tensor
>>> X = grid.get_prediction_input(timestamp=0.5)
>>>
>>> # After prediction, reshape to grid
>>> pred_map = grid.reshape_to_grid(predictions)
>>> grid.plot(pred_map, title='NO2 Prediction')
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch

logger = logging.getLogger(__name__)


def _check_rasterio():
    """Check if rasterio is available."""
    try:
        import rasterio
        return rasterio
    except ImportError:
        raise ImportError(
            "rasterio is required for GeoTIFF operations. "
            "Install with: pip install rasterio"
        )


# Earth constants for coordinate conversion
METERS_PER_DEGREE_LAT = 111_320.0  # Approximately constant


def meters_per_degree_lon(lat: float) -> float:
    """
    Calculate meters per degree longitude at a given latitude.

    Parameters
    ----------
    lat : float
        Latitude in degrees.

    Returns
    -------
    float
        Meters per degree longitude.
    """
    return METERS_PER_DEGREE_LAT * np.cos(np.deg2rad(lat))


@dataclass
class GridInfo:
    """
    Container for grid metadata.

    Attributes
    ----------
    lon_min, lon_max : float
        Longitude bounds.
    lat_min, lat_max : float
        Latitude bounds.
    resolution_m : float
        Grid resolution in meters.
    n_lon, n_lat : int
        Number of grid cells in each dimension.
    lon_res, lat_res : float
        Resolution in degrees for each dimension.
    lons, lats : np.ndarray
        1D arrays of longitude and latitude values.
    """
    lon_min: float
    lon_max: float
    lat_min: float
    lat_max: float
    resolution_m: float
    n_lon: int
    n_lat: int
    lon_res: float
    lat_res: float
    lons: np.ndarray
    lats: np.ndarray

    @property
    def center(self) -> Tuple[float, float]:
        """Get grid center (lon, lat)."""
        return (
            (self.lon_min + self.lon_max) / 2,
            (self.lat_min + self.lat_max) / 2
        )

    @property
    def extent(self) -> List[float]:
        """Get extent for matplotlib imshow [lon_min, lon_max, lat_min, lat_max]."""
        return [self.lon_min, self.lon_max, self.lat_min, self.lat_max]

    @property
    def shape(self) -> Tuple[int, int]:
        """Get grid shape (n_lat, n_lon) for array indexing."""
        return (self.n_lat, self.n_lon)

    @property
    def n_points(self) -> int:
        """Total number of grid points."""
        return self.n_lat * self.n_lon

    def __repr__(self) -> str:
        return (
            f"GridInfo(\n"
            f"  bounds: [{self.lon_min:.4f}, {self.lon_max:.4f}] x "
            f"[{self.lat_min:.4f}, {self.lat_max:.4f}]\n"
            f"  resolution: {self.resolution_m:.1f}m\n"
            f"  shape: {self.n_lon} x {self.n_lat} = {self.n_points:,} points\n"
            f")"
        )


class AnalysisGrid:
    """
    Analysis grid for spatial predictions with covariate interpolation.

    This class defines a regular prediction grid over a study area,
    handles covariate interpolation to grid points, and provides
    utilities for visualization and export.

    Parameters
    ----------
    lon_min, lon_max : float
        Longitude bounds of study area.
    lat_min, lat_max : float
        Latitude bounds of study area.
    resolution_m : float, default=100.0
        Grid resolution in meters.
    name : str, optional
        Name for this grid (e.g., 'Dublin').

    Attributes
    ----------
    info : GridInfo
        Grid metadata.
    lon_grid, lat_grid : np.ndarray
        2D meshgrid arrays of coordinates.
    covariates : Dict[str, np.ndarray]
        Dictionary of covariate grids.

    Example
    -------
    >>> grid = AnalysisGrid(
    ...     lon_min=-6.45, lon_max=-6.10,
    ...     lat_min=53.25, lat_max=53.45,
    ...     resolution_m=100.0,
    ...     name='Dublin'
    ... )
    >>> print(grid.info)
    GridInfo(
      bounds: [-6.4500, -6.1000] x [53.2500, 53.4500]
      resolution: 100.0m
      shape: 525 x 223 = 117075 points
    )
    """

    def __init__(
        self,
        lon_min: float,
        lon_max: float,
        lat_min: float,
        lat_max: float,
        resolution_m: float = 100.0,
        name: Optional[str] = None,
    ):
        """
        Initialize the analysis grid.

        Parameters
        ----------
        lon_min, lon_max : float
            Longitude bounds.
        lat_min, lat_max : float
            Latitude bounds.
        resolution_m : float
            Resolution in meters.
        name : str, optional
            Grid name.
        """
        self.name = name or "AnalysisGrid"

        # Calculate center latitude for aspect ratio correction
        center_lat = (lat_min + lat_max) / 2

        # Convert resolution from meters to degrees
        lat_res = resolution_m / METERS_PER_DEGREE_LAT
        lon_res = resolution_m / meters_per_degree_lon(center_lat)

        # Create coordinate arrays
        lons = np.arange(lon_min, lon_max, lon_res)
        lats = np.arange(lat_min, lat_max, lat_res)

        n_lon = len(lons)
        n_lat = len(lats)

        # Store grid info
        self.info = GridInfo(
            lon_min=lon_min,
            lon_max=lon_max,
            lat_min=lat_min,
            lat_max=lat_max,
            resolution_m=resolution_m,
            n_lon=n_lon,
            n_lat=n_lat,
            lon_res=lon_res,
            lat_res=lat_res,
            lons=lons,
            lats=lats,
        )

        # Create meshgrid (lat varies along rows, lon along columns)
        self.lon_grid, self.lat_grid = np.meshgrid(lons, lats)

        # Flat coordinates for prediction
        self._flat_lons = self.lon_grid.ravel()
        self._flat_lats = self.lat_grid.ravel()

        # Covariate storage
        self.covariates: Dict[str, np.ndarray] = {}
        self._covariate_points: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}

        logger.info(
            f"Created {self.name}: {n_lon}x{n_lat} = {n_lon * n_lat:,} points "
            f"at {resolution_m:.1f}m resolution"
        )

    @classmethod
    def from_latlon_arrays(
        cls,
        lats: np.ndarray,
        lons: np.ndarray,
        name: Optional[str] = None,
    ) -> "AnalysisGrid":
        """
        Create an AnalysisGrid directly from latitude/longitude arrays.

        This preserves the native grid shape (useful for matching a prior grid).
        """
        lats = np.asarray(lats)
        lons = np.asarray(lons)
        if lats.ndim != 1 or lons.ndim != 1:
            raise ValueError("lats and lons must be 1D arrays")

        lat_min = float(lats.min())
        lat_max = float(lats.max())
        lon_min = float(lons.min())
        lon_max = float(lons.max())
        center_lat = (lat_min + lat_max) / 2

        # Estimate resolution from median spacing
        lat_diffs = np.diff(np.sort(lats))
        lon_diffs = np.diff(np.sort(lons))
        lat_res = float(np.median(lat_diffs)) if len(lat_diffs) else 0.0
        lon_res = float(np.median(lon_diffs)) if len(lon_diffs) else 0.0

        resolution_m = (
            lat_res * METERS_PER_DEGREE_LAT
            if lat_res > 0
            else 0.0
        )
        if lon_res > 0:
            resolution_m = max(
                resolution_m,
                lon_res * meters_per_degree_lon(center_lat),
            )

        grid = cls.__new__(cls)
        grid.name = name or "AnalysisGrid"
        grid.info = GridInfo(
            lon_min=lon_min,
            lon_max=lon_max,
            lat_min=lat_min,
            lat_max=lat_max,
            resolution_m=resolution_m,
            n_lon=len(lons),
            n_lat=len(lats),
            lon_res=lon_res,
            lat_res=lat_res,
            lons=np.sort(lons),
            lats=np.sort(lats),
        )
        grid.lon_grid, grid.lat_grid = np.meshgrid(grid.info.lons, grid.info.lats)
        grid._flat_lons = grid.lon_grid.ravel()
        grid._flat_lats = grid.lat_grid.ravel()
        grid.covariates = {}
        grid._covariate_points = {}

        logger.info(
            f"Created {grid.name}: {grid.info.n_lon}x{grid.info.n_lat} = "
            f"{grid.info.n_lon * grid.info.n_lat:,} points "
            f"from native lat/lon arrays"
        )
        return grid

    @classmethod
    def from_bounds(
        cls,
        bounds: Tuple[float, float, float, float],
        resolution_m: float = 100.0,
        name: Optional[str] = None,
    ) -> 'AnalysisGrid':
        """
        Create grid from bounds tuple.

        Parameters
        ----------
        bounds : Tuple[float, float, float, float]
            (lon_min, lon_max, lat_min, lat_max).
        resolution_m : float
            Resolution in meters.
        name : str, optional
            Grid name.

        Returns
        -------
        AnalysisGrid
            New grid instance.
        """
        lon_min, lon_max, lat_min, lat_max = bounds
        return cls(lon_min, lon_max, lat_min, lat_max, resolution_m, name)

    @classmethod
    def from_center(
        cls,
        center_lon: float,
        center_lat: float,
        width_km: float,
        height_km: float,
        resolution_m: float = 100.0,
        name: Optional[str] = None,
    ) -> 'AnalysisGrid':
        """
        Create grid centered on a point with given extent.

        Parameters
        ----------
        center_lon, center_lat : float
            Center coordinates.
        width_km, height_km : float
            Extent in kilometers.
        resolution_m : float
            Resolution in meters.
        name : str, optional
            Grid name.

        Returns
        -------
        AnalysisGrid
            New grid instance.
        """
        # Convert km to degrees
        half_height_deg = (height_km * 1000 / 2) / METERS_PER_DEGREE_LAT
        half_width_deg = (width_km * 1000 / 2) / meters_per_degree_lon(center_lat)

        return cls(
            lon_min=center_lon - half_width_deg,
            lon_max=center_lon + half_width_deg,
            lat_min=center_lat - half_height_deg,
            lat_max=center_lat + half_height_deg,
            resolution_m=resolution_m,
            name=name,
        )

    def add_covariate_points(
        self,
        name: str,
        lons: np.ndarray,
        lats: np.ndarray,
        values: np.ndarray,
        interpolation: str = 'idw',
        idw_power: float = 2.0,
        fill_value: Optional[float] = None,
    ) -> np.ndarray:
        """
        Add covariate from scattered points, interpolating to grid.

        Parameters
        ----------
        name : str
            Covariate name.
        lons, lats : np.ndarray
            Point coordinates.
        values : np.ndarray
            Point values.
        interpolation : str, default='idw'
            Interpolation method: 'idw', 'nearest', 'linear'.
        idw_power : float, default=2.0
            Power for IDW interpolation.
        fill_value : float, optional
            Value for points outside convex hull (for linear interp).

        Returns
        -------
        np.ndarray
            Interpolated grid, shape (n_lat, n_lon).
        """
        # Store original points
        self._covariate_points[name] = (
            np.asarray(lons),
            np.asarray(lats),
            np.asarray(values),
        )

        # Interpolate to grid
        if interpolation == 'idw':
            grid = self._idw_interpolate(lons, lats, values, idw_power)
        elif interpolation == 'nearest':
            grid = self._nearest_interpolate(lons, lats, values)
        elif interpolation == 'linear':
            grid = self._linear_interpolate(lons, lats, values, fill_value)
        else:
            raise ValueError(f"Unknown interpolation method: {interpolation}")

        self.covariates[name] = grid

        logger.info(
            f"Added covariate '{name}': {len(values)} points -> "
            f"{self.info.shape} grid, range [{grid.min():.2f}, {grid.max():.2f}]"
        )

        return grid

    def add_covariate_grid(
        self,
        name: str,
        grid: np.ndarray,
    ) -> None:
        """
        Add pre-computed covariate grid.

        Parameters
        ----------
        name : str
            Covariate name.
        grid : np.ndarray
            Grid values, shape (n_lat, n_lon).
        """
        if grid.shape != self.info.shape:
            raise ValueError(
                f"Grid shape {grid.shape} doesn't match analysis grid {self.info.shape}"
            )

        self.covariates[name] = grid
        logger.info(f"Added covariate grid '{name}'")

    def load_geotiff_covariate(
        self,
        name: str,
        filepath: Union[str, Path],
        band: int = 1,
        resampling: str = 'average',
        nodata_value: Optional[float] = None,
        fill_nodata: bool = True,
        fill_method: str = 'nearest',
    ) -> np.ndarray:
        """
        Load a GeoTIFF raster and resample to match the analysis grid.

        This method reads a GeoTIFF file (e.g., ATMO-Plan background map),
        reprojects/resamples it to match the analysis grid resolution and
        extent, and stores it as a covariate.

        Parameters
        ----------
        name : str
            Name for this covariate (e.g., 'atmo_plan', 'background_no2').
        filepath : str or Path
            Path to the GeoTIFF file.
        band : int, default=1
            Band number to read (1-indexed).
        resampling : str, default='average'
            Resampling method when downsampling. Options:
            - 'average': Mean of source pixels (recommended for downsampling)
            - 'bilinear': Bilinear interpolation
            - 'nearest': Nearest neighbor
            - 'cubic': Cubic interpolation
        nodata_value : float, optional
            Value to treat as nodata. If None, uses the raster's nodata value.
        fill_nodata : bool, default=True
            Whether to fill nodata pixels after resampling.
        fill_method : str, default='nearest'
            Method for filling nodata: 'nearest', 'mean', or a numeric value.

        Returns
        -------
        np.ndarray
            Resampled grid, shape (n_lat, n_lon).

        Example
        -------
        >>> grid = AnalysisGrid(
        ...     lon_min=-6.45, lon_max=-6.10,
        ...     lat_min=53.25, lat_max=53.45,
        ...     resolution_m=100.0
        ... )
        >>> grid.load_geotiff_covariate(
        ...     'atmo_plan',
        ...     'data/atmo_plan_dublin_no2.tif',
        ...     resampling='average'
        ... )
        """
        rasterio = _check_rasterio()
        from rasterio.enums import Resampling
        from rasterio.warp import reproject, calculate_default_transform
        from rasterio.transform import from_bounds

        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"GeoTIFF file not found: {filepath}")

        # Map resampling method names
        resampling_methods = {
            'average': Resampling.average,
            'bilinear': Resampling.bilinear,
            'nearest': Resampling.nearest,
            'cubic': Resampling.cubic,
            'lanczos': Resampling.lanczos,
            'mode': Resampling.mode,
        }

        if resampling not in resampling_methods:
            raise ValueError(
                f"Unknown resampling method '{resampling}'. "
                f"Options: {list(resampling_methods.keys())}"
            )

        resamp = resampling_methods[resampling]

        with rasterio.open(filepath) as src:
            logger.info(
                f"Loading GeoTIFF: {filepath.name}\n"
                f"  Source: {src.width}x{src.height} pixels, CRS={src.crs}\n"
                f"  Bounds: {src.bounds}"
            )

            # Determine nodata value
            src_nodata = nodata_value if nodata_value is not None else src.nodata

            # Define target transform and shape
            dst_transform = from_bounds(
                self.info.lon_min,
                self.info.lat_min,
                self.info.lon_max,
                self.info.lat_max,
                self.info.n_lon,
                self.info.n_lat,
            )

            # Create output array
            dst_data = np.zeros((self.info.n_lat, self.info.n_lon), dtype=np.float32)

            # Reproject/resample
            reproject(
                source=rasterio.band(src, band),
                destination=dst_data,
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=dst_transform,
                dst_crs='EPSG:4326',  # Assume WGS84 for analysis grid
                resampling=resamp,
                src_nodata=src_nodata,
                dst_nodata=np.nan,
            )

            # Report source resolution
            src_res_x = abs(src.transform[0])
            src_res_y = abs(src.transform[4])
            logger.info(
                f"  Source resolution: ~{src_res_x*111320:.1f}m x {src_res_y*111320:.1f}m\n"
                f"  Target resolution: {self.info.resolution_m:.1f}m\n"
                f"  Resampling: {resampling}"
            )

        # Handle nodata/NaN values
        nan_count = np.isnan(dst_data).sum()
        if nan_count > 0:
            logger.info(f"  Found {nan_count} nodata pixels ({100*nan_count/dst_data.size:.1f}%)")

            if fill_nodata:
                dst_data = self._fill_nodata(dst_data, fill_method)
                logger.info(f"  Filled nodata using '{fill_method}' method")

        # Flip to match grid orientation (rasterio is top-down, we use bottom-up)
        dst_data = np.flipud(dst_data)

        # Store as covariate
        self.covariates[name] = dst_data

        logger.info(
            f"Added covariate '{name}' from GeoTIFF\n"
            f"  Shape: {dst_data.shape}\n"
            f"  Range: [{np.nanmin(dst_data):.2f}, {np.nanmax(dst_data):.2f}]"
        )

        return dst_data

    def _fill_nodata(
        self,
        data: np.ndarray,
        method: str = 'nearest',
    ) -> np.ndarray:
        """
        Fill nodata (NaN) values in a grid.

        Parameters
        ----------
        data : np.ndarray
            Grid with NaN values.
        method : str
            Fill method: 'nearest', 'mean', or a numeric value.

        Returns
        -------
        np.ndarray
            Grid with NaN values filled.
        """
        if not np.any(np.isnan(data)):
            return data

        result = data.copy()
        mask = np.isnan(result)

        if method == 'mean':
            result[mask] = np.nanmean(data)
        elif method == 'nearest':
            from scipy.ndimage import distance_transform_edt

            # Find nearest valid pixel for each NaN
            indices = distance_transform_edt(
                mask,
                return_distances=False,
                return_indices=True
            )
            result = data[tuple(indices)]
        else:
            # Assume numeric fill value
            try:
                fill_val = float(method)
                result[mask] = fill_val
            except ValueError:
                raise ValueError(
                    f"Unknown fill method '{method}'. "
                    f"Use 'nearest', 'mean', or a numeric value."
                )

        return result

    def _idw_interpolate(
        self,
        lons: np.ndarray,
        lats: np.ndarray,
        values: np.ndarray,
        power: float = 2.0,
    ) -> np.ndarray:
        """
        Inverse Distance Weighting interpolation.

        Parameters
        ----------
        lons, lats : np.ndarray
            Point coordinates.
        values : np.ndarray
            Point values.
        power : float
            IDW power parameter.

        Returns
        -------
        np.ndarray
            Interpolated grid.
        """
        result = np.zeros(self.info.shape)

        lons = np.asarray(lons)
        lats = np.asarray(lats)
        values = np.asarray(values)

        for i in range(self.info.n_lat):
            for j in range(self.info.n_lon):
                # Distance from grid point to all data points
                dist = np.sqrt(
                    (lons - self.lon_grid[i, j])**2 +
                    (lats - self.lat_grid[i, j])**2
                )

                # Handle exact matches
                if np.any(dist < 1e-10):
                    result[i, j] = values[dist < 1e-10][0]
                else:
                    weights = 1.0 / (dist ** power)
                    result[i, j] = np.sum(weights * values) / np.sum(weights)

        return result

    def _nearest_interpolate(
        self,
        lons: np.ndarray,
        lats: np.ndarray,
        values: np.ndarray,
    ) -> np.ndarray:
        """Nearest neighbor interpolation."""
        from scipy.interpolate import NearestNDInterpolator

        points = np.column_stack([lons, lats])
        interp = NearestNDInterpolator(points, values)

        return interp(self.lon_grid, self.lat_grid)

    def _linear_interpolate(
        self,
        lons: np.ndarray,
        lats: np.ndarray,
        values: np.ndarray,
        fill_value: Optional[float] = None,
    ) -> np.ndarray:
        """Linear interpolation."""
        from scipy.interpolate import LinearNDInterpolator

        points = np.column_stack([lons, lats])
        interp = LinearNDInterpolator(points, values, fill_value=fill_value or np.nan)

        return interp(self.lon_grid, self.lat_grid)

    def get_flat_coords(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get flattened coordinate arrays.

        Returns
        -------
        lons, lats : np.ndarray
            Flattened coordinates, shape (n_points,).
        """
        return self._flat_lons.copy(), self._flat_lats.copy()

    def get_prediction_input(
        self,
        timestamp: Optional[float] = None,
        timestamps: Optional[np.ndarray] = None,
        include_covariates: bool = True,
        covariate_names: Optional[List[str]] = None,
        normalize: bool = False,
        scalers: Optional['Scalers'] = None,
    ) -> Union[np.ndarray, torch.Tensor]:
        """
        Get input array for GP prediction.

        Parameters
        ----------
        timestamp : float, optional
            Single timestamp for all grid points.
        timestamps : np.ndarray, optional
            Array of timestamps (one per grid point or to broadcast).
        include_covariates : bool, default=True
            Include covariate columns.
        covariate_names : List[str], optional
            Specific covariates to include (default: all).
        normalize : bool, default=False
            Normalize coordinates and covariates.
        scalers : Scalers, optional
            Scalers for normalization.

        Returns
        -------
        np.ndarray
            Input array, shape (n_points, 3 + n_covariates) or
            (n_points * n_timestamps, 3 + n_covariates) if multiple timestamps.
        """
        n_points = self.info.n_points

        # Handle timestamps
        if timestamps is not None:
            timestamps = np.asarray(timestamps)
            if timestamps.ndim == 0:
                timestamps = np.array([timestamps])

            # Broadcast: create full spatio-temporal grid
            n_times = len(timestamps)
            lons = np.tile(self._flat_lons, n_times)
            lats = np.tile(self._flat_lats, n_times)
            times = np.repeat(timestamps, n_points)
        elif timestamp is not None:
            lons = self._flat_lons
            lats = self._flat_lats
            times = np.full(n_points, timestamp)
        else:
            lons = self._flat_lons
            lats = self._flat_lats
            times = np.zeros(n_points)

        # Base coordinates: [lat, lon, time] (FusionGP convention)
        X = np.column_stack([lats, lons, times])

        # Add covariates
        if include_covariates and self.covariates:
            cov_names = covariate_names or list(self.covariates.keys())

            for name in cov_names:
                if name not in self.covariates:
                    raise ValueError(f"Covariate '{name}' not found")

                cov_flat = self.covariates[name].ravel()

                # Tile if multiple timestamps
                if timestamps is not None and len(timestamps) > 1:
                    cov_flat = np.tile(cov_flat, len(timestamps))

                X = np.column_stack([X, cov_flat])

        # Normalize if requested
        if normalize and scalers is not None:
            coords = X[:, :2]
            times = X[:, 2:3]
            covs = X[:, 3:] if X.shape[1] > 3 else None

            coords_norm = (coords - scalers.coord_min) / scalers.coord_scale
            times_norm = (times - scalers.time_min) / (scalers.time_max - scalers.time_min)

            if covs is not None and hasattr(scalers, 'transform_covariates'):
                covs_norm = scalers.transform_covariates(covs)
                X = np.column_stack([coords_norm, times_norm, covs_norm])
            else:
                X = np.column_stack([coords_norm, times_norm])
                if covs is not None:
                    X = np.column_stack([X, covs])

        return X

    def get_prediction_tensor(
        self,
        timestamp: Optional[float] = None,
        timestamps: Optional[np.ndarray] = None,
        include_covariates: bool = True,
        covariate_names: Optional[List[str]] = None,
        normalize: bool = False,
        scalers: Optional['Scalers'] = None,
        device: str = 'cpu',
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        """
        Get input tensor for GP prediction.

        Same as get_prediction_input but returns PyTorch tensor.

        Returns
        -------
        torch.Tensor
            Input tensor.
        """
        X = self.get_prediction_input(
            timestamp=timestamp,
            timestamps=timestamps,
            include_covariates=include_covariates,
            covariate_names=covariate_names,
            normalize=normalize,
            scalers=scalers,
        )

        return torch.tensor(X, dtype=dtype, device=device)

    def reshape_to_grid(
        self,
        flat_array: np.ndarray,
        n_timestamps: int = 1,
    ) -> np.ndarray:
        """
        Reshape flat prediction array to grid.

        Parameters
        ----------
        flat_array : np.ndarray
            Flat array, shape (n_points,) or (n_points * n_timestamps,).
        n_timestamps : int, default=1
            Number of timestamps (for spatio-temporal predictions).

        Returns
        -------
        np.ndarray
            Grid array, shape (n_lat, n_lon) or (n_timestamps, n_lat, n_lon).
        """
        flat_array = np.asarray(flat_array)

        if n_timestamps == 1:
            return flat_array.reshape(self.info.shape)
        else:
            return flat_array.reshape(n_timestamps, *self.info.shape)

    def plot(
        self,
        data: np.ndarray,
        ax=None,
        cmap: str = 'jet',
        vmin: Optional[float] = None,
        vmax: Optional[float] = None,
        title: Optional[str] = None,
        colorbar: bool = True,
        colorbar_label: Optional[str] = None,
        show_points: Optional[str] = None,
        point_color: str = 'white',
        point_size: int = 30,
        figsize: Tuple[int, int] = (10, 8),
    ):
        """
        Plot gridded data.

        Parameters
        ----------
        data : np.ndarray
            Grid data, shape (n_lat, n_lon).
        ax : matplotlib.axes.Axes, optional
            Axes to plot on.
        cmap : str
            Colormap.
        vmin, vmax : float, optional
            Color scale limits.
        title : str, optional
            Plot title.
        colorbar : bool
            Show colorbar.
        colorbar_label : str, optional
            Colorbar label.
        show_points : str, optional
            Name of covariate to show source points for.
        point_color : str
            Color for source points.
        point_size : int
            Size for source points.
        figsize : Tuple[int, int]
            Figure size if creating new figure.

        Returns
        -------
        matplotlib.axes.Axes
            The axes.
        """
        import matplotlib.pyplot as plt

        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)

        # Ensure data is 2D
        if data.ndim == 1:
            data = self.reshape_to_grid(data)

        im = ax.imshow(
            data,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            extent=self.info.extent,
            origin='lower',
            aspect='auto',
        )

        if colorbar:
            cbar = plt.colorbar(im, ax=ax, shrink=0.8)
            if colorbar_label:
                cbar.set_label(colorbar_label)

        if show_points and show_points in self._covariate_points:
            lons, lats, _ = self._covariate_points[show_points]
            ax.scatter(
                lons, lats,
                c=point_color,
                s=point_size,
                edgecolors='black',
                alpha=0.7,
                zorder=5,
            )

        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')

        if title:
            ax.set_title(title)

        return ax

    def plot_covariate(
        self,
        name: str,
        ax=None,
        show_points: bool = True,
        **kwargs,
    ):
        """
        Plot a covariate grid.

        Parameters
        ----------
        name : str
            Covariate name.
        ax : matplotlib.axes.Axes, optional
            Axes to plot on.
        show_points : bool
            Show source data points.
        **kwargs
            Additional arguments passed to plot().

        Returns
        -------
        matplotlib.axes.Axes
            The axes.
        """
        if name not in self.covariates:
            raise ValueError(f"Covariate '{name}' not found")

        return self.plot(
            self.covariates[name],
            ax=ax,
            title=kwargs.pop('title', f'{name}'),
            show_points=name if show_points else None,
            **kwargs,
        )

    def to_dataframe(
        self,
        include_covariates: bool = True,
    ) -> 'pd.DataFrame':
        """
        Convert grid to pandas DataFrame.

        Parameters
        ----------
        include_covariates : bool
            Include covariate columns.

        Returns
        -------
        pd.DataFrame
            DataFrame with grid coordinates and covariates.
        """
        import pandas as pd

        data = {
            'longitude': self._flat_lons,
            'latitude': self._flat_lats,
            'grid_i': np.tile(np.arange(self.info.n_lat), self.info.n_lon),
            'grid_j': np.repeat(np.arange(self.info.n_lon), self.info.n_lat),
        }

        if include_covariates:
            for name, grid in self.covariates.items():
                data[name] = grid.ravel()

        return pd.DataFrame(data)

    def to_xarray(
        self,
        data: Optional[np.ndarray] = None,
        data_name: str = 'prediction',
        include_covariates: bool = True,
    ) -> 'xr.Dataset':
        """
        Convert to xarray Dataset.

        Parameters
        ----------
        data : np.ndarray, optional
            Prediction data to include.
        data_name : str
            Name for prediction variable.
        include_covariates : bool
            Include covariate variables.

        Returns
        -------
        xr.Dataset
            Dataset with coordinates and variables.
        """
        import xarray as xr

        coords = {
            'latitude': self.info.lats,
            'longitude': self.info.lons,
        }

        data_vars = {}

        if data is not None:
            if data.ndim == 1:
                data = self.reshape_to_grid(data)
            data_vars[data_name] = (['latitude', 'longitude'], data)

        if include_covariates:
            for name, grid in self.covariates.items():
                data_vars[name] = (['latitude', 'longitude'], grid)

        ds = xr.Dataset(data_vars, coords=coords)

        # Add attributes
        ds.attrs['resolution_m'] = self.info.resolution_m
        ds.attrs['grid_name'] = self.name

        return ds

    def save_geotiff(
        self,
        data: np.ndarray,
        filepath: str,
        crs: str = 'EPSG:4326',
    ) -> None:
        """
        Save grid data as GeoTIFF.

        Parameters
        ----------
        data : np.ndarray
            Grid data.
        filepath : str
            Output file path.
        crs : str
            Coordinate reference system.
        """
        try:
            import rasterio
            from rasterio.transform import from_bounds
        except ImportError:
            raise ImportError("rasterio required for GeoTIFF export")

        if data.ndim == 1:
            data = self.reshape_to_grid(data)

        # Flip data for GeoTIFF (origin at top-left)
        data = np.flipud(data)

        transform = from_bounds(
            self.info.lon_min,
            self.info.lat_min,
            self.info.lon_max,
            self.info.lat_max,
            self.info.n_lon,
            self.info.n_lat,
        )

        with rasterio.open(
            filepath,
            'w',
            driver='GTiff',
            height=self.info.n_lat,
            width=self.info.n_lon,
            count=1,
            dtype=data.dtype,
            crs=crs,
            transform=transform,
        ) as dst:
            dst.write(data, 1)

        logger.info(f"Saved GeoTIFF: {filepath}")

    def __repr__(self) -> str:
        cov_str = ", ".join(self.covariates.keys()) if self.covariates else "none"
        return (
            f"AnalysisGrid('{self.name}')\n"
            f"  {self.info}\n"
            f"  covariates: [{cov_str}]"
        )


def sample_geotiff_at_points(
    filepath: Union[str, Path],
    lons: np.ndarray,
    lats: np.ndarray,
    band: int = 1,
    method: str = 'bilinear',
    nodata_fill: Optional[float] = None,
    src_crs: str = 'EPSG:4326',
) -> np.ndarray:
    """
    Sample GeoTIFF raster values at given point locations.

    This function extracts values from a GeoTIFF file at specified
    longitude/latitude coordinates. Useful for adding raster-based
    covariates (e.g., ATMO-Plan background map) to point observations.

    Automatically handles coordinate transformation if the GeoTIFF
    is in a different CRS than the input coordinates.

    Parameters
    ----------
    filepath : str or Path
        Path to the GeoTIFF file.
    lons : np.ndarray
        Longitude coordinates of sample points.
    lats : np.ndarray
        Latitude coordinates of sample points.
    band : int, default=1
        Band number to read (1-indexed).
    method : str, default='bilinear'
        Interpolation method: 'nearest' or 'bilinear'.
    nodata_fill : float, optional
        Value to use for points outside raster or at nodata locations.
        If None, returns NaN for such points.
    src_crs : str, default='EPSG:4326'
        CRS of the input coordinates (default: WGS84 lat/lon).

    Returns
    -------
    np.ndarray
        Sampled values at each point, shape (n_points,).

    Example
    -------
    >>> # Add ATMO-Plan values to observation dataframe
    >>> df['atmo_plan_no2'] = sample_geotiff_at_points(
    ...     'data/atmo_plan_dublin.tif',
    ...     df['longitude'].values,
    ...     df['latitude'].values
    ... )
    """
    rasterio = _check_rasterio()
    from rasterio.crs import CRS
    from rasterio.warp import transform as transform_coords

    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"GeoTIFF file not found: {filepath}")

    lons = np.asarray(lons)
    lats = np.asarray(lats)

    if lons.shape != lats.shape:
        raise ValueError("lons and lats must have the same shape")

    with rasterio.open(filepath) as src:
        logger.info(
            f"Sampling GeoTIFF at {len(lons.ravel())} points: {filepath.name}"
        )

        # Transform coordinates if CRS differs
        input_crs = CRS.from_string(src_crs)
        raster_crs = src.crs

        if raster_crs and raster_crs != input_crs:
            logger.info(
                f"  Transforming coordinates from {src_crs} to {raster_crs}"
            )
            # Transform lon/lat to raster CRS
            xs, ys = transform_coords(
                input_crs,
                raster_crs,
                lons.ravel(),
                lats.ravel()
            )
            coords = list(zip(xs, ys))
        else:
            # No transformation needed
            coords = list(zip(lons.ravel(), lats.ravel()))

        # Sample using rasterio
        if method == 'nearest':
            # Use rasterio's built-in sampling (nearest)
            samples = list(src.sample(coords, indexes=band))
            values = np.array([s[0] for s in samples], dtype=np.float32)
        elif method == 'bilinear':
            # Bilinear interpolation
            values = _bilinear_sample(src, coords, band)
        else:
            raise ValueError(f"Unknown method '{method}'. Use 'nearest' or 'bilinear'.")

        # Handle nodata
        nodata = src.nodata
        if nodata is not None:
            values[values == nodata] = np.nan

    # Fill nodata if requested
    if nodata_fill is not None:
        values = np.where(np.isnan(values), nodata_fill, values)

    n_valid = np.sum(~np.isnan(values))
    if n_valid > 0:
        logger.info(
            f"  Sampled {n_valid}/{len(values)} valid points, "
            f"range: [{np.nanmin(values):.2f}, {np.nanmax(values):.2f}]"
        )
    else:
        logger.warning(f"  No valid samples found (all {len(values)} points outside raster)")

    return values.reshape(lons.shape)


def _bilinear_sample(
    src,
    coords: list,
    band: int,
) -> np.ndarray:
    """
    Bilinear interpolation sampling from rasterio dataset.

    Parameters
    ----------
    src : rasterio.DatasetReader
        Open rasterio dataset.
    coords : list
        List of (x, y) coordinate tuples.
    band : int
        Band number.

    Returns
    -------
    np.ndarray
        Interpolated values.
    """
    from rasterio.transform import rowcol

    data = src.read(band)
    transform = src.transform

    values = []
    for x, y in coords:
        # Get fractional row/col
        # Note: rowcol returns integer indices, we need fractional
        col_f = (x - transform.c) / transform.a
        row_f = (y - transform.f) / transform.e

        # Integer indices
        col0 = int(np.floor(col_f))
        row0 = int(np.floor(row_f))
        col1 = col0 + 1
        row1 = row0 + 1

        # Bounds check
        if (row0 < 0 or row1 >= data.shape[0] or
            col0 < 0 or col1 >= data.shape[1]):
            values.append(np.nan)
            continue

        # Fractional parts
        dx = col_f - col0
        dy = row_f - row0

        # Get four corner values
        v00 = data[row0, col0]
        v01 = data[row0, col1]
        v10 = data[row1, col0]
        v11 = data[row1, col1]

        # Check for nodata in any corner
        nodata = src.nodata
        if nodata is not None:
            if v00 == nodata or v01 == nodata or v10 == nodata or v11 == nodata:
                values.append(np.nan)
                continue

        # Bilinear interpolation
        val = (v00 * (1 - dx) * (1 - dy) +
               v01 * dx * (1 - dy) +
               v10 * (1 - dx) * dy +
               v11 * dx * dy)

        values.append(val)

    return np.array(values, dtype=np.float32)


# Convenience function for Dublin
def dublin_grid(resolution_m: float = 100.0) -> AnalysisGrid:
    """
    Create default Dublin study area grid.

    Parameters
    ----------
    resolution_m : float
        Resolution in meters.

    Returns
    -------
    AnalysisGrid
        Dublin analysis grid.
    """
    return AnalysisGrid(
        lon_min=-6.45,
        lon_max=-6.10,
        lat_min=53.25,
        lat_max=53.45,
        resolution_m=resolution_m,
        name='Dublin',
    )
