"""
GP-Kalman Filter for proper sequential spatio-temporal fusion in FusionGP.

Unlike the batch SVGP (which sees all days simultaneously), the GPKalmanFilter
starts with the LUR prior and updates its belief as each day's observations arrive.
This is the correct implementation of Bayesian data fusion:

    t=0: state = LUR prior (before seeing any data)
    t=1: EPA + satellite arrive → Kalman update → prediction improves
    t=2: new data arrives → state improves further
    ...

The SVGP is used only for hyperparameter learning (spatial ℓ, temporal ℓ, σ², noise).
Inference is done by the Kalman filter using actual daily observations.

Mathematical foundation
-----------------------
State: m_t ∈ R^{M_s}, P_t ∈ R^{M_s × M_s}
       (inducing values at M_s spatial locations)

Prior at t=0:
    m_0 = LUR(Z_s)   [normalised space]
    P_0 = K_MM       [spatial kernel matrix at inducing locs]

Predict (OU transition, each day):
    m_pred = A_t * m_{t-1}
    P_pred = A_t² * P_{t-1} + Q_t * K_MM

Update (actual EPA/satellite observations at day t):
    H    = K(s_obs, Z_s) @ K_MM⁻¹           # (n_obs, M_s)
    S    = H @ P_pred @ H.T + diag(σ²_src)  # innovation covariance
    K    = P_pred @ H.T @ S⁻¹               # Kalman gain
    m_t  = m_pred + K @ (y_obs - H @ m_pred)
    P_t  = (I - K @ H) @ P_pred

Predict at grid locations s:
    f_mean(s) = μ(s) + k(s, Z_s) @ K_MM⁻¹ @ (m_t - μ(Z_s))
    f_var(s)  = k(s,s) - k(s,Z_s) @ K_MM⁻¹ @ k(Z_s,s)
              + k(s,Z_s) @ K_MM⁻¹ @ P_t @ K_MM⁻¹ @ k(Z_s,s)

Example
-------
>>> gpkf = GPKalmanFilter(model, scalers)
>>> unique_days = np.unique(data.timestamps)
>>> seq = gpkf.filter(data, unique_days, grid_coords)
>>> seq.mean_ug   # shape (n_grid, n_days), µg/m³
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, TYPE_CHECKING

import numpy as np
import torch
from tqdm import tqdm

if TYPE_CHECKING:
    from src.models.svgp import FusionSVGP
    from src.data.loader import FusionData
    from src.data.preprocessor import Scalers

logger = logging.getLogger(__name__)


@dataclass
class SequentialPredictions:
    """
    Container for GP-Kalman filter sequential predictions.

    Attributes
    ----------
    mean_ug : np.ndarray
        Posterior mean in µg/m³, shape (n_locations, n_days).
        Starts at LUR prior on day 0, updates as observations arrive.
    std_ug : np.ndarray
        Posterior std in µg/m³, shape (n_locations, n_days).
        Decreases as more observations are incorporated.
    coords : np.ndarray
        Grid coordinates in original (lat, lon) scale, shape (n_locations, 2).
    day_numbers : np.ndarray
        Original day values (days since first observation), shape (n_days,).
    n_obs_per_day : np.ndarray
        Number of observations (EPA + satellite) assimilated on each day,
        shape (n_days,). Days with n_obs=0 have uncertainty growing from prior.
    """
    mean_ug: np.ndarray
    std_ug: np.ndarray
    coords: np.ndarray
    day_numbers: np.ndarray
    n_obs_per_day: np.ndarray
    # Optional: predictions at evaluation coordinates (e.g. EPA test locations)
    eval_mean_ug: Optional[np.ndarray] = field(default=None)   # (n_eval, n_days)
    eval_std_ug: Optional[np.ndarray] = field(default=None)    # (n_eval, n_days)
    eval_coords: Optional[np.ndarray] = field(default=None)    # (n_eval, 2)


class GPKalmanFilter:
    """
    Sequential GP-Kalman filter for day-by-day pollution fusion.

    Uses the trained SVGP's hyperparameters but performs inference
    sequentially with actual daily observations — not the batch posterior.

    Parameters
    ----------
    model : FusionSVGP
        Trained model. Used for spatial kernel, temporal kernel params,
        noise variances, and prior mean function.
    scalers : Scalers
        Fitted scalers for coordinate normalisation and inverse transform.
    device : str, default='cpu'
        Device for torch computations.
    covariate_df : pd.DataFrame, optional
        Grid-level dataframe containing upwind traffic and wind-speed covariate
        columns plus 'latitude' / 'longitude'.  When provided together with
        ``covariate_cols`` and ``train_data``, a ridge-regression correction is
        fitted on training EPA residuals (observed − LUR) and added to the LUR
        prior mean at each inducing location.  This encodes the static
        traffic/wind-sector loading into the prior so the Kalman filter starts
        from a more informed background field.
    covariate_cols : list of str, optional
        Names of the covariate columns in ``covariate_df`` to use.
    train_data : FusionData, optional
        Normalised training split used to fit the covariate-correction
        regression (EPA observations only).
    """

    def __init__(
        self,
        model: "FusionSVGP",
        scalers: "Scalers",
        device: str = "cpu",
        covariate_df=None,
        covariate_cols=None,
        train_data=None,
        pre_calibrated_sources=None,
    ) -> None:
        self.model = model
        self.scalers = scalers
        self.device = torch.device(device)
        self.pre_calibrated_sources = set(pre_calibrated_sources or [])

        # Pre-build spatial state components
        self._Z_s = None          # (M_s, 2) normalised spatial inducing locs
        self._K_MM = None         # (M_s, M_s)
        self._K_MM_inv = None     # (M_s, M_s) — via Cholesky
        self._L_MM = None         # (M_s, M_s) Cholesky factor
        self._build_spatial_state()

        # Optional covariate correction to prior mean
        self._cov_beta = None          # (n_cov + 1,) ridge-regression coefficients
        self._cov_at_Z_s = None        # (M_s, n_cov) covariate values at inducing pts
        self._prior_mean_cache = {}
        use_covariate_prior_correction = (
            os.getenv("GPKF_COVARIATE_PRIOR_CORRECTION", "1").lower()
            not in {"0", "false", "no"}
        )
        if (
            use_covariate_prior_correction
            and covariate_df is not None
            and covariate_cols
            and train_data is not None
        ):
            self._build_covariate_correction(covariate_df, covariate_cols, train_data)
        elif covariate_df is not None and covariate_cols and train_data is not None:
            logger.info("Skipping GPKF covariate prior correction by environment flag.")

    # ------------------------------------------------------------------
    # State construction
    # ------------------------------------------------------------------

    def _build_spatial_state(self) -> None:
        """
        Extract spatial inducing points and precompute K_MM from trained model.

        Inducing points have shape (M, 3+K) with columns [lat, lon, time, ...].
        We take unique spatial locations (lat, lon) only.
        """
        with torch.no_grad():
            inducing = self.model.variational_strategy.inducing_points.detach()
            spatial = inducing[:, :2]  # (M, 2)
            # Unique spatial locations (inducing may have M_s × M_t combos)
            Z_s, _ = torch.unique(spatial, dim=0, return_inverse=True)
            self._Z_s = Z_s  # (M_s, 2)

            # Pad with zeros for time + covariates (kernel uses active_dims)
            n_s = Z_s.shape[0]
            n_cov = self.model.n_covariates
            pad = torch.zeros(n_s, 1 + n_cov, device=inducing.device, dtype=Z_s.dtype)
            Z_padded = torch.cat([Z_s, pad], dim=-1)  # (M_s, 3+K)

            # Compute spatial kernel matrix K_MM
            K_MM = self.model.covar_module.spatial_kernel(
                Z_padded, Z_padded, diag=False
            )
            if hasattr(K_MM, "to_dense"):
                K_MM = K_MM.to_dense()
            K_MM = K_MM.double()  # higher precision for inversion
            self._K_MM = K_MM

            # Cholesky factorisation for stable inversion
            jitter = 1e-6 * torch.eye(n_s, dtype=torch.float64)
            try:
                L = torch.linalg.cholesky(K_MM + jitter)
            except RuntimeError:
                jitter = 1e-4 * torch.eye(n_s, dtype=torch.float64)
                L = torch.linalg.cholesky(K_MM + jitter)
            self._L_MM = L
            self._K_MM_inv = torch.cholesky_inverse(L)

        logger.info(
            f"GPKalmanFilter: M_s={n_s} spatial inducing points, "
            f"K_MM cond≈{(K_MM.diag().max() / K_MM.diag().min()).item():.1f}"
        )

    def _build_covariate_correction(self, covariate_df, covariate_cols, train_data) -> None:
        """
        Fit a ridge regression on training EPA residuals to learn the
        traffic/wind contribution to NO2, and pre-cache the daily covariate
        matrices at every inducing location.

        The wind sector covariates (traffic_wind_i, wind_speed_w_i) are already
        feature-engineered per (grid cell, day) in covariate_df.  This method:
          1. Maps each inducing point to its nearest grid cell (once).
          2. Pre-caches X(Z_s, t) for every unique day → O(1) lookup at runtime.
          3. Fits ridge regression: (EPA_norm - LUR_norm) ~ X(s,t) @ β.

        Prior mean during filter:  μ(t) = LUR(Z_s) + X(Z_s, t) @ β
        """
        lat_col, lon_col, ts_col = "latitude", "longitude", "timestamp"
        covariate_cols = list(covariate_cols)
        self._covariate_cols = covariate_cols
        n_cov = len(covariate_cols)

        # ---- 1. Unique grid cells and NN map: inducing pts → grid cell ------
        unique_cells = (
            covariate_df[[lat_col, lon_col]]
            .drop_duplicates()
            .reset_index(drop=True)
        )
        grid_lat = unique_cells[lat_col].to_numpy()
        grid_lon = unique_cells[lon_col].to_numpy()
        grid_coords = np.stack([grid_lat, grid_lon], axis=1)  # (n_cells, 2)
        self._cov_grid_coords = grid_coords
        try:
            from scipy.spatial import cKDTree
            self._cov_grid_tree = cKDTree(grid_coords)
        except ImportError:
            self._cov_grid_tree = None

        Z_raw = (
            self._Z_s.cpu().numpy() * self.scalers.coord_scale + self.scalers.coord_min
        )  # (M_s, 2)
        diff = Z_raw[:, None, :] - grid_coords[None, :, :]
        nn_Z = np.argmin((diff ** 2).sum(axis=2), axis=1)  # (M_s,) cell index per inducing pt
        self._nn_Z = nn_Z  # save for use in _prior_mean_for_day

        # ---- 2. Pre-cache X(grid_cells, t) for every unique day -------------
        # Build a (n_cells, n_cov) array per day using the already-computed features.
        # Vectorized: for each day, match all rows to grid cells via NN in one pass.
        # Cache stores full grid (n_cells, n_cov) so it can be indexed by either
        # nn_Z (inducing point lookup) or nn_epa (EPA observation lookup).
        self._daily_X_cache: dict = {}   # day_val → (n_cells, n_cov)
        ts_arr_full = covariate_df[ts_col].to_numpy()
        lat_arr_full = covariate_df[lat_col].to_numpy()
        lon_arr_full = covariate_df[lon_col].to_numpy()
        cov_arr_full = covariate_df[covariate_cols].to_numpy(dtype=float)
        unique_days_cov = np.unique(ts_arr_full)
        for day_val in unique_days_cov:
            day_mask = np.isclose(ts_arr_full, day_val, atol=0.1)
            row_lats = lat_arr_full[day_mask]
            row_lons = lon_arr_full[day_mask]
            row_covs = cov_arr_full[day_mask]           # (n_rows_today, n_cov)
            # Deduplicate by (lat, lon) — covariates are grid-cell-level, not source-specific
            uniq_coords, first_idx = np.unique(
                np.stack([row_lats, row_lons], axis=1), axis=0, return_index=True
            )
            uniq_covs = row_covs[first_idx]             # (n_unique_cells, n_cov)
            # Unique cells in this day → global grid cells. Use the KD-tree to
            # avoid an O(n_unique * n_cells) distance matrix for every day.
            if self._cov_grid_tree is not None:
                _, ci_all = self._cov_grid_tree.query(uniq_coords)
            else:
                ci_all = np.empty(len(uniq_coords), dtype=int)
                for start in range(0, len(uniq_coords), 5000):
                    end = min(start + 5000, len(uniq_coords))
                    diff_r = uniq_coords[start:end, None, :] - grid_coords[None, :, :]
                    ci_all[start:end] = np.argmin((diff_r ** 2).sum(axis=2), axis=1)
            cell_X = np.zeros((len(grid_coords), n_cov))
            cell_X[ci_all] = uniq_covs
            # Store full grid — indexed by grid cell index (0..n_cells-1)
            self._daily_X_cache[float(day_val)] = cell_X  # (n_cells, n_cov)

        # ---- 3. LUR baseline (computed once, reused every day) --------------
        M_s = self._Z_s.shape[0]
        n_cov_model = self.model.n_covariates
        pad = torch.zeros(M_s, 1 + n_cov_model, dtype=torch.float32)
        Z_padded = torch.cat([self._Z_s.float(), pad], dim=-1)
        with torch.no_grad():
            self._lur_at_Z = self.model.mean_module(Z_padded).cpu().numpy().astype(np.float64)

        # ---- 4. Training EPA residuals (y_obs_norm − LUR_norm) -------------
        epa_mask = train_data.source_masks.get("epa", None)
        if epa_mask is None or not epa_mask.any():
            logger.warning("No EPA training observations — skipping covariate correction.")
            return

        epa_coords_norm = train_data.coords[epa_mask].astype(np.float64)
        epa_ts_norm     = train_data.timestamps[epa_mask].astype(np.float64)
        epa_obs_norm    = train_data.observations["epa"][epa_mask].astype(np.float64)

        n_obs = len(epa_obs_norm)
        obs_t = torch.tensor(epa_coords_norm, dtype=torch.float32)
        pad_obs = torch.zeros(n_obs, 1 + n_cov_model, dtype=torch.float32)
        with torch.no_grad():
            lur_at_epa = self.model.mean_module(
                torch.cat([obs_t, pad_obs], dim=-1)
            ).cpu().numpy().astype(np.float64)
        residuals = epa_obs_norm - lur_at_epa  # (N,)

        # ---- 5. Covariate values at each EPA observation via NN + day cache -
        epa_raw    = epa_coords_norm * self.scalers.coord_scale + self.scalers.coord_min
        time_range = float(self.scalers.time_max - self.scalers.time_min)
        epa_ts_raw = epa_ts_norm * time_range + float(self.scalers.time_min)

        diff_epa = epa_raw[:, None, :] - grid_coords[None, :, :]
        nn_epa   = np.argmin((diff_epa ** 2).sum(axis=2), axis=1)  # (N,)

        # For each EPA observation, look up covariates from the pre-built cache:
        # cache[day_val][nn] gives the (n_cov,) covariate vector for that cell+day.
        X_epa = np.zeros((n_obs, n_cov))
        for i, (day_val, nn) in enumerate(zip(epa_ts_raw, nn_epa)):
            best_day = min(self._daily_X_cache, key=lambda d: abs(d - day_val))
            X_epa[i] = self._daily_X_cache[best_day][nn]

        # ---- 6. Normalise and fit ridge regression -------------------------
        # Use GLOBAL statistics from the full covariate grid (all cells × all days),
        # not just EPA station locations. This prevents out-of-distribution extrapolation
        # when the covariate correction is applied at inducing points that may lie in
        # high-traffic areas never covered by EPA stations.
        all_X_vals = np.concatenate(
            list(self._daily_X_cache.values()), axis=0
        )  # (n_days * n_cells, n_cov)
        self._cov_mean = all_X_vals.mean(axis=0, keepdims=True)
        self._cov_std  = all_X_vals.std(axis=0, keepdims=True) + 1e-8
        del all_X_vals

        X_epa_norm = (X_epa - self._cov_mean) / self._cov_std

        X_design = np.column_stack([np.ones(n_obs), X_epa_norm])
        alpha_ridge = 1.0
        ATA = X_design.T @ X_design + alpha_ridge * np.eye(X_design.shape[1])
        self._cov_beta = np.linalg.solve(ATA, X_design.T @ residuals)

        # Release large intermediate arrays — everything needed is in the cache + beta
        del ts_arr_full, lat_arr_full, lon_arr_full, cov_arr_full
        del X_epa, X_epa_norm, X_design, ATA, residuals

        logger.info(
            f"Covariate correction: {n_cov} daily features pre-cached for "
            f"{len(self._daily_X_cache)} days, N_train_EPA={n_obs}, "
            f"β range=[{self._cov_beta.min():.4f}, {self._cov_beta.max():.4f}]"
        )

    def _prior_mean_for_day(self, day_val: float) -> np.ndarray:
        """
        Prior mean at inducing locations for a specific day (normalised space).

        Returns LUR(Z_s) + daily traffic/wind correction.  Shape (M_s,).
        Falls back to LUR-only if covariate correction was not fitted.
        """
        if not hasattr(self, "_lur_at_Z"):
            # Compute LUR on the fly if _build_covariate_correction was not called
            with torch.no_grad():
                n_s = self._Z_s.shape[0]
                n_cov = self.model.n_covariates
                pad = torch.zeros(n_s, 1 + n_cov, device=self._Z_s.device,
                                  dtype=self._Z_s.dtype)
                Z_padded = torch.cat([self._Z_s, pad], dim=-1)
                return self.model.mean_module(Z_padded).cpu().numpy().astype(np.float64)

        m0 = self._lur_at_Z.copy()

        if self._cov_beta is None or not self._daily_X_cache:
            return m0

        # O(1) cache lookup — find closest cached day
        best_day = min(self._daily_X_cache, key=lambda d: abs(d - day_val))
        X_day = self._daily_X_cache[best_day][self._nn_Z]      # (M_s, n_cov)
        # Clip to ±5σ to prevent extreme extrapolation at inducing points that
        # may have covariate values far from the training distribution.
        X_day_norm = np.clip((X_day - self._cov_mean) / self._cov_std, -5.0, 5.0)
        X_design   = np.column_stack([np.ones(len(m0)), X_day_norm])
        return m0 + X_design @ self._cov_beta

    def _prior_mean_at_locations(
        self,
        coords_norm: np.ndarray,
        day_val: float,
        batch_size: int = 5000,
    ) -> np.ndarray:
        """
        Prior mean at arbitrary locations for a specific day (normalised space).

        This evaluates the full-resolution prior mean directly at the requested
        locations, then adds the same daily traffic/wind correction used at the
        inducing points when that correction is available.
        """
        coords_norm = np.asarray(coords_norm, dtype=np.float64)
        n = coords_norm.shape[0]
        cache_key = None
        if n > 1000:
            cache_key = (
                round(float(day_val), 6),
                coords_norm.shape,
                tuple(np.round(coords_norm[0], 8)),
                tuple(np.round(coords_norm[-1], 8)),
            )
            cached = self._prior_mean_cache.get(cache_key)
            if cached is not None:
                return cached.copy()

        n_cov = self.model.n_covariates
        prior = np.zeros(n, dtype=np.float64)

        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            c_t = torch.tensor(coords_norm[start:end], dtype=torch.float32)
            pad = torch.zeros(end - start, 1 + n_cov, dtype=torch.float32)
            x = torch.cat([c_t, pad], dim=-1)
            with torch.no_grad():
                prior[start:end] = (
                    self.model.mean_module(x).cpu().numpy().astype(np.float64)
                )

        if self._cov_beta is None or not getattr(self, "_daily_X_cache", None):
            return prior

        coords_raw = coords_norm * self.scalers.coord_scale + self.scalers.coord_min
        if getattr(self, "_cov_grid_tree", None) is not None:
            _, nn = self._cov_grid_tree.query(coords_raw)
        else:
            grid_coords = self._cov_grid_coords
            nn = np.empty(n, dtype=int)
            for start in range(0, n, batch_size):
                end = min(start + batch_size, n)
                diff = coords_raw[start:end, None, :] - grid_coords[None, :, :]
                nn[start:end] = np.argmin((diff ** 2).sum(axis=2), axis=1)

        best_day = min(self._daily_X_cache, key=lambda d: abs(d - day_val))
        X_day = self._daily_X_cache[best_day][nn]
        X_day_norm = np.clip((X_day - self._cov_mean) / self._cov_std, -5.0, 5.0)
        X_design = np.column_stack([np.ones(n), X_day_norm])
        prior = prior + X_design @ self._cov_beta
        if cache_key is not None:
            self._prior_mean_cache[cache_key] = prior.copy()
        return prior

    # ------------------------------------------------------------------
    # OU transition parameters
    # ------------------------------------------------------------------

    def _ou_params(self, unique_days: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute OU transition A[t] and process noise Q[t] for each day gap.

        Returns
        -------
        A_arr : (n_days-1,)
        Q_arr : (n_days-1,)
        """
        ell = self.model.covar_module.temporal_lengthscale.detach().squeeze().item()
        sigma2 = self.model.covar_module.outputscale.detach().item()
        time_range = float(self.scalers.time_max - self.scalers.time_min)

        A_arr = np.zeros(len(unique_days) - 1)
        Q_arr = np.zeros(len(unique_days) - 1)
        for i in range(len(unique_days) - 1):
            dt_norm = (unique_days[i + 1] - unique_days[i]) / time_range
            A = float(np.exp(-dt_norm / ell))
            A_arr[i] = A
            Q_arr[i] = sigma2 * (1.0 - A ** 2)

        logger.info(f"OU params: ℓ={ell:.4f}, σ²={sigma2:.4f}, "
                    f"A range=[{A_arr.min():.3f}, {A_arr.max():.3f}]")
        return A_arr, Q_arr

    # ------------------------------------------------------------------
    # Observation helpers
    # ------------------------------------------------------------------

    def _get_day_observations(
        self,
        data: "FusionData",
        day_val: float,
        sources: List[str],
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Extract all observations at a given day from FusionData.

        Parameters
        ----------
        data : FusionData
            Full dataset (in normalised space — timestamps and coords normalised).
        day_val : float
            The original (pre-normalisation) day value to select.
        sources : list of str
            Sources to include (e.g. ["epa", "satellite"]).

        Returns
        -------
        y_norm : np.ndarray, shape (n_obs,)
            Observations in normalised target space.
        coords_norm : np.ndarray, shape (n_obs, 2)
            Normalised spatial coordinates.
        noise_var : np.ndarray, shape (n_obs,)
            Per-observation noise variance in observation space.
        slope : np.ndarray, shape (n_obs,)
            Per-observation calibration slope (observation = slope * f + intercept).
        intercept : np.ndarray, shape (n_obs,)
            Per-observation calibration intercept.
        """
        # Find indices matching this day (original timestamp scale)
        day_mask = np.isclose(data.timestamps, day_val, atol=0.1)

        y_list, coord_list, noise_list = [], [], []
        slope_list, intercept_list = [], []

        for source in sources:
            if source not in data.source_masks:
                continue
            src_mask = data.source_masks[source]
            combined = day_mask & src_mask

            if not combined.any():
                continue

            y_vals = data.observations[source][combined].astype(np.float64)
            coords = data.coords[combined]
            # NOTE: data.observations is already in the Kalman state space.
            # The preprocessor normalises observations before passing data here
            # (normalize_targets=True), so no further normalisation is needed.

            # Observation-space noise variance and calibration parameters
            noise_var_src = self._source_noise_var_obs(source)
            slope_src, intercept_src = self._source_calibration_params(source)

            y_list.append(y_vals)
            coord_list.append(coords)
            noise_list.append(np.full(combined.sum(), noise_var_src))
            slope_list.append(np.full(combined.sum(), slope_src))
            intercept_list.append(np.full(combined.sum(), intercept_src))

        if not y_list:
            return (
                np.empty(0),
                np.empty((0, 2)),
                np.empty(0),
                np.empty(0),
                np.empty(0),
            )

        return (
            np.concatenate(y_list).astype(np.float64),
            np.concatenate(coord_list, axis=0).astype(np.float64),
            np.concatenate(noise_list).astype(np.float64),
            np.concatenate(slope_list).astype(np.float64),
            np.concatenate(intercept_list).astype(np.float64),
        )

    def _source_noise_var_obs(self, source: str) -> float:
        """Noise variance for a given source in observation space."""
        with torch.no_grad():
            var = self.model.likelihood.noise_variance[source].item()
        return float(var)

    def _source_calibration_params(self, source: str) -> Tuple[float, float]:
        """Calibration slope/intercept for a given source."""
        if source in self.pre_calibrated_sources:
            return 1.0, 0.0
        if source == "satellite":
            with torch.no_grad():
                slope = float(self.model.likelihood.sat_slope.item())
                intercept = float(self.model.likelihood.sat_intercept.item())
            return slope, intercept
        if source == "low_cost":
            with torch.no_grad():
                slope = float(self.model.likelihood.lc_slope.item())
                intercept = float(self.model.likelihood.lc_intercept.item())
            return slope, intercept
        return 1.0, 0.0

    def _observation_matrix(self, obs_coords_norm: np.ndarray) -> np.ndarray:
        """
        Compute H = K(s_obs, Z_s) @ K_MM_inv.

        Parameters
        ----------
        obs_coords_norm : np.ndarray, shape (n_obs, 2)
            Normalised observation coordinates.

        Returns
        -------
        H : np.ndarray, shape (n_obs, M_s)
        """
        n_obs = obs_coords_norm.shape[0]
        n_cov = self.model.n_covariates
        n_s = self._Z_s.shape[0]

        obs_t = torch.tensor(obs_coords_norm, dtype=torch.float32)
        pad_obs = torch.zeros(n_obs, 1 + n_cov, dtype=torch.float32)
        obs_padded = torch.cat([obs_t, pad_obs], dim=-1)

        pad_z = torch.zeros(n_s, 1 + n_cov, dtype=torch.float32)
        Z_padded = torch.cat([self._Z_s.float(), pad_z], dim=-1)

        with torch.no_grad():
            K_sZ = self.model.covar_module.spatial_kernel(
                obs_padded, Z_padded, diag=False
            )
            if hasattr(K_sZ, "to_dense"):
                K_sZ = K_sZ.to_dense()

        K_sZ_np = K_sZ.cpu().numpy().astype(np.float64)
        H = K_sZ_np @ self._K_MM_inv.numpy()  # (n_obs, M_s)
        return H

    # ------------------------------------------------------------------
    # Grid prediction
    # ------------------------------------------------------------------

    def _predict_grid(
        self,
        grid_coords_norm: np.ndarray,
        m_t: np.ndarray,
        P_t: np.ndarray,
        day_val: float,
        prior_at_Z: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict mean and variance at grid locations given current state.

        f_mean(s) = μ(s) + K(s, Z_s) @ K_MM⁻¹ @ (m_t - μ(Z_s))
        f_var(s)  = k(s,s) - k(s,Z_s)@K_MM⁻¹@k(Z_s,s)
                  + k(s,Z_s)@K_MM⁻¹@P_t@K_MM⁻¹@k(Z_s,s)

        Parameters
        ----------
        grid_coords_norm : np.ndarray, shape (n_grid, 2)
        m_t : np.ndarray, shape (M_s,)
        P_t : np.ndarray, shape (M_s, M_s)
        day_val : float
        prior_at_Z : np.ndarray, optional, shape (M_s,)

        Returns
        -------
        mean : np.ndarray, shape (n_grid,)
        var  : np.ndarray, shape (n_grid,)
        """
        n_grid = grid_coords_norm.shape[0]
        n_cov = self.model.n_covariates
        n_s = self._Z_s.shape[0]
        batch = 2000  # spatial batch to avoid memory issues

        means = np.zeros(n_grid)
        vars_ = np.zeros(n_grid)
        prior_full = self._prior_mean_at_locations(grid_coords_norm, day_val)
        if prior_at_Z is None:
            prior_at_Z = self._prior_mean_for_day(day_val)
        residual_t = m_t - prior_at_Z

        pad_z = torch.zeros(n_s, 1 + n_cov, dtype=torch.float32)
        Z_padded = torch.cat([self._Z_s.float(), pad_z], dim=-1)

        for start in range(0, n_grid, batch):
            end = min(start + batch, n_grid)
            g_np = grid_coords_norm[start:end]
            n_b = g_np.shape[0]

            g_t = torch.tensor(g_np, dtype=torch.float32)
            pad_g = torch.zeros(n_b, 1 + n_cov, dtype=torch.float32)
            g_padded = torch.cat([g_t, pad_g], dim=-1)

            with torch.no_grad():
                # k(s, Z_s)
                K_gZ = self.model.covar_module.spatial_kernel(
                    g_padded, Z_padded, diag=False
                )
                if hasattr(K_gZ, "to_dense"):
                    K_gZ = K_gZ.to_dense()
                K_gZ_np = K_gZ.cpu().numpy().astype(np.float64)

                # k(s, s) diagonal
                k_diag = self.model.covar_module.spatial_kernel(
                    g_padded, g_padded, diag=True
                )
                if hasattr(k_diag, "to_dense"):
                    k_diag = k_diag.to_dense()
                k_diag_np = k_diag.cpu().numpy().astype(np.float64)

            K_MM_inv_np = self._K_MM_inv.numpy()
            # α = K_MM_inv @ k(Z_s, s)  (M_s, n_b)
            alpha = K_MM_inv_np @ K_gZ_np.T  # (M_s, n_b)

            # Reconstruct the full field as fine-grid prior plus smooth residual.
            means[start:end] = prior_full[start:end] + residual_t @ alpha

            # prior variance reduction: k_ss - k_sZ K_MM_inv k_Zs
            prior_reduction = np.einsum("ij,ji->i", K_gZ_np, alpha)  # (n_b,)

            # posterior variance addition: k_sZ K_MM_inv P_t K_MM_inv k_Zs
            P_alpha = P_t @ alpha  # (M_s, n_b)
            post_addition = np.einsum("ij,ji->i", K_gZ_np, K_MM_inv_np @ P_alpha)

            # Add a minimum variance floor (fraction of prior k(s,s)) to account for
            # irreducible spatial interpolation uncertainty. P_t collapses to ~0 when
            # n_obs >> M_s, so without this the predictive variance is near-zero.
            # This inflates predictive variance without touching the state P_t,
            # so the Kalman mean m_t and all future updates are unaffected.
            vars_[start:end] = np.maximum(
                k_diag_np - prior_reduction + post_addition + 0.5 * k_diag_np, 1e-12
            )

        return means, vars_

    # ------------------------------------------------------------------
    # Inverse transform
    # ------------------------------------------------------------------

    def _inverse_transform(
        self, mean_norm: np.ndarray, std_norm: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        scalers = self.scalers
        if getattr(scalers, "normalize_targets", False) and "epa" in scalers.target_std:
            t_mean = float(scalers.target_mean["epa"])
            t_std = float(scalers.target_std["epa"])
            return mean_norm * t_std + t_mean, std_norm * t_std
        return mean_norm.copy(), std_norm.copy()

    # ------------------------------------------------------------------
    # Main filter
    # ------------------------------------------------------------------

    def filter(
        self,
        data: "FusionData",
        unique_days: np.ndarray,
        grid_coords_original: np.ndarray,
        eval_coords_original: Optional[np.ndarray] = None,
    ) -> SequentialPredictions:
        """
        Run GP-Kalman filter sequentially over all days.

        Parameters
        ----------
        data : FusionData
            Full dataset. Timestamps should be in original (pre-normalised) scale.
            Observations and coords should be in normalised space.
        unique_days : np.ndarray
            Sorted unique day values in original time scale, shape (n_days,).
        grid_coords_original : np.ndarray
            Grid locations in original (lat, lon) scale, shape (n_grid, 2).
        eval_coords_original : np.ndarray, optional
            Additional evaluation coordinates in original (lat, lon) scale,
            shape (n_eval, 2). If provided, predictions are also made here each
            day and returned in ``SequentialPredictions.eval_mean_ug``.

        Returns
        -------
        SequentialPredictions
        """
        n_days = len(unique_days)
        n_grid = len(grid_coords_original)
        n_s = self._Z_s.shape[0]
        sources = [s for s in self.model.sources if s != "low_cost"]
        # Diagnostics: likelihood calibration params (obs space)
        try:
            with torch.no_grad():
                sat_slope = float(self.model.likelihood.sat_slope.item())
                sat_intercept = float(self.model.likelihood.sat_intercept.item())
                noise_epa = float(self.model.likelihood.noise_variance["epa"].item())
                noise_sat = float(self.model.likelihood.noise_variance["satellite"].item())
            logger.info(
                f"GPKF likelihood params: sat_slope={sat_slope:.4f}, "
                f"sat_intercept={sat_intercept:.4f}, "
                f"noise_var_epa={noise_epa:.4f}, noise_var_sat={noise_sat:.4f}"
            )
        except (AttributeError, KeyError, TypeError, ValueError) as exc:
            logger.debug("Skipping GPKF likelihood diagnostics: %s", exc)

        # Normalise grid coordinates
        grid_coords_norm = (
            (grid_coords_original - self.scalers.coord_min) / self.scalers.coord_scale
        ).astype(np.float64)

        # Optional eval coordinates (e.g. EPA test locations)
        has_eval = eval_coords_original is not None
        if has_eval:
            n_eval = len(eval_coords_original)
            eval_coords_norm = (
                (eval_coords_original - self.scalers.coord_min) / self.scalers.coord_scale
            ).astype(np.float64)
            eval_mean_norm = np.zeros((n_eval, n_days))
            eval_var_norm = np.zeros((n_eval, n_days))

        # OU params
        A_arr, Q_arr = self._ou_params(unique_days)
        K_MM_np = self._K_MM.numpy()

        # --- Initialise state at prior for day 0 ---
        # Prior mean is daily: LUR(Z_s) + traffic/wind correction for that day
        mu_prev = self._prior_mean_for_day(unique_days[0])   # (M_s,)
        m_t = mu_prev.copy()                                  # (M_s,)  running mean
        P_t = K_MM_np.copy()                                  # (M_s, M_s)

        # Output arrays
        mean_norm = np.zeros((n_grid, n_days))
        var_norm = np.zeros((n_grid, n_days))
        n_obs_per_day = np.zeros(n_days, dtype=int)

        logger.info(f"Running GP-Kalman filter: {n_days} days, {n_grid} grid cells, "
                    f"M_s={n_s} inducing points")

        for d_idx in tqdm(range(n_days), desc="GP-Kalman filter"):
            day_val = unique_days[d_idx]

            # Prior mean for today (daily wind changes which upwind traffic matters)
            mu_today = self._prior_mean_for_day(day_val)      # (M_s,)

            # --- Predict (OU transition) for t > 0 ---
            if d_idx > 0:
                A = A_arr[d_idx - 1]
                Q = Q_arr[d_idx - 1]
                # Mean reverts to today's prior (wind/traffic-informed background)
                m_t = mu_today + A * (m_t - mu_prev)
                P_t = A ** 2 * P_t + Q * K_MM_np

            mu_prev = mu_today    # carry forward for next day's OU transition

            # --- Get actual observations at this day ---
            y_obs, obs_coords, noise_var, obs_slope, obs_intercept = self._get_day_observations(
                data, day_val, sources
            )
            n_obs = len(y_obs)
            n_obs_per_day[d_idx] = n_obs

            # Diagnostic: print state prior to update on the first 3 days
            if d_idx < 3:
                m_grid_diag, _ = self._predict_grid(
                    grid_coords_norm, m_t, P_t, day_val, prior_at_Z=mu_today
                )
                tqdm.write(
                    f"  [DIAG] Day {d_idx:02d} PRE-UPDATE: state m_t "
                    f"range=[{m_t.min():.3f}, {m_t.max():.3f}], "
                    f"grid_norm range=[{m_grid_diag.min():.3f}, {m_grid_diag.max():.3f}]"
                )
                if n_obs > 0:
                    tqdm.write(
                        f"  [DIAG] Day {d_idx:02d} y_obs "
                        f"range=[{y_obs.min():.3f}, {y_obs.max():.3f}], "
                        f"mean={y_obs.mean():.3f}  "
                        f"slope=[{obs_slope.min():.3f}, {obs_slope.max():.3f}] "
                        f"intercept=[{obs_intercept.min():.3f}, {obs_intercept.max():.3f}] "
                        f"noise_var=[{noise_var.min():.3f}, {noise_var.max():.3f}]  "
                        f"obs_coords lat=[{obs_coords[:,0].min():.4f}, {obs_coords[:,0].max():.4f}] "
                        f"lon=[{obs_coords[:,1].min():.4f}, {obs_coords[:,1].max():.4f}]"
                    )

            # --- Kalman update (only if there are observations) ---
            if n_obs > 0:
                # Normalise observation coordinates to match the state space
                obs_coords_norm = (
                    (obs_coords - self.scalers.coord_min) / self.scalers.coord_scale
                ).astype(np.float64)
                H = self._observation_matrix(obs_coords_norm)  # (n_obs, M_s)
                # Observation model: y = a * H f + b + eps
                H_scaled = H * obs_slope[:, None]
                R = np.diag(noise_var)                        # (n_obs, n_obs)
                prior_obs = self._prior_mean_at_locations(obs_coords_norm, day_val)
                residual_t = m_t - mu_today

                if d_idx < 3:
                    tqdm.write(
                        f"  [DIAG] Day {d_idx:02d} obs_coords_norm "
                        f"range=[{obs_coords_norm.min():.4f}, {obs_coords_norm.max():.4f}], "
                        f"H row-sums range=[{H_scaled.sum(axis=1).min():.4f}, {H_scaled.sum(axis=1).max():.4f}]"
                    )

                # Innovation covariance S = H P_pred H.T + R
                HP = H_scaled @ P_t                           # (n_obs, M_s)
                S = HP @ H_scaled.T + R                       # (n_obs, n_obs)

                # Cholesky solve for stability: K = P_t H.T S^{-1}
                try:
                    L_S = np.linalg.cholesky(S + 1e-8 * np.eye(n_obs))
                    # K P_t H.T L_S^{-T} → K = P_t H.T S^{-1}
                    Kt = np.linalg.solve(L_S, H_scaled @ P_t.T).T  # (M_s, n_obs)
                    Kalman_gain = np.linalg.solve(L_S.T, Kt.T).T  # (M_s, n_obs)
                except np.linalg.LinAlgError:
                    Kalman_gain = P_t @ H_scaled.T @ np.linalg.pinv(S)

                # Innovation under nonzero prior-mean residual reconstruction:
                # y = slope * (mu(obs) + H @ (m_t - mu(Z))) + intercept + eps
                pred_obs = obs_slope * (prior_obs + H @ residual_t) + obs_intercept
                innovation = y_obs - pred_obs  # (n_obs,)

                # State update (standard form)
                m_t = m_t + Kalman_gain @ innovation
                P_t = (np.eye(n_s) - Kalman_gain @ H_scaled) @ P_t
                # Symmetrise to prevent numerical drift
                P_t = 0.5 * (P_t + P_t.T)

                tqdm.write(
                    f"  Day {d_idx:02d} (t={day_val:.0f}): "
                    f"{n_obs} obs → updated state "
                    f"[m_t range: {m_t.min():.3f}, {m_t.max():.3f}]"
                )
            else:
                tqdm.write(
                    f"  Day {d_idx:02d} (t={day_val:.0f}): "
                    f"no obs → propagated prior"
                )

            # --- Predict at all grid locations ---
            m_grid, v_grid = self._predict_grid(
                grid_coords_norm, m_t, P_t, day_val, prior_at_Z=mu_today
            )
            mean_norm[:, d_idx] = m_grid
            var_norm[:, d_idx] = v_grid

            # --- Predict at eval locations if provided ---
            if has_eval:
                m_eval, v_eval = self._predict_grid(
                    eval_coords_norm, m_t, P_t, day_val, prior_at_Z=mu_today
                )
                eval_mean_norm[:, d_idx] = m_eval
                eval_var_norm[:, d_idx] = v_eval

        # --- Inverse transform to µg/m³ ---
        mean_ug, std_ug = self._inverse_transform(mean_norm, np.sqrt(var_norm))

        eval_mean_ug = eval_std_ug = None
        if has_eval:
            eval_mean_ug, eval_std_ug = self._inverse_transform(
                eval_mean_norm, np.sqrt(eval_var_norm)
            )

        return SequentialPredictions(
            mean_ug=mean_ug,
            std_ug=std_ug,
            coords=grid_coords_original,
            day_numbers=unique_days,
            n_obs_per_day=n_obs_per_day,
            eval_mean_ug=eval_mean_ug,
            eval_std_ug=eval_std_ug,
            eval_coords=eval_coords_original,
        )
