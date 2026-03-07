"""
Kalman smoother for day-by-day pollution tracking in FusionGP.

This module implements a hybrid approach: the trained SVGP produces a Gaussian
posterior at each (location, day), which is then used as a pseudo-measurement in
a 1D Kalman filter. The temporal dynamics come from the OU process that corresponds
exactly to the exponential (Matérn-1/2) temporal kernel.

Mathematical foundation
-----------------------
The exponential kernel k(t, t') = σ² exp(-|t-t'|/ℓ) is the covariance of an
Ornstein-Uhlenbeck (OU) process with SDE:

    df(t) = -(1/ℓ) f(t) dt + σ√(2/ℓ) dW(t)

Discretising over Δt gives the state-space transition:

    f_{t+1} = A_t f_t + q_t,   q_t ~ N(0, Q_t)
    A_t = exp(-Δt_norm / ℓ)
    Q_t = σ² (1 - A_t²)

where Δt_norm is the time gap in the same [0,1] normalised scale used by the kernel.

Pseudo-measurement approach
---------------------------
The SVGP posterior mean m_t(s) and latent variance v_t(s) at location s and day t
are used as a Gaussian pseudo-measurement:

    z_t(s) = m_t(s),   R_t(s) = v_t(s)

This absorbs the multi-source (EPA + satellite) fusion already done by the SVGP,
and the Kalman filter adds temporal dynamics on top.

All operations run in the normalised target space. The inverse transform to µg/m³
is applied once at the end.

Example
-------
>>> smoother = KalmanSmoother(model, predictor, scalers)
>>> unique_days = np.unique(data.timestamps)   # original day values
>>> smoothed = smoother.smooth(grid_coords, unique_days)
>>> smoothed.mean_ug   # shape (n_locations, n_days), µg/m³
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional, Tuple, TYPE_CHECKING

import numpy as np
import torch
from tqdm import tqdm

if TYPE_CHECKING:
    from src.models.svgp import FusionSVGP
    from src.inference.predictor import Predictor
    from src.data.preprocessor import Scalers

logger = logging.getLogger(__name__)


@dataclass
class SmoothedPredictions:
    """
    Container for Kalman-smoothed spatio-temporal predictions.

    Attributes
    ----------
    mean_ug : np.ndarray
        Smoothed posterior mean in µg/m³, shape (n_locations, n_days).
    std_ug : np.ndarray
        Smoothed posterior std in µg/m³, shape (n_locations, n_days).
    coords : np.ndarray
        Spatial coordinates in original (lat, lon) scale, shape (n_locations, 2).
    day_numbers : np.ndarray
        Original day values (days since first observation), shape (n_days,).
    svgp_mean_ug : np.ndarray
        Batch SVGP mean in µg/m³ before smoothing, shape (n_locations, n_days).
        Useful for comparing the effect of the smoother.
    svgp_std_ug : np.ndarray
        Batch SVGP std in µg/m³ before smoothing, shape (n_locations, n_days).
    """
    mean_ug: np.ndarray
    std_ug: np.ndarray
    coords: np.ndarray
    day_numbers: np.ndarray
    svgp_mean_ug: np.ndarray
    svgp_std_ug: np.ndarray


class KalmanSmoother:
    """
    Hybrid SVGP + Kalman smoother for day-by-day pollution tracking.

    Uses the trained SVGP's posterior marginals as pseudo-measurements and
    applies an Ornstein-Uhlenbeck Kalman filter + RTS smoother to impose
    Markovian temporal dynamics across days.

    Parameters
    ----------
    model : FusionSVGP
        Trained model. Must have an exponential/matern12 temporal kernel so that
        the OU process interpretation is exact.
    predictor : Predictor
        Existing Predictor instance (with scalers already set).
    scalers : Scalers
        Fitted scalers. Used for coordinate normalisation and inverse transform.
    run_backward_smoother : bool, default=True
        If True, run RTS backward smoother after forward Kalman filter.
        Set to False for online/causal filtering only.
    batch_size : int, default=1024
        Number of spatial locations to predict at once per SVGP call.
    """

    def __init__(
        self,
        model: "FusionSVGP",
        predictor: "Predictor",
        scalers: "Scalers",
        run_backward_smoother: bool = True,
        batch_size: int = 1024,
    ) -> None:
        self.model = model
        self.predictor = predictor
        self.scalers = scalers
        self.run_backward_smoother = run_backward_smoother
        self.batch_size = batch_size

    def _extract_ou_params(self) -> Tuple[float, float]:
        """
        Extract OU process parameters from the trained model's temporal kernel.

        Returns
        -------
        ell : float
            Temporal lengthscale in normalised [0,1] time.
        sigma2 : float
            Output variance = stationary variance of the OU process (P_∞).
        """
        ell = self.model.covar_module.temporal_lengthscale.detach().squeeze().item()
        sigma2 = self.model.covar_module.outputscale.detach().item()
        logger.info(f"OU params: ℓ={ell:.4f}, σ²={sigma2:.4f}")
        return ell, sigma2

    def _compute_transitions(
        self,
        unique_days: np.ndarray,
        ell: float,
        sigma2: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute OU transition A[t] and process noise Q[t] for each day gap.

        Parameters
        ----------
        unique_days : np.ndarray
            Sorted unique day values in original time scale, shape (n_days,).
        ell : float
            Temporal lengthscale in normalised [0,1] time.
        sigma2 : float
            Stationary variance.

        Returns
        -------
        A_arr : np.ndarray, shape (n_days - 1,)
        Q_arr : np.ndarray, shape (n_days - 1,)
        """
        time_range = float(self.scalers.time_max - self.scalers.time_min)
        if time_range <= 0:
            raise ValueError(f"Invalid time range: {time_range}")

        A_arr = np.zeros(len(unique_days) - 1)
        Q_arr = np.zeros(len(unique_days) - 1)

        for i in range(len(unique_days) - 1):
            delta_t_norm = (unique_days[i + 1] - unique_days[i]) / time_range
            A = np.exp(-delta_t_norm / ell)
            Q = sigma2 * (1.0 - A ** 2)
            A_arr[i] = A
            Q_arr[i] = Q

        return A_arr, Q_arr

    def _get_svgp_predictions_for_day(
        self,
        coords_norm: np.ndarray,
        t_norm: float,
        covariates_norm: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get SVGP latent posterior mean and variance at given locations for one day.

        Operates in normalised space — no inverse transform applied here.
        Uses include_noise=False so that R_t is the latent posterior variance,
        not the predictive (observation) variance.

        Parameters
        ----------
        coords_norm : np.ndarray, shape (n_locations, 2)
            Normalised spatial coordinates.
        t_norm : float
            Normalised timestamp in [0, 1].
        covariates_norm : np.ndarray, shape (n_locations, K)
            Normalised covariates (zeros if not available).

        Returns
        -------
        means : np.ndarray, shape (n_locations,)
        variances : np.ndarray, shape (n_locations,)
        """
        n_loc = len(coords_norm)
        t_col = np.full((n_loc, 1), t_norm, dtype=np.float32)
        x_np = np.concatenate([coords_norm, t_col, covariates_norm], axis=1)
        x = torch.tensor(x_np, dtype=torch.float32)

        # _predict_batched returns (mean, std) in normalised space when no scalers
        # are applied. We call it directly with include_noise=False for latent variance.
        means, stds = self.predictor._predict_batched(
            x,
            include_noise=False,
            verbose=False,
        )
        variances = stds ** 2
        return means, variances

    def smooth(
        self,
        coords_original: np.ndarray,
        unique_days: np.ndarray,
        covariates_original: Optional[np.ndarray] = None,
    ) -> SmoothedPredictions:
        """
        Run SVGP + Kalman smoother for all locations across all days.

        Parameters
        ----------
        coords_original : np.ndarray, shape (n_locations, 2)
            Spatial coordinates in original (lat, lon) scale.
        unique_days : np.ndarray, shape (n_days,)
            Sorted unique day values in original time scale (days since first obs).
        covariates_original : np.ndarray, optional, shape (n_locations, K)
            Covariate values at each location in original scale.
            If None, zeros are used (consistent with Predictor behaviour).

        Returns
        -------
        SmoothedPredictions
            Smoothed posterior mean and std in µg/m³ for each (location, day).
        """
        n_loc = len(coords_original)
        n_days = len(unique_days)

        if n_days == 1:
            logger.warning("Only one day — returning SVGP predictions directly.")
            coords_norm = (coords_original - self.scalers.coord_min) / self.scalers.coord_scale
            t_norm = 0.0
            cov_norm = self._normalise_covariates(covariates_original, n_loc)
            means, variances = self._get_svgp_predictions_for_day(coords_norm, t_norm, cov_norm)
            mean_ug, std_ug = self._inverse_transform(means, np.sqrt(variances))
            return SmoothedPredictions(
                mean_ug=mean_ug[:, None],
                std_ug=std_ug[:, None],
                coords=coords_original,
                day_numbers=unique_days,
                svgp_mean_ug=mean_ug[:, None],
                svgp_std_ug=std_ug[:, None],
            )

        # --- Normalise inputs ---
        coords_norm = (coords_original - self.scalers.coord_min) / self.scalers.coord_scale
        coords_norm = coords_norm.astype(np.float32)
        cov_norm = self._normalise_covariates(covariates_original, n_loc)

        time_range = float(self.scalers.time_max - self.scalers.time_min)

        # --- OU parameters ---
        ell, sigma2 = self._extract_ou_params()
        A_arr, Q_arr = self._compute_transitions(unique_days, ell, sigma2)

        # --- Collect SVGP posteriors for all days ---
        svgp_means = np.zeros((n_loc, n_days), dtype=np.float64)
        svgp_vars = np.zeros((n_loc, n_days), dtype=np.float64)

        logger.info(f"Collecting SVGP posteriors for {n_days} days × {n_loc} locations...")
        for d_idx in tqdm(range(n_days), desc="SVGP per day"):
            t_norm = float((unique_days[d_idx] - self.scalers.time_min) / time_range)
            t_norm = float(np.clip(t_norm, 0.0, 1.0))
            m, v = self._get_svgp_predictions_for_day(coords_norm, t_norm, cov_norm)
            svgp_means[:, d_idx] = m
            svgp_vars[:, d_idx] = v

        # --- Forward Kalman filter ---
        logger.info("Running forward Kalman filter...")
        m_fwd = np.zeros((n_loc, n_days), dtype=np.float64)
        P_fwd = np.zeros((n_loc, n_days), dtype=np.float64)

        # Initialise at day 0 with SVGP posterior
        m_fwd[:, 0] = svgp_means[:, 0]
        P_fwd[:, 0] = svgp_vars[:, 0]

        for t in range(1, n_days):
            A = A_arr[t - 1]
            Q = Q_arr[t - 1]

            # Predict
            m_pred = A * m_fwd[:, t - 1]
            P_pred = A ** 2 * P_fwd[:, t - 1] + Q

            # Update (SVGP posterior as pseudo-measurement)
            R = svgp_vars[:, t]
            S = P_pred + R
            K = P_pred / S                              # Kalman gain
            m_fwd[:, t] = m_pred + K * (svgp_means[:, t] - m_pred)
            P_fwd[:, t] = (1.0 - K) * P_pred

        # --- RTS Backward smoother ---
        if self.run_backward_smoother:
            logger.info("Running RTS backward smoother...")
            m_smo = m_fwd.copy()
            P_smo = P_fwd.copy()

            for t in range(n_days - 2, -1, -1):
                A = A_arr[t]
                Q = Q_arr[t]

                P_pred_next = A ** 2 * P_fwd[:, t] + Q
                G = P_fwd[:, t] * A / np.maximum(P_pred_next, 1e-12)  # smoother gain

                m_smo[:, t] = m_fwd[:, t] + G * (m_smo[:, t + 1] - A * m_fwd[:, t])
                P_smo[:, t] = P_fwd[:, t] + G ** 2 * (P_smo[:, t + 1] - P_pred_next)
                P_smo[:, t] = np.maximum(P_smo[:, t], 1e-12)  # keep positive
        else:
            m_smo = m_fwd
            P_smo = P_fwd

        # --- Inverse transform to µg/m³ ---
        mean_ug, std_ug = self._inverse_transform(m_smo, np.sqrt(P_smo))
        svgp_mean_ug, svgp_std_ug = self._inverse_transform(svgp_means, np.sqrt(svgp_vars))

        logger.info("Kalman smoother complete.")
        return SmoothedPredictions(
            mean_ug=mean_ug,
            std_ug=std_ug,
            coords=coords_original,
            day_numbers=unique_days,
            svgp_mean_ug=svgp_mean_ug,
            svgp_std_ug=svgp_std_ug,
        )

    def _normalise_covariates(
        self,
        covariates_original: Optional[np.ndarray],
        n_loc: int,
    ) -> np.ndarray:
        """Return normalised covariates, using zeros if not provided."""
        n_cov = self.model.n_covariates
        if n_cov == 0:
            return np.empty((n_loc, 0), dtype=np.float32)
        if covariates_original is not None:
            cov = (covariates_original - self.scalers.covariate_mean) / (
                self.scalers.covariate_std + 1e-8
            )
        else:
            cov = np.zeros((n_loc, n_cov), dtype=np.float32)
        return cov.astype(np.float32)

    def _inverse_transform(
        self,
        mean_norm: np.ndarray,
        std_norm: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Inverse-transform from normalised target space to µg/m³.

        Works for arrays of any shape (broadcasts over all locations/days).
        """
        scalers = self.scalers
        if getattr(scalers, "normalize_targets", False) and "epa" in scalers.target_std:
            t_mean = float(scalers.target_mean["epa"])
            t_std = float(scalers.target_std["epa"])
            mean_ug = mean_norm * t_std + t_mean
            std_ug = std_norm * t_std
        else:
            mean_ug = mean_norm.copy()
            std_ug = std_norm.copy()
        return mean_ug, std_ug
