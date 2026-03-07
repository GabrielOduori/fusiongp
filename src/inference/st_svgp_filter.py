"""
ST-SVGP filtering and smoothing scaffold.

Implements the state-space filtering/smoothing over spatial inducing
points as described in arXiv:2111.01732v1. This module will be used by
STSVGPTrainer for CVI natural-gradient updates.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import torch

from src.models.st_svgp import STSVGPModel


@dataclass
class FilterResult:
    m_filt: np.ndarray  # (T, M_s)
    P_filt: np.ndarray  # (T, M_s, M_s)
    loglik: float = 0.0


@dataclass
class SmoothResult:
    m_smooth: np.ndarray  # (T, M_s)
    P_smooth: np.ndarray  # (T, M_s, M_s)


class STSVGPFilter:
    """
    Sequential filter/smoother over spatial inducing points.

    This is a scaffold: the full algorithm (Alg. 2/4 in the paper) needs
    implementation for a production path. The interfaces are defined to
    keep the training pipeline stable.
    """

    def __init__(self, model: STSVGPModel, device: str = "cpu") -> None:
        self.model = model
        self.device = torch.device(device)
        self._source_idx_to_name = list(getattr(self.model.likelihood, "sources", []))

    def build_state_space(
        self,
        unique_days: np.ndarray,
        time_range: Optional[float] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Build OU-style state transition (A) and process noise (Q) arrays.

        This implements the exponential/Matern-1/2 temporal kernel conversion
        to an OU process. For other kernels, this should be replaced with the
        appropriate state-space conversion from the paper.
        """
        if unique_days.ndim != 1 or len(unique_days) < 2:
            raise ValueError("unique_days must be 1D with length >= 2")

        # Temporal lengthscale/outputscale from the shared kernel
        ell = self.model.covar_module.temporal_kernel.lengthscale.detach().squeeze().item()
        sigma2 = self.model.covar_module.outputscale.detach().item()

        if time_range is None:
            time_range = float(unique_days.max() - unique_days.min())
            if time_range <= 0:
                time_range = 1.0

        A_arr = np.zeros(len(unique_days) - 1, dtype=np.float64)
        Q_arr = np.zeros(len(unique_days) - 1, dtype=np.float64)
        for i in range(len(unique_days) - 1):
            dt_norm = (unique_days[i + 1] - unique_days[i]) / time_range
            A = float(np.exp(-dt_norm / ell))
            A_arr[i] = A
            Q_arr[i] = sigma2 * (1.0 - A ** 2)

        return A_arr, Q_arr

    def build_state_space_torch(
        self,
        unique_days: torch.Tensor,
        time_range: Optional[float] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Torch version of build_state_space (OU conversion).
        """
        if unique_days.ndim != 1 or unique_days.numel() < 2:
            raise ValueError("unique_days must be 1D with length >= 2")

        ell = self.model.covar_module.temporal_kernel.lengthscale.squeeze()
        sigma2 = self.model.covar_module.outputscale

        if time_range is None:
            time_range = float((unique_days.max() - unique_days.min()).item())
            if time_range <= 0:
                time_range = 1.0

        dt = (unique_days[1:] - unique_days[:-1]) / time_range
        A = torch.exp(-dt / ell)
        Q = sigma2 * (1.0 - A ** 2)
        return A, Q

    def filter(
        self,
        y_obs: np.ndarray,
        H: np.ndarray,
        R: np.ndarray,
        unique_days: np.ndarray,
        m0: np.ndarray,
        P0: np.ndarray,
    ) -> FilterResult:
        """
        Run sequential filtering.

        NOTE: Placeholder for Alg. 3/4 (sequential/parallel filtering).
        This assumes a single shared H/R across all days. Prefer filter_by_day
        for per-day observation updates.
        """
        if len(unique_days) < 2:
            raise ValueError("unique_days must have length >= 2")

        n_days = len(unique_days)
        m_filt = np.zeros((n_days, m0.shape[0]), dtype=np.float64)
        P_filt = np.zeros((n_days, P0.shape[0], P0.shape[1]), dtype=np.float64)

        A_arr, Q_arr = self.build_state_space(unique_days)

        m_t = m0.copy()
        P_t = P0.copy()

        for t in range(n_days):
            if t > 0:
                A = A_arr[t - 1]
                Q = Q_arr[t - 1]
                m_t = A * m_t
                P_t = A ** 2 * P_t + Q * np.eye(P_t.shape[0])

            # Single shared H/R for now (placeholder). Multi-day batching should
            # construct per-day H/R and call filter per day.
            if y_obs is not None and H is not None and R is not None and len(y_obs) > 0:
                HP = H @ P_t
                S = HP @ H.T + R
                try:
                    L_S = np.linalg.cholesky(S + 1e-8 * np.eye(S.shape[0]))
                    Kt = np.linalg.solve(L_S, H @ P_t.T).T
                    K = np.linalg.solve(L_S.T, Kt.T).T
                except np.linalg.LinAlgError:
                    K = P_t @ H.T @ np.linalg.pinv(S)
                innovation = y_obs - H @ m_t
                m_t = m_t + K @ innovation
                P_t = (np.eye(P_t.shape[0]) - K @ H) @ P_t
                P_t = 0.5 * (P_t + P_t.T)

            m_filt[t] = m_t
            P_filt[t] = P_t

        return FilterResult(m_filt=m_filt, P_filt=P_filt)

    def filter_by_day(
        self,
        day_obs: list[dict],
        unique_days: np.ndarray,
        m0: np.ndarray,
        P0: np.ndarray,
    ) -> FilterResult:
        """
        Run sequential filtering with per-day, per-source observations.

        day_obs: list of dicts, length n_days. Each dict can include:
          - y: np.ndarray (n_obs,)
          - coords_norm: np.ndarray (n_obs, 2)
          - sources: np.ndarray (n_obs,) of source labels
        """
        if len(unique_days) != len(day_obs):
            raise ValueError("unique_days and day_obs must have the same length")

        n_days = len(unique_days)
        m_filt = np.zeros((n_days, m0.shape[0]), dtype=np.float64)
        P_filt = np.zeros((n_days, P0.shape[0], P0.shape[1]), dtype=np.float64)
        loglik = 0.0

        A_arr, Q_arr = self.build_state_space(unique_days)

        m_t = m0.copy()
        P_t = P0.copy()

        for t in range(n_days):
            if t > 0:
                A = A_arr[t - 1]
                Q = Q_arr[t - 1]
                m_t = A * m_t
                P_t = A ** 2 * P_t + Q * np.eye(P_t.shape[0])

            obs = day_obs[t]
            if obs and len(obs.get("y", [])) > 0:
                y_all = obs["y"]
                coords = obs["coords_norm"]
                sources = obs["sources"]

                H_blocks = []
                y_blocks = []
                intercept_blocks = []
                noise_blocks = []

                for src in np.unique(sources):
                    src_mask = sources == src
                    y_src = y_all[src_mask]
                    coords_src = coords[src_mask]
                    H, slope, intercept = self.observation_matrices(coords_src, src)
                    H_scaled = H * slope

                    H_blocks.append(H_scaled)
                    y_blocks.append(y_src)
                    intercept_blocks.append(np.full(len(y_src), intercept))
                    noise_blocks.append(
                        np.full(len(y_src), self._source_noise_var_obs(src))
                    )

                H_all = np.vstack(H_blocks)
                y_vec = np.concatenate(y_blocks)
                intercept_vec = np.concatenate(intercept_blocks)
                noise_vec = np.concatenate(noise_blocks)

                HP = H_all @ P_t
                S = HP @ H_all.T + np.diag(noise_vec)
                try:
                    L_S = np.linalg.cholesky(S + 1e-8 * np.eye(S.shape[0]))
                    Kt = np.linalg.solve(L_S, H_all @ P_t.T).T
                    K = np.linalg.solve(L_S.T, Kt.T).T
                except np.linalg.LinAlgError:
                    K = P_t @ H_all.T @ np.linalg.pinv(S)

                innovation = y_vec - (H_all @ m_t + intercept_vec)
                # Log-likelihood contribution (Gaussian)
                try:
                    v = np.linalg.solve(L_S, innovation)
                    logdet = 2.0 * np.sum(np.log(np.diag(L_S)))
                    loglik += -0.5 * (len(innovation) * np.log(2.0 * np.pi) + logdet + v.T @ v)
                except Exception:
                    pass
                m_t = m_t + K @ innovation
                P_t = (np.eye(P_t.shape[0]) - K @ H_all) @ P_t
                P_t = 0.5 * (P_t + P_t.T)

            m_filt[t] = m_t
            P_filt[t] = P_t

        return FilterResult(m_filt=m_filt, P_filt=P_filt, loglik=loglik)

    def filter_by_day_torch(
        self,
        day_obs: list[dict],
        unique_days: torch.Tensor,
        m0: torch.Tensor,
        P0: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Torch sequential filtering with per-day, per-source observations.
        Returns (m_filt, P_filt, loglik).
        """
        if len(unique_days) != len(day_obs):
            raise ValueError("unique_days and day_obs must have the same length")

        n_days = len(unique_days)
        n_state = m0.shape[0]
        m_filt = torch.zeros((n_days, n_state), dtype=m0.dtype, device=m0.device)
        P_filt = torch.zeros((n_days, n_state, n_state), dtype=P0.dtype, device=P0.device)
        loglik = torch.tensor(0.0, dtype=m0.dtype, device=m0.device)

        A_arr, Q_arr = self.build_state_space_torch(unique_days)

        m_t = m0.clone()
        P_t = P0.clone()

        eye = torch.eye(n_state, dtype=P0.dtype, device=P0.device)

        for t in range(n_days):
            if t > 0:
                A = A_arr[t - 1]
                Q = Q_arr[t - 1]
                m_t = A * m_t
                P_t = A ** 2 * P_t + Q * eye

            obs = day_obs[t]
            if obs and len(obs.get("y", [])) > 0:
                use_latent = "y_latent" in obs and "noise_var" in obs
                y_all = obs["y_latent"] if use_latent else obs["y"]
                coords = obs["coords_norm"]
                sources = obs["sources"]

                H_blocks = []
                y_blocks = []
                intercept_blocks = []
                noise_blocks = []

                for src in torch.unique(sources):
                    src_mask = sources == src
                    y_src = y_all[src_mask]
                    coords_src = coords[src_mask]
                    src_name = self._source_idx_to_name[int(src.item())]
                    H, slope, intercept = self.observation_matrices_torch(coords_src, src_name)
                    if use_latent:
                        H_scaled = H
                        intercept = torch.tensor(0.0, device=H.device, dtype=H.dtype)
                    else:
                        H_scaled = H * slope

                    H_blocks.append(H_scaled)
                    y_blocks.append(y_src)
                    intercept_blocks.append(torch.full_like(y_src, intercept))
                    noise_blocks.append(
                        (obs["noise_var"][src_mask] if use_latent else
                         torch.full_like(y_src, self._source_noise_var_obs(src_name), dtype=P0.dtype))
                    )

                H_all = torch.cat(H_blocks, dim=0)
                y_vec = torch.cat(y_blocks, dim=0)
                intercept_vec = torch.cat(intercept_blocks, dim=0)
                noise_vec = torch.cat(noise_blocks, dim=0)

                HP = H_all @ P_t
                S = HP @ H_all.T + torch.diag(noise_vec)

                L_S = torch.linalg.cholesky(S + 1e-8 * torch.eye(S.shape[0], device=S.device, dtype=S.dtype))
                # Solve S X = H P^T  -> X = S^{-1} H P^T
                Kt = torch.cholesky_solve((H_all @ P_t.T), L_S)
                # K = P H^T S^{-1} = (S^{-1} H P^T)^T
                K = Kt.T

                innovation = y_vec - (H_all @ m_t + intercept_vec)
                v = torch.cholesky_solve(innovation.unsqueeze(1), L_S).squeeze(1)
                logdet = 2.0 * torch.sum(torch.log(torch.diag(L_S)))
                loglik = loglik + (-0.5 * (len(innovation) * np.log(2.0 * np.pi) + logdet + v @ v))

                m_t = m_t + K @ innovation
                P_t = (eye - K @ H_all) @ P_t
                P_t = 0.5 * (P_t + P_t.T)

            m_filt[t] = m_t
            P_filt[t] = P_t

        return m_filt, P_filt, loglik

    def smooth(
        self,
        filt: FilterResult,
        A: np.ndarray,
        Q: np.ndarray,
    ) -> SmoothResult:
        """
        Run smoothing over filtered states.

        NOTE: Placeholder for Alg. 2/4 (smoothing).
        """
        m_filt = filt.m_filt
        P_filt = filt.P_filt
        n_days, n_state = m_filt.shape

        m_smooth = m_filt.copy()
        P_smooth = P_filt.copy()

        for t in range(n_days - 2, -1, -1):
            A_t = A[t]
            Q_t = Q[t]
            P_pred = A_t ** 2 * P_filt[t] + Q_t * np.eye(n_state)
            try:
                C = P_filt[t] * A_t @ np.linalg.pinv(P_pred)
            except np.linalg.LinAlgError:
                C = P_filt[t] * A_t @ np.linalg.pinv(P_pred + 1e-8 * np.eye(n_state))
            m_smooth[t] = m_filt[t] + C @ (m_smooth[t + 1] - A_t * m_filt[t])
            P_smooth[t] = P_filt[t] + C @ (P_smooth[t + 1] - P_pred) @ C.T
            P_smooth[t] = 0.5 * (P_smooth[t] + P_smooth[t].T)

        return SmoothResult(m_smooth=m_smooth, P_smooth=P_smooth)

    def smooth_torch(
        self,
        m_filt: torch.Tensor,
        P_filt: torch.Tensor,
        A: torch.Tensor,
        Q: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Torch RTS smoother (dense).
        Returns (m_smooth, P_smooth).
        """
        n_days, n_state = m_filt.shape
        m_smooth = m_filt.clone()
        P_smooth = P_filt.clone()
        eye = torch.eye(n_state, dtype=P_filt.dtype, device=P_filt.device)

        for t in range(n_days - 2, -1, -1):
            A_t = A[t]
            Q_t = Q[t]
            P_pred = A_t ** 2 * P_filt[t] + Q_t * eye
            # RTS gain
            C = (P_filt[t] * A_t) @ torch.linalg.pinv(P_pred)
            m_smooth[t] = m_filt[t] + C @ (m_smooth[t + 1] - A_t * m_filt[t])
            P_smooth[t] = P_filt[t] + C @ (P_smooth[t + 1] - P_pred) @ C.T
            P_smooth[t] = 0.5 * (P_smooth[t] + P_smooth[t].T)

        return m_smooth, P_smooth

    def observation_matrices(
        self,
        obs_coords_norm: np.ndarray,
        source: str,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Build observation model for a given source.

        Returns (H, slope, intercept) where y = slope * H f + intercept + eps.
        """
        # Build H = K(s_obs, Z_s) @ K_MM^{-1}
        Z_s = self.model.Z_s.detach()
        obs_t = torch.tensor(obs_coords_norm, dtype=torch.float32, device=self.device)
        Z_t = Z_s.to(self.device).float()

        with torch.no_grad():
            K_sZ = self.model.spatial_kernel(obs_t, Z_t)
            if hasattr(K_sZ, "to_dense"):
                K_sZ = K_sZ.to_dense()

        K_sZ_np = K_sZ.cpu().numpy().astype(np.float64)

        # Cache K_MM_inv for reuse
        if not hasattr(self, "_K_MM_inv"):
            with torch.no_grad():
                K_MM = self.model.spatial_kernel(Z_t, Z_t)
                if hasattr(K_MM, "to_dense"):
                    K_MM = K_MM.to_dense()
                K_MM = K_MM.double()
                jitter = 1e-6 * torch.eye(K_MM.shape[0], dtype=torch.float64, device=K_MM.device)
                try:
                    L = torch.linalg.cholesky(K_MM + jitter)
                except RuntimeError:
                    jitter = 1e-4 * torch.eye(K_MM.shape[0], dtype=torch.float64, device=K_MM.device)
                    L = torch.linalg.cholesky(K_MM + jitter)
                K_MM_inv = torch.cholesky_inverse(L)
            self._K_MM_inv = K_MM_inv.cpu().numpy().astype(np.float64)

        H = K_sZ_np @ self._K_MM_inv

        # Calibration parameters for the observation model
        if source == "satellite":
            with torch.no_grad():
                slope = float(self.model.likelihood.sat_slope.item())
                intercept = float(self.model.likelihood.sat_intercept.item())
        elif source == "low_cost":
            with torch.no_grad():
                slope = float(self.model.likelihood.lc_slope.item())
                intercept = float(self.model.likelihood.lc_intercept.item())
        else:
            slope, intercept = 1.0, 0.0

        return H, np.array(slope, dtype=np.float64), np.array(intercept, dtype=np.float64)

    def _source_noise_var_obs(self, source: str) -> float:
        with torch.no_grad():
            var = self.model.likelihood.noise_variance[source].item()
        return float(var)

    def observation_matrices_torch(
        self,
        obs_coords_norm: torch.Tensor,
        source: str,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Torch version of observation_matrices with gradients.
        Returns (H, slope, intercept).
        """
        Z_s = self.model.Z_s.to(self.device)
        obs_t = obs_coords_norm.to(self.device).to(Z_s.dtype)
        Z_t = Z_s

        K_sZ = self.model.spatial_kernel(obs_t, Z_t)
        if hasattr(K_sZ, "to_dense"):
            K_sZ = K_sZ.to_dense()

        # Cache K_MM_inv in torch
        K_MM = self.model.spatial_kernel(Z_t, Z_t)
        if hasattr(K_MM, "to_dense"):
            K_MM = K_MM.to_dense()
        K_MM = K_MM.to(torch.float64)
        jitter = 1e-6 * torch.eye(K_MM.shape[0], dtype=torch.float64, device=K_MM.device)
        try:
            L = torch.linalg.cholesky(K_MM + jitter)
        except RuntimeError:
            jitter = 1e-4 * torch.eye(K_MM.shape[0], dtype=torch.float64, device=K_MM.device)
            L = torch.linalg.cholesky(K_MM + jitter)
        K_MM_inv = torch.cholesky_inverse(L).to(obs_t.dtype)

        H = K_sZ @ K_MM_inv

        if source == "satellite":
            slope = self.model.likelihood.sat_slope
            intercept = self.model.likelihood.sat_intercept
        elif source == "low_cost":
            slope = self.model.likelihood.lc_slope
            intercept = self.model.likelihood.lc_intercept
        else:
            slope = torch.tensor(1.0, device=obs_t.device, dtype=obs_t.dtype)
            intercept = torch.tensor(0.0, device=obs_t.device, dtype=obs_t.dtype)

        return H, slope, intercept

    def predict_grid_torch(
        self,
        grid_coords_norm: torch.Tensor,
        m_t: torch.Tensor,
        P_t: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict mean/variance at grid coords given current state.
        """
        Z_t = self.model.Z_s.to(self.device)
        g_t = grid_coords_norm.to(self.device).to(Z_t.dtype)

        K_gZ = self.model.spatial_kernel(g_t, Z_t)
        if hasattr(K_gZ, "to_dense"):
            K_gZ = K_gZ.to_dense()

        K_MM = self.model.spatial_kernel(Z_t, Z_t)
        if hasattr(K_MM, "to_dense"):
            K_MM = K_MM.to_dense()
        K_MM = K_MM.to(torch.float64)
        jitter = 1e-6 * torch.eye(K_MM.shape[0], dtype=torch.float64, device=K_MM.device)
        try:
            L = torch.linalg.cholesky(K_MM + jitter)
        except RuntimeError:
            jitter = 1e-4 * torch.eye(K_MM.shape[0], dtype=torch.float64, device=K_MM.device)
            L = torch.linalg.cholesky(K_MM + jitter)
        K_MM_inv = torch.cholesky_inverse(L).to(g_t.dtype)

        alpha = K_MM_inv @ K_gZ.T
        mean = m_t @ alpha

        k_diag = self.model.covar_module.spatial_kernel(g_t, g_t, diag=True)
        if hasattr(k_diag, "to_dense"):
            k_diag = k_diag.to_dense()
        prior_reduction = torch.einsum("ij,ji->i", K_gZ, alpha)
        P_alpha = P_t @ alpha
        post_add = torch.einsum("ij,ji->i", K_gZ, K_MM_inv @ P_alpha)
        var = torch.clamp(k_diag - prior_reduction + post_add, min=1e-12)

        return mean, var
