"""
ST-SVGP trainer scaffold.

Implements the CVI natural-gradient training loop (Alg. 1 in
arXiv:2111.01732v1). This class is designed to run in parallel to the
existing Trainer (SVGP) without breaking current workflows.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

import logging
import torch

from src.data.loader import FusionData
from src.inference.st_svgp_filter import STSVGPFilter, FilterResult
from src.models.st_svgp import STSVGPModel

logger = logging.getLogger(__name__)


@dataclass
class STSVGPHistory:
    elbo: List[float] = field(default_factory=list)
    epoch_times: List[float] = field(default_factory=list)

    def to_dict(self) -> Dict[str, List[float]]:
        return {
            "elbo": self.elbo,
            "epoch_time": self.epoch_times,
        }


class STSVGPTrainer:
    """
    CVI natural-gradient trainer for ST-SVGP.

    NOTE: This is a scaffold. The full ELBO computation, natural-gradient
    updates, and filtering/smoothing steps must be implemented.
    """

    def __init__(
        self,
        model: STSVGPModel,
        n_epochs: int = 100,
        device: str = "cpu",
        cvi_lr: float = 0.5,
        lr: float = 0.01,
    ) -> None:
        self.model = model
        self.n_epochs = n_epochs
        self.device = device
        self.filter = STSVGPFilter(model, device=device)
        self.history = STSVGPHistory()
        self.cvi_lr = cvi_lr
        self.lr = lr

    def fit(self, data: FusionData) -> STSVGPHistory:
        """
        Train ST-SVGP on the full dataset.

        NOTE: Placeholder. The ST-SVGP algorithm requires:
        - building the state-space model
        - filtering/smoothing over time
        - natural-gradient updates of q(u)
        - hyperparameter optimization (e.g., Adam on ELBO)
        """
        # Build per-day observation batches for filtering
        unique_days = np.unique(data.timestamps)
        day_obs = self._build_day_obs(data, unique_days)

        # Initialize prior mean/cov (placeholders; to be replaced by q(u) init)
        M_s = self.model.Z_s.shape[0]
        dtype = self.model.Z_s.dtype
        m0 = torch.zeros(M_s, dtype=dtype)
        P0 = torch.eye(M_s, dtype=dtype)
        if M_s > 500:
            logger.warning(
                "ST-SVGP dense covariance on CPU can be very slow for M_s > 500 "
                f"(M_s={M_s}). Consider reducing inducing points."
            )

        # CVI loop scaffold: for Gaussian likelihoods, CVI reduces to exact filtering.
        # We record the log-likelihood proxy ELBO per epoch.
        logger.warning(
            "ST-SVGP CVI loop is a scaffold: no natural-gradient updates "
            "or hyperparameter optimization are performed yet."
        )
        # Convert day_obs to torch for differentiable filtering
        day_obs_t = self._build_day_obs_torch(day_obs)
        unique_days_t = torch.tensor(unique_days, dtype=dtype)

        # Initialize CVI site parameters (scaffold)
        site_params = self._init_site_params(day_obs_t)

        # Optimizer over kernel + likelihood + inducing locations
        params = list(self.model.parameters())
        optim = torch.optim.Adam(params, lr=self.lr)

        for _ in range(self.n_epochs):
            site_params, day_obs_t = self._cvi_update(site_params, day_obs_t)
            m_filt, P_filt, loglik = self.filter.filter_by_day_torch(
                day_obs_t, unique_days_t, m0, P0
            )
            A, Q = self.filter.build_state_space_torch(unique_days_t)
            m_smooth, P_smooth = self.filter.smooth_torch(m_filt, P_filt, A, Q)
            # KL(q(u)||p(u)) at t=0 (placeholder for full ELBO term)
            kl = self._kl_qp(m_smooth[0], P_smooth[0])
            elbo = loglik - kl
            loss = -elbo
            optim.zero_grad()
            loss.backward()
            optim.step()
            self.history.elbo.append(float(elbo.detach().cpu().item()))

        # Cache last smoother outputs for prediction
        self._last_smooth = (m_smooth.detach(), P_smooth.detach(), unique_days_t.detach())

        return self.history

    def _kl_qp(self, m: torch.Tensor, P: torch.Tensor) -> torch.Tensor:
        """
        KL between q(u)=N(m,P) and p(u)=N(0,K_MM) for spatial inducing points.
        This is a placeholder term; full ST-SVGP ELBO requires time-structured KL.
        """
        Z = self.model.Z_s
        K_MM = self.model.spatial_kernel(Z, Z)
        if hasattr(K_MM, "to_dense"):
            K_MM = K_MM.to_dense()
        K_MM = K_MM.to(torch.float64)
        P = P.to(torch.float64)
        m = m.to(torch.float64)

        jitter = 1e-6 * torch.eye(K_MM.shape[0], dtype=torch.float64, device=K_MM.device)
        Lp = torch.linalg.cholesky(K_MM + jitter)
        Lq = torch.linalg.cholesky(P + jitter)

        K_inv = torch.cholesky_inverse(Lp)
        trace_term = torch.trace(K_inv @ P)
        quad_term = m @ (K_inv @ m)
        logdet_p = 2.0 * torch.sum(torch.log(torch.diag(Lp)))
        logdet_q = 2.0 * torch.sum(torch.log(torch.diag(Lq)))
        k = K_MM.shape[0]
        return 0.5 * (trace_term + quad_term - k + logdet_p - logdet_q)

    def predict(self, grid_coords_norm: torch.Tensor, day_idx: Optional[int] = None):
        """
        Predict at grid coordinates using the last smoothed posterior.

        grid_coords_norm: normalized coords (N, 2).
        day_idx: if provided, return a single day's prediction.
        """
        if not hasattr(self, "_last_smooth"):
            raise RuntimeError("No fitted posterior found. Call fit() first.")

        m_smooth, P_smooth, unique_days_t = self._last_smooth
        if day_idx is None:
            means = []
            vars_ = []
            for t in range(m_smooth.shape[0]):
                m_t = m_smooth[t]
                P_t = P_smooth[t]
                mean, var = self.filter.predict_grid_torch(grid_coords_norm, m_t, P_t)
                means.append(mean)
                vars_.append(var)
            return torch.stack(means, dim=1), torch.stack(vars_, dim=1), unique_days_t
        else:
            m_t = m_smooth[day_idx]
            P_t = P_smooth[day_idx]
            mean, var = self.filter.predict_grid_torch(grid_coords_norm, m_t, P_t)
            return mean, var, unique_days_t[day_idx]

    def _build_day_obs(self, data: FusionData, unique_days: np.ndarray) -> list[dict]:
        """
        Build per-day observation dicts for filter_by_day.

        Each entry contains:
          - y: (n_obs,)
          - coords_norm: (n_obs, 2)
          - sources: (n_obs,)
        """
        source_list = list(getattr(self.model.likelihood, "sources", []))
        source_to_idx = {s: i for i, s in enumerate(source_list)}
        day_obs: list[dict] = []
        for day_val in unique_days:
            day_mask = np.isclose(data.timestamps, day_val, atol=0.1)
            y_list = []
            coord_list = []
            src_list = []

            for source, src_mask in data.source_masks.items():
                combined = day_mask & src_mask
                if not combined.any():
                    continue
                src_idx = source_to_idx.get(source, None)
                if src_idx is None:
                    continue
                y_vals = data.observations[source][combined].astype(np.float64)
                coords = data.coords[combined].astype(np.float64)
                y_list.append(y_vals)
                coord_list.append(coords)
                src_list.append(np.full(len(y_vals), src_idx))

            if y_list:
                day_obs.append({
                    "y": np.concatenate(y_list),
                    "coords_norm": np.concatenate(coord_list, axis=0),
                    "sources": np.concatenate(src_list),
                })
            else:
                day_obs.append({})
        return day_obs

    def _build_day_obs_torch(self, day_obs: list[dict]) -> list[dict]:
        out = []
        for obs in day_obs:
            if not obs:
                out.append({})
                continue
            out.append({
                "y": torch.tensor(obs["y"], dtype=self.model.Z_s.dtype),
                "coords_norm": torch.tensor(obs["coords_norm"], dtype=self.model.Z_s.dtype),
                "sources": torch.tensor(obs["sources"], dtype=torch.int64),
            })
        return out

    def _init_site_params(self, day_obs_t: list[dict]) -> list[dict]:
        """
        Initialize CVI site parameters for each day.

        For now, we store placeholders for natural parameters. For Gaussian
        likelihoods this is equivalent to the observation model used directly
        in filtering.
        """
        sites = []
        for obs in day_obs_t:
            if not obs:
                sites.append({})
                continue
            # Placeholder: natural params for a Gaussian site.
            sites.append({
                "eta1": torch.zeros_like(obs["y"]),  # precision * mean
                "eta2": torch.zeros_like(obs["y"]),  # -0.5 * precision
            })
        return sites

    def _cvi_update(self, site_params: list[dict], day_obs_t: list[dict]) -> tuple[list[dict], list[dict]]:
        """
        CVI natural-gradient update (scaffold).

        For Gaussian likelihoods, the exact update yields closed-form
        natural parameters. We compute latent-space pseudo-observations:
          y_latent = (y - b) / a
          noise_var = sigma^2 / a^2
        This is equivalent to the Gaussian CVI site for each observation.
        """
        source_list = list(getattr(self.model.likelihood, "sources", []))
        source_to_idx = {s: i for i, s in enumerate(source_list)}

        updated_obs = []
        for obs in day_obs_t:
            if not obs:
                updated_obs.append({})
                continue

            y = obs["y"]
            sources = obs["sources"]

            y_latent = torch.zeros_like(y)
            noise_var = torch.zeros_like(y)

            for src_name, src_idx in source_to_idx.items():
                mask = sources == src_idx
                if not torch.any(mask):
                    continue
                if src_name == "satellite":
                    a = self.model.likelihood.sat_slope
                    b = self.model.likelihood.sat_intercept
                elif src_name == "low_cost":
                    a = self.model.likelihood.lc_slope
                    b = self.model.likelihood.lc_intercept
                else:
                    a = torch.tensor(1.0, dtype=y.dtype, device=y.device)
                    b = torch.tensor(0.0, dtype=y.dtype, device=y.device)

                sigma2 = self.model.likelihood.noise_variance[src_name]
                y_latent[mask] = (y[mask] - b) / a
                noise_var[mask] = sigma2 / (a ** 2)

            obs2 = dict(obs)
            obs2["y_latent"] = y_latent
            obs2["noise_var"] = noise_var
            updated_obs.append(obs2)

        return site_params, updated_obs
