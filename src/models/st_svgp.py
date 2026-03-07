"""
Spatio-Temporal SVGP (ST-SVGP) model scaffold.

This module defines a minimal, fusion-ready model wrapper for the
ST-SVGP algorithm described in arXiv:2111.01732v1. The full variational
inference is implemented in src/training/st_svgp_trainer.py and the
filter/smoother in src/inference/st_svgp_filter.py.

This is a scaffold intended to keep integration points stable while the
filtering/smoothing and CVI natural-gradient updates are implemented.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch

from src.models.kernels import SpatioTemporalKernel
from src.models.likelihoods import MultiSourceLikelihood


@dataclass
class STSVGPConfig:
    spatial_kernel_type: str = "matern32"
    temporal_kernel_type: str = "exponential"
    spatial_ard: bool = True
    n_spatial_inducing: int = 100
    learn_inducing_locations: bool = True


class STSVGPModel(torch.nn.Module):
    """
    Minimal ST-SVGP model wrapper.

    Holds:
    - Spatio-temporal kernel
    - Multi-source likelihood
    - Spatial inducing locations (Z_s)

    The full variational posterior q(u) is maintained by the STSVGPTrainer.
    """

    def __init__(
        self,
        config: STSVGPConfig,
        likelihood: Optional[MultiSourceLikelihood] = None,
        initial_Zs: Optional[torch.Tensor] = None,
        device: str = "cpu",
    ) -> None:
        super().__init__()
        self.config = config
        self.device = torch.device(device)

        self.covar_module = SpatioTemporalKernel(
            spatial_kernel_type=config.spatial_kernel_type,
            temporal_kernel_type=config.temporal_kernel_type,
            spatial_ard=config.spatial_ard,
        )
        self.likelihood = likelihood or MultiSourceLikelihood()

        if initial_Zs is None:
            # Caller should set Z_s after data load; keep a placeholder tensor.
            initial_Zs = torch.zeros((config.n_spatial_inducing, 2), dtype=torch.float32)

        if config.learn_inducing_locations:
            self.Z_s = torch.nn.Parameter(initial_Zs.to(self.device))
        else:
            self.register_buffer("Z_s", initial_Zs.to(self.device))

    def spatial_kernel(self, X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
        """Spatial kernel K_s(X, Y)."""
        pad_x = torch.zeros((X.shape[0], 1), device=X.device, dtype=X.dtype)
        pad_y = torch.zeros((Y.shape[0], 1), device=Y.device, dtype=Y.dtype)
        X3 = torch.cat([X, pad_x], dim=-1)
        Y3 = torch.cat([Y, pad_y], dim=-1)
        return self.covar_module.spatial_kernel(X3, Y3, diag=False)

    def temporal_kernel_params(self) -> dict:
        """Return temporal kernel hyperparameters needed for state-space conversion."""
        return self.covar_module.get_hyperparameters()
