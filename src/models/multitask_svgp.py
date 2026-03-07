"""
Multitask Sparse Variational GP model with coregionalization.

Implements a shared spatio-temporal kernel with a task covariance matrix
to produce per-source outputs in a single model.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

import torch
from gpytorch.models import ApproximateGP
from gpytorch.variational import (
    CholeskyVariationalDistribution,
    VariationalStrategy,
    LMCVariationalStrategy,
)
from gpytorch.means import ConstantMean
from gpytorch.distributions import MultivariateNormal

from src.models.kernels import SpatioTemporalKernel
from src.models.likelihoods import MaskedMultitaskGaussianLikelihood

logger = logging.getLogger(__name__)


class MultiTaskSVGP(ApproximateGP):
    """
    Multitask SVGP with a coregionalization kernel.
    """

    def __init__(
        self,
        n_inducing: int = 500,
        kernel_type: str = "matern32",
        spatial_kernel_type: Optional[str] = None,
        temporal_kernel_type: Optional[str] = None,
        spatial_ard: bool = True,
        learn_inducing_locations: bool = True,
        n_covariates: int = 0,
        initial_lengthscales: Optional[Dict[str, float]] = None,
        initial_noise: Optional[Dict[str, float]] = None,
        learn_kernel_hyperparams: bool = True,
        learn_noise: bool = True,
        learn_noise_sources: Optional[List[str]] = None,
        sources: Optional[List[str]] = None,
        task_rank: int = 1,
        learn_calibration: bool = True,
    ):
        self.n_inducing = n_inducing
        self.kernel_type = kernel_type
        self.learn_inducing_locations = learn_inducing_locations
        self.n_covariates = n_covariates
        self.spatial_kernel_type = spatial_kernel_type or kernel_type
        self.temporal_kernel_type = temporal_kernel_type or kernel_type
        self.sources = sources if sources else ["epa", "low_cost", "satellite"]
        self.num_tasks = len(self.sources)
        self.task_rank = task_rank

        ls_init = initial_lengthscales or {}
        spatial_ls = (
            ls_init.get("spatial_x", 0.1),
            ls_init.get("spatial_y", 0.1),
        )
        temporal_ls = ls_init.get("temporal", 0.1)

        inducing_points = torch.rand(n_inducing, 3 + self.n_covariates)
        variational_distribution = CholeskyVariationalDistribution(
            n_inducing,
            batch_shape=torch.Size([self.task_rank]),
        )
        base_variational_strategy = VariationalStrategy(
            self,
            inducing_points,
            variational_distribution,
            learn_inducing_locations=learn_inducing_locations,
        )
        variational_strategy = LMCVariationalStrategy(
            base_variational_strategy,
            num_tasks=self.num_tasks,
            num_latents=self.task_rank,
        )

        super().__init__(variational_strategy)

        self.mean_module = ConstantMean(batch_shape=torch.Size([self.task_rank]))

        covariate_dims = (
            list(range(3, 3 + self.n_covariates)) if self.n_covariates > 0 else None
        )
        base_kernel = SpatioTemporalKernel(
            spatial_kernel_type=self.spatial_kernel_type,
            temporal_kernel_type=self.temporal_kernel_type,
            spatial_ard=spatial_ard,
            initial_spatial_lengthscale=spatial_ls,
            initial_temporal_lengthscale=temporal_ls,
            covariate_dims=covariate_dims,
            batch_shape=torch.Size([self.task_rank]),
        )
        if not learn_kernel_hyperparams:
            base_kernel.spatial_kernel.raw_lengthscale.requires_grad_(False)
            base_kernel.temporal_kernel.raw_lengthscale.requires_grad_(False)
            base_kernel.outputscale_param.requires_grad_(False)

        self.covar_module = base_kernel

        self.likelihood = MaskedMultitaskGaussianLikelihood(
            sources=self.sources,
            initial_noise=initial_noise,
            learn_noise=learn_noise,
            learn_noise_sources=learn_noise_sources,
        )

        logger.info(
            f"Created MultiTaskSVGP: n_inducing={n_inducing}, tasks={self.sources}, "
            f"kernel={kernel_type}, rank={task_rank}"
        )

    def forward(self, x: torch.Tensor) -> MultivariateNormal:
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)

    def initialize_inducing_points(
        self,
        x: torch.Tensor,
        method: str = "kmeans",
        random_state: int = 42,
        spatial_inducing: Optional[int] = None,
        temporal_inducing: Optional[int] = None,
    ) -> None:
        from sklearn.cluster import KMeans

        n_data = x.shape[0]
        n_inducing = min(self.n_inducing, n_data)

        if n_inducing != self.n_inducing:
            logger.warning(
                f"Requested {self.n_inducing} inducing points, but only {n_data} data points available. "
                f"Using {n_inducing} inducing points instead."
            )

        if method == "kmeans":
            x_np = x.detach().cpu().numpy()
            kmeans = KMeans(
                n_clusters=n_inducing,
                random_state=random_state,
                n_init=10,
            )
            kmeans.fit(x_np)
            inducing_points = torch.tensor(
                kmeans.cluster_centers_,
                dtype=x.dtype,
                device=x.device,
            )
        elif method == "random":
            torch.manual_seed(random_state)
            indices = torch.randperm(n_data)[:n_inducing]
            inducing_points = x[indices].clone()
        elif method == "structured_grid":
            inducing_points, _ = self._create_structured_grid(
                x, n_inducing, spatial_inducing, temporal_inducing
            )
            n_inducing = inducing_points.shape[0]
        else:
            raise ValueError(f"Unknown initialization method: {method}")

        base_strategy = self.variational_strategy.base_variational_strategy
        if n_inducing != base_strategy.inducing_points.shape[0]:
            variational_distribution = CholeskyVariationalDistribution(
                n_inducing,
                batch_shape=torch.Size([self.task_rank]),
            )
            base_variational_strategy = VariationalStrategy(
                self,
                inducing_points,
                variational_distribution,
                learn_inducing_locations=self.learn_inducing_locations,
            )
            self.variational_strategy = LMCVariationalStrategy(
                base_variational_strategy,
                num_tasks=self.num_tasks,
                num_latents=self.task_rank,
            )
        else:
            base_strategy.inducing_points.data = inducing_points

    def _create_structured_grid(
        self,
        x: torch.Tensor,
        n_inducing: int,
        spatial_inducing: Optional[int] = None,
        temporal_inducing: Optional[int] = None,
    ) -> tuple[torch.Tensor, tuple[int, int]]:
        unique_spatial, _ = torch.unique(x[:, :2], dim=0, return_inverse=True)
        unique_temporal, _ = torch.unique(x[:, 2], dim=0, return_inverse=True)

        n_s = len(unique_spatial)
        n_t = len(unique_temporal)

        if spatial_inducing is not None and temporal_inducing is not None:
            M_s = spatial_inducing
            M_t = temporal_inducing
        else:
            ratio = max(n_s / max(n_t, 1), 1e-6)
            M_s = int((n_inducing * ratio) ** 0.5)
            M_s = max(1, min(M_s, n_inducing))
            M_t = max(1, n_inducing // max(M_s, 1))
            while M_s * M_t < n_inducing:
                M_t += 1
            M_s = min(M_s, n_s)
            M_t = min(M_t, n_t)

        if M_s < n_s:
            spatial_indices = torch.linspace(0, n_s - 1, M_s).long()
            inducing_spatial = unique_spatial[spatial_indices]
        else:
            inducing_spatial = unique_spatial

        if M_t < n_t:
            temporal_indices = torch.linspace(0, n_t - 1, M_t).long()
            inducing_temporal = unique_temporal[temporal_indices]
        else:
            inducing_temporal = unique_temporal

        n_inducing_actual = M_s * M_t
        inducing_points = torch.zeros(
            n_inducing_actual,
            3 + self.n_covariates,
            dtype=x.dtype,
            device=x.device,
        )
        covariate_mean = None
        if self.n_covariates > 0:
            covariate_mean = x[:, 3:].mean(dim=0)

        idx = 0
        for s_idx in range(M_s):
            for t_idx in range(M_t):
                inducing_points[idx, :2] = inducing_spatial[s_idx]
                inducing_points[idx, 2] = inducing_temporal[t_idx]
                if covariate_mean is not None:
                    inducing_points[idx, 3:] = covariate_mean
                idx += 1

        return inducing_points, (M_s, M_t)

    def elbo(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        source_masks: torch.Tensor,
        n_data: int = None,
    ) -> torch.Tensor:
        variational_dist = self.variational_strategy(x)

        expected_log_lik = self.likelihood.expected_log_prob(
            y, variational_dist, source_masks
        )
        if n_data is not None:
            expected_log_lik = expected_log_lik * (n_data / x.shape[0])

        kl_divergence = self.variational_strategy.kl_divergence()
        return expected_log_lik - kl_divergence

    def get_hyperparameters(self) -> Dict[str, any]:
        """
        Get all model hyperparameters.
        """
        kernel_params = self.covar_module.get_hyperparameters()
        likelihood_params = self.likelihood.get_parameters()

        mean_params = {}
        if hasattr(self.mean_module, "constant"):
            mean_params["mean"] = self.mean_module.constant.detach().cpu().numpy()

        return {
            **mean_params,
            **kernel_params,
            **{k: v.cpu().numpy() for k, v in likelihood_params.items()},
            "n_inducing": self.n_inducing,
            "task_rank": self.task_rank,
        }
