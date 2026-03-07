"""
Model components for FusionGP.

This module provides the core model components:
- FusionSVGP: The main Sparse Variational GP model
- SpatioTemporalKernel: Separable spatio-temporal covariance function
- MultiSourceLikelihood: Heteroscedastic likelihood with source-specific parameters

Classes
-------
FusionSVGP
    Main SVGP model for multi-source fusion.
SpatioTemporalKernel
    Separable product kernel for space-time covariance.
MultiSourceLikelihood
    Heteroscedastic Gaussian likelihood with calibration.

References
----------
.. [1] Hensman, J., Fusi, N., & Lawrence, N. D. (2013). 
       Gaussian processes for big data. UAI.
.. [2] Titsias, M. (2009). Variational learning of inducing variables 
       in sparse Gaussian processes. AISTATS.
"""

from src.models.kernels import SpatioTemporalKernel
from src.models.likelihoods import MultiSourceLikelihood, MaskedMultitaskGaussianLikelihood
from src.models.svgp import FusionSVGP
from src.models.st_svgp import STSVGPModel, STSVGPConfig
from src.models.multitask_svgp import MultiTaskSVGP
from src.models.prior_mean import GridPriorMean, RasterMean, ATMOPlanMean

__all__ = [
    "FusionSVGP",
    "MultiTaskSVGP",
    "SpatioTemporalKernel",
    "MultiSourceLikelihood",
    "MaskedMultitaskGaussianLikelihood",
    "STSVGPModel",
    "STSVGPConfig",
    "GridPriorMean",
    "RasterMean",
    "ATMOPlanMean",
]
