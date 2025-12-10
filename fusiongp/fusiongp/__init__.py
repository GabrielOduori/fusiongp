"""
FusionGP: Scalable Probabilistic Multi-Source NO₂ Fusion
=========================================================

A framework for fusing heterogeneous air quality observations using 
Sparse Variational Gaussian Processes (SVGPs).

Main Components
---------------
- data: Data loading, preprocessing, and batching utilities
- models: SVGP model, kernels, and likelihood definitions
- training: Training loop, callbacks, and optimization
- inference: Prediction and uncertainty quantification
- evaluation: Comprehensive metrics for probabilistic models
- visualization: Plotting utilities for maps and diagnostics

Example
-------
>>> from fusiongp import FusionSVGP, DataLoader, Trainer
>>> 
>>> # Load data
>>> loader = DataLoader("data.csv")
>>> data = loader.load()
>>> 
>>> # Create and train model
>>> model = FusionSVGP(n_inducing=500)
>>> trainer = Trainer(model)
>>> trainer.fit(data)
>>> 
>>> # Make predictions
>>> predictions = model.predict(test_locations)

References
----------
.. [1] Hensman, J., Fusi, N., & Lawrence, N. D. (2013). 
       Gaussian processes for big data. UAI.
.. [2] Titsias, M. (2009). Variational learning of inducing variables 
       in sparse Gaussian processes. AISTATS.
.. [3] Hamelijnck, O., et al. (2021). Spatio-temporal variational 
       Gaussian processes. NeurIPS.

License
-------
MIT License
"""

__version__ = "0.1.0"
__author__ = "Your Name"
__email__ = "your.email@example.com"

# Public API imports
from fusiongp.data import DataLoader, DataPreprocessor
from fusiongp.models import FusionSVGP, SpatioTemporalKernel, MultiSourceLikelihood
from fusiongp.training import Trainer, EarlyStopping, ModelCheckpoint
from fusiongp.inference import Predictor
from fusiongp.evaluation import Evaluator, compute_metrics
from fusiongp.visualization import (
    plot_predictions,
    plot_uncertainty,
    plot_training_history,
    plot_calibration,
)

__all__ = [
    # Version info
    "__version__",
    "__author__",
    # Data
    "DataLoader",
    "DataPreprocessor",
    # Models
    "FusionSVGP",
    "SpatioTemporalKernel",
    "MultiSourceLikelihood",
    # Training
    "Trainer",
    "EarlyStopping",
    "ModelCheckpoint",
    # Inference
    "Predictor",
    # Evaluation
    "Evaluator",
    "compute_metrics",
    # Visualization
    "plot_predictions",
    "plot_uncertainty",
    "plot_training_history",
    "plot_calibration",
]
