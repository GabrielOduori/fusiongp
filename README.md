# FusionGP: Scalable Probabilistic Multi-Source NO₂ Fusion

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![GPyTorch](https://img.shields.io/badge/GPyTorch-1.11+-green.svg)](https://gpytorch.ai/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

**FusionGP** is a scalable probabilistic framework for fusing heterogeneous NO₂ observations into a unified spatio-temporal field using Sparse Variational Gaussian Processes (SVGPs). The framework integrates:

- **EPA Regulatory Monitors**: High-accuracy reference measurements (sparse spatial coverage)
- **Satellite Retrievals**: Broad spatial coverage with coarser resolution (TROPOMI)
- **LUR Prior Mean**: Land-use regression baseline as the GP mean function

### Key Features

- **Single Latent Field**: One GP models the true NO₂ concentration; all sources observe it through source-specific likelihoods
- **Heteroscedastic Noise**: Source-specific noise variances (σ²_EPA, σ²_SAT)
- **Scalable Inference**: SVGP with inducing points enables O(NM²) complexity instead of O(N³)
- **Uncertainty Quantification**: Full predictive distributions with calibrated uncertainty
- **LUR Prior Mean**: Optional land-use regression (LUR) prior provides a spatial baseline
- **Sequential Fusion**: GP-Kalman Filter (GPKF) for day-by-day updates from a LUR prior
- **Smoothing**: Kalman smoother for temporally consistent posterior maps

## End-to-End Pipeline (Demo)

The full demo pipeline in `experiments/run_demo_pipeline.py` includes:

1. Build merged daily dataset from raw sources
2. Load data + covariates, preprocess, and split
3. Build a LUR prior mean over the grid
4. Train FusionSVGP (batch variational GP)
5. Predict and evaluate (SVGP)
6. Run Kalman smoother for day-by-day tracking
7. Run GP-Kalman Filter (GPKF) for sequential fusion from the LUR prior
8. Save maps, metrics, and reports

Run it with:

```bash
python experiments/run_demo_pipeline.py
```

### Pipeline Diagram

```mermaid
flowchart TD
  A[Raw Sources] --> B[Merged Daily Dataset]
  B --> C[Covariates + Preprocess + Split]
  C --> D[LUR Prior Mean]
  C --> E[Train FusionSVGP]
  E --> F[SVGP Predictions + Eval]
  D --> G[GP-Kalman Filter (Sequential Fusion)]
  C --> G
  G --> H[GPKF Maps + Metrics]
  E --> I[Kalman Smoother]
  I --> J[Smoothed Maps]
  F --> K[Reports + Figures]
  H --> K
  J --> K
```

## Maps and Visualisations

After the pipeline has run, all figures can be regenerated at any time without re-running the full pipeline:

```bash
# Uses the most recent outputs/demo_run_* automatically
python experiments/generate_figures.py

# Or point to a specific run directory
python experiments/generate_figures.py outputs/demo_run_20240601_120000
```

This reads the CSVs saved by the pipeline and produces all figures inside `<run_dir>/publication_maps/`:

| Sub-directory | Contents |
|---|---|
| `gpkf/` | Per-day GPKF mean + uncertainty side-by-side maps |
| `svgp/` | Per-day SVGP fusion mean + uncertainty maps |
| `baselines/` | LUR and Atmo-Plan static baseline maps |
| `timeseries/` | Domain-averaged NO₂ over time; EPA observed vs SVGP predicted |
| `diagnostics/` | Training loss curves; SVGP calibration + residuals; GPKF calibration |

All spatial maps use Gaussian-smoothed heatmaps overlaid on a basemap with EPA station locations.

**Optional:** For higher-quality basemap tiles, add a Stadia Maps API key to a `.env` file at the project root:

```
STADIA_API_KEY=your_key_here
```

Without a key, it falls back to CartoDB Positron tiles.

## Mathematical Formulation

### Generative Model

Let `f(s,t)` denote the true latent NO₂ concentration at location `s=(x,y)` and time `t`.

Each source `q ∈ {EPA, SAT}` produces observations:

```
y_i^(q) = h_q(f(s_i, t_i)) + ε_i^(q),  ε_i^(q) ~ N(0, σ_q²)
```

Where the link functions are:
- **EPA**: `h_EPA(f) = f` (unbiased reference)
- **Satellite**: `h_SAT(f) = f` (unbiased but higher noise)

### GP Prior

```
f(s,t) ~ GP(m(s,t), k_space(s,s') · k_time(t,t'))
```

- `k_space`: Matérn-3/2 kernel with ARD over (x, y)
- `k_time`: Matérn-3/2 kernel over time
- `m(·)`: Constant mean function

### SVGP Inference

We maximize the Evidence Lower Bound (ELBO):

```
L = E_q(f)[log p(y|f)] - KL(q(u) || p(u))
```

Where:
- `Z ∈ R^(M×3)`: Inducing points in (x, y, t) space
- `u = f(Z)`: Inducing values
- `q(u) = N(m, S)`: Variational posterior

## Installation

```bash
# Clone the repository
git clone https://github.com/GabrielOduori/fusiongp.git
cd fusiongp

# Install dependencies
pip install -e .

# Or install dependencies directly
pip install torch gpytorch pandas numpy scipy scikit-learn matplotlib seaborn tqdm pyyaml
```

## Data Setup

Place your data files in the `data/` folder at the repository root:

```
fusiongp/
└── data/
    ├── satellite_retreavals.csv       # TROPOMI satellite NO₂ retrievals
    ├── epa_timeseries.csv             # EPA reference station time series
    ├── traffic_timeseries.csv         # Traffic volume time series
    ├── lur_predictions.csv            # Land-use regression predictions
    ├── grids_coordinates.csv          # Grid cell coordinates (lat/lon)
    ├── atmos_plan_model_no2.csv       # AtmoPlan model NO₂ (optional baseline)
    └── wind_sector_features_era5land_2023-06_daily.csv  # Wind covariates (optional)
```

The pipeline automatically discovers data from `data/` when run without arguments:

```bash
python experiments/run_demo_pipeline.py
```

To use a different data directory:

```bash
python experiments/run_demo_pipeline.py --data-dir /path/to/your/data
```

### Verifying your setup

The pipeline checks for all required files before training starts and will immediately report any that are missing. To validate your data without running any training (~2 minutes):

```bash
python experiments/run_demo_pipeline.py --data-only
```

This builds the merged dataset and exports UQ CSVs, confirming all data files are readable and correctly formatted.

## Quick Start

```python
from fusiongp.data import DataLoader, DataPreprocessor
from fusiongp.models import FusionSVGP
from fusiongp.training import Trainer
from fusiongp.inference import Predictor
from fusiongp.evaluation import Evaluator
from fusiongp.visualization import plot_predictions, plot_uncertainty

# 1. Load and preprocess data
loader = DataLoader("path/to/data.csv")
data = loader.load()

preprocessor = DataPreprocessor()
train_data, test_data, scalers = preprocessor.fit_transform(data)

# 2. Initialize model
model = FusionSVGP(
    n_inducing=500,
    kernel_type="matern32",
    learn_inducing_locations=True
)

# 3. Train
trainer = Trainer(model, learning_rate=0.01, n_epochs=500)
trainer.fit(train_data)

# 4. Predict
predictor = Predictor(model, scalers)
predictions = predictor.predict_grid(test_data)

# 5. Evaluate
evaluator = Evaluator()
metrics = evaluator.compute_all_metrics(predictions, test_data)
print(metrics)

# 6. Visualize
plot_predictions(predictions, save_path="results/predictions.png")
plot_uncertainty(predictions, save_path="results/uncertainty.png")
```

## Project Structure

```
fusiongp/
├── README.md                    # This file
├── pyproject.toml              # Project configuration and dependencies
├── src/
│   ├── __init__.py             # Package initialization
│   ├── data/
│   │   ├── __init__.py
│   │   ├── loader.py           # Data loading and validation
│   │   └── preprocessor.py     # Normalization, splitting, batching
│   ├── models/
│   │   ├── __init__.py
│   │   ├── kernels.py          # Spatio-temporal kernel definitions
│   │   ├── svgp.py             # Core SVGP model implementation
│   │   └── likelihoods.py      # Multi-source heteroscedastic likelihood
│   ├── training/
│   │   ├── __init__.py
│   │   ├── trainer.py          # Training loop with logging
│   │   └── callbacks.py        # Early stopping, checkpointing, scheduling
│   ├── inference/
│   │   ├── __init__.py
│   │   ├── predictor.py        # Gridded predictions with uncertainty
│   │   ├── kalman_smoother.py   # Temporal smoothing over SVGP outputs
│   │   └── gp_kalman_filter.py  # Sequential fusion from LUR prior
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── metrics.py          # Comprehensive evaluation metrics
│   │   └── cross_validation.py # Spatial and temporal cross-validation
│   └── visualization/
│       ├── __init__.py
│       ├── maps.py             # Spatial prediction maps
│       ├── uncertainty.py      # Uncertainty visualization
│       └── diagnostics.py      # Training curves, residuals, calibration
├── notebooks/
│   └── demo.ipynb              # End-to-end demonstration
├── tests/
│   ├── __init__.py
│   ├── test_data.py
│   ├── test_models.py
│   └── test_metrics.py
└── configs/
    └── default.yaml            # Default hyperparameters
```

## Evaluation Metrics

FusionGP computes comprehensive metrics for probabilistic model evaluation:

### Point Prediction Metrics
- **RMSE**: Root Mean Squared Error
- **MAE**: Mean Absolute Error
- **R²**: Coefficient of Determination
- **Bias**: Mean prediction error

### Probabilistic/Gaussian Metrics
- **NLL**: Negative Log-Likelihood
- **CRPS**: Continuous Ranked Probability Score
- **Calibration**: Coverage at various confidence levels
- **Sharpness**: Average predictive standard deviation
- **DSS**: Dawid-Sebastiani Score
- **Energy Score**: Multivariate probabilistic metric

### Per-Source Metrics
All metrics computed separately for EPA and Satellite observations.

## Configuration

See `configs/default.yaml` for all configurable parameters:

```yaml
model:
  n_inducing: 500
  kernel:
    spatial: matern32
    temporal: matern32
    learn_lengthscales: true
  likelihood:
    learn_noise: true
    learn_calibration: true

training:
  n_epochs: 500
  batch_size: 1024
  learning_rate: 0.01
  optimizer: adam
  scheduler:
    type: reduce_on_plateau
    patience: 20
    factor: 0.5

inference:
  n_samples: 100
  batch_size: 2048
```

## References

1. Hensman, J., Fusi, N., & Lawrence, N. D. (2013). Gaussian processes for big data. UAI.
2. Titsias, M. (2009). Variational learning of inducing variables in sparse Gaussian processes. AISTATS.
3. Hamelijnck, O., et al. (2021). Spatio-temporal variational Gaussian processes. NeurIPS.
4. Williams, C. K., & Rasmussen, C. E. (2006). Gaussian processes for machine learning. MIT Press.

## License

MIT License - see LICENSE file for details.

## Citation

```bibtex
@software{fusiongp2024,
  title={FusionGP: Scalable Probabilistic Multi-Source NO₂ Fusion},
  author={Your Name},
  year={2024},
  url={https://github.com/GabrielOduori/fusiongp}
}
```
