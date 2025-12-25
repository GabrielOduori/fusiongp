# FusionGP Experiments

This directory contains experimental scripts and benchmarks for the FusionGP framework.

## Structure

```
experiments/
├── README.md              # This file
├── basic_usage.py         # Basic workflow demonstration
├── reproduce_paper.py     # Paper reproduction experiment
├── results/               # Experiment results (auto-generated)
├── checkpoints/           # Model checkpoints (auto-generated)
├── outputs/              # Visualization outputs (auto-generated)
└── .gitignore            # Ignore generated files
```

## Running Experiments

### 1. Paper Reproduction

Reproduces the main experimental results from the FusionGP paper with the full configuration (800 inducing points, fixed hyperparameters, IDW baseline comparison):

```bash
cd experiments
python reproduce_paper.py
```

This will:
1. Load and preprocess synthetic NO₂ data
2. Train FusionSVGP model with 800 inducing points (high capacity)
3. Compare against IDW baseline
4. Evaluate on all sources (EPA, low-cost, satellite)
5. Generate comprehensive visualizations

**Outputs:** (timestamped in `results/experiment_YYYYMMDD_HHMMSS/`)
- `experiment_summary.txt` - Configuration and metrics
- `predictions.png` - Mean predictions
- `uncertainty.png` - Uncertainty estimates
- `uncertainty_ci.png` - Confidence intervals
- `predictions_surface.png` - Gridded mean surface
- `uncertainty_surface.png` - Gridded uncertainty surface
- `spatial_maps/` - Time-series spatial maps with contours

**Expected runtime:** ~5-10 minutes (300 epochs, CPU)

### 2. Basic Usage Example

Quick demonstration of the FusionGP workflow with reduced model capacity for faster execution:

```bash
cd experiments
python basic_usage.py
```

This will:
1. Load synthetic NO₂ data from multiple sources (EPA, low-cost sensors, satellite)
2. Preprocess and split data (60% train, 20% val, 20% test)
3. Initialize and train a FusionSVGP model with 300 inducing points (smaller for speed)
4. Make predictions on test set
5. Evaluate performance (RMSE, MAE, R², CRPS)
6. Generate visualization plots

**Outputs:**
- `checkpoints/best_model.pt` - Trained model checkpoint
- `outputs/basic_usage_results.png` - Training history and predictions
- `outputs/uncertainty_calibration.png` - Uncertainty quantification analysis

**Expected runtime:** ~2-5 minutes (100 epochs, CPU)

## Experiment Ideas

Future experiments to add:

### Performance Benchmarks
- `benchmark_inducing_points.py` - Compare different numbers of inducing points (100, 300, 500, 1000)
- `benchmark_kernels.py` - Compare kernel types (Matérn-1/2, 3/2, 5/2, RBF)
- `benchmark_scalability.py` - Test scaling with dataset size

### Optimization Studies
- `kronecker_grid_prediction.py` - Implement and benchmark Kronecker-aware grid prediction
- `gpu_acceleration.py` - Compare CPU vs GPU training/inference
- `inducing_point_strategies.py` - Compare K-means vs grid vs random initialization

### Ablation Studies
- `ablation_calibration.py` - Impact of learning calibration parameters
- `ablation_noise.py` - Fixed vs learned noise variances
- `ablation_sources.py` - Single source vs multi-source fusion

### Cross-Validation
- `spatial_cv.py` - Spatial block cross-validation
- `temporal_cv.py` - Temporal holdout validation

## Configuration

Most experiments can be configured via command-line arguments:

```bash
python basic_usage.py --n-inducing 500 --epochs 200 --lr 0.01
```

See individual scripts for available options.

## Results Tracking

For systematic experiment tracking, consider using:
- **MLflow** for experiment management
- **TensorBoard** for training visualization
- **Weights & Biases** for collaborative tracking

## Notes

- All experiments use the synthetic NO₂ data from `fusiongp/notebooks/synthetic_no2_data.csv`
- Model checkpoints and outputs are gitignored by default
- For reproducibility, set random seeds in each script
