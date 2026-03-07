# FusionGP Tests

This directory contains test scripts for validating and analyzing the FusionGP implementation.

## Grid Structure Analysis

### Quick Start

```bash
# From project root
python tests/test_grid_structure.py
```

This will:
1. Load your data
2. Analyze the spatial/temporal grid structure
3. Benchmark kernel evaluation strategies
4. Test inducing point initialization methods
5. Generate diagnostic plots in `outputs/`

### Detailed Usage

```bash
# Use custom data file
python tests/test_grid_structure.py --data-path /path/to/your/data.csv

# Run only specific tests
python tests/test_grid_structure.py --no-benchmark --visualize

# Adjust benchmark trials
python tests/test_grid_structure.py --n-trials 20
```

### Options

- `--data-path`: Path to CSV data file (default: `data/realistic_synthetic_no2.csv`)
- `--benchmark`: Run kernel evaluation benchmark (default: True)
- `--visualize`: Generate visualization plots (default: True)
- `--test-inducing`: Test inducing point strategies (default: True)
- `--n-trials`: Number of benchmark trials (default: 10)

### Expected Output

The test script will print:

```
======================================================================
GRID STRUCTURE ANALYSIS
======================================================================

Spatial Structure:
  Unique spatial locations (centroids): 100
  Average observations per centroid: 12.0
  Std dev of repeats: 2.34
  Min/Max repeats: 5/18

Temporal Structure:
  Unique temporal points: 120
  Average observations per time: 10.0
  Std dev of repeats: 1.45
  Min/Max repeats: 6/15

Grid Properties:
  Potential grid size: 100 × 120 = 12,000
  Actual observations: 7,200
  Grid coverage: 60.0%
  Is gridded: ✓ YES

Kronecker Structure:
  ✓ Can exploit Kronecker structure!
  ✓ K_space size: (100, 100)
  ✓ K_time size: (120, 120)
  ✓ Storage reduction: 480.0×
======================================================================
```

### Generated Files

All outputs are saved to `outputs/`:

1. **grid_structure_analysis.png**: Six-panel visualization showing:
   - Spatial distribution of centroids
   - Temporal distribution
   - Coverage heatmap (which grid cells are observed)
   - Spatial repeats histogram
   - Temporal repeats histogram
   - Summary statistics

2. **inducing_points_comparison.png**: Comparison of:
   - K-means on all data (scattered)
   - Structured grid approach (spatial × temporal)

### Interpreting Results

#### If "Is gridded: ✓ YES"

Your data is on a regular grid! You can:
- ✅ Use Kronecker-structured inducing points
- ✅ Exploit efficient kernel evaluation
- ✅ Get massive speedups (100-1000×)

**Next steps**: Implement the optimized kernel evaluation in `src/models/kernels.py`

#### If "Is gridded: ✗ NO"

Your data is scattered (irregular). You should:
- ✅ Keep current element-wise kernel evaluation
- ✅ Use K-means inducing points
- ❌ Cannot fully exploit Kronecker structure

**Note**: Partial benefits still possible if spatial locations are mostly regular.

### Benchmark Results

The benchmark compares:

1. **Element-wise** (current implementation):
   - Computes full (N×N) spatial and temporal covariance matrices
   - Element-wise multiplication

2. **Gridded** (proposed optimization):
   - Computes only (N_s×N_s) and (N_t×N_t) matrices
   - Indexes to reconstruct full covariance

Expected speedup: **2-10×** depending on data structure

### Troubleshooting

**Error: "No such file or directory"**
```bash
# Generate synthetic data first
python experiments/generate_realistic_data.py

# Or specify your data path
python tests/test_grid_structure.py --data-path /your/path/data.csv
```

**Error: "Module not found"**
```bash
# Make sure you're in the project root and dependencies are installed
cd /path/to/FusionGP
pip install -e .
python tests/test_grid_structure.py
```

## Other Tests

### Unit Tests
```bash
# Run all unit tests
pytest tests/

# Run specific test file
pytest tests/test_kernels.py

# Run with coverage
pytest tests/ --cov=src --cov-report=html
```

### Integration Tests
```bash
# Full end-to-end test
python tests/test_end_to_end.py
```

## Contributing

When adding new tests:
1. Place unit tests in `tests/test_*.py`
2. Place integration/system tests in `tests/integration/`
3. Place benchmarks in `tests/benchmarks/`
4. Update this README with usage instructions
