"""
Reproduce FusionGP paper results.

Multi-source NO2 fusion (EPA + satellite + LUR/ATMO-Plan prior) with SVGP
training, GP-Kalman Filter (GPKF) sequential fusion, and Kalman-smoothed
day-by-day maps, evaluated against held-out EPA ground truth.

This is the canonical entry point — it runs the same pipeline implementation
as experiments/run_demo_pipeline.py (kept only as a backward-compatible
alias for existing subprocess callers); both call
experiments/paper_pipeline/full_pipeline.py.

Usage:
    # Standard run (M=300, 200 epochs):
    python experiments/reproduce_paper.py

    # Run pipeline then M-sweep for RQ2:
    python experiments/reproduce_paper.py --sweep

    # Custom M and epochs:
    python experiments/reproduce_paper.py --n-inducing 400 --epochs 150

    # Leave-one-station-out cross-validation (see reproduce_paper_results.py
    # and run_loso_and_sweep_overnight.sh for the full 9-station sweep):
    EPA_HOLDOUT_MODE=station EPA_HOLDOUT_GRID_IDS=<grid_id> \\
        PRIOR_MEAN_SOURCE=atmo python experiments/reproduce_paper.py

Outputs saved to outputs/demo_run_<tag>/, including per-day gpkf_maps/ and
kalman_maps/, baseline comparison tables, and a final_results_report.md.
See experiments/paper_pipeline/full_pipeline.py's module docstring and
--help for the full env-var and CLI surface.
"""

import sys
from pathlib import Path

# Add repo root (for `src`) and this directory (for the `paper_pipeline` package) to path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from paper_pipeline import full_pipeline

if __name__ == "__main__":
    full_pipeline.main()
