"""
Deprecated alias for experiments/reproduce_paper.py.

The full pipeline implementation moved to
experiments/paper_pipeline/full_pipeline.py so that reproduce_paper.py runs
the same GPKF/Kalman-smoother/LOSO-capable code path. This file is kept only
so existing subprocess callers (experiments/reproduce_paper_results.py,
experiments/run_loso_and_sweep_overnight.sh) keep working unchanged by path.
Prefer invoking experiments/reproduce_paper.py directly for new work.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from paper_pipeline import full_pipeline

if __name__ == "__main__":
    full_pipeline.main()
