"""
Reproduce FusionGP paper results.

Multi-source NO2 fusion with evaluation against EPA ground truth.

Methodology:
- When USE_EPA_IN_TRAINING=False: Train on Satellite (and prior), evaluate on held-out EPA
- When USE_EPA_IN_TRAINING=True: Legacy mode with all sources (has data leakage)

Pre-calibration uses colocated EPA-LCS observations to learn a linear correction
before training, avoiding data leakage while bringing sources to EPA scale.

Usage:
    python reproduce_paper.py

Outputs saved to results/experiment_YYYYMMDD_HHMMSS/ with subdirectories:
    figures/ - Training curves, predictions, uncertainty maps
    models/  - Trained model checkpoint
    tables/  - Metrics CSVs and experiment summary

Implementation lives in experiments/paper_pipeline/ (config, data, evaluation,
plotting, reporting, inference, pipeline) — this file is just the entry point.
"""

import sys
from pathlib import Path

# Add repo root (for `src`) and this directory (for the `paper_pipeline` package) to path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from paper_pipeline import pipeline

if __name__ == "__main__":
    pipeline.run()
