"""
Training utilities for FusionGP.

This module provides training infrastructure including:
- Trainer: Main training loop with logging and validation
- EarlyStopping: Callback for early stopping
- ModelCheckpoint: Callback for saving best models
- LRScheduler: Learning rate scheduling

Classes
-------
Trainer
    Main training class with full training loop.
EarlyStopping
    Callback to stop training when validation loss stops improving.
ModelCheckpoint
    Callback to save model checkpoints.
"""

from src.training.trainer import Trainer
from src.training.st_svgp_trainer import STSVGPTrainer
from src.training.callbacks import EarlyStopping, ModelCheckpoint, LRScheduler

__all__ = [
    "Trainer",
    "STSVGPTrainer",
    "EarlyStopping",
    "ModelCheckpoint",
    "LRScheduler",
]
