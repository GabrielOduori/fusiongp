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

from fusiongp.training.trainer import Trainer
from fusiongp.training.callbacks import EarlyStopping, ModelCheckpoint, LRScheduler

__all__ = [
    "Trainer",
    "EarlyStopping",
    "ModelCheckpoint",
    "LRScheduler",
]
