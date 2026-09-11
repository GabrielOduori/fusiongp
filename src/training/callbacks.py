"""
Training callbacks for FusionGP.

This module provides callback classes for controlling the training loop:
- EarlyStopping: Stop training when validation metric stops improving
- ModelCheckpoint: Save model checkpoints during training
- LRScheduler: Adjust learning rate based on training progress

References
----------
.. [1] Prechelt, L. (1998). Early stopping-but when? Neural Networks.

Example
-------
>>> early_stopping = EarlyStopping(patience=50, min_delta=1e-4)
>>> checkpoint = ModelCheckpoint(save_dir='checkpoints', save_best_only=True)
>>> 
>>> trainer = Trainer(model, callbacks=[early_stopping, checkpoint])
>>> trainer.fit(train_data, val_data)
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Optional

import torch

logger = logging.getLogger(__name__)


class Callback(ABC):
    """
    Abstract base class for training callbacks.
    
    Callbacks are called at various points during training to perform
    actions like early stopping, checkpointing, or learning rate scheduling.
    """
    
    @abstractmethod
    def on_epoch_end(
        self,
        epoch: int,
        logs: Dict[str, float],
        model: Any,
    ) -> bool:
        """
        Called at the end of each epoch.
        
        Parameters
        ----------
        epoch : int
            Current epoch number.
        logs : Dict[str, float]
            Dictionary of metrics for this epoch.
        model : Any
            The model being trained.
            
        Returns
        -------
        bool
            True to continue training, False to stop.
        """
        raise NotImplementedError
    
    def on_train_begin(self, model: Any) -> None:
        """Called at the start of training."""
        return None
    
    def on_train_end(self, model: Any, logs: Dict[str, float]) -> None:
        """Called at the end of training."""
        return None


class EarlyStopping(Callback):
    """
    Stop training when a monitored metric has stopped improving.
    
    Training is stopped when the monitored metric does not improve by
    at least `min_delta` for `patience` consecutive epochs.
    
    Parameters
    ----------
    monitor : str, default='val_loss'
        Metric to monitor.
    patience : int, default=50
        Number of epochs with no improvement before stopping.
    min_delta : float, default=1e-4
        Minimum change to qualify as an improvement.
    mode : str, default='min'
        'min' for metrics to minimize (loss), 'max' for metrics to maximize.
    restore_best_weights : bool, default=True
        Whether to restore model weights from best epoch.
        
    Attributes
    ----------
    best_value : float
        Best metric value observed.
    best_epoch : int
        Epoch with best metric value.
    wait : int
        Number of epochs since last improvement.
    stopped_epoch : int
        Epoch at which training was stopped (0 if not stopped).
    best_weights : Dict
        Model state dict from best epoch.
        
    Example
    -------
    >>> early_stopping = EarlyStopping(
    ...     monitor='val_loss',
    ...     patience=50,
    ...     min_delta=1e-4,
    ...     restore_best_weights=True
    ... )
    """
    
    def __init__(
        self,
        monitor: str = 'val_loss',
        patience: int = 50,
        min_delta: float = 1e-4,
        mode: str = 'min',
        restore_best_weights: bool = True,
    ):
        """
        Initialize EarlyStopping callback.
        
        Parameters
        ----------
        monitor : str
            Metric name to monitor.
        patience : int
            Epochs to wait before stopping.
        min_delta : float
            Minimum improvement threshold.
        mode : str
            'min' or 'max'.
        restore_best_weights : bool
            Restore best weights at end.
        """
        self.monitor = monitor
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.restore_best_weights = restore_best_weights
        
        # State
        self.best_value = float('inf') if mode == 'min' else float('-inf')
        self.best_epoch = 0
        self.wait = 0
        self.stopped_epoch = 0
        self.best_weights = None
        
        # Comparison function
        if mode == 'min':
            self._is_improvement = lambda new, best: new < (best - min_delta)
        else:
            self._is_improvement = lambda new, best: new > (best + min_delta)
        
        logger.info(
            f"EarlyStopping: monitor={monitor}, patience={patience}, "
            f"min_delta={min_delta}, mode={mode}"
        )
    
    def on_train_begin(self, model: Any) -> None:
        """Reset state at start of training."""
        self.best_value = float('inf') if self.mode == 'min' else float('-inf')
        self.best_epoch = 0
        self.wait = 0
        self.stopped_epoch = 0
        self.best_weights = None
    
    def on_epoch_end(
        self,
        epoch: int,
        logs: Dict[str, float],
        model: Any,
    ) -> bool:
        """
        Check for improvement and potentially stop training.
        
        Parameters
        ----------
        epoch : int
            Current epoch.
        logs : Dict[str, float]
            Metrics dictionary.
        model : Any
            Model being trained.
            
        Returns
        -------
        bool
            True to continue, False to stop.
        """
        current = logs.get(self.monitor)
        
        if current is None:
            logger.warning(
                f"EarlyStopping: metric '{self.monitor}' not found in logs. "
                f"Available: {list(logs.keys())}"
            )
            return True
        
        if self._is_improvement(current, self.best_value):
            self.best_value = current
            self.best_epoch = epoch
            self.wait = 0
            
            if self.restore_best_weights:
                self.best_weights = {
                    k: v.cpu().clone() 
                    for k, v in model.state_dict().items()
                }
            
            logger.debug(
                f"EarlyStopping: improvement at epoch {epoch}, "
                f"{self.monitor}={current:.6f}"
            )
        else:
            self.wait += 1
            
            if self.wait >= self.patience:
                self.stopped_epoch = epoch
                logger.info(
                    f"EarlyStopping: stopping at epoch {epoch}. "
                    f"Best {self.monitor}={self.best_value:.6f} at epoch {self.best_epoch}"
                )
                return False
        
        return True
    
    def on_train_end(self, model: Any, logs: Dict[str, float]) -> None:
        """Restore best weights if enabled."""
        if self.restore_best_weights and self.best_weights is not None:
            model.load_state_dict(self.best_weights)
            logger.info(
                f"EarlyStopping: restored best weights from epoch {self.best_epoch}"
            )


class ModelCheckpoint(Callback):
    """
    Save model checkpoints during training.
    
    Can save the best model only (based on a monitored metric) or
    save checkpoints at regular intervals.
    
    Parameters
    ----------
    save_dir : str or Path, default='checkpoints'
        Directory to save checkpoints.
    monitor : str, default='val_loss'
        Metric to monitor for best model.
    mode : str, default='min'
        'min' or 'max' for the monitored metric.
    save_best_only : bool, default=True
        Only save when metric improves.
    save_freq : int, default=1
        Save every N epochs (if save_best_only=False).
    filename_template : str, default='model_epoch{epoch:04d}.pt'
        Template for checkpoint filenames.
        
    Attributes
    ----------
    best_value : float
        Best metric value observed.
    best_path : Path
        Path to best model checkpoint.
        
    Example
    -------
    >>> checkpoint = ModelCheckpoint(
    ...     save_dir='checkpoints',
    ...     monitor='val_loss',
    ...     save_best_only=True
    ... )
    """
    
    def __init__(
        self,
        save_dir: str | Path = 'checkpoints',
        monitor: str = 'val_loss',
        mode: str = 'min',
        save_best_only: bool = True,
        save_freq: int = 1,
        filename_template: str = 'model_epoch{epoch:04d}.pt',
    ):
        """
        Initialize ModelCheckpoint callback.
        
        Parameters
        ----------
        save_dir : str or Path
            Checkpoint directory.
        monitor : str
            Metric to monitor.
        mode : str
            'min' or 'max'.
        save_best_only : bool
            Only save best model.
        save_freq : int
            Save frequency in epochs.
        filename_template : str
            Filename template.
        """
        self.save_dir = Path(save_dir)
        self.monitor = monitor
        self.mode = mode
        self.save_best_only = save_best_only
        self.save_freq = save_freq
        self.filename_template = filename_template
        
        # State
        self.best_value = float('inf') if mode == 'min' else float('-inf')
        self.best_path: Optional[Path] = None
        
        # Comparison function
        if mode == 'min':
            self._is_improvement = lambda new, best: new < best
        else:
            self._is_improvement = lambda new, best: new > best
        
        logger.info(
            f"ModelCheckpoint: save_dir={save_dir}, monitor={monitor}, "
            f"save_best_only={save_best_only}"
        )
    
    def on_train_begin(self, model: Any) -> None:
        """Create save directory."""
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.best_value = float('inf') if self.mode == 'min' else float('-inf')
        self.best_path = None
    
    def on_epoch_end(
        self,
        epoch: int,
        logs: Dict[str, float],
        model: Any,
    ) -> bool:
        """
        Save checkpoint if conditions are met.
        
        Parameters
        ----------
        epoch : int
            Current epoch.
        logs : Dict[str, float]
            Metrics dictionary.
        model : Any
            Model to save.
            
        Returns
        -------
        bool
            Always True (continue training).
        """
        if self.save_best_only:
            current = logs.get(self.monitor)
            
            if current is None:
                return True
            
            if self._is_improvement(current, self.best_value):
                self.best_value = current
                
                # Remove old best
                if self.best_path is not None and self.best_path.exists():
                    self.best_path.unlink()
                
                # Save new best
                self.best_path = self.save_dir / f'best_model.pt'
                self._save_checkpoint(model, epoch, logs, self.best_path)
                
                logger.info(
                    f"ModelCheckpoint: saved best model at epoch {epoch}, "
                    f"{self.monitor}={current:.6f}"
                )
        else:
            if (epoch + 1) % self.save_freq == 0:
                path = self.save_dir / self.filename_template.format(epoch=epoch)
                self._save_checkpoint(model, epoch, logs, path)
        
        return True
    
    def _save_checkpoint(
        self,
        model: Any,
        epoch: int,
        logs: Dict[str, float],
        path: Path,
    ) -> None:
        """
        Save a checkpoint to disk.
        
        Parameters
        ----------
        model : Any
            Model to save.
        epoch : int
            Current epoch.
        logs : Dict[str, float]
            Metrics.
        path : Path
            Save path.
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'metrics': logs,
        }
        
        # Include likelihood if present
        if hasattr(model, 'likelihood'):
            checkpoint['likelihood_state_dict'] = model.likelihood.state_dict()
        
        torch.save(checkpoint, path)


class LRScheduler(Callback):
    """
    Learning rate scheduler callback.
    
    Wraps PyTorch learning rate schedulers for use with the Trainer.
    
    Parameters
    ----------
    scheduler : torch.optim.lr_scheduler._LRScheduler
        PyTorch LR scheduler instance.
    monitor : str, default='val_loss'
        Metric to monitor (for ReduceLROnPlateau).
        
    Example
    -------
    >>> optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    >>> scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    ...     optimizer, mode='min', patience=20, factor=0.5
    ... )
    >>> lr_callback = LRScheduler(scheduler, monitor='val_loss')
    """
    
    def __init__(
        self,
        scheduler: torch.optim.lr_scheduler.LRScheduler,
        monitor: str = 'val_loss',
    ):
        """
        Initialize LRScheduler callback.
        
        Parameters
        ----------
        scheduler : _LRScheduler
            PyTorch scheduler.
        monitor : str
            Metric for ReduceLROnPlateau.
        """
        self.scheduler = scheduler
        self.monitor = monitor
        self._is_reduce_on_plateau = isinstance(
            scheduler, 
            torch.optim.lr_scheduler.ReduceLROnPlateau
        )
    
    def on_epoch_end(
        self,
        epoch: int,
        logs: Dict[str, float],
        model: Any,
    ) -> bool:
        """
        Step the scheduler.
        
        Parameters
        ----------
        epoch : int
            Current epoch.
        logs : Dict[str, float]
            Metrics.
        model : Any
            Model (unused).
            
        Returns
        -------
        bool
            Always True.
        """
        if self._is_reduce_on_plateau:
            value = logs.get(self.monitor)
            if value is not None:
                self.scheduler.step(value)
        else:
            self.scheduler.step()
        
        # Log current LR
        current_lr = self.scheduler.optimizer.param_groups[0]['lr']
        logger.debug(f"LRScheduler: lr={current_lr:.2e}")
        
        return True
