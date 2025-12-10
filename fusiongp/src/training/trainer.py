"""
Training loop for FusionGP.

This module provides the Trainer class which handles the complete training
workflow including optimization, validation, logging, and callbacks.

The training procedure follows the SVGP framework:
1. Sample mini-batch of observations
2. Compute ELBO on mini-batch
3. Backpropagate and update parameters
4. Periodically evaluate on validation set

References
----------
.. [1] Hensman, J., Fusi, N., & Lawrence, N. D. (2013). 
       Gaussian processes for big data. UAI.

Example
-------
>>> trainer = Trainer(
...     model,
...     learning_rate=0.01,
...     n_epochs=500,
...     batch_size=1024
... )
>>> history = trainer.fit(train_data, val_data)
>>> print(history['val_loss'][-1])
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.loader import FusionData, FusionDataset
from src.models.svgp import FusionSVGP
from src.training.callbacks import Callback, EarlyStopping, ModelCheckpoint, LRScheduler

logger = logging.getLogger(__name__)


@dataclass
class TrainingHistory:
    """
    Container for training history.
    
    Stores metrics from each epoch for analysis and visualization.
    
    Attributes
    ----------
    train_loss : List[float]
        Training loss per epoch.
    val_loss : List[float]
        Validation loss per epoch.
    learning_rates : List[float]
        Learning rate per epoch.
    epoch_times : List[float]
        Time per epoch in seconds.
    metrics : Dict[str, List[float]]
        Additional metrics per epoch.
    """
    train_loss: List[float] = field(default_factory=list)
    val_loss: List[float] = field(default_factory=list)
    learning_rates: List[float] = field(default_factory=list)
    epoch_times: List[float] = field(default_factory=list)
    metrics: Dict[str, List[float]] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, List[float]]:
        """Convert to dictionary."""
        result = {
            'train_loss': self.train_loss,
            'val_loss': self.val_loss,
            'learning_rate': self.learning_rates,
            'epoch_time': self.epoch_times,
        }
        result.update(self.metrics)
        return result
    
    @property
    def best_val_loss(self) -> float:
        """Get best validation loss."""
        return min(self.val_loss) if self.val_loss else float('inf')
    
    @property
    def best_epoch(self) -> int:
        """Get epoch with best validation loss."""
        if not self.val_loss:
            return 0
        return int(np.argmin(self.val_loss))


class Trainer:
    """
    Trainer for FusionSVGP models.
    
    Handles the complete training workflow including:
    - Mini-batch stochastic optimization
    - ELBO maximization
    - Validation monitoring
    - Callbacks (early stopping, checkpointing, LR scheduling)
    - Progress logging
    
    Parameters
    ----------
    model : FusionSVGP
        The model to train.
    learning_rate : float, default=0.01
        Initial learning rate.
    n_epochs : int, default=500
        Maximum number of training epochs.
    batch_size : int, default=1024
        Mini-batch size for training.
    optimizer_type : str, default='adam'
        Optimizer type: 'adam', 'adamw', or 'sgd'.
    weight_decay : float, default=0.0
        L2 regularization weight.
    gradient_clip : float, optional
        Max gradient norm for clipping.
    val_interval : int, default=5
        Validate every N epochs.
    callbacks : List[Callback], optional
        List of callbacks.
    device : str, default='cpu'
        Device to train on.
    seed : int, default=42
        Random seed.
        
    Attributes
    ----------
    model : FusionSVGP
        The model.
    optimizer : torch.optim.Optimizer
        The optimizer.
    history : TrainingHistory
        Training history.
        
    Example
    -------
    >>> model = FusionSVGP(n_inducing=500)
    >>> trainer = Trainer(
    ...     model,
    ...     learning_rate=0.01,
    ...     n_epochs=500,
    ...     callbacks=[
    ...         EarlyStopping(patience=50),
    ...         ModelCheckpoint(save_dir='checkpoints')
    ...     ]
    ... )
    >>> 
    >>> history = trainer.fit(train_data, val_data)
    """
    
    def __init__(
        self,
        model: FusionSVGP,
        learning_rate: float = 0.01,
        n_epochs: int = 500,
        batch_size: int = 1024,
        optimizer_type: str = 'adam',
        weight_decay: float = 0.0,
        gradient_clip: Optional[float] = 1.0,
        val_interval: int = 5,
        callbacks: Optional[List[Callback]] = None,
        device: str = 'cpu',
        seed: int = 42,
    ):
        """
        Initialize the Trainer.
        
        Parameters
        ----------
        model : FusionSVGP
            Model to train.
        learning_rate : float
            Initial learning rate.
        n_epochs : int
            Maximum epochs.
        batch_size : int
            Batch size.
        optimizer_type : str
            Optimizer type.
        weight_decay : float
            L2 regularization.
        gradient_clip : float
            Gradient clipping norm.
        val_interval : int
            Validation frequency.
        callbacks : List[Callback]
            Training callbacks.
        device : str
            Device ('cpu' or 'cuda').
        seed : int
            Random seed.
        """
        self.model = model
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.gradient_clip = gradient_clip
        self.val_interval = val_interval
        self.device = torch.device(device)
        self.seed = seed
        
        # Move model to device
        self.model = self.model.to(self.device)
        self.model.likelihood = self.model.likelihood.to(self.device)
        
        # Create optimizer
        self.optimizer = self._create_optimizer(optimizer_type, weight_decay)
        
        # Set up callbacks
        self.callbacks = callbacks or []
        
        # History
        self.history = TrainingHistory()
        
        # Set random seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        logger.info(
            f"Trainer initialized: lr={learning_rate}, epochs={n_epochs}, "
            f"batch_size={batch_size}, device={device}"
        )
    
    def _create_optimizer(
        self,
        optimizer_type: str,
        weight_decay: float,
    ) -> torch.optim.Optimizer:
        """
        Create the optimizer.
        
        Parameters
        ----------
        optimizer_type : str
            Type of optimizer.
        weight_decay : float
            L2 regularization weight.
            
        Returns
        -------
        torch.optim.Optimizer
            Configured optimizer.
        """
        # Collect all parameters
        params = list(self.model.parameters()) + list(self.model.likelihood.parameters())
        
        if optimizer_type.lower() == 'adam':
            return torch.optim.Adam(
                params,
                lr=self.learning_rate,
                weight_decay=weight_decay,
            )
        elif optimizer_type.lower() == 'adamw':
            return torch.optim.AdamW(
                params,
                lr=self.learning_rate,
                weight_decay=weight_decay,
            )
        elif optimizer_type.lower() == 'sgd':
            return torch.optim.SGD(
                params,
                lr=self.learning_rate,
                weight_decay=weight_decay,
                momentum=0.9,
            )
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_type}")
    
    def fit(
        self,
        train_data: FusionData,
        val_data: Optional[FusionData] = None,
        verbose: bool = True,
    ) -> TrainingHistory:
        """
        Train the model.
        
        Parameters
        ----------
        train_data : FusionData
            Training data.
        val_data : FusionData, optional
            Validation data for monitoring.
        verbose : bool, default=True
            Show progress bar.
            
        Returns
        -------
        TrainingHistory
            Training history with losses and metrics.
        """
        logger.info("Starting training...")
        
        # Initialize inducing points from training data
        train_dataset = FusionDataset(train_data)
        train_x = train_dataset.get_input_tensor().to(self.device)
        self.model.initialize_inducing_points(train_x, method='kmeans')
        
        # Create data loader
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=False,
        )
        
        # Prepare validation data
        if val_data is not None:
            val_dataset = FusionDataset(val_data)
            val_x = val_dataset.get_input_tensor().to(self.device)
            val_y = val_dataset.observations.to(self.device)
            val_masks = val_dataset.source_masks.to(self.device)
        
        # Notify callbacks
        for callback in self.callbacks:
            callback.on_train_begin(self.model)
        
        # Training loop
        self.history = TrainingHistory()
        
        epoch_iterator = tqdm(range(self.n_epochs), desc="Training", disable=not verbose)
        
        for epoch in epoch_iterator:
            epoch_start = time.time()
            
            # Training step
            train_loss = self._train_epoch(train_loader)
            
            # Validation step
            val_loss = None
            if val_data is not None and (epoch + 1) % self.val_interval == 0:
                val_loss = self._validate(val_x, val_y, val_masks)
            
            # Record history
            epoch_time = time.time() - epoch_start
            self.history.train_loss.append(train_loss)
            if val_loss is not None:
                self.history.val_loss.append(val_loss)
            self.history.learning_rates.append(
                self.optimizer.param_groups[0]['lr']
            )
            self.history.epoch_times.append(epoch_time)
            
            # Update progress bar
            desc = f"Epoch {epoch+1}/{self.n_epochs} | Train: {train_loss:.4f}"
            if val_loss is not None:
                desc += f" | Val: {val_loss:.4f}"
            epoch_iterator.set_description(desc)
            
            # Callbacks
            logs = {
                'train_loss': train_loss,
                'val_loss': val_loss if val_loss is not None else self.history.val_loss[-1] if self.history.val_loss else train_loss,
                'epoch': epoch,
            }
            
            continue_training = True
            for callback in self.callbacks:
                if not callback.on_epoch_end(epoch, logs, self.model):
                    continue_training = False
                    break
            
            if not continue_training:
                logger.info(f"Training stopped at epoch {epoch+1}")
                break
        
        # Notify callbacks of end
        final_logs = {
            'train_loss': self.history.train_loss[-1],
            'val_loss': self.history.val_loss[-1] if self.history.val_loss else None,
        }
        for callback in self.callbacks:
            callback.on_train_end(self.model, final_logs)
        
        logger.info(
            f"Training complete. Best val_loss: {self.history.best_val_loss:.4f} "
            f"at epoch {self.history.best_epoch + 1}"
        )
        
        return self.history
    
    def _train_epoch(self, train_loader: DataLoader) -> float:
        """
        Run one training epoch.
        
        Parameters
        ----------
        train_loader : DataLoader
            Training data loader.
            
        Returns
        -------
        float
            Average training loss.
        """
        self.model.train()
        self.model.likelihood.train()
        
        total_loss = 0.0
        n_batches = 0
        
        for batch in train_loader:
            coords, timestamps, observations, masks, indices = batch
            
            # Combine coords and timestamps
            x = torch.cat([
                coords.to(self.device),
                timestamps.unsqueeze(-1).to(self.device)
            ], dim=-1)
            y = observations.to(self.device)
            source_masks = masks.to(self.device)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Compute ELBO (negative loss)
            elbo = self.model.elbo(x, y, source_masks)
            loss = -elbo / len(x)  # Normalize by batch size
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            if self.gradient_clip is not None:
                torch.nn.utils.clip_grad_norm_(
                    list(self.model.parameters()) + list(self.model.likelihood.parameters()),
                    self.gradient_clip
                )
            
            # Update parameters
            self.optimizer.step()
            
            total_loss += loss.item()
            n_batches += 1
        
        return total_loss / n_batches
    
    def _validate(
        self,
        val_x: torch.Tensor,
        val_y: torch.Tensor,
        val_masks: torch.Tensor,
    ) -> float:
        """
        Compute validation loss.
        
        Parameters
        ----------
        val_x : torch.Tensor
            Validation inputs.
        val_y : torch.Tensor
            Validation observations.
        val_masks : torch.Tensor
            Validation masks.
            
        Returns
        -------
        float
            Validation loss.
        """
        self.model.eval()
        self.model.likelihood.eval()
        
        with torch.no_grad():
            elbo = self.model.elbo(val_x, val_y, val_masks)
            loss = -elbo / len(val_x)
        
        return loss.item()
    
    def save(self, path: str | Path) -> None:
        """
        Save the trained model.
        
        Parameters
        ----------
        path : str or Path
            Save path.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'likelihood_state_dict': self.model.likelihood.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'history': self.history.to_dict(),
            'config': {
                'learning_rate': self.learning_rate,
                'n_epochs': self.n_epochs,
                'batch_size': self.batch_size,
            }
        }
        
        torch.save(checkpoint, path)
        logger.info(f"Model saved to {path}")
    
    def load(self, path: str | Path) -> None:
        """
        Load a saved model.
        
        Parameters
        ----------
        path : str or Path
            Path to saved checkpoint.
        """
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.likelihood.load_state_dict(checkpoint['likelihood_state_dict'])
        
        if 'optimizer_state_dict' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        logger.info(f"Model loaded from {path}")
