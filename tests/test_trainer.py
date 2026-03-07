"""
Tests for Trainer (probabilistic core).

Run with: pytest tests/test_trainer.py -v
"""

import numpy as np
import torch

from src.data.loader import FusionData
from src.models.svgp import FusionSVGP
from src.training.trainer import Trainer


def _make_fusion_data(n: int = 40, include_epa: bool = True) -> FusionData:
    rng = np.random.default_rng(0)
    coords = np.column_stack([
        rng.uniform(53.25, 53.45, size=n),
        rng.uniform(-6.45, -6.10, size=n),
    ])
    timestamps = rng.uniform(0.0, 1.0, size=n)
    grid_ids = rng.integers(0, 10, size=n)
    raw_timestamps = np.array([f"t{i}" for i in range(n)], dtype=object)

    epa = rng.normal(20.0, 5.0, size=n) if include_epa else np.full(n, np.nan)
    low_cost = rng.normal(22.0, 6.0, size=n)
    satellite = rng.normal(18.0, 4.0, size=n)

    # Introduce some missing values
    low_cost[rng.random(n) < 0.1] = np.nan
    satellite[rng.random(n) < 0.1] = np.nan

    observations = {
        "epa": epa,
        "low_cost": low_cost,
        "satellite": satellite,
    }
    source_masks = {
        "epa": ~np.isnan(epa),
        "low_cost": ~np.isnan(low_cost),
        "satellite": ~np.isnan(satellite),
    }

    return FusionData(
        coords=coords,
        timestamps=timestamps,
        observations=observations,
        source_masks=source_masks,
        grid_ids=grid_ids,
        raw_timestamps=raw_timestamps,
    )


def test_trainer_fit_smoke():
    torch.manual_seed(0)
    data = _make_fusion_data(n=50, include_epa=True)

    model = FusionSVGP(n_inducing=10)
    trainer = Trainer(
        model,
        n_epochs=2,
        batch_size=16,
        val_interval=1,
        learning_rate=0.01,
    )

    history = trainer.fit(data, val_data=data, verbose=False)

    assert len(history.train_loss) == 2
    assert len(history.val_loss) >= 1
    assert np.isfinite(history.train_loss).all()
    assert np.isfinite(history.val_loss).all()


def test_trainer_fit_no_epa_in_val():
    torch.manual_seed(0)
    train_data = _make_fusion_data(n=40, include_epa=True)
    val_data = _make_fusion_data(n=30, include_epa=False)

    model = FusionSVGP(n_inducing=8)
    trainer = Trainer(
        model,
        n_epochs=1,
        batch_size=16,
        val_interval=1,
        learning_rate=0.01,
    )

    history = trainer.fit(train_data, val_data=val_data, verbose=False)

    assert len(history.train_loss) == 1
    assert len(history.val_loss) >= 1
