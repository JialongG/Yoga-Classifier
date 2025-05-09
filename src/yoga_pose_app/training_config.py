"""Typed configuration for training and fine-tuning pipelines."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple


@dataclass(frozen=True)
class DatasetPaths:
    """Dataset split roots expected by the training pipeline."""

    train_dir: Path
    val_dir: Path
    test_dir: Path


@dataclass(frozen=True)
class TrainingConfig:
    """Hyperparameters used by transfer learning and CNN baseline training."""

    image_size: Tuple[int, int] = (300, 300)
    cnn_image_size: Tuple[int, int] = (224, 224)
    batch_size: int = 32
    epochs: int = 90
    phase1_learning_rate: float = 1e-3
    finetune_learning_rate: float = 1e-4
    cnn_learning_rate: float = 1e-3
    dropout_rate: float = 0.3
    head_dense_units: int = 256
    unfreeze_variants: Tuple[int, ...] = (30, 20, 40)
    random_seed: int = 42

