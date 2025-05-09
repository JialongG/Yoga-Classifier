"""Dataset construction utilities for model training."""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import tensorflow as tf
from tensorflow import keras

from .training_config import DatasetPaths, TrainingConfig


def resolve_dataset_paths(dataset_root: str | Path) -> DatasetPaths:
    """Resolve and validate train/val/test split directories."""
    root = Path(dataset_root).expanduser().resolve()
    train_dir = root / "train"
    val_dir = root / "val"
    test_dir = root / "test"
    for split_dir in (train_dir, val_dir, test_dir):
        if not split_dir.exists():
            raise FileNotFoundError(
                f"Missing dataset split directory: {split_dir}. "
                "Expected layout: <dataset_root>/{train,val,test}/<class_name>/*.png"
            )
    return DatasetPaths(train_dir=train_dir, val_dir=val_dir, test_dir=test_dir)


def _image_dataset(
    directory: Path,
    image_size: Tuple[int, int],
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> tf.data.Dataset:
    return keras.utils.image_dataset_from_directory(
        directory,
        image_size=image_size,
        color_mode="rgb",
        batch_size=batch_size,
        shuffle=shuffle,
        seed=seed,
        label_mode="categorical",
    )


def build_datasets(
    paths: DatasetPaths, config: TrainingConfig
) -> tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
    """Build EfficientNet-compatible datasets at ``config.image_size``."""
    train_ds = _image_dataset(
        paths.train_dir,
        image_size=config.image_size,
        batch_size=config.batch_size,
        shuffle=True,
        seed=config.random_seed,
    )
    val_ds = _image_dataset(
        paths.val_dir,
        image_size=config.image_size,
        batch_size=config.batch_size,
        shuffle=False,
        seed=config.random_seed,
    )
    test_ds = _image_dataset(
        paths.test_dir,
        image_size=config.image_size,
        batch_size=config.batch_size,
        shuffle=False,
        seed=config.random_seed,
    )
    autotune = tf.data.AUTOTUNE
    return (
        train_ds.prefetch(autotune),
        val_ds.prefetch(autotune),
        test_ds.prefetch(autotune),
    )


def build_cnn_datasets(
    paths: DatasetPaths, config: TrainingConfig
) -> tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
    """Build CNN baseline datasets at ``config.cnn_image_size``."""
    train_ds = _image_dataset(
        paths.train_dir,
        image_size=config.cnn_image_size,
        batch_size=config.batch_size,
        shuffle=True,
        seed=config.random_seed,
    )
    val_ds = _image_dataset(
        paths.val_dir,
        image_size=config.cnn_image_size,
        batch_size=config.batch_size,
        shuffle=False,
        seed=config.random_seed,
    )
    test_ds = _image_dataset(
        paths.test_dir,
        image_size=config.cnn_image_size,
        batch_size=config.batch_size,
        shuffle=False,
        seed=config.random_seed,
    )
    autotune = tf.data.AUTOTUNE
    return (
        train_ds.prefetch(autotune),
        val_ds.prefetch(autotune),
        test_ds.prefetch(autotune),
    )

