"""Model builders mirroring the core training strategies used in the project."""
# pyright: reportMissingImports=false

from __future__ import annotations

from typing import Sequence

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import EfficientNetB3

from .training_config import TrainingConfig


def build_effb3_transfer(
    num_classes: int, config: TrainingConfig, input_shape: tuple[int, int, int]
) -> keras.Model:
    """Build phase-1 EfficientNetB3 transfer model with a custom dense head."""
    base = EfficientNetB3(
        include_top=False,
        weights="imagenet",
        input_shape=input_shape,
        pooling="avg",
    )
    base.trainable = False
    return keras.Sequential(
        [
            base,
            layers.Dropout(config.dropout_rate),
            layers.Dense(config.head_dense_units, activation="relu"),
            layers.Dropout(config.dropout_rate),
            layers.Dense(num_classes, activation="softmax"),
        ],
        name="effb3_transfer",
    )


def configure_finetuning(model: keras.Model, unfreeze_count: int) -> None:
    """Unfreeze the last ``unfreeze_count`` EfficientNet layers, keep BN frozen."""
    base = model.layers[0]
    base.trainable = True
    for layer in base.layers[:-unfreeze_count]:
        layer.trainable = False
    for layer in base.layers[-unfreeze_count:]:
        if isinstance(layer, layers.BatchNormalization):
            layer.trainable = False


def compile_effb3_phase1(model: keras.Model, config: TrainingConfig) -> None:
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=config.phase1_learning_rate),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )


def compile_effb3_finetune(model: keras.Model, config: TrainingConfig) -> None:
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=config.finetune_learning_rate),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )


def build_cnn_baseline(num_classes: int) -> keras.Model:
    """Construct CNN baseline used for comparison with transfer learning models."""
    model = keras.Sequential(name="cnn_baseline")
    model.add(layers.Rescaling(1.0 / 255, input_shape=(224, 224, 3)))
    model.add(layers.Conv2D(32, (3, 3), activation="relu", padding="same"))
    model.add(layers.BatchNormalization())
    model.add(layers.Conv2D(32, (3, 3), activation="relu", padding="same"))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Conv2D(64, (3, 3), activation="relu", padding="same"))
    model.add(layers.BatchNormalization())
    model.add(layers.Conv2D(64, (3, 3), activation="relu", padding="same"))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Conv2D(128, (3, 3), activation="relu", padding="same"))
    model.add(layers.BatchNormalization())
    model.add(layers.Conv2D(128, (3, 3), activation="relu", padding="same"))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Conv2D(256, (3, 3), activation="relu", padding="same"))
    model.add(layers.BatchNormalization())
    model.add(layers.Conv2D(256, (3, 3), activation="relu", padding="same"))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.1))

    model.add(layers.Conv2D(512, (3, 3), activation="relu", padding="same"))
    model.add(layers.BatchNormalization())
    model.add(layers.Conv2D(512, (3, 3), activation="relu", padding="same"))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.15))

    model.add(layers.GlobalAveragePooling2D())
    model.add(layers.Dense(512, activation="relu"))
    model.add(layers.Dropout(0.35))
    model.add(layers.Dense(num_classes, activation="softmax"))
    return model


def compile_cnn_baseline(model: keras.Model, config: TrainingConfig) -> None:
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=config.cnn_learning_rate),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )


def standard_training_callbacks(log_dir: str) -> Sequence[keras.callbacks.Callback]:
    return [
        keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=15, restore_best_weights=True
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=8, verbose=1
        ),
        keras.callbacks.TensorBoard(log_dir=log_dir),
    ]

