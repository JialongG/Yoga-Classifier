"""End-to-end training pipeline using EfficientNetB3 and FFN."""
# pyright: reportMissingImports=false

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import tensorflow as tf
from tensorflow import keras

from .training_config import TrainingConfig
from .training_data import build_cnn_datasets, build_datasets, resolve_dataset_paths
from .training_models import (
    build_cnn_baseline,
    build_effb3_transfer,
    compile_cnn_baseline,
    compile_effb3_finetune,
    compile_effb3_phase1,
    configure_finetuning,
    standard_training_callbacks,
)

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class TrainingArtifacts:
    """Paths of saved model outputs for later inference/export."""

    phase1_keras: str
    finetuned_keras: dict[str, str]
    finetuned_tflite_fp16: dict[str, str]
    cnn_baseline_keras: str
    metrics_json: str


def _save_tflite_fp16(model: keras.Model, output_path: Path) -> None:
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_types = [tf.float16]
    output_path.write_bytes(converter.convert())


def run_training_pipeline(
    dataset_root: str | Path,
    output_dir: str | Path = "artifacts/training_runs",
    config: TrainingConfig | None = None,
) -> TrainingArtifacts:
    """Train phase-1, three fine-tune variants, and the CNN baseline."""
    cfg = config or TrainingConfig()
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    out_root = Path(output_dir).expanduser().resolve() / timestamp
    out_root.mkdir(parents=True, exist_ok=True)
    tb_log_dir = out_root / "tb_logs"
    tb_log_dir.mkdir(parents=True, exist_ok=True)

    paths = resolve_dataset_paths(dataset_root)
    train_ds, val_ds, test_ds = build_datasets(paths, cfg)
    train_ds_cnn, val_ds_cnn, test_ds_cnn = build_cnn_datasets(paths, cfg)

    class_names = list(train_ds.class_names)
    num_classes = len(class_names)
    if num_classes == 0:
        raise ValueError("Dataset contains zero classes.")

    LOGGER.info("Training phase-1 transfer head model on %d classes", num_classes)
    phase1 = build_effb3_transfer(
        num_classes=num_classes, config=cfg, input_shape=cfg.image_size + (3,)
    )
    compile_effb3_phase1(phase1, cfg)
    phase1.fit(
        train_ds,
        validation_data=val_ds,
        epochs=cfg.epochs,
        callbacks=standard_training_callbacks(str(tb_log_dir / "phase1")),
        verbose=1,
    )
    phase1_weights = phase1.get_weights()
    phase1_path = out_root / "effb3_phase1.keras"
    phase1.save(phase1_path)

    finetuned_keras: dict[str, str] = {}
    finetuned_tflite: dict[str, str] = {}
    metrics: dict[str, Any] = {
        "num_classes": num_classes,
        "class_names": class_names,
        "unfreeze_variants": list(cfg.unfreeze_variants),
        "evaluations": {},
    }

    for unfreeze_count in cfg.unfreeze_variants:
        run_name = f"effb3_ft_unfreeze_{unfreeze_count}"
        LOGGER.info("Fine-tuning variant: unfreeze last %d layers", unfreeze_count)
        ft_model = build_effb3_transfer(
            num_classes=num_classes, config=cfg, input_shape=cfg.image_size + (3,)
        )
        ft_model.set_weights(phase1_weights)
        configure_finetuning(ft_model, unfreeze_count)
        compile_effb3_finetune(ft_model, cfg)
        ft_model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=cfg.epochs,
            callbacks=standard_training_callbacks(str(tb_log_dir / run_name)),
            verbose=1,
        )
        val_loss, val_acc = ft_model.evaluate(val_ds, verbose=0)
        test_loss, test_acc = ft_model.evaluate(test_ds, verbose=0)
        metrics["evaluations"][run_name] = {
            "val_loss": float(val_loss),
            "val_accuracy": float(val_acc),
            "test_loss": float(test_loss),
            "test_accuracy": float(test_acc),
        }

        keras_path = out_root / f"{run_name}.keras"
        tflite_path = out_root / f"{run_name}_fp16.tflite"
        ft_model.save(keras_path)
        _save_tflite_fp16(ft_model, tflite_path)
        finetuned_keras[str(unfreeze_count)] = str(keras_path)
        finetuned_tflite[str(unfreeze_count)] = str(tflite_path)

    LOGGER.info("Training CNN baseline model for comparison")
    cnn = build_cnn_baseline(num_classes=num_classes)
    compile_cnn_baseline(cnn, cfg)
    cnn.fit(
        train_ds_cnn,
        validation_data=val_ds_cnn,
        epochs=cfg.epochs,
        callbacks=standard_training_callbacks(str(tb_log_dir / "cnn_baseline")),
        verbose=1,
    )
    cnn_val_loss, cnn_val_acc = cnn.evaluate(val_ds_cnn, verbose=0)
    cnn_test_loss, cnn_test_acc = cnn.evaluate(test_ds_cnn, verbose=0)
    metrics["evaluations"]["cnn_baseline"] = {
        "val_loss": float(cnn_val_loss),
        "val_accuracy": float(cnn_val_acc),
        "test_loss": float(cnn_test_loss),
        "test_accuracy": float(cnn_test_acc),
    }

    cnn_path = out_root / "cnn_baseline.keras"
    cnn.save(cnn_path)

    metrics_path = out_root / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    return TrainingArtifacts(
        phase1_keras=str(phase1_path),
        finetuned_keras=finetuned_keras,
        finetuned_tflite_fp16=finetuned_tflite,
        cnn_baseline_keras=str(cnn_path),
        metrics_json=str(metrics_path),
    )


def artifacts_to_dict(artifacts: TrainingArtifacts) -> dict[str, Any]:
    """Serialize artifacts dataclass for CLI output."""
    return asdict(artifacts)

