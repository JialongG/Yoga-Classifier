"""Pose feature extraction and FFN training pipeline using YOLO and FFN."""
# pyright: reportMissingImports=false

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from tensorflow import keras
from tensorflow.keras import layers
from ultralytics import YOLO

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class PoseExtractionConfig:
    """Configuration for YOLO pose extraction."""

    yolo_model_path: str = "yolo11s-pose.pt"
    image_size: int = 640
    max_images_per_class: int | None = None
    extensions: tuple[str, ...] = (".jpg", ".jpeg", ".png", ".bmp")


@dataclass(frozen=True)
class PoseTrainingConfig:
    """Training configuration for the FFN classifier on pose keypoints."""

    epochs: int = 120
    batch_size: int = 32
    learning_rate: float = 1e-3
    early_stopping_patience: int = 30
    reduce_lr_factor: float = 0.75
    reduce_lr_patience: int = 5


@dataclass(frozen=True)
class PosePipelineArtifacts:
    """Output artifact paths for pose extraction and FFN training."""

    output_dir: str
    class_names_json: str
    train_features_npz: str
    val_features_npz: str
    test_features_npz: str
    ffn_model_keras: str
    metrics_json: str


class PosePipelineError(Exception):
    """Raised when pose extraction or training pipeline fails."""


def _select_largest_bounding_box(result: Any) -> int | None:
    boxes = getattr(result, "boxes", None)
    if boxes is None or len(boxes) == 0:
        return None
    xyxy = boxes.xyxy.cpu().numpy()
    areas = (xyxy[:, 2] - xyxy[:, 0]) * (xyxy[:, 3] - xyxy[:, 1])
    return int(np.argmax(areas))


def extract_keypoints(yolo: YOLO, image: Image.Image, image_size: int) -> np.ndarray | None:
    """Extract 17x2 keypoints from the largest detected person."""
    rgb = image.convert("RGB")
    np_image = np.array(rgb)
    result = yolo.predict(source=np_image, imgsz=image_size, verbose=False)[0]
    main_idx = _select_largest_bounding_box(result)
    if main_idx is None or result.keypoints is None:
        return None

    kpts_tensor = result.keypoints.xy
    if kpts_tensor is None:
        return None
    kpts = kpts_tensor.cpu().numpy()
    if kpts.ndim == 3:
        if main_idx >= kpts.shape[0]:
            return None
        return kpts[main_idx]
    if kpts.ndim == 2:
        return kpts
    return None


def normalize_keypoints(kpts: np.ndarray) -> np.ndarray:
    """Center and scale keypoints, then flatten to a fixed 34D vector."""
    if kpts.ndim != 2 or kpts.shape[1] != 2:
        raise PosePipelineError(f"Expected keypoints shape (N,2), got {kpts.shape}")

    hip_ids = [11, 12]
    valid_hips = [i for i in hip_ids if i < kpts.shape[0]]
    if valid_hips:
        center = kpts[valid_hips].mean(axis=0)
    else:
        center = kpts.mean(axis=0)
    centered = kpts - center

    shoulder_ids = [5, 6]
    if all(i < kpts.shape[0] for i in shoulder_ids):
        shoulder_dist = np.linalg.norm(centered[shoulder_ids[0]] - centered[shoulder_ids[1]])
    else:
        shoulder_dist = 0.0
    scale = float(shoulder_dist) if shoulder_dist > 1e-6 else float(np.max(np.linalg.norm(centered, axis=1)) + 1e-6)
    normalized = centered / scale
    return normalized.reshape(-1).astype(np.float32)


def _extract_feature_from_path(
    yolo: YOLO,
    image_path: Path,
    config: PoseExtractionConfig,
) -> np.ndarray | None:
    try:
        with Image.open(image_path) as image:
            keypoints = extract_keypoints(yolo, image, image_size=config.image_size)
        if keypoints is None:
            return None
        return normalize_keypoints(keypoints)
    except Exception as exc:
        LOGGER.warning("Skipping '%s' due to extraction failure: %s", image_path, exc)
        return None


def _split_dir(dataset_root: Path, split: str) -> Path:
    split_dir = dataset_root / split
    if not split_dir.exists():
        raise PosePipelineError(f"Missing split directory: {split_dir}")
    return split_dir


def build_pose_feature_dataset(
    dataset_root: str | Path,
    split: str,
    yolo: YOLO,
    config: PoseExtractionConfig,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Extract normalized pose vectors and labels from one dataset split."""
    root = Path(dataset_root).expanduser().resolve()
    split_dir = _split_dir(root, split)
    class_dirs = sorted([p for p in split_dir.iterdir() if p.is_dir()])
    class_names = [p.name for p in class_dirs]
    class_to_idx = {name: idx for idx, name in enumerate(class_names)}

    x_list: list[np.ndarray] = []
    y_list: list[int] = []
    for class_dir in class_dirs:
        image_paths = [p for p in class_dir.iterdir() if p.suffix.lower() in config.extensions]
        if config.max_images_per_class is not None:
            image_paths = image_paths[: config.max_images_per_class]
        for image_path in image_paths:
            feat = _extract_feature_from_path(yolo, image_path, config)
            if feat is None:
                continue
            x_list.append(feat)
            y_list.append(class_to_idx[class_dir.name])

    if not x_list:
        raise PosePipelineError(
            f"No pose features extracted from split '{split}' under {split_dir}. "
            "Check dataset quality and YOLO model availability."
        )
    x = np.vstack(x_list).astype(np.float32)
    y = np.array(y_list, dtype=np.int64)
    return x, y, class_names


def build_ffn_classifier(num_classes: int, input_dim: int = 34) -> keras.Model:
    model = keras.Sequential(
        [
            layers.Input(shape=(input_dim,)),
            layers.Dense(128, activation="relu"),
            layers.Dropout(0.1),
            layers.Dense(128, activation="relu"),
            layers.Dropout(0.15),
            layers.Dense(num_classes, activation="softmax"),
        ],
        name="yolo_pose_ffn",
    )
    return model


def train_pose_pipeline(
    dataset_root: str | Path,
    output_dir: str | Path = "artifacts/pose_runs",
    extraction_config: PoseExtractionConfig | None = None,
    training_config: PoseTrainingConfig | None = None,
) -> PosePipelineArtifacts:
    """Run YOLO keypoint extraction on train/val/test and train FFN classifier."""
    e_cfg = extraction_config or PoseExtractionConfig()
    t_cfg = training_config or PoseTrainingConfig()
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    run_dir = Path(output_dir).expanduser().resolve() / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)

    yolo = YOLO(e_cfg.yolo_model_path)
    x_train, y_train, class_names = build_pose_feature_dataset(dataset_root, "train", yolo, e_cfg)
    x_val, y_val, class_names_val = build_pose_feature_dataset(dataset_root, "val", yolo, e_cfg)
    x_test, y_test, class_names_test = build_pose_feature_dataset(dataset_root, "test", yolo, e_cfg)
    if class_names != class_names_val or class_names != class_names_test:
        raise PosePipelineError("Class ordering mismatch across train/val/test splits.")

    num_classes = len(class_names)
    model = build_ffn_classifier(num_classes=num_classes, input_dim=x_train.shape[1])
    model.compile(
        optimizer=keras.optimizers.Adam(t_cfg.learning_rate),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    y_train_cat = keras.utils.to_categorical(y_train, num_classes)
    y_val_cat = keras.utils.to_categorical(y_val, num_classes)
    y_test_cat = keras.utils.to_categorical(y_test, num_classes)

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=t_cfg.early_stopping_patience,
            restore_best_weights=True,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_accuracy",
            factor=t_cfg.reduce_lr_factor,
            patience=t_cfg.reduce_lr_patience,
        ),
    ]

    history = model.fit(
        x_train,
        y_train_cat,
        validation_data=(x_val, y_val_cat),
        epochs=t_cfg.epochs,
        batch_size=t_cfg.batch_size,
        callbacks=callbacks,
        verbose=1,
    )

    val_loss, val_acc = model.evaluate(x_val, y_val_cat, verbose=0)
    test_loss, test_acc = model.evaluate(x_test, y_test_cat, verbose=0)

    (run_dir / "class_names.json").write_text(json.dumps(class_names, indent=2), encoding="utf-8")
    np.savez(run_dir / "train_features.npz", x=x_train, y=y_train)
    np.savez(run_dir / "val_features.npz", x=x_val, y=y_val)
    np.savez(run_dir / "test_features.npz", x=x_test, y=y_test)
    model.save(run_dir / "yolo_pose_ffn.keras")
    metrics = {
        "val_loss": float(val_loss),
        "val_accuracy": float(val_acc),
        "test_loss": float(test_loss),
        "test_accuracy": float(test_acc),
        "epochs_ran": len(history.history.get("loss", [])),
    }
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    return PosePipelineArtifacts(
        output_dir=str(run_dir),
        class_names_json=str(run_dir / "class_names.json"),
        train_features_npz=str(run_dir / "train_features.npz"),
        val_features_npz=str(run_dir / "val_features.npz"),
        test_features_npz=str(run_dir / "test_features.npz"),
        ffn_model_keras=str(run_dir / "yolo_pose_ffn.keras"),
        metrics_json=str(run_dir / "metrics.json"),
    )


def artifacts_to_dict(artifacts: PosePipelineArtifacts) -> dict[str, Any]:
    return asdict(artifacts)

