"""Inference service for the yoga pose classifier.

Loads TFLite EfficientNetB3 models declared in ``configs/inference_config.json``
(with a fallback to ``configs/inference_config.example.json``) and exposes a
small :class:`InferenceService` facade that the Streamlit UI and the smoke
script consume.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from PIL import Image

# Prefer the standalone tflite_runtime wheel when available.
try:
    from tflite_runtime.interpreter import Interpreter
except Exception:  # pragma: no cover - fallback depends on local install
    from tensorflow.lite.python.interpreter import Interpreter


LOGGER = logging.getLogger(__name__)

DEFAULT_CONFIG_PATH = Path("configs/inference_config.json")
DEFAULT_EXAMPLE_CONFIG_PATH = Path("configs/inference_config.example.json")


class InferenceConfigError(Exception):
    """Raised when inference configuration is missing required fields or values."""


class InferenceRuntimeError(Exception):
    """Raised when model loading or prediction fails at runtime."""


@dataclass(frozen=True)
class InferenceConfig:
    """Immutable, validated view of the on-disk inference configuration."""

    labels_path: str
    model_paths: Dict[str, str]
    image_size: Tuple[int, int]
    top_k: int


@dataclass(frozen=True)
class PredictionResult:
    """Structured output returned by :meth:`InferenceService.predict`."""

    model_name: str
    predicted_label: str
    confidence: float
    inference_time_ms: float
    top_k_scores: List[Tuple[str, float]]


def get_repo_root() -> Path:
    """Return the repository root, resolved relative to this module's location."""
    return Path(__file__).resolve().parents[2]


def _validate_relative_path(path_str: str, key: str) -> None:
    if Path(path_str).is_absolute():
        raise InferenceConfigError(
            f"{key} must be a repository-relative path, got absolute path: {path_str}"
        )


def _require_key(data: Dict[str, Any], key: str) -> Any:
    if key not in data:
        raise InferenceConfigError(f"Missing required config key: '{key}'")
    return data[key]


def load_inference_config(config_path: str | Path | None = None) -> InferenceConfig:
    """Load and validate an :class:`InferenceConfig` from disk.

    When ``config_path`` is ``None`` the loader looks for
    ``configs/inference_config.json``; if that file is absent it falls back to
    ``configs/inference_config.example.json`` so the app remains runnable in a
    fresh clone.
    """
    repo_root = get_repo_root()
    selected = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
    absolute_selected = selected if selected.is_absolute() else repo_root / selected

    if not absolute_selected.exists():
        fallback = repo_root / DEFAULT_EXAMPLE_CONFIG_PATH
        if not fallback.exists():
            raise InferenceConfigError(
                f"Config file not found: {absolute_selected}. "
                f"Missing fallback example config at {fallback}."
            )
        absolute_selected = fallback
        LOGGER.warning(
            "Using example config because configs/inference_config.json is missing."
        )

    with open(absolute_selected, "r", encoding="utf-8") as handle:
        data: Dict[str, Any] = json.load(handle)

    labels_path = str(_require_key(data, "labels_path"))
    model_paths_raw = _require_key(data, "model_paths")
    image_size_raw = _require_key(data, "image_size")
    top_k = int(_require_key(data, "top_k"))

    _validate_relative_path(labels_path, "labels_path")
    if not isinstance(model_paths_raw, dict) or not model_paths_raw:
        raise InferenceConfigError("model_paths must be a non-empty mapping.")
    for model_name, model_path in model_paths_raw.items():
        if not isinstance(model_name, str) or not model_name.strip():
            raise InferenceConfigError("model_paths contains an empty model name.")
        _validate_relative_path(str(model_path), f"model_paths[{model_name}]")

    if not isinstance(image_size_raw, (list, tuple)) or len(image_size_raw) != 2:
        raise InferenceConfigError(
            "image_size must contain two integer values [width, height]."
        )
    image_size = (int(image_size_raw[0]), int(image_size_raw[1]))
    if image_size[0] <= 0 or image_size[1] <= 0:
        raise InferenceConfigError("image_size values must be positive.")
    if top_k <= 0:
        raise InferenceConfigError("top_k must be positive.")

    return InferenceConfig(
        labels_path=labels_path,
        model_paths={str(k): str(v) for k, v in model_paths_raw.items()},
        image_size=image_size,
        top_k=top_k,
    )


def load_labels(labels_path: str) -> List[str]:
    """Read class labels from a text file, one label per non-empty line."""
    labels_full_path = get_repo_root() / labels_path
    if not labels_full_path.exists():
        raise InferenceRuntimeError(f"Label file not found: {labels_full_path}")
    with open(labels_full_path, "r", encoding="utf-8") as handle:
        labels = [line.strip() for line in handle.readlines() if line.strip()]
    if not labels:
        raise InferenceRuntimeError(f"Label file is empty: {labels_full_path}")
    return labels


def load_models(models: Dict[str, str]) -> Dict[str, dict]:
    """Instantiate a TFLite interpreter for each entry in ``models``.

    Failures are captured per-model rather than raised so the UI can still
    start and surface a clear error only when the user selects an
    unavailable model.
    """
    loaded: Dict[str, dict] = {}
    for name, rel_path in models.items():
        model_path = get_repo_root() / rel_path
        try:
            interp = Interpreter(model_path=str(model_path))
            interp.allocate_tensors()
            loaded[name] = {
                "type": "tflite",
                "interpreter": interp,
                "path": str(model_path),
            }
        except Exception as exc:
            LOGGER.exception(
                "Failed to load model '%s' at '%s': %s", name, model_path, exc
            )
            loaded[name] = {"type": "error", "error": str(exc), "path": str(model_path)}
    return loaded


def preprocess_image(pil_img: Image.Image, image_size: Tuple[int, int]) -> np.ndarray:
    """Convert a PIL image to a batched float32 tensor of shape ``(1, H, W, 3)``.

    The EfficientNetB3 graphs exported for this app have Keras' built-in
    preprocessing baked in, so the interpreter expects raw 0-255 float
    pixels rather than a normalized input.
    """
    img = pil_img.convert("RGB").resize(image_size)
    x = np.asarray(img, dtype=np.float32)
    return np.expand_dims(x, axis=0)


def predict_with_tflite(
    interp: Interpreter, x: np.ndarray
) -> Tuple[np.ndarray, float]:
    """Run one forward pass and return ``(probabilities, elapsed_seconds)``.

    The input tensor is cast to match the interpreter's declared dtype so the
    same code path works for both float32 and float16-quantized exports.
    Integer-quantized models are rejected explicitly because they require a
    different preprocessing contract.
    """
    input_details = interp.get_input_details()[0]
    output_details = interp.get_output_details()[0]

    x_in = x
    if input_details["dtype"] == np.float16:
        x_in = x.astype(np.float16)
    elif input_details["dtype"] == np.uint8:
        raise InferenceRuntimeError("This app expects float16/float32 TFLite models.")

    interp.set_tensor(input_details["index"], x_in)
    t0 = time.perf_counter()
    interp.invoke()
    dt = time.perf_counter() - t0

    y = interp.get_tensor(output_details["index"])
    return np.squeeze(y, axis=0), dt


def topk_labels(
    probs: np.ndarray, labels: List[str], k: int
) -> List[Tuple[str, float]]:
    """Return the ``k`` highest-probability ``(label, score)`` pairs."""
    idx = np.argsort(-probs)[:k]
    return [(labels[i], float(probs[i])) for i in idx]


class InferenceService:
    """High-level inference facade used by the Streamlit UI and the smoke script.

    The service eagerly loads every configured interpreter at construction
    time. Per-model load failures are recorded in :attr:`models` rather than
    raised, so the UI can still start when one export is corrupt; the
    failure is surfaced only when that specific model is selected.
    """

    def __init__(self, config: InferenceConfig) -> None:
        self.config = config
        self.labels = load_labels(config.labels_path)
        self.models = load_models(config.model_paths)

    def available_model_names(self) -> List[str]:
        return list(self.config.model_paths.keys())

    def validate_model(self, model_name: str) -> dict:
        if model_name not in self.models:
            raise InferenceRuntimeError(f"Unknown model name: {model_name}")
        model_info = self.models[model_name]
        if model_info.get("type") == "error":
            raise InferenceRuntimeError(
                f"Model '{model_name}' is unavailable: {model_info.get('error')}"
            )
        return model_info

    def predict(self, model_name: str, pil_image: Image.Image) -> PredictionResult:
        """Run inference with the named model and return a :class:`PredictionResult`."""
        model_info = self.validate_model(model_name)
        x = preprocess_image(pil_image, image_size=self.config.image_size)
        probs, dt = predict_with_tflite(model_info["interpreter"], x)
        predicted_index = int(np.argmax(probs))
        top_scores = topk_labels(probs, self.labels, self.config.top_k)
        return PredictionResult(
            model_name=model_name,
            predicted_label=self.labels[predicted_index],
            confidence=float(probs[predicted_index]),
            inference_time_ms=dt * 1000.0,
            top_k_scores=top_scores,
        )
