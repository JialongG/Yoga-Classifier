from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from PIL import Image

try:
    from tflite_runtime.interpreter import Interpreter
except Exception:  # pragma: no cover - fallback depends on local install
    from tensorflow.lite.python.interpreter import Interpreter


LOGGER = logging.getLogger(__name__)
DEFAULT_CONFIG_PATH = Path("configs/inference_config.json")
DEFAULT_EXAMPLE_CONFIG_PATH = Path("configs/inference_config.example.json")

IMG_SIZE: Tuple[int, int] = (300, 300)
TOPK: int = 5

MODEL_PATHS: Dict[str, str] = {
    "EffNetB3 FT (unfreeze 20)": "assets/models/tflite/effb3_ft1_fp16_UI.tflite",
    "EffNetB3 FT (unfreeze 30)": "assets/models/tflite/effb3_ft2_fp16_UI.tflite",
    "EffNetB3 FT (unfreeze 40)": "assets/models/tflite/effb3_ft3_fp16_UI.tflite",
}
LABELS_PATH: str = "assets/models/tflite/yoga-poses-english.txt"


class InferenceConfigError(Exception):
    """Raised when inference configuration is invalid."""


class InferenceRuntimeError(Exception):
    """Raised when model loading or prediction fails."""


@dataclass(frozen=True)
class InferenceConfig:
    labels_path: str
    model_paths: Dict[str, str]
    image_size: Tuple[int, int]
    top_k: int


@dataclass(frozen=True)
class PredictionResult:
    model_name: str
    predicted_label: str
    confidence: float
    inference_time_ms: float
    top_k: List[Tuple[str, float]]


def get_repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _validate_relative_path(path_str: str, key: str) -> None:
    path = Path(path_str)
    if path.is_absolute():
        raise InferenceConfigError(f"{key} must be a repository-relative path, got absolute path: {path_str}")


def load_inference_config(config_path: str | Path | None = None) -> InferenceConfig:
    repo_root = get_repo_root()
    selected = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
    absolute_selected = selected if selected.is_absolute() else repo_root / selected

    if not absolute_selected.exists():
        fallback = repo_root / DEFAULT_EXAMPLE_CONFIG_PATH
        if not fallback.exists():
            raise InferenceConfigError(
                f"Config file not found: {absolute_selected}. Missing fallback example config at {fallback}."
            )
        absolute_selected = fallback
        LOGGER.warning("Using example config because configs/inference_config.json is missing.")

    with open(absolute_selected, "r", encoding="utf-8") as handle:
        data: Dict[str, Any] = json.load(handle)

    labels_path = data.get("labels_path", LABELS_PATH)
    model_paths = data.get("model_paths", MODEL_PATHS)
    image_size_raw = data.get("image_size", list(IMG_SIZE))
    top_k = int(data.get("top_k", TOPK))

    _validate_relative_path(labels_path, "labels_path")
    if not isinstance(model_paths, dict) or not model_paths:
        raise InferenceConfigError("model_paths must be a non-empty mapping.")
    for model_name, model_path in model_paths.items():
        if not isinstance(model_name, str) or not model_name.strip():
            raise InferenceConfigError("model_paths contains an empty model name.")
        _validate_relative_path(str(model_path), f"model_paths[{model_name}]")

    if not isinstance(image_size_raw, (list, tuple)) or len(image_size_raw) != 2:
        raise InferenceConfigError("image_size must contain two integer values [width, height].")
    image_size = (int(image_size_raw[0]), int(image_size_raw[1]))
    if image_size[0] <= 0 or image_size[1] <= 0:
        raise InferenceConfigError("image_size values must be positive.")
    if top_k <= 0:
        raise InferenceConfigError("top_k must be positive.")

    return InferenceConfig(
        labels_path=labels_path,
        model_paths={str(k): str(v) for k, v in model_paths.items()},
        image_size=image_size,
        top_k=top_k,
    )


def load_labels(labels_path: str = LABELS_PATH) -> List[str]:
    labels_full_path = get_repo_root() / labels_path
    if not labels_full_path.exists():
        raise InferenceRuntimeError(f"Label file not found: {labels_full_path}")
    with open(labels_full_path, "r", encoding="utf-8") as f:
        labels = [line.strip() for line in f.readlines() if line.strip()]
    if not labels:
        raise InferenceRuntimeError(f"Label file is empty: {labels_full_path}")
    return labels


def load_models(models: Dict[str, str] | None = None) -> Dict[str, dict]:
    models = models or MODEL_PATHS
    loaded: Dict[str, dict] = {}
    for name, rel_path in models.items():
        model_path = get_repo_root() / rel_path
        try:
            interp = Interpreter(model_path=str(model_path))
            interp.allocate_tensors()
            loaded[name] = {"type": "tflite", "interpreter": interp, "path": str(model_path)}
        except Exception as exc:
            LOGGER.exception("Failed to load model '%s' at '%s': %s", name, model_path, exc)
            loaded[name] = {"type": "error", "error": str(exc), "path": str(model_path)}
    return loaded


def preprocess_image(pil_img: Image.Image, image_size: Tuple[int, int] = IMG_SIZE) -> np.ndarray:
    img = pil_img.convert("RGB").resize(image_size)
    x = np.asarray(img, dtype=np.float32)
    return np.expand_dims(x, axis=0)


def predict_with_tflite(interp: Interpreter, x: np.ndarray) -> Tuple[np.ndarray, float]:
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


def topk_labels(probs: np.ndarray, labels: List[str], k: int = TOPK) -> List[Tuple[str, float]]:
    idx = np.argsort(-probs)[:k]
    return [(labels[i], float(probs[i])) for i in idx]


class InferenceService:
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
            raise InferenceRuntimeError(f"Model '{model_name}' is unavailable: {model_info.get('error')}")
        return model_info

    def predict(self, model_name: str, pil_image: Image.Image) -> PredictionResult:
        model_info = self.validate_model(model_name)
        x = preprocess_image(pil_image, image_size=self.config.image_size)
        probs, dt = predict_with_tflite(model_info["interpreter"], x)
        predicted_index = int(np.argmax(probs))
        topk = topk_labels(probs, self.labels, self.config.top_k)
        return PredictionResult(
            model_name=model_name,
            predicted_label=self.labels[predicted_index],
            confidence=float(probs[predicted_index]),
            inference_time_ms=dt * 1000.0,
            top_k=topk,
        )

