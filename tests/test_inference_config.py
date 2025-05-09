from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from yoga_pose_app.inference import InferenceConfigError, load_inference_config


def test_load_inference_config_has_models() -> None:
    config = load_inference_config()
    assert config.labels_path
    assert config.model_paths
    assert config.image_size[0] > 0
    assert config.image_size[1] > 0
    assert config.top_k > 0


def test_load_inference_config_rejects_absolute_labels_path(tmp_path: Path) -> None:
    bad = {
        "labels_path": "/etc/passwd",
        "model_paths": {"m": "assets/models/tflite/effb3_ft1_fp16_UI.tflite"},
        "image_size": [300, 300],
        "top_k": 5,
    }
    config_file = tmp_path / "bad.json"
    config_file.write_text(json.dumps(bad))
    with pytest.raises(InferenceConfigError):
        load_inference_config(config_file)


def test_load_inference_config_rejects_missing_required_key(tmp_path: Path) -> None:
    config_file = tmp_path / "bad.json"
    config_file.write_text(json.dumps({"labels_path": "assets/labels.txt"}))
    with pytest.raises(InferenceConfigError):
        load_inference_config(config_file)


def test_load_inference_config_rejects_non_positive_top_k(tmp_path: Path) -> None:
    bad = {
        "labels_path": "assets/labels/yoga-poses-english.txt",
        "model_paths": {"m": "assets/models/tflite/effb3_ft1_fp16_UI.tflite"},
        "image_size": [300, 300],
        "top_k": 0,
    }
    config_file = tmp_path / "bad.json"
    config_file.write_text(json.dumps(bad))
    with pytest.raises(InferenceConfigError):
        load_inference_config(config_file)
