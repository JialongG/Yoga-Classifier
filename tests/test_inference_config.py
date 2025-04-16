from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from yoga_pose_app.inference import load_inference_config


def test_load_inference_config_has_models() -> None:
    config = load_inference_config()
    assert config.labels_path
    assert config.model_paths
    assert config.image_size[0] > 0
    assert config.image_size[1] > 0
    assert config.top_k > 0
