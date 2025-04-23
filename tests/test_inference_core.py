from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from yoga_pose_app.inference import preprocess_image, topk_labels


def test_preprocess_image_shape_and_dtype() -> None:
    img = Image.new("RGB", (50, 80), color=(10, 20, 30))
    x = preprocess_image(img, image_size=(300, 300))
    assert x.shape == (1, 300, 300, 3)
    assert x.dtype == np.float32


def test_preprocess_image_upcasts_grayscale_to_rgb() -> None:
    gray = Image.new("L", (40, 40), color=128)
    x = preprocess_image(gray, image_size=(16, 16))
    assert x.shape == (1, 16, 16, 3)


def test_topk_labels_orders_by_descending_probability() -> None:
    probs = np.array([0.10, 0.50, 0.30, 0.05, 0.05], dtype=np.float32)
    labels = ["a", "b", "c", "d", "e"]
    result = topk_labels(probs, labels, k=3)
    assert [name for name, _ in result] == ["b", "c", "a"]
    assert result[0][1] == pytest.approx(0.5)


def test_topk_labels_respects_k_smaller_than_num_classes() -> None:
    probs = np.array([0.4, 0.6], dtype=np.float32)
    result = topk_labels(probs, ["x", "y"], k=1)
    assert len(result) == 1
    assert result[0][0] == "y"
    assert result[0][1] == pytest.approx(0.6)
