from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from yoga_pose_app.pose_pipeline import _select_largest_bounding_box, normalize_keypoints


class _DummyTensor:
    def __init__(self, arr: np.ndarray) -> None:
        self._arr = arr

    def cpu(self) -> "_DummyTensor":
        return self

    def numpy(self) -> np.ndarray:
        return self._arr


class _DummyBoxes:
    def __init__(self, xyxy: np.ndarray) -> None:
        self.xyxy = _DummyTensor(xyxy)

    def __len__(self) -> int:
        return int(self.xyxy.numpy().shape[0])


class _DummyResult:
    def __init__(self, xyxy: np.ndarray) -> None:
        self.boxes = _DummyBoxes(xyxy)


def test_select_largest_bounding_box_returns_max_area_index() -> None:
    result = _DummyResult(
        np.array(
            [
                [0, 0, 10, 10],  # 100
                [0, 0, 20, 5],   # 100
                [0, 0, 30, 10],  # 300
            ],
            dtype=np.float32,
        )
    )
    assert _select_largest_bounding_box(result) == 2


def test_normalize_keypoints_outputs_34d_vector() -> None:
    # 17 keypoints with simple grid-like coordinates.
    kpts = np.stack([np.array([i, i + 1], dtype=np.float32) for i in range(17)], axis=0)
    feat = normalize_keypoints(kpts)
    assert feat.shape == (34,)
    assert feat.dtype == np.float32
    assert np.all(np.isfinite(feat))


def test_normalize_keypoints_rejects_wrong_shape() -> None:
    with pytest.raises(Exception):
        normalize_keypoints(np.array([1.0, 2.0], dtype=np.float32))

