#!/usr/bin/env python3
"""CLI entrypoint for YOLO pose extraction + FFN classifier training."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from yoga_pose_app.pose_pipeline import (
    PoseExtractionConfig,
    PoseTrainingConfig,
    artifacts_to_dict,
    train_pose_pipeline,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train FFN classifier on YOLO11s-pose extracted keypoints."
    )
    parser.add_argument("--dataset-root", required=True, help="Dataset root with train/val/test splits.")
    parser.add_argument("--output-dir", default="artifacts/pose_runs", help="Output directory for pose run artifacts.")
    parser.add_argument("--yolo-model-path", default="yolo11s-pose.pt", help="YOLO pose model path or model name.")
    parser.add_argument("--img-size", type=int, default=640, help="YOLO inference image size.")
    parser.add_argument("--max-images-per-class", type=int, default=None, help="Optional cap per class for faster experiments.")
    parser.add_argument("--epochs", type=int, default=120, help="FFN training epochs.")
    parser.add_argument("--batch-size", type=int, default=32, help="FFN training batch size.")
    parser.add_argument("--verbose", action="store_true", help="Enable info-level logs.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    extraction_config = PoseExtractionConfig(
        yolo_model_path=args.yolo_model_path,
        image_size=args.img_size,
        max_images_per_class=args.max_images_per_class,
    )
    training_config = PoseTrainingConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
    )
    artifacts = train_pose_pipeline(
        dataset_root=args.dataset_root,
        output_dir=args.output_dir,
        extraction_config=extraction_config,
        training_config=training_config,
    )
    print(json.dumps(artifacts_to_dict(artifacts), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

