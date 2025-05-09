#!/usr/bin/env python3
"""CLI entrypoint for training pipeline using EfficientNetB3 and FFN."""

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

from yoga_pose_app.training_config import TrainingConfig
from yoga_pose_app.training_pipeline import artifacts_to_dict, run_training_pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train EfficientNetB3 transfer + fine-tune (30/20/40) and CNN baseline."
        )
    )
    parser.add_argument(
        "--dataset-root",
        required=True,
        help="Root directory with train/ val/ test/ subfolders.",
    )
    parser.add_argument(
        "--output-dir",
        default="artifacts/training_runs",
        help="Directory for saved models, logs, and metrics.",
    )
    parser.add_argument("--epochs", type=int, default=90, help="Training epochs.")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size.")
    parser.add_argument(
        "--unfreeze-variants",
        nargs="+",
        type=int,
        default=[30, 20, 40],
        help="Fine-tuning unfreeze layer counts.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable info-level logs.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    config = TrainingConfig(
        batch_size=args.batch_size,
        epochs=args.epochs,
        unfreeze_variants=tuple(args.unfreeze_variants),
    )
    artifacts = run_training_pipeline(
        dataset_root=args.dataset_root,
        output_dir=args.output_dir,
        config=config,
    )
    print(json.dumps(artifacts_to_dict(artifacts), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

