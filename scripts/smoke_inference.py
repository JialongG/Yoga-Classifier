#!/usr/bin/env python3
"""Low-cost smoke checks for inference configuration and model loading."""

from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from yoga_pose_app.inference import InferenceService, get_repo_root, load_inference_config


def main() -> int:
    config = load_inference_config()
    repo_root = get_repo_root()

    missing = []
    for model_path in config.model_paths.values():
        if not (repo_root / model_path).exists():
            missing.append(model_path)
    if not (repo_root / config.labels_path).exists():
        missing.append(config.labels_path)

    if missing:
        print("Smoke check failed. Missing required artifacts:")
        for item in missing:
            print(f"- {item}")
        return 1

    service = InferenceService(config)
    model_names = service.available_model_names()
    if not model_names:
        print("Smoke check failed. No models are available from configuration.")
        return 1

    print("Smoke check passed.")
    print(f"- Loaded labels: {len(service.labels)}")
    print(f"- Configured models: {len(model_names)}")
    print(f"- Top-k setting: {service.config.top_k}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
