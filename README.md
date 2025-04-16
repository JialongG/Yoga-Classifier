# Yoga Pose Classifier

## Project Overview

This repository provides a Streamlit-based yoga pose classification application backed by exported EfficientNetB3 TFLite models.

The production entrypoint is the UI + inference service path. Notebook material in `notebooks/` is retained as supporting experimentation history, not as the canonical runtime.

## Core Pipeline

1. `yoga_pose_ui.py` initializes `InferenceService` from `src/yoga_pose_app/inference.py`.
2. `load_inference_config()` reads `configs/inference_config.json` (falls back to `configs/inference_config.example.json`).
3. The service resolves repository-relative label/model paths and loads TFLite interpreters.
4. Uploaded images are converted to RGB, resized, and passed into the selected model.
5. The UI presents top prediction, confidence, latency, and top-k class scores.

## Repository Structure

```text
.
├── yoga_pose_ui.py
├── run_streamlit.sh
├── configs/
│   └── inference_config.example.json
├── scripts/
│   └── smoke_inference.py
├── src/
│   └── yoga_pose_app/
│       ├── __init__.py
│       └── inference.py
├── tests/
│   └── test_inference_config.py
├── assets/
│   ├── models/
│   │   ├── tflite/
│   │   ├── effb3/
│   │   └── best_cnn/
│   └── screenshots/
│       └── ui/
├── data/
│   └── 107_yoga_poses/
├── notebooks/
├── requirements.txt
└── README.md
```

## Installation

- Python `3.11+`
- Create and activate a virtual environment
- Install dependencies:

```bash
pip install -r requirements.txt
```

## Configuration

- Example config: `configs/inference_config.example.json`
- Optional local override: `configs/inference_config.json`

Supported keys:

- `labels_path`: repository-relative label path
- `model_paths`: mapping of display name to repository-relative model path
- `image_size`: `[width, height]`
- `top_k`: number of classes shown in output

## Usage

Run the application:

```bash
streamlit run yoga_pose_ui.py
```

Or run with the provided launcher:

```bash
bash run_streamlit.sh
```

Smoke check (config + required artifacts + model loading):

```bash
python scripts/smoke_inference.py
```

Tests:

```bash
pytest -q
```

