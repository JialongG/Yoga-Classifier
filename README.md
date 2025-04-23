# Yoga Pose Classifier

## Project Overview

This repository provides a Streamlit-based yoga pose classification application backed by exported EfficientNetB3 TFLite models. The production entrypoint is the UI + inference service path.

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
│   ├── test_inference_config.py
│   └── test_inference_core.py
├── assets/
│   ├── models/
│   │   └── tflite/
│   └── screenshots/
│       └── ui/
├── data/
│   └── README.md
├── requirements.txt
├── requirements-dev.txt
└── README.md
```

## Installation

- Python `3.11+`
- Create and activate a virtual environment
- Install runtime dependencies:

```bash
pip install -r requirements.txt
```

Install development dependencies (tests):

```bash
pip install -r requirements-dev.txt
```

## Configuration

- Example config: `configs/inference_config.example.json`
- Optional local override: `configs/inference_config.json`

Required keys:

- `labels_path`: repository-relative label path
- `model_paths`: mapping of display name to repository-relative model path
- `image_size`: `[width, height]`
- `top_k`: number of classes shown in output

## Data

The training/validation/test imagery used to produce the shipped models is a third-party Yoga-107 corpus and is not redistributed here. See [`data/README.md`](data/README.md) for the expected on-disk layout and guidance on obtaining the dataset. The Streamlit inference app does not require the dataset to run.

## Usage

Run the application:

```bash
streamlit run yoga_pose_ui.py
```

Or run with the provided macOS launcher:

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

## Model Artifacts

Exported TFLite models under `assets/models/tflite/` are approximately 21 MB each. A `.gitattributes` file declares model binaries (`*.tflite`, `*.keras`, `*.h5`, `*.onnx`, `*.pb`) as Git LFS targets for future commits.

```bash
git lfs install
git lfs migrate import --include="*.tflite,*.keras,*.h5" --everything
```

## Attribution

- **Backbone**: [EfficientNetB3](https://arxiv.org/abs/1905.11946) via `keras.applications`, initialized from ImageNet weights and fine-tuned for 107 yoga pose classes.
- **Dataset**: a third-party Yoga-107 image corpus, not redistributed with this repository. See [`data/README.md`](data/README.md).
- **Academic context**: the training workflow and UI integration was developed as part of a rapidly executed academic project.
