# Yoga Pose Classifier

## Project Overview

This repository provides:

- an inference/UI path (`yoga_pose_ui.py` + `src/yoga_pose_app/inference.py`) for serving exported EfficientNetB3 TFLite models;
- a training path (`scripts/train_models.py` + `src/yoga_pose_app/training_*`) that implements data preprocessing, transfer learning, and fine-tuning variants;
- a pose-estimation path (`scripts/train_pose_ffn.py` + `src/yoga_pose_app/pose_pipeline.py`) that extracts YOLO11s-pose keypoints and trains an FFN classifier.

## Core Pipeline

1. `yoga_pose_ui.py` initializes `InferenceService` from `src/yoga_pose_app/inference.py`.
2. `load_inference_config()` reads `configs/inference_config.json` (falls back to `configs/inference_config.example.json`).
3. The service resolves repository-relative label/model paths and loads TFLite interpreters.
4. Uploaded images are converted to RGB, resized, and passed into the selected model.
5. The UI presents top prediction, confidence, latency, and top-k class scores.

Training pipeline:

1. `scripts/train_models.py` resolves dataset root and CLI hyperparameters.
2. `src/yoga_pose_app/training_data.py` builds RGB directory datasets with prefetch.
3. `src/yoga_pose_app/training_models.py` builds:
   - EfficientNetB3 transfer model (phase-1 head training),
   - fine-tune variants with configurable layer unfreezing (default: 30/20/40),
   - CNN baseline comparator.
4. `src/yoga_pose_app/training_pipeline.py` runs phase-1 training, each fine-tune run, CNN training, evaluation, and FP16 TFLite export.
5. Artifacts are written under `artifacts/training_runs/<timestamp>/`.

Pose-estimation pipeline:

1. `scripts/train_pose_ffn.py` configures dataset root, YOLO model, and FFN training settings.
2. `src/yoga_pose_app/pose_pipeline.py` loads YOLO11s-pose and extracts 17x2 keypoints from the largest detected person in each image.
3. Extracted keypoints are normalized (hip-centered + shoulder-scale normalization) and flattened to fixed 34D vectors.
4. A feed-forward network is trained on train/val keypoint vectors and evaluated on val/test splits.
5. Features, class names, model, and metrics are written under `artifacts/pose_runs/<timestamp>/`.

## Repository Structure

```text
.
├── yoga_pose_ui.py
├── run_streamlit.sh
├── configs/
│   └── inference_config.example.json
├── scripts/
│   ├── smoke_inference.py
│   ├── train_models.py
│   └── train_pose_ffn.py
├── src/
│   └── yoga_pose_app/
│       ├── __init__.py
│       ├── inference.py
│       ├── pose_pipeline.py
│       ├── training_config.py
│       ├── training_data.py
│       ├── training_models.py
│       └── training_pipeline.py
├── tests/
│   ├── test_inference_config.py
│   ├── test_inference_core.py
│   └── test_pose_pipeline_core.py
├── assets/
│   ├── labels/
│   │   └── yoga-poses-english.txt
│   ├── models/
│   │   └── tflite/
│   └── screenshots/
│       └── ui/
├── notebooks/
│   └── yoga_classifier_notebook.ipynb
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

The training/validation/test imagery used to produce the shipped models is a third-party Yoga-107 corpus from Kaggle and is not redistributed here. Dataset link: [Yoga Pose Image Classification Dataset](https://www.kaggle.com/datasets/shrutisaxena/yoga-pose-image-classification-dataset). See [`data/README.md`](data/README.md) for the expected on-disk layout and setup guidance. The Streamlit inference app does not require the dataset to run.

Quick download example:
```bash
kaggle datasets download -d shrutisaxena/yoga-pose-image-classification-dataset -p data && unzip -o data/yoga-pose-image-classification-dataset.zip -d data
```

## Entry Scripts

- `yoga_pose_ui.py`: Streamlit inference application entrypoint.
- `scripts/smoke_inference.py`: validates config, labels, and model loadability.
- `scripts/train_models.py`: training/fine-tuning export pipeline.
- `scripts/train_pose_ffn.py`: YOLO keypoint extraction + FFN classifier training pipeline.

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

Training run:

```bash
python scripts/train_models.py \
  --dataset-root data/107_yoga_poses \
  --epochs 90 \
  --unfreeze-variants 30 20 40 \
  --verbose
```

The training command requires the Kaggle Yoga-107 dataset downloaded locally (see the Data section above).

Pose-estimation + FFN run:

```bash
python scripts/train_pose_ffn.py \
  --dataset-root data/107_yoga_poses \
  --yolo-model-path yolo11s-pose.pt \
  --epochs 120 \
  --verbose
```

The first run may download YOLO weights if they are not available locally.

Tests:

```bash
pytest -q
```

## Results

Three modelling approaches were evaluated on the Yoga-107 test split. The fine-tuned EfficientNetB3 variant with the last 40 backbone layers unfrozen is the selected model and is the configuration shipped as TFLite for the Streamlit UI.

| Approach | Test Loss | Test Accuracy | Inference Latency (s/img) |
|---|---:|---:|---:|
| **EfficientNetB3, fine-tune last 40 layers** (selected) | **1.3490** | **0.6463** | 0.013477 |
| CNN baseline (trained from scratch) | 2.0747 | 0.5779 | 0.001468 |
| YOLO11s-pose keypoints + FFN classifier | 2.5020 | 0.4857 | — |

Relative accuracy improvement of the selected model:

- **+11.83%** over the CNN baseline
- **+33.06%** over the YOLO-pose + FFN pipeline

### Why this approach

- **Representation quality.** ImageNet-pretrained EfficientNetB3 provides stronger visual features than a CNN trained from scratch on a 107-class, moderately sized dataset, and captures pose-irrelevant visual context (clothing, background, limb appearance) that the 17-keypoint skeleton used by the YOLO-pose path discards.
- **Two-stage training.** Phase 1 freezes the backbone and trains a custom classification head on top of the pretrained features; phase 2 unfreezes the last *N* backbone layers and fine-tunes end-to-end at a lower learning rate. Running both phases is what closes the accuracy gap against the baselines.
- **Unfreezing sweep.** The `{20, 30, 40}`-layer variants are all trained and exported; the 40-layer configuration yielded the best test metrics and is listed first in `configs/inference_config.example.json`. All three TFLite exports are shipped so the UI can compare them side by side.

### Trade-offs

- **Compute.** Fine-tuning a pretrained backbone in two phases is the most expensive training path in the project; a mid-range GPU (e.g., Colab T4) is recommended for reproducing the full sweep.
- **Inference latency.** The EfficientNetB3 TFLite path is roughly **9.18× slower** per image than the CNN baseline (≈13.5 ms/img vs. ≈1.5 ms/img on CPU). This remains well within interactive response times for a single-image UI.
- **Design choice.** Because the target deployment is an interactive classifier rather than a real-time pipeline, predictive accuracy was prioritized over inference throughput.

## Model Artifacts

Exported TFLite models under `assets/models/tflite/` are approximately 21 MB each. Training-time exports from `scripts/train_models.py` are generated under `artifacts/training_runs/` (gitignored by default). A `.gitattributes` file declares model binaries (`*.tflite`, `*.keras`, `*.h5`, `*.onnx`, `*.pb`) as Git LFS targets for future commits.

```bash
git lfs install
git lfs migrate import --include="*.tflite,*.keras,*.h5" --everything
```

## Attribution

- **Backbone**: [EfficientNetB3](https://arxiv.org/abs/1905.11946) via `keras.applications`, initialized from ImageNet weights and fine-tuned for 107 yoga pose classes.
- **Pose estimator**: [Ultralytics YOLO pose](https://docs.ultralytics.com/tasks/pose/) (`yolo11s-pose`) for keypoint extraction in the yolo-pose pipeline.
- **Dataset**: a third-party Yoga-107 image corpus from Kaggle, not redistributed with this repository. See [`data/README.md`](data/README.md).
- **Academic context**: the training workflow and UI integration was developed as part of a rapidly executed academic project.
