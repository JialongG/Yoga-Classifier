# Dataset

The `data/107_yoga_poses/` directory is where the Yoga-107 image corpus used to
train and evaluate the models under `assets/models/` is expected to live. The
dataset itself is a third-party resource and is **not redistributed** with
this repository; this folder is listed in `.gitignore`.

## Expected layout

```
data/107_yoga_poses/
├── train/<class_name>/*.png
├── val/<class_name>/*.png
└── test/<class_name>/*.png
```

Every split must contain one subdirectory per class, and the class names must
match the entries in `assets/models/tflite/yoga-poses-english.txt` (107
labels, lowercase, one per line).

## Obtaining the data

The models shipped under `assets/models/tflite/` were trained against a
publicly available 107-class yoga pose dataset. Place a compatible copy of
that dataset under the layout above before running any training or
evaluation code. The Streamlit inference app under `yoga_pose_ui.py` does
**not** require this directory and can be exercised with arbitrary user
uploads.

## Git policy

Do not commit raw dataset contents back to the repository. The `data/` entry
in `.gitignore` exists specifically to prevent multi-gigabyte accidental
pushes.
