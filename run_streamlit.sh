#!/bin/bash
# Streamlit launcher tuned for macOS. The exported TF env vars dampen
# oversubscription when TensorFlow is pulled in as the tflite fallback on
# Apple Silicon; Linux users can typically run `streamlit run yoga_pose_ui.py`
# directly and skip this script.
set -e

export TF_CPP_MIN_LOG_LEVEL=2
export TF_ENABLE_ONEDNN_OPTS=0
export OMP_NUM_THREADS=1
export TF_NUM_INTEROP_THREADS=1
export TF_NUM_INTRAOP_THREADS=1

cd "$(dirname "$0")"

if [ ! -f venv/bin/activate ]; then
    echo "Error: venv/bin/activate not found at $(pwd)/venv." >&2
    echo "Create the virtualenv first, for example:" >&2
    echo "  python3.11 -m venv venv" >&2
    echo "  source venv/bin/activate" >&2
    echo "  pip install -r requirements.txt" >&2
    exit 1
fi

# shellcheck disable=SC1091
source venv/bin/activate
streamlit run yoga_pose_ui.py
