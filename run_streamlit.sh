#!/bin/bash
# Fix TensorFlow threading issues on macOS
export TF_CPP_MIN_LOG_LEVEL=2
export TF_ENABLE_ONEDNN_OPTS=0
export OMP_NUM_THREADS=1
export TF_NUM_INTEROP_THREADS=1
export TF_NUM_INTRAOP_THREADS=1

# Activate virtual environment and run streamlit
cd "$(dirname "$0")"
source venv/bin/activate
streamlit run yoga_pose_ui.py

