"""Streamlit entrypoint for yoga pose inference."""

import io
import sys
from pathlib import Path

import streamlit as st
from PIL import Image

# Make the in-repo package importable without requiring an editable install.
ROOT_DIR = Path(__file__).resolve().parent
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from yoga_pose_app.inference import (
    InferenceConfigError,
    InferenceRuntimeError,
    InferenceService,
    load_inference_config,
)

st.set_page_config(
    page_title="Yoga-107 Classifier (EfficientNetB3 TL)", layout="centered"
)


@st.cache_resource
def load_service_cached() -> InferenceService:
    """Build an :class:`InferenceService` once per Streamlit session.

    The interpreters are expensive to allocate, so ``cache_resource`` keeps
    them resident across reruns triggered by widget interactions.
    """
    config = load_inference_config()
    return InferenceService(config)


st.title("Yoga-107 Classifier (Transfer-learned EfficientNetB3)")
st.caption(
    "Upload a yoga image, choose one exported EfficientNetB3 TFLite model, "
    "and run top-k classification inference."
)

# Configuration and model-loading errors are fatal for the page; surface them
# as visible errors rather than letting Streamlit show a stack trace.
try:
    service = load_service_cached()
except InferenceConfigError as exc:
    st.error(f"Configuration error: {exc}")
    st.stop()
except Exception as exc:
    st.error(f"Failed to initialize inference service: {exc}")
    st.stop()

col_left, col_right = st.columns([1, 1])

with col_left:
    model_name = st.selectbox(
        "Model version", service.available_model_names(), index=0
    )

with col_right:
    uploaded = st.file_uploader(
        "Upload an image (jpg/png)", type=["jpg", "jpeg", "png"]
    )

st.divider()

if uploaded is not None:
    img = Image.open(io.BytesIO(uploaded.read()))
    st.image(img, caption="Uploaded image", use_container_width=True)
    try:
        result = service.predict(model_name=model_name, pil_image=img)
    except InferenceRuntimeError as exc:
        st.error(f"Inference error: {exc}")
        st.stop()
    except Exception as exc:
        st.error(f"Unexpected prediction failure: {exc}")
        st.stop()

    st.markdown(f"### Predicted: **{result.predicted_label}**")
    st.write(f"Inference time: **{result.inference_time_ms:.1f} ms**")
    # ``st.progress`` clamps to [0, 1]; min() protects against rare cases
    # where a model output drifts slightly above 1.0 due to quantization.
    st.progress(min(1.0, result.confidence))
    st.write(f"Confidence: **{result.confidence:.3f}**")

    st.subheader(f"Top {service.config.top_k} classes")
    for name, score in result.top_k_scores:
        st.write(f"- {name}: {score:.3f}")
else:
    st.info("Upload an image to run inference.")
