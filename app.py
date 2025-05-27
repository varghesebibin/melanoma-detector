"""
app.py – Streamlit front-end for the Melanoma Detector
-----------------------------------------------------
✓ sample-image buttons
✓ manual uploader
✓ reset button (no page refresh)
✓ cached model load
"""

import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# -------------------------------------------------- #
# 1️⃣  Page and model
# -------------------------------------------------- #
st.set_page_config(page_title="Melanoma Detector", page_icon="🩺", layout="centered")

@st.cache_resource(show_spinner=False)
def load_trained_model():
    return load_model("melanoma_mobilenetv2_finetuned.h5")

model = load_trained_model()

# -------------------------------------------------- #
# 2️⃣  Session state
# -------------------------------------------------- #
if "image_path" not in st.session_state:
    st.session_state.image_path = None        # str | BytesIO | None
if "prediction" not in st.session_state:
    st.session_state.prediction = None        # (label:str, confidence:float)

def set_image(path):
    """Register a new image & clear old prediction."""
    st.session_state.image_path = path
    st.session_state.prediction = None

def reset():
    """Clear everything so user can choose a new image."""
    st.session_state.image_path = None
    st.session_state.prediction = None

# -------------------------------------------------- #
# 3️⃣  Header
# -------------------------------------------------- #
st.title("🩺 Melanoma Skin-Cancer Detector")
st.markdown(
    """
Upload a skin-lesion image and this AI model will predict whether it is **Benign** or **Malignant**.  
The model is fine-tuned from **MobileNetV2** and trained on 10 000 labelled images.
"""
)

# -------------------------------------------------- #
# 4️⃣  Sample buttons + uploader
# -------------------------------------------------- #
st.markdown("### 🧪 Try a sample image:")
c1, c2 = st.columns(2)

if c1.button("🎯 Sample Benign"):
    set_image("sample_images/benign.jpg")

if c2.button("⚠️ Sample Malignant"):
    set_image("sample_images/malignant.jpg")

st.file_uploader(
    "📤 Upload your own image (jpg / png)",
    type=["jpg", "jpeg", "png"],
    on_change=lambda: set_image(st.session_state.user_upload),
    key="user_upload",
)

# -------------------------------------------------- #
# 5️⃣  Optional reset button
# -------------------------------------------------- #
if st.session_state.image_path:
    st.button("🔄 Upload Another Image", on_click=reset)

# -------------------------------------------------- #
# 6️⃣  Run prediction once per image
# -------------------------------------------------- #
if st.session_state.image_path and st.session_state.prediction is None:
    img = Image.open(st.session_state.image_path).convert("RGB")
    st.image(img, caption="Uploaded Image", use_column_width=True)

    # preprocess
    img_resized = img.resize((224, 224))
    arr = preprocess_input(img_to_array(img_resized))
    arr = np.expand_dims(arr, axis=0)

    raw = float(model.predict(arr, verbose=0)[0][0])
    label = "Malignant" if raw >= 0.5 else "Benign"
    conf  = raw if raw >= 0.5 else 1 - raw
    st.session_state.prediction = (label, conf)

# -------------------------------------------------- #
# 7️⃣  Display result
# -------------------------------------------------- #
if st.session_state.prediction:
    label, conf = st.session_state.prediction

    st.markdown(f"### 🔍 Prediction: **{label}**")
    st.markdown(f"### 📊 Confidence: `{conf:.2f}`")

    # confidence bar
    color = "red" if label == "Malignant" else "green"
    fig, ax = plt.subplots(figsize=(6, 0.4))
    ax.barh([0], [conf], color=color)
    ax.set_xlim(0, 1)
    ax.set_yticks([])
    ax.set_xticks([0.0, 0.5, 1.0])
    ax.set_title("Confidence Level")
    st.pyplot(fig)

    # low-confidence warning
    if 0.4 <= conf <= 0.6:
        st.warning("⚠️ Model not confident – please consult a dermatologist.")

    # download text
    result_txt = f"Prediction: {label}\nConfidence: {conf:.2f}"
    st.download_button("📥 Download Result", result_txt, file_name="melanoma_result.txt")

    # info expander
    with st.expander("🧬 What does this mean?"):
        if label == "Malignant":
            st.error(
                "Melanoma is a dangerous form of skin cancer. "
                "This prediction suggests the lesion may be **malignant**. "
                "Please consult a certified dermatologist immediately."
            )
        else:
            st.success(
                "This lesion appears **benign**. "
                "However, if you have concerns, consider professional evaluation."
            )