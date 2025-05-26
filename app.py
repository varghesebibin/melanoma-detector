import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os

# Set page config
st.set_page_config(page_title="Melanoma Detector", page_icon="🩺", layout="centered")

# Load the model
model = load_model("melanoma_mobilenetv2_finetuned.h5")

# Title and description
st.title("🩺 Melanoma Skin Cancer Detector")
st.markdown("""
Upload a skin lesion image and this AI model will predict whether it is **Benign** or **Malignant**.  
The model is fine-tuned from MobileNetV2 and trained on 10,000 labeled melanoma images.
""")

# Sample test buttons
st.markdown("### 🧪 Try a sample image:")
col1, col2 = st.columns(2)

if col1.button("Use Sample Benign Image"):
    uploaded_file = "sample_images/benign.jpg"
elif col2.button("Use Sample Malignant Image"):
    uploaded_file = "sample_images/malignant.jpg"
else:
    uploaded_file = st.file_uploader("📤 Upload your own image (jpg/png)", type=["jpg", "jpeg", "png"])

# Predict if file is uploaded
if uploaded_file:
    # Load and display image
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess
    image = image.resize((224, 224))
    image_array = img_to_array(image)
    image_array = preprocess_input(image_array)
    image_array = np.expand_dims(image_array, axis=0)

    # Predict
    prediction = model.predict(image_array)[0][0]
    label = "Malignant" if prediction >= 0.5 else "Benign"
    confidence = prediction if prediction >= 0.5 else 1 - prediction

    # Results
    st.markdown(f"### 🔍 Prediction: **{label}**")
    st.markdown(f"### 📊 Confidence: `{confidence:.2f}`")

    # Visual confidence bar
    fig, ax = plt.subplots(figsize=(6, 0.4))
    bar_color = "red" if label == "Malignant" else "green"
    ax.barh([0], [confidence], color=bar_color)
    ax.set_xlim(0, 1)
    ax.set_yticks([])
    ax.set_xticks([0.0, 0.5, 1.0])
    ax.set_title("Confidence Level")
    st.pyplot(fig)

    # Warning if confidence is low
    if 0.4 <= confidence <= 0.6:
        st.warning("⚠️ The model is not confident about this prediction. Please consult a doctor.")

    # Downloadable result
    result_text = f"Prediction: {label}\nConfidence: {confidence:.2f}"
    st.download_button("📥 Download Result", result_text, file_name="melanoma_result.txt")

    # Info box
    with st.expander("🧬 What does this mean?"):
        if label == "Malignant":
            st.error("Melanoma is a dangerous form of skin cancer. This prediction suggests the lesion may be malignant. Please consult a certified dermatologist immediately.")
        else:
            st.success("This lesion appears benign. However, if you have concerns, consider clinical evaluation.")