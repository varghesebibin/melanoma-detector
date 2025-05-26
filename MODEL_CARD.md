# 🧾 Model Card – Melanoma Detector

## Overview
This model performs binary classification to detect melanoma from dermoscopic images. It's designed to support dermatologists with second opinions and is **not** intended for standalone diagnosis.

---

## Intended Use
- Assist clinical decision-making with image classification.
- Educational or demonstration purposes via Streamlit web app.

---

## Dataset
- **Source:** ISIC 2020 Challenge + HAM10000
- **Size:** ~10,000 images
- **Balanced:** Yes — classes (benign vs malignant) were sampled to maintain balance during training.

---

## Training Procedure
- **Base model:** MobileNetV2 (pretrained on ImageNet)
- **Loss function:** Binary Crossentropy
- **Optimizer:** Adam (lr = 1e-4)
- **Epochs:** ~10 with EarlyStopping (patience = 5)
- **Augmentation:** Horizontal/vertical flip, rotation, brightness

---

## Evaluation Metrics
- **Accuracy:** ~90% on validation set
- **Precision/Recall/F1:** Refer to final model notebook
- **ROC-AUC:** Included in training logs

---

## Limitations
- Performance may drop on:
  - Smartphone-captured or non-dermoscopic images
  - Highly occluded or blurry lesions
  - Rare subtypes not present in training data

---

## Ethical Considerations
- This tool is **not a diagnostic instrument**.
- Intended for use by qualified professionals.
- All input data must be de-identified.
