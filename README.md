# Melanoma Detector – End‑to‑End Skin‑Cancer Classification

<p align="center">
  <img src="Logo/Dermatology.jpeg"alt="banner"/>
</p>

> **A lightweight MobileNetV2‑based classifier delivered through an interactive Streamlit web app.** Trained on 10 000 dermoscopic images and fine‑tuned to reach **≈ 90 % val‑accuracy**.

---

## 🌟 Project Highlights

| Feature                       | What it means                                                                         |
| ----------------------------- | ------------------------------------------------------------------------------------- |
| **MobileNetV2 fine‑tuned**    | Fast inference on CPU‑only back‑ends (Streamlit Cloud, Heroku, low‑spec laptops).     |
| **End‑to‑end code**           | From raw dataset → data loaders → training notebook → web app → one‑click deploy.     |
| **Reproducible environment**  | Pinned `python‑3.9` + `tensorflow‑cpu==2.16.2` – wheels exist for all platforms.     |
| **Model card & explanations** | Clear description of dataset splits, metrics, limitations and ethical considerations. |
| **CI hook**                   | Optional GitHub Action to smoke‑test the app on every push (disabled by default).     |

---

## 🔗 Live Demo

> **Streamlit Cloud** → [https://melanoma-cancer-detector.streamlit.app](https://melanoma-cancer-detector.streamlit.app) *(opens in same tab – right-click to open in new tab)*

Feel free to upload your own dermoscopic image *(⚠️ non‑identifiable images only!)* and see the predicted probability.

---

## 🗂️ Repository Structure

```
melanoma-detector/
├─ app.py                     # Streamlit front‑end
├─ melanoma_mobilenetv2_finetuned.h5  # 12 MB model weights (MobileNetV2)
├─ melanoma_dataset/          # raw & pre‑processed images (≈ 200 MB after pruning)
├─ notebooks/
│   └─ Melanoma_Detection.ipynb  # training & evaluation notebook
├─ requirements.txt           # runtime deps – pinned for reproducibility
├─ runtime.txt                # python version hint for Streamlit
├─ .github/
│   ├─ workflows/deploy.yml   # optional CI deploy action
│   └─ banner.png             # repo header image
└─ README.md                  # you are here
```

---

## 🚀 Quick Start (Local)

```bash
# 1) Clone & enter repo
git clone https://github.com/varghesebibin/melanoma-detector.git
cd melanoma-detector

# 2) Create Python 3.10 virtual‑env
python3.10 -m venv .venv
source .venv/bin/activate

# 3) Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# 4) Launch the app
streamlit run app.py
```

The first run may take \~30 s while TensorFlow initialises.

> **Tip:** To re‑train the model, open `notebooks/Melanoma_Detection.ipynb` *(Jupyter/Colab)* and run the cells. Trained weights are saved as `melanoma_mobilenetv2_finetuned.h5`.

---

## ☁️  One‑Click Deploy to Streamlit Cloud

1. Push your fork to GitHub (model < 100 MB or use \[Git LFS]).
2. Log‑in to [https://streamlit.io/cloud](https://streamlit.io/cloud), click **New app** → select repo & `main` branch.
3. Ensure **Advanced → “Clear build cache”** is ticked the first time.
4. Wait for the log to show:

   ```
   Successfully installed tensorflow-cpu‑2.16.2‑cp310‑...manylinux...
   🎈  Your app is live!
   ```
5. Share the public URL with your users.

---

## 🏋🏽‍♂️  Training Pipeline

| Stage                    | File / Notebook section                                                                  |
| ------------------------ | ---------------------------------------------------------------------------------------- |
| **Data ingestion**       | `notebooks/01_data_preparation` – loads ISIC‑2020 subset, removes duplicates & corrupts. |
| **Augmentation**         | Random flips, rotations, colour‑jitter via `tf.image` / `keras.preprocessing`.           |
| **Transfer‑learning**    | Base = `MobileNetV2(weights="imagenet")` → replace top with GAP + Dense(1, sigmoid).     |
| **Optimiser & schedule** | Adam (1e‑4), early‑stop on `val_loss` patience = 5.                                      |
| **Metrics**              | Accuracy, Precision, Recall, AUC. Final test set accuracy ≈ 85 %.                        |

Full hyper‑params are logged to TensorBoard under `./logs/`.

---

## 📝  Model Card (Short)

**Intended use:** Support dermatologists with preliminary triage and second opinions. ⚠️ Not a substitute for clinical diagnosis.

**Training data:** ~10 000 dermoscopic images sourced from ISIC 2020 and HAM10000. The dataset was balanced between benign and malignant cases to prevent model bias.

**Limitations:** May underperform on:
- Non‑dermoscopic images (e.g. smartphone photos),
- Occluded or poor-quality samples,
- Rare or ambiguous melanoma subtypes.

**Ethical considerations:** This model does **not** replace professional diagnosis. Use only with de-identified, high-quality clinical imagery. Always consult a healthcare provider.

*See [`MODEL_CARD.md`](MODEL_CARD.md) for details.*

---

## 🙌  Contributing

PRs and issues are welcome!  Please open an issue describing the bug / feature first.  When adding code:

```text
├─ feature‑branch
│  ├─ your_code.py
│  └─ tests/test_your_code.py     <- add a minimal test   
```

Run `pytest` locally before pushing.

---

## 📜 License

This project is released under the **MIT License** – see [`LICENSE`](LICENSE) for details.

---

## 👍 Acknowledgements

* ISIC Archive & HAM10000 researchers for the open dataset.
* Keras Team for the high‑level API.
* Streamlit for the free community cloud hosting.

---

> *“Early detection saves lives – this repo is a small step toward accessible screening tools for everyone.”*
# melanoma-detector
