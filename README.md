# 🫁 COVID-19 Chest X-Ray Detection System

> **AI-powered chest X-ray classification using Deep Learning (VGG16 Transfer Learning)**  
> Detects COVID-19, Viral Pneumonia, and Normal lung conditions from chest X-ray images.

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-app-name.streamlit.app)
![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?logo=tensorflow)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 📌 Project Overview

A healthcare-focused deep learning project that automatically detects COVID-19 from chest X-ray images. Built as part of a Data Science Mini Project to demonstrate real-world applications of Convolutional Neural Networks (CNNs) in medical imaging.

### 🎯 Objective
Help hospitals:
- ⚡ Reduce diagnosis time
- 🏥 Minimize burden on radiologists
- 🌍 Scale testing across regions with limited radiological expertise

---

## 🖼️ App Preview

| Upload Screen | Prediction Result |
|:---:|:---:|
| Upload a chest X-ray image | AI returns class + confidence gauge |

---

## 🗂️ Dataset

**Source:** [COVID-19 Image Dataset — Kaggle](https://www.kaggle.com/datasets/pranavraikokte/covid19-image-dataset)

| Class | Description |
|---|---|
| **COVID-19** | Confirmed COVID infection |
| **Viral Pneumonia** | Non-COVID lung infection |
| **Normal** | No visible lung abnormality |

```
covid19-image-dataset/
├── train/
│   ├── Covid/
│   ├── Normal/
│   └── Viral Pneumonia/
└── test/
    ├── Covid/
    ├── Normal/
    └── Viral Pneumonia/
```

---

## 🏗️ Project Structure

```
📁 COVID-19-XRay-Detection/
├── 📓 Mini Project-8 Image classification using CNN.ipynb   # Full analysis notebook
├── 🚀 app.py                                                 # Streamlit web app
├── 🧠 covid_detection_model.keras                            # Trained VGG16 model
├── 📋 requirements.txt                                       # Python dependencies
└── 📖 README.md                                              # You are here
```

---

## 🧠 Models Built

| Model | Architecture | Description |
|---|---|---|
| **Model 1** | Basic CNN | Conv2D → MaxPool → Flatten → Dense |
| **Model 2** | VGG16 Transfer Learning | Pre-trained VGG16 + custom head |
| **Model 3** | VGG16 + Augmentation | Transfer learning with ImageDataGenerator |

### 📊 Model Comparison

| Model | Train Acc | Test Acc | F1 Score | Overfitting |
|---|---|---|---|---|
| Basic CNN | — | — | — | — |
| Deep CNN | — | — | — | — |
| ResNet50 | — | — | — | — |
| **VGG16 ✅** | — | — | — | No |

> 🏆 **Best Model:** VGG16 with Transfer Learning + Early Stopping

---

## ⚙️ Tech Stack

| Tool | Purpose |
|---|---|
| `TensorFlow / Keras` | Model building & training |
| `VGG16` | Pre-trained base model |
| `OpenCV / PIL` | Image processing |
| `Streamlit` | Web app deployment |
| `Plotly` | Interactive charts |
| `NumPy / Pandas` | Data manipulation |
| `Matplotlib / Seaborn` | Visualization |
| `Scikit-learn` | Evaluation metrics |

---

## 🚀 Run Locally

### 1. Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
cd YOUR_REPO_NAME
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Launch the Streamlit app
```bash
streamlit run app.py
```

### 4. Open in browser
```
http://localhost:8501
```

---

## 🌐 Deploy to Streamlit Cloud

1. Push this repo to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub repo
4. Set **main file** = `app.py`
5. Click **Deploy** 🎉

---

## 📱 App Features

- 🩻 **Upload** any chest X-ray (JPG/PNG)
- 🧠 **AI Prediction** — COVID / Normal / Viral Pneumonia
- 📊 **Confidence Gauge** — Plotly animated gauge chart
- 📈 **Probability Bars** — All 3 class scores visualized
- 🎨 **Risk Badges** — Color-coded HIGH / MEDIUM / LOW risk
- 🌑 **Dark Theme** — Premium GitHub-style UI
- ℹ️ **Model Info Sidebar** — Architecture details & dataset stats

---

## 🔍 Key Techniques Used

- ✅ Transfer Learning (VGG16)
- ✅ Data Augmentation (ImageDataGenerator)
- ✅ Early Stopping & Model Checkpoints
- ✅ Class Imbalance Handling (class weights)
- ✅ Keras Tuner for Hyperparameter Optimization
- ✅ Confusion Matrix, Classification Report, ROC-AUC
- ✅ One-Hot Encoding & Label Encoding

---

## ⚕️ Medical Disclaimer

> This tool is for **educational and research purposes only**.  
> Predictions are **not a substitute** for professional medical advice, diagnosis, or treatment.  
> Always consult a qualified healthcare provider for medical decisions.

---

## 👤 Author

**Purna** — Data Science Student  
📧 *[Your Email]*  
🔗 *[Your LinkedIn]*  
📁 *[Your Kaggle]*

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).

---

<p align="center">
  Made with ❤️ using TensorFlow · VGG16 · Streamlit
</p>
