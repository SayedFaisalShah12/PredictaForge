---
title: PredictaForge
emoji: 🧠
colorFrom: red
colorTo: purple
sdk: streamlit
sdk_version: "1.30.0"
app_file: app.py
pinned: false
short_description: AI-powered breast cancer prediction web application
license: odbl
---

# 🧠 PredictaForge – Breast Cancer Prediction App

PredictaForge is an AI-powered web application designed to assist in the early detection of breast cancer. The app analyzes cell nuclei measurements obtained from cytology tests and predicts whether a tumor is **benign** or **malignant** using machine learning.

> ⚠️ This application is intended for educational and research purposes only and should not replace professional medical diagnosis.

---

## 🚀 Features

- AI-based breast cancer prediction  
- Interactive Streamlit web interface  
- Radar chart visualization of cell features  
- Probability scores for benign and malignant classes  
- Fast and lightweight deployment  
- User-friendly sliders for input data  

---

## 🧪 Dataset Used

**Breast Cancer Wisconsin (Diagnostic) Dataset**  
- Source: UCI Machine Learning Repository  
- Total samples: 569  
- Features: 30 numerical cytology measurements  
- Labels:  
  - **Benign (0)**  
  - **Malignant (1)**  

---

## 🛠️ Tech Stack

| Category | Tools |
|--------|------|
| Programming Language | Python |
| Web Framework | Streamlit |
| Machine Learning | Scikit-learn |
| Data Processing | Pandas, NumPy |
| Visualization | Plotly |
| Model Serialization | Pickle |
| Deployment | Hugging Face Spaces |

---

## Links

Huggingface Link: https://huggingface.co/spaces/Sayed-Shah/PredictaFridge
Streamlit Link: https://sayedfaisalshah12-predictaforge-appmain-bmeihg.streamlit.app/


## 📂 Project Structure

```text
.
├── app/main.py
├── requirements.txt
├── README.md
├── model/
│   ├── model.pkl
│   └── scalar.pkl
├── data/
│   └── data.csv
├── assets/
│   └── style.css
