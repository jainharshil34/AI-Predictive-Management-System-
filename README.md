# ⚡ NeuralGuard — Neural Network-Based Predictive Maintenance

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.13%2B-orange?logo=tensorflow)](https://tensorflow.org)
[![Streamlit](https://img.shields.io/badge/Dashboard-Streamlit-red?logo=streamlit)](https://streamlit.io)
[![Flask](https://img.shields.io/badge/API-Flask%205001-lightgrey?logo=flask)](https://flask.palletsprojects.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-green)](LICENSE)

> Predict CNC machine failures **before they happen** using a hybrid LSTM + Autoencoder deep-learning pipeline on the Honeywell AI4I dataset.

---

## 📌 Overview

NeuralGuard is an end-to-end **AI Predictive Maintenance (PdM)** system that analyses multi-sensor data from industrial CNC machines to forecast equipment failures. The system combines:

- **LSTM + Dot-Attention classifier** — temporal failure probability over 10-timestep windows
- **Autoencoder anomaly detector** — unsupervised anomaly flagging via reconstruction MSE
- **Flask REST API** — real-time inference backend
- **Streamlit dashboard** — 8-page interactive analytics interface

### Key Metrics

| Metric | Value |
|---|---|
| Classification Accuracy | **83%** |
| Fault Recall | **86%** |
| False Alarm Rate | **17.5%** |
| AE Anomaly Threshold (MSE) | **0.025** |
| LSTM Model Size | 293 KB |
| TFLite Quantized | 44 KB (6.6× compression) |

---

## 🏗️ Project Structure

```
NeuralGuard-PdM/
├── deployment/
│   ├── app.py                              # Flask backend (port 5001)
│   ├── dashboard.py                        # Streamlit dashboard
│   ├── requirements.txt                    # Python dependencies
│   ├── best_lstm_attention_model_v2.keras  # LSTM+Attention model
│   ├── autoencoder_model.h5                # Autoencoder model
│   └── scaler.joblib                       # Pre-fitted StandardScaler
├── Notebooks/
│   ├── 01_eda.ipynb                        # Exploratory Data Analysis
│   ├── 02_PREPROCESS_FUSION.ipynb          # Feature engineering & preprocessing
│   ├── 03_anomaly_and_model_design.ipynb   # Model architecture design
│   ├── 04_training_evaluation.ipynb        # Training, evaluation & TFLite export
│   └── 05_xai.ipynb                        # SHAP + Attention explainability
├── Data/
│   ├── ai4i_engineered_features.csv        # Preprocessed dataset (8 features)
│   └── honeywell.csv                       # Raw Honeywell AI4I CNC dataset
├── .gitignore
└── README.md
```

---

## 🚀 Quick Start

### 1. Clone & Install

```bash
git clone https://github.com/Mehakdeep/AI-Predictive-Maintenance-System.git
cd NeuralGuard-PdM/deployment
pip install -r requirements.txt
```

### 2. Start Flask Backend (Terminal 1)

```bash
cd deployment
python app.py
# → Flask API running at http://0.0.0.0:5001
```

### 3. Launch Streamlit Dashboard (Terminal 2)

```bash
cd deployment
streamlit run dashboard.py
# → Dashboard at http://localhost:8501
```

> **Offline mode:** The dashboard works without Flask — it falls back to a local heuristic estimator. Real model predictions require Flask running.

---

## 🧠 Model Pipeline

```
5 Raw Sensors
  └─ Air T · Process T · RPM · Torque · Tool Wear
        │
        ▼
  Feature Engineering           ← Notebook 02
  temp_diff = Proc_T − Air_T
  stress_index = Torque / (RPM + ε)
  torque_wear  = Torque × Tool_wear
        │
        ▼
  StandardScaler (5 raw only)   ← scaler.joblib
        │
        ├──────────────────────────────────────┐
        ▼                                      ▼
  LSTM Sequence Buffer                  Autoencoder
  deque(maxlen=10) → (1,10,8)          (1,8) → reconstruct
        │                                      │
  LSTM(64, return_sequences=True)        MSE > 0.025?
  Dropout(0.3)                                 │
  Dot-Attention (axes=1)               anomaly_flag: bool
  Dense(32, ReLU)
  Sigmoid output
        │
  failure_probability ∈ [0,1]
```

---

## 📡 API Reference

### `POST /predict`

Accepts either **full column names** or **short aliases**.

**Request body (full names):**
```json
{
  "Air temperature [K]": 303.5,
  "Process temperature [K]": 314.2,
  "Rotational speed [rpm]": 1280,
  "Torque [Nm]": 68.5,
  "Tool wear [min]": 220
}
```

**Request body (short aliases):**
```json
{
  "air_t": 303.5, "proc_t": 314.2,
  "rpm": 1280, "torque": 68.5, "wear": 220
}
```

**Response:**
```json
{
  "failure_probability": 0.7823,
  "anomaly_flag": true,
  "anomaly_mse": 0.03841,
  "status": "Active",
  "buffer_size": 10,
  "input_warnings": []
}
```

> `status = "Stabilizing"` means the 10-step buffer is filling — LSTM returns `0.0` until full.

### `GET /health`

Returns model load status and buffer state.

### `POST /reset`

Clears the rolling LSTM buffer. Call this when switching between machines or starting a new monitoring session.

---

## 📊 Dataset

**Honeywell AI4I 2020 Predictive Maintenance Dataset**
- 10,000 records × 14 columns
- 5 failure types: TWF, HDF, PWF, OSF, RNF
- ~3.4% failure rate (class-imbalanced)
- No missing values

### Engineered Features

| # | Feature | Formula | Rationale |
|---|---|---|---|
| 6 | `temp_diff` | `Process_T − Air_T` | Thermal differential |
| 7 | `stress_index` | `Torque / (RPM + ε)` | Mechanical load proxy |
| 8 | `torque_wear` | `Torque × Tool_wear` | Compound degradation |

---

## 🔬 Explainability (XAI)

Notebook `05_xai.ipynb` uses `shap.KernelExplainer` with the LSTM flattened to 2D (80-dim). SHAP values are summed across 10 timesteps per feature.

**Global feature importance (Mean |SHAP|):**

| Feature | Importance |
|---|---|
| Tool wear [min] | 0.38 |
| torque_wear | 0.29 |
| Torque [Nm] | 0.24 |
| stress_index | 0.18 |
| Rotational speed [rpm] | 0.12 |

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Deep Learning | TensorFlow 2.x / Keras |
| Backend API | Flask + flask-cors |
| Dashboard | Streamlit + Plotly |
| ML Utilities | scikit-learn, joblib |
| Explainability | SHAP (KernelExplainer) |
| Edge Deployment | TFLite (post-training quantization) |
| Data | pandas, numpy |

---

## 📓 Notebooks

| Notebook | Description |
|---|---|
| `01_eda.ipynb` | EDA — distributions, failure clusters, correlations |
| `02_PREPROCESS_FUSION.ipynb` | Feature engineering, IQR capping, scaler fit |
| `03_anomaly_and_model_design.ipynb` | Autoencoder design, threshold τ = μ + 2σ |
| `04_training_evaluation.ipynb` | LSTM training, ROC/PR, TFLite export |
| `05_xai.ipynb` | SHAP KernelExplainer, attention visualisation |

---


