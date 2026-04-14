# Hybrid Ensemble Intelligence for Multi-Domain ECG Arrhythmia Classification
[![Project Status: Active](https://img.shields.io/badge/Project%20Status-Active-brightgreen.svg)](https://github.com/HemachandRavulapalli/ECG-Arrhythmia-Classification)
[![Python 3.12](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![TensorFlow 2.15](https://img.shields.io/badge/TensorFlow-2.15-orange.svg)](https://www.tensorflow.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A high-precision, research-grade ECG classification system utilizing a **Triple-Branch Hybrid Ensemble** (CNN-1D, CNN-2D Spectrogram, and Handcrafted ML) to detect common arrhythmias with clinical-grade accuracy.

---

## 🛠️ Integrated Model Intelligence

Our architecture captures both temporal rhythms and spectral signatures of cardiac signals through a sophisticated ensemble approach.

### The Triple-Branch Architecture:
- **Temporal Rhythm Analysis**: Deep Residual CNN-1D processing for rhythmic anomalies.
- **Spectral Feature Mapping**: CNN-2D analysis of log-power spectrograms for frequency-domain signatures.
- **Clinical Morphologic Features**: Extraction of 16 research-grade HRV, morphologic, and statistical features processed by an XGBoost/Random Forest ensemble.

---

## 📊 Performance Benchmarks (Digital Test Set)

The model was validated on a high-fidelity patient-wise split of standard digital datasets (MIT-BIH & PTB-XL).

| Metric | Value |
| :--- | :--- |
| **Accuracy** | **86.44%** |
| **AUC-ROC** | **97.52%** |
| **Specificity** | **94.77%** |

### Per-Class Precision breakdown:
- **Normal Sinus Rhythm**: High Precision rhythmic detection.
- **Atrial Fibrillation**: Spectral signature identification.
- **Arrhythmia Classes**: Categorization across Bradycardia, Tachycardia, and Ventricular anomalies.

---

## 👨‍🔬 Feature Engineering (16-Dim Handcrafted)

High-precision features representing cardiac health:
- **HRV**: RMSSD, RR-Std, RR-Mean.
- **Morphology**: QRS-width, Peak Amplitude, PR-Proxy.
- **Spectral**: Spectral Entropy, Band Energies (LF, MF, HF).
- **Statistical**: Skewness, Kurtosis, Signal Energy.

---

## Deployment (Azure VPS Quick Start)

### Installation
```bash
git clone https://github.com/HemachandRavulapalli/ECG-Arrhythmia-Classification.git
cd ECG-Arrhythmia-Classification
pip install -r requirements.txt
```

### Starting the System
The system is optimized for Azure VPS deployment with discrete start scripts:

```bash
# Start Backend (FastAPI on Port 8000)
bash start_backend.sh

# Start Frontend (React/Vite on Port 3002)
bash start_frontend.sh
```

---

## 📦 Model Files
This repository includes the fully trained models (managed via Git LFS). Ensure you have `git-lfs` installed to retrieve weights correctly.

- **Location**: `backend/src/saved_models/`
- **Models**: KNN, Random Forest, SVM, XGBoost, CNN-1D, and CNN-2D.

---

## 👨‍🔬 Authors
- **Hemachand Ravulapalli** - *Lead Researcher & Developer*
- **Hemachand Ravulapalli** - *Researcher & Developer*

---
© 2026 ECG Classification Major Project. All Rights Reserved.
