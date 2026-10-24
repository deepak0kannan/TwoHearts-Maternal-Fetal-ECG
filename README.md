# TwoHearts: Multi-Task Deep Learning for Maternal-Fetal Cardiac Screening & Autonomic Triage

[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red.svg)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115%2B-green.svg)](https://fastapi.tiangolo.com/)
[![ROC-AUC](https://img.shields.io/badge/Fetal%20Arrhythmia%20ROC--AUC-0.9844-teal.svg)]()
[![Specificity](https://img.shields.io/badge/Diagnostic%20Specificity-97.80%25-brightgreen.svg)]()
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)]()

> **TwoHearts** is an explainable, multi-task deep learning platform for simultaneous non-invasive fetal electrocardiogram (NI-fECG) arrhythmia screening and maternal autonomic stress tele-triage in resource-limited rural Primary Health Centres (PHCs).

---

## ⚡ Key Innovations & Highlights
- **End-to-End Learning:** Ingests 4-channel cutaneous abdominal ECG signals directly, eliminating fragile blind source separation (FastICA) polarity ambiguity, channel permutation, and peak loss failures.
- **Multi-Scale Temporal Stem:** Parallel 1D convolutional branches with receptive fields $k \in \{7, 15, 31\}$ simultaneously capture narrow 40 ms fetal QRS spikes and broad maternal P-QRS-T complexes.
- **4-Stage SE-ResNet-1D Backbone:** Squeeze-and-Excitation channel attention dynamically re-weights lead importance, suppressing maternal baseline wander and motion artifacts while boosting subtle fetal electrical dipoles.
- **Dual Multi-Task Prediction Heads:**
  - **Head 1 (Classification):** Fetal Arrhythmia Classifier ($\text{ROC-AUC} = \mathbf{0.9844}$, $\text{F1} = \mathbf{0.9249}$, $\text{Specificity} = \mathbf{97.80\%}$).
  - **Head 2 (Regression):** Maternal Autonomic Stress Regressor (Continuous Baevsky Stress Index & RMSSD tracking, $\text{MAE} = \mathbf{0.1703}$).
- **Explainable AI (1D Grad-CAM):** Millisecond-aligned temporal attention heatmaps overlaid on raw multi-lead waveforms ($\text{SAMI} = \mathbf{0.841}$), giving clinicians visual explainability for detected ectopic complexes.
- **Hardware-Accelerated Web Dashboard:** 60 FPS HTML5 Canvas medical oscilloscope with calibrated $25\text{ mm/s}$ ECG grid lines, interactive window navigation, patient triage directives, and printable diagnostic PDF reports.

---

## 📂 Repository Structure

```
TwoHearts/
├── app.py                             # FastAPI clinical tele-triage dashboard server
├── requirements.txt                   # Production Python package requirements
├── README.md                          # Project documentation and setup guide
├── LICENSE                            # MIT Open Source License
│
├── templates/
│   └── index.html                     # Clinical dashboard user interface
│
├── static/
│   ├── css/
│   │   └── style.css                  # Zero-CDN clinical styling & responsive theme
│   └── js/
│       ├── app.js                     # Dashboard interaction, metrics & triage logic
│       └── ecg_canvas.js              # 60 FPS Canvas multi-lead ECG renderer
│
├── TwoHearts_DL/                      # Core Deep Learning Package
│   ├── run_pipeline.py                # End-to-end 5-fold CV training & evaluation script
│   ├── src/
│   │   ├── dataset.py                 # Multi-channel data loader & window segmenter
│   │   ├── model.py                   # TwoHeartsNet (Multi-scale stem + SE-ResNet-1D + Dual Heads)
│   │   ├── train.py                   # Multi-task training engine with Cosine Annealing
│   │   ├── evaluate.py                # Patient-level stratified CV evaluation suite
│   │   └── explainability.py          # 1D Grad-CAM engine & temporal interpolation
│   ├── tests/
│   │   ├── test_model.py              # Unit tests for tensor shapes & forward passes
│   │   └── test_dataset.py            # Unit tests for preprocessing & window segmentation
│   └── results/
│       ├── checkpoints/               # Trained PyTorch model weights (Folds 1 to 5)
│       │   ├── twohearts_best_fold1.pt
│       │   └── ...
│       ├── twohearts_samples.npz      # Curated multi-patient demonstration cache (~8.5 MB)
│       ├── patient_summary.csv        # Demographic & clinical metadata for 24 patients
│       └── cv_performance_metrics.csv # 5-fold cross-validation results table
│
├── Preprocessing/                     # Signal Preprocessing & Filtering Notebooks
│   ├── Data_preprocessing.ipynb
│   ├── Data_Preprocessing_Full_Pipeline.ipynb
│   └── fetal_ecg_preprocessing.ipynb
│
└── Feature_extraction/                # Heart Rate Variability (HRV) Feature Extraction
    ├── Fetal_ECG_Feature_Extraction_FIXED_v2.ipynb
    └── fetal_maternal_features_(edited).csv
```

---

## 🚀 Quick Start: Running the Web Dashboard

### 1. Prerequisites
- Python 3.10, 3.11, 3.12, or 3.13 installed.
- Git & modern web browser (Chrome, Edge, Firefox, Safari).

### 2. Environment Setup

```bash
# Clone the repository
git clone https://github.com/deepak0kannan/TwoHearts-Maternal-Fetal-ECG.git
cd TwoHearts-Maternal-Fetal-ECG

# Create a virtual environment
python -m venv .venv

# Activate the virtual environment
# On Windows:
.\.venv\Scripts\activate
# On Linux / macOS:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Start the Server

```bash
python app.py
```
*The server will start at: `http://127.0.0.1:8000`*

### 4. Access the Dashboard
Open your web browser and navigate to:  
👉 **`http://127.0.0.1:8000`**

- **Explore Patients:** Select any patient from the dropdown (`ARR_01` for critical arrhythmia, `NR_01` for normal sinus rhythm) for instant analysis.
- **Navigate Windows:** Use the window slider to inspect consecutive 10-second segments.
- **Explainable AI:** View synchronized 1D Grad-CAM saliency heatmaps highlighting ectopic complexes.
- **Custom Uploads:** Upload your own multi-lead ECG recordings (`.dat` + `.hea`, `.npz`, or `.csv`) via the file manager.

---

## 📊 Benchmark Validation Results (5-Fold Patient-Stratified CV)

Evaluated across the PhysioNet Non-Invasive Fetal ECG Arrhythmia database (24 patients, 3,304 windows):

| Metric | Score | Clinical Interpretation |
| :--- | :---: | :--- |
| **ROC-AUC** | **0.9844** (98.44%) | Exceptional discriminatory power across threshold range |
| **F1-Score** | **0.9249** (92.49%) | Strong harmonic balance between precision and sensitivity |
| **Accuracy** | **91.95%** | High window-level diagnostic reliability |
| **Specificity** | **97.80%** | Minimal false alarms; prevents unnecessary emergency transfers |
| **Precision** | **97.20%** | 97.2% of flagged arrhythmia alerts are true cases |
| **Sensitivity (Recall)** | **88.16%** | Reliable detection of lethal fetal conduction blocks & ectopy |
| **Maternal Stress MAE** | **0.1703** | Accurate tracking of maternal autonomic strain |
| **Patient-Level Diagnosis** | **91.67%** (22/24) | Majority-consensus triage over full 10-minute records |

---

## 🔬 Reproducing Full Training & Benchmarks

To retrain the 5-fold cross-validation models from scratch:

1. Download the PhysioNet [Non-Invasive Fetal ECG Arrhythmia Database](https://physionet.org/content/nifeadb/1.0.0/) into `non-invasive-fetal-ecg-arrhythmia-database-1.0.0/`.
2. Run the end-to-end training pipeline:
   ```bash
   python TwoHearts_DL/run_pipeline.py
   ```
3. To run unit tests:
   ```bash
   pytest TwoHearts_DL/tests/
   ```

---

## 📜 License
This project is licensed under the MIT License - see the LICENSE file for details.
