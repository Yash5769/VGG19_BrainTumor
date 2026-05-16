# 🧠 Brain Tumor MRI Classification using VGGNet-19

<p align="center">
  <img src="outputs/sample_images.png" alt="Sample MRI Images" width="750"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Framework-PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white"/>
  <img src="https://img.shields.io/badge/Model-VGGNet--19-blueviolet?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/Platform-Google%20Colab-F9AB00?style=for-the-badge&logo=googlecolab&logoColor=white"/>
  <img src="https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white"/>
</p>

---

## 📌 Project Overview

This project implements a deep learning pipeline for **automated brain tumor classification** from MRI scans using a fine-tuned **VGGNet-19** convolutional neural network.

The model classifies MRI images into **4 categories**:

| Class | Description |
|---|---|
| 🔴 Glioma | Tumor arising from glial cells |
| 🟠 Meningioma | Tumor from the meninges (brain lining) |
| 🟣 Pituitary | Tumor in the pituitary gland |
| 🟢 No Tumor | Healthy brain scan |

---

## 🎯 Key Features

- **Transfer Learning** — VGGNet-19 pretrained on ImageNet
- **Two-phase training** — frozen backbone → selective fine-tuning
- **Stratified split** — 70% Train / 15% Validation / 15% Test
- **Full evaluation suite** — Accuracy, Precision, Recall, F1, AUC-ROC, Confusion Matrix
- **Data augmentation** — rotation, flip, color jitter, affine, perspective
- **Class weighting** — handles any class imbalance automatically
- **Colab-ready** — runs on free T4 GPU in ~20 minutes

---

## 🗂️ Dataset

**7,200 brain MRI images** across 4 balanced classes, curated from three public sources:

| Source | Link |
|---|---|
| figshare Brain Tumor Dataset | [figshare.com](https://figshare.com/articles/dataset/brain_tumor_dataset/1512427) |
| SARTAJ Dataset | [kaggle.com/sartajbhuvaji](https://www.kaggle.com/sartajbhuvaji/brain-tumor-classification-mri) |
| Br35H Dataset | [kaggle.com/ahmedhamada0](https://www.kaggle.com/datasets/ahmedhamada0/brain-tumor-detection) |

```
Dataset/
├── Training/
│   ├── glioma/        (1400 images)
│   ├── meningioma/    (1400 images)
│   ├── notumor/       (1400 images)
│   └── pituitary/     (1400 images)
└── Testing/
    ├── glioma/        (400 images)
    ├── meningioma/    (400 images)
    ├── notumor/       (400 images)
    └── pituitary/     (400 images)
```

> The script merges Training + Testing (7200 total) and re-splits them
> with stratification into 70/15/15 to ensure balanced class distribution
> across all three sets.

---

## 🏗️ Model Architecture

```
Input Image (224 × 224 × 3)
        │
        ▼
┌─────────────────────────────┐
│   VGG19 Feature Extractor   │  ← Pretrained on ImageNet
│   (13 conv layers, 5 blocks)│
└─────────────────────────────┘
        │
        ▼
  AdaptiveAvgPool (7×7)
        │
        ▼
     Flatten (25088)
        │
        ▼
  Linear(25088 → 4096)
  BatchNorm → ReLU → Dropout(0.5)
        │
        ▼
  Linear(4096 → 4096)
  BatchNorm → ReLU → Dropout(0.5)
        │
        ▼
  Linear(4096 → 1024)
  BatchNorm → ReLU → Dropout(0.3)
        │
        ▼
  Linear(1024 → 4)
  Softmax Output
```

### Training Strategy

| Phase | Backbone | Learning Rate | Scheduler | Epochs |
|---|---|---|---|---|
| Phase 1 | Frozen | 1e-4 | ReduceLROnPlateau | 15 |
| Phase 2 | Blocks 4 & 5 unfrozen | 1e-5 | CosineAnnealing | 10 |

---

## 📊 Results

### Overall Metrics (Test Set)

| Metric | Value |
|---|---|
| Accuracy | 93.61% |
| Precision (weighted) | 0.9366 |
| Recall (weighted) |  0.9361 |
| F1-Score (weighted) | 0.9355 |
| AUC-ROC (weighted OvR) | 0.9922 |


### Visualizations

<table>
  <tr>
    <td align="center"><b>Loss & Accuracy Curves</b></td>
    <td align="center"><b>Confusion Matrix</b></td>
  </tr>
  <tr>
    <td><img src="outputs/loss_accuracy_curves.png" width="370"/></td>
    <td><img src="outputs/confusion_matrix.png" width="370"/></td>
  </tr>
  <tr>
    <td align="center"><b>AUC-ROC Curves</b></td>
    <td align="center"><b>Per-Class Metrics</b></td>
  </tr>
  <tr>
    <td><img src="outputs/auc_roc_curves.png" width="370"/></td>
    <td><img src="outputs/per_class_metrics.png" width="370"/></td>
  </tr>
</table>

---

## 🚀 Quick Start

### Option 1 — Google Colab (Recommended)

1. Open **`VGG19_BrainTumor_Colab.ipynb`** in Colab
2. `Runtime` → `Change runtime type` → **T4 GPU**
3. Upload dataset to Google Drive:
   ```
   MyDrive/brain-tumor-mri/Training/
   MyDrive/brain-tumor-mri/Testing/
   ```
4. Update paths in **Step 2** cell:
   ```python
   TRAIN_DIR = '/content/drive/MyDrive/brain-tumor-mri/Training'
   TEST_DIR  = '/content/drive/MyDrive/brain-tumor-mri/Testing'
   ```
5. `Runtime` → `Run all`

### Option 2 — Local

```bash
# 1. Clone the repo
git clone https://github.com/your_username/VGGNet19-Brain-Tumor-MRI.git
cd VGGNet19-Brain-Tumor-MRI

# 2. Install dependencies
pip install -r requirements.txt

# 3. Update dataset paths in Config class inside main.py
#    DATASET_DIR = "path/to/Training"
#    EXTRA_DIR   = "path/to/Testing"

# 4. Run
python main.py
```

> ⚠️ Local CPU training is very slow. Google Colab with GPU is strongly recommended.

---

## 📦 Requirements

```
torch >= 2.0.0
torchvision >= 0.15.0
scikit-learn >= 1.2.0
matplotlib >= 3.7.0
seaborn >= 0.12.0
Pillow >= 9.5.0
numpy >= 1.23.0
```

Install all at once:
```bash
pip install -r requirements.txt
```

---

## 📁 Repository Structure

```
VGGNet19-Brain-Tumor-MRI/
│
├── VGG19_BrainTumor_Colab.ipynb   # Google Colab notebook (recommended)
├── main.py                         # Local training script
├── requirements.txt                # Python dependencies
├── README.md
│
└── outputs/
    ├── sample_images.png           # Augmented training samples
    ├── loss_accuracy_curves.png    # Phase 1 + Phase 2 training curves
    ├── confusion_matrix.png        # Raw + normalized confusion matrix
    ├── auc_roc_curves.png          # Per-class ROC curves + micro-avg
    └── per_class_metrics.png       # Accuracy/Precision/Recall/F1 per class
```

---

## 🧪 Evaluation Metrics Explained

| Metric | What it measures |
|---|---|
| **Accuracy** | Overall correct predictions / total predictions |
| **Precision** | Of all predicted positives, how many were actually positive |
| **Recall** | Of all actual positives, how many were correctly identified |
| **F1-Score** | Harmonic mean of Precision and Recall |
| **AUC-ROC** | Area under ROC curve — model's ability to distinguish classes |
| **Confusion Matrix** | Per-class breakdown of correct vs incorrect predictions |

---

## 📋 Assignment Details

| Item | Detail |
|---|---|
| Subject | Deep Learning / Computer Vision |
| Model | VGGNet-19 (CNN) |
| Framework | PyTorch |
| Dataset Split | 70% Train · 15% Val · 15% Test (Stratified) |
| Task | Multi-class Image Classification |

---

## 👤 Author

**Yash Srivastava**  
B.Tech Computer Science Engineering  

---

## 📄 License

This project is for academic purposes. Dataset credits go to the original authors
linked in the Dataset section above.
