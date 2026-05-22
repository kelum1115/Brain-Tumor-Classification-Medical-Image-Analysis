# Brain-Tumor-Classification-Medical-Image-Analysis
Deep learning-based brain tumor detection and classification system using MRI scans. Implements a two-stage CNN pipeline with VGG16 and Streamlit for real-time prediction of Glioma, Meningioma, Pituitary, and No Tumor cases.

## Overview
Brain tumors can significantly impair neurological functions such as memory, mobility, speech, and sensory processing. Early and accurate diagnosis is essential for effective treatment and improved patient outcomes.

This project introduces an AI pipeline for:
1. Binary Classification — Detects whether an MRI scan contains a tumor  
2. Multi-Class Classification — Classifies tumors into:
   - Glioma  
   - Meningioma  
   - Pituitary Tumor  
   - No Tumor  

A Streamlit web app enables real-time MRI predictions with confidence scores.

---

## Features
- Two-stage CNN architecture for improved reliability
- MRI image preprocessing and augmentation
- Binary tumor detection model
- Multi-class tumor classification model
- VGG16-based classifier
- Entropy-based uncertainty gate — ambiguous predictions routed to "Doctor Review" instead of forcing a confident wrong answer
- Real-time Streamlit web interface
- Evaluation using accuracy, precision, recall, and F1-score

---

## Dataset
Brain Tumor MRI Dataset (Kaggle)

### Classes
- Glioma
- Meningioma
- Pituitary
- No Tumor

### Dataset Size
- Total images: ~7,200 MRI scans  
- Training: 5,600 images (1,400 per class)  
- Testing: 1,600 images (400 per class)

The dataset is balanced, improving model reliability and reducing bias.

---

## Methodology

### Preprocessing
- Image resizing
- Normalization
- Data augmentation

### Model Pipeline

#### Stage 1: Binary CNN (PyTorch)
- Detects tumor vs no tumor
- Optimized for high recall (minimizing missed tumors)
- Threshold set to 0.2 for safety-first design

#### Stage 2: Multi-Class CNN (VGG16 / TensorFlow)
- Classifies tumor type
- Trained on full dataset (7,200 images)
- Two-phase training: frozen base → fine-tune block5
- Entropy-based OOD gate flags uncertain predictions for Doctor Review

---

## Results

### Binary Model
- 100% tumor recall
- Safety-first design to avoid false negatives

### Multi-Class Model
| Class        | F1-Score |
|--------------|----------|
| Pituitary    | 0.92     |
| No Tumor     | 0.91     |
| Glioma       | 0.83     |
| Meningioma   | 0.76     |

### Key Insights
- High performance on Pituitary and No Tumor classes
- Glioma and Meningioma are harder due to visual similarity
- Ambiguous cases are flagged for expert review

---

## Setup

```bash
pip install -r requirements.txt
```

## Run the App

```bash
streamlit run code/app.py
```

## Retrain Models

```bash
python code/train.py
```

---

## Tech Stack
- Python
- PyTorch
- TensorFlow / Keras
- CNN / VGG16
- OpenCV
- NumPy
- Matplotlib
- Streamlit
