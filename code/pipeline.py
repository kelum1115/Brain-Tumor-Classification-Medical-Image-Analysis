import torch
import numpy as np
import cv2
import streamlit as st
from PIL import Image

from tensorflow.keras.models import load_model
from tensorflow.keras.applications.vgg16 import preprocess_input
from torchvision import transforms

from config import DEVICE, OPTIMAL_THRESHOLD, MODEL1_PATH, MODEL2_H5_PATH

# -----------------------------
# Settings
# -----------------------------

# Must match training order: glioma=0, meningioma=1, notumor=2, pituitary=3
# "notumor" here is Model 2's second-pass filter for leakage from Model 1
CLASS_LABELS = ["Glioma", "Meningioma", "No Tumor", "Pituitary"]

# -----------------------------
# Cached Model Loaders
# @st.cache_resource: loads once per session, survives reruns
# Without this, Streamlit re-executes top-to-bottom on every widget
# interaction, reloading ~500MB of weights each time → unusable latency
# -----------------------------

@st.cache_resource
def load_binary_model():
    model = torch.load(
        MODEL1_PATH,
        map_location=DEVICE,
        weights_only=False
    )
    model.eval()
    return model


@st.cache_resource
def load_multiclass_model():
    return load_model(MODEL2_H5_PATH)


# Same transform used during binary model training
binary_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# -----------------------------
# Image Preprocessing Helpers
# -----------------------------

def crop_brain_contour(image: Image.Image, margin: int = 15) -> Image.Image:
    """Remove non-brain pixels via largest contour bounding box."""
    img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
    _, thresh = cv2.threshold(img_cv, 45, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(
        thresh,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if contours:
        c = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(c)

        img_h, img_w = img_cv.shape
        x_min = max(0, x - margin)
        y_min = max(0, y - margin)
        x_max = min(img_w, x + w + margin)
        y_max = min(img_h, y + h + margin)

        cropped = np.array(image)[y_min:y_max, x_min:x_max]
        return Image.fromarray(cropped)

    return image  # fallback: return original if no contour found


def normalize_mri_contrast(image: Image.Image) -> Image.Image:
    """CLAHE contrast enhancement — reduces scanner variance."""
    img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
    clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(img_cv)
    return Image.fromarray(cv2.cvtColor(enhanced, cv2.COLOR_GRAY2RGB))


# -----------------------------
# Prediction Pipeline
# -----------------------------

def predict_pipeline(image: Image.Image) -> dict:
    """
    Run MRI image through two-stage pipeline.

    Streamlit usage:
        uploaded = st.file_uploader("Upload MRI", type=["jpg", "png"])
        if uploaded:
            image = Image.open(uploaded).convert("RGB")  # convert here, not inside
            result = predict_pipeline(image)

    Returns dict with keys: prediction, confidence, stage,
    and optionally: all_probabilities, class_labels.
    """
    # Load models via cache (no-op if already loaded)
    binary_model     = load_binary_model()
    multiclass_model = load_multiclass_model()

    # Guarantee PIL RGB input
    if not isinstance(image, Image.Image):
        image = Image.fromarray(image)
    image = image.convert("RGB")

    # Preprocessing
    norm_image    = normalize_mri_contrast(image)
    cropped_image = crop_brain_contour(norm_image)

    # -----------------------------
    # Stage 1: Binary PyTorch Model
    # -----------------------------
    binary_input = binary_transform(cropped_image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        binary_logit      = binary_model(binary_input).squeeze(1)
        tumor_probability = torch.sigmoid(binary_logit).cpu().item()

    tumor_probability = float(tumor_probability)

    if tumor_probability < OPTIMAL_THRESHOLD:
        return {
            "prediction": "No Tumor",
            "confidence": round(1 - tumor_probability, 4),
            "stage": "Model 1 Binary Filter"
        }

    # -----------------------------
    # Stage 2: Multi-class Keras Model
    # -----------------------------
    img_array = cv2.resize(np.array(cropped_image), (224, 224))
    img_array = preprocess_input(img_array.astype(np.float32))
    img_array = np.expand_dims(img_array, axis=0)

    probs      = multiclass_model.predict(img_array, verbose=0).squeeze()
    confidence = float(np.max(probs))
    pred_idx   = int(np.argmax(probs))
    prediction = CLASS_LABELS[pred_idx]

    # Entropy gate: catches high-uncertainty cases even when conf > 0.45
    # e.g. glioma=0.47 / meningioma=0.44 — conf passes 0.45 but model is split
    entropy     = float(-np.sum(probs * np.log(probs + 1e-8)))
    max_entropy = float(np.log(len(CLASS_LABELS)))  # log(4)

    # -----------------------------
    # Uncertainty / OOD Gates
    # -----------------------------
    if confidence < 0.45 or entropy > 0.85 * max_entropy:
        return {
            "prediction": "Uncertain → Doctor Review",
            "confidence": round(confidence, 4),
            "stage": "Model 2 Uncertainty OOD Gate",
            "entropy": round(entropy, 4),
            "all_probabilities": probs.tolist(),
            "class_labels": CLASS_LABELS
        }

    if confidence < 0.85:
        return {
            "prediction": f"{prediction} suspected → Doctor Review",
            "confidence": round(confidence, 4),
            "stage": "Model 2 Low Confidence Gate",
            "all_probabilities": probs.tolist(),
            "class_labels": CLASS_LABELS
        }

    return {
        "prediction": prediction,
        "confidence": round(confidence, 4),
        "stage": "Final Tumor Classification",
        "all_probabilities": probs.tolist(),
        "class_labels": CLASS_LABELS
    }
