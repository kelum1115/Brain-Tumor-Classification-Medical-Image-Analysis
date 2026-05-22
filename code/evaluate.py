"""
evaluate.py — Model evaluation, confusion matrices, and clinical benchmark.

Clinical benchmark metrics (what actually matters for deployment):
  - Automation rate:    % of cases pipeline handles without doctor
  - Doctor review rate: % routed to review (acceptable cost for safety)
  - Missed tumors:      definitive wrong on a tumor case (zero-tolerance target)
  - Per-class accuracy: F1 breakdown by tumor type
  - Latency:            not measured here; benchmark separately in prod
"""

import os
import random
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix

from config import DEVICE, OPTIMAL_THRESHOLD, MODEL1_PATH
from pipeline import predict_pipeline


# ── Model 1 Evaluation ────────────────────────────────────────────────────────

def evaluate_model1(model, val_loader, test_loader=None):
    """Classification report + confusion matrix for binary screener."""
    model.load_state_dict(torch.load(MODEL1_PATH))
    model.eval()

    for loader, split in [(val_loader, "Val"), (test_loader, "Test")]:
        if loader is None:
            continue
        all_preds, all_labels = [], []
        with torch.no_grad():
            for imgs, labels in loader:
                imgs   = imgs.to(DEVICE)
                logits = model(imgs).squeeze(1)
                preds  = (torch.sigmoid(logits) >= OPTIMAL_THRESHOLD).long()
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        print(f"\nModel 1 — {split} Classification Report")
        print(classification_report(all_labels, all_preds, target_names=["No Tumor", "Tumor"]))
        _plot_confusion_matrix(all_labels, all_preds, ["No Tumor (0)", "Tumor (1)"],
                               f"Model 1 Confusion Matrix — {split} (Threshold={OPTIMAL_THRESHOLD})", "Reds")


# ── Model 2 Evaluation ────────────────────────────────────────────────────────

def evaluate_model2(model2, test_gen):
    """Classification report + confusion matrix for multiclass classifier."""
    test_gen.reset()
    preds      = model2.predict(test_gen, verbose=1)
    y_pred     = np.argmax(preds, axis=1)
    y_true     = test_gen.classes
    cls_names  = list(test_gen.class_indices.keys())

    print("\nModel 2 (VGG16) — Test Classification Report")
    print(classification_report(y_true, y_pred, target_names=cls_names))

    test_loss, test_acc = model2.evaluate(test_gen, verbose=0)
    print(f"Test Loss: {test_loss:.4f} | Test Accuracy: {test_acc:.4f}")

    _plot_confusion_matrix(y_true, y_pred, cls_names,
                           "Model 2 Confusion Matrix (Multi-Class)", "Blues")


def plot_training_history(history):
    """Accuracy and loss curves for Model 2 phase 1 training."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    ax1.plot(history.history["accuracy"],     label="Train",      color="blue")
    ax1.plot(history.history["val_accuracy"], label="Validation", color="orange")
    ax1.set_title("Model Accuracy")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Accuracy")
    ax1.legend()

    ax2.plot(history.history["loss"],     label="Train",      color="blue")
    ax2.plot(history.history["val_loss"], label="Validation", color="orange")
    ax2.set_title("Model Loss")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Loss")
    ax2.legend()

    plt.tight_layout()
    plt.show()


# ── Clinical Pipeline Benchmark ───────────────────────────────────────────────

def clinical_benchmark(test_dir: str, n_samples: int = 50, seed: int = 67):
    """
    Runs pipeline on n_samples from test_dir.
    Reports the metrics that map to clinical deployment value:
      - Automation rate (reduces radiologist workload)
      - Doctor review rate (safety overhead)
      - Missed tumors / definitive incorrect (patient risk)
    """
    all_images = [
        os.path.join(root, f)
        for root, _, files in os.walk(test_dir)
        for f in files if f.lower().endswith(("jpg", "jpeg", "png"))
    ]
    random.seed(seed)
    sample = random.sample(all_images, min(n_samples, len(all_images)))

    definitive_correct   = 0
    definitive_incorrect = 0
    tumor_to_review      = 0
    notumor_to_review    = 0
    stage_counts         = defaultdict(int)

    for img_path in sample:
        actual = os.path.basename(os.path.dirname(img_path))
        img    = Image.open(img_path).convert("RGB")
        result = predict_pipeline(img)
        stage  = result["stage"]
        pred   = result["prediction"].strip().lower()

        stage_counts[stage] += 1

        if stage in ("Final Classification", "Model 1 filtering"):
            if actual in pred or (pred == "no tumor" and actual == "notumor"):
                definitive_correct += 1
            else:
                definitive_incorrect += 1
        else:
            if actual != "notumor":
                tumor_to_review += 1
            else:
                notumor_to_review += 1

        print(f"{actual:12s} → {result['prediction']:50s} | conf={result['confidence']:.3f}")

    total          = len(sample)
    total_reviewed = tumor_to_review + notumor_to_review

    print(f"\n{'─'*60}")
    print(f"Clinical Benchmark  (n={total})")
    print(f"{'─'*60}")
    print(f"Automation rate              : {definitive_correct/total*100:.1f}%  ({definitive_correct}/{total})")
    print(f"Doctor review rate           : {total_reviewed/total*100:.1f}%  ({total_reviewed}/{total})")
    print(f"Definitive incorrect (risk)  : {definitive_incorrect/total*100:.1f}%  ({definitive_incorrect}/{total})")
    print(f"Tumor correctly flagged      : {tumor_to_review/total*100:.1f}%  ({tumor_to_review}/{total})")
    print(f"NoTumor flagged (false hold) : {notumor_to_review/total*100:.1f}%  ({notumor_to_review}/{total})")
    print(f"Stage routing: {dict(stage_counts)}")

    effective = (definitive_correct + total_reviewed) / total * 100
    print(f"Effective safe rate          : {effective:.1f}%  (correct + reviewed)")


# ── Single Image Debug ────────────────────────────────────────────────────────

def test_single_image(image_path: str):
    """Visual prediction check for a single MRI image."""
    img    = Image.open(image_path).convert("RGB")
    result = predict_pipeline(img)

    plt.figure(figsize=(6, 6))
    plt.imshow(img)
    plt.axis("off")

    color = "green"
    if "No Tumor" not in result["prediction"] and "Tumor" in result["prediction"]:
        color = "red"
    if "Uncertain" in result["prediction"]:
        color = "orange"

    plt.title(
        f"Prediction: {result['prediction']}\nConfidence: {result['confidence']:.2%}",
        fontsize=14, color=color, fontweight="bold",
    )
    plt.show()

    print("Pipeline Metadata:")
    for k, v in result.items():
        print(f"  {k}: {v}")


# ── Internal Helpers ──────────────────────────────────────────────────────────

def _plot_confusion_matrix(y_true, y_pred, labels, title, cmap):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap=cmap,
                xticklabels=labels, yticklabels=labels, annot_kws={"size": 14})
    plt.title(title, fontsize=16, fontweight="bold")
    plt.ylabel("Actual",    fontsize=14)
    plt.xlabel("Predicted", fontsize=14)
    plt.tight_layout()
    plt.show()
