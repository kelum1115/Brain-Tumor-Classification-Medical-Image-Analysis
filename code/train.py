"""
train.py — Orchestrates the full two-stage training pipeline.

Run:
    python train.py

Outputs:
    best_model_full.pth             — Model 1 best checkpoint
    best_model2_phase1.keras   — Model 2 phase 1 best checkpoint
    best_model2_vgg16.keras    — Model 2 fine-tuned final checkpoint
    multiclass_tumor_model.h5  — Legacy format for deployment compatibility
"""

import os
from preprocessing import get_dataset_path, load_binary_loaders, load_keras_generators
from model1 import build_model1, train_model1
from model2 import build_model2, train_model2
from evaluate import evaluate_model1, evaluate_model2, plot_training_history, clinical_benchmark
from config import MODEL2_H5_PATH


def main():
    # ── 1. Data ───────────────────────────────────────────────────────────────
    data_dir = get_dataset_path()

    print("\n=== Loading binary loaders (Model 1) ===")
    train_loader, val_loader, test_loader = load_binary_loaders(data_dir)

    print("\n=== Loading Keras generators (Model 2) ===")
    train_gen, val_gen, test_gen, class_weights = load_keras_generators(data_dir)

    # ── 2. Model 1 ────────────────────────────────────────────────────────────
    print("\n=== Training Model 1 (Binary Screener) ===")
    model1 = build_model1()
    model1 = train_model1(model1, train_loader, val_loader)
    evaluate_model1(model1, val_loader, test_loader)

    # ── 3. Model 2 ────────────────────────────────────────────────────────────
    print("\n=== Training Model 2 (Multiclass Classifier) ===")
    model2   = build_model2()
    model2   = train_model2(model2, train_gen, val_gen, class_weights)
    evaluate_model2(model2, test_gen)

    # ── 4. Save legacy format ─────────────────────────────────────────────────
    model2.save(MODEL2_H5_PATH)
    print(f"Saved: {MODEL2_H5_PATH}")

    # ── 5. Clinical benchmark ─────────────────────────────────────────────────
    test_dir = os.path.join(data_dir, "Testing")
    print("\n=== Clinical Pipeline Benchmark ===")
    clinical_benchmark(test_dir, n_samples=50)


if __name__ == "__main__":
    main()
