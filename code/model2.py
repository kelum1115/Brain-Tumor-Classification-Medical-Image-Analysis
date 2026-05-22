"""
model2.py — TF/Keras VGG16 multiclass classifier (Model 2).

Two-phase training strategy:
  Phase 1 — Frozen VGG16 base, train classifier head only (10 epochs).
             Fast convergence; avoids destroying ImageNet features early.
  Phase 2 — Unfreeze block5 (last 8 VGG16 layers), fine-tune at LR=1e-5.
             Adapts high-level features to MRI-specific patterns.

Loss: CategoricalFocalCrossentropy(alpha=0.25, gamma=2.0)
  - Reduces class-imbalance sensitivity vs. plain cross-entropy
  - alpha=0.25 standard for 4-class; gamma=2.0 moderate focus on hard examples

OOD handling is in pipeline.py (entropy gate + conf thresholds).
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

from config import IMG_SIZE, LR, EPOCHS_M2, EPOCHS_M2_FT, MODEL2_PHASE1_PATH, MODEL2_FINAL_PATH


# ── Architecture ──────────────────────────────────────────────────────────────

def build_model2() -> Sequential:
    """
    VGG16 transfer learning pipeline.
    Base frozen at build time; call unfreeze_block5() for phase 2.
    """
    base = VGG16(weights="imagenet", include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))
    base.trainable = False  # Phase 1: head-only training

    model = Sequential([
        base,
        layers.GlobalAveragePooling2D(),
        layers.BatchNormalization(),
        layers.Dense(256, activation="relu"),
        layers.Dropout(0.5),
        layers.Dense(64,  activation="relu"),
        layers.Dropout(0.4),
        layers.Dense(4,   activation="softmax"),  # glioma / meningioma / notumor / pituitary
    ])
    return model


def _compile_model2(model: Sequential, lr: float) -> Sequential:
    model.compile(
        optimizer=Adam(learning_rate=lr),
        loss=tf.keras.losses.CategoricalFocalCrossentropy(alpha=0.25, gamma=2.0),
        metrics=["accuracy"],
    )
    return model


# ── Training ──────────────────────────────────────────────────────────────────

def train_model2(model: Sequential, train_gen, val_gen, class_weights: dict) -> Sequential:
    """
    Two-phase training. Returns model with best phase-2 weights loaded.
    """
    # ── Phase 1: frozen base ──────────────────────────────────────────────────
    _compile_model2(model, lr=LR)
    print("Phase 1: Training classifier head (VGG16 frozen)")
    model.fit(
        train_gen,
        validation_data=val_gen,
        class_weight=class_weights,
        epochs=EPOCHS_M2,
        verbose=1,
        callbacks=[
            EarlyStopping(monitor="val_accuracy", patience=5, restore_best_weights=True, verbose=1),
            ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6, verbose=1),
            ModelCheckpoint(MODEL2_PHASE1_PATH, monitor="val_accuracy", save_best_only=True, verbose=1),
        ],
    )

    # ── Phase 2: unfreeze block5 ──────────────────────────────────────────────
    # Only the last 8 VGG16 layers (block5_conv1–3 + pooling) become trainable.
    # Re-freezing earlier blocks preserves low/mid-level features; prevents catastrophic forgetting.
    base_model = model.layers[0]
    base_model.trainable = True
    for layer in base_model.layers[:-8]:
        layer.trainable = False

    trainable_count = sum(1 for l in base_model.layers if l.trainable)
    print(f"Phase 2: Fine-tuning {trainable_count}/{len(base_model.layers)} VGG16 layers")

    _compile_model2(model, lr=1e-5)  # 10x lower LR prevents blowing up phase-1 weights
    model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=EPOCHS_M2_FT,
        class_weight=class_weights,
        verbose=1,
        callbacks=[
            EarlyStopping(monitor="val_accuracy", patience=4, restore_best_weights=True, verbose=1),
            ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-7, verbose=1),
            ModelCheckpoint(MODEL2_FINAL_PATH, monitor="val_accuracy", save_best_only=True, verbose=1),
        ],
    )

    return model


# ── Convenience Loader ────────────────────────────────────────────────────────

def load_model2() -> keras.Model:
    """Load trained Model 2 from disk. Used at inference time."""
    return keras.models.load_model(MODEL2_FINAL_PATH)
