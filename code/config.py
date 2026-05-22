"""
config.py — Single source of truth for all hyperparams and paths.
Modify here; all modules read from this. No magic numbers in model code.
"""

import torch

# ── Hardware ──────────────────────────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Image & Training ──────────────────────────────────────────────────────────
IMG_SIZE   = 224
BATCH_SIZE = 32
LR         = 1e-4
EPOCHS_M1  = 10   # Model 1: binary screener
EPOCHS_M2  = 10   # Model 2: multiclass phase 1 (frozen base)
EPOCHS_M2_FT = 5  # Model 2: fine-tune phase 2 (unfreeze block5)

# ── Classes ───────────────────────────────────────────────────────────────────
CLASSES = ["glioma", "meningioma", "notumor", "pituitary"]

# ── Thresholds ────────────────────────────────────────────────────────────────
# Model 1: lower = higher recall (safety bias — catch every tumor)
OPTIMAL_THRESHOLD = 0.2

# Model 2 OOD gates: route to "Uncertain → Doctor Review" when below
CONF_LOW_GATE  = 0.45   # entropy-based OOD gate
CONF_HIGH_GATE = 0.85   # low-confidence tumor-suspected gate

# ── Artifact Paths ────────────────────────────────────────────────────────────
MODEL1_PATH        = "best_model_full.pth"
MODEL2_PHASE1_PATH = "best_model2_phase1.keras"
MODEL2_FINAL_PATH  = "best_model2_vgg16.keras"
MODEL2_H5_PATH     = "multiclass_tumor_model.h5"
