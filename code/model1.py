"""
model1.py — PyTorch binary tumor screener (Model 1).

Design priorities:
  - Recall > Precision: a missed tumor is worse than a false positive
  - FocalLoss(gamma=3.0): aggressively penalizes confident wrong answers
  - Threshold=0.2: shifts decision boundary further toward recall
  - Best checkpoint saved on val recall, not accuracy

Architecture: 4-block CNN → AdaptiveAvgPool → FC head
Kept shallow intentionally; deep features aren't needed for binary detection.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from config import DEVICE, LR, EPOCHS_M1, OPTIMAL_THRESHOLD, MODEL1_PATH


# ── Loss ──────────────────────────────────────────────────────────────────────

class FocalLoss(nn.Module):
    """
    Binary focal loss.
    alpha=0.75 upweights the tumor class (positive).
    gamma=3.0  aggressively down-weights easy negatives.
    """
    def __init__(self, alpha: float = 0.75, gamma: float = 3.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce   = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        pt    = torch.exp(-bce)
        loss  = self.alpha * (1 - pt) ** self.gamma * bce
        return loss.mean()


# ── Architecture ──────────────────────────────────────────────────────────────

def build_model1() -> nn.Sequential:
    """4-block CNN binary screener. Returns model on DEVICE."""
    model = nn.Sequential(
        # Block 1
        nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
        # Block 2
        nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
        # Block 3
        nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2),
        # Global pool → FC head
        nn.AdaptiveAvgPool2d((1, 1)),
        nn.Flatten(),
        nn.Linear(128, 256), nn.ReLU(), nn.Dropout(0.5),
        nn.Linear(256, 1),   # raw logit; sigmoid applied at inference
    ).to(DEVICE)
    return model


# ── Training ──────────────────────────────────────────────────────────────────

def train_model1(
    model: nn.Sequential,
    train_loader: DataLoader,
    val_loader: DataLoader,
) -> nn.Sequential:
    """
    Train binary screener. Checkpoints best val-recall model to MODEL1_PATH.
    Returns model with best weights loaded.
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = FocalLoss(alpha=0.75, gamma=3.0)

    best_recall, best_precision, best_epoch = 0.0, 0.0, 0

    for epoch in range(EPOCHS_M1):
        # ── Train pass ────────────────────────────────────────────────────────
        model.train()
        train_loss = 0.0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(DEVICE), labels.float().to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(imgs).squeeze(1), labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        # ── Val pass ──────────────────────────────────────────────────────────
        model.eval()
        val_loss = 0.0
        tp = fn = fp = tn = 0

        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(DEVICE), labels.float().to(DEVICE)
                logits = model(imgs).squeeze(1)
                val_loss += criterion(logits, labels).item()

                preds  = (torch.sigmoid(logits) >= OPTIMAL_THRESHOLD).long()
                labels = labels.long()
                tp += ((preds == 1) & (labels == 1)).sum().item()
                fn += ((preds == 0) & (labels == 1)).sum().item()
                fp += ((preds == 1) & (labels == 0)).sum().item()
                tn += ((preds == 0) & (labels == 0)).sum().item()

        val_loss  /= len(val_loader)
        recall     = tp / (tp + fn + 1e-8)
        precision  = tp / (tp + fp + 1e-8)
        accuracy   = (tp + tn) / (tp + tn + fp + fn)

        # Save on recall improvement; tie-break by precision
        if recall > best_recall or (recall == best_recall and precision > best_precision):
            best_recall, best_precision, best_epoch = recall, precision, epoch + 1
            torch.save(model.state_dict(), MODEL1_PATH)

        print(
            f"Epoch {epoch+1}/{EPOCHS_M1} | "
            f"Train {train_loss:.4f} | Val {val_loss:.4f} | "
            f"Recall {recall:.4f} | Prec {precision:.4f} | Acc {accuracy:.4f} | "
            f"Best Recall {best_recall:.4f} (ep {best_epoch})"
        )

    model.load_state_dict(torch.load(MODEL1_PATH))
    return model


# ── Convenience Loader ────────────────────────────────────────────────────────

def load_model1() -> nn.Sequential:
    """Load trained Model 1 weights from disk. Used at inference time."""
    model = build_model1()
    model.load_state_dict(torch.load(MODEL1_PATH, map_location=DEVICE))
    model.eval()
    return model
