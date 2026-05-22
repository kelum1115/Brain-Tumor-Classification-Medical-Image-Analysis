"""
preprocessing.py — Data loading, augmentation, and binary/multiclass generators.

Two independent pipelines:
  - PyTorch  (Model 1): binary tumor/no-tumor screener
  - Keras    (Model 2): 4-class classifier (glioma, meningioma, notumor, pituitary)

ImageNet normalization stats are standard; do not change unless switching backbone.
"""

import os
import numpy as np
import cv2
import torch
from collections import Counter
from PIL import Image

import kagglehub
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, random_split, Subset
from tensorflow.keras.preprocessing.image import ImageDataGenerator as Imgen
from tensorflow.keras.applications.vgg16 import preprocess_input
from sklearn.utils.class_weight import compute_class_weight

from config import (
    IMG_SIZE, BATCH_SIZE, CLASSES
)

# ── Dataset Download ──────────────────────────────────────────────────────────

def get_dataset_path() -> str:
    """Download dataset via kagglehub; returns local root path."""
    path = kagglehub.dataset_download("masoudnickparvar/brain-tumor-mri-dataset")
    print(f"Dataset path: {path}")
    return path


# ── PyTorch Transforms (Model 1) ──────────────────────────────────────────────

# OOD-robust augmentation: affine scale+translate hardens against scanner variance
train_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomRotation(10),
    transforms.RandomHorizontalFlip(),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.8, 1.2)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

val_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def _make_binary(dataset: ImageFolder) -> ImageFolder:
    """Remap multiclass labels → binary (0=no tumor, 1=tumor)."""
    notumor_idx = dataset.class_to_idx["notumor"]
    dataset.targets = [0 if t == notumor_idx else 1 for t in dataset.targets]
    dataset.samples = [
        (p, dataset.targets[i]) for i, (p, _) in enumerate(dataset.samples)
    ]
    return dataset


def load_binary_loaders(data_dir: str):
    """
    Returns (train_loader, val_loader, test_loader) for Model 1.
    80/20 train/val split from Training folder; Testing folder as held-out test.
    """
    base_ds    = ImageFolder(f"{data_dir}/Training")
    val_size   = int(0.2 * len(base_ds))
    train_size = len(base_ds) - val_size

    train_idx, val_idx = random_split(
        range(len(base_ds)),
        [train_size, val_size],
        generator=torch.Generator().manual_seed(67),
    )

    train_full = _make_binary(ImageFolder(f"{data_dir}/Training", transform=train_tf))
    val_full   = _make_binary(ImageFolder(f"{data_dir}/Training", transform=val_tf))
    test_full  = _make_binary(ImageFolder(f"{data_dir}/Testing",  transform=val_tf))

    train_ds = Subset(train_full, train_idx)
    val_ds   = Subset(val_full,   val_idx)

    for name, ds in [("Train", train_ds), ("Val", val_ds), ("Test", test_full)]:
        labels = [lbl for _, lbl in ds]
        print(f"{name}: {len(ds)} | {Counter(labels)}")

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=2)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    test_loader  = DataLoader(test_full, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    return train_loader, val_loader, test_loader


# ── Keras Preprocessing (Model 2) ─────────────────────────────────────────────

def _clahe_preprocess(image: Image.Image) -> np.ndarray:
    """CLAHE contrast enhancement → VGG16 preprocess_input normalization."""
    img_uint8 = np.array(image, dtype=np.uint8)
    gray      = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2GRAY)
    clahe     = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced  = clahe.apply(gray)
    img_rgb   = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2RGB)
    return preprocess_input(img_rgb.astype(np.float32))


def load_keras_generators(data_dir: str):
    """
    Returns (train_gen, val_gen, test_gen, class_weights) for Model 2.
    Generators apply CLAHE + VGG16 normalization + light augmentation.
    """
    train_dir = os.path.join(data_dir, "Training")
    test_dir  = os.path.join(data_dir, "Testing")

    train_idg = Imgen(
        preprocessing_function=_clahe_preprocess,
        rotation_range=10,
        zoom_range=0.05,
        width_shift_range=0.05,
        height_shift_range=0.05,
        horizontal_flip=True,
        fill_mode="nearest",
        validation_split=0.2,
    )
    val_idg  = Imgen(preprocessing_function=_clahe_preprocess, validation_split=0.2)
    test_idg = Imgen(preprocessing_function=_clahe_preprocess)

    gen_kwargs = dict(
        directory=train_dir,
        classes=CLASSES,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        seed=67,
        class_mode="categorical",
    )

    train_gen = train_idg.flow_from_directory(**gen_kwargs, subset="training")
    val_gen   = val_idg.flow_from_directory(**gen_kwargs,   subset="validation")
    test_gen  = test_idg.flow_from_directory(
        directory=test_dir, classes=CLASSES,
        target_size=(IMG_SIZE, IMG_SIZE), batch_size=BATCH_SIZE,
        seed=67, class_mode="categorical", shuffle=False,
    )

    # Balanced class weights compensate for meningioma underrepresentation
    raw_labels  = train_gen.classes
    unique_cls  = np.unique(raw_labels)
    weights     = compute_class_weight("balanced", classes=unique_cls, y=raw_labels)
    class_weights = dict(zip(unique_cls.tolist(), weights.tolist()))
    print("Class weights:", {CLASSES[k]: f"{v:.3f}" for k, v in class_weights.items()})

    print(f"Train: {train_gen.samples} | Val: {val_gen.samples} | Test: {test_gen.samples}")
    print("Class indices:", train_gen.class_indices)

    return train_gen, val_gen, test_gen, class_weights
