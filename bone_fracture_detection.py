"""
Multi-Regional Bone Fracture Identification Using X-ray Images
Advanced Topics in Image Analysis - Course Project
Author: Mehmet Bedirhan Onder

Dataset: Fracture Multi-Region X-ray Dataset
  https://www.kaggle.com/datasets/bmadushanirodrigo/fracture-multi-region-x-ray-data
"""

import os
import argparse
import warnings
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
from pathlib import Path
from PIL import ImageFile

# Allow Pillow to load truncated/corrupted images
ImageFile.LOAD_TRUNCATED_IMAGES = True

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
warnings.filterwarnings("ignore")

import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_score,
    f1_score,
)

# Configurations
CONFIG = {
    "dataset_name": "bmadushanirodrigo/fracture-multi-region-x-ray-data",
    "data_dir": os.path.join(os.path.dirname(__file__), "dataset"),
    "img_size": (224, 224),
    "batch_size": 32,
    "epochs_baseline": 20,
    "epochs_transfer": 15,
    "epochs_finetune": 5,
    "learning_rate_baseline": 1e-3,
    "learning_rate_transfer": 1e-4,
    "learning_rate_finetune": 1e-5,
    "seed": 42,
    "output_dir": os.path.join(os.path.dirname(__file__), "results"),
}

np.random.seed(CONFIG["seed"])
tf.random.set_seed(CONFIG["seed"])


# Download Dataset
def download_dataset(dataset_name: str, data_dir: str) -> str:
    if os.path.exists(data_dir) and any(os.scandir(data_dir)):
        print(f"[INFO] Dataset directory already exists at '{data_dir}'. Skipping download.")
        return data_dir

    print("[INFO] Downloading dataset from Kaggle...")
    os.makedirs(data_dir, exist_ok=True)

    try:
        import kagglehub

        # Use a short path to avoid Windows MAX_PATH limit.
        short_cache = "C:\\kgdata"
        os.environ["KAGGLEHUB_CACHE"] = short_cache
        os.makedirs(short_cache, exist_ok=True)

        path = kagglehub.dataset_download(dataset_name)
        print(f"[INFO] Dataset downloaded to: {path}")

        # Copy into our project data_dir if kagglehub stored it elsewhere
        import shutil

        if os.path.abspath(path) != os.path.abspath(data_dir):
            for item in os.listdir(path):
                src = os.path.join(path, item)
                dst = os.path.join(data_dir, item)
                if os.path.isdir(src):
                    shutil.copytree(src, dst, dirs_exist_ok=True)
                else:
                    shutil.copy2(src, dst)
    except ImportError:
        print("[INFO] kagglehub not found, trying kaggle CLI...")
        ret = os.system(
            f'kaggle datasets download -d {dataset_name} --unzip -p "{data_dir}"'
        )
        if ret != 0:
            raise RuntimeError(
                "Failed to download dataset. Install kagglehub (`pip install kagglehub`) "
                "or set up the Kaggle CLI (pip install kaggle) with your API key."
            )

    print("[INFO] Dataset ready.\n")
    return data_dir


def resolve_data_dir(data_dir: str) -> str:
    if not os.path.isdir(data_dir):
        return data_dir

    contents = [
        d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))
    ]

    # If train/ is directly in path dataset is already extracted correctly
    if "train" in contents:
        return data_dir

    # One level of nesting
    if len(contents) >= 1:
        for sub in contents:
            candidate = os.path.join(data_dir, sub)
            sub_contents = os.listdir(candidate)
            if "train" in sub_contents:
                return candidate
            # Two levels of nesting
            for sub2 in sub_contents:
                candidate2 = os.path.join(candidate, sub2)
                if os.path.isdir(candidate2) and "train" in os.listdir(candidate2):
                    return candidate2

    return data_dir

# Image Preprocessing Functions
def apply_clahe(image: np.ndarray, clip_limit: float = 2.0) -> np.ndarray:
    """
    CLAHE - Contrast Limited Adaptive Histogram Equalization.

    Why CLAHE for X-rays:
      X-ray images often suffer from uneven illumination and low contrast.
      CLAHE divides the image into small tiles and applies histogram equalization
      locally, which enhances subtle fracture lines without over-amplifying noise.
      The clip limit prevents excessive contrast amplification in homogeneous regions.
    """
    img = image.astype(np.uint8)
    if len(img.shape) == 3 and img.shape[2] == 3:
        lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
        l_channel = clahe.apply(l_channel)
        lab = cv2.merge([l_channel, a_channel, b_channel])
        return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    else:
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
        return clahe.apply(img)


def apply_median_filter(image: np.ndarray, ksize: int = 3) -> np.ndarray:
    """
    Median Filter for noise reduction.

    Why Median over Gaussian for fracture detection:
      Median filtering removes salt-and-pepper noise (common in X-ray digitization)
      while preserving sharp edges. This is critical because fracture lines ARE edges;
      Gaussian blur would soften them and make detection harder.
    """
    return cv2.medianBlur(image.astype(np.uint8), ksize)


def apply_unsharp_mask(
    image: np.ndarray, sigma: float = 1.0, strength: float = 1.5
) -> np.ndarray:
    """
    Unsharp Masking for edge enhancement.

    After denoising, this sharpens fine details (fracture lines, bone edges)
    by subtracting a blurred version from the original and adding the difference back.
    """
    img = image.astype(np.float32)
    blurred = cv2.GaussianBlur(img, (0, 0), sigma)
    sharpened = cv2.addWeighted(img, 1.0 + strength, blurred, -strength, 0)
    return np.clip(sharpened, 0, 255).astype(np.uint8)


def preprocess_xray(image: np.ndarray) -> np.ndarray:
    """
    Full preprocessing pipeline for X-ray images:
      1. Median filtering  -> remove noise, keep edges
      2. CLAHE             -> enhance local contrast to reveal fracture lines
      3. Unsharp masking   -> sharpen fine fracture details
      4. Normalize to [0, 1]
    """
    image = apply_median_filter(image, ksize=3)
    image = apply_clahe(image, clip_limit=2.0)
    image = apply_unsharp_mask(image, sigma=1.0, strength=1.0)
    image = image.astype(np.float32) / 255.0
    return image

# 3. Visualization of Preprocessing
def visualize_preprocessing(data_dir: str, img_size: tuple):
    train_dir = os.path.join(data_dir, "train")
    if not os.path.isdir(train_dir):
        print("[WARN] Cannot visualize preprocessing: train directory not found.")
        return

    classes = sorted(os.listdir(train_dir))
    if not classes:
        return

    fig, axes = plt.subplots(len(classes), 5, figsize=(20, 4 * len(classes)))
    if len(classes) == 1:
        axes = [axes]

    col_titles = ["Original", "Median Filter", "CLAHE", "Unsharp Mask", "Final Result"]

    for row, cls_name in enumerate(classes):
        cls_dir = os.path.join(train_dir, cls_name)
        images = [
            f
            for f in os.listdir(cls_dir)
            if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp"))
        ]
        if not images:
            continue

        img_path = os.path.join(cls_dir, images[0])
        img = cv2.imread(img_path)
        if img is None:
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, img_size)

        # Step-by-step preprocessing
        step1 = apply_median_filter(img.copy())
        step2 = apply_clahe(step1.copy())
        step3 = apply_unsharp_mask(step2.copy())
        final = step3.astype(np.float32) / 255.0

        steps = [img, step1, step2, step3, (final * 255).astype(np.uint8)]

        for col, (step_img, title) in enumerate(zip(steps, col_titles)):
            axes[row][col].imshow(step_img if step_img.max() > 1 else step_img)
            if row == 0:
                axes[row][col].set_title(title, fontsize=12, fontweight="bold")
            axes[row][col].set_ylabel(cls_name, fontsize=10)
            axes[row][col].axis("off")

    plt.suptitle(
        "X-ray Preprocessing Pipeline: Step-by-Step", fontsize=14, fontweight="bold"
    )
    plt.tight_layout()
    os.makedirs(CONFIG["output_dir"], exist_ok=True)
    plt.savefig(
        os.path.join(CONFIG["output_dir"], "preprocessing_steps.png"),
        dpi=150,
        bbox_inches="tight",
    )
    plt.show()
    print("[INFO] Preprocessing visualization saved.\n")

# 4. Data Loading with Augmentation
def create_data_generators_no_preprocessing(data_dir: str, img_size: tuple, batch_size: int):
    """
    Create data generators WITHOUT custom preprocessing.
    Only rescales pixel values to [0, 1] — no CLAHE, median filter, or unsharp mask.
    Used to measure the impact of preprocessing on model performance.
    """
    train_dir = os.path.join(data_dir, "train")
    val_dir = os.path.join(data_dir, "val")
    test_dir = os.path.join(data_dir, "test")

    if not os.path.exists(val_dir):
        val_dir = os.path.join(data_dir, "validation")

    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255.0,
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        brightness_range=[0.9, 1.1],
        fill_mode="nearest",
    )

    eval_datagen = ImageDataGenerator(rescale=1.0 / 255.0)

    print("[INFO] Loading datasets (NO preprocessing)...")

    train_gen = train_datagen.flow_from_directory(
        train_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode="binary",
        shuffle=True,
        seed=CONFIG["seed"],
    )

    val_gen = eval_datagen.flow_from_directory(
        val_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode="binary",
        shuffle=False,
    )

    test_gen = eval_datagen.flow_from_directory(
        test_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode="binary",
        shuffle=False,
    )

    print(f"\n[INFO] Class mapping: {train_gen.class_indices}")
    print(f"[INFO] Training samples:   {train_gen.samples}")
    print(f"[INFO] Validation samples: {val_gen.samples}")
    print(f"[INFO] Test samples:       {test_gen.samples}\n")

    return train_gen, val_gen, test_gen


def create_data_generators(data_dir: str, img_size: tuple, batch_size: int):
    """
    Create data generators with:
      - Custom X-ray preprocessing (CLAHE + median filter + unsharp mask)
      - Data augmentation for training (rotation, shift, zoom, flip, brightness)
      - No augmentation for validation/test
    """
    train_dir = os.path.join(data_dir, "train")
    val_dir = os.path.join(data_dir, "val")
    test_dir = os.path.join(data_dir, "test")

    # Handle alternative directory names
    if not os.path.exists(val_dir):
        val_dir = os.path.join(data_dir, "validation")

    # Training: preprocessing + augmentation
    train_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_xray,
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        brightness_range=[0.9, 1.1],
        fill_mode="nearest",
    )

    # Validation & Test: preprocessing only
    eval_datagen = ImageDataGenerator(preprocessing_function=preprocess_xray)

    print("[INFO] Loading datasets...")

    train_gen = train_datagen.flow_from_directory(
        train_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode="binary",
        shuffle=True,
        seed=CONFIG["seed"],
    )

    val_gen = eval_datagen.flow_from_directory(
        val_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode="binary",
        shuffle=False,
    )

    test_gen = eval_datagen.flow_from_directory(
        test_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode="binary",
        shuffle=False,
    )

    print(f"\n[INFO] Class mapping: {train_gen.class_indices}")
    print(f"[INFO] Training samples:   {train_gen.samples}")
    print(f"[INFO] Validation samples: {val_gen.samples}")
    print(f"[INFO] Test samples:       {test_gen.samples}\n")

    return train_gen, val_gen, test_gen

# 5. Visualize Sample Augmented Images
def visualize_augmentation(train_gen):
    """Show samples of augmented training images."""
    batch_images, batch_labels = next(train_gen)
    class_names = {v: k for k, v in train_gen.class_indices.items()}

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    for i, ax in enumerate(axes.flat):
        if i >= len(batch_images):
            break
        img = batch_images[i]
        # Undo normalization for display if needed
        if img.max() <= 1.0:
            img = (img * 255).astype(np.uint8)
        ax.imshow(img)
        label = class_names[int(batch_labels[i])]
        ax.set_title(f"Label: {label}", fontsize=10)
        ax.axis("off")

    plt.suptitle("Augmented Training Samples", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(
        os.path.join(CONFIG["output_dir"], "augmented_samples.png"),
        dpi=150,
        bbox_inches="tight",
    )
    plt.show()
    print("[INFO] Augmented samples visualization saved.\n")

# 6. Model Definitions
def build_baseline_cnn(input_shape: tuple) -> tf.keras.Model:
    """
    Architecture:
      4 convolutional blocks (Conv2D -> BatchNorm -> ReLU -> MaxPool)
      with increasing filter counts (32 -> 64 -> 128 -> 256),
      followed by GlobalAveragePooling and Dense layers with Dropout.
    """
    model = models.Sequential(
        [
            # Block 1
            layers.Conv2D(32, (3, 3), padding="same", input_shape=input_shape),
            layers.BatchNormalization(),
            layers.Activation("relu"),
            layers.MaxPooling2D((2, 2)),
            # Block 2
            layers.Conv2D(64, (3, 3), padding="same"),
            layers.BatchNormalization(),
            layers.Activation("relu"),
            layers.MaxPooling2D((2, 2)),
            # Block 3
            layers.Conv2D(128, (3, 3), padding="same"),
            layers.BatchNormalization(),
            layers.Activation("relu"),
            layers.MaxPooling2D((2, 2)),
            # Block 4
            layers.Conv2D(256, (3, 3), padding="same"),
            layers.BatchNormalization(),
            layers.Activation("relu"),
            layers.GlobalAveragePooling2D(),
            # Classification head
            layers.Dense(256, activation="relu"),
            layers.Dropout(0.5),
            layers.Dense(128, activation="relu"),
            layers.Dropout(0.3),
            layers.Dense(1, activation="sigmoid"),
        ]
    )

    model.compile(
        optimizer=optimizers.Adam(learning_rate=CONFIG["learning_rate_baseline"]),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )

    print("BASELINE CNN MODEL")
    model.summary()
    return model


def build_transfer_model(input_shape: tuple) -> tf.keras.Model:
    """
    Transfer Learning model using ResNet50 pre-trained on ImageNet.

    Why ResNet50:
      - Residual connections allow learning fine-grained features
        without vanishing gradients.
      - Well-validated in medical imaging literature for X-ray analysis.
      - ImageNet pre-training provides strong low-level feature extractors
        (edges, textures) that transfer well to bone fracture detection.

    Strategy:
      Phase 1: Freeze base, train classification head only.
      Phase 2: Unfreeze top layers of ResNet50 and fine-tune with lower LR.
    """
    base_model = ResNet50(
        weights="imagenet", include_top=False, input_shape=input_shape
    )
    base_model.trainable = False  # Freeze for Phase 1

    model = models.Sequential(
        [
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.Dense(256, activation="relu"),
            layers.Dropout(0.5),
            layers.Dense(128, activation="relu"),
            layers.Dropout(0.3),
            layers.Dense(1, activation="sigmoid"),
        ]
    )

    model.compile(
        optimizer=optimizers.Adam(learning_rate=CONFIG["learning_rate_transfer"]),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )

    print("TRANSFER LEARNING MODEL (ResNet50)")
    model.summary()
    return model


def fine_tune_model(model: tf.keras.Model, num_layers_to_unfreeze: int = 30):
    """
    Fine-tuning phase: unfreeze the top layers of ResNet50 base
    and retrain with a very low learning rate.
    """
    base_model = model.layers[0]
    base_model.trainable = True

    # Freeze all layers except the last `num_layers_to_unfreeze`
    for layer in base_model.layers[:-num_layers_to_unfreeze]:
        layer.trainable = False

    model.compile(
        optimizer=optimizers.Adam(learning_rate=CONFIG["learning_rate_finetune"]),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )

    trainable = sum(1 for l in model.layers[0].layers if l.trainable)
    total = len(model.layers[0].layers)
    print(f"\n[INFO] Fine-tuning: {trainable}/{total} ResNet50 layers are now trainable.")
    return model

# 7. Training
def get_callbacks(model_name: str) -> list:
    """Create training callbacks: EarlyStopping, ReduceLR, ModelCheckpoint."""
    os.makedirs(CONFIG["output_dir"], exist_ok=True)
    return [
        callbacks.EarlyStopping(
            monitor="val_loss", patience=4, restore_best_weights=True, verbose=1
        ),
        callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=2, min_lr=1e-7, verbose=1
        ),
        callbacks.ModelCheckpoint(
            filepath=os.path.join(CONFIG["output_dir"], f"{model_name}_best.keras"),
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1,
        ),
    ]


def train_model(model, train_gen, val_gen, epochs: int, model_name: str):
    """Train the model and return training history."""
    print(f"TRAINING: {model_name}")

    history = model.fit(
        train_gen,
        epochs=epochs,
        validation_data=val_gen,
        callbacks=get_callbacks(model_name),
        verbose=1,
    )
    return history

# 8. Evaluation & Visualization
def plot_training_history(history, model_name: str):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(history.history["accuracy"], label="Train Accuracy", linewidth=2)
    ax1.plot(history.history["val_accuracy"], label="Val Accuracy", linewidth=2)
    ax1.set_title(f"{model_name} - Accuracy")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Accuracy")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(history.history["loss"], label="Train Loss", linewidth=2)
    ax2.plot(history.history["val_loss"], label="Val Loss", linewidth=2)
    ax2.set_title(f"{model_name} - Loss")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Loss")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(
        os.path.join(CONFIG["output_dir"], f"{model_name}_training_curves.png"),
        dpi=150,
        bbox_inches="tight",
    )
    plt.show()
    print(f"[INFO] Training curves saved for {model_name}.\n")


def evaluate_model(model, test_gen, model_name: str) -> dict:
    print(f"EVALUATION: {model_name}")

    test_gen.reset()
    y_pred_prob = model.predict(test_gen, verbose=1)
    y_pred = (y_pred_prob > 0.5).astype(int).flatten()
    y_true = test_gen.classes
    class_names = list(test_gen.class_indices.keys())

    # Compute metrics
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average="weighted")
    f1 = f1_score(y_true, y_pred, average="weighted")

    print(f"\n  Accuracy:  {acc:.4f}")
    print(f"  Precision: {prec:.4f}")
    print(f"  F1-Score:  {f1:.4f}")
    print(f"\n  Classification Report:\n")
    print(classification_report(y_true, y_pred, target_names=class_names))

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
    )
    plt.title(f"{model_name} - Confusion Matrix")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    plt.savefig(
        os.path.join(CONFIG["output_dir"], f"{model_name}_confusion_matrix.png"),
        dpi=150,
        bbox_inches="tight",
    )
    plt.show()
    print(f"[INFO] Confusion matrix saved for {model_name}.\n")

    return {"accuracy": acc, "precision": prec, "f1_score": f1}


def compare_models(results: dict):
    model_names = list(results.keys())
    metrics = ["accuracy", "precision", "f1_score"]
    metric_labels = ["Accuracy", "Precision", "F1-Score"]

    x = np.arange(len(metrics))
    width = 0.8 / len(model_names)

    fig, ax = plt.subplots(figsize=(10, 6))

    colors = ["#2196F3", "#FF9800", "#4CAF50"]
    for i, name in enumerate(model_names):
        values = [results[name][m] for m in metrics]
        bars = ax.bar(
            x + i * width, values, width, label=name, color=colors[i % len(colors)]
        )
        for bar, val in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                bar.get_height() + 0.008,
                f"{val:.3f}",
                ha="center",
                va="bottom",
                fontsize=10,
                fontweight="bold",
            )

    ax.set_xlabel("Metric", fontsize=12)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("Model Comparison", fontsize=14, fontweight="bold")
    ax.set_xticks(x + width * (len(model_names) - 1) / 2)
    ax.set_xticklabels(metric_labels)
    ax.legend()
    ax.set_ylim(0, 1.12)
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(
        os.path.join(CONFIG["output_dir"], "model_comparison.png"),
        dpi=150,
        bbox_inches="tight",
    )
    plt.show()
    print("[INFO] Model comparison chart saved.\n")

# 9. Main
def main():
    parser = argparse.ArgumentParser(
        description="Multi-Regional Bone Fracture Identification Using X-ray Images"
    )
    parser.add_argument(
        "--dataset-path",
        type=str,
        default=None,
        help="Path to the dataset directory (must contain train/val/test subdirs). "
             "If not provided or invalid, the dataset will be downloaded from Kaggle.",
    )
    args = parser.parse_args()

    # --- Step 1: Locate or Download Dataset ---
    if args.dataset_path and os.path.isdir(args.dataset_path):
        # User provided a path — validate it has a train/ subfolder
        candidate = resolve_data_dir(args.dataset_path)
        train_check = os.path.join(candidate, "train")
        if os.path.isdir(train_check):
            data_dir = candidate
            print(f"[INFO] Using provided dataset path: {data_dir}")
        else:
            print(f"[WARN] Provided path '{args.dataset_path}' has no train/ subfolder.")
            print("[INFO] Falling back to Kaggle download...")
            download_dataset(CONFIG["dataset_name"], CONFIG["data_dir"])
            data_dir = resolve_data_dir(CONFIG["data_dir"])
    else:
        if args.dataset_path:
            print(f"[WARN] Provided path '{args.dataset_path}' does not exist.")
            print("[INFO] Falling back to Kaggle download...")
        download_dataset(CONFIG["dataset_name"], CONFIG["data_dir"])
        data_dir = resolve_data_dir(CONFIG["data_dir"])

    print(f"[INFO] Using data directory: {data_dir}\n")

    os.makedirs(CONFIG["output_dir"], exist_ok=True)
    img_shape = CONFIG["img_size"] + (3,)

    # --- Step 2: Create Data Generators ---
    # train_gen, val_gen, test_gen = create_data_generators(
    #     data_dir, CONFIG["img_size"], CONFIG["batch_size"]
    # )

    results = {}

    # MODEL 0: Baseline CNN WITHOUT Preprocessing (to measure preprocessing impact)
    print("\n" + "=" * 70)
    print("MODEL: Baseline CNN WITHOUT Preprocessing")
    print("=" * 70)
    train_gen_raw, val_gen_raw, test_gen_raw = create_data_generators_no_preprocessing(
        data_dir, CONFIG["img_size"], CONFIG["batch_size"]
    )
    raw_model = build_baseline_cnn(img_shape)
    raw_history = train_model(
        raw_model,
        train_gen_raw,
        val_gen_raw,
        CONFIG["epochs_baseline"],
        "Baseline_CNN_NoPreprocessing",
    )
    plot_training_history(raw_history, "Baseline_CNN_NoPreprocessing")
    results["Baseline CNN (No Preprocessing)"] = evaluate_model(
        raw_model, test_gen_raw, "Baseline_CNN_NoPreprocessing"
    )

    # MODEL 1: Baseline CNN (with preprocessing)
    baseline_model = build_baseline_cnn(img_shape)
    baseline_history = train_model(
        baseline_model,
        train_gen,
        val_gen,
        CONFIG["epochs_baseline"],
        "Baseline_CNN",
    )
    plot_training_history(baseline_history, "Baseline_CNN")
    results["Baseline CNN"] = evaluate_model(baseline_model, test_gen, "Baseline_CNN")

    # MODEL 2: Transfer Learning (ResNet50) - Phase 1 (Frozen Base)
    transfer_model = build_transfer_model(img_shape)
    transfer_history = train_model(
        transfer_model,
        train_gen,
        val_gen,
        CONFIG["epochs_transfer"],
        "ResNet50_Transfer",
    )
    plot_training_history(transfer_history, "ResNet50_Transfer_Phase1")

    # --- Phase 2: Fine-tuning ---
    print("\n[INFO] Starting fine-tuning phase (unfreezing top ResNet50 layers)...")
    transfer_model = fine_tune_model(transfer_model, num_layers_to_unfreeze=30)
    finetune_history = train_model(
        transfer_model,
        train_gen,
        val_gen,
        CONFIG["epochs_finetune"],
        "ResNet50_FineTuned",
    )
    plot_training_history(finetune_history, "ResNet50_FineTuned")
    results["ResNet50 (Fine-Tuned)"] = evaluate_model(
        transfer_model, test_gen, "ResNet50_FineTuned"
    )

    compare_models(results)

    print(" FINAL RESULTS SUMMARY")
    for name, metrics in results.items():
        print(f"\n  {name}:")
        print(f"    Accuracy:  {metrics['accuracy']:.4f}")
        print(f"    Precision: {metrics['precision']:.4f}")
        print(f"    F1-Score:  {metrics['f1_score']:.4f}")

    best_model = max(results, key=lambda k: results[k]["f1_score"])
    print(f"\n  >>> Best Model (by F1-Score): {best_model}")
    print(f"\n[INFO] All results saved to '{CONFIG['output_dir']}/' directory.")


if __name__ == "__main__":
    main()

    # --- Optional: Uncomment below to generate visualization outputs ---
    # data_dir = resolve_data_dir(CONFIG["data_dir"])
    # visualize_preprocessing(data_dir, CONFIG["img_size"])
    # train_gen, _, _ = create_data_generators(data_dir, CONFIG["img_size"], CONFIG["batch_size"])
    # visualize_augmentation(train_gen)
