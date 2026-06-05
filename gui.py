"""
Bone Fracture Detection - GUI Application
Uses trained .keras models for inference on user-provided X-ray images.

Usage: python gui.py
"""

import os
import sys
import tkinter as tk
from tkinter import ttk, filedialog
import numpy as np
import cv2
from PIL import Image, ImageTk

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf

# Import preprocessing functions from the training script
from bone_fracture_detection import (
    preprocess_xray,
    apply_median_filter,
    apply_clahe,
    apply_unsharp_mask,
    CONFIG,
)

# ================================================================
# Constants
# ================================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")
IMG_SIZE = CONFIG["img_size"]  # (224, 224)

MODELS = {
    "Baseline CNN": os.path.join(RESULTS_DIR, "Baseline_CNN_best.keras"),
    "ResNet50 Transfer": os.path.join(RESULTS_DIR, "ResNet50_Transfer_best.keras"),
    "ResNet50 Fine-Tuned": os.path.join(RESULTS_DIR, "ResNet50_FineTuned_best.keras"),
}

# Class mapping (alphabetical order from flow_from_directory)
# Class 0 = "fractured", Class 1 = "not fractured"
CLASS_NAMES = {0: "Fractured", 1: "Not Fractured"}

DISPLAY_SIZE = (280, 280)
THUMB_SIZE = (130, 130)


# ================================================================
# Application
# ================================================================
class FractureDetectionApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Bone Fracture Detection System")
        self.root.configure(bg="#f0f0f0")
        self.root.resizable(False, False)

        self.model = None
        self.model_name = None
        self.original_image = None  # RGB numpy array (original size)

        self._build_ui()
        self._center_window()

    # ----------------------------------------------------------
    # UI Construction
    # ----------------------------------------------------------
    def _build_ui(self):
        style = ttk.Style()
        style.theme_use("clam")

        # --- Top controls ---
        ctrl_frame = ttk.Frame(self.root, padding=10)
        ctrl_frame.pack(fill="x")

        ttk.Label(ctrl_frame, text="Model:", font=("Segoe UI", 10, "bold")).pack(
            side="left", padx=(0, 5)
        )

        self.model_var = tk.StringVar(value="ResNet50 Fine-Tuned")
        model_combo = ttk.Combobox(
            ctrl_frame,
            textvariable=self.model_var,
            values=list(MODELS.keys()),
            state="readonly",
            width=25,
        )
        model_combo.pack(side="left", padx=(0, 15))
        model_combo.bind("<<ComboboxSelected>>", self._on_model_change)

        self.load_btn = ttk.Button(
            ctrl_frame, text="Load X-ray Image", command=self._load_image
        )
        self.load_btn.pack(side="left", padx=(0, 10))

        self.predict_btn = ttk.Button(
            ctrl_frame, text="Predict", command=self._predict, state="disabled"
        )
        self.predict_btn.pack(side="left")

        self.status_var = tk.StringVar(value="Select a model and load an X-ray image.")
        ttk.Label(ctrl_frame, textvariable=self.status_var, foreground="gray").pack(
            side="right"
        )

        # --- Image display (original + preprocessed) ---
        img_frame = ttk.Frame(self.root, padding=10)
        img_frame.pack(fill="x")

        # Original
        orig_container = ttk.LabelFrame(img_frame, text="Original Image", padding=5)
        orig_container.pack(side="left", padx=(0, 10))
        self.orig_label = ttk.Label(orig_container)
        self.orig_label.pack()
        self._set_placeholder(self.orig_label, DISPLAY_SIZE)

        # Preprocessed
        proc_container = ttk.LabelFrame(
            img_frame, text="Preprocessed Image", padding=5
        )
        proc_container.pack(side="left")
        self.proc_label = ttk.Label(proc_container)
        self.proc_label.pack()
        self._set_placeholder(self.proc_label, DISPLAY_SIZE)

        # --- Result panel ---
        self.result_frame = tk.Frame(self.root, bg="#e0e0e0", height=70)
        self.result_frame.pack(fill="x", padx=10, pady=(0, 5))
        self.result_frame.pack_propagate(False)

        self.result_label = tk.Label(
            self.result_frame,
            text="No prediction yet",
            font=("Segoe UI", 16, "bold"),
            bg="#e0e0e0",
            fg="#666666",
        )
        self.result_label.pack(expand=True)

        # --- Preprocessing steps strip ---
        steps_frame = ttk.LabelFrame(
            self.root, text="Preprocessing Steps", padding=5
        )
        steps_frame.pack(fill="x", padx=10, pady=(0, 10))

        self.step_labels = []
        step_names = ["Original", "Median Filter", "CLAHE", "Unsharp Mask"]
        for name in step_names:
            col = ttk.Frame(steps_frame)
            col.pack(side="left", padx=5)
            lbl = ttk.Label(col)
            lbl.pack()
            self._set_placeholder(lbl, THUMB_SIZE)
            ttk.Label(col, text=name, font=("Segoe UI", 8)).pack()
            self.step_labels.append(lbl)

    def _center_window(self):
        self.root.update_idletasks()
        w = self.root.winfo_width()
        h = self.root.winfo_height()
        x = (self.root.winfo_screenwidth() // 2) - (w // 2)
        y = (self.root.winfo_screenheight() // 2) - (h // 2)
        self.root.geometry(f"+{x}+{y}")

    @staticmethod
    def _set_placeholder(label: ttk.Label, size: tuple):
        placeholder = Image.new("RGB", size, (200, 200, 200))
        photo = ImageTk.PhotoImage(placeholder)
        label.configure(image=photo)
        label.image = photo  # keep reference

    @staticmethod
    def _numpy_to_photo(img_array: np.ndarray, size: tuple) -> ImageTk.PhotoImage:
        if img_array.dtype != np.uint8:
            if img_array.max() <= 1.0:
                img_array = (img_array * 255).astype(np.uint8)
            else:
                img_array = img_array.astype(np.uint8)
        pil_img = Image.fromarray(img_array)
        pil_img = pil_img.resize(size, Image.LANCZOS)
        return ImageTk.PhotoImage(pil_img)

    # ----------------------------------------------------------
    # Event Handlers
    # ----------------------------------------------------------
    def _on_model_change(self, event=None):
        selected = self.model_var.get()
        if selected == self.model_name:
            return
        self.model = None
        self.model_name = None
        self.status_var.set(f"Model will be loaded on first prediction.")

    def _load_image(self):
        path = filedialog.askopenfilename(
            title="Select an X-ray Image",
            filetypes=[
                ("Image files", "*.png *.jpg *.jpeg *.bmp"),
                ("All files", "*.*"),
            ],
        )
        if not path:
            return

        img = cv2.imread(path)
        if img is None:
            self.status_var.set("Error: Could not read the image file.")
            return

        self.original_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self._display_original()
        self._display_preprocessing_steps()
        self.predict_btn.configure(state="normal")
        self.status_var.set(f"Loaded: {os.path.basename(path)}")

        # Reset result
        self.result_label.configure(
            text="No prediction yet", bg="#e0e0e0", fg="#666666"
        )
        self.result_frame.configure(bg="#e0e0e0")
        self._set_placeholder(self.proc_label, DISPLAY_SIZE)

    def _display_original(self):
        photo = self._numpy_to_photo(self.original_image, DISPLAY_SIZE)
        self.orig_label.configure(image=photo)
        self.orig_label.image = photo

    def _display_preprocessing_steps(self):
        img = cv2.resize(self.original_image.copy(), IMG_SIZE)

        step0 = img.copy()
        step1 = apply_median_filter(img.copy())
        step2 = apply_clahe(step1.copy())
        step3 = apply_unsharp_mask(step2.copy())

        for lbl, step_img in zip(self.step_labels, [step0, step1, step2, step3]):
            photo = self._numpy_to_photo(step_img, THUMB_SIZE)
            lbl.configure(image=photo)
            lbl.image = photo

    def _load_model(self):
        selected = self.model_var.get()
        model_path = MODELS.get(selected)
        if not model_path or not os.path.exists(model_path):
            self.status_var.set(f"Error: Model file not found at {model_path}")
            return False

        self.status_var.set(f"Loading {selected} model...")
        self.root.update_idletasks()

        self.model = tf.keras.models.load_model(model_path)
        self.model_name = selected
        self.status_var.set(f"Model loaded: {selected}")
        return True

    def _predict(self):
        if self.original_image is None:
            self.status_var.set("Please load an image first.")
            return

        # Load model if needed
        if self.model is None or self.model_name != self.model_var.get():
            if not self._load_model():
                return

        self.status_var.set("Running prediction...")
        self.root.update_idletasks()

        # Preprocess (same pipeline as training)
        img = cv2.resize(self.original_image.copy(), IMG_SIZE)
        processed = preprocess_xray(img)

        # Display preprocessed image
        proc_display = (processed * 255).astype(np.uint8)
        photo = self._numpy_to_photo(proc_display, DISPLAY_SIZE)
        self.proc_label.configure(image=photo)
        self.proc_label.image = photo

        # Inference
        batch = np.expand_dims(processed, axis=0)
        prediction = self.model.predict(batch, verbose=0)[0][0]

        # Class mapping: sigmoid output
        # Class 0 = "fractured" (pred <= 0.5), Class 1 = "not fractured" (pred > 0.5)
        if prediction > 0.5:
            class_label = CLASS_NAMES[1]
            confidence = prediction * 100
            bg_color = "#4CAF50"  # green
            fg_color = "white"
        else:
            class_label = CLASS_NAMES[0]
            confidence = (1 - prediction) * 100
            bg_color = "#F44336"  # red
            fg_color = "white"

        result_text = f"{class_label}   |   Confidence: {confidence:.1f}%"
        self.result_label.configure(text=result_text, bg=bg_color, fg=fg_color)
        self.result_frame.configure(bg=bg_color)

        self.status_var.set(
            f"Prediction complete ({self.model_name})"
        )


# ================================================================
# Main
# ================================================================
def main():
    # Verify at least one model exists
    available = {k: v for k, v in MODELS.items() if os.path.exists(v)}
    if not available:
        print(f"[ERROR] No trained model files found in '{RESULTS_DIR}'.")
        print("Run bone_fracture_detection.py first to train the models.")
        sys.exit(1)

    # Update MODELS to only show available ones
    for k in list(MODELS.keys()):
        if k not in available:
            del MODELS[k]

    root = tk.Tk()
    FractureDetectionApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
