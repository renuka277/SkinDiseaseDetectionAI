"""
Project Configuration File
Skin Disease Detection using EfficientNetB0
"""

import tensorflow as tf

# ==========================================================
# Image Configuration
# ==========================================================

IMAGE_HEIGHT = 224
IMAGE_WIDTH = 224
IMAGE_SIZE = (IMAGE_HEIGHT, IMAGE_WIDTH)

CHANNELS = 3

# ==========================================================
# Dataset Configuration
# ==========================================================

BATCH_SIZE = 32

AUTOTUNE = tf.data.AUTOTUNE

RANDOM_STATE = 42

# ==========================================================
# Training Configuration
# ==========================================================

INITIAL_EPOCHS = 5

FINE_TUNE_EPOCHS = 10

TOTAL_EPOCHS = INITIAL_EPOCHS + FINE_TUNE_EPOCHS

INITIAL_LEARNING_RATE = 1e-3

FINE_TUNE_LEARNING_RATE = 1e-5

DROPOUT_RATE = 0.3

# ==========================================================
# Class Names
# ==========================================================

CLASS_NAMES = [
    "MEL",
    "NV",
    "BCC",
    "AKIEC",
    "BKL",
    "DF",
    "VASC"
]

NUM_CLASSES = len(CLASS_NAMES)

# ==========================================================
# Label Mapping
# ==========================================================

label_to_index = {
    label: idx
    for idx, label in enumerate(CLASS_NAMES)
}

index_to_label = {
    idx: label
    for idx, label in enumerate(CLASS_NAMES)
}

# ==========================================================
# Model Saving
# ==========================================================

BEST_MODEL_PATH = "models/best_model.keras"

FINAL_MODEL_PATH = "models/final_model.keras"

# ==========================================================
# TensorBoard Logs
# ==========================================================

LOG_DIR = "logs"

# ==========================================================
# Fine-Tuning
# ==========================================================

UNFREEZE_LAYERS = 30