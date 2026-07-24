import os

import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

from sklearn.model_selection import train_test_split

from src.config import (
    CLASS_NAMES,
    label_to_index,
    BATCH_SIZE,
    RANDOM_STATE,
    BEST_MODEL_PATH
)

from src.dataset_loader import load_and_preprocess


# ==========================================================
# Paths
# ==========================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CSV_PATH = os.path.join(BASE_DIR, "dataset", "GroundTruth.csv")

IMAGE_DIR = os.path.join(BASE_DIR, "dataset", "images")


# ==========================================================
# Load Dataset
# ==========================================================

df = pd.read_csv(CSV_PATH)

df["label"] = df[CLASS_NAMES].idxmax(axis=1)

df["image_path"] = df["image"].apply(
    lambda x: os.path.join(IMAGE_DIR, x + ".jpg")
)


# ==========================================================
# Dataset Split
# ==========================================================

train_df, test_df = train_test_split(
    df,
    test_size=0.15,
    stratify=df["label"],
    random_state=RANDOM_STATE
)

train_df, val_df = train_test_split(
    train_df,
    test_size=0.15,
    stratify=train_df["label"],
    random_state=RANDOM_STATE
)

test_df["label"] = test_df["label"].map(label_to_index)


# ==========================================================
# Test Dataset
# ==========================================================

test_dataset = (
    tf.data.Dataset
    .from_tensor_slices(
        (
            test_df["image_path"].values,
            test_df["label"].values
        )
    )
    .map(load_and_preprocess)
    .batch(BATCH_SIZE)
    .prefetch(tf.data.AUTOTUNE)
)


# ==========================================================
# Load Model
# ==========================================================

print("\nLoading Best Model...\n")

model = tf.keras.models.load_model(BEST_MODEL_PATH)


# ==========================================================
# Prediction
# ==========================================================

print("Evaluating Model...\n")

predictions = model.predict(test_dataset)

y_pred = np.argmax(predictions, axis=1)

y_true = test_df["label"].values


# ==========================================================
# Accuracy
# ==========================================================

accuracy = accuracy_score(y_true, y_pred)

print("=" * 50)
print(f"Test Accuracy : {accuracy:.4f}")
print("=" * 50)


# ==========================================================
# Classification Report
# ==========================================================

print("\nClassification Report\n")

print(
    classification_report(
        y_true,
        y_pred,
        target_names=CLASS_NAMES
    )
)


# ==========================================================
# Confusion Matrix
# ==========================================================

cm = confusion_matrix(
    y_true,
    y_pred
)

print("\nConfusion Matrix\n")

print(cm)


# ==========================================================
# Plot Confusion Matrix
# ==========================================================

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=CLASS_NAMES
)

fig, ax = plt.subplots(figsize=(8, 8))

disp.plot(
    ax=ax,
    cmap="Blues",
    values_format="d"
)

plt.title("Skin Disease Confusion Matrix")
plt.show()


# ==========================================================
# Sample Predictions
# ==========================================================

print("\nSample Predictions\n")

for i in range(10):

    actual = CLASS_NAMES[y_true[i]]

    predicted = CLASS_NAMES[y_pred[i]]

    confidence = np.max(predictions[i]) * 100

    print(
        f"{i+1}. Actual: {actual:<8}"
        f" Predicted: {predicted:<8}"
        f" Confidence: {confidence:.2f}%"
    )