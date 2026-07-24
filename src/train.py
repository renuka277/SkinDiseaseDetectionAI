import os
import datetime

import pandas as pd
import tensorflow as tf

from sklearn.model_selection import train_test_split

from tensorflow.keras.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    ReduceLROnPlateau,
    TensorBoard
)

from src.config import (
    CLASS_NAMES,
    label_to_index,
    BATCH_SIZE,
    RANDOM_STATE,
    INITIAL_EPOCHS,
    FINE_TUNE_EPOCHS,
    INITIAL_LEARNING_RATE,
    FINE_TUNE_LEARNING_RATE,
    BEST_MODEL_PATH,
    FINAL_MODEL_PATH,
    LOG_DIR,
    UNFREEZE_LAYERS
)

from src.dataset_loader import (
    load_and_preprocess_train,
    load_and_preprocess
)

from src.model import build_model


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
# Train / Validation / Test Split
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


# ==========================================================
# Encode Labels
# ==========================================================

train_df["label"] = train_df["label"].map(label_to_index)
val_df["label"] = val_df["label"].map(label_to_index)
test_df["label"] = test_df["label"].map(label_to_index)


# ==========================================================
# TensorFlow Dataset
# ==========================================================

train_dataset = (
    tf.data.Dataset
    .from_tensor_slices(
        (
            train_df["image_path"].values,
            train_df["label"].values
        )
    )
    .shuffle(len(train_df))
    .map(load_and_preprocess_train, num_parallel_calls=tf.data.AUTOTUNE)
    .batch(BATCH_SIZE)
    .prefetch(tf.data.AUTOTUNE)
)

val_dataset = (
    tf.data.Dataset
    .from_tensor_slices(
        (
            val_df["image_path"].values,
            val_df["label"].values
        )
    )
    .map(load_and_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    .batch(BATCH_SIZE)
    .prefetch(tf.data.AUTOTUNE)
)


# ==========================================================
# Build Model
# ==========================================================

model, base_model = build_model()


# ==========================================================
# Stage 1 Training
# ==========================================================

model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=INITIAL_LEARNING_RATE
    ),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)


# ==========================================================
# Callbacks
# ==========================================================

checkpoint = ModelCheckpoint(
    BEST_MODEL_PATH,
    monitor="val_accuracy",
    save_best_only=True,
    verbose=1
)

early_stop = EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True,
    verbose=1
)

reduce_lr = ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.2,
    patience=2,
    verbose=1
)

log_dir = os.path.join(
    LOG_DIR,
    datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
)

tensorboard = TensorBoard(
    log_dir=log_dir
)


# ==========================================================
# Stage 1
# ==========================================================

print("\n========== Stage 1 ==========")
print("Training Classifier Head...\n")

history1 = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=INITIAL_EPOCHS,
    callbacks=[
        checkpoint,
        early_stop,
        reduce_lr,
        tensorboard
    ]
)


# ==========================================================
# Fine Tuning
# ==========================================================

print("\n========== Stage 2 ==========")
print("Fine Tuning EfficientNet...\n")

base_model.trainable = True

for layer in base_model.layers[:-UNFREEZE_LAYERS]:
    layer.trainable = False

for layer in base_model.layers[-UNFREEZE_LAYERS:]:
    layer.trainable = True


# ==========================================================
# Recompile
# ==========================================================

model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=FINE_TUNE_LEARNING_RATE
    ),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)


# ==========================================================
# Stage 2
# ==========================================================

history2 = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=FINE_TUNE_EPOCHS,
    callbacks=[
        checkpoint,
        early_stop,
        reduce_lr,
        tensorboard
    ]
)


# ==========================================================
# Save Final Model
# ==========================================================

model.save(FINAL_MODEL_PATH)

print("\n=================================")
print("Training Completed Successfully!")
print("=================================")