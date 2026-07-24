import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import EfficientNetB0

from src.config import (
    IMAGE_SIZE,
    NUM_CLASSES,
    DROPOUT_RATE
)


def build_model():
    """
    Builds an EfficientNetB0 transfer learning model.
    Returns:
        model
        base_model
    """

    # --------------------------------------------------
    # Base Model
    # --------------------------------------------------

    base_model = EfficientNetB0(
        weights="imagenet",
        include_top=False,
        input_shape=(
            IMAGE_SIZE[0],
            IMAGE_SIZE[1],
            3
        )
    )

    # Freeze during initial training
    base_model.trainable = False

    # --------------------------------------------------
    # Input Layer
    # --------------------------------------------------

    inputs = tf.keras.Input(
        shape=(
            IMAGE_SIZE[0],
            IMAGE_SIZE[1],
            3
        )
    )

    # --------------------------------------------------
    # Feature Extraction
    # --------------------------------------------------

    x = base_model(inputs, training=False)

    # --------------------------------------------------
    # Classification Head
    # --------------------------------------------------

    x = layers.GlobalAveragePooling2D()(x)

    x = layers.BatchNormalization()(x)

    x = layers.Dropout(DROPOUT_RATE)(x)

    x = layers.Dense(
        256,
        activation="relu"
    )(x)

    x = layers.Dropout(DROPOUT_RATE)(x)

    outputs = layers.Dense(
        NUM_CLASSES,
        activation="softmax"
    )(x)

    # --------------------------------------------------
    # Final Model
    # --------------------------------------------------

    model = Model(
        inputs,
        outputs
    )

    return model, base_model