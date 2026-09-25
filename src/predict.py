import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["TF_NUM_INTRAOP_THREADS"] = "1"
os.environ["TF_NUM_INTEROP_THREADS"] = "1"

import numpy as np
import tensorflow as tf

from src.config import (
    IMAGE_SIZE,
    CLASS_NAMES,
    BEST_MODEL_PATH
)

from src.disease_info import DISEASE_INFO


# ==========================================================
# TensorFlow Memory Optimization
# ==========================================================

tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)


# ==========================================================
# Load Model (only once)
# ==========================================================

model = tf.keras.models.load_model(
    BEST_MODEL_PATH,
    compile=False
)


# ==========================================================
# Preprocess Image
# ==========================================================

def preprocess_image(image_path):
    image = tf.io.read_file(image_path)

    image = tf.image.decode_jpeg(
        image,
        channels=3
    )

    image = tf.image.resize(
        image,
        IMAGE_SIZE
    )

    image = tf.cast(
        image,
        tf.float32
    )

    image = tf.keras.applications.efficientnet.preprocess_input(
        image
    )

    image = tf.expand_dims(
        image,
        axis=0
    )

    return image


# ==========================================================
# Predict Disease
# ==========================================================

def predict(image_path):

    image = preprocess_image(image_path)

    prediction = model(
        image,
        training=False
    )

    prediction = prediction.numpy()[0]

    predicted_index = int(
        np.argmax(prediction)
    )

    confidence = float(
        np.max(prediction)
    ) * 100

    disease_code = CLASS_NAMES[predicted_index]

    info = DISEASE_INFO[disease_code]

    return {
        "disease": info["name"],
        "code": disease_code,
        "confidence": round(confidence, 2),
        "severity": info["severity"],
        "description": info["description"],
        "precautions": info["precautions"]
    }


# ==========================================================
# Test
# ==========================================================

if __name__ == "__main__":

    image_path = input("Enter image path: ")

    if not os.path.exists(image_path):

        print("Image not found!")

    else:

        result = predict(image_path)

        print("\nPrediction")
        print("-" * 40)

        print("Disease      :", result["disease"])
        print("Confidence   :", result["confidence"], "%")
        print("Severity     :", result["severity"])
        print("Description  :", result["description"])
        print("Precautions  :", result["precautions"])