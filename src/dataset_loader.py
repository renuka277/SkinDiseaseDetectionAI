import tensorflow as tf

from tensorflow.keras.applications.efficientnet import preprocess_input

from src.config import IMAGE_SIZE


# ==========================================================
# Data Augmentation
# ==========================================================

data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.1),
    tf.keras.layers.RandomZoom(0.1),
    tf.keras.layers.RandomContrast(0.1),
])


# ==========================================================
# Load Image
# ==========================================================

def load_image(image_path):
    """
    Reads an image from disk.
    """

    image = tf.io.read_file(image_path)

    image = tf.image.decode_jpeg(image, channels=3)

    image = tf.image.resize(image, IMAGE_SIZE)

    image = tf.cast(image, tf.float32)

    return image


# ==========================================================
# Training Preprocessing
# ==========================================================

def load_and_preprocess_train(image_path, label):
    """
    Used for training dataset.
    Includes augmentation.
    """

    image = load_image(image_path)

    image = data_augmentation(image)

    image = preprocess_input(image)

    return image, label


# ==========================================================
# Validation/Test Preprocessing
# ==========================================================

def load_and_preprocess(image_path, label):
    """
    Used for validation and testing.
    No augmentation.
    """

    image = load_image(image_path)

    image = preprocess_input(image)

    return image, label