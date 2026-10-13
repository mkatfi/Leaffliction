"""
Model architecture, data pipelines, and artifact packaging helpers
shared by train.py and predict.py.
"""
from __future__ import annotations

import json
import os
import zipfile

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def build_model(num_classes: int, img_size: int = 128) -> keras.Model:
    """A compact CNN trained from scratch (no external weights needed,
    so training never depends on internet access)."""
    inputs = keras.Input(shape=(img_size, img_size, 3))
    x = layers.Rescaling(1.0 / 255)(inputs)

    x = layers.RandomFlip("horizontal")(x)
    x = layers.RandomRotation(0.05)(x)

    for filters in (32, 64, 128, 128):
        x = layers.Conv2D(filters, 3, padding="same", activation="relu")(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D()(x)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.4)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = keras.Model(inputs, outputs)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def _load_and_resize(path: tf.Tensor, label: tf.Tensor, img_size: int):
    raw = tf.io.read_file(path)
    image = tf.io.decode_image(
        raw, channels=3, expand_animations=False,
    )
    image = tf.image.resize(image, (img_size, img_size))
    image.set_shape((img_size, img_size, 3))
    return image, label


def make_dataset(
    items: list[tuple[str, str]],
    label_to_id: dict[str, int],
    img_size: int,
    batch_size: int,
    shuffle: bool,
) -> tf.data.Dataset:
    paths = [p for p, _ in items]
    labels = [label_to_id[c] for _, c in items]
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(paths), seed=42)
    ds = ds.map(
        lambda p, y: _load_and_resize(p, y, img_size),
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    ds = ds.batch(batch_size)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


def save_labels(labels: list[str], path: str) -> None:
    with open(path, "w") as fh:
        json.dump(labels, fh, indent=2)


def load_labels(path: str) -> list[str]:
    with open(path) as fh:
        return json.load(fh)


def preprocess_for_inference(image_bgr: np.ndarray, img_size: int):
    """OpenCV BGR image -> model-ready RGB batch of shape (1, s, s, 3)."""
    import cv2

    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (img_size, img_size))
    return np.expand_dims(resized.astype("float32"), axis=0)


def zip_artifacts(
    output_zip: str,
    model_path: str,
    labels_path: str,
    augmented_dir: str,
    extra_files: list[str] | None = None,
) -> str:
    """
    Bundle the trained model, the label map and the augmented image
    directory (the "increased/modified images") into a single .zip,
    as required by the subject.
    """
    extra_files = extra_files or []
    with zipfile.ZipFile(output_zip, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(model_path, arcname=os.path.basename(model_path))
        zf.write(labels_path, arcname=os.path.basename(labels_path))
        for extra in extra_files:
            if os.path.isfile(extra):
                zf.write(extra, arcname=os.path.basename(extra))
        base = os.path.basename(os.path.normpath(augmented_dir))
        for root, _, files in os.walk(augmented_dir):
            for fname in files:
                full = os.path.join(root, fname)
                rel = os.path.relpath(full, augmented_dir)
                zf.write(full, arcname=os.path.join(base, rel))
    return output_zip
