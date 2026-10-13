#!/usr/bin/env python3
"""
train.py - Part 4: Classification (training program).

Usage:
    ./train.py ./Apple/

Reads every sub-directory of the given directory as one disease class,
balances/augments the training images, trains a CNN to recognise the
disease from a leaf photo, and packages the model together with the
augmented ("increased/modified") images into a .zip file, as required
by the subject.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import dataset_utils as ds  # noqa: E402
import model_utils as mu  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a leaf disease classifier.",
    )
    parser.add_argument(
        "directory",
        help="Path to a plant directory containing one "
             "sub-directory of images per disease class.",
    )
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--img-size", type=int, default=128)
    parser.add_argument("--val-split", type=float, default=0.2)
    parser.add_argument("--min-val", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output", default=None,
        help="Name of the output .zip file "
             "(default: <directory-name>_learnings.zip).",
    )
    parser.add_argument(
        "--work-dir", default=None,
        help="Where to write the augmented_directory and model "
             "artifacts (default: ./<directory-name>_training/).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    directory = args.directory.rstrip("/")
    if not os.path.isdir(directory):
        print(f"error: no such directory: {directory}", file=sys.stderr)
        sys.exit(1)

    plant_name = os.path.basename(os.path.abspath(directory))
    work_dir = args.work_dir or f"./{plant_name}_training"
    augmented_dir = os.path.join(work_dir, "augmented_directory")
    os.makedirs(work_dir, exist_ok=True)

    print(f"[1/6] Loading dataset from {directory} ...")
    items = ds.load_dataset(directory)
    classes = ds.list_classes(directory)
    print(f"       {len(items)} images across {len(classes)} classes:")
    for name, count in ds.class_counts(items).items():
        print(f"         - {name}: {count}")

    print("[2/6] Splitting into train / validation (validation is "
          "held out untouched, before any augmentation) ...")
    train_items, val_items = ds.train_val_split(
        items, val_ratio=args.val_split,
        min_val=args.min_val, seed=args.seed,
    )
    print(f"       train={len(train_items)}  val={len(val_items)}")

    print("[3/6] Balancing classes with augmentation "
          "(Flip, Rotate, Skew, Shear, Crop, Distortion) ...")
    t0 = time.time()
    train_items = ds.balance_and_augment(
        train_items, augmented_dir, seed=args.seed,
    )
    print(f"       wrote {len(train_items)} training images to "
          f"{augmented_dir} in {time.time() - t0:.1f}s")
    for name, count in ds.class_counts(train_items).items():
        print(f"         - {name}: {count}")

    print("[4/6] Building tf.data pipelines and the model ...")
    label_to_id = {name: i for i, name in enumerate(classes)}
    train_ds = mu.make_dataset(
        train_items, label_to_id, args.img_size, args.batch_size,
        shuffle=True,
    )
    val_ds = mu.make_dataset(
        val_items, label_to_id, args.img_size, args.batch_size,
        shuffle=False,
    )
    model = mu.build_model(len(classes), args.img_size)
    model.summary()

    print(f"[5/6] Training for up to {args.epochs} epochs ...")
    import tensorflow as tf

    ckpt_path = os.path.join(work_dir, "best.keras")
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy", patience=6,
            restore_best_weights=True,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=3,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            ckpt_path, monitor="val_accuracy", save_best_only=True,
        ),
    ]
    history = model.fit(
        train_ds, validation_data=val_ds,
        epochs=args.epochs, callbacks=callbacks,
    )

    print("[6/6] Evaluating and packaging the results ...")
    val_loss, val_acc = model.evaluate(val_ds, verbose=0)
    print(f"       validation accuracy: {val_acc * 100:.2f}% "
          f"(n={len(val_items)})")
    if val_acc < 0.90:
        print("       warning: accuracy is below the 90% target; "
              "consider more epochs, more data, or a larger model.")

    history_png = os.path.join(work_dir, "training_history.png")
    _plot_history(history.history, history_png)

    model_path = os.path.join(work_dir, "model.keras")
    labels_path = os.path.join(work_dir, "labels.json")
    model.save(model_path)
    mu.save_labels(classes, labels_path)

    output_zip = args.output or f"{plant_name}_learnings.zip"
    mu.zip_artifacts(
        output_zip, model_path, labels_path, augmented_dir,
        extra_files=[history_png],
    )
    print(f"\nDone. Learnings saved to: {output_zip}")
    print("      (contains model.keras, labels.json, "
          "training_history.png and the augmented_directory)")
    _print_sha1(output_zip)


def _plot_history(history: dict, out_path: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(history["accuracy"], label="train")
    axes[0].plot(history["val_accuracy"], label="val")
    axes[0].set_title("Accuracy")
    axes[0].set_xlabel("epoch")
    axes[0].legend()

    axes[1].plot(history["loss"], label="train")
    axes[1].plot(history["val_loss"], label="val")
    axes[1].set_title("Loss")
    axes[1].set_xlabel("epoch")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(out_path, dpi=120)


def _print_sha1(path: str) -> None:
    import hashlib

    h = hashlib.sha1()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    print(f"sha1({path}) = {h.hexdigest()}")


if __name__ == "__main__":
    main()
