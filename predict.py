#!/usr/bin/env python3
"""
predict.py - Part 4: Classification (prediction program).

Usage:
    ./predict.py ./Apple/apple_healthy/image\\ \\(1\\).JPG

Loads the model produced by train.py, displays the original image next
to its leaf-segmented ("transformed") version, and prints/shows the
predicted disease.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import tempfile
import zipfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import model_utils as mu  # noqa: E402
import transformation as tr  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Predict the disease of a leaf image.",
    )
    parser.add_argument("image", help="Path to the leaf image.")
    parser.add_argument(
        "--zip", dest="zip_path", default=None,
        help="Path to the *_learnings.zip produced by train.py "
             "(default: auto-detect a *_learnings.zip in cwd).",
    )
    parser.add_argument(
        "--model-dir", dest="model_dir", default=None,
        help="Path to an already-extracted directory containing "
             "model.keras and labels.json.",
    )
    parser.add_argument("--img-size", type=int, default=128)
    parser.add_argument(
        "--no-display", action="store_true",
        help="Do not open a window; only print the result and save "
             "a PNG next to the input image.",
    )
    return parser.parse_args()


def _locate_model(args: argparse.Namespace) -> str:
    if args.model_dir:
        return args.model_dir

    zip_path = args.zip_path
    if not zip_path:
        candidates = sorted(glob.glob("*_learnings.zip"))
        if not candidates:
            print(
                "error: no *_learnings.zip found in the current "
                "directory. Train one first with train.py, or pass "
                "--zip/--model-dir explicitly.",
                file=sys.stderr,
            )
            sys.exit(1)
        if len(candidates) > 1:
            print(
                f"note: multiple learnings found ({candidates}); "
                f"using {candidates[0]}. Pass --zip to choose another.",
            )
        zip_path = candidates[0]

    extract_dir = tempfile.mkdtemp(prefix="leaffliction_model_")
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(extract_dir)
    print(f"(using model from {zip_path})")
    return extract_dir


def main() -> None:
    args = parse_args()
    if not os.path.isfile(args.image):
        print(f"error: no such file: {args.image}", file=sys.stderr)
        sys.exit(1)

    model_dir = _locate_model(args)
    model_path = os.path.join(model_dir, "model.keras")
    labels_path = os.path.join(model_dir, "labels.json")
    if not os.path.isfile(model_path) or not os.path.isfile(labels_path):
        print(
            f"error: {model_dir} does not contain model.keras and "
            "labels.json",
            file=sys.stderr,
        )
        sys.exit(1)

    import cv2
    from tensorflow import keras

    model = keras.models.load_model(model_path)
    with open(labels_path) as fh:
        labels = json.load(fh)

    image_bgr = cv2.imread(args.image)
    if image_bgr is None:
        print(f"error: could not read image {args.image}", file=sys.stderr)
        sys.exit(1)

    mask = tr.create_mask(image_bgr)
    transformed_bgr = tr.apply_mask(image_bgr, mask)

    batch = mu.preprocess_for_inference(image_bgr, args.img_size)
    probs = model.predict(batch, verbose=0)[0]
    pred_idx = int(probs.argmax())
    pred_label = labels[pred_idx]
    confidence = float(probs[pred_idx]) * 100

    print("=" * 48)
    print(f"  Class predicted : {pred_label}")
    print(f"  Confidence      : {confidence:.1f}%")
    print("=" * 48)
    top3 = sorted(
        zip(labels, probs), key=lambda kv: kv[1], reverse=True,
    )[:3]
    for name, p in top3:
        print(f"    {name:<30s} {p * 100:5.1f}%")

    _show(image_bgr, transformed_bgr, pred_label, confidence, args)


def _show(image_bgr, transformed_bgr, pred_label, confidence, args):
    import matplotlib
    if args.no_display:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(8, 4.2))
    axes[0].imshow(cv2_to_rgb(image_bgr))
    axes[0].set_title("Original")
    axes[0].axis("off")
    axes[1].imshow(cv2_to_rgb(transformed_bgr))
    axes[1].set_title("Transformed (leaf mask)")
    axes[1].axis("off")
    fig.suptitle(f"Class predicted: {pred_label}  ({confidence:.1f}%)")
    fig.tight_layout()

    out_png = os.path.splitext(args.image)[0] + "_prediction.png"
    fig.savefig(out_png, dpi=120)
    print(f"(figure saved to {out_png})")
    if not args.no_display:
        plt.show()


def cv2_to_rgb(image_bgr):
    import cv2
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


if __name__ == "__main__":
    main()
