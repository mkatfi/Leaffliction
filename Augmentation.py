#!/usr/bin/env python3
"""Part 2 - Data augmentation of one image.

Usage: ./Augmentation.py <image> [--no-display]

Creates 6 augmented versions of the image (Flip, Rotate, Skew, Shear, Crop,
Distortion). They are saved next to the original, named
<original name>_<Augmentation>.<ext>, and displayed with the original.
"""

import argparse
import os
import sys

import cv2

from augmentation_lib import AUGMENTATIONS, apply_and_save, read_image
from dataset import is_image
from viewer import plt, show


def parse_args():
    parser = argparse.ArgumentParser(
        description="Apply 6 data augmentations to an image.")
    parser.add_argument("image",
                        help="path of the image "
                             "(ex: ./Apple/apple_healthy/image (1).JPG)")
    parser.add_argument("--no-display", action="store_true",
                        help="only save the files, do not open a window")
    return parser.parse_args()


def display(original, results):
    """Show the original image and its 6 augmentations on one figure."""
    fig, axes = plt.subplots(1, len(results) + 1, figsize=(18, 3.5))
    images = [("Original", original)] + list(results.items())
    for ax, (title, img) in zip(axes, images):
        ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        ax.set_title(title)
        ax.axis("off")
    fig.tight_layout()
    show("augmentation_preview.png")


def main():
    args = parse_args()
    path = args.image
    if not os.path.isfile(path):
        print(f"Error: '{path}' is not a file", file=sys.stderr)
        return 1
    if not is_image(os.path.basename(path)):
        print(f"Error: '{path}' is not a supported image", file=sys.stderr)
        return 1
    try:
        original = read_image(path)
    except ValueError as err:
        print(f"Error: {err}", file=sys.stderr)
        return 1

    results = {}
    for name in AUGMENTATIONS:
        new_path = apply_and_save(path, name, img=original)
        results[name] = read_image(new_path)
        print(f"saved: {new_path}")
    if not args.no_display:
        display(original, results)
    return 0


if __name__ == "__main__":
    sys.exit(main())
