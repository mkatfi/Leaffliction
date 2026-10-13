"""
Dataset loading, train/validation splitting and class balancing.

The validation split is taken BEFORE any augmentation happens, and
validation images are never augmented or otherwise modified, so that
train.py has no way of "seeing" augmented variants of a validation
leaf during training (this is the "no overfitting / no cheating"
requirement of Part 4).
"""
from __future__ import annotations

import os
import random
import shutil
from collections import defaultdict

import cv2

import augmentation_lib as aug

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG")


def list_classes(directory: str) -> list[str]:
    """Return the sorted list of class (sub-directory) names."""
    return sorted(
        d for d in os.listdir(directory)
        if os.path.isdir(os.path.join(directory, d))
    )


def load_dataset(directory: str) -> list[tuple[str, str]]:
    """
    Walk `directory`, treating each immediate sub-directory as a class.
    Returns a list of (image_path, class_name) pairs.
    """
    items: list[tuple[str, str]] = []
    for class_name in list_classes(directory):
        class_dir = os.path.join(directory, class_name)
        for root, _, files in os.walk(class_dir):
            for fname in files:
                if fname.endswith(IMAGE_EXTENSIONS):
                    items.append((os.path.join(root, fname), class_name))
    if not items:
        raise ValueError(
            f"No images found under {directory}. Expected one "
            "sub-directory per class, each containing images."
        )
    return items


def train_val_split(
    items: list[tuple[str, str]],
    val_ratio: float = 0.2,
    min_val: int = 100,
    seed: int = 42,
) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    """
    Stratified split (per class) into (train, val), guaranteeing the
    validation set has at least `min_val` images in total whenever the
    dataset is large enough to allow it.
    """
    rng = random.Random(seed)
    by_class: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for item in items:
        by_class[item[1]].append(item)

    total = len(items)
    effective_ratio = val_ratio
    if total * val_ratio < min_val and total >= min_val:
        effective_ratio = min_val / total

    train, val = [], []
    for class_name, class_items in by_class.items():
        rng.shuffle(class_items)
        n_val = max(1, round(len(class_items) * effective_ratio))
        val.extend(class_items[:n_val])
        train.extend(class_items[n_val:])

    if len(val) < min_val:
        print(
            f"warning: only {len(val)} validation images available "
            f"(wanted >= {min_val}); the data set is small.",
        )
    return train, val


def balance_and_augment(
    train_items: list[tuple[str, str]],
    output_dir: str,
    seed: int = 42,
) -> list[tuple[str, str]]:
    """
    Copy every training image into output_dir/<class>/, then generate
    just enough augmented variants (Flip, Rotate, Skew, Shear, Crop,
    Distortion - cycled in that order) per class so all classes reach
    the size of the largest class.

    Returns the new list of (path, class_name) pairs living under
    output_dir - this directory is the "increased/modified images"
    the subject asks train.py to produce.
    """
    rng = random.Random(seed)
    by_class: dict[str, list[str]] = defaultdict(list)
    for path, class_name in train_items:
        by_class[class_name].append(path)

    if os.path.isdir(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    target = max(len(paths) for paths in by_class.values())
    aug_names = list(aug.AUGMENTATIONS.keys())
    new_items: list[tuple[str, str]] = []

    for class_name, paths in by_class.items():
        class_out = os.path.join(output_dir, class_name)
        os.makedirs(class_out, exist_ok=True)

        # 1) copy the originals untouched
        for src in paths:
            dst = os.path.join(class_out, os.path.basename(src))
            shutil.copy2(src, dst)
            new_items.append((dst, class_name))

        # 2) top up with augmented variants until we reach `target`.
        # Cycle through (image, augmentation-type) pairs; once a full
        # cycle is exhausted, start a new cycle with a unique suffix
        # so file names never collide (the augmentation functions are
        # randomized internally, so a repeated pair still yields a
        # visually different image).
        needed = target - len(paths)
        if needed <= 0:
            continue
        pool = paths.copy()
        rng.shuffle(pool)
        combo_i = 0
        pairs_per_cycle = len(pool) * len(aug_names)
        while needed > 0:
            cycle, offset = divmod(combo_i, pairs_per_cycle)
            src_i, aug_i = divmod(offset, len(aug_names))
            src = pool[src_i]
            aug_name = aug_names[aug_i]
            stem, ext = os.path.splitext(os.path.basename(src))
            suffix = f"_{aug_name}" if cycle == 0 else f"_{aug_name}{cycle}"
            out_path = os.path.join(class_out, f"{stem}{suffix}{ext}")
            image = aug.AUGMENTATIONS[aug_name](cv2.imread(src))
            cv2.imwrite(out_path, image)
            new_items.append((out_path, class_name))
            needed -= 1
            combo_i += 1

    return new_items


def class_counts(items: list[tuple[str, str]]) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for _, class_name in items:
        counts[class_name] += 1
    return dict(sorted(counts.items()))
