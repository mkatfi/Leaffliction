#!/usr/bin/env python3
"""Part 2 - Balance the data set with data augmentation.

Usage: ./Balance.py <directory> [-dst augmented_directory] [--seed N]

Copies the data set into "augmented_directory" (the original is untouched),
then adds augmented images (same 6 techniques as Augmentation.py) to the
smallest classes until every class of a plant type has as many images as
the biggest class of that plant type.
"""

import argparse
import os
import random
import shutil
import sys

from augmentation_lib import AUGMENTATIONS, apply_and_save, is_augmented
from dataset import group_by_plant, scan_dataset


def parse_args():
    parser = argparse.ArgumentParser(
        description="Balance a leaf data set with data augmentation.")
    parser.add_argument("directory", help="data set directory "
                                          "(ex: ./images or ./Apple)")
    parser.add_argument("-dst", "--dst", default="augmented_directory",
                        help="output directory (default: augmented_directory)")
    parser.add_argument("--seed", type=int, default=42,
                        help="random seed, for reproducible results")
    return parser.parse_args()


def balance_class(cls, target, rng):
    """Create `target - current` augmented images inside the class folder.

    Each original image can give up to 6 new images (one per technique).
    Returns the number of images created.
    """
    originals = [p for p in cls["images"] if not is_augmented(p)]
    deficit = target - len(cls["images"])
    # for every original image, a random order of the 6 techniques
    plans = {p: rng.sample(list(AUGMENTATIONS), len(AUGMENTATIONS))
             for p in originals}
    created = 0
    for round_index in range(len(AUGMENTATIONS)):
        order = originals[:]
        rng.shuffle(order)
        for path in order:
            if created >= deficit:
                return created
            apply_and_save(path, plans[path][round_index])
            created += 1
    return created


def main():
    args = parse_args()
    src, dst = args.directory, args.dst
    try:
        scan_dataset(src)  # validates the source
    except ValueError as err:
        print(f"Error: {err}", file=sys.stderr)
        return 1
    if os.path.exists(dst):
        print(f"Error: '{dst}' already exists, remove it or choose "
              f"another -dst", file=sys.stderr)
        return 1
    if os.path.abspath(dst).startswith(os.path.abspath(src) + os.sep):
        print("Error: destination can not be inside the source",
              file=sys.stderr)
        return 1

    print(f"Copying '{src}' to '{dst}' ...")
    shutil.copytree(src, dst)
    rng = random.Random(args.seed)

    for plant, classes in group_by_plant(scan_dataset(dst)).items():
        target = max(len(c["images"]) for c in classes)
        print(f"[{plant}] target: {target} images per class")
        for cls in classes:
            before = len(cls["images"])
            created = balance_class(cls, target, rng)
            print(f"  {cls['name']:<25} {before:>5} -> {before + created:>5}"
                  f"  (+{created})")
            if before + created < target:
                print("  warning: not enough originals to reach the target")
    print(f"Done. Balanced data set: {dst}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
