"""Helpers to explore a leaf data set (directory of class sub-directories)."""

import os

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp")


def is_image(filename):
    """Return True if the file name has an image extension."""
    return (not filename.startswith(".")
            and filename.lower().endswith(IMAGE_EXTENSIONS))


def list_images(directory):
    """Return the sorted image paths located directly in `directory`."""
    names = sorted(os.listdir(directory))
    paths = [os.path.join(directory, n) for n in names if is_image(n)]
    return [p for p in paths if os.path.isfile(p)]


def plant_name(class_name, parent_name):
    """Guess the plant type of a class.

    'Apple_Black_rot' -> 'Apple'. If the class name has no '_',
    the name of its parent directory is used.
    """
    if "_" in class_name:
        return class_name.split("_", 1)[0]
    return parent_name


def scan_dataset(root):
    """Walk `root` and return the classes found in its sub-directories.

    A class is any sub-directory that directly contains images.
    Returns a list of dicts: {"name", "plant", "path", "images"},
    sorted by plant then class name.
    Raises ValueError if `root` is not a directory or has no class.
    """
    if not os.path.isdir(root):
        raise ValueError(f"'{root}' is not a directory")
    classes = []
    for current, dirs, _ in os.walk(root, followlinks=True):
        dirs[:] = sorted(d for d in dirs if not d.startswith("."))
        if os.path.abspath(current) == os.path.abspath(root):
            continue  # only look at the sub-directories
        images = list_images(current)
        if images:
            name = os.path.basename(current)
            parent = os.path.basename(os.path.dirname(current))
            classes.append({
                "name": name,
                "plant": plant_name(name, parent),
                "path": current,
                "images": images,
            })
    if not classes:
        raise ValueError(f"no sub-directory containing images in '{root}'")
    classes.sort(key=lambda c: (c["plant"].lower(), c["name"].lower()))
    return classes


def group_by_plant(classes):
    """Group classes by plant type (case-insensitive).

    Returns an ordered dict {plant display name: [class, ...]}.
    """
    groups = {}
    keys = {}
    for cls in classes:
        key = cls["plant"].lower()
        display = keys.setdefault(key, cls["plant"])
        groups.setdefault(display, []).append(cls)
    return groups
