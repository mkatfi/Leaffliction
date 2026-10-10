"""The 6 data augmentation techniques used by the project (OpenCV)."""

import os
from collections import OrderedDict

import cv2
import numpy as np


def background_color(img):
    """Median colour of the image edges (used to fill empty corners)."""
    edges = np.concatenate([img[0], img[-1], img[:, 0], img[:, -1]])
    return tuple(float(v) for v in np.median(edges, axis=0))


def fill(img):
    """OpenCV border options: empty areas get the background colour."""
    return {"borderMode": cv2.BORDER_CONSTANT,
            "borderValue": background_color(img)}


def flip(img):
    """Mirror the image horizontally."""
    return cv2.flip(img, 1)


def rotate(img, angle=30):
    """Rotate the image around its centre."""
    h, w = img.shape[:2]
    matrix = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    return cv2.warpAffine(img, matrix, (w, h), **fill(img))


def skew(img, ratio=0.15):
    """Perspective skew: the top edge is squeezed (trapezoid)."""
    h, w = img.shape[:2]
    shift = w * ratio
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    dst = np.float32([[shift, 0], [w - shift, 0], [w, h], [0, h]])
    matrix = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(img, matrix, (w, h), **fill(img))


def shear(img, factor=0.25):
    """Horizontal shear (affine transform)."""
    h, w = img.shape[:2]
    matrix = np.float32([[1, factor, -factor * h / 2], [0, 1, 0]])
    return cv2.warpAffine(img, matrix, (w, h), **fill(img))


def crop(img, ratio=0.75):
    """Keep the centre of the image and resize it to the original size."""
    h, w = img.shape[:2]
    ch, cw = int(h * ratio), int(w * ratio)
    top, left = (h - ch) // 2, (w - cw) // 2
    centre = img[top:top + ch, left:left + cw]
    return cv2.resize(centre, (w, h), interpolation=cv2.INTER_LINEAR)


def distortion(img, strength=0.25):
    """Barrel distortion: pixels move according to their distance to centre."""
    h, w = img.shape[:2]
    xs, ys = np.meshgrid(np.linspace(-1, 1, w), np.linspace(-1, 1, h))
    factor = 1 + strength * (xs ** 2 + ys ** 2)
    map_x = ((xs * factor + 1) * (w - 1) / 2).astype(np.float32)
    map_y = ((ys * factor + 1) * (h - 1) / 2).astype(np.float32)
    return cv2.remap(img, map_x, map_y, cv2.INTER_LINEAR, **fill(img))


AUGMENTATIONS = OrderedDict([
    ("Flip", flip),
    ("Rotate", rotate),
    ("Skew", skew),
    ("Shear", shear),
    ("Crop", crop),
    ("Distortion", distortion),
])


def read_image(path):
    """Read an image (BGR). Raises ValueError if it can not be decoded."""
    try:
        data = np.fromfile(path, dtype=np.uint8)
    except OSError as err:
        raise ValueError(f"cannot read '{path}': {err}")
    img = cv2.imdecode(data, cv2.IMREAD_COLOR) if data.size else None
    if img is None:
        raise ValueError(f"'{path}' is not a valid image")
    return img


def write_image(path, img):
    """Write an image, keeping the extension of `path`."""
    ext = os.path.splitext(path)[1]
    params = []
    if ext.lower() in (".jpg", ".jpeg"):
        params = [cv2.IMWRITE_JPEG_QUALITY, 95]
    ok, buf = cv2.imencode(ext, img, params)
    if not ok:
        raise ValueError(f"cannot encode '{path}'")
    buf.tofile(path)


def augmented_path(path, name, out_dir=None):
    """'dir/image (1).JPG' + 'Flip' -> 'dir/image (1)_Flip.JPG'."""
    folder, filename = os.path.split(path)
    stem, ext = os.path.splitext(filename)
    return os.path.join(out_dir or folder, f"{stem}_{name}{ext}")


def is_augmented(path):
    """True if the file name already ends with '_<augmentation name>'."""
    stem = os.path.splitext(os.path.basename(path))[0]
    return any(stem.endswith("_" + name) for name in AUGMENTATIONS)


def apply_and_save(path, name, img=None, out_dir=None):
    """Apply one augmentation to an image file and save the result."""
    if img is None:
        img = read_image(path)
    new_path = augmented_path(path, name, out_dir)
    write_image(new_path, AUGMENTATIONS[name](img))
    return new_path
