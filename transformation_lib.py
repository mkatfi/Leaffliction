"""Core image transformations and feature extraction for leaves (OpenCV)."""

import os
from collections import OrderedDict

import cv2
import numpy as np


def read_image(path):
    """Read an image as RGB. Raises ValueError if unreadable."""
    try:
        data = np.fromfile(path, dtype=np.uint8)
    except OSError as err:
        raise ValueError(f"cannot read '{path}': {err}")
    bgr = cv2.imdecode(data, cv2.IMREAD_COLOR) if data.size else None
    if bgr is None:
        raise ValueError(f"'{path}' is not a valid image")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def write_image(path, img):
    """Save an RGB or grayscale image to disk."""
    ext = os.path.splitext(path)[1]
    params = []
    if ext.lower() in (".jpg", ".jpeg"):
        params = [cv2.IMWRITE_JPEG_QUALITY, 95]
    if len(img.shape) == 3 and img.shape[2] == 3:
        bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    else:
        bgr = img
    ok, buf = cv2.imencode(ext, bgr, params)
    if not ok:
        raise ValueError(f"cannot encode '{path}'")
    buf.tofile(path)


def gaussian_blur(img, ksize=(15, 15)):
    """Apply Gaussian blur for smoothing and noise reduction."""
    return cv2.GaussianBlur(img, ksize, 0)


def create_mask(img):
    """Create a clean binary mask segmenting leaf from background."""
    # Convert RGB to LAB - channel 'a' isolates green leaf from background
    bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    _, a, _ = cv2.split(lab)

    # Otsu threshold on inverted 'a' channel
    _, mask = cv2.threshold(a, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Morphological cleaning
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    # Keep only the largest connected component (the leaf)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        cnt = max(contours, key=cv2.contourArea)
        clean = np.zeros_like(mask)
        cv2.drawContours(clean, [cnt], -1, 255, -1)
        return clean
    return mask


def roi_objects(img, mask=None):
    """Highlight leaf contour and bounding ROI rectangle."""
    if mask is None:
        mask = create_mask(img)
    h, w = img.shape[:2]
    res = img.copy()
    cv2.rectangle(res, (0, 0), (w - 1, h - 1), (0, 0, 255), 2)  # Blue ROI box
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        cv2.drawContours(res, contours, -1, (0, 255, 0), 2)  # Green contour
    return res


def analyze_object(img, mask=None):
    """Extract convex hull, center of mass, and principal axes in magenta."""
    if mask is None:
        mask = create_mask(img)
    res = img.copy()
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return res

    cnt = max(contours, key=cv2.contourArea)
    hull = cv2.convexHull(cnt)
    cv2.polylines(res, [hull], True, (255, 0, 255), 2)  # Magenta hull

    moments = cv2.moments(cnt)
    if moments["m00"] > 0:
        cx = int(moments["m10"] / moments["m00"])
        cy = int(moments["m01"] / moments["m00"])
        cv2.circle(res, (cx, cy), 5, (255, 0, 255), -1)

        if len(cnt) >= 5:
            _, axes, angle = cv2.fitEllipse(cnt)
            rad = np.deg2rad(angle)
            sin_a, cos_a = np.sin(rad), np.cos(rad)
            maj_len = axes[1] / 2
            min_len = axes[0] / 2
            pt1 = (int(cx - maj_len * sin_a), int(cy + maj_len * cos_a))
            pt2 = (int(cx + maj_len * sin_a), int(cy - maj_len * cos_a))
            cv2.line(res, pt1, pt2, (255, 0, 255), 2)
            pt3 = (int(cx - min_len * cos_a), int(cy - min_len * sin_a))
            pt4 = (int(cx + min_len * cos_a), int(cy + min_len * sin_a))
            cv2.line(res, pt3, pt4, (255, 0, 255), 2)
    return res


def pseudolandmarks(img, mask=None):
    """Sample morphological landmarks along perimeter and main axis."""
    if mask is None:
        mask = create_mask(img)
    h, w = img.shape[:2]
    res = img.copy()
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return res

    cnt = max(contours, key=cv2.contourArea)
    pts = cnt.reshape(-1, 2)
    indices = np.linspace(0, len(pts) - 1, 28, dtype=int)
    for idx in indices:
        pt = tuple(pts[idx])
        cv2.circle(res, pt, 4, (255, 0, 0), -1)  # Red dots along perimeter

    moments = cv2.moments(cnt)
    if moments["m00"] > 0 and len(cnt) >= 5:
        cx = int(moments["m10"] / moments["m00"])
        cy = int(moments["m01"] / moments["m00"])
        _, axes, angle = cv2.fitEllipse(cnt)
        rad = np.deg2rad(angle)
        sin_a, cos_a = np.sin(rad), np.cos(rad)
        maj_len = axes[1] / 2
        for t in np.linspace(-maj_len * 0.8, maj_len * 0.8, 14):
            c_pt = (int(cx + t * sin_a), int(cy - t * cos_a))
            in_bounds = 0 <= c_pt[0] < w and 0 <= c_pt[1] < h
            if in_bounds and mask[c_pt[1], c_pt[0]] > 0:
                cv2.circle(res, c_pt, 4, (0, 200, 255), -1)  # Cyan axis dots
    return res


def compute_color_histogram(img, mask=None):
    """Compute normalized intensity histograms for 9 color channels."""
    if mask is None:
        mask = create_mask(img)

    bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)

    channels = [
        ("blue", img[:, :, 2], "blue"),
        ("blue-yellow", lab[:, :, 2], "#D4AF37"),
        ("green", img[:, :, 1], "green"),
        ("green-magenta", lab[:, :, 1], "#BA55D3"),
        ("hue", hsv[:, :, 0], "#8A2BE2"),
        ("lightness", lab[:, :, 0], "#4B0082"),
        ("red", img[:, :, 0], "red"),
        ("saturation", hsv[:, :, 1], "cyan"),
        ("value", hsv[:, :, 2], "#FFA500"),
    ]

    leaf_pixels = mask > 0
    total = np.count_nonzero(leaf_pixels)
    if total == 0:
        total = 1

    hist_data = OrderedDict()
    for label, channel_arr, color in channels:
        vals = channel_arr[leaf_pixels]
        hist, _ = np.histogram(vals, bins=256, range=(0, 256))
        pct = (hist / total) * 100
        hist_data[label] = (pct, color)
    return hist_data


TRANSFORMATIONS = OrderedDict([
    ("blur", ("Blur", gaussian_blur)),
    ("mask", ("Mask", create_mask)),
    ("roi", ("Roi", roi_objects)),
    ("analyze", ("Analyze", analyze_object)),
    ("landmarks", ("Landmarks", pseudolandmarks)),
])
