#!/usr/bin/env python3
"""Part 3 - Direct feature extraction and image transformation.

Usage:
  Single image display mode:
    ./Transformation.py <image_path> [--no-display]

  Batch directory processing mode:
    ./Transformation.py -src <source_dir> -dst <dest_dir> [-mask] [-blur]
                        [-roi] [-analyze] [-landmarks] [-all]
"""

import argparse
import os
import sys

from dataset import is_image
from transformation_lib import (
    TRANSFORMATIONS,
    analyze_object,
    compute_color_histogram,
    create_mask,
    gaussian_blur,
    read_image,
    roi_objects,
    write_image,
)
from viewer import plt, show


def parse_args():
    """Parse CLI arguments for single image or batch transformation."""
    parser = argparse.ArgumentParser(
        description="Extract characteristics and transform leaf images.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "image",
        nargs="?",
        default=None,
        help="path to a single leaf image to analyze and display",
    )
    parser.add_argument(
        "-src",
        "--src",
        metavar="DIR",
        help="source directory containing images for batch processing",
    )
    parser.add_argument(
        "-dst",
        "--dst",
        metavar="DIR",
        help="destination directory where transformed images will be saved",
    )
    parser.add_argument(
        "-mask",
        "--mask",
        action="store_true",
        help="generate and save binary leaf mask",
    )
    parser.add_argument(
        "-blur",
        "--blur",
        action="store_true",
        help="generate and save Gaussian blurred image",
    )
    parser.add_argument(
        "-roi",
        "--roi",
        action="store_true",
        help="generate and save ROI bounding objects image",
    )
    parser.add_argument(
        "-analyze",
        "--analyze",
        action="store_true",
        help="generate and save morphological analyze object image",
    )
    parser.add_argument(
        "-landmarks",
        "--landmarks",
        action="store_true",
        help="generate and save pseudolandmarks image",
    )
    parser.add_argument(
        "-all",
        "--all",
        action="store_true",
        help="generate and save all transformations in batch mode",
    )
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="do not open window, save figures instead",
    )
    return parser.parse_args()


def display_transformations(path, no_display=False):
    """Show the 6 image transformations and the color histogram."""
    try:
        rgb = read_image(path)
    except ValueError as err:
        print(f"Error: {err}", file=sys.stderr)
        return 1

    mask = create_mask(rgb)
    blur = gaussian_blur(rgb)
    roi = roi_objects(rgb, mask)
    analyze = analyze_object(rgb, mask)
    landmarks = analyze_object_landmarks = None  # noqa: F841
    from transformation_lib import pseudolandmarks
    landmarks = pseudolandmarks(rgb, mask)

    # 1. Six transformation figures
    fig1, axes = plt.subplots(3, 2, figsize=(8, 10))
    titles = [
        "Figure IV.1: Original",
        "Figure IV.2: Gaussian blur",
        "Figure IV.3: Mask",
        "Figure IV.4: Roi objects",
        "Figure IV.5: Analyze object",
        "Figure IV.6: Pseudolandmarks",
    ]
    imgs = [rgb, blur, mask, roi, analyze, landmarks]

    for ax, title, im in zip(axes.flat, titles, imgs):
        if len(im.shape) == 2:
            ax.imshow(im, cmap="gray")
        else:
            ax.imshow(im)
        ax.set_title(title, fontsize=10, pad=6)
        ax.tick_params(labelsize=8)

    fig1.tight_layout()

    # 2. Color histogram figure
    hist_data = compute_color_histogram(rgb, mask)
    fig2, ax_hist = plt.subplots(figsize=(8, 6))
    for label, (pct, color) in hist_data.items():
        ax_hist.plot(range(256), pct, label=label, color=color, linewidth=1.2)

    ax_hist.set_title("Figure IV.7: Color histogram", fontsize=11, pad=10)
    ax_hist.set_xlabel("Pixel intensity", fontsize=10)
    ax_hist.set_ylabel("Proportion of pixels (%)", fontsize=10)
    ax_hist.set_xlim(0, 256)
    ax_hist.set_ylim(0, 11)
    ax_hist.grid(True, linestyle="-", alpha=0.3)
    ax_hist.legend(
        title="color Channel",
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        frameon=True,
        fontsize=8,
    )
    fig2.tight_layout()

    if not no_display:
        show("transformation_preview.png")
    else:
        fig1.savefig("transformation_preview.png")
        fig2.savefig("histogram_preview.png")
        print("Previews saved: transformation_preview.png, "
              "histogram_preview.png")
    return 0


def collect_images(directory):
    """Recursively collect all image paths in a directory."""
    images = []
    for root, _, files in os.walk(directory):
        for f in sorted(files):
            if is_image(f):
                images.append(os.path.join(root, f))
    return images


def run_batch(src, dst, selected_keys):
    """Process all images in src directory and save results to dst."""
    if not os.path.isdir(src):
        print(f"Error: source '{src}' is not a directory", file=sys.stderr)
        return 1
    if not dst:
        print("Error: -dst destination directory is required", file=sys.stderr)
        return 1

    images = collect_images(src)
    if not images:
        print(f"Error: no images found in '{src}'", file=sys.stderr)
        return 1

    os.makedirs(dst, exist_ok=True)
    print(f"Processing {len(images)} images from '{src}' -> '{dst}' ...")

    for idx, img_path in enumerate(images, 1):
        try:
            rgb = read_image(img_path)
        except ValueError as err:
            print(f"Warning: skipping {img_path}: {err}", file=sys.stderr)
            continue

        rel_path = os.path.relpath(img_path, src)
        sub_dir = os.path.dirname(rel_path)
        target_dir = os.path.join(dst, sub_dir)
        os.makedirs(target_dir, exist_ok=True)

        stem, ext = os.path.splitext(os.path.basename(img_path))
        mask = create_mask(rgb)

        for key in selected_keys:
            name, func = TRANSFORMATIONS[key]
            if key in ("roi", "analyze", "landmarks"):
                out_img = func(rgb, mask)
            elif key == "mask":
                out_img = mask
            else:
                out_img = func(rgb)

            out_name = f"{stem}_{name}{ext}"
            out_path = os.path.join(target_dir, out_name)
            write_image(out_path, out_img)

        if idx % 50 == 0 or idx == len(images):
            print(f"  [{idx}/{len(images)}] completed")

    print(f"Done. Transformations saved in '{dst}'.")
    return 0


def main():
    """Main CLI entrypoint."""
    args = parse_args()

    # Batch mode: -src provided
    if args.src:
        selected = []
        if args.mask or args.all:
            selected.append("mask")
        if args.blur or args.all:
            selected.append("blur")
        if args.roi or args.all:
            selected.append("roi")
        if args.analyze or args.all:
            selected.append("analyze")
        if args.landmarks or args.all:
            selected.append("landmarks")

        # If no flag specified, default to all transformations
        if not selected:
            selected = list(TRANSFORMATIONS.keys())

        return run_batch(args.src, args.dst, selected)

    # Single image mode: positional image argument provided
    if args.image:
        if not os.path.isfile(args.image):
            print(f"Error: '{args.image}' is not a file", file=sys.stderr)
            return 1
        if not is_image(os.path.basename(args.image)):
            print(f"Error: '{args.image}' is not a supported image file",
                  file=sys.stderr)
            return 1
        return display_transformations(args.image, args.no_display)

    # No arguments provided
    print("Error: specify an image path or use -src and -dst for batch mode.",
          file=sys.stderr)
    print("Use ./Transformation.py -h for help.", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
