#!/usr/bin/env python3
"""Part 1 - Analysis of the data set.

Usage: ./Distribution.py <directory> [--save OUTPUT_DIR]

Reads the images of the sub-directories of <directory>, then displays a pie
chart and a bar chart (one figure per plant type). The columns are named
after the sub-directories.
"""

import argparse
import os
import sys

from dataset import group_by_plant, scan_dataset
from viewer import can_display, open_file, plt, show

# Same colours as the example of the subject (plotly default palette)
COLORS = ["#636EFA", "#EF553B", "#00CC96", "#AB63FA", "#FFA15A",
          "#19D3F3", "#FF6692", "#B6E880", "#FF97FF", "#FECB52"]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Show pie and bar charts of the class distribution "
                    "of a leaf data set.")
    parser.add_argument("directory",
                        help="directory whose sub-directories are classes "
                             "(ex: ./Apple or ./images)")
    parser.add_argument("--save", metavar="OUTPUT_DIR",
                        help="save the charts as PNG files in OUTPUT_DIR "
                             "instead of opening windows")
    return parser.parse_args()


def make_figure(plant, classes):
    """Build one figure (pie + bar) for a plant type."""
    labels = [c["name"] for c in classes]
    counts = [len(c["images"]) for c in classes]
    colors = [COLORS[i % len(COLORS)] for i in range(len(classes))]

    fig, (ax_pie, ax_bar) = plt.subplots(1, 2, figsize=(13, 6))
    fig.suptitle(f"{plant.lower()} class distribution")
    ax_pie.pie(counts, colors=colors, autopct="%1.1f%%", startangle=90,
               counterclock=False, textprops={"color": "white"})
    ax_pie.legend(labels, loc="lower center", bbox_to_anchor=(0.5, -0.15),
                  fontsize=8)
    bars = ax_bar.bar(labels, counts, color=colors)
    ax_bar.bar_label(bars)
    ax_bar.set_ylabel("number of images")
    ax_bar.tick_params(axis="x", labelrotation=20, labelsize=8)
    fig.tight_layout()
    return fig


def main():
    args = parse_args()
    try:
        classes = scan_dataset(args.directory)
    except ValueError as err:
        print(f"Error: {err}", file=sys.stderr)
        return 1

    # no window available -> save the charts instead of losing them
    save_dir = args.save
    auto_open = False
    if not save_dir and not can_display():
        save_dir = "distribution_charts"
        auto_open = True
        print("No window backend available: charts are saved in "
              f"'{save_dir}/' and opened with the system viewer")

    groups = group_by_plant(classes)
    for plant, plant_classes in groups.items():
        print(f"[{plant}]")
        for cls in plant_classes:
            print(f"  {cls['name']:<25} {len(cls['images'])}")
        make_figure(plant, plant_classes)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            out = os.path.join(save_dir, f"{plant.lower()}_distribution.png")
            plt.gcf().savefig(out)
            print(f"  chart saved: {out}")
            plt.close("all")
            if auto_open:
                open_file(out)
    if not save_dir:
        show()
    return 0


if __name__ == "__main__":
    sys.exit(main())
