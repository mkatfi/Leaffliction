# Leaffliction - Part 1 & Part 2

Computer vision project (1337 / 42). This repository contains **only Part 1
(analysis of the data set) and Part 2 (data augmentation)**.

## Quick start (Makefile)

```bash
make                    # creates ./venv and installs the requirements
make help               # list all commands
```

| Command | What it does |
|---------|--------------|
| `make` | create `venv` + install requirements (only once) |
| `make distribution [DIR=./images] [SAVE=charts]` | Part 1: pie + bar charts |
| `make augmentation IMG="./images/Apple_healthy/image (1).JPG"` | Part 2: 6 augmentations of one image |
| `make balance [DIR=./images] [DST=augmented_directory]` | Part 2: balance the whole data set |
| `make test` | unit tests |
| `make lint` | flake8 (the Python norm) |
| `make clean` / `make fclean` / `make re` | remove cache / cache + venv + augmented_directory / reinstall |

Requires Python 3.9+ and `python3-venv`. Put (or symlink) the data set as `./images`.

## Files

| File | Role |
|------|------|
| `Distribution.py`   | Part 1 - pie + bar charts of each plant type |
| `Augmentation.py`   | Part 2 - 6 augmentations of one image (saved + displayed) |
| `Balance.py`        | Part 2 - balances the whole data set into `augmented_directory` |
| `augmentation_lib.py` | the 6 augmentation functions + image read/write |
| `dataset.py`        | scans a data set directory (classes, plant types) |
| `tests/`            | unit tests (`make test`) |

## Usage

The data set is expected as a directory of class sub-directories, e.g.
`images/Apple_healthy`, `images/Grape_Esca`... or `Apple/apple_healthy`...

### Part 1 - Distribution

```bash
./Distribution.py ./images            # one figure (pie + bar) per plant type
./Distribution.py ./Apple             # a single plant type
./Distribution.py ./images --save charts/   # save PNGs instead of opening windows
```
It also prints the number of images of each class. Column names are the
sub-directory names.

### Part 2 - Augmentation

```bash
./Augmentation.py "./images/Apple_healthy/image (1).JPG"
```
Creates next to the original image:
`image (1)_Flip.JPG`, `_Rotate`, `_Skew`, `_Shear`, `_Crop`, `_Distortion`
and displays them with the original (`--no-display` to only save).

Balance the whole data set (the original is **not modified**, the result is
written into `augmented_directory`):
```bash
./Balance.py ./images                  # -> ./augmented_directory
./Balance.py ./images -dst my_dir --seed 1
./Distribution.py ./augmented_directory   # check: classes are now equal
```
Within each plant type (Apple, Grape) every class is completed with augmented
images until it has as many images as the biggest class of that plant.
Apple -> 1640 per class, Grape -> 1382 per class on the given data set.

## Without make

```bash
python3 -m venv venv && . venv/bin/activate && pip install -r requirements.txt
./Distribution.py ./images
```

## Charts and windows

`viewer.py` chooses the matplotlib backend safely (no Qt, so no crash):

1. If **Tk** works on your machine (`sudo apt install python3-tk` on Linux),
   `Distribution.py` and `Augmentation.py` open a normal window.
2. Otherwise the figures are **saved as PNG and opened with the system image
   viewer** (`xdg-open`): `distribution_charts/*.png` and
   `augmentation_preview.png`.
3. With no display at all (SSH...) they are only saved and the path is printed.

`--save DIR` (Distribution) and `--no-display` (Augmentation) never open anything.

## Good to know

* `Augmentation.py` writes the 6 new images **next to the original** (as the
  subject asks). If you run it on `./images/...`, those 6 files stay in your
  data set and are counted by `Distribution.py` / `Balance.py`. Remove them
  with: `rm "images/Apple_healthy/image (1)_"{Flip,Rotate,Skew,Shear,Crop,Distortion}.JPG`
  or test on a copy of the image.
* `Balance.py` never modifies the source data set; delete the output folder
  (`make fclean`) before running it again.
