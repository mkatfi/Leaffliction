# Leaffliction

**Plant leaf disease classification — Computer vision project (1337 / 42)**

Leaffliction is a computer vision pipeline that recognizes plant diseases from
leaf photographs. Before any machine learning, the project forces you to
**understand and fix your dataset**: analyze its distribution (Part 1), balance
it with data augmentation (Part 2), extract visual features (Part 3), and only
then train a classifier (Part 4).

```
Given a photo of a leaf  →  which disease is it? (or: healthy)
Dataset: Apple & Grape leaves, 8 classes (see table below)
```

---

## 1. The pipeline at a glance

```
 Part 1            Part 2                Part 3                Part 4
Distribution.py → Augmentation.py  →  Transformation.py  →  train.py / predict.py
"what do I have?" "fix the balance"   "extract features"    "learn & classify"
 pie + bar charts   6 aug types,        blur → mask →         CNN training,
                    balancing           ROI → analyze →       >90% validation
                                        landmarks → histogram accuracy required
```

**Why this order?** A neural network trained on unbalanced data learns the lazy
shortcut "always answer the majority class". Parts 1 and 2 make you *see* and
*fix* that imbalance before training — otherwise Part 4 cannot reach the
required 90% validation accuracy.

---

## 2. The dataset

Directory layout (folder name = class label, no annotation files):

```
images/
├── Apple_Black_rot/
├── Apple_healthy/
├── Apple_rust/
├── Apple_scab/
├── Grape_Black_rot/
├── Grape_Esca/
├── Grape_healthy/
└── Grape_spot/
```

| Plant | Class | ~Images (original) |
|---|---|---|
| Apple | Apple_healthy | 1640 |
| Apple | Apple_Black_rot | 620 |
| Apple | Apple_scab | 629 |
| Apple | Apple_rust | 275 |
| Grape | Grape_Esca | 1382 |
| Grape | Grape_Black_rot | 1178 |
| Grape | Grape_spot | 1075 |
| Grape | Grape_healthy | 422 |

> The dataset is **deliberately unbalanced** (healthy Apple = 4.3× Apple_rust).
> That imbalance is the motivation for Part 2. It is provided by the school and
> must NOT be committed to git (see `signature.txt`, section 8).

---

## 3. Project structure

```
Leaffliction/
├── Distribution.py        # Part 1 — dataset analysis (pie + bar charts)
├── Augmentation.py        # Part 2 — 6 augmentation types + balancing
├── Transformation.py      # Part 3 — feature extraction pipeline (PlantCV/OpenCV)
├── train.py               # Part 4 — model training
├── predict.py             # Part 4 — single-image prediction
├── tests/                 # unit tests (unittest)
├── Makefile               # single entry point for every command
├── requirements.txt       # python dependencies
├── .gitignore             # excludes images/, augmented_directory/, *.zip
└── README.md
```

---

## 4. Setup

Requires Python 3.10+.

```bash
make            # creates venv/ and installs requirements.txt  (first time only)
make test       # unit tests — expect "OK"
make lint       # flake8 style check — expect no output
```

Equivalent manual setup:

```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

---

## 5. Usage

### Part 1 — Distribution

```bash
make distribution                         # charts for ./images (interactive window)
make distribution SAVE=charts             # save PNGs under ./charts instead
make distribution DIR=./images/Grape      # any directory works (generic requirement)
```

Prints per-class counts, then shows/saves:
- a **pie chart** — proportions at a glance (reveals imbalance)
- a **bar chart** — absolute counts (tells you how many images Part 2 must add)

Error handling (never crashes, always a clean message + exit 1):
```bash
./venv/bin/python Distribution.py          # missing argument
./venv/bin/python Distribution.py /nope    # invalid directory
```

### Part 2 — Augmentation

Six techniques, implemented as geometric transforms on the pixel array
(Flip = mirror, Rotate = rotation matrix, Skew/Shear = affine matrices,
Crop = random region + resize, Distortion = perspective transform):

```bash
# single image → 6 new files next to the original:
#   image (1)_Flip.JPG, image (1)_Rotate.JPG, ... image (1)_Distortion.JPG
make augmentation IMG="/path/to/image (1).JPG"

# balance the whole dataset → creates ./augmented_directory
# (small classes are augmented with random originals × random types
#  until every class reaches the size of the biggest class)
make balance
```

Verify the balance worked:

```bash
make distribution DIR=./augmented_directory   # all bars ~equal
for d in augmented_directory/*/; do echo "$d $(ls "$d" | wc -l)"; done
```

Safety properties (all verified by tests):
- original images are **never modified** — outputs are new files only
- generated files are **never re-augmented** (no copies-of-copies drift)
- a second `make balance` **refuses to overwrite** an existing
  `augmented_directory`
- unreadable / non-image input prints an error and continues, no crash

### Part 3 — Transformation

Direct feature extraction per leaf image (OpenCV / PlantCV):

```
Original → Gaussian blur → Mask (background removal) → ROI objects
        → Analyze object (shape/outline) → Pseudolandmarks → Color histogram
```

```bash
./venv/bin/python Transformation.py <image>                # display the 7 outputs
./venv/bin/python Transformation.py -src <dir> -dst <dir>  # batch: save to dst
./venv/bin/python Transformation.py -h                     # full usage
```

### Part 4 — Classification

```bash
./venv/bin/python train.py <dataset_dir>     # trains, saves model + zip
./venv/bin/python predict.py <image.JPG>     # shows original + transformed image,
                                             # prints predicted disease class
```

Requirements from the subject:
- train/validation split (≥100 images in validation)
- **validation accuracy > 90%**, provable at evaluation time
- training outputs (model + augmented data) packaged in a `.zip`
- no knowledge of the validation set before assessment (no overfitting tricks)

---

## 6. Key concepts (the WHY)

| Concept | What it means here |
|---|---|
| **Class imbalance** | Classes have wildly different sizes → model learns "always guess the majority class" |
| **Data augmentation** | Realistic modified copies (flip/rotate/...) — multiplies small classes *and* teaches the model orientation/position invariance |
| **Affine vs perspective transform** | Skew/shear keep parallel lines parallel; distortion (perspective) does not |
| **Mask / ROI** | Remove the background so features describe the leaf, not the photo |
| **Pseudolandmarks** | Sampled contour points — a compact shape fingerprint of the leaf |
| **Color histogram** | Distribution of pixel intensities per channel — disease changes leaf color statistics |
| **Overfitting** | Memorizing the training set instead of learning disease patterns; fought with augmentation + validation |

## 7. Technologies

- **Python 3** (only standard library + the packages in `requirements.txt`)
- **OpenCV (`cv2`)** + **numpy** — all image operations (images are arrays)
- **matplotlib** — charts and histograms
- **unittest** + **flake8** — tests and style (42 norm-friendly toolchain)
- Part 4: **TensorFlow/Keras** or **PyTorch** (either is acceptable)

---

## 8. Turn-in notes (1337 requirements)

- Git repo contains **programs only** — the dataset is **forbidden** in git.
- `signature.txt` at the repo root = SHA-1 of your dataset zip:
  ```bash
  sha1sum directory.zip        # Linux
  shasum directory.zip         # macOS
  certUtil -hashfile directory.zip sha1   # Windows
  ```
  The defense compares this hash against your actual dataset.
- Everything must run from the repo root with `make`-level commands —
  the evaluator will not read your mind, only your README and Makefile.

---

## 9. Full command reference

| Command | Effect |
|---|---|
| `make` / `make setup` | create venv + install requirements |
| `make re` | full reset, then setup |
| `make test` | unit tests |
| `make lint` | flake8 |
| `make distribution [DIR=...] [SAVE=charts]` | Part 1 charts |
| `make augmentation IMG="..."` | Part 2, six augmentations of one image |
| `make balance` | Part 2, build balanced `augmented_directory/` |
| `make clean` | remove `__pycache__` |
| `make fclean` | also remove venv, `augmented_directory/`, `charts/` |

---

*Educational project — 1337 School (42 network). Dataset: PlantVillage-derived
leaf images. Do not use as a substitute for professional plant pathology diagnosis.*
