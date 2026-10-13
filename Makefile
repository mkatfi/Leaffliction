# ============================================================
#  Leaffliction - Part 1 & 2
#  Just type `make` to install everything, `make help` for more.
# ============================================================

VENV    := venv
PY      := $(VENV)/bin/python
STAMP   := $(VENV)/.installed

# Arguments you can change on the command line, ex:
#   make distribution DIR=./Apple
DIR     ?= ./images
DST     ?= augmented_directory
IMG     ?= ./images/Apple_healthy/image (1).JPG
SAVE    ?=

.DEFAULT_GOAL := all

# ---------- install ----------
all: $(STAMP)

$(STAMP): requirements.txt
	python3 -m venv $(VENV)
	$(PY) -m pip install --upgrade pip
	$(PY) -m pip install -r requirements.txt
	$(PY) -m pip install flake8
	@touch $(STAMP)
	@echo "Installation done. Type 'make help' to see the commands."

# source /home/mar-one/Desktop/Leaffliction/venv/bin/activate
# ---------- Part 1 ----------
distribution: $(STAMP)
	$(PY) Distribution.py "$(DIR)" $(if $(SAVE),--save "$(SAVE)")

# ---------- Part 2 ----------
augmentation: $(STAMP)
	$(PY) Augmentation.py "$(IMG)"

balance: $(STAMP)
	$(PY) Balance.py "$(DIR)" -dst "$(DST)"

# ---------- Part 3 ----------
transformation: $(STAMP)
	$(PY) Transformation.py "$(IMG)"

# ---------- quality ----------
lint: $(STAMP)
	$(PY) -m flake8 *.py

# ---------- cleaning ----------
clean:
	find . -name "__pycache__" -type d -prune -exec rm -rf {} +

fclean: clean
	rm -rf $(VENV) $(DST) distribution_charts augmentation_preview.png \
	       transformation_preview.png histogram_preview.png

re: fclean all

help:
	@echo "make                  create venv + install requirements"
	@echo "make distribution     Part 1: charts          (DIR=./images [SAVE=charts])"
	@echo "make augmentation     Part 2: 1 image, 6 augs (IMG=\"path/to/image (1).JPG\")"
	@echo "make balance          Part 2: balance dataset (DIR=./images DST=augmented_directory)"
	@echo "make transformation   Part 3: feature extract (IMG=\"path/to/image (1).JPG\")"
	@echo "make lint             style check with flake8"
	@echo "make clean            remove __pycache__"
	@echo "make fclean           clean + remove venv and augmented_directory"
	@echo "make re               fclean + install again"

.PHONY: all distribution augmentation balance transformation lint clean fclean re help
