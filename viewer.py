"""Show figures safely: a window if possible, otherwise a saved PNG.

Import this module BEFORE matplotlib.pyplot: it picks the matplotlib
backend. A window backend (Tk) is used only if it really works here;
otherwise the figures are saved as PNG files and opened with the image
viewer of the system. Nothing here can crash the program.
Set the MPLBACKEND environment variable to force another backend.
"""

import os
import shutil
import subprocess
import sys

import matplotlib

# backends that only draw into memory / files (they can not open a window)
NON_INTERACTIVE = ("agg", "cairo", "pdf", "pgf", "ps", "svg", "template")


def has_display():
    """True if a graphical session seems available."""
    if sys.platform in ("darwin", "win32"):
        return True
    return bool(os.environ.get("DISPLAY")
                or os.environ.get("WAYLAND_DISPLAY"))


def tk_works():
    """True if tkinter is installed and can really open a window."""
    if not has_display():
        return False
    try:
        import tkinter
        tkinter.Tk().destroy()
        return True
    except Exception:
        return False


def choose_backend():
    """Select a safe matplotlib backend (unless MPLBACKEND is set)."""
    if os.environ.get("MPLBACKEND"):
        return
    matplotlib.use("TkAgg" if tk_works() else "Agg")


choose_backend()
import matplotlib.pyplot as plt  # noqa: E402


def can_display():
    """True if matplotlib has an interactive (window) backend."""
    return matplotlib.get_backend().lower() not in NON_INTERACTIVE


def open_file(path):
    """Open an image with the system viewer (silently ignore failures)."""
    if not has_display():
        return False
    if sys.platform == "darwin":
        tool = "open"
    elif sys.platform == "win32":
        os.startfile(path)
        return True
    else:
        tool = "xdg-open"
    if shutil.which(tool) is None:
        return False
    try:
        subprocess.Popen([tool, path], stdout=subprocess.DEVNULL,
                         stderr=subprocess.DEVNULL)
        return True
    except OSError:
        return False


def show(fallback_path=None):
    """Open the figure windows, or save the current figure and open it."""
    if can_display():
        plt.show()
        return
    if fallback_path:
        os.makedirs(os.path.dirname(os.path.abspath(fallback_path)),
                    exist_ok=True)
        plt.gcf().savefig(fallback_path)
        opened = open_file(fallback_path)
        print(f"No window backend available: figure saved in "
              f"'{fallback_path}'" + (" and opened" if opened else ""))
