"""
Environment check — run this first, and any time something feels broken.

Works identically on macOS, Windows and Linux. No 'make' required.

    python verify.py
"""

import importlib
import platform
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent
DATA = BASE / "data" / "cases"
OK, BAD, WARN = "  [OK]  ", "  [FAIL]", "  [warn]"

PACKAGES = [
    ("numpy", "1.26"), ("nibabel", "5.0"), ("matplotlib", "3.7"),
    ("scipy", "1.10"), ("sklearn", "1.3"), ("skimage", "0.21"),
    ("streamlit", "1.35"),
]

problems = []


def check_python():
    v = sys.version_info
    print(f"\nPython {v.major}.{v.minor}.{v.micro}  ({platform.system()} {platform.machine()})")
    if v < (3, 10):
        print(BAD + f"Python 3.10+ required, found {v.major}.{v.minor}")
        problems.append("Install Python 3.11 or 3.12 from python.org (tick 'Add to PATH').")
    else:
        print(OK + "Python version")
    if sys.prefix == sys.base_prefix:
        print(WARN + "not running inside a virtual environment")


def check_packages():
    print("\nPackages")
    for mod, minv in PACKAGES:
        try:
            m = importlib.import_module(mod)
            print(OK + f"{mod:<12} {getattr(m, '__version__', '?')}")
        except ImportError:
            print(BAD + f"{mod} not installed")
            problems.append("Run:  python -m pip install -r requirements.txt")


def check_data():
    print("\nData")
    img, lab = DATA / "images", DATA / "labels"
    if not img.is_dir():
        print(BAD + f"missing {img}")
        problems.append("Data folder not found — did the clone finish? Try: git status")
        return
    n_img = len(list(img.glob("case_*.nii.gz")))
    n_lab = len(list(lab.glob("case_*.nii.gz")))
    print(OK + f"{n_img} images, {n_lab} label maps")
    if n_img == 0 or n_img != n_lab:
        print(BAD + "image/label count mismatch")
        problems.append("Re-clone the repository — the data appears incomplete.")
        return
    try:
        import nibabel as nib
        import numpy as np
        pid = sorted(p.name for p in img.glob("case_*.nii.gz"))[0]
        a = np.asanyarray(nib.load(str(img / pid)).dataobj)
        m = np.asanyarray(nib.load(str(lab / pid)).dataobj)
        print(OK + f"loaded {pid}: image {a.shape}, mask {m.shape}, "
                   f"tumour {100 * (m > 0).mean():.1f}% of volume")
    except Exception as exc:
        print(BAD + f"could not read a volume: {exc}")
        problems.append("nibabel could not open the data — check the clone completed.")


def check_torch():
    print("\nPyTorch (optional — only needed for the CNN demo)")
    try:
        import torch
        dev = ("cuda" if torch.cuda.is_available() else
               "mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()
               else "cpu")
        print(OK + f"torch {torch.__version__}, device: {dev}")
        if dev == "cpu":
            print(WARN + "no GPU detected — CNN training will be slow (that is fine)")
    except ImportError:
        print(WARN + "torch not installed — skip unless you reach the CNN exercise")


if __name__ == "__main__":
    print("=" * 60)
    print("  Agentic Medical AI Lab — environment check")
    print("=" * 60)
    check_python()
    check_packages()
    check_data()
    check_torch()
    print("\n" + "=" * 60)
    if problems:
        print("  NOT READY — do this:\n")
        for p in dict.fromkeys(problems):
            print("   * " + p)
        print("\n  Still stuck? Open the Codespace (see README) and carry on there.")
        sys.exit(1)
    print("  ALL GOOD — you are ready to start.")
    print("=" * 60)
