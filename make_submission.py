"""Validate evaluation masks and encode them as a Kaggle submission."""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import nibabel as nib
import numpy as np

EXPECTED_IDS = tuple(f"eval_{number:03d}" for number in range(1, 61))
EXPECTED_SHAPE = (240, 240, 155)


def rle_encode(mask: np.ndarray) -> str:
    """Encode a mask in one-indexed Fortran voxel order."""
    pixels = mask.astype(np.uint8).ravel(order="F")
    padded = np.pad(pixels, (1, 1))
    changes = np.flatnonzero(padded[1:] != padded[:-1])
    changes[1::2] -= changes[::2]
    changes[::2] += 1
    return " ".join(map(str, changes))


def build_submission(prediction_dir: Path, output: Path) -> list[str]:
    """Strictly validate one mask per expected case and write its RLE."""
    files = sorted(prediction_dir.glob("*.nii.gz"))
    ids = [path.name.removesuffix(".nii.gz") for path in files]
    malformed = [case_id for case_id in ids if not re.fullmatch(r"eval_\d{3}", case_id)]
    duplicates = sorted({case_id for case_id in ids if ids.count(case_id) > 1})
    missing = sorted(set(EXPECTED_IDS) - set(ids))
    unexpected = sorted(set(ids) - set(EXPECTED_IDS))
    if len(files) != len(EXPECTED_IDS) or malformed or duplicates or missing or unexpected:
        raise ValueError(
            f"Expected exactly eval_001..eval_060 once: count={len(files)}, "
            f"malformed={malformed}, duplicates={duplicates}, missing={missing}, "
            f"unexpected={unexpected}"
        )

    rows: list[tuple[str, str]] = []
    empty: list[str] = []
    volumes: list[int] = []
    for case_id, path in zip(ids, files):
        mask = np.asarray(nib.load(path).dataobj)
        if mask.shape != EXPECTED_SHAPE:
            raise ValueError(f"{path}: expected shape {EXPECTED_SHAPE}, found {mask.shape}")
        if not np.isfinite(mask).all():
            raise ValueError(f"{path}: mask contains NaN or infinite values")
        values = np.unique(mask)
        if np.any(values < 0) or not np.all(np.isin(values, (0, 1))):
            raise ValueError(f"{path}: mask must be binary 0/1, found {values.tolist()}")
        binary = mask.astype(bool)
        volume = int(binary.sum())
        volumes.append(volume)
        encoded = rle_encode(binary)
        if not encoded:
            empty.append(case_id)
            encoded = "EMPTY"
        rows.append((case_id, encoded))

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("case_id", "mask_rle"))
        writer.writerows(rows)

    print(f"Validated masks: {len(rows)}")
    print(f"Empty masks: {len(empty)} ({', '.join(empty) if empty else 'none'})")
    print(
        "Predicted lesion volume (voxels): "
        f"min={min(volumes)}, median={float(np.median(volumes)):.1f}, max={max(volumes)}"
    )
    for case_id in empty:
        print(f"WARNING: {case_id} is a valid empty mask and was written as EMPTY")
    print(f"Wrote {output} with {len(rows)} rows")
    return empty


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prediction_dir", type=Path)
    parser.add_argument("--output", type=Path, default=Path("submission.csv"))
    args = parser.parse_args()
    build_submission(args.prediction_dir, args.output)


if __name__ == "__main__":
    main()
