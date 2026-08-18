import csv

import nibabel as nib
import numpy as np
import pytest

import make_submission


def test_build_submission_validates_and_writes_rows(tmp_path, monkeypatch):
    monkeypatch.setattr(make_submission, "EXPECTED_IDS", ("eval_001", "eval_002"))
    monkeypatch.setattr(make_submission, "EXPECTED_SHAPE", (2, 3, 1))
    for number in (1, 2):
        mask = np.zeros((2, 3, 1), dtype=np.uint8)
        if number == 1:
            mask[0, 0, 0] = 1
        nib.save(nib.Nifti1Image(mask, np.eye(4)), tmp_path / f"eval_{number:03d}.nii.gz")

    output = tmp_path / "submission.csv"
    empty = make_submission.build_submission(tmp_path, output)
    with output.open(newline="") as handle:
        rows = list(csv.DictReader(handle))

    assert empty == ["eval_002"]
    assert rows == [
        {"case_id": "eval_001", "mask_rle": "1 1"},
        {"case_id": "eval_002", "mask_rle": "EMPTY"},
    ]


def test_build_submission_rejects_nonbinary_mask(tmp_path, monkeypatch):
    monkeypatch.setattr(make_submission, "EXPECTED_IDS", ("eval_001",))
    monkeypatch.setattr(make_submission, "EXPECTED_SHAPE", (2, 3, 1))
    mask = np.zeros((2, 3, 1), dtype=np.uint8)
    mask[0, 0, 0] = 2
    nib.save(nib.Nifti1Image(mask, np.eye(4)), tmp_path / "eval_001.nii.gz")

    with pytest.raises(ValueError, match="binary"):
        make_submission.build_submission(tmp_path, tmp_path / "submission.csv")
