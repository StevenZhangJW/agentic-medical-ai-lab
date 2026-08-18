import numpy as np
import pytest

torch = pytest.importorskip("torch")

from segmentation_pipeline import Config, UNet3D, dice_score, fixed_split, load_normalized, loss_function, rle_encode


def test_normalization_preserves_background_and_standardizes_brain():
    image, _ = load_normalized(__import__("pathlib").Path("data/cases/images/case_001.nii.gz"))
    for channel in range(4):
        brain = image[..., channel] != 0
        assert np.all(image[..., channel][~brain] == 0)
        assert abs(float(image[..., channel][brain].mean())) < 1e-5
        assert abs(float(image[..., channel][brain].std()) - 1) < 1e-5


def test_fixed_split_is_reproducible_and_disjoint():
    cases = [f"case_{x:03d}" for x in range(1, 61)]
    train, validation = fixed_split(cases, Config().seed)
    assert len(train) == 48 and len(validation) == 12
    assert not set(train) & set(validation)
    assert (train, validation) == fixed_split(cases, Config().seed)


def test_unet_shape_loss_and_backward():
    model = UNet3D(base=2)
    inputs = torch.randn(1, 4, 16, 16, 16)
    target = torch.zeros(1, 1, 16, 16, 16); target[..., 4:9, 4:9, 4:9] = 1
    output = model(inputs); loss = loss_function(output, target); loss.backward()
    assert output.shape == target.shape and torch.isfinite(loss)


def test_dice_and_rle_known_example():
    mask = np.zeros((2, 3, 1), dtype=bool); mask[0, 0, 0] = True; mask[1, 1, 0] = True
    assert dice_score(mask, mask) == 1
    assert rle_encode(mask) == "1 1 4 1"
