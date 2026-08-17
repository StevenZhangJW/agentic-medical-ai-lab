import matplotlib.pyplot as plt
import numpy as np

from app import CHANNELS, available_cases, display_mask, display_slice, load_case, make_figure


def test_dataset_and_case_structure():
    cases = available_cases()
    assert len(cases) == 60
    assert cases[0] == "case_001"

    image, mask = load_case("case_001")
    assert image.shape == (240, 240, 155, 4)
    assert mask.shape == image.shape[:3]
    assert set(np.unique(mask)) == {0, 1, 2, 3}


def test_one_index_extracts_same_plane_for_every_channel():
    volume = np.zeros((3, 4, 5, 4), dtype=np.float32)
    for channel in range(4):
        for z in range(5):
            volume[:, :, z, channel] = channel * 10 + z

    for channel in range(4):
        plane = display_slice(volume, slice_index=3, channel=channel)
        assert np.all(plane == channel * 10 + 3)
        assert plane.shape == (4, 3)


def test_mask_and_images_have_identical_display_transform():
    image = np.zeros((3, 4, 2, 4), dtype=np.float32)
    mask = np.zeros((3, 4, 2), dtype=np.uint8)
    image[1, 2, 1, :] = 7
    mask[1, 2, 1] = 3

    lesion_location = np.argwhere(display_mask(mask, 1))[0]
    for channel in range(4):
        image_location = np.argwhere(display_slice(image, 1, channel) == 7)[0]
        np.testing.assert_array_equal(image_location, lesion_location)


def test_figure_has_all_named_channels_and_optional_overlay():
    image, mask = load_case("case_001")
    figure = make_figure(image, mask, slice_index=80, show_mask=True)
    assert [axis.get_title() for axis in figure.axes] == list(CHANNELS)
    assert all(axis.collections for axis in figure.axes)
    plt.close(figure)

    figure = make_figure(image, mask, slice_index=80, show_mask=False)
    assert all(not axis.collections for axis in figure.axes)
    plt.close(figure)
