"""Interactive four-channel brain MRI case viewer."""

from pathlib import Path

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import streamlit as st


DATA_DIR = Path(__file__).resolve().parent / "data" / "cases"
IMAGE_DIR = DATA_DIR / "images"
LABEL_DIR = DATA_DIR / "labels"
CHANNELS = ("FLAIR", "T1w", "T1-Gd", "T2w")


def available_cases(image_dir: Path = IMAGE_DIR) -> list[str]:
    """Return case identifiers that have both an image and a label volume."""
    return [
        path.stem.removesuffix(".nii")
        for path in sorted(image_dir.glob("case_*.nii.gz"))
        if (LABEL_DIR / path.name).is_file()
    ]


@st.cache_data(show_spinner="Loading MRI volume…", max_entries=3)
def load_case(case_id: str) -> tuple[np.ndarray, np.ndarray]:
    """Load and validate one channel-last image and its registered label map."""
    image_nii = nib.load(IMAGE_DIR / f"{case_id}.nii.gz")
    mask_nii = nib.load(LABEL_DIR / f"{case_id}.nii.gz")
    image = np.asarray(image_nii.dataobj, dtype=np.float32)
    mask = np.asarray(mask_nii.dataobj, dtype=np.uint8)

    if image.ndim != 4 or image.shape[-1] != len(CHANNELS):
        raise ValueError(f"Expected an (X, Y, Z, 4) image, found {image.shape}")
    if mask.shape != image.shape[:3]:
        raise ValueError(f"Image {image.shape[:3]} and mask {mask.shape} do not match")
    if not np.allclose(image_nii.affine, mask_nii.affine):
        raise ValueError("Image and mask affines do not match")
    return image, mask


def display_slice(volume: np.ndarray, slice_index: int, channel: int) -> np.ndarray:
    """Extract an axial slice and orient it consistently for display."""
    return np.rot90(volume[:, :, slice_index, channel])


def display_mask(mask: np.ndarray, slice_index: int) -> np.ndarray:
    """Extract the binary whole-lesion mask in the same display orientation."""
    return np.rot90(mask[:, :, slice_index] > 0)


def intensity_window(channel_slice: np.ndarray) -> tuple[float, float]:
    """Use robust per-panel contrast while ignoring skull-stripped background."""
    foreground = channel_slice[channel_slice > 0]
    if foreground.size == 0:
        return 0.0, 1.0
    low, high = np.percentile(foreground, (1, 99))
    return float(low), float(high if high > low else low + 1)


def make_figure(
    image: np.ndarray, mask: np.ndarray, slice_index: int, show_mask: bool
) -> plt.Figure:
    """Render all co-registered channels using one shared axial index."""
    figure, axes = plt.subplots(2, 2, figsize=(9, 9), facecolor="#0e1117")
    lesion = display_mask(mask, slice_index)
    for channel, (axis, name) in enumerate(zip(axes.flat, CHANNELS)):
        plane = display_slice(image, slice_index, channel)
        low, high = intensity_window(plane)
        axis.imshow(plane, cmap="gray", vmin=low, vmax=high, interpolation="nearest")
        if show_mask and lesion.any():
            axis.contour(lesion, levels=[0.5], colors=["#ff3b3b"], linewidths=1.4)
        axis.set_title(name, color="white", fontsize=15, fontweight="bold")
        axis.axis("off")
    figure.suptitle(
        f"Axial slice {slice_index + 1} / {image.shape[2]}",
        color="white",
        fontsize=13,
    )
    figure.tight_layout()
    return figure


def main() -> None:
    st.set_page_config(page_title="Four-channel MRI viewer", page_icon="🧠", layout="wide")
    st.title("🧠 Four-channel MRI viewer")
    st.caption("Co-registered axial FLAIR, T1w, T1-Gd, and T2w with expert lesion outline")

    cases = available_cases()
    if not cases:
        st.error("No complete image/label case pairs were found.")
        st.stop()

    with st.sidebar:
        st.header("Viewer controls")
        case_id = st.selectbox("Case", cases)
        show_mask = st.toggle("Show segmentation", value=True)
        st.markdown("**Overlay:** red outline = any expert label > 0")

    image, mask = load_case(case_id)
    state_key = f"slice_{case_id}"
    if state_key not in st.session_state:
        tumour_by_slice = np.count_nonzero(mask > 0, axis=(0, 1))
        st.session_state[state_key] = int(np.argmax(tumour_by_slice))

    previous, slider_col, following = st.columns([1, 8, 1], vertical_alignment="bottom")
    with previous:
        if st.button("◀ Previous", width="stretch"):
            st.session_state[state_key] = max(0, st.session_state[state_key] - 1)
    with following:
        if st.button("Next ▶", width="stretch"):
            st.session_state[state_key] = min(image.shape[2] - 1, st.session_state[state_key] + 1)
    with slider_col:
        slice_index = st.slider(
            "Axial slice",
            min_value=0,
            max_value=image.shape[2] - 1,
            key=state_key,
            format="Slice %d",
        )

    lesion_pixels = int(np.count_nonzero(mask[:, :, slice_index]))
    st.subheader(f"{case_id} · axial slice {slice_index + 1} of {image.shape[2]}")
    status = "lesion present" if lesion_pixels else "no lesion on this slice"
    st.caption(f"Shared source index z={slice_index} across all four panels · {status}")

    figure = make_figure(image, mask, slice_index, show_mask)
    st.pyplot(figure, width="stretch")
    plt.close(figure)


if __name__ == "__main__":
    main()
