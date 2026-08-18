"""Reproducible four-channel 3-D U-Net whole-lesion pipeline.

Run ``python segmentation_pipeline.py --help`` for the staged workflow.  The
script deliberately uses one fixed model and one fixed 48/12 development split;
it is intended as a defensible baseline, not a hyperparameter-search framework.
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import torch
from scipy.ndimage import gaussian_filter
from torch import nn
from torch.utils.data import DataLoader, Dataset

CHANNELS = ("FLAIR", "T1w", "T1-Gd", "T2w")
THRESHOLDS = (0.30, 0.40, 0.50, 0.60, 0.70)


@dataclass(frozen=True)
class Config:
    seed: int = 20260818
    patch_size: tuple[int, int, int] = (96, 96, 96)
    base_channels: int = 16
    batch_size: int = 1
    train_patches_per_case: int = 4
    val_patches_per_case: int = 2
    max_epochs: int = 100
    patience: int = 12
    learning_rate: float = 2e-4
    foreground_probability: float = 0.7


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def case_paths(image_dir: Path, label_dir: Path | None = None) -> list[tuple[str, Path, Path | None]]:
    """Discover cases and, when requested, require one matching label each."""
    found = []
    for image in sorted(image_dir.glob("*.nii.gz")):
        case_id = image.name.removesuffix(".nii.gz")
        label = label_dir / image.name if label_dir else None
        if label is not None and not label.is_file():
            raise FileNotFoundError(f"Missing label for {case_id}: {label}")
        found.append((case_id, image, label))
    if not found:
        raise FileNotFoundError(f"No .nii.gz cases found in {image_dir}")
    return found


def load_normalized(path: Path) -> tuple[np.ndarray, nib.Nifti1Image]:
    """Load X,Y,Z,4 data and z-score each channel inside its non-zero brain."""
    nii = nib.load(path)
    image = np.asarray(nii.dataobj, dtype=np.float32)
    if image.ndim != 4 or image.shape[-1] != 4:
        raise ValueError(f"{path}: expected (X,Y,Z,4), found {image.shape}")
    output = np.zeros_like(image, dtype=np.float32)
    for channel in range(4):
        source = image[..., channel]
        brain = source != 0
        if not brain.any():
            raise ValueError(f"{path}: channel {CHANNELS[channel]} has no brain voxels")
        values = source[brain]
        sd = float(values.std())
        if not np.isfinite(sd) or sd <= 0:
            raise ValueError(f"{path}: channel {CHANNELS[channel]} has invalid SD {sd}")
        output[..., channel][brain] = (values - float(values.mean())) / sd
    return output, nii


def load_label(path: Path, shape: tuple[int, int, int], affine: np.ndarray) -> np.ndarray:
    nii = nib.load(path)
    label = np.asarray(nii.dataobj)
    if label.shape != shape or not np.allclose(nii.affine, affine):
        raise ValueError(f"{path}: label geometry does not match its image")
    return (label > 0).astype(np.float32)


def fixed_split(case_ids: list[str], seed: int, val_count: int = 12) -> tuple[list[str], list[str]]:
    shuffled = sorted(case_ids)
    random.Random(seed).shuffle(shuffled)
    return sorted(shuffled[val_count:]), sorted(shuffled[:val_count])


def _patch(array: np.ndarray, center: np.ndarray, size: tuple[int, int, int]) -> np.ndarray:
    spatial = np.asarray(array.shape[:3])
    size_a = np.asarray(size)
    start = np.minimum(np.maximum(center - size_a // 2, 0), np.maximum(spatial - size_a, 0))
    end = np.minimum(start + size_a, spatial)
    slices = tuple(slice(int(a), int(b)) for a, b in zip(start, end))
    result = array[slices]
    pads = [(0, int(n - result.shape[i])) for i, n in enumerate(size)]
    if array.ndim == 4:
        pads.append((0, 0))
    return np.pad(result, pads)


class PatchDataset(Dataset):
    """Lazy, tumour-aware 3-D patch sampler (no modified data is written)."""

    def __init__(self, cases, config: Config, patches_per_case: int, augment: bool):
        self.cases = cases
        self.config = config
        self.patches_per_case = patches_per_case
        self.augment = augment
        self.cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    def __len__(self):
        return len(self.cases) * self.patches_per_case

    def _load(self, item):
        case_id, image_path, label_path = item
        if case_id not in self.cache:
            image, nii = load_normalized(image_path)
            self.cache[case_id] = (image, load_label(label_path, image.shape[:3], nii.affine))
        return self.cache[case_id]

    def __getitem__(self, index):
        image, mask = self._load(self.cases[index % len(self.cases)])
        tumour = np.argwhere(mask > 0)
        if tumour.size and random.random() < self.config.foreground_probability:
            center = tumour[random.randrange(len(tumour))]
        else:
            brain = np.argwhere(np.any(image != 0, axis=-1))
            center = brain[random.randrange(len(brain))]
        image = _patch(image, center, self.config.patch_size)
        mask = _patch(mask, center, self.config.patch_size)
        if self.augment:
            for axis in range(3):
                if random.random() < 0.5:
                    image, mask = np.flip(image, axis).copy(), np.flip(mask, axis).copy()
            image = image * random.uniform(0.9, 1.1) + random.uniform(-0.1, 0.1) * (image != 0)
            if random.random() < 0.3:
                image = image + np.random.normal(0, 0.03, image.shape).astype(np.float32) * (image != 0)
        return torch.from_numpy(np.moveaxis(image.astype(np.float32), -1, 0)), torch.from_numpy(mask[None].astype(np.float32))


def _block(inputs: int, outputs: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv3d(inputs, outputs, 3, padding=1, bias=False), nn.InstanceNorm3d(outputs), nn.LeakyReLU(0.01),
        nn.Conv3d(outputs, outputs, 3, padding=1, bias=False), nn.InstanceNorm3d(outputs), nn.LeakyReLU(0.01),
    )


class UNet3D(nn.Module):
    def __init__(self, base: int = 16):
        super().__init__()
        self.enc1, self.enc2, self.enc3 = _block(4, base), _block(base, base * 2), _block(base * 2, base * 4)
        self.pool = nn.MaxPool3d(2)
        self.bottleneck = _block(base * 4, base * 8)
        self.up3, self.dec3 = nn.ConvTranspose3d(base * 8, base * 4, 2, 2), _block(base * 8, base * 4)
        self.up2, self.dec2 = nn.ConvTranspose3d(base * 4, base * 2, 2, 2), _block(base * 4, base * 2)
        self.up1, self.dec1 = nn.ConvTranspose3d(base * 2, base, 2, 2), _block(base * 2, base)
        self.output = nn.Conv3d(base, 1, 1)

    def forward(self, x):
        e1 = self.enc1(x); e2 = self.enc2(self.pool(e1)); e3 = self.enc3(self.pool(e2))
        x = self.dec3(torch.cat((self.up3(self.bottleneck(self.pool(e3))), e3), 1))
        x = self.dec2(torch.cat((self.up2(x), e2), 1))
        return self.output(self.dec1(torch.cat((self.up1(x), e1), 1)))


def soft_dice(logits, target, epsilon=1e-5):
    probability = torch.sigmoid(logits)
    axes = tuple(range(1, probability.ndim))
    return ((2 * (probability * target).sum(axes) + epsilon) / (probability.sum(axes) + target.sum(axes) + epsilon)).mean()


def loss_function(logits, target):
    return nn.functional.binary_cross_entropy_with_logits(logits, target) + (1 - soft_dice(logits, target))


def sliding_probability(model, image: np.ndarray, patch_size, device) -> np.ndarray:
    """Overlap-tile inference with Gaussian blending and complete edge coverage."""
    spatial, patch = np.asarray(image.shape[:3]), np.asarray(patch_size)
    padded_shape = np.maximum(spatial, patch)
    padded = np.pad(image, [(0, int(padded_shape[i] - spatial[i])) for i in range(3)] + [(0, 0)])
    stride = np.maximum(patch // 2, 1)
    starts = [sorted(set(list(range(0, int(padded_shape[i] - patch[i] + 1), int(stride[i]))) + [int(padded_shape[i] - patch[i])])) for i in range(3)]
    weight = gaussian_filter(np.pad(np.ones(tuple(np.maximum(patch // 2, 1))), [(int(p // 4), int(p - max(p // 2, 1) - p // 4)) for p in patch]), sigma=tuple(patch / 8))
    weight = np.maximum(weight, 1e-3).astype(np.float32)
    total, weights = np.zeros(tuple(padded_shape), np.float32), np.zeros(tuple(padded_shape), np.float32)
    model.eval()
    with torch.inference_mode():
        for x in starts[0]:
            for y in starts[1]:
                for z in starts[2]:
                    crop = padded[x:x+patch[0], y:y+patch[1], z:z+patch[2]]
                    tensor = torch.from_numpy(np.moveaxis(crop, -1, 0)[None]).to(device)
                    pred = torch.sigmoid(model(tensor))[0, 0].cpu().numpy()
                    total[x:x+patch[0], y:y+patch[1], z:z+patch[2]] += pred * weight
                    weights[x:x+patch[0], y:y+patch[1], z:z+patch[2]] += weight
    return (total / np.maximum(weights, 1e-6))[tuple(slice(0, int(n)) for n in spatial)]


def dice_score(prediction: np.ndarray, target: np.ndarray) -> float:
    denominator = prediction.sum() + target.sum()
    return float((2 * np.logical_and(prediction, target).sum()) / denominator) if denominator else 1.0


def train_model(train_cases, val_cases, config: Config, output: Path, *, final: bool = False) -> Path:
    seed_everything(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet3D(config.base_channels).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    loader = DataLoader(PatchDataset(train_cases, config, config.train_patches_per_case, True), batch_size=config.batch_size, shuffle=True, num_workers=0)
    validation = DataLoader(PatchDataset(val_cases or train_cases, config, config.val_patches_per_case, False), batch_size=1, num_workers=0)
    output.mkdir(parents=True, exist_ok=True)
    checkpoint = output / ("final_model.pt" if final else "best_validation_model.pt")
    best, stale = -1.0, 0
    history = []
    for epoch in range(1, config.max_epochs + 1):
        model.train(); losses = []
        for images, masks in loader:
            images, masks = images.to(device), masks.to(device); optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                loss = loss_function(model(images), masks)
            scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update(); losses.append(float(loss))
        model.eval(); dices = []
        with torch.inference_mode():
            for images, masks in validation:
                dices.append(float(soft_dice(model(images.to(device)), masks.to(device))))
        score = float(np.mean(dices)); history.append({"epoch": epoch, "loss": float(np.mean(losses)), "patch_dice": score})
        print(json.dumps(history[-1]), flush=True)
        if score > best:
            best, stale = score, 0
            torch.save({"model": model.state_dict(), "config": asdict(config), "epoch": epoch, "validation_patch_dice": score}, checkpoint)
        else:
            stale += 1
            if stale >= config.patience: break
    (output / ("final_history.json" if final else "validation_history.json")).write_text(json.dumps(history, indent=2))
    return checkpoint


def load_model(checkpoint: Path, device):
    state = torch.load(checkpoint, map_location=device, weights_only=False)
    config = Config(**{**state["config"], "patch_size": tuple(state["config"]["patch_size"])})
    model = UNet3D(config.base_channels).to(device); model.load_state_dict(state["model"])
    return model, config


def validate(checkpoint: Path, cases, output: Path) -> float:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, config = load_model(checkpoint, device); output.mkdir(parents=True, exist_ok=True)
    records, cached = [], {}
    for case_id, image_path, label_path in cases:
        image, nii = load_normalized(image_path); truth = load_label(label_path, image.shape[:3], nii.affine).astype(bool)
        probability = sliding_probability(model, image, config.patch_size, device); cached[case_id] = (image, truth, probability)
        records.append({"case_id": case_id, **{str(t): dice_score(probability >= t, truth) for t in THRESHOLDS}})
    medians = {t: float(np.median([row[str(t)] for row in records])) for t in THRESHOLDS}
    threshold = max(THRESHOLDS, key=lambda t: (medians[t], -abs(t - .5)))
    scores = np.asarray([row[str(threshold)] for row in records])
    metrics = {"locked_threshold": threshold, "estimated_median_dice_on_labelled_development_data": float(np.median(scores)), "mean_dice": float(scores.mean()), "iqr": [float(np.percentile(scores, 25)), float(np.percentile(scores, 75))], "per_case_dice": {r["case_id"]: r[str(threshold)] for r in records}, "threshold_median_dice": medians}
    (output / "validation_metrics.json").write_text(json.dumps(metrics, indent=2))
    order = np.argsort(scores); selected = {"poor": order[0], "near_median": order[len(order)//2], "good": order[-1]}
    for name, index in selected.items():
        case_id = records[int(index)]["case_id"]; image, truth, probability = cached[case_id]; prediction = probability >= threshold
        count = np.count_nonzero(truth, axis=(0, 1)); slices = np.unique(np.linspace(np.flatnonzero(count)[0], np.flatnonzero(count)[-1], 5).astype(int))
        fig, axes = plt.subplots(1, len(slices), figsize=(4 * len(slices), 4), squeeze=False)
        for axis, z in zip(axes[0], slices):
            base = np.rot90(image[:, :, z, 0]); lo, hi = np.percentile(base[base != 0], (1, 99)); axis.imshow(base, cmap="gray", vmin=lo, vmax=hi)
            rgb = np.zeros((*truth.shape[:2], 4), np.float32); rgb[..., 3] = 0
            tp, fp, fn = prediction[:,:,z] & truth[:,:,z], prediction[:,:,z] & ~truth[:,:,z], ~prediction[:,:,z] & truth[:,:,z]
            rgb[tp], rgb[fp], rgb[fn] = (0,1,0,.65), (1,0,0,.65), (0,0,1,.65)
            axis.imshow(np.rot90(rgb)); axis.set_title(f"z={z}"); axis.axis("off")
        fig.suptitle(f"{name}: {case_id}, Dice={scores[int(index)]:.3f} | green TP, red FP, blue FN"); fig.tight_layout(); fig.savefig(output / f"overlay_{name}_{case_id}.png", dpi=140); plt.close(fig)
    return threshold


def rle_encode(mask: np.ndarray) -> str:
    """Encode in Fortran voxel order as one-indexed start/length pairs."""
    pixels = mask.astype(np.uint8).ravel(order="F")
    padded = np.pad(pixels, (1, 1)); changes = np.flatnonzero(padded[1:] != padded[:-1])
    changes[1::2] -= changes[::2]
    changes[::2] += 1
    return " ".join(map(str, changes))


def predict(checkpoint: Path, image_dir: Path, threshold: float, output: Path) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu"); model, config = load_model(checkpoint, device)
    output.mkdir(parents=True, exist_ok=True); rows = []
    for case_id, image_path, _ in case_paths(image_dir):
        image, nii = load_normalized(image_path); probability = sliding_probability(model, image, config.patch_size, device)
        mask = (probability >= threshold).astype(np.uint8)
        mask_path = output / f"{case_id}.nii.gz"
        nib.save(nib.Nifti1Image(mask, nii.affine, nii.header), mask_path); rows.append((case_id, rle_encode(mask)))
    if len(rows) != 60 or len({row[0] for row in rows}) != 60:
        raise ValueError(f"Expected 60 unique competition cases, found {len(rows)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("split", "train", "validate", "train-final", "predict"))
    parser.add_argument("--images", type=Path, default=Path("data/cases/images")); parser.add_argument("--labels", type=Path, default=Path("data/cases/labels"))
    parser.add_argument("--competition-images", type=Path, default=Path("data/competition/images")); parser.add_argument("--output", type=Path, default=Path("artifacts"))
    parser.add_argument("--checkpoint", type=Path); parser.add_argument("--threshold", type=float); parser.add_argument("--max-epochs", type=int, default=100)
    args = parser.parse_args(); config = Config(max_epochs=args.max_epochs); labelled = case_paths(args.images, args.labels)
    train_ids, val_ids = fixed_split([c[0] for c in labelled], config.seed); by_id = {c[0]: c for c in labelled}
    split = {"seed": config.seed, "training": train_ids, "validation": val_ids}; args.output.mkdir(parents=True, exist_ok=True); (args.output / "split.json").write_text(json.dumps(split, indent=2))
    if args.stage == "split": print(json.dumps(split, indent=2)); return
    if args.stage == "train": train_model([by_id[x] for x in train_ids], [by_id[x] for x in val_ids], config, args.output); return
    if args.stage == "validate": validate(args.checkpoint or args.output / "best_validation_model.pt", [by_id[x] for x in val_ids], args.output); return
    if args.stage == "train-final": train_model(labelled, [], config, args.output, final=True); return
    if args.threshold is None: raise ValueError("--threshold must be the locked validation threshold")
    predict(args.checkpoint or args.output / "final_model.pt", args.competition_images, args.threshold, args.output)


if __name__ == "__main__":
    main()
