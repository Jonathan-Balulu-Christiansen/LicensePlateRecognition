"""Visualize the training-time image augmentations (a one-time inspection tool).

Reads the augmentation settings straight out of `src/train_detector.py`, applies them to a few
sample training images using Ultralytics' own transforms, and saves side-by-side pictures
to `runs/augmentation_examples/` — so you can actually see what the detector trains on
(flips, color shifts, mosaics, etc.). Not part of the demo.
"""

from __future__ import annotations

import ast
import importlib.util
import json
import math
import random
import sys
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import matplotlib
import numpy as np
import torch
import yaml

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import ultralytics  # noqa: E402
from matplotlib import gridspec  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402
from ultralytics.cfg import DEFAULT_CFG, get_cfg  # noqa: E402
from ultralytics.data.augment import LetterBox, Mosaic, RandomFlip, RandomHSV, RandomPerspective  # noqa: E402
from ultralytics.data.dataset import YOLODataset  # noqa: E402
from ultralytics.data.utils import img2label_paths  # noqa: E402
from ultralytics.utils.instance import Instances  # noqa: E402

SEED = 42
DATASET_YAML = Path("configs/dataset.yaml")
TRAIN_PY = Path("src/train_detector.py")
OUTPUT_DIR = Path("runs/augmentation_examples")
CLASS_NAME = "license plate"
GREEN = "#00b050"
IMAGE_EXTS = {".bmp", ".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp"}
SEGMENT_PLACEHOLDER = np.zeros((0, 1000, 2), dtype=np.float32)
AUGMENTATION_KEYS = [
    "hsv_h",
    "hsv_s",
    "hsv_v",
    "degrees",
    "translate",
    "scale",
    "shear",
    "perspective",
    "flipud",
    "fliplr",
    "bgr",
    "mosaic",
    "mixup",
    "cutmix",
    "copy_paste",
    "copy_paste_mode",
    "close_mosaic",
    "multi_scale",
    "auto_augment",
    "erasing",
]


@dataclass(frozen=True)
class SourceSample:
    """One picked training image with its plate boxes (and how prominent the plate is)."""

    image_path: Path
    label_path: Path
    boxes_xywhn: np.ndarray
    cls: np.ndarray
    score: float


class NoCacheWriteYOLODataset(YOLODataset):
    """Use the real YOLODataset transform pipeline without creating dataset cache files."""

    def get_labels(self) -> list[dict]:
        """Load each training image's size + boxes (without writing a cache file)."""
        self.label_files = img2label_paths(self.im_files)
        labels = []
        for im_file, label_file in zip(self.im_files, self.label_files):
            image = cv2.imread(im_file)
            if image is None:
                raise FileNotFoundError(f"Could not read training image: {im_file}")
            h, w = image.shape[:2]
            boxes, cls = read_yolo_label(Path(label_file))
            labels.append(
                {
                    "im_file": im_file,
                    "shape": (h, w),
                    "cls": cls,
                    "bboxes": boxes,
                    "segments": [],
                    "keypoints": None,
                    "normalized": True,
                    "bbox_format": "xywh",
                }
            )
        if not labels:
            raise RuntimeError(f"No valid labels found under {self.img_path}")
        return labels


class DummyMosaicDataset:
    """A stand-in dataset the Mosaic transform can pull extra images from."""

    cache = None

    def __len__(self) -> int:
        return 0


def project_root() -> Path:
    """The project's top folder."""
    return Path(__file__).resolve().parents[1]


def seed_all(seed: int) -> None:
    """Fix all random seeds so the example images come out the same each run."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def parse_train_config(train_path: Path) -> dict[str, Any]:
    """Read the training settings out of src/train_detector.py by parsing the file (no import/run)."""
    if not train_path.exists():
        raise FileNotFoundError(f"Missing training script: {train_path}")

    tree = ast.parse(train_path.read_text(encoding="utf-8"), filename=str(train_path))
    result: dict[str, Any] = {"base_model": None, "run_name": None, "device": None, "train_params": {}}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if not isinstance(target, ast.Name):
                continue
            if target.id in {"base_model", "run_name", "device"}:
                try:
                    result[target.id] = ast.literal_eval(node.value)
                except (ValueError, SyntaxError):
                    result[target.id] = None
            elif target.id == "train_params":
                try:
                    value = ast.literal_eval(node.value)
                except (ValueError, SyntaxError) as exc:
                    raise RuntimeError(f"Could not parse train_params in {train_path}") from exc
                if not isinstance(value, dict):
                    raise RuntimeError(f"train_params in {train_path} is not a dictionary")
                result["train_params"] = value

    result["augmentation_overrides"] = {
        k: v for k, v in result["train_params"].items() if k in AUGMENTATION_KEYS
    }
    return result


def load_dataset_yaml(yaml_path: Path) -> dict[str, Any]:
    """Load and sanity-check the dataset YAML (paths + class names)."""
    if not yaml_path.exists():
        raise FileNotFoundError(f"Missing dataset YAML: {yaml_path}")
    with yaml_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise RuntimeError(f"Dataset YAML is not a mapping: {yaml_path}")
    for key in ("path", "train", "names"):
        if key not in data:
            raise RuntimeError(f"Dataset YAML missing required key '{key}': {yaml_path}")
    names = data["names"]
    if isinstance(names, dict):
        data["names"] = {int(k): str(v) for k, v in names.items()}
    elif isinstance(names, list):
        data["names"] = {i: str(v) for i, v in enumerate(names)}
    else:
        raise RuntimeError("Dataset YAML 'names' must be a list or dictionary")
    data["nc"] = len(data["names"])
    return data


def resolve_dataset_paths(root: Path, dataset_yaml_path: Path, data: dict[str, Any]) -> tuple[Path, Path]:
    """Work out the train image + label folders from the dataset YAML."""
    data_root = Path(data["path"])
    if not data_root.is_absolute():
        data_root = (dataset_yaml_path.parent / data_root).resolve()

    train_images = data_root / Path(data["train"])
    label_parts = ["labels" if part == "images" else part for part in Path(data["train"]).parts]
    train_labels = data_root / Path(*label_parts)

    if not train_images.exists():
        raise FileNotFoundError(f"Missing train image directory from {dataset_yaml_path}: {train_images}")
    if not train_labels.exists():
        raise FileNotFoundError(f"Missing train label directory inferred from dataset YAML: {train_labels}")
    if not train_images.is_relative_to(root) or not train_labels.is_relative_to(root):
        raise RuntimeError("Resolved dataset paths must remain inside the opened project folder")
    return train_images, train_labels


def read_yolo_label(label_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Read one YOLO label file into arrays of boxes and class ids."""
    if not label_path.exists():
        raise FileNotFoundError(f"Missing YOLO label file: {label_path}")
    boxes: list[list[float]] = []
    classes: list[list[float]] = []
    for line_no, line in enumerate(label_path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        parts = line.split()
        if len(parts) < 5:
            raise RuntimeError(f"Invalid YOLO label at {label_path}:{line_no}")
        cls, x, y, w, h = map(float, parts[:5])
        if w <= 0 or h <= 0:
            continue
        classes.append([cls])
        boxes.append([x, y, w, h])
    return np.array(boxes, dtype=np.float32), np.array(classes, dtype=np.float32)


def yolo_to_xyxy(boxes_xywhn: np.ndarray, w: int, h: int) -> np.ndarray:
    """Convert normalized YOLO boxes to pixel corner coordinates (x1,y1,x2,y2)."""
    if len(boxes_xywhn) == 0:
        return np.zeros((0, 4), dtype=np.float32)
    boxes = boxes_xywhn.astype(np.float32).copy()
    boxes[:, [0, 2]] *= w
    boxes[:, [1, 3]] *= h
    xyxy = np.empty_like(boxes)
    xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2
    xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2
    xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2
    xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2
    xyxy[:, [0, 2]] = xyxy[:, [0, 2]].clip(0, w)
    xyxy[:, [1, 3]] = xyxy[:, [1, 3]].clip(0, h)
    return xyxy


def select_training_samples(train_images: Path, train_labels: Path, count: int = 4) -> list[SourceSample]:
    """Pick a few good training images (prefer ones with a clear, sizeable plate)."""
    candidates: list[SourceSample] = []
    fallback: list[SourceSample] = []
    for image_path in sorted(p for p in train_images.iterdir() if p.suffix.lower() in IMAGE_EXTS):
        label_path = train_labels / f"{image_path.stem}.txt"
        if not label_path.exists():
            continue
        image = cv2.imread(str(image_path))
        if image is None:
            continue
        h, w = image.shape[:2]
        boxes, cls = read_yolo_label(label_path)
        if len(boxes) == 0:
            continue

        xyxy = yolo_to_xyxy(boxes, w, h)
        widths = xyxy[:, 2] - xyxy[:, 0]
        heights = xyxy[:, 3] - xyxy[:, 1]
        areas = widths * heights / float(w * h)
        aspect = widths / np.maximum(heights, 1e-6)
        best_i = int(np.argmax(areas))
        best = SourceSample(image_path, label_path, boxes, cls, float(areas[best_i]))
        fallback.append(best)
        if areas[best_i] >= 0.006 and widths[best_i] >= 70 and heights[best_i] >= 25 and 0.8 <= aspect[best_i] <= 5.0:
            candidates.append(best)

    if len(candidates) >= count:
        return random.Random(SEED).sample(candidates, count)
    if len(fallback) >= count:
        fallback = sorted(fallback, key=lambda s: (s.score, s.image_path.name), reverse=True)
        return fallback[:count]
    raise RuntimeError(f"Need at least {count} labeled training images, found {len(fallback)}")


def load_training_image(image_path: Path, imgsz: int) -> tuple[np.ndarray, tuple[int, int], tuple[int, int]]:
    """Read an image and resize its long side to imgsz (like training does)."""
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
    h0, w0 = image.shape[:2]
    r = imgsz / max(h0, w0)
    if r != 1:
        resized_w, resized_h = min(math.ceil(w0 * r), imgsz), min(math.ceil(h0 * r), imgsz)
        image = cv2.resize(image, (resized_w, resized_h), interpolation=cv2.INTER_LINEAR)
    return image, (h0, w0), image.shape[:2]


def make_ultralytics_label(sample: SourceSample, imgsz: int) -> dict[str, Any]:
    """Package a sample into the dict format Ultralytics' transforms expect."""
    image, ori_shape, resized_shape = load_training_image(sample.image_path, imgsz)
    return {
        "im_file": str(sample.image_path),
        "ori_shape": ori_shape,
        "resized_shape": resized_shape,
        "img": image,
        "cls": sample.cls.copy(),
        "instances": Instances(
            bboxes=sample.boxes_xywhn.copy(),
            segments=SEGMENT_PLACEHOLDER.copy(),
            bbox_format="xywh",
            normalized=True,
        ),
    }


def labels_to_boxes_xyxy(labels: dict[str, Any]) -> np.ndarray:
    """Pull the (denormalized, cleaned) corner boxes out of a transformed label dict."""
    image = labels["img"]
    h, w = image.shape[:2]
    instances = deepcopy(labels["instances"])
    instances.convert_bbox("xyxy")
    instances.denormalize(w, h)
    instances.clip(w, h)
    instances.remove_zero_area_boxes()
    boxes = instances.bboxes.astype(np.float32)
    if len(boxes) == 0:
        return boxes
    widths = boxes[:, 2] - boxes[:, 0]
    heights = boxes[:, 3] - boxes[:, 1]
    keep = (widths >= 2.0) & (heights >= 2.0)
    return boxes[keep]


def formatted_sample_to_image_boxes(sample: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    """Turn a finished pipeline sample (tensor) back into an image + pixel boxes."""
    image = sample["img"].detach().cpu().permute(1, 2, 0).numpy()
    boxes = sample["bboxes"].detach().cpu().numpy().astype(np.float32)
    h, w = image.shape[:2]
    if len(boxes) == 0:
        return image, np.zeros((0, 4), dtype=np.float32)
    xyxy = np.empty_like(boxes)
    xyxy[:, 0] = (boxes[:, 0] - boxes[:, 2] / 2) * w
    xyxy[:, 1] = (boxes[:, 1] - boxes[:, 3] / 2) * h
    xyxy[:, 2] = (boxes[:, 0] + boxes[:, 2] / 2) * w
    xyxy[:, 3] = (boxes[:, 1] + boxes[:, 3] / 2) * h
    xyxy[:, [0, 2]] = xyxy[:, [0, 2]].clip(0, w)
    xyxy[:, [1, 3]] = xyxy[:, [1, 3]].clip(0, h)
    widths = xyxy[:, 2] - xyxy[:, 0]
    heights = xyxy[:, 3] - xyxy[:, 1]
    keep = (widths >= 2.0) & (heights >= 2.0)
    return image, xyxy[keep]


def bgr_to_rgb(image: np.ndarray) -> np.ndarray:
    """Swap OpenCV's BGR channel order to RGB for matplotlib."""
    if image.ndim == 3 and image.shape[2] == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


def draw_boxes(ax: plt.Axes, boxes: np.ndarray, annotate: bool = True) -> None:
    """Draw green rectangles for each box on a matplotlib axis."""
    h = ax.images[0].get_array().shape[0]
    lw = max(1.8, h / 480.0)
    for box in boxes:
        x1, y1, x2, y2 = box.tolist()
        bw, bh = x2 - x1, y2 - y1
        rect = Rectangle((x1, y1), bw, bh, linewidth=lw, edgecolor=GREEN, facecolor="none")
        ax.add_patch(rect)
        if annotate and bw >= 120 and bh >= 28:
            ax.text(
                x1,
                max(0, y1 - 4),
                CLASS_NAME,
                fontsize=7,
                color="white",
                va="bottom",
                bbox={"facecolor": GREEN, "edgecolor": "none", "alpha": 0.9, "pad": 1.5},
            )


def add_panel(
    ax: plt.Axes,
    image: np.ndarray,
    boxes: np.ndarray,
    title: str,
    rgb: bool = False,
    annotate: bool = True,
) -> None:
    """Add one titled image panel (with its boxes) to a figure."""
    ax.imshow(image if rgb else bgr_to_rgb(image))
    draw_boxes(ax, boxes, annotate=annotate and len(boxes) <= 2)
    ax.set_title(title, fontsize=11, pad=7)
    ax.set_axis_off()


def visible_transform(
    base_label: dict[str, Any],
    transform: Any,
    seed_start: int,
    trials: int = 80,
) -> dict[str, Any]:
    """Apply one augmentation a few times and keep the clearest, most visible result."""
    original = base_label["img"]
    best: dict[str, Any] | None = None
    best_score = -1.0
    for offset in range(trials):
        seed_all(seed_start + offset)
        candidate = transform(deepcopy(base_label))
        boxes = labels_to_boxes_xyxy(candidate)
        if len(boxes) == 0:
            continue
        image = candidate["img"]
        reference = cv2.resize(original, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_LINEAR)
        diff = float(np.mean(np.abs(image.astype(np.float32) - reference.astype(np.float32))))
        score = diff + 0.01 * float(np.mean((boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])))
        if score > best_score:
            best = candidate
            best_score = score
    if best is None:
        raise RuntimeError(f"Could not produce a valid transformed sample with {transform.__class__.__name__}")
    return best


def create_individual_figures(samples: list[SourceSample], cfg: Any, out_dir: Path) -> list[str]:
    """Save one before/after picture per augmentation type (HSV, flip, scale, combined)."""
    written: list[str] = []
    for i, sample in enumerate(samples[:3], start=1):
        base = make_ultralytics_label(sample, int(cfg.imgsz))
        panels: list[tuple[str, dict[str, Any]]] = [("Original", base)]

        if any(getattr(cfg, k) > 0 for k in ("hsv_h", "hsv_s", "hsv_v")):
            hsv = RandomHSV(hgain=cfg.hsv_h, sgain=cfg.hsv_s, vgain=cfg.hsv_v)
            panels.append(("HSV", visible_transform(base, hsv, SEED + 100 * i)))

        if cfg.fliplr > 0:
            hflip = RandomFlip(direction="horizontal", p=cfg.fliplr)
            panels.append(("Horizontal Flip", visible_transform(base, hflip, SEED + 200 * i)))

        if cfg.translate > 0 or cfg.scale > 0:
            affine = RandomPerspective(
                degrees=0.0,
                translate=cfg.translate,
                scale=cfg.scale,
                shear=0.0,
                perspective=0.0,
                pre_transform=LetterBox(new_shape=(cfg.imgsz, cfg.imgsz)),
            )
            panels.append(("Scale + Translate", visible_transform(base, affine, SEED + 300 * i)))

        combined = [
            RandomPerspective(
                degrees=cfg.degrees,
                translate=cfg.translate,
                scale=cfg.scale,
                shear=cfg.shear,
                perspective=cfg.perspective,
                pre_transform=LetterBox(new_shape=(cfg.imgsz, cfg.imgsz)),
            )
        ]
        if any(getattr(cfg, k) > 0 for k in ("hsv_h", "hsv_s", "hsv_v")):
            combined.append(RandomHSV(hgain=cfg.hsv_h, sgain=cfg.hsv_s, vgain=cfg.hsv_v))
        if cfg.flipud > 0:
            combined.append(RandomFlip(direction="vertical", p=cfg.flipud))
        if cfg.fliplr > 0:
            combined.append(RandomFlip(direction="horizontal", p=cfg.fliplr))
        if len(combined) > 1 or cfg.translate > 0 or cfg.scale > 0:
            panels.append(("Combined Augmentations", visible_transform(base, _Sequential(combined), SEED + 400 * i)))

        fig, axes = plt.subplots(1, len(panels), figsize=(3.25 * len(panels), 3.6), dpi=300)
        if len(panels) == 1:
            axes = [axes]
        for ax, (title, labels) in zip(axes, panels):
            add_panel(ax, labels["img"], labels_to_boxes_xyxy(labels), title)
        fig.patch.set_facecolor("white")
        fig.tight_layout(pad=0.6, w_pad=0.9)
        output = out_dir / f"{i:02d}_sample_individual_augmentations.png"
        fig.savefig(output, dpi=300, facecolor="white")
        plt.close(fig)
        written.append(output.name)
    return written


class _Sequential:
    """Chain several transforms so they run one after another."""

    def __init__(self, transforms: list[Any]) -> None:
        self.transforms = transforms

    def __call__(self, labels: dict[str, Any]) -> dict[str, Any]:
        """Apply each transform in turn to the sample."""
        for transform in self.transforms:
            labels = transform(labels)
        return labels


def create_mosaic_figure(samples: list[SourceSample], cfg: Any, out_dir: Path) -> str:
    """Save an example of the Mosaic augmentation (4 images stitched into one)."""
    labels = [make_ultralytics_label(sample, int(cfg.imgsz)) for sample in samples[:4]]
    mosaic = Mosaic(DummyMosaicDataset(), imgsz=int(cfg.imgsz), p=cfg.mosaic, n=4)
    expected_boxes = sum(len(label["cls"]) for label in labels)
    best_output: dict[str, Any] | None = None
    best_score = -1.0
    for offset in range(300):
        seed_all(SEED + 800 + offset)
        mosaic_input = deepcopy(labels[0])
        mosaic_input["mix_labels"] = [deepcopy(x) for x in labels[1:4]]
        candidate = mosaic._mix_transform(mosaic_input)
        boxes = labels_to_boxes_xyxy(candidate)
        gray = np.array([114, 114, 114], dtype=np.uint8)
        gray_ratio = float(np.mean(np.all(candidate["img"] == gray, axis=2)))
        box_area = 0.0
        if len(boxes):
            box_area = float(np.sum((boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])))
            box_area /= float(candidate["img"].shape[0] * candidate["img"].shape[1])
        score = 1000.0 * len(boxes) + 20.0 * box_area - gray_ratio
        if score > best_score:
            best_output = candidate
            best_score = score
        if len(boxes) >= expected_boxes and gray_ratio < 0.08:
            break
    if best_output is None:
        raise RuntimeError("Could not produce a valid mosaic output")
    mosaic_output = best_output

    fig = plt.figure(figsize=(12.4, 6.4), dpi=300, facecolor="white")
    spec = gridspec.GridSpec(2, 4, figure=fig, width_ratios=[1, 1, 1.75, 1.75], wspace=0.18, hspace=0.24)
    positions = [(0, 0), (0, 1), (1, 0), (1, 1)]
    for idx, (row, col) in enumerate(positions):
        ax = fig.add_subplot(spec[row, col])
        add_panel(ax, labels[idx]["img"], labels_to_boxes_xyxy(labels[idx]), f"Original {idx + 1}", annotate=False)

    ax_mosaic = fig.add_subplot(spec[:, 2:])
    add_panel(ax_mosaic, mosaic_output["img"], labels_to_boxes_xyxy(mosaic_output), "Mosaic", annotate=False)
    output = out_dir / "04_mosaic_augmentation.png"
    fig.savefig(output, dpi=300, facecolor="white", bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)
    return output.name


def build_actual_dataset(train_images: Path, data: dict[str, Any], cfg: Any) -> NoCacheWriteYOLODataset:
    """Build the real Ultralytics dataset using the exact training transform pipeline."""
    return NoCacheWriteYOLODataset(
        img_path=str(train_images),
        imgsz=int(cfg.imgsz),
        batch_size=int(cfg.batch),
        augment=True,
        hyp=cfg,
        rect=False,
        cache=None,
        single_cls=False,
        stride=32,
        pad=0.0,
        prefix="train: ",
        task="detect",
        classes=None,
        data=data,
        fraction=float(cfg.fraction),
    )


def create_actual_pipeline_figure(train_images: Path, data: dict[str, Any], cfg: Any, out_dir: Path) -> str:
    """Save a grid of images as the full training pipeline actually produces them."""
    dataset = build_actual_dataset(train_images, data, cfg)
    rng = random.Random(SEED)
    warmup = rng.sample(range(len(dataset)), min(64, len(dataset)))
    for idx in warmup:
        dataset.get_image_and_label(idx)

    preview_indices = rng.sample(range(len(dataset)), min(6, len(dataset)))
    panels: list[tuple[np.ndarray, np.ndarray, str]] = []
    for i, idx in enumerate(preview_indices, start=1):
        seed_all(SEED + 1000 + i)
        sample = dataset[idx]
        image, boxes = formatted_sample_to_image_boxes(sample)
        panels.append((image, boxes, f"Sample {i}"))

    cols = 3
    rows = math.ceil(len(panels) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(10.5, 3.75 * rows), dpi=300, facecolor="white")
    axes_arr = np.array(axes).reshape(-1)
    for ax, (image, boxes, title) in zip(axes_arr, panels):
        add_panel(ax, image, boxes, title, rgb=True, annotate=False)
    for ax in axes_arr[len(panels) :]:
        ax.set_axis_off()
    fig.tight_layout(pad=0.7, w_pad=0.8, h_pad=1.0)
    output = out_dir / "05_actual_training_pipeline_samples.png"
    fig.savefig(output, dpi=300, facecolor="white")
    plt.close(fig)
    return output.name


def active_and_omitted(cfg: Any, albumentations_installed: bool) -> tuple[list[str], dict[str, str]]:
    """List which augmentations are turned on vs off (and the reason each is off)."""
    active: list[str] = []
    omitted: dict[str, str] = {}
    if any(getattr(cfg, k) > 0 for k in ("hsv_h", "hsv_s", "hsv_v")):
        active.append("HSV color augmentation")
    else:
        omitted["HSV color augmentation"] = "hsv_h=hsv_s=hsv_v=0"
    if cfg.fliplr > 0:
        active.append("Horizontal flip")
    else:
        omitted["Horizontal flip"] = "fliplr=0"
    if cfg.translate > 0 or cfg.scale > 0:
        active.append("Scale + translation affine")
    else:
        omitted["Scale + translation affine"] = "translate=0 and scale=0"
    if cfg.mosaic > 0:
        active.append("Mosaic")
    else:
        omitted["Mosaic"] = "mosaic=0"

    inactive_checks = {
        "Rotation": ("degrees", cfg.degrees),
        "Shear": ("shear", cfg.shear),
        "Perspective": ("perspective", cfg.perspective),
        "Vertical flip": ("flipud", cfg.flipud),
        "MixUp": ("mixup", cfg.mixup),
        "CutMix": ("cutmix", cfg.cutmix),
        "Copy-Paste": ("copy_paste", cfg.copy_paste),
        "BGR channel swap": ("bgr", cfg.bgr),
        "Multi-scale training": ("multi_scale", cfg.multi_scale),
    }
    for name, (key, value) in inactive_checks.items():
        if float(value) <= 0:
            omitted[name] = f"{key}={value}"
        elif name not in active:
            active.append(name)
    if not albumentations_installed:
        omitted["Default Albumentations wrapper"] = "albumentations is not installed; wrapper is a no-op"
    omitted["Classification auto_augment"] = (
        "auto_augment is classification-only and is not used by detect v8_transforms"
    )
    omitted["Classification random erasing"] = "erasing is classification-only and is not used by detect v8_transforms"
    return active, omitted


def write_config_record(
    out_dir: Path,
    root: Path,
    dataset_yaml: Path,
    train_info: dict[str, Any],
    data: dict[str, Any],
    train_images: Path,
    train_labels: Path,
    samples: list[SourceSample],
    cfg: Any,
    included_figures: list[str],
    active: list[str],
    omitted: dict[str, str],
    albumentations_installed: bool,
) -> str:
    """Save a JSON record of the exact settings and files used for these examples."""
    params = {key: getattr(cfg, key) for key in AUGMENTATION_KEYS if hasattr(cfg, key)}
    record = {
        "ultralytics_version": ultralytics.__version__,
        "random_seed": SEED,
        "dataset_yaml_path": str(dataset_yaml.relative_to(root)),
        "train_py_path": str((root / TRAIN_PY).relative_to(root)),
        "training_image_size": int(cfg.imgsz),
        "detector_model": train_info["base_model"],
        "detection_class_names": data["names"],
        "train_split": {
            "images_dir": str(train_images.relative_to(root)),
            "labels_dir": str(train_labels.relative_to(root)),
            "selected_source_images": [str(s.image_path.relative_to(root)) for s in samples],
        },
        "train_py": {
            "train_params": train_info["train_params"],
            "augmentation_overrides_in_train_py": train_info["augmentation_overrides"],
        },
        "augmentation_parameters": params,
        "albumentations_installed": albumentations_installed,
        "included_in_figures": active,
        "generated_files": included_figures,
        "omitted_inactive_or_not_applicable": omitted,
        "notes": [
            "Mosaic is active during normal training with mosaic=1.0, then disabled for the final close_mosaic epochs.",
            "Individual figures use the installed Ultralytics transform classes with deterministic seeds.",
            "The actual pipeline preview is generated through the installed YOLODataset transform stack.",
        ],
    }
    output = out_dir / "augmentation_config.json"
    output.write_text(json.dumps(record, indent=2), encoding="utf-8")
    return output.name


def print_config_summary(cfg: Any, active: list[str], omitted: dict[str, str], dataset_yaml: Path) -> None:
    """Print the active/omitted augmentations and their values to the console."""
    params = {key: getattr(cfg, key) for key in AUGMENTATION_KEYS if hasattr(cfg, key)}
    print(f"Ultralytics version: {ultralytics.__version__}")
    print(f"Dataset YAML: {dataset_yaml}")
    print(f"Training image size: {cfg.imgsz}")
    print("Active augmentation parameters:")
    print(json.dumps(params, indent=2))
    print("Included active augmentations:")
    for item in active:
        print(f"  - {item}")
    print("Omitted inactive/not-applicable augmentations:")
    for name, reason in omitted.items():
        print(f"  - {name}: {reason}")


def main() -> None:
    """Generate all the augmentation example images from src/train_detector.py's settings."""
    root = project_root()
    dataset_yaml = root / DATASET_YAML
    train_py = root / TRAIN_PY
    out_dir = root / OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    train_info = parse_train_config(train_py)
    data = load_dataset_yaml(dataset_yaml)
    train_images, train_labels = resolve_dataset_paths(root, dataset_yaml, data)
    overrides = {
        "task": "detect",
        "mode": "train",
        "model": train_info["base_model"],
        "data": str(dataset_yaml),
        **train_info["train_params"],
    }
    cfg = get_cfg(DEFAULT_CFG, overrides)
    albumentations_installed = importlib.util.find_spec("albumentations") is not None
    active, omitted = active_and_omitted(cfg, albumentations_installed)
    print_config_summary(cfg, active, omitted, dataset_yaml)

    selected = select_training_samples(train_images, train_labels, count=4)
    print("Selected training images:")
    for sample in selected:
        print(f"  - {sample.image_path.relative_to(root)}")

    written = []
    written.extend(create_individual_figures(selected, cfg, out_dir))
    written.append(create_mosaic_figure(selected, cfg, out_dir))
    written.append(create_actual_pipeline_figure(train_images, data, cfg, out_dir))
    written.append(
        write_config_record(
            out_dir,
            root,
            dataset_yaml,
            train_info,
            data,
            train_images,
            train_labels,
            selected,
            cfg,
            written,
            active,
            omitted,
            albumentations_installed,
        )
    )

    missing = [name for name in written if not (out_dir / name).exists()]
    if missing:
        raise RuntimeError(f"Expected output files were not created: {missing}")
    print("Generated output files:")
    for name in written:
        print(f"  - {OUTPUT_DIR / name}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise
