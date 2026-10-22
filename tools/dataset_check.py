"""Validate the detector's label files before training (one-time safety check).

Checks each YOLO label is well-formed (5 numbers, normalized 0-1, class 0, box inside
the image), that every image has a matching label, and -- most important -- that no image
appears in more than one split (which would leak test data into training and fake the
metrics). Writes a report + preview images to `runs/checks/`.
"""

from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import pandas as pd

# Run as `python tools/dataset_check.py`; make the shared src/ helpers importable.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from common import (
    IMAGE_EXTENSIONS,
    PROJECT_ROOT,
    SPLITS,
    infer_label_dir,
    load_dataset_yaml,
    resolve_dataset_root,
    resolve_project_path,
)

# Tolerance for box-edge-inside-image checks: absorbs float rounding from the
# annotation export (worst real overflow in this dataset is ~5e-7) without
# hiding genuine out-of-bounds boxes.
EDGE_TOLERANCE = 1e-3


@dataclass
class Issue:
    """One problem found in the dataset (a warning or an error) and where it is."""

    split: str
    image: str
    label: str
    severity: str
    message: str
    line_number: int | str = ""


@dataclass
class Box:
    """A parsed YOLO bounding box (center x/y, width, height — all normalized 0-1)."""

    class_id: int
    x_center: float
    y_center: float
    width: float
    height: float


def selected_splits(split: str) -> tuple[str, ...]:
    """Turn the --split option into the list of splits to check ("all" -> all three)."""
    if split == "all":
        return SPLITS
    return (split,)


def list_images(image_dir: Path) -> list[Path]:
    """List the image files in a folder, sorted."""
    return sorted(
        path
        for path in image_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def list_labels(label_dir: Path) -> list[Path]:
    """List the .txt label files in a folder, sorted."""
    return sorted(path for path in label_dir.iterdir() if path.is_file() and path.suffix.lower() == ".txt")


def add_issue(
    issues: list[Issue],
    split: str,
    image: Path | str,
    label: Path | str,
    severity: str,
    message: str,
    line_number: int | str = "",
) -> None:
    """Record one problem (called wherever a check fails)."""
    issues.append(
        Issue(
            split=split,
            image=str(image),
            label=str(label),
            severity=severity,
            message=message,
            line_number=line_number,
        )
    )


def parse_label_file(label_path: Path, split: str, image_path: Path, issues: list[Issue]) -> list[Box]:
    """Read one label file, validate every box, and return the valid boxes (bad rows logged)."""
    text = label_path.read_text(encoding="utf-8").strip()
    if not text:
        add_issue(issues, split, image_path, label_path, "warning", "Label file is empty.")
        return []

    boxes: list[Box] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        fields = line.split()
        if len(fields) != 5:
            add_issue(
                issues,
                split,
                image_path,
                label_path,
                "error",
                f"Label row must contain exactly 5 fields, found {len(fields)}.",
                line_number,
            )
            continue

        try:
            class_id = int(fields[0])
            values = [float(value) for value in fields[1:]]
            # float() accepts "nan" and "inf". Every range check below is False for NaN,
            # so a NaN box would be dropped silently instead of reported; reject it here.
            if not all(math.isfinite(value) for value in values):
                raise ValueError("non-finite box value")
        except ValueError:
            add_issue(
                issues,
                split,
                image_path,
                label_path,
                "error",
                "Class id and bounding box values must be finite numbers.",
                line_number,
            )
            continue

        if class_id != 0:
            add_issue(
                issues,
                split,
                image_path,
                label_path,
                "error",
                f"Invalid class id {class_id}; expected 0 for license_plate.",
                line_number,
            )

        x_center, y_center, width, height = values
        for name, value in zip(("x_center", "y_center", "width", "height"), values):
            if value < 0.0 or value > 1.0:
                add_issue(
                    issues,
                    split,
                    image_path,
                    label_path,
                    "error",
                    f"{name} must be normalized to [0, 1], found {value}.",
                    line_number,
                )

        if width <= 0.0 or height <= 0.0:
            add_issue(
                issues,
                split,
                image_path,
                label_path,
                "error",
                "Width and height must be positive.",
                line_number,
            )

        edges = {
            "left": x_center - width / 2.0,
            "top": y_center - height / 2.0,
            "right": x_center + width / 2.0,
            "bottom": y_center + height / 2.0,
        }
        for edge_name, edge_value in edges.items():
            if edge_value < -EDGE_TOLERANCE or edge_value > 1.0 + EDGE_TOLERANCE:
                add_issue(
                    issues,
                    split,
                    image_path,
                    label_path,
                    "error",
                    f"Box {edge_name} edge {edge_value:.4f} falls outside the image [0, 1].",
                    line_number,
                )

        if class_id == 0 and all(0.0 <= value <= 1.0 for value in values) and width > 0.0 and height > 0.0:
            boxes.append(Box(class_id, x_center, y_center, width, height))

    return boxes


def draw_preview(image_path: Path, boxes: list[Box], output_path: Path) -> bool:
    """Draw the label boxes on a copy of the image so you can eyeball the annotations."""
    image = cv2.imread(str(image_path))
    if image is None:
        return False

    height, width = image.shape[:2]
    for box in boxes:
        x_center = box.x_center * width
        y_center = box.y_center * height
        box_width = box.width * width
        box_height = box.height * height

        x1 = int(round(x_center - box_width / 2))
        y1 = int(round(y_center - box_height / 2))
        x2 = int(round(x_center + box_width / 2))
        y2 = int(round(y_center + box_height / 2))

        x1 = max(0, min(width - 1, x1))
        x2 = max(0, min(width - 1, x2))
        y1 = max(0, min(height - 1, y1))
        y2 = max(0, min(height - 1, y2))

        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 220, 80), 2)
        cv2.putText(
            image,
            "license_plate",
            (x1, max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 220, 80),
            2,
            cv2.LINE_AA,
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    return cv2.imwrite(str(output_path), image)


def validate_split(
    split: str,
    dataset_root: Path,
    split_image_value: str,
    max_images: int,
    output_dir: Path,
    issues: list[Issue],
) -> dict[str, int]:
    """Check one split: images↔labels match, labels are valid, and save preview images."""
    image_dir = (dataset_root / split_image_value).resolve()
    label_dir = infer_label_dir(dataset_root, split_image_value)

    counts = {
        "images": 0,
        "labels": 0,
        "previews": 0,
    }

    if not image_dir.exists():
        add_issue(issues, split, image_dir, "", "error", "Image folder does not exist.")
        return counts

    if not label_dir.exists():
        add_issue(issues, split, "", label_dir, "error", "Label folder does not exist.")
        return counts

    images = list_images(image_dir)
    labels = list_labels(label_dir)
    counts["images"] = len(images)
    counts["labels"] = len(labels)

    if not images:
        add_issue(issues, split, image_dir, label_dir, "warning", "No images found in split.")

    image_by_stem = {image.stem: image for image in images}
    label_by_stem = {label.stem: label for label in labels}

    for image in images:
        label = label_dir / f"{image.stem}.txt"
        if not label.exists():
            add_issue(issues, split, image, label, "error", "Image has no matching label file.")

    for label in labels:
        if label.stem not in image_by_stem:
            add_issue(issues, split, "", label, "error", "Label file has no matching image file.")

    preview_candidates = images[: max(0, max_images)]
    for index, image in enumerate(preview_candidates, start=1):
        label = label_by_stem.get(image.stem)
        if label is None:
            continue

        boxes = parse_label_file(label, split, image, issues)
        preview_path = output_dir / f"preview_{split}_{index:04d}_{image.stem}.jpg"
        if draw_preview(image, boxes, preview_path):
            counts["previews"] += 1
        else:
            add_issue(issues, split, image, label, "error", "Could not read image for preview.")

    for image in images[max(0, max_images) :]:
        if cv2.imread(str(image)) is None:
            add_issue(issues, split, image, "", "error", "Image could not be read (corrupt or unsupported).")
        label = label_by_stem.get(image.stem)
        if label is not None:
            parse_label_file(label, split, image, issues)

    return counts


def write_outputs(output_dir: Path, issues: list[Issue], split_counts: dict[str, dict[str, int]]) -> tuple[Path, Path]:
    """Write the full issue list to a CSV and a short human-readable summary."""
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "dataset_check_report.csv"
    summary_path = output_dir / "dataset_check_summary.txt"

    report_rows = [
        {
            "split": issue.split,
            "image": issue.image,
            "label": issue.label,
            "severity": issue.severity,
            "message": issue.message,
            "line_number": issue.line_number,
        }
        for issue in issues
    ]
    pd.DataFrame(
        report_rows,
        columns=("split", "image", "label", "severity", "message", "line_number"),
    ).to_csv(report_path, index=False)

    errors = sum(1 for issue in issues if issue.severity == "error")
    warnings = sum(1 for issue in issues if issue.severity == "warning")

    lines = [
        "Dataset check summary",
        f"project_root: {PROJECT_ROOT}",
        f"errors: {errors}",
        f"warnings: {warnings}",
        "",
    ]
    for split, counts in split_counts.items():
        lines.extend(
            [
                f"[{split}]",
                f"images: {counts['images']}",
                f"labels: {counts['labels']}",
                f"previews: {counts['previews']}",
                "",
            ]
        )

    summary_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path, summary_path


def check_cross_split_duplicates(
    dataset_root: Path,
    config: dict[str, Any],
    splits: tuple[str, ...],
    issues: list[Issue],
) -> None:
    """Flag any image stem that appears in more than one split (data leakage)."""
    stem_to_splits: dict[str, list[tuple[str, Path]]] = {}

    for split in splits:
        split_image_value = config.get(split)
        if not split_image_value:
            continue
        image_dir = (dataset_root / str(split_image_value)).resolve()
        if not image_dir.exists():
            continue
        for image in list_images(image_dir):
            stem_to_splits.setdefault(image.stem, []).append((split, image))

    for stem, occurrences in stem_to_splits.items():
        splits_seen = sorted({split for split, _ in occurrences})
        if len(splits_seen) > 1:
            add_issue(
                issues,
                "+".join(splits_seen),
                occurrences[0][1],
                "",
                "error",
                f"Image stem '{stem}' appears in multiple splits {splits_seen} (data leakage).",
            )


def parse_args() -> argparse.Namespace:
    """Read the command-line options (which split to check, how many previews)."""
    parser = argparse.ArgumentParser(description="Validate a YOLO license plate dataset.")
    parser.add_argument("--data", default="configs/dataset.yaml", help="Dataset YAML path.")
    parser.add_argument("--split", choices=("train", "val", "test", "all"), default="all")
    parser.add_argument("--max-images", type=int, default=100, help="Maximum preview images per split.")
    return parser.parse_args()


def main() -> None:
    """Validate the dataset and exit non-zero if any hard errors were found."""
    args = parse_args()
    data_path = resolve_project_path(args.data)
    config = load_dataset_yaml(data_path)
    dataset_root = resolve_dataset_root(data_path, config)
    output_dir = (PROJECT_ROOT / "runs" / "checks").resolve()

    issues: list[Issue] = []
    split_counts: dict[str, dict[str, int]] = {}

    selected = selected_splits(args.split)
    for split in selected:
        # Previews are numbered per run, so after a re-split an old run's files would
        # linger beside the new ones and show images that are no longer in this split.
        for old_preview in output_dir.glob(f"preview_{split}_*.jpg"):
            old_preview.unlink()

        split_image_value = config.get(split)
        if not split_image_value:
            add_issue(issues, split, "", "", "error", f"Dataset YAML does not define '{split}'.")
            continue

        split_counts[split] = validate_split(
            split=split,
            dataset_root=dataset_root,
            split_image_value=str(split_image_value),
            max_images=args.max_images,
            output_dir=output_dir,
            issues=issues,
        )

    if len(selected) > 1:
        check_cross_split_duplicates(dataset_root, config, selected, issues)

    report_path, summary_path = write_outputs(output_dir, issues, split_counts)
    errors = sum(1 for issue in issues if issue.severity == "error")
    warnings = sum(1 for issue in issues if issue.severity == "warning")

    print(f"Dataset check report saved to: {report_path}")
    print(f"Dataset check summary saved to: {summary_path}")
    print(f"Errors: {errors}")
    print(f"Warnings: {warnings}")

    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
