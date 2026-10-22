"""Create human-reviewable YOLO pre-labels for a folder of plate images.

One-time data-prep helper (not part of the live demo): run the trained detector
over raw images and triage each into one of three piles for a human to review:

  auto_labeled/  -- one confident, in-bounds box; label written
  review/        -- a box was found but something is off (low conf, multiple
                    boxes, clamped edges, very large); label written, eyeball it
  needs_manual/  -- no usable detection; draw the box by hand

A manifest CSV records the decision for every image. This is a review draft, not
final training data -- verify the piles before training.
"""

from __future__ import annotations

import argparse
import csv
import shutil
import sys
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

from ultralytics import YOLO

# Run as `python tools/auto_label.py`; make the shared src/ helpers importable.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from common import IMAGE_EXTENSIONS, check_cuda_or_fail, resolve_project_path

MANIFEST_FIELDS = [
    "image_path",
    "pile",
    "num_boxes_kept",
    "num_boxes_discarded_small",
    "best_confidence",
    "flags",
]


def parse_args() -> argparse.Namespace:
    """Read the command-line options (weights, source/output folders, thresholds)."""
    parser = argparse.ArgumentParser(
        description="Create human-reviewable YOLO license-plate pre-labels."
    )
    parser.add_argument("--weights", required=True, help="Path to trained YOLO weights.")
    parser.add_argument("--source", required=True, help="Folder containing input images.")
    parser.add_argument("--output", required=True, help="Root folder for triaged output.")
    parser.add_argument("--conf", type=float, default=0.25, help="Inference threshold.")
    parser.add_argument(
        "--review-conf",
        type=float,
        default=0.45,
        help="Detections below this confidence go to review.",
    )
    parser.add_argument(
        "--min-box-h",
        type=float,
        default=0.02,
        help="Minimum box height as a fraction of image height.",
    )
    parser.add_argument("--imgsz", type=int, default=640, help="Inference image size (the detector is trained at 640).")
    parser.add_argument("--device", default="0", help="Inference device, for example 0 or cpu.")
    parser.add_argument("--batch", type=int, default=16, help="Images per streamed inference batch.")
    return parser.parse_args()


def validate_inputs(args: argparse.Namespace) -> tuple[Path, Path, Path]:
    """Check the weights/source exist and the output isn't inside the source; return the paths."""
    weights = resolve_project_path(args.weights)
    source = resolve_project_path(args.source)
    output = resolve_project_path(args.output)

    if not weights.is_file():
        raise FileNotFoundError(f"Weights not found: {weights}")
    if not source.is_dir():
        raise NotADirectoryError(f"Source folder not found: {source}")
    if output == source or output.is_relative_to(source):
        raise ValueError("Output must not be the source folder or a folder inside it.")
    return weights, source, output


def find_images(source: Path) -> list[Path]:
    """Find every image under the source folder (recursively)."""
    images = sorted(
        path.resolve()
        for path in source.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not images:
        raise FileNotFoundError(f"No supported images found under: {source}")
    return images


def make_output_folders(output: Path) -> dict[str, Path]:
    """Create the three review piles (auto_labeled / review / needs_manual)."""
    folders = {
        "auto_images": output / "auto_labeled" / "images",
        "auto_labels": output / "auto_labeled" / "labels",
        "review_images": output / "review" / "images",
        "review_labels": output / "review" / "labels",
        "manual_images": output / "needs_manual" / "images",
    }
    for folder in folders.values():
        folder.mkdir(parents=True, exist_ok=True)
    return folders


def chunks(items: Sequence[Path], size: int) -> Iterator[Sequence[Path]]:
    """Yield the list in batches of `size` (so YOLO processes several images at once)."""
    for start in range(0, len(items), size):
        yield items[start : start + size]


def analyze_result(
    result: Any,
    conf_threshold: float,
    review_conf: float,
    min_box_h: float,
) -> dict[str, object]:
    """Look at one image's detections and decide its pile + build the YOLO label line."""
    image_height, image_width = result.orig_shape

    candidates: list[dict[str, object]] = []
    discarded_small = 0
    for box in result.boxes or []:
        x1, y1, x2, y2 = box.xyxy[0].detach().cpu().tolist()
        if (y2 - y1) / image_height < min_box_h:
            discarded_small += 1
            continue
        confidence = float(box.conf[0].detach().cpu().item())
        if confidence >= conf_threshold:
            candidates.append({"confidence": confidence, "xyxy": (x1, y1, x2, y2)})

    if not candidates:
        return {
            "pile": "needs_manual",
            "num_boxes_kept": 0,
            "num_boxes_discarded_small": discarded_small,
            "best_confidence": None,
            "flags": [],
            "label": None,
        }

    best = max(candidates, key=lambda candidate: float(candidate["confidence"]))
    confidence = float(best["confidence"])
    x1, y1, x2, y2 = best["xyxy"]

    normalized = (x1 / image_width, y1 / image_height, x2 / image_width, y2 / image_height)
    cx1, cy1, cx2, cy2 = (min(1.0, max(0.0, value)) for value in normalized)
    clamped = (cx1, cy1, cx2, cy2) != normalized
    box_width = cx2 - cx1
    box_height = cy2 - cy1

    flags: list[str] = []
    if confidence < review_conf:
        flags.append("low_confidence")
    if len(candidates) > 1:
        flags.append("multiple_boxes")
    if clamped:
        flags.append("out_of_bounds")
    if box_width * box_height > 0.5:
        flags.append("large_box")

    label = (0, (cx1 + cx2) / 2.0, (cy1 + cy2) / 2.0, box_width, box_height)
    return {
        "pile": "review" if flags else "auto_labeled",
        "num_boxes_kept": len(candidates),
        "num_boxes_discarded_small": discarded_small,
        "best_confidence": confidence,
        "flags": flags,
        "label": label,
    }


def write_output(
    source: Path,
    image_path: Path,
    analysis: dict[str, object],
    folders: dict[str, Path],
) -> None:
    """Copy the image into its pile and, if labeled, write the label .txt next to it."""
    relative_image = image_path.relative_to(source)
    pile = analysis["pile"]

    if pile == "auto_labeled":
        image_destination = folders["auto_images"] / relative_image
        label_destination = folders["auto_labels"] / relative_image.with_suffix(".txt")
    elif pile == "review":
        image_destination = folders["review_images"] / relative_image
        label_destination = folders["review_labels"] / relative_image.with_suffix(".txt")
    else:
        image_destination = folders["manual_images"] / relative_image
        label_destination = None

    image_destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(image_path, image_destination)

    label = analysis["label"]
    if label_destination is not None and label is not None:
        label_destination.parent.mkdir(parents=True, exist_ok=True)
        class_id, x_center, y_center, width, height = label
        label_destination.write_text(
            f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n",
            encoding="utf-8",
        )


def manifest_row(image_path: Path, analysis: dict[str, object]) -> dict[str, object]:
    """Build one CSV row summarizing what was decided for an image."""
    confidence = analysis["best_confidence"]
    return {
        "image_path": str(image_path),
        "pile": analysis["pile"],
        "num_boxes_kept": analysis["num_boxes_kept"],
        "num_boxes_discarded_small": analysis["num_boxes_discarded_small"],
        "best_confidence": "" if confidence is None else f"{float(confidence):.6f}",
        "flags": ";".join(str(flag) for flag in analysis["flags"]),
    }


def run_auto_labeling(
    model: YOLO,
    images: Sequence[Path],
    source: Path,
    output: Path,
    folders: dict[str, Path],
    args: argparse.Namespace,
) -> dict[str, int]:
    """Run the detector over all images, sort each into a pile, and write the manifest CSV."""
    counts = {"auto_labeled": 0, "review": 0, "needs_manual": 0, "total": 0}
    manifest_path = output / "auto_label_manifest.csv"

    with manifest_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()

        for image_batch in chunks(images, args.batch):
            results = model.predict(
                source=[str(path) for path in image_batch],
                imgsz=args.imgsz,
                conf=args.conf,
                device=args.device,
                stream=True,
                verbose=False,
            )
            for image_path, result in zip(image_batch, results):
                analysis = analyze_result(result, args.conf, args.review_conf, args.min_box_h)
                write_output(source, image_path, analysis, folders)
                writer.writerow(manifest_row(image_path, analysis))

                counts[str(analysis["pile"])] += 1
                counts["total"] += 1
                if counts["total"] % 25 == 0 or counts["total"] == len(images):
                    print(f"Processed {counts['total']}/{len(images)} images")

    return counts


def print_summary(counts: dict[str, int], output: Path) -> None:
    """Print how many images went to each pile, plus the review reminder."""
    print()
    print("Auto-labeling summary")
    print(f"Total images processed: {counts['total']}")
    print(f"auto_labeled: {counts['auto_labeled']}")
    print(f"review: {counts['review']}")
    print(f"needs_manual: {counts['needs_manual']}")
    print(f"Output root: {output}")
    print()
    print("This output is a review draft, not final training data.")
    print("Verify auto_labeled/ and review/ before training; draw boxes for needs_manual/.")


def main() -> int:
    """Run the auto-labeling pass and print a summary (returns an exit code)."""
    args = parse_args()
    try:
        weights, source, output = validate_inputs(args)
        images = find_images(source)
        check_cuda_or_fail(args.device)
        folders = make_output_folders(output)

        print(f"Weights: {weights}")
        print(f"Source: {source} ({len(images)} images)")
        print(f"Output: {output}")

        model = YOLO(str(weights))
        counts = run_auto_labeling(model, images, source, output, folders, args)
        print_summary(counts, output)
        return 0
    except (FileNotFoundError, NotADirectoryError, ValueError, RuntimeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
