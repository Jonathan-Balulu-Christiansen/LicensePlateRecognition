"""Run the DETECTOR on images/videos and save the results (detector only — no OCR).

Loads trained weights, runs YOLO on an image / folder / video / camera, and writes an
annotated copy plus a `detections.csv` (one row per detected plate box). Handy for
checking the detector on its own, separate from the full app.
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import pandas as pd
from ultralytics import YOLO

from common import (
    PROJECT_ROOT,
    VIDEO_EXTENSIONS,
    check_cuda_or_fail,
    create_run_dir,
    is_camera_source,
    resolve_project_path,
    write_run_metadata,
)


def source_is_video(source: str) -> bool:
    """True if the source is a video or camera (so we number frames instead of just 0)."""
    if is_camera_source(source):
        return True
    source_path = resolve_project_path(source)
    return source_path.is_file() and source_path.suffix.lower() in VIDEO_EXTENSIONS


def parse_args() -> argparse.Namespace:
    """Read the command-line options (weights, source, thresholds, and save toggles)."""
    parser = argparse.ArgumentParser(description="Run YOLO license plate detection.")
    parser.add_argument("--weights", required=True, help="Path to trained YOLO weights.")
    parser.add_argument("--source", required=True, help="Image, folder, video, or camera index.")
    parser.add_argument("--imgsz", type=int, default=640, help="Inference image size (the detector is trained at 640).")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold.")
    parser.add_argument("--device", default="0", help="Inference device, for example 0 or cpu.")
    parser.add_argument("--save-txt", action="store_true", help="Save YOLO text detections.")
    parser.add_argument("--save-crops", action="store_true", help="Save detected plate crops.")
    parser.add_argument("--show", action="store_true", help="Display predictions.")
    return parser.parse_args()


def validate_inputs(weights: Path, source: str) -> str | int:
    """Fail early if the weights/source are missing; return a camera index or a file path."""
    if not weights.exists():
        raise FileNotFoundError(f"Weights not found: {weights}")

    if is_camera_source(source):
        return int(source)

    source_path = resolve_project_path(source)
    if not source_path.exists():
        raise FileNotFoundError(f"Source not found: {source_path}")

    return str(source_path)


def row_for_no_detection(source: str, frame_index: int) -> dict[str, object]:
    """One CSV row marking a frame where no plate was found (keeps it in the record)."""
    return {
        "source": source,
        "frame_index": frame_index,
        "x1": "",
        "y1": "",
        "x2": "",
        "y2": "",
        "confidence": 0.0,
        "class_id": -1,
        "class_name": "no_detection",
    }


def main() -> None:
    """Run detection on the source and write annotated media + a detections.csv table."""
    args = parse_args()
    weights = resolve_project_path(args.weights)
    source_for_yolo = validate_inputs(weights, args.source)
    check_cuda_or_fail(args.device)

    started_at = datetime.now()
    output_base = (PROJECT_ROOT / "runs" / "predict").resolve()
    run_dir = create_run_dir(output_base, started_at)
    write_run_metadata(
        run_dir,
        {
            "timestamp": started_at.isoformat(timespec="seconds"),
            "weights": str(weights),
            "source": args.source,
            "imgsz": args.imgsz,
            "conf": args.conf,
            "device": args.device,
            "save_txt": args.save_txt,
            "save_crops": args.save_crops,
        },
    )

    model = YOLO(str(weights))
    results = model.predict(
        source=source_for_yolo,
        imgsz=args.imgsz,
        conf=args.conf,
        device=args.device,
        project=str(output_base),
        name=run_dir.name,
        exist_ok=True,
        save=True,
        save_txt=args.save_txt,
        save_crop=args.save_crops,
        show=args.show,
        stream=True,
        verbose=True,
    )

    rows: list[dict[str, object]] = []
    video_like = source_is_video(args.source)

    for result_index, result in enumerate(results):
        frame_index = result_index if video_like else 0
        result_source = str(result.path)
        boxes = result.boxes

        if boxes is None or len(boxes) == 0:
            rows.append(row_for_no_detection(result_source, frame_index))
            continue

        for box in boxes:
            xyxy = box.xyxy[0].detach().cpu().tolist()
            class_id = int(box.cls[0].detach().cpu().item())
            confidence = float(box.conf[0].detach().cpu().item())
            class_name = model.names.get(class_id, str(class_id))
            rows.append(
                {
                    "source": result_source,
                    "frame_index": frame_index,
                    "x1": float(xyxy[0]),
                    "y1": float(xyxy[1]),
                    "x2": float(xyxy[2]),
                    "y2": float(xyxy[3]),
                    "confidence": confidence,
                    "class_id": class_id,
                    "class_name": class_name,
                }
            )

    if not rows:
        rows.append(row_for_no_detection(str(source_for_yolo), 0))

    csv_path = run_dir / "detections.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    print(f"Prediction outputs saved under: {run_dir}")
    print(f"Detection CSV saved to: {csv_path}")


if __name__ == "__main__":
    main()
