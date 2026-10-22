"""Extract still frames from one video for dataset creation.

One-time data-prep helper (not part of the live demo). Saves timestamped JPGs at
a fixed interval plus a small metadata.csv for provenance.

Example:
  python tools/extract_frames_from_video.py \
      --video "data/raw/videos/night1.mp4" \
      --start 0 --end 10 --interval 0.5
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path

import cv2

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_ROOT = SCRIPT_DIR / "ready images"
DATASET_WARNING = (
    "Frames from the same video are highly correlated. Keep whole video sources\n"
    "separated when building train/val/test splits (do not randomly mix them)."
)


def parse_time(value: str, field_name: str) -> float:
    """Parse a time given as seconds, mm:ss, or hh:mm:ss into seconds."""
    parts = value.strip().split(":")
    if len(parts) > 3:
        raise ValueError(f"{field_name} must be seconds, mm:ss, or hh:mm:ss.")
    try:
        numbers = [float(part) for part in parts]
    except ValueError as exc:
        raise ValueError(f"{field_name} must be seconds, mm:ss, or hh:mm:ss.") from exc
    if any(number < 0 for number in numbers):
        raise ValueError(f"{field_name} cannot be negative.")
    seconds = 0.0
    for number in numbers:  # left to right: h, m, s (or just s / m,s)
        seconds = seconds * 60 + number
    return seconds


def parse_args() -> argparse.Namespace:
    """Read the command-line options (video path, start/end time, interval)."""
    parser = argparse.ArgumentParser(
        description="Extract timestamped JPG frames from one video."
    )
    parser.add_argument("--video", type=Path, required=True, help="Path to a video file.")
    parser.add_argument("--start", required=True, help="Start time: seconds, mm:ss, or hh:mm:ss.")
    parser.add_argument("--interval", type=float, required=True, help="Seconds between frames.")
    parser.add_argument("--end", help="End time (seconds, mm:ss, hh:mm:ss); default: end of video.")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT_ROOT, help="Output root folder.")
    parser.add_argument("--jpg-quality", type=int, default=95, help="JPEG quality, 0-100.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing JPGs.")
    return parser.parse_args()


def extract_frames(
    video_path: Path,
    output_root: Path,
    start: float,
    end: float | None,
    interval: float,
    jpg_quality: int,
    overwrite: bool,
) -> tuple[int, Path]:
    """Step through the video at a fixed interval and save each chosen frame as a JPG."""
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"OpenCV could not open the video: {video_path}")

    try:
        fps = capture.get(cv2.CAP_PROP_FPS)
        if not math.isfinite(fps) or fps <= 0:
            raise RuntimeError(f"Video has an invalid FPS value: {fps}")
        total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps if total_frames > 0 else None

        output_folder = output_root / video_path.stem
        output_folder.mkdir(parents=True, exist_ok=True)
        print(f"Video: {video_path.name}  ({fps:.2f} FPS, {total_frames} frames)")
        print(f"Output folder: {output_folder}")

        rows: list[dict[str, object]] = []
        extracted = 0
        target_number = 0

        while True:
            target_time = start + target_number * interval
            if end is not None and target_time > end:
                break
            if duration is not None and target_time >= duration:
                break

            capture.set(cv2.CAP_PROP_POS_MSEC, target_time * 1000.0)
            read_ok, frame = capture.read()
            if not read_ok or frame is None:
                break

            frame_index = int(round(target_time * fps))
            filename = f"{video_path.stem}_t{target_time:010.3f}_f{frame_index:06d}.jpg"
            image_path = output_folder / filename
            if image_path.exists() and not overwrite:
                target_number += 1
                continue

            if not cv2.imwrite(str(image_path), frame, [cv2.IMWRITE_JPEG_QUALITY, jpg_quality]):
                raise RuntimeError(f"OpenCV could not save image: {image_path}")
            rows.append(
                {
                    "video_path": str(video_path),
                    "output_image": str(image_path),
                    "timestamp_sec": f"{target_time:.3f}",
                    "frame_index": frame_index,
                }
            )
            extracted += 1
            target_number += 1
            if extracted % 25 == 0:
                print(f"Saved {extracted} frames...")

        metadata_path = output_folder / "metadata.csv"
        with metadata_path.open("w", newline="", encoding="utf-8") as csv_file:
            writer = csv.DictWriter(
                csv_file, fieldnames=["video_path", "output_image", "timestamp_sec", "frame_index"]
            )
            writer.writeheader()
            writer.writerows(rows)
    finally:
        capture.release()

    return extracted, output_folder


def main() -> int:
    """Parse the options, extract the frames, and print where they were saved."""
    args = parse_args()
    try:
        start = parse_time(args.start, "Start time")
        end = parse_time(args.end, "End time") if args.end else None
        if end is not None and end <= start:
            raise ValueError("End time must be greater than start time.")
        if not math.isfinite(args.interval) or args.interval <= 0:
            raise ValueError("Interval must be greater than 0.")

        video_path = args.video.expanduser().resolve()
        if not video_path.is_file():
            raise ValueError(f"Video not found: {video_path}")

        extracted, output_folder = extract_frames(
            video_path=video_path,
            output_root=args.out.expanduser().resolve(),
            start=start,
            end=end,
            interval=args.interval,
            jpg_quality=args.jpg_quality,
            overwrite=args.overwrite,
        )
    except (ValueError, RuntimeError, OSError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(f"\nFrames extracted: {extracted}")
    print(f"Results saved in: {output_folder}")
    print(f"\n{DATASET_WARNING}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
