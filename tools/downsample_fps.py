"""One-time helper: write lower-fps copies of a video by dropping frames.

The `test*` day clips were shot at 120 fps, which is wasted on plate detection (a plate
barely moves in 1/120 s) and makes the live app 4x slower than a 30 fps clip. This keeps
the source frame nearest each output time step (every 4th frame for 120 -> 30) and
rewrites a new mp4 at the target fps, so the content and total duration are unchanged but
there are far fewer frames to process.

Originals are never touched, and a run that would write over its own source is refused.
Run:

    python tools/downsample_fps.py --source data/raw/videos/test1_original.mp4 --fps 30
    python tools/downsample_fps.py --source data/raw/videos --pattern original --fps 30
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from common import VIDEO_EXTENSIONS, resolve_project_path  # noqa: E402


def short_name(source: Path) -> str:
    """`test1_original.mp4` -> `test1` (first underscore-token), for clear output names."""
    return source.stem.split("_")[0]


def keep_frame(index: int, kept: int, src_fps: float, target_fps: float) -> bool:
    """Keep source frame `index` if the output (with `kept` frames so far) is due a frame.

    Frames are picked by time, not by a fixed stride: output frame k shows the moment
    k / target_fps, so the copy lasts exactly as long as the source at any fps ratio.
    A rounded stride only does that when the ratio is a whole number (120 -> 30 keeps
    every 4th frame either way); for 60 -> 24 it kept every 2nd frame, i.e. 30 frames per
    second written at 24 fps, and one second of video became 1.25 s.
    """
    return index * target_fps >= kept * src_fps


def output_path(source: Path, target_fps: int) -> Path:
    """`<folder>/<short>_<fps>fps.mp4`, refusing to name the source itself.

    `test1_30fps.mp4` downsampled to 30 fps would otherwise be written over itself.
    """
    out_path = source.with_name(f"{short_name(source)}_{target_fps}fps.mp4")
    if out_path.resolve() == source.resolve():
        raise ValueError(f"Output would overwrite the source: {source.name}. Downsample the original clip instead.")
    return out_path


def downsample(source: Path, target_fps: int) -> Path:
    """Write `<folder>/<short>_<fps>fps.mp4` at the target fps. Returns the new path."""
    out_path = output_path(source, target_fps)
    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {source}")

    src_fps = capture.get(cv2.CAP_PROP_FPS) or float(target_fps)
    if target_fps > src_fps:
        capture.release()
        raise ValueError(f"{source.name} is {src_fps:.0f} fps; cannot raise it to {target_fps} fps by dropping frames.")
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # VideoWriter.fourcc, not the cv2.VideoWriter_fourcc alias: the alias is missing from
    # cv2's type stubs, so editors flag it as an unknown attribute. Identical call.
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter.fourcc(*"mp4v"),
                             float(target_fps), (width, height))

    read = kept = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if keep_frame(read, kept, src_fps, target_fps):
                writer.write(frame)
                kept += 1
            read += 1
    finally:
        capture.release()
        writer.release()

    print(f"  {source.name}: {src_fps:.0f}fps x{read} -> {target_fps}fps x{kept}  =>  {out_path.name}")
    return out_path


def iter_sources(source: Path, pattern: str) -> list[Path]:
    """A single video, or the video files in a folder whose name contains `pattern`."""
    if source.is_dir():
        return sorted(p for p in source.iterdir()
                      if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS
                      and pattern.lower() in p.name.lower())
    return [source]


def parse_args() -> argparse.Namespace:
    """Read the command-line options for the downsampler."""
    parser = argparse.ArgumentParser(description="Write lower-fps copies of a video (frame drop).")
    parser.add_argument("--source", required=True, help="Video file or a folder of videos.")
    parser.add_argument("--fps", type=int, nargs="+", required=True, help="Target fps value(s).")
    parser.add_argument("--pattern", default="", help="If --source is a folder, only files whose "
                        "name contains this substring (e.g. 'test').")
    return parser.parse_args()


def main() -> None:
    """Create the requested lower-fps copies next to each source video."""
    args = parse_args()
    source = resolve_project_path(args.source)
    videos = iter_sources(source, args.pattern)
    if not videos:
        raise FileNotFoundError(f"No matching video found at: {source}")

    print(f"Downsampling {len(videos)} clip(s) to fps {args.fps}:")
    for video in videos:
        for target_fps in args.fps:
            downsample(video, target_fps)


if __name__ == "__main__":
    main()
