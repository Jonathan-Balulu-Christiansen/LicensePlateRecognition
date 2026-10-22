"""The real-time engine behind the Visus window: capture -> detect + track -> read -> vote.

Everything heavy runs in a worker QThread, so the window stays responsive. The worker
does not draw on the frame: it sends each frame together with a FrameInfo (the tracked
boxes, per-stage timings, the plate in focus), and the window paints the overlay and the
side panel itself. A plate whose number locks is sent once more as a PlateEvent, which
is also what gets appended to the gate log.
"""

from __future__ import annotations

import math
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import torch
from PySide6.QtCore import QThread, Signal
from PySide6.QtGui import QImage
from ultralytics import YOLO

from common import PROJECT_ROOT, append_jsonl, is_camera_source, resolve_project_path
from plate_origin import UNKNOWN, Origin, origin_from_samples, plate_color
from plate_reader import READER_PATH, PlateVoter, load_reader, read_plates

# The promoted models. `runs/` is throwaway (see .gitignore) and gets pruned, so the models
# the app uses by default live in `weights/`: the YOLO26m detector and the CNN+CTC reader.
# Re-promote a detector with a copy of runs/train/<run>/weights/best.pt, a reader with
# `python src/train_reader.py --promote`.
DEFAULT_DETECTOR = PROJECT_ROOT / "weights" / "detector" / "best.pt"
DEFAULT_READER = READER_PATH
DEFAULT_DEVICE = "0" if torch.cuda.is_available() else "cpu"
# Cumulative log of passing vehicles: one line per plate, appended across runs.
GATE_LOG = PROJECT_ROOT / "reports" / "gate_log.jsonl"
READ_EVERY_N_FRAMES = 4  # read each unlocked plate on every 4th frame (1 = every frame)
MIN_READ_WIDTH = 90  # px; narrower plate crops hold too few pixels per digit to read
INFER_IMGSZ = 640  # inference size; measured to match 960 in quality here, and ~2x faster
DETECT_EVERY_N_FRAMES = 1  # run YOLO every Nth frame, reusing the previous boxes in between.
# 1 on this desktop (RTX 4070 Ti). Not because 2 is slower here — it is faster
# (best-of-3 over test1_30fps.mp4: 1 -> 1.97x of real time, 2 -> 2.84x) — but because
# 1 already runs faster than real time, and video files are paced to the clip's own fps
# (see _loop()), so more speed would only buy idle time at the cost of the one-frame box
# trail. Set it to 2 on a weak GPU, where the problem is the opposite:
# measured on the MX450, 1 -> 16.2fps (0.54x of real time), 2 -> 23.6fps (0.79x).
# Only a 1.46x gain there, not 2x: skipped frames still pay a fixed ~23ms of decode +
# QImage convert + emit, which on that machine dominates the ~39ms inference.
# The cost of 2 is that boxes trail one frame (~12px at 30fps) on skipped frames —
# cosmetic only, reading uses the current box. Speeding this up further means attacking the
# display path, not YOLO (raising N caps out at ~43fps).
# The box drawn is the one model.track returns: ByteTrack's Kalman-filter estimate,
# updated with that frame's detection — not YOLO's raw box (on test1_30fps.mp4 the two
# differ by ~1px median, 14px at most). A moving-average smoother on top of it was
# measured and removed: the tracked box's jitter is already sub-pixel (0.04px median at
# 120fps, 0.18px at 30fps) and smoothing added ~3x more lag than it removed.
TIMING_WINDOW = 30  # frames averaged for every number on screen (~1s at 30fps)
FORGET_AFTER_FRAMES = 300  # drop a track's bookkeeping once it has been gone this long


@dataclass(frozen=True)
class TrackView:
    """One tracked plate as the window draws it over the video."""

    box: tuple[int, int, int, int]
    track_id: int | None
    detector_conf: float
    locked: bool
    label: str  # e.g. "ID 7 · LOCKED" or "ID 9 · reading 2/5"


@dataclass(frozen=True)
class PlateView:
    """The plate shown in the side panel: locked, or the leading candidate while reading."""

    track_id: int
    number: str
    locked: bool
    votes: int
    min_votes: int
    reads: int
    digit_confs: tuple[float, ...]
    detector_conf: float
    origin: Origin
    visible: bool


@dataclass(frozen=True)
class FrameStats:
    """Per-stage timings, each averaged over the last TIMING_WINDOW frames.

    `frame_ms` is the processing time of one frame — detect + read + image conversion. It
    leaves out `capture_ms` (for a camera that is mostly waiting for the next frame) and the
    pacing sleep that holds a video file to its own fps, so it measures the pipeline, not
    the source. `fps` is the rate frames actually reach the window.
    """

    capture_ms: float
    detect_ms: float
    read_ms: float
    convert_ms: float
    frame_ms: float
    last_frame_ms: float  # this frame alone, for the sparkline
    fps: float
    source_fps: float
    width: int
    height: int


@dataclass(frozen=True)
class FrameInfo:
    """Everything the window needs about one frame besides its pixels."""

    tracks: list[TrackView]
    focus: PlateView | None
    stats: FrameStats


@dataclass(frozen=True)
class PlateEvent:
    """One vehicle whose number locked: a row in the passages table and in the gate log."""

    when: datetime
    track_id: int
    number: str
    origin: Origin
    detector_conf: float
    digit_confs: tuple[float, ...]

    @property
    def read_score(self) -> float:
        """Product of the per-digit confidences: the reader's score for the whole number."""
        return math.prod(self.digit_confs)


@dataclass
class _TrackState:
    """What the worker remembers about one track between frames."""

    detector_conf: float = 0.0
    last_seen: int = 0
    colors: list[tuple[float, float]] = field(default_factory=list)  # plate_color() per read
    confs_by_text: dict[str, tuple[float, ...]] = field(default_factory=dict)
    lock_confs: tuple[float, ...] = ()
    origin: Origin = UNKNOWN


def device_label(device: str = DEFAULT_DEVICE) -> str:
    """'CUDA · RTX 4070 Ti' or 'CPU', for the window."""
    if device != "cpu" and torch.cuda.is_available():
        return f"CUDA · {torch.cuda.get_device_name(0).replace('NVIDIA ', '').replace('GeForce ', '')}"
    return "CPU"


def find_detector_models() -> list[Path]:
    """Detector weights the app can offer: the promoted ones first, then training runs."""
    promoted = sorted((PROJECT_ROOT / "weights" / "detector").glob("*.pt"))
    runs = sorted((PROJECT_ROOT / "runs" / "train").glob("*/weights/best.pt"))
    return promoted + runs


def find_reader_models() -> list[Path]:
    """Plate readers the app can offer: the promoted one first, then training runs."""
    promoted = sorted((PROJECT_ROOT / "weights" / "reader").glob("*.keras"))
    runs = sorted((PROJECT_ROOT / "runs" / "reader").glob("*/plate_reader.keras"))
    return promoted + runs


def open_capture(source: str) -> cv2.VideoCapture:
    """Open a camera (by index) or a video file for frame-by-frame reading."""
    if is_camera_source(source):
        capture_source: int | str = int(source)
    else:
        source_path = resolve_project_path(source)
        if not source_path.exists():
            raise FileNotFoundError(f"Video source not found: {source_path}")
        capture_source = str(source_path)
    capture = cv2.VideoCapture(capture_source)
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video source: {source}")
    return capture


def update_tracks(result) -> list[tuple[tuple[int, int, int, int], float, int | None]]:
    """One frame's tracked boxes as (box, confidence, track_id).

    The boxes are the tracker's: after model.track, each one is ByteTrack's Kalman-filter
    estimate updated with that frame's detection, not YOLO's raw box.
    """
    detections = []
    for box in result.boxes or []:
        confidence = float(box.conf[0].detach().cpu().item())
        xyxy = box.xyxy[0].detach().cpu().tolist()
        track_id = int(box.id[0].item()) if box.id is not None else None
        # int(round()) on each value: consumers expect exactly four ints, and round() on a
        # numpy scalar would return a numpy scalar.
        x1, y1, x2, y2 = (int(round(value)) for value in xyxy)
        detections.append(((x1, y1, x2, y2), confidence, track_id))
    return detections


class VideoWorker(QThread):
    """Runs capture + YOLO tracking + plate reading off the UI thread and emits ready frames.

    The window learns the worker is done from QThread's own `finished` signal, which fires
    after run() returns — whether it ended normally, was stopped, or failed.
    """

    frame_ready = Signal(QImage, object)  # (frame, FrameInfo)
    plate_locked = Signal(object)  # PlateEvent
    status = Signal(str)  # progress while starting ("Loading detector…")
    failed = Signal(str)  # a setup or runtime error, already worded for the user

    def __init__(
        self,
        source: str,
        detector: Path = DEFAULT_DETECTOR,
        reader: Path = DEFAULT_READER,
        min_votes: int = 5,
        *,
        imgsz: int = INFER_IMGSZ,
        conf: float = 0.25,
        device: str = DEFAULT_DEVICE,
        log_path: Path | None = GATE_LOG,
    ) -> None:
        """Store the settings for this run; `log_path=None` turns the gate log off."""
        super().__init__()
        self._source = source
        self._detector = Path(detector)
        self._reader = Path(reader)
        self._min_votes = min_votes
        self._imgsz = imgsz
        self._conf = conf
        self._device = device
        self._log_path = log_path
        self._running = True

    def stop(self) -> None:
        """Ask the loop to finish after the current frame."""
        self._running = False

    def run(self) -> None:
        """Load both models, open the source, and run the frame loop until stopped.

        Every failure — a missing model, a camera that will not open, an error mid-run —
        is reported through `failed` instead of escaping the thread, and the capture is
        always released.
        """
        capture = None
        try:
            if not self._detector.is_file():
                raise FileNotFoundError(f"Detector weights not found: {self._detector}")
            self.status.emit("Loading detector…")
            model = YOLO(str(self._detector))
            self.status.emit("Loading plate reader…")
            reader = load_reader(self._reader)
            if not self._running:
                return
            self.status.emit("Opening source…")
            capture = open_capture(self._source)
            self._loop(model, reader, capture)
        except Exception as exc:  # noqa: BLE001 - every failure must reach the window
            self.failed.emit(f"{type(exc).__name__}: {exc}")
        finally:
            if capture is not None:
                capture.release()

    def _loop(self, model: YOLO, reader, capture: cv2.VideoCapture) -> None:
        """The per-frame pipeline: capture -> detect + track -> read until locked -> vote -> emit."""
        voter = PlateVoter(min_votes=self._min_votes)
        # Video files are paced to their own fps: without that, a fast GPU plays them in
        # fast-forward (~2x on the 4070 Ti). Cameras are never paced — read() already
        # waits for the camera's next frame.
        source_fps = capture.get(cv2.CAP_PROP_FPS)
        frame_period = 0.0 if is_camera_source(self._source) or source_fps <= 0 else 1.0 / source_fps
        next_frame_at = time.perf_counter()
        last_shown: float | None = None
        frame_index = 0
        last_detections = []  # reused on skipped frames so boxes keep drawing
        states: dict[int, _TrackState] = {}
        focus_id: int | None = None
        # Average every readout over ~1s of frames. A single frame swings wildly (a detect
        # frame costs ~65ms, a skipped one ~5ms), which would hide the real numbers.
        # Frame 0 is never averaged: it pays the lazy CUDA/cuDNN init (~2.8s vs ~51ms
        # steady on the MX450) and would drag the first 30 readings down ~2.4x.
        window = {name: deque(maxlen=TIMING_WINDOW) for name in ("capture", "detect", "read", "convert", "shown")}

        def mean(name: str) -> float:
            values = window[name]
            return sum(values) / len(values) if values else 0.0

        self.status.emit("Running")

        while self._running:
            t_start = time.perf_counter()
            ok, frame = capture.read()
            if not ok:
                break
            t_read = time.perf_counter()

            run_detect = frame_index % DETECT_EVERY_N_FRAMES == 0
            if run_detect:
                result = model.track(
                    source=frame,
                    imgsz=self._imgsz,
                    conf=self._conf,
                    device=self._device,
                    tracker="bytetrack.yaml",
                    persist=True,
                    verbose=False,
                )[0]
                last_detections = update_tracks(result)
            t_detect = time.perf_counter()

            # Pass 1: bookkeeping, and which plates to read this frame — only on detect frames,
            # until locked, every Nth frame, and only if big enough.
            to_read: list[tuple[int, float, np.ndarray]] = []
            for box, confidence, track_id in last_detections:
                if track_id is None:
                    continue
                state = states.setdefault(track_id, _TrackState())
                state.detector_conf = confidence
                state.last_seen = frame_index
                x1, y1, x2, y2 = box
                if (
                    run_detect
                    and voter.locked(track_id) is None
                    and frame_index % READ_EVERY_N_FRAMES == 0
                    and x2 - x1 >= MIN_READ_WIDTH
                ):
                    to_read.append((track_id, confidence, frame[y1:y2, x1:x2]))

            # Pass 2: every plate of the frame in ONE forward pass of the reader, then vote.
            reads = read_plates([crop for _, _, crop in to_read], reader) if to_read else []
            for (track_id, confidence, crop), (text, confs) in zip(to_read, reads):
                state = states[track_id]
                state.colors.append(plate_color(crop))
                if text:
                    state.confs_by_text[text] = confs
                number = voter.update(track_id, text, confs)
                # `number` goes None -> locked exactly once per vehicle, which is the moment
                # to report it. The read that triggers the lock is always the locked string
                # (the vote that reaches the threshold is what wins it), so `confs` describes it.
                if number:
                    state.lock_confs = confs
                    state.origin = origin_from_samples(state.colors)
                    focus_id = track_id
                    self._report(PlateEvent(datetime.now(), track_id, number, state.origin, confidence, confs))

            tracks: list[TrackView] = []
            for box, confidence, track_id in last_detections:
                if track_id is None:
                    tracks.append(TrackView(box, None, confidence, False, f"plate {confidence:.2f}"))
                elif voter.locked(track_id):
                    tracks.append(TrackView(box, track_id, confidence, True, f"ID {track_id} · LOCKED"))
                else:
                    _, votes, _ = voter.tally(track_id)
                    tracks.append(TrackView(box, track_id, confidence, False,
                                            f"ID {track_id} · reading {votes}/{self._min_votes}"))
            t_read_plates = time.perf_counter()

            image = self._to_qimage(frame)
            t_convert = time.perf_counter()

            focus_id = self._choose_focus(focus_id, tracks, voter)
            focus = self._plate_view(focus_id, states, voter, tracks) if focus_id is not None else None
            for stale in [tid for tid, s in states.items()
                          if frame_index - s.last_seen > FORGET_AFTER_FRAMES and tid != focus_id]:
                del states[stale]

            if frame_period:  # a video file: never show it faster than real time
                wait = next_frame_at - time.perf_counter()
                if wait > 0:
                    time.sleep(wait)
                elif -wait > frame_period:
                    # Fell behind (e.g. the slow first frame): restart the schedule
                    # from now instead of racing through frames to catch up.
                    next_frame_at = time.perf_counter()
                next_frame_at += frame_period

            now = time.perf_counter()
            last_frame_ms = (t_convert - t_read) * 1000
            if frame_index > 0:
                window["capture"].append((t_read - t_start) * 1000)
                window["detect"].append((t_detect - t_read) * 1000)
                window["read"].append((t_read_plates - t_detect) * 1000)
                window["convert"].append((t_convert - t_read_plates) * 1000)
                if last_shown is not None:
                    window["shown"].append(now - last_shown)
            last_shown = now

            shown = window["shown"]
            stats = FrameStats(
                capture_ms=mean("capture"),
                detect_ms=mean("detect"),
                read_ms=mean("read"),
                convert_ms=mean("convert"),
                frame_ms=mean("detect") + mean("read") + mean("convert"),
                last_frame_ms=last_frame_ms,
                fps=len(shown) / sum(shown) if shown else 0.0,
                source_fps=source_fps if source_fps > 0 else 0.0,
                width=frame.shape[1],
                height=frame.shape[0],
            )
            self.frame_ready.emit(image, FrameInfo(tracks, focus, stats))
            frame_index += 1

    @staticmethod
    def _choose_focus(focus_id: int | None, tracks: list[TrackView], voter: PlateVoter) -> int | None:
        """Keep the side panel on one plate while it is on screen (it does not jump to every
        new track); when it leaves, move to the visible plate furthest along — a locked one
        first, then the one with the most agreeing reads."""
        visible = [t.track_id for t in tracks if t.track_id is not None]
        if focus_id in visible or not visible:
            return focus_id
        return max(visible, key=lambda tid: (voter.locked(tid) is not None, voter.tally(tid)[1]))

    def _plate_view(
        self, track_id: int, states: dict[int, _TrackState], voter: PlateVoter, tracks: list[TrackView]
    ) -> PlateView | None:
        """Snapshot of the focused plate for the side panel."""
        state = states.get(track_id)
        if state is None:
            return None
        visible = any(t.track_id == track_id for t in tracks)
        number = voter.locked(track_id)
        best, votes, reads = voter.tally(track_id)
        if number:
            return PlateView(track_id, number, True, votes, self._min_votes, reads, state.lock_confs,
                             state.detector_conf, state.origin, visible)
        return PlateView(track_id, best, False, votes, self._min_votes, reads, state.confs_by_text.get(best, ()),
                         state.detector_conf, origin_from_samples(state.colors), visible)

    def _report(self, event: PlateEvent) -> None:
        """Send a locked plate to the window and append it to the gate log.

        `detector_confidence` is YOLO's confidence that the box is a plate. `ocr_digit_mean`
        says how clean the digits looked on average; `ocr_plate_confidence` is the product
        of the per-digit confidences — the reader's score for the whole number, which a
        single shaky digit pulls down where a mean would hide it. It is a model score, not
        a measured probability that the number is right.
        """
        self.plate_locked.emit(event)
        if self._log_path is None:
            return
        append_jsonl(
            self._log_path,
            {
                "date": event.when.strftime("%d/%m/%Y"),
                "time": event.when.strftime("%H:%M:%S"),
                "plate": event.number,
                "origin": event.origin.code,
                "detector_confidence": round(event.detector_conf, 4),
                "ocr_digit_mean": round(sum(event.digit_confs) / len(event.digit_confs), 4),
                "ocr_plate_confidence": round(event.read_score, 4),
            },
        )

    @staticmethod
    def _to_qimage(frame) -> QImage:
        """Wrap a BGR OpenCV frame as a QImage and copy it (safe across threads).

        Format_BGR888 reads OpenCV's byte order directly, which saves the BGR->RGB
        conversion the window used to pay on every frame.
        """
        height, width, _ = frame.shape
        return QImage(frame.data, width, height, 3 * width, QImage.Format.Format_BGR888).copy()
