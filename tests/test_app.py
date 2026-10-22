"""The window, the engine's failure path, and the data tools' safety rules."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pytest

from common import PROJECT_ROOT

VIDEOS = PROJECT_ROOT / "data" / "raw" / "videos"
DETECTOR = PROJECT_ROOT / "weights" / "detector" / "best.pt"
READER = PROJECT_ROOT / "weights" / "reader" / "plate_reader.keras"
HAVE_MODELS = DETECTOR.is_file() and READER.is_file()
# The demo clips and the numbers they should lock are local data, never in git:
# {"<clip file name>": "<number>"} next to the clips.
EXPECTED_PLATES = VIDEOS / "expected_plates.json"


def run_worker(worker, stop_after_lock: bool = False) -> dict[str, list]:
    """Run a VideoWorker in this thread and collect what it emitted."""
    seen: dict[str, list] = {"failed": [], "status": [], "frames": [], "locked": []}
    worker.failed.connect(seen["failed"].append)
    worker.status.connect(seen["status"].append)
    worker.frame_ready.connect(lambda image, info: seen["frames"].append(info))

    def on_lock(event):
        seen["locked"].append(event)
        if stop_after_lock:
            worker.stop()

    worker.plate_locked.connect(on_lock)
    worker.run()  # synchronous: same thread, so every signal is delivered directly
    return seen


# ------------------------------------------------------------------------- engine


def test_missing_detector_is_reported_not_raised(qapp, tmp_path):
    from pipeline import VideoWorker

    seen = run_worker(VideoWorker("0", tmp_path / "missing.pt", READER, log_path=None))
    assert len(seen["failed"]) == 1
    assert "Detector weights not found" in seen["failed"][0]
    assert seen["frames"] == []


@pytest.mark.slow
@pytest.mark.skipif(not HAVE_MODELS, reason="models not present (weights/ is not in git)")
def test_missing_video_is_reported_and_models_are_cached(qapp, tmp_path):
    from pipeline import VideoWorker

    seen = run_worker(VideoWorker(str(tmp_path / "missing.mp4"), DETECTOR, READER, log_path=None))
    assert len(seen["failed"]) == 1
    assert "Video source not found" in seen["failed"][0]


@pytest.mark.slow
@pytest.mark.skipif(not (HAVE_MODELS and EXPECTED_PLATES.is_file()), reason="demo clips not present")
def test_demo_clip_locks_the_expected_number(qapp, tmp_path):
    from pipeline import VideoWorker

    clip, number = next(iter(json.loads(EXPECTED_PLATES.read_text(encoding="utf-8")).items()))
    log = tmp_path / "gate_log.jsonl"
    worker = VideoWorker(str(VIDEOS / clip), DETECTOR, READER, log_path=log)
    seen = run_worker(worker, stop_after_lock=True)
    assert seen["failed"] == []
    assert [event.number for event in seen["locked"]] == [number]
    assert seen["locked"][0].origin.code == "IL"
    stats = seen["frames"][-1].stats
    assert 0 < stats.frame_ms < 1000 and stats.detect_ms > 0
    assert '"detector_confidence"' in log.read_text(encoding="utf-8")


# ------------------------------------------------------------------------- window


def test_window_builds_and_shows_several_plates(qapp):
    from app import MainWindow
    from pipeline import PlateView
    from plate_origin import UNKNOWN, Origin

    window = MainWindow()
    window.show()
    assert window.start_button.text() == "Start"
    if DETECTOR.is_file():
        assert window.detector_combo.value() == str(DETECTOR)

    israel = Origin("IL", "Israel", "plate color · yellow", 0.57)
    window._show_plates([PlateView(7, "12345601", True, 5, 5, 7, (0.9998,) * 8, 0.78, israel, True),
                         PlateView(9, "1234567", False, 2, 5, 3, (0.97,) * 7, 0.66, UNKNOWN, True)])
    first, second = window.plate_items[:2]
    assert first.isVisible() and second.isVisible() and not window.plate_items[2].isVisible()
    assert first.chip.text() == "LOCKED 5/5" and first.track_label.text() == "ID 7"
    assert first.reads_label.text() == "· 7 reads" and first.property("locked") is True
    assert first.origin_label.text() == "Israel · 8 digits"
    assert first.scores_label.text() == "read 0.998 · det 0.78"
    assert second.chip.text() == "READING 2/5"
    assert sum(cell.isVisible() for cell, _, _ in second.digit_cells) == 7
    assert window.plates_count.text() == "2 · 1 locked"
    assert not window.grab().isNull()  # every custom paintEvent runs without error

    window._show_plates([])
    assert not any(item.isVisible() for item in window.plate_items)
    assert window.no_plate_label.isVisible()
    window.close()


def test_a_locked_plate_goes_to_the_top_of_the_passages(qapp):
    from app import MainWindow
    from pipeline import PlateEvent
    from plate_origin import Origin

    window = MainWindow()
    israel = Origin("IL", "Israel", "plate color · yellow", 0.6)
    for number in ("87654302", "12345601"):
        window._on_plate_locked(PlateEvent(datetime(2026, 10, 21, 18, 31), 7, number, israel, 0.78, (0.999,) * 8))
    assert window.passages.rowCount() == 2
    newest = window.passages.item(0, 1)
    assert newest is not None and newest.text() == "123-45-601"
    assert window.logged_value.text() == "2 logged"
    window.close()


@pytest.mark.parametrize(("size", "side", "inline"), [((1100, 720), 360, False), ((1920, 1080), 480, True),
                                                      ((3440, 1400), 500, True)])
def test_layout_adapts_to_the_window_size(qapp, size, side, inline):
    from app import MainWindow

    window = MainWindow()
    window.resize(*size)
    window.show()
    qapp.processEvents()
    assert window.side_panel.width() == side  # ~25% of the width, kept within 360-500 px
    assert window._sparkline_inline is inline  # beside the numbers when there is room, else below
    assert not window.grab().isNull()
    window.close()


def test_fullscreen_toggles_and_escape_leaves_it(qapp):
    from app import MainWindow

    window = MainWindow()
    window.show()
    window._toggle_fullscreen()
    assert window.isFullScreen()
    window._leave_fullscreen()
    assert not window.isFullScreen()
    window.close()


def test_start_without_a_detector_explains_instead_of_starting(qapp):
    from app import MainWindow

    window = MainWindow()
    window.detector_combo.clear_choices()
    window._start()
    assert window.worker is None
    assert "detector" in window.status_label.text()
    window.close()


# -------------------------------------------------------------------------- tools


def test_downsample_keeps_the_duration_at_any_ratio():
    from downsample_fps import keep_frame

    def kept(src_fps: float, target_fps: float, frames: int) -> int:
        count = 0
        for index in range(frames):
            if keep_frame(index, count, src_fps, target_fps):
                count += 1
        return count

    assert kept(120, 30, 1200) == 300  # 10 s stays 10 s
    assert kept(60, 24, 600) == 240  # the rounded stride used to give 300 here (12.5 s)
    assert kept(30, 30, 90) == 90


def test_downsample_refuses_to_overwrite_its_source(tmp_path):
    from downsample_fps import output_path

    with pytest.raises(ValueError):
        output_path(tmp_path / "test1_30fps.mp4", 30)
    assert output_path(tmp_path / "test1_original.mp4", 30).name == "test1_30fps.mp4"


def test_label_check_reports_nan_boxes(tmp_path):
    from dataset_check import parse_label_file

    label = tmp_path / "a.txt"
    label.write_text("0 nan 0.5 0.2 0.1\n0 0.5 0.5 0.2 0.1\n", encoding="utf-8")
    issues: list = []
    boxes = parse_label_file(label, "train", Path("a.jpg"), issues)
    assert len(boxes) == 1
    assert [issue.severity for issue in issues] == ["error"]


def test_dataset_yaml_is_resolved_against_the_yaml_not_the_cwd(tmp_path):
    # Ultralytics resolves a relative `path` against its global datasets_dir; training gets a
    # copy with the path made absolute relative to the YAML file itself.
    import yaml

    from common import resolved_dataset_yaml

    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    source = config_dir / "dataset.yaml"
    source.write_text("path: ../data/detector\ntrain: images/train\nnames:\n  0: license_plate\n", encoding="utf-8")
    resolved = yaml.safe_load(resolved_dataset_yaml(source, tmp_path / "runs").read_text(encoding="utf-8"))
    assert resolved["path"] == (tmp_path / "data" / "detector").resolve().as_posix()
    assert resolved["train"] == "images/train" and resolved["names"] == {0: "license_plate"}
