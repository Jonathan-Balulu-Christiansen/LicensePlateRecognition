"""Shared helpers used by the train / predict / live / dataset scripts.

Kept in one place so the individual entry-point scripts stay short and there is
a single source of truth for path resolution, dataset-YAML parsing, device
selection, and run-directory bookkeeping.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
IMAGE_EXTENSIONS = {".bmp", ".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp"}
VIDEO_EXTENSIONS = {".avi", ".m4v", ".mkv", ".mov", ".mp4", ".mpeg", ".mpg", ".wmv"}
SPLITS = ("train", "val", "test")


def resolve_project_path(path_value: str | Path) -> Path:
    """Resolve a path relative to the project root unless it is already absolute."""
    path = Path(path_value)
    if path.is_absolute():
        return path
    return (PROJECT_ROOT / path).resolve()


def load_dataset_yaml(data_path: Path) -> dict[str, Any]:
    """Read the dataset config file (which folders hold the train/val/test images)."""
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset YAML not found: {data_path}")

    with data_path.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file)

    if not isinstance(data, dict):
        raise ValueError(f"Dataset YAML must contain a mapping: {data_path}")

    return data


def resolve_dataset_root(dataset_yaml: Path, config: dict[str, Any]) -> Path:
    """Find the dataset's base folder that the split paths in the config are relative to."""
    dataset_path_value = config.get("path")
    if not dataset_path_value:
        raise ValueError("Dataset YAML must define 'path'.")

    dataset_root = Path(str(dataset_path_value))
    if not dataset_root.is_absolute():
        dataset_root = dataset_yaml.parent / dataset_root

    return dataset_root.resolve()


def resolved_dataset_yaml(dataset_yaml: Path, out_dir: Path) -> Path:
    """Write a copy of the dataset YAML whose `path` is absolute, and return it.

    Our YAML gives `path` relative to the YAML file itself (that is how dataset_check.py
    and split_dataset.py read it). Ultralytics instead resolves a relative `path` against
    its global `datasets_dir` setting, which on the dev machine only happened to land on
    the same folder. Training and validation get this resolved copy, so the project can
    be moved or cloned anywhere.

    Optional `train_extra` / `val_extra` lists (more folders for those splits, e.g. harvested
    video frames) are merged into `train` / `val` here, so the other tools keep reading each
    split as one folder. An extra folder that does not exist (data not built on this machine) is
    skipped with a note. The test split is never extended.
    """
    config = load_dataset_yaml(dataset_yaml)
    root = resolve_dataset_root(dataset_yaml, config)
    config["path"] = root.as_posix()
    for split in ("train", "val"):
        extra = []
        for folder in config.pop(f"{split}_extra", None) or []:
            if (root / folder).is_dir():
                extra.append(folder)
            else:
                print(f"note: {split}_extra folder {folder} not found, skipped")
        if extra:
            config[split] = [config[split], *extra]
    out_dir.mkdir(parents=True, exist_ok=True)
    resolved = out_dir / f"{dataset_yaml.stem}.resolved.yaml"
    resolved.write_text(yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return resolved


def infer_label_dir(dataset_root: Path, image_dir_value: str) -> Path:
    """Map an 'images/<split>' path to its sibling 'labels/<split>' directory."""
    parts = Path(image_dir_value).parts
    if not parts or parts[0] != "images":
        raise ValueError(
            f"Expected split image path to start with 'images': {image_dir_value}"
        )
    return dataset_root.joinpath("labels", *parts[1:]).resolve()


def is_camera_source(source: str) -> bool:
    """True if the source is a webcam index (e.g. "0") rather than a file path."""
    return source.strip().isdigit()


def check_cuda_or_fail(device: str) -> None:
    """Print the selected device and fail early if a GPU was requested but is unavailable."""
    if str(device).strip().lower() == "cpu":
        print("Selected device: CPU")
        return

    if not torch.cuda.is_available():
        raise RuntimeError(
            f"CUDA device '{device}' was requested, but PyTorch does not detect CUDA. "
            "Check the NVIDIA driver, CUDA-compatible PyTorch installation, and active virtual environment."
        )

    print(f"Selected device: {device} ({torch.cuda.get_device_name(0)})")


def create_run_dir(base: Path, started_at: datetime) -> Path:
    """Make a unique timestamped run directory so runs never overwrite each other."""
    stamp = started_at.strftime("%Y%m%d_%H%M%S")
    run_dir = base / stamp
    suffix = 2
    while run_dir.exists():
        run_dir = base / f"{stamp}_{suffix}"
        suffix += 1
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def write_run_metadata(run_dir: Path, metadata: dict[str, object]) -> None:
    """Record what produced this run (weights, source, params) for traceability."""
    (run_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def append_jsonl(path: Path, row: dict[str, object]) -> None:
    """Append one record to a JSON Lines file, creating it (and its folder) if needed.

    One JSON object per line, rather than a JSON array: an array would have to be read,
    parsed and rewritten in full for every appended row, which is the wrong shape for a
    log that grows one entry per event.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(row, ensure_ascii=False) + "\n")
