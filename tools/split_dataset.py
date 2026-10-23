"""Deterministically re-split the YOLO plate dataset into train/val/test.

A pristine pool (``data/detector/pool/``) is the single source of truth: it
holds every image+label pair and is never modified by a re-split. Each split
folder holds one subfolder per source; this tool owns ``photos/``
(``images/{train,val,test}/photos`` + ``labels/{train,val,test}/photos``), a
disposable, derived copy -- on every run it is cleared and re-populated from the
pool. Other sources (``video/``, written by tools/export_harvest.py) are left
alone. This means re-splitting is always safe and never destroys data.

The assignment is reproducible and idempotent: pairs are sorted by stem (so the
result is independent of where a file currently lives) and shuffled with a fixed
seed, so re-running with the same seed reproduces the same split.

A manifest CSV is written as the reproducibility record. Use --dry-run to
preview the planned split without touching any files.
"""

from __future__ import annotations

import argparse
import random
import shutil
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

# Run as `python tools/split_dataset.py`; make the shared src/ helpers importable.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from common import (
    IMAGE_EXTENSIONS,
    SPLITS,
    infer_label_dir,
    load_dataset_yaml,
    resolve_dataset_root,
    resolve_project_path,
)

SOURCE = "photos"  # the subfolder of each split that this tool owns


@dataclass
class Pair:
    """One image and its matching label file, kept together as a unit."""

    stem: str
    image: Path
    label: Path


def split_dirs(dataset_root: Path, config: dict[str, Any], split: str) -> tuple[Path, Path]:
    """Return the (images, labels) folders for a split, from the config."""
    image_dir_value = config.get(split)
    if not image_dir_value:
        raise ValueError(f"Dataset YAML must define '{split}'.")
    image_dir = (dataset_root / str(image_dir_value)).resolve()
    label_dir = infer_label_dir(dataset_root, str(image_dir_value))
    return image_dir, label_dir


def photo_dirs(dataset_root: Path, config: dict[str, Any], split: str) -> tuple[Path, Path]:
    """Return the (images, labels) folders this tool owns inside a split: the photos/ subfolder."""
    image_dir, label_dir = split_dirs(dataset_root, config, split)
    return image_dir / SOURCE, label_dir / SOURCE


def pool_dirs(dataset_root: Path) -> tuple[Path, Path]:
    """Return the (images, labels) folders of the pristine 'pool' (the untouched master copy)."""
    return (dataset_root / "pool" / "images", dataset_root / "pool" / "labels")


def pairs_in(image_dir: Path, label_dir: Path) -> tuple[list[Pair], list[str]]:
    """Pair every image with a matching .txt label; collect orphans separately."""
    pairs: list[Pair] = []
    orphans: list[str] = []
    if not image_dir.exists():
        return pairs, orphans

    images = sorted(
        path
        for path in image_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    image_stems = {image.stem for image in images}
    for image in images:
        label = label_dir / f"{image.stem}.txt"
        if label.exists():
            pairs.append(Pair(image.stem, image, label))
        else:
            orphans.append(f"image without label: {image}")

    if label_dir.exists():
        for label in label_dir.iterdir():
            if label.is_file() and label.suffix.lower() == ".txt" and label.stem not in image_stems:
                orphans.append(f"label without image: {label}")

    return pairs, orphans


def gather_split_pairs(dataset_root: Path, config: dict[str, Any]) -> tuple[list[Pair], list[str]]:
    """Pool all pairs currently living in the working split folders."""
    by_stem: dict[str, Pair] = {}
    orphans: list[str] = []
    for split in SPLITS:
        image_dir, label_dir = photo_dirs(dataset_root, config, split)
        pairs, split_orphans = pairs_in(image_dir, label_dir)
        orphans.extend(split_orphans)
        for pair in pairs:
            if pair.stem in by_stem:
                orphans.append(f"duplicate stem across splits: {pair.image}")
                continue
            by_stem[pair.stem] = pair
    return [by_stem[stem] for stem in sorted(by_stem)], orphans


def ensure_pool(
    dataset_root: Path, config: dict[str, Any], dry_run: bool
) -> tuple[list[Pair], list[str], str]:
    """Return the authoritative pool pairs, building the pool once if absent."""
    pool_image_dir, pool_label_dir = pool_dirs(dataset_root)
    existing, _ = pairs_in(pool_image_dir, pool_label_dir)
    if existing:
        return sorted(existing, key=lambda item: item.stem), [], "exists"

    # First run: capture the current working-split files as the pristine pool.
    source_pairs, orphans = gather_split_pairs(dataset_root, config)
    if dry_run:
        return source_pairs, orphans, "would-build"

    pool_image_dir.mkdir(parents=True, exist_ok=True)
    pool_label_dir.mkdir(parents=True, exist_ok=True)
    pool_pairs: list[Pair] = []
    for pair in source_pairs:
        target_image = pool_image_dir / pair.image.name
        target_label = pool_label_dir / pair.label.name
        shutil.copy2(pair.image, target_image)
        shutil.copy2(pair.label, target_label)
        pool_pairs.append(Pair(pair.stem, target_image, target_label))
    return sorted(pool_pairs, key=lambda item: item.stem), orphans, "built"


def assign_splits(
    pairs: list[Pair], seed: int, val_frac: float, test_frac: float
) -> dict[str, str]:
    """Return stem -> target split. Sorted-then-seeded-shuffle = reproducible."""
    if val_frac < 0 or test_frac < 0 or (val_frac + test_frac) >= 1.0:
        raise ValueError("val_frac and test_frac must be >= 0 and sum to < 1.")

    order = sorted(pairs, key=lambda item: item.stem)
    random.Random(seed).shuffle(order)

    total = len(order)
    n_val = int(round(total * val_frac))
    n_test = int(round(total * test_frac))
    n_train = total - n_val - n_test

    assignment: dict[str, str] = {}
    for index, pair in enumerate(order):
        if index < n_train:
            assignment[pair.stem] = "train"
        elif index < n_train + n_val:
            assignment[pair.stem] = "val"
        else:
            assignment[pair.stem] = "test"
    return assignment


def clear_working_splits(dataset_root: Path, config: dict[str, Any]) -> None:
    """Remove image/label files from the derived photos/ split dirs (never the pool)."""
    for split in SPLITS:
        image_dir, label_dir = photo_dirs(dataset_root, config, split)
        for directory, suffixes in ((image_dir, IMAGE_EXTENSIONS), (label_dir, {".txt"})):
            if not directory.exists():
                continue
            for path in directory.iterdir():
                if path.is_file() and path.suffix.lower() in suffixes:
                    path.unlink()


def populate_splits(
    pool_pairs: list[Pair],
    assignment: dict[str, str],
    dataset_root: Path,
    config: dict[str, Any],
) -> int:
    """Copy each pool pair into its assigned split folder."""
    target_dirs = {split: photo_dirs(dataset_root, config, split) for split in SPLITS}
    for image_dir, label_dir in target_dirs.values():
        image_dir.mkdir(parents=True, exist_ok=True)
        label_dir.mkdir(parents=True, exist_ok=True)

    copied = 0
    for pair in pool_pairs:
        image_dir, label_dir = target_dirs[assignment[pair.stem]]
        shutil.copy2(pair.image, image_dir / pair.image.name)
        shutil.copy2(pair.label, label_dir / pair.label.name)
        copied += 1
    return copied


def clear_label_caches(dataset_root: Path, config: dict[str, Any]) -> None:
    """Stale Ultralytics .cache files must go; they are rebuilt on next train/val."""
    for split in SPLITS:
        _, label_dir = split_dirs(dataset_root, config, split)
        # Ultralytics names the cache after the folder of the first label file it reads.
        for cache_file in [*label_dir.glob("*.cache"), label_dir.with_name(f"{label_dir.name}.cache")]:
            cache_file.unlink(missing_ok=True)


def write_manifest(
    dataset_root: Path,
    pool_pairs: list[Pair],
    assignment: dict[str, str],
    seed: int,
) -> Path:
    """Write a CSV recording which split each plate went to (a reproducibility record)."""
    timestamp = datetime.now().isoformat(timespec="seconds")
    rows = [
        {
            "stem": pair.stem,
            "split": assignment[pair.stem],
            "image_name": pair.image.name,
            "seed": seed,
            "timestamp": timestamp,
        }
        for pair in sorted(pool_pairs, key=lambda item: item.stem)
    ]
    manifest_path = dataset_root / "split_manifest.csv"
    pd.DataFrame(rows, columns=("stem", "split", "image_name", "seed", "timestamp")).to_csv(
        manifest_path, index=False
    )
    return manifest_path


def parse_args() -> argparse.Namespace:
    """Read the command-line options (seed, val/test fractions, dry-run)."""
    parser = argparse.ArgumentParser(description="Re-split the YOLO plate dataset.")
    parser.add_argument("--data", default="configs/dataset.yaml", help="Dataset YAML path.")
    parser.add_argument("--seed", type=int, default=42, help="Shuffle seed for reproducibility.")
    parser.add_argument("--val-frac", type=float, default=0.1, help="Validation fraction.")
    parser.add_argument("--test-frac", type=float, default=0.1, help="Test fraction.")
    parser.add_argument("--dry-run", action="store_true", help="Preview the split without changing files.")
    return parser.parse_args()


def main() -> None:
    """Re-split the detection dataset 80/10/10 from the immutable pool, deterministically."""
    args = parse_args()
    data_path = resolve_project_path(args.data)
    config = load_dataset_yaml(data_path)
    dataset_root = resolve_dataset_root(data_path, config)

    pool_pairs, orphans, pool_status = ensure_pool(dataset_root, config, args.dry_run)
    if not pool_pairs:
        raise SystemExit("No image+label pairs found; nothing to split.")

    assignment = assign_splits(pool_pairs, args.seed, args.val_frac, args.test_frac)

    pool_image_dir, _ = pool_dirs(dataset_root)
    print(f"Dataset root: {dataset_root}")
    print(f"Pool ({pool_status}): {pool_image_dir}")
    print(f"Pooled pairs: {len(pool_pairs)} (seed={args.seed})")
    counts = {split: sum(1 for value in assignment.values() if value == split) for split in SPLITS}
    print(f"Target split: train={counts['train']} val={counts['val']} test={counts['test']}")

    if orphans:
        print(f"\nOrphans/strays skipped ({len(orphans)}):")
        for orphan in orphans:
            print(f"  - {orphan}")

    if args.dry_run:
        print("\n[dry-run] No files changed. Pool would be preserved; splits would be rebuilt from it.")
        return

    clear_working_splits(dataset_root, config)
    copied = populate_splits(pool_pairs, assignment, dataset_root, config)
    clear_label_caches(dataset_root, config)
    manifest_path = write_manifest(dataset_root, pool_pairs, assignment, args.seed)

    print(f"\nCopied {copied} pair(s) from pool into train/val/test.")
    print(f"Manifest written to: {manifest_path}")


if __name__ == "__main__":
    main()
