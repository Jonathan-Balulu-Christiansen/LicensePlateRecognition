"""Train the plate READER (CNN + CTC, see plate_reader.py) and save it as a run.

Two stages, because real labelled plates are far too few for a 1.85M-parameter network on
their own:
1. Pretrain on the synthetic plates in data/synthetic/plates/v1 (made by tools/synth_plates.py;
   two-row plates are left out — the reader reads one row).
2. Fine-tune on every real plate in data/ocr/plates (file name = the number), half of every
   batch real and half synthetic, so the network does not forget the synthetic variety.

Augmentation is ONLINE: every batch is built fresh from the plate images on disk, so each epoch
sees new versions of every plate and nothing is memorised (validation is never augmented). Each
sample gets a "difficulty budget" — changes from 1-3 groups (geometry, image quality, light,
dirt), never all at once — and 15% stay clean. Every change keeps the label true: a rotated or
re-framed plate is drawn on an enlarged canvas so no digit ever leaves the crop.

    python src/train_reader.py              # train; the run goes to runs/reader/<timestamp>/
    python src/train_reader.py --promote    # ... and put it in service: weights/reader/plate_reader.keras
    python src/train_reader.py --cv         # 5-fold cross-validation by plate instead (no model saved)
    python src/train_reader.py --preview    # write a sheet of augmented training samples, then stop

The app can also try any run directly: its Reader list includes runs/reader/.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import shutil
import time
from datetime import datetime
from pathlib import Path
from typing import Any

os.environ.setdefault("KERAS_BACKEND", "torch")

import cv2
import keras
import numpy as np

from common import PROJECT_ROOT, create_run_dir, write_run_metadata
from plate_reader import build_model, decode, encode, prep

cv2.setNumThreads(0)  # OpenCV's thread pool crashes intermittently next to PyTorch in one process

SYNTH_DIR = PROJECT_ROOT / "data" / "synthetic" / "plates" / "v1"
REAL_DIR = PROJECT_ROOT / "data" / "ocr" / "plates"
RUNS_DIR = PROJECT_ROOT / "runs" / "reader"
LONG_TYPES = ("long_strip", "long_plain")
PROGRESS: Any = 2  # Keras fit(verbose=2): one line per epoch (its stubs type verbose as str)

# A training sample: (image path, the number, True if synthetic, plate type).
Sample = tuple[Path, str, bool, str]


def log(message: str) -> None:
    """Print with a timestamp (training runs take a while)."""
    print(time.strftime("%H:%M:%S"), message, flush=True)


def read_image(path: Path) -> np.ndarray:
    """cv2.imread that fails loudly instead of returning None."""
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return image


# ------------------------------------------------------------------- augmentation
# Each function takes a BGR plate crop and a random.Random and returns a new crop whose label
# is still exactly the same number.


def _bumper(img: np.ndarray, height: int, width: int, rng: random.Random) -> np.ndarray:
    """A dark, slightly noisy, blurred background — what surrounds a plate on a car — for the
    margins that rotation and re-framing create (instead of smeared edge pixels)."""
    tone = img.reshape(-1, 3).mean(0) * rng.uniform(0.2, 0.5)
    canvas = np.full((height, width, 3), tone, np.float32)
    canvas += np.random.default_rng(rng.randrange(1 << 30)).normal(0, 6, canvas.shape)
    return cv2.GaussianBlur(np.clip(canvas, 0, 255).astype(np.uint8), (0, 0), 3)


def _warp(img: np.ndarray, matrix: np.ndarray, size: tuple[int, int], rng: random.Random, affine: bool) -> np.ndarray:
    """Warp onto a bumper-coloured canvas of `size` (w, h)."""
    h, w = img.shape[:2]
    warp = cv2.warpAffine if affine else cv2.warpPerspective
    out = warp(img, matrix, size, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
    mask = warp(np.full((h, w), 255, np.uint8), matrix, size, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return np.where(mask[..., None] > 0, out, _bumper(img, size[1], size[0], rng))


def rotate(img: np.ndarray, rng: random.Random, max_deg: float = 12.0) -> np.ndarray:
    """A crooked plate or camera: a random angle within +-max_deg, on an enlarged canvas so no
    corner — no digit — leaves the frame."""
    h, w = img.shape[:2]
    deg = rng.uniform(-max_deg, max_deg)
    rad = np.radians(abs(deg))
    new_w = int(w * np.cos(rad) + h * np.sin(rad)) + 2
    new_h = int(h * np.cos(rad) + w * np.sin(rad)) + 2
    matrix = cv2.getRotationMatrix2D((w / 2, h / 2), deg, 1.0)
    matrix[0, 2] += (new_w - w) / 2
    matrix[1, 2] += (new_h - h) / 2
    return _warp(img, matrix, (new_w, new_h), rng, affine=True)


def view_angle(img: np.ndarray, rng: random.Random) -> np.ndarray:
    """Seen from the side (10-40 degrees): the far edge gets shorter and the plate narrower."""
    h, w = img.shape[:2]
    yaw = np.radians(rng.uniform(10, 40))
    shrink = 0.35 * np.sin(yaw) * h / 2
    new_w = max(8, int(w * np.cos(yaw)))
    src = np.array([[0, 0], [w, 0], [w, h], [0, h]], np.float32)
    if rng.random() < 0.5:  # the left edge is the far one
        dst = np.array([[0, shrink], [new_w, 0], [new_w, h], [0, h - shrink]], np.float32)
    else:
        dst = np.array([[0, 0], [new_w, shrink], [new_w, h - shrink], [0, h]], np.float32)
    return _warp(img, cv2.getPerspectiveTransform(src, dst), (new_w, h), rng, affine=False)


def pitch(img: np.ndarray, rng: random.Random) -> np.ndarray:
    """A camera above (or below) the plate: one horizontal edge looks wider than the other."""
    h, w = img.shape[:2]
    inset = w * rng.uniform(0.02, 0.06)
    src = np.array([[0, 0], [w, 0], [w, h], [0, h]], np.float32)
    if rng.random() < 0.7:  # cameras are usually above
        dst = np.array([[0, 0], [w, 0], [w - inset, h], [inset, h]], np.float32)
    else:
        dst = np.array([[inset, 0], [w - inset, 0], [w, h], [0, h]], np.float32)
    return _warp(img, cv2.getPerspectiveTransform(src, dst), (w, h), rng, affine=False)


def reframe(img: np.ndarray, rng: random.Random) -> np.ndarray:
    """The detector's box is never exact: a little tight (inside the plate border, not into the
    digits) or a little loose and off-centre (bumper around the plate)."""
    h, w = img.shape[:2]
    if rng.random() < 0.5:  # tight
        dy, dx = int(h * rng.uniform(0, 0.04)), int(w * rng.uniform(0, 0.03))
        return img[dy: h - dy, dx: w - dx]
    top, bottom = int(h * rng.uniform(0, 0.15)), int(h * rng.uniform(0, 0.15))
    left, right = int(w * rng.uniform(0, 0.06)), int(w * rng.uniform(0, 0.06))
    canvas = _bumper(img, h + top + bottom, w + left + right, rng)
    canvas[top: top + h, left: left + w] = img
    return canvas


def squeeze(img: np.ndarray, rng: random.Random, low: float = 0.7, high: float = 1.3) -> np.ndarray:
    """Real crops come in many aspect ratios: rescale the width only."""
    h, w = img.shape[:2]
    new_w = max(8, round(w * rng.uniform(low, high)))
    return cv2.resize(img, (new_w, h), interpolation=cv2.INTER_AREA if new_w < w else cv2.INTER_LINEAR)


def blur(img: np.ndarray, rng: random.Random) -> np.ndarray:
    """Gentle focus blur, scaled to the plate height so the digits stay readable."""
    return cv2.GaussianBlur(img, (0, 0), max(0.4, img.shape[0] / rng.uniform(55, 90)))


def motion_blur(img: np.ndarray, rng: random.Random) -> np.ndarray:
    """Horizontal motion blur (a passing car), its length scaled to the plate height."""
    length = max(3, int(img.shape[0] / rng.uniform(10, 16)) | 1)
    kernel = np.zeros((length, length), np.float32)
    kernel[length // 2, :] = 1.0 / length
    return cv2.filter2D(img, -1, kernel)


def low_resolution(img: np.ndarray, rng: random.Random) -> np.ndarray:
    """The plate reached the reader only 90-160 px wide (90 is the smallest the app reads)."""
    h, w = img.shape[:2]
    small_w = min(w, rng.randint(90, 160))
    small = cv2.resize(img, (small_w, max(8, int(h * small_w / w))), interpolation=cv2.INTER_AREA)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)


def noise(img: np.ndarray, rng: random.Random) -> np.ndarray:
    """Sensor noise."""
    grain = np.random.default_rng(rng.randrange(1 << 30)).normal(0, rng.uniform(3, 10), img.shape)
    return np.clip(img.astype(np.float32) + grain, 0, 255).astype(np.uint8)


def jpeg(img: np.ndarray, rng: random.Random) -> np.ndarray:
    """Compression artefacts of a cheap camera or a recorded stream."""
    _, encoded = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, rng.randint(35, 85)])
    return np.asarray(cv2.imdecode(encoded, cv2.IMREAD_COLOR))


def brightness(img: np.ndarray, rng: random.Random) -> np.ndarray:
    """Under- or over-exposure."""
    return np.clip(img.astype(np.float32) * rng.uniform(0.55, 1.3) + rng.uniform(-20, 30), 0, 255).astype(np.uint8)


def low_contrast(img: np.ndarray, rng: random.Random) -> np.ndarray:
    """Haze, fog, a dirty lens."""
    pixels = img.astype(np.float32)
    mean = pixels.mean()
    return np.clip((pixels - mean) * rng.uniform(0.5, 0.8) + mean, 0, 255).astype(np.uint8)


def night(img: np.ndarray, rng: random.Random) -> np.ndarray:
    """Dim, warm street light plus sensor noise."""
    cast = np.array([rng.uniform(0.4, 0.6), rng.uniform(0.55, 0.75), rng.uniform(0.8, 1.0)], np.float32)
    dim = np.clip(img.astype(np.float32) * cast * rng.uniform(0.6, 0.9), 0, 255).astype(np.uint8)
    return noise(dim, rng)


def glare(img: np.ndarray, rng: random.Random) -> np.ndarray:
    """A headlight or sun reflection: a soft bright blob on part of the plate."""
    h, w = img.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w]
    cx, cy, radius = rng.uniform(0.15, 0.85) * w, rng.uniform(0.2, 0.7) * h, rng.uniform(0.2, 0.4)
    blob = np.exp(-(((xx - cx) / (radius * w)) ** 2 + ((yy - cy) / (radius * 2.2 * h)) ** 2) * 2.5)
    return np.clip(img.astype(np.float32) + blob[..., None] * rng.uniform(80, 150), 0, 255).astype(np.uint8)


def shadow(img: np.ndarray, rng: random.Random) -> np.ndarray:
    """A shadow band (a pole, the bumper) over part of the plate, with a soft edge."""
    h, w = img.shape[:2]
    mask = np.ones((h, w), np.float32)
    start = rng.uniform(0, 0.6)
    mask[:, int(start * w): int((start + rng.uniform(0.2, 0.45)) * w)] = rng.uniform(0.4, 0.7)
    mask = cv2.GaussianBlur(mask, (0, 0), max(1.0, w / 60))
    return np.clip(img.astype(np.float32) * mask[..., None], 0, 255).astype(np.uint8)


def dirt(img: np.ndarray, rng: random.Random) -> np.ndarray:
    """One to three small dark smudges on the border rows/columns — never over a whole digit."""
    out = img.copy()
    h, w = img.shape[:2]
    for _ in range(rng.randint(1, 3)):
        if rng.random() < 0.5:  # top or bottom border
            x, y = rng.uniform(0.05, 0.95) * w, rng.choice([rng.uniform(0, 0.12), rng.uniform(0.88, 1)]) * h
        else:  # left or right end
            x, y = rng.choice([rng.uniform(0, 0.06), rng.uniform(0.94, 1)]) * w, rng.uniform(0.1, 0.9) * h
        tone = int(rng.uniform(25, 70))
        cv2.circle(out, (int(x), int(y)), max(1, int(h * rng.uniform(0.04, 0.09))), (tone, tone + 4, tone + 8), -1)
    return cv2.GaussianBlur(out, (3, 3), 0)


AUGMENT_GROUPS = {
    "geometry": (view_angle, pitch, reframe, squeeze),
    "quality": (blur, motion_blur, low_resolution, noise, jpeg),
    "light": (brightness, low_contrast, night, glare, shadow),
    "dirt": (dirt,),
}


def digit_band(bgr: np.ndarray, n_digits: int) -> tuple[int, int, int, int] | None:
    """(top, bottom, left, right) of the digits on a synthetic plate, from dark digit-shaped
    components of similar height and centre; None if fewer than max(4, n - 3) digits agree."""
    grey = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    h, w = grey.shape
    core = grey[int(h * 0.2): int(h * 0.8), int(w * 0.15): int(w * 0.95)]
    threshold, _ = cv2.threshold(core, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    _, _, stats, _ = cv2.connectedComponentsWithStats((grey < threshold).astype(np.uint8), connectivity=8)
    boxes = [(x, y, bw, bh) for x, y, bw, bh, _ in stats[1:]
             if 0.2 * h <= bh <= 0.95 * h and bw <= 0.2 * w and 1.1 * bw <= bh <= 6 * bw
             and x > 0 and y > 0 and x + bw < w and y + bh < h]
    best: list[tuple[int, int, int, int]] = []
    for ref in boxes:
        group = [b for b in boxes if abs(b[3] - ref[3]) <= 0.2 * ref[3]
                 and abs((b[1] + b[3] / 2) - (ref[1] + ref[3] / 2)) <= 0.25 * ref[3]]
        if len(group) > len(best):
            best = group
    if len(best) < max(4, n_digits - 3):
        return None
    return (min(b[1] for b in best), max(b[1] + b[3] for b in best),
            min(b[0] for b in best), max(b[0] + b[2] for b in best))


def tight_crop(bgr: np.ndarray, band, rng: random.Random, fill=(0.65, 0.90), side=0.06) -> np.ndarray:
    """Synthetic plates keep margins that real detector crops lack: crop so the digits fill
    U(fill) of the height and trim up to 6% per side — never into a digit."""
    h, w = bgr.shape[:2]
    top, bottom, left, right = band
    crop_h = min(h, max(bottom - top + 2, round((bottom - top) / rng.uniform(*fill))))
    y0 = top - round((crop_h - (bottom - top)) * rng.uniform(0.25, 0.75))
    y0 = max(0, min(y0, h - crop_h, top - 1))
    y1 = min(h, max(y0 + crop_h, bottom + 1))
    x0 = min(round(w * rng.uniform(0, side)), max(0, left - 2))
    x1 = max(w - round(w * rng.uniform(0, side)), min(w, right + 2))
    return bgr[y0:y1, x0:x1]


def augment_plate(img: np.ndarray, sample: Sample, rng: random.Random, cfg: dict) -> np.ndarray:
    """One fresh training version of a plate crop (the label stays exactly the same).

    Synthetic plates are first cropped tight (60%), like real detector crops. Then 15% of all
    samples stay clean; the rest get a random rotation within +-12 degrees (60% of them) and a
    difficulty budget of changes from 1-3 groups, one change per group.
    """
    _, number, synthetic, plate_type = sample
    if synthetic and rng.random() < cfg["tight_share"]:
        band = digit_band(img, len(number))
        if band is not None:
            img = tight_crop(img, band, rng)
    if rng.random() < cfg["clean_share"]:
        return img
    if rng.random() < cfg["rotate_share"]:
        img = rotate(img, rng, cfg["max_rotation"])
    for group in rng.sample(list(AUGMENT_GROUPS), rng.choice((1, 2, 2, 3))):
        change = rng.choice(AUGMENT_GROUPS[group])
        if change is squeeze and plate_type in LONG_TYPES:  # long plates are often compressed hard
            img = squeeze(img, rng, 0.45, 1.3)
        else:
            img = change(img, rng)
    return img


# --------------------------------------------------------------------------------- data


def synthetic_samples(cfg: dict) -> tuple[list[Sample], list[Sample]]:
    """(train, validation) one-row synthetic plates; two-row plates are left out."""
    with (SYNTH_DIR / "labels.csv").open(newline="", encoding="utf-8") as file:
        rows = [r for r in csv.DictReader(file) if r["plate_type"] != "two_row" and r["rows"] == "1"]
    random.Random(cfg["seed"]).shuffle(rows)
    samples = [(SYNTH_DIR / r["file"], r["number"], True, r["plate_type"]) for r in rows]
    return samples[cfg["synthetic_val"]:], samples[: cfg["synthetic_val"]]


def real_samples() -> list[Sample]:
    """Every labelled real plate crop in data/ocr/plates. The file name starts with the number:
    "12345678.jpg", or "12345678_<video>_f<frame>.png" for several crops of one plate."""
    paths = sorted(p for p in REAL_DIR.rglob("*") if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
    return [(path, path.stem.split("_")[0], False, "real") for path in paths]


def clean_arrays(samples: list[Sample]) -> tuple[np.ndarray, list[str]]:
    """Network inputs without any augmentation (validation and evaluation)."""
    return np.stack([prep(read_image(s[0])) for s in samples]).astype(np.float32), [s[1] for s in samples]


class PlateBatches(keras.utils.PyDataset):
    """Batches built on the fly: each draws plates at random (a `real_share` of them real, the
    rest synthetic), augments every one afresh, and prepares them for the network."""

    def __init__(self, real: list[Sample], synthetic: list[Sample], real_share: float, steps: int,
                 cfg: dict, seed: int, **kwargs) -> None:
        """`steps` batches per epoch; `kwargs` go to PyDataset (workers etc.)."""
        super().__init__(**kwargs)
        self.real, self.synthetic, self.real_share = real, synthetic, real_share
        self.steps, self.cfg, self.seed, self.epoch = steps, cfg, seed, 0

    def __len__(self) -> int:
        return self.steps

    def __getitem__(self, index: int):
        rng = random.Random(self.seed * 1_000_003 + self.epoch * 10_007 + index)
        picks = [rng.choice(self.real) if self.real and rng.random() < self.real_share else rng.choice(self.synthetic)
                 for _ in range(self.cfg["batch"])]
        x = np.stack([prep(augment_plate(read_image(s[0]), s, rng, self.cfg)) for s in picks]).astype(np.float32)
        return x, encode([s[1] for s in picks])

    def on_epoch_end(self) -> None:
        self.epoch += 1


# ----------------------------------------------------------------------------- training


def exact(model, x: np.ndarray, numbers: list[str]) -> float:
    """Share of plates read exactly (constrained decoding)."""
    reads = decode(np.asarray(model.predict(x, verbose=0, batch_size=128)))
    return float(np.mean([text == number for (text, _), number in zip(reads, numbers)]))


def pretrain(cfg: dict, synthetic: list[Sample], x_val: np.ndarray, y_val: np.ndarray, weights: Path):
    """Stage 1 on the synthetic plates (reused when the same configuration was pretrained before)."""
    keras.utils.set_random_seed(cfg["seed"])
    model = build_model()
    model.compile(optimizer=keras.optimizers.Adam(cfg["pretrain_lr"]), loss=keras.losses.CTC())
    key = weights.with_suffix(".json")
    settings = {k: v for k, v in cfg.items() if not k.startswith("finetune")}
    if weights.exists() and key.exists() and json.loads(key.read_text()) == settings:
        log(f"reusing pretrained weights {weights.relative_to(PROJECT_ROOT).as_posix()}")
        model.load_weights(weights)
        return model
    steps = len(synthetic) // cfg["batch"]
    log(f"pretraining on {len(synthetic)} synthetic plates, augmented online ({steps} batches per epoch)")
    batches = PlateBatches([], synthetic, 0.0, steps, cfg, seed=cfg["seed"], workers=cfg["workers"])
    model.fit(batches, validation_data=(x_val, y_val), epochs=cfg["pretrain_epochs"], verbose=PROGRESS,
              callbacks=[keras.callbacks.ReduceLROnPlateau(patience=2, factor=0.5),
                         keras.callbacks.EarlyStopping(patience=4, restore_best_weights=True)])
    weights.parent.mkdir(parents=True, exist_ok=True)
    model.save_weights(weights)
    key.write_text(json.dumps(settings))
    return model


def fine_tune(cfg: dict, pretrained: Path, real: list[Sample], synthetic: list[Sample], seed: int):
    """Stage 2: the pretrained network on the real plates, half of each batch synthetic."""
    keras.utils.set_random_seed(seed)
    model = build_model()
    model.load_weights(pretrained)
    model.compile(optimizer=keras.optimizers.Adam(cfg["finetune_lr"]), loss=keras.losses.CTC())
    batches = PlateBatches(real, synthetic, cfg["finetune_real_share"], cfg["finetune_steps"], cfg,
                           seed=100 + seed, workers=cfg["workers"])
    model.fit(batches, epochs=cfg["finetune_epochs"], verbose=PROGRESS)
    return model


def preview(cfg: dict, real: list[Sample], synthetic: list[Sample], out: Path) -> None:
    """A sheet of 24 freshly augmented training samples (real top, synthetic bottom), each next
    to what the network receives, to check the augmentation by eye."""
    rng = random.Random(7)
    tiles = []
    for sample in [rng.choice(real) for _ in range(12)] + [rng.choice(synthetic) for _ in range(12)]:
        img = augment_plate(read_image(sample[0]), sample, rng, cfg)
        pair = []
        for picture in (img, cv2.cvtColor(prep(img)[..., 0], cv2.COLOR_GRAY2BGR)):
            pair.append(cv2.copyMakeBorder(cv2.resize(picture, (240, 62)), 0, 18, 3, 3,
                                           cv2.BORDER_CONSTANT, value=(24, 24, 28)))
        cv2.putText(pair[0], sample[1], (5, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (235, 235, 235), 1, cv2.LINE_AA)
        tiles.append(np.hstack(pair))
    rows = [np.hstack(tiles[i: i + 3]) for i in range(0, len(tiles), 3)]
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), np.vstack(rows))
    log(f"preview written to {out}")


def parse_args() -> argparse.Namespace:
    """Read the command-line options."""
    parser = argparse.ArgumentParser(description="Train the CNN+CTC plate reader.")
    parser.add_argument("--promote", action="store_true", help="Also copy the run to weights/reader/.")
    parser.add_argument("--cv", action="store_true", help="5-fold cross-validation by plate instead (no model saved).")
    parser.add_argument("--preview", action="store_true", help="Write a sheet of augmented samples and stop.")
    return parser.parse_args()


def main() -> None:
    """Pretrain on synthetic plates, fine-tune on the real ones, save (and optionally promote)."""
    args = parse_args()
    # --- Parameters (edit these directly) ---
    cfg = {"seed": 1, "synthetic_val": 2000, "batch": 64, "workers": 8,
           # augmentation (online): see augment_plate()
           "tight_share": 0.6, "clean_share": 0.15, "rotate_share": 0.6, "max_rotation": 12.0,
           "pretrain_epochs": 20, "pretrain_lr": 1e-3,
           "finetune_epochs": 12, "finetune_lr": 3e-4, "finetune_steps": 200, "finetune_real_share": 0.5}

    synthetic, synthetic_val = synthetic_samples(cfg)
    real = real_samples()
    if args.preview:
        preview(cfg, real, synthetic, RUNS_DIR / "augmentation_preview.png")
        return
    x_val, val_numbers = clean_arrays(synthetic_val)
    pretrained = RUNS_DIR / "pretrained" / "synthetic.weights.h5"
    model = pretrain(cfg, synthetic, x_val, encode(val_numbers), pretrained)
    log(f"synthetic validation, whole plates exact: {exact(model, x_val, val_numbers):.4f}")
    real_x, real_numbers = clean_arrays(real)
    log(f"real plates read by the synthetic-only model: {exact(model, real_x, real_numbers):.4f}")

    if args.cv:  # folds by plate, seed-0 shuffle
        # Folds are by plate number, so crops of one plate never sit on both sides of a fold.
        order = list(dict.fromkeys(s[1] for s in real))
        random.Random(0).shuffle(order)
        correct = 0
        for k, test_ids in enumerate(set(order[i::5]) for i in range(5)):
            test = [s for s in real if s[1] in test_ids]
            fold_model = fine_tune(cfg, pretrained, [s for s in real if s[1] not in test_ids], synthetic, k)
            x_test, test_numbers = clean_arrays(test)
            hits = round(exact(fold_model, x_test, test_numbers) * len(test))
            correct += hits
            log(f"fold {k}: {hits}/{len(test)} whole plates exact")
        log(f"cross-validation: {correct}/{len(real)} = {correct / len(real):.4f} of whole plates exact")
        return

    model = fine_tune(cfg, pretrained, real, synthetic, 0)
    started_at = datetime.now()
    run_dir = create_run_dir(RUNS_DIR, started_at)
    model.save(run_dir / "plate_reader.keras")
    write_run_metadata(run_dir, {"timestamp": started_at.isoformat(timespec="seconds"), "config": cfg,
                                 "real_plates": len(real), "synthetic_train": len(synthetic)})
    log(f"reader saved to {(run_dir / 'plate_reader.keras').relative_to(PROJECT_ROOT).as_posix()}")
    if args.promote:
        target = PROJECT_ROOT / "weights" / "reader" / "plate_reader.keras"
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(run_dir / "plate_reader.keras", target)
        log(f"promoted to {target.relative_to(PROJECT_ROOT).as_posix()}")


if __name__ == "__main__":
    main()
