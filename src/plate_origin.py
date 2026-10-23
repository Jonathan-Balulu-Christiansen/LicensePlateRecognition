"""Which country a plate belongs to, and how its number is written there.

For now there is one rule: Israeli civilian plates are yellow, so a plate whose colored
pixels are mostly yellow is called Israeli. The decision is returned together with the
method and its score, so the UI can say *how* it knows — and so a later model (flag,
character pattern) can replace the heuristic behind the same `Origin` result.

How "yellow" is measured, and why (all numbers measured on this project's data):
- Only *colored* pixels count — saturated and bright enough to have a hue. A plate far
  from the headlights at night is grey-blue: it carries no color evidence, which is not
  the same as evidence that it is not yellow. A read whose plate is under 30% colored is
  therefore skipped instead of voting "not yellow". (night1 locked as "Unknown" before
  this: its first reads were 0-5% yellow while the car was still dark.)
- Hue 10-40 on OpenCV's 0-180 scale: plate yellow drifts toward orange under headlights.
- Per read, the yellow share of the colored pixels: 0.74-1.00 on the day clips, 0.32-1.00
  on usable night reads (every night track's median >= 0.6), 0.87+ on all 107 labelled
  crops in data/reader/plates; white, grey, blue and red plates score 0.
- Per track, the median over its usable reads, against a 0.5 threshold.
Known blind spot: a white plate under orange sodium street light would read as yellow.
"""

from __future__ import annotations

from dataclasses import dataclass
from statistics import median

import cv2
import numpy as np


def crop_blue_strip(plate_bgr: np.ndarray) -> np.ndarray:
    """Remove the blue EU-style 'IL' strip on the left, if one is present, so it does not
    count against the plate's yellow."""
    b, g, r = (plate_bgr[:, :, i].astype(int) for i in range(3))
    is_blue = (b - r > 40) & (b - g > 20)
    width = plate_bgr.shape[1]
    left = width // 4  # the strip lives in the left quarter
    blue_frac = is_blue[:, :left].mean(axis=0)  # fraction of blue pixels per column
    blue_cols = np.where(blue_frac > 0.5)[0]
    if blue_cols.size and blue_cols.min() < 0.1 * width:
        return plate_bgr[:, blue_cols.max() + 1:]
    return plate_bgr

YELLOW_THRESHOLD = 0.5  # median yellow share (of colored pixels) that makes a track IL
MIN_COLOR_COVERAGE = 0.3  # reads with less of the plate colored carry no color evidence
_HUE_RANGE = (10, 40)
_MIN_SATURATION = 70
_MIN_VALUE = 60


@dataclass(frozen=True)
class Origin:
    """Where a plate is from, plus how that was decided."""

    code: str  # ISO 3166 alpha-2, or "" when unknown
    country: str
    method: str  # human-readable, shown in the UI
    score: float  # the method's own measurement (here: median yellow share)


UNKNOWN = Origin("", "Unknown", "plate color · too dark to judge", 0.0)


def plate_color(plate_bgr: np.ndarray) -> tuple[float, float]:
    """(yellow share of the colored pixels, share of the plate that is colored) for one crop,
    blue side strip removed."""
    if plate_bgr.size == 0:
        return 0.0, 0.0
    hsv = cv2.cvtColor(crop_blue_strip(plate_bgr), cv2.COLOR_BGR2HSV)
    hue, saturation, value = cv2.split(hsv)
    colored = (saturation >= _MIN_SATURATION) & (value >= _MIN_VALUE)
    yellow = colored & (hue >= _HUE_RANGE[0]) & (hue <= _HUE_RANGE[1])
    colored_count = int(colored.sum())
    coverage = colored_count / colored.size
    return (float(yellow.sum()) / colored_count if colored_count else 0.0), coverage


def origin_from_samples(samples: list[tuple[float, float]]) -> Origin:
    """Decide the origin from one track's plate_color() samples: the median yellow share
    over the reads that carry color evidence (one odd crop cannot flip it)."""
    usable = [share for share, coverage in samples if coverage >= MIN_COLOR_COVERAGE]
    if not usable:
        return UNKNOWN
    score = median(usable)
    if score >= YELLOW_THRESHOLD:
        return Origin("IL", "Israel", "plate color · yellow", score)
    return Origin("", "Unknown", "plate color · not yellow", score)


def format_plate(number: str, code: str) -> str:
    """Write a plate number the way it appears on the plate itself.

    Israel: 8 digits as 123-45-678, 7 digits as 12-345-67. Anything else is returned as is.
    """
    if code == "IL" and number.isdigit():
        if len(number) == 8:
            return f"{number[:3]}-{number[3:5]}-{number[5:]}"
        if len(number) == 7:
            return f"{number[:2]}-{number[2:5]}-{number[5:]}"
    return number


def plate_format_label(number: str, code: str) -> str:
    """A short description of the plate's format, e.g. "8 digits · 3-2-3"."""
    if code == "IL" and number.isdigit() and len(number) in (7, 8):
        return "8 digits · 3-2-3" if len(number) == 8 else "7 digits · 2-3-2"
    return f"{len(number)} characters" if number else ""
