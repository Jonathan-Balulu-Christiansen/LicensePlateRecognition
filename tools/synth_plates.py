"""Generate synthetic Israeli licence-plate crops with exact labels, for OCR pretraining.

Six layouts cover long plates with/without a strip, short plates with a strip or bottom
badge or no badge, and two-row plates. Numbers are chosen first; rendering and camera
effects never change their labels. Only Pillow, numpy and CPU OpenCV are used.

Run (outputs go to data/synthetic/plates/<name>/, which is not in git):

    python tools/synth_plates.py --count 96 --name samples_v2 --seed 7 --preview
    python tools/synth_plates.py --count 40000 --name v1 --seed 1 --preview
    python tools/synth_plates.py --count 96 --name short --types short_strip short_badge

Each run contains <number>_<index>.png, labels.csv and optionally preview.png (first 96).
--types restricts the layouts; --type-mix accepts TYPE=WEIGHT pairs and normalizes them.
"""

from __future__ import annotations

import argparse
import csv
import math
import random
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont, features

PROJECT_ROOT = Path(__file__).resolve().parents[1]  # Keep generation independent of torch/CUDA.
cv2.setNumThreads(0)  # OpenCV's thread pool crashes intermittently next to PyTorch in one process
FONT_PATH = Path(r"C:\Windows\Fonts\bahnschrift.ttf")
TEXT_FONT_PATH = Path(r"C:\Windows\Fonts\arial.ttf")
HAS_RAQM = features.check("raqm")
OUT_ROOT = PROJECT_ROOT / "data" / "synthetic" / "plates"
PLATE_SIZES = {
    "long_strip": (520, 112),
    "long_plain": (520, 112),
    "short_strip": (520, 260),
    "short_badge": (520, 260),
    "short_plain": (520, 260),
    "two_row": (312, 260),
}
# Requested shares sum to 95. Preserve their ratios; split short-with-flag 7:8.
DEFAULT_MIX = dict(zip(PLATE_SIZES, (55, 12, 7, 8, 5, 8)))
LABEL_COLUMNS = ("file", "number", "plate_type", "digits", "rows", "has_flag", "has_il", "separator", "clean")
SEPARATORS = {"hyphen": "-", "dot": "\u00b7", "colon": ":", "none": ""}
_fonts: dict[tuple[int, int, int], ImageFont.FreeTypeFont] = {}


class Rng:
    """random.Random plus numpy's normal(), seeded together, so a run is reproducible."""

    def __init__(self, seed: int) -> None:
        self.py = random.Random(seed)
        self.np = np.random.default_rng(seed)

    def random(self) -> float:
        return self.py.random()

    def uniform(self, low: float, high: float) -> float:
        return self.py.uniform(low, high)

    def randint(self, low: int, high: int) -> int:
        return self.py.randint(low, high)

    def choice(self, options):
        return self.py.choice(options)

    def normal(self, mean: float, std: float, shape) -> np.ndarray:
        return self.np.normal(mean, std, shape)


@dataclass(frozen=True)
class PlateStyle:
    """Sample decorations once so the manifest describes exactly what was rendered."""

    plate_type: str
    has_flag: bool
    has_il: bool
    separator: str
    triangle: bool
    small_print: bool
    country_text: bool
    badge_left: bool

    @property
    def rows(self) -> int:
        return 2 if self.plate_type == "two_row" else 1


def choose_style(plate_type: str, rng: Rng) -> PlateStyle:
    if plate_type not in PLATE_SIZES:
        raise ValueError(f"Unknown Israeli plate type: {plate_type}")
    decorated = not plate_type.endswith("_plain")
    # Independent draws: missing flag does not force missing IL, or vice versa.
    flag, il = rng.random() < 0.94, rng.random() < 0.96
    return PlateStyle(
        plate_type, decorated and flag, decorated and il,
        rng.choice(("hyphen", "dot", "colon", "dot", "hyphen", "none")),
        rng.random() < 0.68, rng.random() < 0.55, rng.random() < 0.75, rng.random() < 0.5,
    )


def type_mix(types: list[str] | tuple[str, ...] | None = None,
             weights: dict[str, float] | None = None) -> dict[str, float]:
    """Validate and normalize weights; an explicit mix selects its own types by default."""
    selected = list(types) if types is not None else list(weights if weights is not None else PLATE_SIZES)
    if not selected or len(set(selected)) != len(selected):
        raise ValueError("--types must contain distinct plate types")
    if any(name not in PLATE_SIZES for name in selected):
        raise ValueError(f"Types must be chosen from: {', '.join(PLATE_SIZES)}")
    if weights is not None and set(weights) != set(selected):
        raise ValueError("--type-mix must give exactly one weight for every selected type")
    values = {name: float((DEFAULT_MIX if weights is None else weights)[name]) for name in selected}
    if any(not math.isfinite(v) or v < 0 for v in values.values()) or sum(values.values()) <= 0:
        raise ValueError("Type weights must be finite, nonnegative and have a positive total")
    total = sum(values.values())
    if not math.isfinite(total):
        raise ValueError("Type weight total must be finite")
    return {name: value / total for name, value in values.items()}


def choose_type(rng: Rng, mix: dict[str, float]) -> str:
    return rng.py.choices(tuple(mix), weights=tuple(mix.values()), k=1)[0]


def _font(size: int, weight: int, width: int) -> ImageFont.FreeTypeFont:
    key = (size, weight, width)
    if key not in _fonts:
        font = ImageFont.truetype(str(FONT_PATH), size)
        font.set_variation_by_axes([weight, width])
        _fonts[key] = font
    return _fonts[key]


@lru_cache(maxsize=16)
def _text_font(size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(str(TEXT_FONT_PATH), size)


@lru_cache(maxsize=120)
def _digit(char: str, weight: int, width: int) -> Image.Image:
    """Cache large, tightly cropped glyphs, with a bounded set of font variants."""
    font = _font(128, weight, width)
    box = font.getbbox(char)
    mask = Image.new("L", (box[2] - box[0], box[3] - box[1]))
    ImageDraw.Draw(mask).text((-box[0], -box[1]), char, font=font, fill=255)
    return mask


def _validate_number(number: str) -> None:
    if len(number) not in (7, 8) or any(char not in "0123456789" for char in number):
        raise ValueError("An Israeli civilian plate label must contain exactly 7 or 8 ASCII digits")


def random_number(rng: Rng, eight_share: float = 0.55) -> str:
    """Keep leading zeroes: numbers are strings throughout rendering and CSV writing."""
    length = 8 if rng.random() < eight_share else 7
    return "".join(rng.choice("0123456789") for _ in range(length))


def plate_text(number: str, rng: Rng, separator: str | None = None) -> str:
    """The number as printed: 123-45-678 or 12-345-67."""
    _validate_number(number)
    sep = SEPARATORS[separator] if separator is not None else rng.choice(("-", "\u00b7", ":", "-", "\u00b7"))
    first = 3 if len(number) == 8 else 2
    return f"{number[:first]}{sep}{number[first:5]}{sep}{number[5:]}"


def _flag(img: Image.Image, box: tuple[int, int, int, int]) -> None:
    """White Israeli flag, two blue horizontal stripes and a six-point Star of David."""
    flag = Image.new("RGB", (160, 112), (246, 246, 240))
    draw = ImageDraw.Draw(flag)
    blue = (23, 69, 158)
    for top in (14, 88):
        draw.rectangle((0, top, 159, top + 10), fill=blue)
    draw.line(((80, 33), (58, 72), (102, 72), (80, 33)), fill=blue, width=4)
    draw.line(((80, 80), (58, 41), (102, 41), (80, 80)), fill=blue, width=4)
    x, y, w, h = box
    img.paste(flag.resize((w, h), Image.Resampling.LANCZOS), (x, y))


def _rtl(text: str) -> str:
    """Display our fixed country wording even in Windows Pillow builds without RAQM."""
    if HAS_RAQM:
        return text
    if text == "\u0625\u0633\u0631\u0627\u0626\u064a\u0644":
        # Contextual Arabic forms for this fixed word only, then visual right-to-left order.
        # Alef and reh break joining; seen/yeh-with-hamza start their following connections.
        text = "\ufe87\ufeb3\ufeae\ufe8d\ufe8b\ufef4\ufede"
    return text[::-1]


def _country_names(draw: ImageDraw.ImageDraw, x: int, y: int, size: int, fill) -> None:
    # Country names are decoration, never characters in the registration number.
    draw.text((x, y), _rtl("\u05d9\u05e9\u05e8\u05d0\u05dc"), font=_text_font(size), fill=fill, anchor="mt")
    draw.text((x, y + size), _rtl("\u0625\u0633\u0631\u0627\u0626\u064a\u0644"),
              font=_text_font(size), fill=fill, anchor="mt")


def _badge(img: Image.Image, style: PlateStyle, box: tuple[int, int, int, int], horizontal: bool) -> None:
    x, y, w, h = box
    draw = ImageDraw.Draw(img)
    draw.rectangle((x, y, x + w, y + h), fill=(27, 75, 158))
    white = (244, 244, 234)
    if horizontal:
        if style.has_flag:
            _flag(img, (x + 5, y + 5, int(w * 0.43), int(h * 0.48)))
        if style.has_il:
            draw.text((x + w * 0.75, y + h * 0.32), "IL",
                      font=_font(max(12, int(h * 0.52)), 600, 100), fill=white, anchor="mm")
            if style.country_text:
                _country_names(draw, int(x + w * 0.72), int(y + h * 0.57), max(6, h // 7), white)
    else:
        if style.has_flag:
            _flag(img, (x + 4, y + 5, w - 8, int((w - 8) * 0.7)))
        if style.has_il:
            draw.text((x + w / 2, y + h * 0.56), "IL", font=_font(int(w * 0.72), 600, 100),
                      fill=white, anchor="mm")
            if style.country_text:
                _country_names(draw, x + w // 2, int(y + h * 0.74), max(6, w // 5), white)


def _draw_number(img: Image.Image, text: str, box: tuple[int, int, int, int],
                 rng: Rng, style: PlateStyle, weight: int, stretch: int) -> None:
    """Place each digit on a shared baseline; separators do not shrink digit height."""
    x, y, room, height = box
    cell, gap = height * rng.uniform(0.44, 0.49), height * rng.uniform(0.055, 0.095)
    sep_width = height * 0.19
    lengths = [cell if char in "0123456789" else sep_width for char in text]
    full_width = sum(lengths) + gap * (len(text) - 1)
    scale = min(1.0, room / full_width) if style.rows == 2 else room / full_width
    x += (room - full_width * scale) / 2
    draw = ImageDraw.Draw(img)
    ink = (rng.randint(13, 36),) * 3
    triangle_drawn = False
    for char, advance in zip(text, lengths):
        if char in "0123456789":
            glyph = _digit(char, weight, stretch)
            # Narrow '1' retains its true width inside an equal-width digit cell.
            reference_width = _digit("0", weight, stretch).width
            glyph_width = max(3, round(cell * scale * glyph.width / reference_width))
            mask = glyph.resize((glyph_width, height), Image.Resampling.LANCZOS)
            img.paste(ink, (round(x + (cell * scale - glyph_width) / 2), y), mask)
        else:
            cx, cy = x + advance * scale / 2, y + height * 0.5
            radius = max(1, round(height * 0.026))
            if char == "-":
                draw.rectangle((cx - advance * scale * 0.38, cy - radius,
                                cx + advance * scale * 0.38, cy + radius), fill=ink)
            else:
                centers = [cy] if char == "\u00b7" else [cy - height * 0.09, cy + height * 0.09]
                for center in centers:
                    draw.ellipse((cx - radius, center - radius, cx + radius, center + radius), fill=ink)
            if style.triangle and not triangle_drawn:
                # The inspection mark sits below a separator, inside its gap.
                ty, r = y + height * 0.81, max(3, height * 0.047 * scale)
                draw.line(((cx, ty - r), (cx - r, ty + r), (cx + r, ty + r), (cx, ty - r)),
                          fill=ink, width=1)
                triangle_drawn = True
        x += (advance + gap) * scale


def render_plate(number: str, rng: Rng, plate_type: str = "long_strip",
                 *, style: PlateStyle | None = None) -> np.ndarray:
    """A front-on plate in BGR. Optional style is also the source of manifest fields."""
    _validate_number(number)
    style = choose_style(plate_type, rng) if style is None else style
    width, height = PLATE_SIZES[style.plate_type]
    old = style.plate_type.endswith("_plain")
    hsv = np.uint8([[[rng.uniform(18, 27), rng.uniform(135 if old else 185, 245), rng.uniform(211, 253)]]])
    body = tuple(int(c) for c in cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)[0, 0])
    img = Image.new("RGB", (width, height), (35, 35, 34))
    draw = ImageDraw.Draw(img)
    draw.rounded_rectangle((2, 2, width - 3, height - 3), radius=8, fill=body,
                           outline=(38, 36, 29), width=rng.choice((1, 2, 2, 3)))
    left = 16
    if style.plate_type in ("long_strip", "short_strip"):
        strip = 39 if height == 112 else 44
        _badge(img, style, (6, 6, strip, height - 13), horizontal=False)
        left = strip + 15
    elif style.plate_type in ("short_badge", "two_row") and (style.has_flag or style.has_il):
        badge_w, badge_h = (94, 47) if style.rows == 1 else (72, 30)
        bx = 24 if style.badge_left else (width - badge_w) // 2
        _badge(img, style, (bx, height - badge_h - 12, badge_w, badge_h), horizontal=True)

    weight, stretch = rng.choice((500, 560, 620, 680)), rng.choice((75, 80, 85))
    text = plate_text(number, rng, style.separator)
    if style.rows == 2:
        # Wrap at a group boundary, preserving 2-3-2 / 3-2-3 and row-major labels.
        first = 3 if len(number) == 8 else 2
        sep = SEPARATORS[style.separator]
        lines = (f"{number[:first]}{sep}{number[first:5]}", number[5:])
        for line, top in zip(lines, (18, 118)):
            _draw_number(img, line, (16, top, width - 32, 88), rng, style, weight, stretch)
    else:
        digit_height = rng.randint(77, 86) if height == 112 else rng.randint(135, 155)
        top = rng.randint(10, 13) if height == 112 else rng.randint(30, 41)
        if style.plate_type == "short_plain":
            top += 10
        _draw_number(img, text, (left, top, width - left - 16, digit_height),
                     rng, style, weight, stretch)

    if style.small_print:
        # Actual country wording instead of the old random Latin placeholder letters.
        size = 6 if height == 112 else 9
        pos = (int(width * 0.68), height - (13 if height == 112 else 25))
        draw.text(pos, _rtl("\u05ea\u05d5\u05e6\u05e8\u05ea \u05d9\u05e9\u05e8\u05d0\u05dc"),
                  font=_text_font(size), fill=(59, 55, 37), anchor="mt")
    if rng.random() < 0.84:
        # Mounting screws stay above/below the number so they cannot replace a digit.
        screw_y = 7 if rng.random() < 0.65 else height - 8
        for sx in (int(width * 0.22), int(width * 0.79)):
            r = 2 if height == 112 else 3
            draw.ellipse((sx - r, screw_y - r, sx + r, screw_y + r), fill=(144, 141, 123),
                         outline=(55, 53, 44))
            draw.line((sx - r + 1, screw_y, sx + r - 1, screw_y), fill=(45, 45, 40))
    if rng.random() < 0.40:
        # Plastic holder lip, not a thick printed plate border.
        draw.rounded_rectangle((0, 0, width - 1, height - 1), radius=8,
                               outline=(22, 23, 22), width=3)
        draw.rectangle((4, height - 5, width - 5, height - 1), fill=(26, 27, 26))
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


def augment(img: np.ndarray, rng: Rng, strength: float = 1.0) -> np.ndarray:
    """Readable CPU camera effects, also usable on real crops at lower strength.

    Long crops retain >=28 pixels of plate height; tall crops retain >=180 pixels of
    width. Blur is budgeted against the resulting digit-cell width, never stacked with
    a second blur. Strength zero is an exact identity.
    """
    if not math.isfinite(strength) or not 0 <= strength <= 1:
        raise ValueError("strength must be finite and between 0 and 1")
    if strength == 0:
        return img.copy()
    h, w = img.shape[:2]
    # Wear changes contrast gently. It never erases or covers a whole digit stroke.
    pixels = img.astype(np.float32)
    yy, xx = np.mgrid[:h, :w].astype(np.float32)
    shade = 1 + strength * rng.uniform(-0.16, 0.16) * (xx / w - 0.5)
    shade = shade + strength * rng.uniform(-0.12, 0.12) * (yy / h - 0.5)
    if rng.random() < 0.35:
        dirt = cv2.resize(rng.np.uniform(0.87, 1.0, (5, 9)).astype(np.float32), (w, h))
        shade *= 1 - strength * (1 - dirt)
    pixels *= shade[:, :, None]
    if rng.random() < 0.30:
        reflection = np.exp(-((xx / w - rng.uniform(0.1, 0.9)) / 0.14) ** 2)
        alpha = reflection[:, :, None] * rng.uniform(0.06, 0.18) * strength
        pixels = pixels * (1 - alpha) + 244 * alpha
    if rng.random() < 0.30:
        # Faded yellow and worn ink, bounded to retain foreground/background contrast.
        pixels = pixels * (1 - 0.12 * strength) + np.float32([125, 191, 216]) * 0.12 * strength
    img = np.clip(pixels, 0, 255).astype(np.uint8)

    # Project all four corners inside the canvas; no accidental clipping of end digits.
    pad_x, pad_y = int(w * rng.uniform(0.005, 0.05) * strength), int(h * rng.uniform(0.01, 0.06) * strength)
    canvas_w, canvas_h = w + 2 * pad_x, h + 2 * pad_y
    jx, jy = w * 0.035 * strength, h * 0.045 * strength
    src = np.float32([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]])
    dst = np.float32([
        [pad_x + rng.uniform(0, jx), pad_y + rng.uniform(0, jy)],
        [pad_x + w - 1 - rng.uniform(0, jx), pad_y + rng.uniform(0, jy)],
        [pad_x + w - 1 - rng.uniform(0, jx), pad_y + h - 1 - rng.uniform(0, jy)],
        [pad_x + rng.uniform(0, jx), pad_y + h - 1 - rng.uniform(0, jy)],
    ])
    background = rng.randint(22, 85)
    img = cv2.warpPerspective(img, cv2.getPerspectiveTransform(src, dst), (canvas_w, canvas_h),
                              borderMode=cv2.BORDER_CONSTANT, borderValue=(background,) * 3)
    minimum = max(180, math.ceil(28 * w / h))
    sampled_w = rng.randint(minimum, max(minimum, 340))
    target_w = round(canvas_w * (1 - strength) + sampled_w * strength)
    target_h = max(28, round(target_w * canvas_h / canvas_w))
    img = cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_AREA)
    # At the lowest resolution, a cell is ~16px wide; sigma then stays below 0.5px.
    cell_width = target_w * 0.075
    blur_budget = min(0.85, max(0.30, cell_width * 0.034)) * strength
    if rng.random() < 0.48 * strength:
        sigma = rng.uniform(0.20, blur_budget) if blur_budget > 0.20 else blur_budget
        if rng.random() < 0.75:
            img = cv2.GaussianBlur(img, (3, 3), sigma)
        else:
            side = min(0.20, blur_budget * 0.23)
            img = cv2.filter2D(img, -1, np.float32([[side, 1 - 2 * side, side]]))

    pixels = img.astype(np.float32) * (1 + rng.uniform(-0.32, 0.12) * strength)
    pixels += rng.uniform(-10, 10) * strength
    if rng.random() < 0.32 * strength:
        cast = rng.choice(((0.82, 0.94, 1.08), (1.08, 0.97, 0.88)))
        pixels *= 1 + (np.float32(cast) - 1) * strength
    pixels += rng.normal(0, rng.uniform(0.3, 3.2) * strength, pixels.shape)
    img = np.clip(pixels, 0, 255).astype(np.uint8)
    if rng.random() < 0.5 * strength:
        ok, encoded = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, rng.randint(72, 95)])
        if not ok:
            raise OSError("JPEG encoding failed")
        img = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    return img


def contact_sheet(images: list[np.ndarray], labels: list[str], columns: int = 6) -> np.ndarray:
    """A grid that preserves every plate's aspect ratio, including two-row plates."""
    if not images or len(images) != len(labels) or columns < 1:
        raise ValueError("A contact sheet needs paired images/labels and at least one column")
    tiles = []
    for img, label in zip(images, labels):
        tile = np.full((156, 246, 3), (24, 24, 28), np.uint8)
        h, w = img.shape[:2]
        scale = min(236 / w, 120 / h)
        resized = cv2.resize(img, (max(1, round(w * scale)), max(1, round(h * scale))),
                             interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC)
        rh, rw = resized.shape[:2]
        tile[4 + (120 - rh) // 2:4 + (120 - rh) // 2 + rh, (246 - rw) // 2:(246 - rw) // 2 + rw] = resized
        cv2.putText(tile, label, (5, 143), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (235, 235, 235), 1, cv2.LINE_AA)
        tiles.append(tile)
    while len(tiles) % columns:
        tiles.append(np.full_like(tiles[0], 11))
    return np.vstack([np.hstack(tiles[i:i + columns]) for i in range(0, len(tiles), columns)])


def _write_image(path: Path, img: np.ndarray) -> None:
    if not cv2.imwrite(str(path), img):
        raise OSError(f"Could not write image: {path}")


def generate(count: int, name: str, seed: int, clean_share: float, preview: bool,
             types: list[str] | None = None, weights: dict[str, float] | None = None,
             eight_share: float = 0.55) -> Path:
    """Write a new run; never reuse an existing directory, even an empty one."""
    if count <= 0:
        raise ValueError("--count must be positive")
    if not all(math.isfinite(v) and 0 <= v <= 1 for v in (clean_share, eight_share)):
        raise ValueError("Clean and eight-digit shares must be finite and between 0 and 1")
    if not name or name in (".", "..") or any(c not in "abcdefghijklmnopqrstuvwxyz"
                                            "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in name):
        raise ValueError("--name must be a single folder name using letters, digits, '_' or '-'")
    mix = type_mix(types, weights)
    if not FONT_PATH.is_file() or not TEXT_FONT_PATH.is_file():
        raise FileNotFoundError("Rendering requires the installed Windows Bahnschrift and Arial fonts")
    out_dir = OUT_ROOT / name
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(exist_ok=False)  # Atomic refusal, including concurrent writers.
    rng = Rng(seed)
    sheet_images, sheet_labels = [], []
    with (out_dir / "labels.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(LABEL_COLUMNS)
        for index in range(count):
            plate_type = choose_type(rng, mix)
            number = random_number(rng, eight_share)
            style = choose_style(plate_type, rng)
            clean = rng.random() < clean_share
            img = render_plate(number, rng, style=style)
            if not clean:
                img = augment(img, rng)
            filename = f"{number}_{index:06d}.png"
            _write_image(out_dir / filename, img)
            writer.writerow((filename, number, plate_type, len(number), style.rows,
                             int(style.has_flag), int(style.has_il), style.separator, int(clean)))
            if preview and len(sheet_images) < 96:
                sheet_images.append(img)
                sheet_labels.append(f"{number} {plate_type}" + (" clean" if clean else ""))
            if count >= 1000 and (index + 1) % 2000 == 0:
                print(f"{index + 1}/{count} plates", flush=True)
    if preview:
        _write_image(out_dir / "preview.png", contact_sheet(sheet_images, sheet_labels))
    return out_dir


def parse_args() -> argparse.Namespace:
    """Read the existing CLI plus layout selection and mixture controls."""
    parser = argparse.ArgumentParser(description="Generate synthetic Israeli plate crops with exact labels.")
    parser.add_argument("--count", type=int, default=48, help="How many plates to write.")
    parser.add_argument("--name", required=True, help="New run folder under data/synthetic/plates/.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed (same seed, same plates).")
    parser.add_argument("--clean-share", type=float, default=0.1, help="Share without camera/wear augmentation.")
    parser.add_argument("--preview", action="store_true", help="Also write preview.png (first 96 plates).")
    parser.add_argument("--types", nargs="+", choices=tuple(PLATE_SIZES), help="Only render these layouts.")
    parser.add_argument("--type-mix", nargs="+", metavar="TYPE=WEIGHT",
                        help="Relative shares; defaults to 55 12 7 8 5 8.")
    parser.add_argument("--eight-share", type=float, default=0.55, help="Share of eight-digit numbers (0 to 1).")
    args = parser.parse_args()
    args.weights = None
    if args.type_mix is not None:
        args.weights = {}
        for item in args.type_mix:
            try:
                name, value = item.split("=", 1)
                if name in args.weights:
                    raise ValueError("duplicate type")
                args.weights[name] = float(value)
            except ValueError:
                parser.error(f"Invalid type weight: {item}; use distinct TYPE=WEIGHT pairs")
    try:
        type_mix(args.types, args.weights)
    except ValueError as exc:
        parser.error(str(exc))
    return args


def main() -> None:
    """Generate one run of synthetic plates."""
    args = parse_args()
    out_dir = generate(args.count, args.name, args.seed, args.clean_share, args.preview,
                       args.types, args.weights, args.eight_share)
    print(f"{args.count} plates -> {out_dir.relative_to(PROJECT_ROOT).as_posix()}")


if __name__ == "__main__":
    main()
