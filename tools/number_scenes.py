"""Print a realistic Israeli number (and strip) onto every blank plate of the generated scenes.

One-time data-prep helper. The scene generator draws Israeli plates blank on purpose (image
models scramble digits), and the scene manifest records each plate's corners. Here every long
plate gets:

  * a number in the real format, as seen on every real plate we have: 8 digits `123:45-601`
    (the 7th digit is always 0, the last 1-4) or 7 digits `12:345-67`, about half each; a colon
    after the first group, a dash or dot after the second, often the small inspection triangle
    under the colon; synth_plates.py's condensed plate font;
  * the blue strip's contents, as on real plates: IL with the flag above it (most), IL only, or
    nothing (older plates).

Nothing is pasted on top: the plate is rectified from its corners, the digits are printed by
*darkening* the blank plate where the ink goes, and the strip's white flag and letters are lit
by the strip's own measured colour and shading — so both inherit the scene's light, reflections,
blur and night colour. Two-row, short and partly hidden plates stay blank ("blank" in the
manifest); their boxes stay labelled, since they are plates.

    python tools/number_scenes.py data/synthetic/scenes/c01 [--preview]

writes <scenes>_numbered/:
    images/, labels/      the scenes with numbers; YOLO labels unchanged (same boxes)
    crops/<number>_<scene>_<k>.png   numbered plates at least 90 px wide, for the reader
    manifest.jsonl        the original manifest; each plate also gets "number", "number_status",
                          "strip", "separators", and "vehicle" (make, model, generation, body,
                          colour) copied from its vehicle, for later colour/make/model work
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from synth_plates import PLATE_SIZES, PlateStyle, Rng, _badge, _draw_number  # noqa: E402

CANVAS_W, CANVAS_H = PLATE_SIZES["long_strip"]  # the plate, rectified: 520 x 112
INK = 0.12  # printed digits reflect ~12% of the light the yellow does
STRIP_BLUE = np.array([158, 75, 27], np.float32)  # BGR of synth_plates' strip blue
MIN_STRIP = 36  # canvas columns; a real strip is ~39 of the plate's 520
MIN_CROP_WIDTH = 90  # px; the app does not read narrower plates (pipeline.MIN_READ_WIDTH)
VEHICLE_FIELDS = ("make", "model", "generation", "body", "view", "color", "color_detail", "finish",
                  "two_tone", "second_color")


def real_number(rng: Rng, used: set[str]) -> str:
    """A number in the real format, never repeated in one batch."""
    while True:
        if rng.random() < 0.55:  # 8 digits: 3-2-3, 7th digit 0, last digit 1-4
            last = rng.py.choices("1234", weights=(50, 12, 20, 18))[0]
            number = f"{rng.randint(100, 999)}{rng.randint(0, 99):02d}{rng.randint(0, 9)}0{last}"
        else:  # 7 digits: 2-3-2
            number = f"{rng.randint(10, 99)}{rng.randint(0, 99999):05d}"
        if number not in used:
            used.add(number)
            return number


def number_ink(number: str, start: int, rng: Rng) -> tuple[np.ndarray, str]:
    """The printed number as ink coverage 0..1 on the rectified plate, right of column `start`."""
    first = 3 if len(number) == 8 else 2
    second = rng.choice(("-", "·"))
    text = f"{number[:first]}:{number[first:5]}{second}{number[5:]}"
    style = PlateStyle("long_strip", True, True, "colon", rng.random() < 0.55, False, False, False)
    height = round(CANVAS_H * rng.uniform(0.62, 0.68))  # of the yellow face, a bit less than the plate
    # As on a real plate, the number spans the face from just after the strip to the right edge.
    room = CANVAS_W - 14 - (start + 8)
    canvas = Image.new("RGB", (CANVAS_W, CANVAS_H), (255, 255, 255))
    _draw_number(canvas, text, (start + 8, (CANVAS_H - height) // 2, room, height), rng, style,
                 rng.choice((560, 620, 680)), rng.choice((75, 80, 85)))
    grey = np.asarray(canvas.convert("L"), np.float32)
    return np.clip((255 - grey) / 220, 0, 1), f":{second}"


def strip_end(canvas: np.ndarray) -> int:
    """Where the blue IL strip ends, in canvas columns (the digits start after it)."""
    hsv = cv2.cvtColor(canvas, cv2.COLOR_BGR2HSV)
    blue = ((hsv[..., 0] > 95) & (hsv[..., 0] < 135) & (hsv[..., 1] > 70)).mean(axis=0)
    columns = np.flatnonzero(blue[: CANVAS_W // 3] > 0.35)
    return int(columns.max()) + 1 if columns.size else 0


def strip_light(canvas: np.ndarray, end: int) -> np.ndarray | None:
    """The strip's light per channel (its measured blue / the rendered blue), or None when the
    strip is too small or too blurred into the yellow to measure — then it is left as it is."""
    if end < 12:
        return None
    region = canvas[6:CANVAS_H - 6, 2:end - 1]
    hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
    blue = (hsv[..., 0] > 95) & (hsv[..., 0] < 135) & (hsv[..., 1] > 70)
    if blue.mean() < 0.6:
        return None
    ratio = np.median(region[blue].astype(np.float32), axis=0) / STRIP_BLUE
    # Brightness in full, colour cast only partly: a full per-channel ratio turns white pink.
    brightness = float(ratio.mean())
    return np.clip(brightness * (1 + 0.3 * (ratio / brightness - 1)), 0.08, 1.6)


def light_strip(canvas: np.ndarray, end: int, kind: str, gain: np.ndarray) -> np.ndarray:
    """The rectified plate with the strip's flag and/or IL drawn in, lit like the strip itself."""
    grey = cv2.cvtColor(canvas[:, :end], cv2.COLOR_BGR2GRAY).astype(np.float32)
    shading = cv2.GaussianBlur(grey, (0, 0), 3) / max(1.0, float(np.median(grey)))
    badge = Image.new("RGB", (end, CANVAS_H), tuple(int(c) for c in STRIP_BLUE[::-1]))
    style = PlateStyle("long_strip", kind == "flag_il", True, "colon", False, False, False, False)
    _badge(badge, style, (1, 5, end - 3, CANVAS_H - 11), horizontal=False)
    lit = cv2.cvtColor(np.asarray(badge), cv2.COLOR_RGB2BGR).astype(np.float32) * gain * shading[..., None]
    out = canvas.copy()
    out[:, :end] = np.clip(lit, 0, 255).astype(np.uint8)
    return out


def print_plate(scene: np.ndarray, corners, number: str, strip: str, rng: Rng) -> tuple[np.ndarray, str, str]:
    """Print the strip and the number onto the blank plate at `corners` in `scene`; returns the
    scene, the separators used and the strip actually drawn ("plain" when it could not be lit)."""
    quad = np.array(corners, np.float32)
    canvas_quad = np.array([[0, 0], [CANVAS_W, 0], [CANVAS_W, CANVAS_H], [0, CANVAS_H]], np.float32)
    to_canvas = cv2.getPerspectiveTransform(quad, canvas_quad)
    to_scene = np.linalg.inv(to_canvas)
    height, width = scene.shape[:2]
    canvas = cv2.warpPerspective(scene, to_canvas, (CANVAS_W, CANVAS_H), flags=cv2.INTER_LINEAR)
    end = max(strip_end(canvas), MIN_STRIP)  # every scene plate has a strip, even when too faint to find
    plate_h = float(np.linalg.norm(quad[3] - quad[0]))
    focus = max(0.35, plate_h / 90) * rng.uniform(0.8, 1.3)  # match the scene's own softness

    out = scene.astype(np.float32)
    if strip == "flag_il" and np.linalg.norm(quad[1] - quad[0]) < MIN_CROP_WIDTH:
        strip = "il"  # a flag on a plate this small is a few coloured pixels at best
    gain = strip_light(canvas, end) if strip != "plain" else None
    if gain is None:
        strip = "plain"
    else:
        printed = cv2.warpPerspective(light_strip(canvas, end, strip, gain), to_scene, (width, height),
                                      flags=cv2.INTER_AREA)
        area = np.zeros((CANVAS_H, CANVAS_W), np.float32)
        area[3:CANVAS_H - 3, 1:end - 1] = 1
        area = cv2.GaussianBlur(cv2.warpPerspective(area, to_scene, (width, height), flags=cv2.INTER_AREA),
                                (0, 0), focus)[..., None]
        out = out * (1 - area) + cv2.GaussianBlur(printed, (0, 0), focus).astype(np.float32) * area

    ink, separators = number_ink(number, end, rng)
    ink = cv2.GaussianBlur(cv2.warpPerspective(ink, to_scene, (width, height), flags=cv2.INTER_AREA), (0, 0), focus)
    out = out * (1 - (1 - INK) * ink[..., None])
    grain = np.random.default_rng(rng.randint(0, 1 << 30)).normal(0, 2.0, scene.shape) * (ink[..., None] > 0.05)
    return np.clip(out + grain, 0, 255).astype(np.uint8), separators, strip


def main() -> None:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    parser.add_argument("scenes", type=Path, help="a scene batch folder (images/, labels/, manifest.jsonl)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--preview", type=Path, default=None, help="only write a before/after sheet to this file")
    args = parser.parse_args()

    src = args.scenes.resolve()
    rows = [json.loads(line) for line in (src / "manifest.jsonl").read_text(encoding="utf-8").splitlines() if line]
    rng = Rng(args.seed)
    used: set[str] = set()
    out_dir = src.parent / f"{src.name}_numbered"
    if not args.preview:
        if out_dir.exists():
            raise SystemExit(f"{out_dir} exists; delete it first")
        for sub in ("images", "labels", "crops"):
            (out_dir / sub).mkdir(parents=True)
    tiles = []
    for row in rows:
        scene = np.asarray(cv2.imread(str(src / "images" / row["file"])))
        original = scene.copy()
        stem = Path(row["file"]).stem
        vehicles = {v["index"]: v for v in row["vehicles"]}
        for k, plate in enumerate(row["plates"], start=1):
            vehicle = vehicles.get(plate["vehicle_index"], {})
            plate["vehicle"] = {name: vehicle.get(name) for name in VEHICLE_FIELDS}
            if plate["plate_type"] != "long" or plate.get("occluded"):
                plate.update(number=None, number_status="blank", strip=None, separators=None)
                continue
            number = real_number(rng, used)
            strip = rng.py.choices(("flag_il", "il", "plain"), weights=(60, 30, 10))[0]
            scene, separators, strip = print_plate(scene, plate["corners"], number, strip, rng)
            plate.update(number=number, number_status="printed", strip=strip, separators=separators)
            x1, y1, x2, y2 = plate["bbox_xyxy"]
            if args.preview:
                w, h = x2 - x1, y2 - y1
                box = (slice(max(0, y1 - h), y2 + h), slice(max(0, x1 - w // 4), x2 + w // 4))
                pair = np.hstack([cv2.resize(original[box], (300, 110)), cv2.resize(scene[box], (300, 110))])
                pair = cv2.copyMakeBorder(pair, 0, 20, 2, 2, cv2.BORDER_CONSTANT, value=(30, 30, 30))
                cv2.putText(pair, f"{stem[-3:]} {w}px {strip} -> {number}", (4, 125), cv2.FONT_HERSHEY_SIMPLEX,
                            0.45, (230, 230, 230), 1)
                tiles.append(pair)
            elif x2 - x1 >= MIN_CROP_WIDTH:
                cv2.imwrite(str(out_dir / "crops" / f"{number}_{stem}_{k}.png"), scene[y1:y2, x1:x2])
        if not args.preview:
            cv2.imwrite(str(out_dir / "images" / row["file"]), scene, [cv2.IMWRITE_JPEG_QUALITY, 95])
            shutil.copy(src / "labels" / f"{stem}.txt", out_dir / "labels" / f"{stem}.txt")
    if args.preview:
        while len(tiles) % 2:
            tiles.append(np.zeros_like(tiles[0]))
        cv2.imwrite(str(args.preview), np.vstack([np.hstack(tiles[i:i + 2]) for i in range(0, len(tiles), 2)]))
        print(f"{len(tiles)} plates -> {args.preview}")
        return
    with (out_dir / "manifest.jsonl").open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")
    plates = [p for r in rows for p in r["plates"]]
    printed = sum(p["number_status"] == "printed" for p in plates)
    crops = len(list((out_dir / "crops").glob("*.png")))
    print(f"{len(rows)} scenes: {printed} plates numbered, {len(plates) - printed} left blank, "
          f"{crops} reader crops -> {out_dir}")


if __name__ == "__main__":
    main()
