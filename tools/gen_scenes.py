"""Plan a balanced batch of synthetic vehicle scenes, drawn free in the Gemini app.

Each scene is described by us before it is drawn — camera, place, time, weather, vehicle
make/model/body/colour, plate type — so the manifest is exact for everything we asked for.
The plates are drawn BLANK; real numbers are pasted on later and every plate box is measured
by our detector (see docs/DATASETS.md: image models' own estimates of plate geometry are not
reliable). Conditions are balanced: cameras, places, times, weathers
and traffic are dealt round-robin from shuffled decks; vehicles and colours follow
real-traffic shares.

The images are made in the Gemini app, one image per request (no paid API is used). Three
steps per batch:

    python tools/gen_scenes.py --name b01 --count 10 --app-prompt   # 1. writes + prints the message
    #   2. paste the message into the Gemini app, then download every image (they land in Downloads)
    python tools/gen_scenes.py --name b01 --count 10 --import       # 3. Downloads -> the batch folder

Images are matched to scenes in download order, so download them in the order shown. If the app
stops early, import what you have, run --app-prompt again (it lists only the missing scenes),
and repeat. Output: data/synthetic/scenes/<name>/scene_<name>_<nnn>.jpg + manifest.jsonl.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from common import PROJECT_ROOT  # noqa: E402

OUT_ROOT = PROJECT_ROOT / "data" / "synthetic" / "scenes"
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp"}

BRIEF = (
    "Photorealistic photo or CCTV/dashcam frame of real vehicles on Israeli roads, for training a "
    "licence-plate detector. Must look like an unedited real photo: real car models with correct "
    "proportions and wheel count, believable reflections and shadows, real Israeli road markings and "
    "kerbs, no CGI look, no merged or melted geometry. Every licence plate is an Israeli-style plate "
    "whose yellow area is completely BLANK: no digits, no letters, no logos, no stickers. Plate fully "
    "visible and facing the camera at a readable angle (0-45 degrees), roughly 3-13% of the image "
    "width - no close-up. No identifiable faces, no watermarks, no captions, no foreign plates. "
    "Landscape 16:9."
)
CAMERAS = {
    "gate": ("fixed gate camera about 1.5 m high facing the oncoming vehicle", "front"),
    "pole": ("parking-lot pole camera about 3 m high looking down at the vehicle", "front"),
    "dashcam": ("dashcam view from the car behind", "rear"),
    "low": ("low camera about 0.8 m high", "rear"),
}
PLACES = ["residential street", "highway", "gated-community entrance with a barrier", "supermarket parking lot",
          "underground car park", "kibbutz road", "city centre street", "petrol station", "school drop-off area",
          "moshav entrance road"]
TIMES = ["morning", "noon", "golden hour", "dusk", "night under street lights", "night, oncoming headlights"]
DAYTIME = {"morning", "noon", "golden hour"}  # low sun glare only happens in these
WEATHERS = ["clear", "overcast", "rain, wet road with reflections", "light fog", "dust haze (sharav)",
            "low winter sun with glare"]
# (make, model, body, plate type, weight). Short plates: cars with a small US-style plate
# recess. Motorcycles and scooters: a two-row plate, rear only.
VEHICLES = [
    ("Kia", "Picanto", "hatchback", "long", 6), ("Kia", "Sportage", "suv", "long", 6),
    ("Kia", "Niro", "suv", "long", 4), ("Hyundai", "i10", "hatchback", "long", 5),
    ("Hyundai", "i20", "hatchback", "long", 5), ("Hyundai", "Tucson", "suv", "long", 6),
    ("Hyundai", "Ioniq 5", "suv", "long", 3), ("Toyota", "Corolla", "sedan", "long", 7),
    ("Toyota", "Yaris", "hatchback", "long", 4), ("Toyota", "C-HR", "suv", "long", 3),
    ("Toyota", "RAV4", "suv", "long", 4), ("Skoda", "Octavia", "sedan", "long", 5),
    ("Skoda", "Kodiaq", "suv", "long", 3), ("Mazda", "3", "sedan", "long", 4),
    ("Mazda", "CX-5", "suv", "long", 3), ("Mitsubishi", "Outlander", "suv", "long", 3),
    ("Suzuki", "Swift", "hatchback", "long", 3), ("Tesla", "Model 3", "sedan", "long", 3),
    ("Tesla", "Model Y", "suv", "long", 3), ("BYD", "Atto 3", "suv", "long", 3),
    ("Geely", "Geometry C", "hatchback", "long", 2), ("Chery", "Tiggo 7", "suv", "long", 2),
    ("Peugeot", "208", "hatchback", "long", 2), ("Volkswagen", "Golf", "hatchback", "long", 2),
    ("Isuzu", "D-Max", "pickup", "long", 3), ("Ford", "Transit", "van", "long", 2),
    ("Mercedes", "Sprinter", "van", "long", 2), ("Jeep", "Grand Cherokee", "suv", "short", 2),
    ("Ford", "Mustang", "sedan", "short", 1), ("Chevrolet", "Corvette", "sedan", "short", 1),
    ("Honda", "PCX", "scooter", "two_row", 3), ("Yamaha", "MT-07", "motorcycle", "two_row", 2),
]
COLORS = [("white", 30), ("silver", 13), ("grey", 12), ("black", 15), ("blue", 10), ("red", 8),
          ("green", 3), ("brown", 2), ("beige", 3), ("yellow", 2), ("orange", 2)]
PLATE_TEXT = {
    "long": "a long Israeli plate (~4.7:1, narrow blue IL strip on the left), blank yellow",
    "short": "a short Israeli plate (~2:1, small blue IL block), blank yellow",
    "two_row": "a small near-square Israeli motorcycle plate, blank yellow",
}
# Extra vehicles in ~70% of scenes: several plates per frame (small, far, partly hidden ones
# included) are what a real camera sees, and what multi-plate detection and tracking train on.
TRAFFIC = ["", "", "",
           "one more vehicle further away, its plate blank too",
           "two more vehicles further away, their plates blank too",
           "two more vehicles further away, their plates blank too",
           "three parked cars in the background, their plates blank too",
           "one more vehicle close behind or beside, partly hiding a plate, plates blank",
           "busy traffic: four to six more vehicles at different distances, all plates blank",
           "busy traffic: four to six more vehicles at different distances, all plates blank"]


def article(word: str) -> str:
    """'a' or 'an' for the word that follows."""
    return "an" if word[0] in "aeiou" else "a"


class Deck:
    """Deal items round-robin from a reshuffled deck, so every option comes up equally often."""

    def __init__(self, items: list, rng: random.Random) -> None:
        self.items, self.rng, self.pile = list(items), rng, []

    def draw(self):
        if not self.pile:
            self.pile = self.items[:]
            self.rng.shuffle(self.pile)
        return self.pile.pop()


def plan_scenes(name: str, count: int, seed: int) -> list[dict]:
    """The full, balanced list of scenes for a batch (deterministic for a given seed)."""
    rng = random.Random(seed)
    cameras, places, times, weathers, traffic = (Deck(list(CAMERAS), rng), Deck(PLACES, rng), Deck(TIMES, rng),
                                                 Deck(WEATHERS, rng), Deck(TRAFFIC, rng))
    scenes = []
    for index in range(1, count + 1):
        make, model, body, plate_type, _ = rng.choices(VEHICLES, weights=[v[4] for v in VEHICLES])[0]
        camera = cameras.draw()
        side = CAMERAS[camera][1]
        if plate_type == "two_row":  # motorcycles carry a rear plate only
            camera, side = rng.choice(["dashcam", "low", "pole"]), "rear"
        place, when, weather = places.draw(), times.draw(), weathers.draw()
        if weather == "low winter sun with glare" and when not in DAYTIME:
            when = rng.choice(sorted(DAYTIME))  # no sun at dusk or night
        if place == "underground car park":
            when, weather = "under fluorescent lights", "indoor"
        color = rng.choices([c for c, _ in COLORS], weights=[w for _, w in COLORS])[0]
        extra = traffic.draw()
        vehicle = f"{article(color)} {color} {make} {model}"
        scene = (f"{CAMERAS[camera][0]}, {place}, {when}, {weather}. Main vehicle: {vehicle}, "
                 f"{side} plate visible: {PLATE_TEXT[plate_type]}." + (f" Also: {extra}." if extra else ""))
        scenes.append({
            "file": f"scene_{name}_{index:03d}.jpg", "camera": camera, "place": place, "time": when,
            "weather": weather,
            "vehicles": [{"make": make, "model": model, "body": body, "color": color, "plate_type": plate_type,
                          "plate_side": side}],
            "scene": scene, "source": "Gemini app (Nano Banana)", "geometry": "to be measured by our detector",
        })
    return scenes


def app_message(scenes: list[dict]) -> str:
    """One message for the Gemini app asking for every listed scene in a single reply."""
    lines = [f"{i}. {s['file']}: {s['scene']}" for i, s in enumerate(scenes, start=1)]
    request = (f"Generate the following {len(scenes)} images, one after another, all in this reply, in exactly "
               "this order. Under each image write only its file name. If you must stop before the end, tell me "
               'the last file name you made and I will say "continue".')
    return "\n\n".join([BRIEF, request, "\n".join(lines)])


def import_downloads(scenes: list[dict], out_dir: Path, source: Path, since: float) -> int:
    """Copy images downloaded after `since` from `source` into the batch as JPEG, in download
    order, filling the scenes that have no image yet. Returns how many were imported."""
    todo = [s for s in scenes if not (out_dir / s["file"]).exists()]
    found = sorted((p for p in source.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES and p.stat().st_mtime > since),
                   key=lambda p: p.stat().st_mtime)
    imported = 0
    for scene, path in zip(todo, found):
        image = cv2.imread(str(path))
        if image is None:
            print(f"  skipped {path.name}: not a readable image")
            continue
        cv2.imwrite(str(out_dir / scene["file"]), image, [cv2.IMWRITE_JPEG_QUALITY, 95])
        print(f"  {path.name}  ->  {scene['file']}")
        imported += 1
    return imported


def parse_args() -> argparse.Namespace:
    """Read the command-line options."""
    parser = argparse.ArgumentParser(description="Plan a batch of synthetic vehicle scenes for the Gemini app.")
    parser.add_argument("--name", required=True, help="Batch name, e.g. b01 (folder under data/synthetic/scenes/).")
    parser.add_argument("--count", type=int, default=10, help="Scenes in the batch.")
    parser.add_argument("--seed", type=int, default=0, help="Seed for the scene plan (same seed, same plan).")
    step = parser.add_mutually_exclusive_group(required=True)
    step.add_argument("--app-prompt", action="store_true", help="Write and print the Gemini-app message.")
    step.add_argument("--import", dest="import_", action="store_true",
                      help="Import the images downloaded since --app-prompt, in download order.")
    parser.add_argument("--from", dest="source", type=Path, default=Path.home() / "Downloads",
                        help="Folder the app's images were downloaded to (default: Downloads).")
    return parser.parse_args()


def main() -> None:
    """Plan the batch and write its manifest, then print the app message or import the images."""
    args = parse_args()
    out_dir = OUT_ROOT / args.name
    out_dir.mkdir(parents=True, exist_ok=True)
    scenes = plan_scenes(args.name, args.count, args.seed)
    with (out_dir / "manifest.jsonl").open("w", encoding="utf-8") as file:
        for scene in scenes:
            file.write(json.dumps(scene, ensure_ascii=False) + "\n")
    where = out_dir.relative_to(PROJECT_ROOT).as_posix()

    if args.app_prompt:
        message = app_message([s for s in scenes if not (out_dir / s["file"]).exists()])
        (out_dir / "app_prompt.txt").write_text(message, encoding="utf-8")  # its time marks where downloads start
        print(message)
        print(f"\n(also saved to {where}/app_prompt.txt)")
        return

    marker = out_dir / "app_prompt.txt"
    if not marker.exists():
        raise SystemExit("Run --app-prompt first: its time marks which downloads belong to this batch.")
    imported = import_downloads(scenes, out_dir, args.source, marker.stat().st_mtime)
    left = sum(not (out_dir / s["file"]).exists() for s in scenes)
    print(f"Imported {imported} image(s) into {where}; {left} scene(s) still without an image"
          + (" — run --app-prompt again for those." if left else "."))


if __name__ == "__main__":
    main()
