"""Export the Visus mark and app icon from src/theme.py into assets/branding/.

The app draws its icon from the SVG in theme.py at runtime and needs none of these files;
they are for everything outside the app — the README, a Windows shortcut, a future
installer. Re-run after changing the mark in theme.py:

    python tools/build_branding.py

Writes:
    visus_mark.svg   the mark alone (full detail), the master for docs and print
    visus_icon.png   the 256 px app icon
    visus.ico        multi-size Windows icon, each size at its own detail level
"""

from __future__ import annotations

import struct
import sys
from pathlib import Path

from PySide6.QtCore import QBuffer, QByteArray, QIODevice
from PySide6.QtGui import QGuiApplication

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import theme  # noqa: E402
from common import PROJECT_ROOT  # noqa: E402

OUT_DIR = PROJECT_ROOT / "assets" / "branding"


def png_bytes(svg: str, size: int) -> bytes:
    """Render SVG source to PNG bytes at size x size."""
    data = QByteArray()
    buffer = QBuffer(data)
    buffer.open(QIODevice.OpenModeFlag.WriteOnly)
    theme.render_svg(svg, size).save(buffer, b"PNG")
    return bytes(data.data())


def write_ico(path: Path, images: dict[int, bytes]) -> None:
    """Write a .ico holding one PNG per size (the format Windows Vista+ reads).

    Qt's ICO writer stores a single image, so the container is written by hand: a 6-byte
    header, one 16-byte directory entry per image, then the PNG data.
    """
    header = struct.pack("<HHH", 0, 1, len(images))
    offset = len(header) + 16 * len(images)
    entries, blobs = b"", b""
    for size, blob in sorted(images.items()):
        side = 0 if size >= 256 else size  # 0 means 256 in an ICO directory entry
        entries += struct.pack("<BBBBHHII", side, side, 0, 0, 1, 32, len(blob), offset)
        blobs += blob
        offset += len(blob)
    path.write_bytes(header + entries + blobs)


def main() -> None:
    """Write the three branding files."""
    app = QGuiApplication.instance() or QGuiApplication(sys.argv)  # QPainter on images needs one
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "visus_mark.svg").write_text(theme.mark_svg(256), encoding="utf-8")
    (OUT_DIR / "visus_icon.png").write_bytes(png_bytes(theme.icon_svg(256), 256))
    write_ico(OUT_DIR / "visus.ico", {size: png_bytes(theme.icon_svg(size), size) for size in theme.ICON_SIZES})
    del app
    for name in ("visus_mark.svg", "visus_icon.png", "visus.ico"):
        print(f"wrote {(OUT_DIR / name).relative_to(PROJECT_ROOT).as_posix()}")


if __name__ == "__main__":
    main()
