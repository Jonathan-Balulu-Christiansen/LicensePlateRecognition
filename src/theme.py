"""Visus look: the Carbon Signal palette, fonts, the Qt stylesheet, the mark and the app icon.

Everything visual lives here so the window code only names roles ("accent", "muted"),
never colors. The mark is kept as SVG source in this file — the app renders it at any
size at runtime, and `tools/build_branding.py` exports the same SVG to assets/branding/
(.svg, .png, multi-size .ico) for docs and Windows shortcuts.

The mark: a V whose tip holds a lens; the arms are its sightlines, and inside the field of
view a reticle has caught a red plate. Two scan rays shorten toward the lens. It comes in
three detail levels, because a 16px taskbar icon cannot carry what a 256px one can.
"""

from __future__ import annotations

import sys

from PySide6.QtCore import QByteArray, QRectF, Qt
from PySide6.QtGui import QColor, QFont, QIcon, QImage, QPainter, QPalette, QPixmap
from PySide6.QtSvg import QSvgRenderer
from PySide6.QtWidgets import QApplication

APP_NAME = "Visus"
APP_USER_MODEL_ID = "Visus.PlateRecognition.1"

# Carbon Signal. Red is reserved for three things: live/active, the primary button, and
# the reticle around a tracked plate. Plate yellow appears only on the rendered plate.
COLORS = {
    "bg": "#0B0B0E",
    "surface": "#141419",
    "raised": "#1C1C23",
    "line": "#2B2B34",
    "text": "#F2F0EB",
    "muted": "#A09FA8",
    "accent": "#D9363E",
    "accent_hover": "#E5484F",
    "accent_dim": "#5A2226",
    "plate": "#F6C21C",
    "plate_ink": "#111111",
    "plate_strip": "#1F4FB4",
}

# Windows ships both: Bahnschrift is a DIN face (the road-signage look), Cascadia Mono
# keeps changing numbers from jittering sideways. The second family is the fallback.
UI_FONTS = ["Bahnschrift", "Segoe UI"]
MONO_FONTS = ["Cascadia Mono", "Consolas"]

_V = "M22 24H58L100 108L142 24H178L100 180Z"
_RETICLE = "M81 41V34H88M112 34H119V41M119 49V56H112M88 56H81V49"


def mark_svg(size: int = 256, fg: str = COLORS["text"], accent: str = COLORS["accent"]) -> str:
    """The mark alone, as SVG source, at the detail level that suits `size` px."""
    if size >= 96:  # full: lens with pupil, reticle, plate, two scan rays
        body = (
            f'<path fill-rule="evenodd" fill="{fg}" d="{_V} M92 146a8 8 0 1 0 16 0a8 8 0 1 0 -16 0Z"/>'
            f'<circle cx="100" cy="146" r="3" fill="{accent}"/>'
            f'<path d="{_RETICLE}" fill="none" stroke="{fg}" stroke-width="3.5"/>'
            f'<rect x="86" y="40" width="28" height="10" rx="2" fill="{accent}"/>'
            f'<path d="M90 72H110" stroke="{fg}" stroke-width="3" stroke-linecap="round" stroke-opacity="0.55"/>'
            f'<path d="M95 86H105" stroke="{fg}" stroke-width="3" stroke-linecap="round" stroke-opacity="0.3"/>'
        )
    elif size >= 40:  # medium: rays dropped, lens opened, heavier reticle
        body = (
            f'<path fill-rule="evenodd" fill="{fg}" d="{_V} M90 146a10 10 0 1 0 20 0a10 10 0 1 0 -20 0Z"/>'
            f'<path d="{_RETICLE}" fill="none" stroke="{fg}" stroke-width="5"/>'
            f'<rect x="86" y="40" width="28" height="10" rx="2" fill="{accent}"/>'
        )
    else:  # small: the V and the plate only
        body = f'<path fill="{fg}" d="{_V}"/><rect x="79" y="35" width="42" height="20" rx="3" fill="{accent}"/>'
    return f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="16 16 168 168">{body}</svg>'


def icon_svg(size: int = 256) -> str:
    """The app icon: the mark on a dark rounded tile, at the detail level that suits `size`."""
    inner = mark_svg(size).split(">", 1)[1].rsplit("</svg>", 1)[0]
    scale = 164 / 168
    return (
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 256 256">'
        f'<rect x="2" y="2" width="252" height="252" rx="56" fill="{COLORS["bg"]}" '
        f'stroke="{COLORS["line"]}" stroke-width="4"/>'
        f'<g transform="translate(46 46) scale({scale:.5f}) translate(-16 -16)">{inner}</g>'
        "</svg>"
    )


def render_svg(svg: str, size: int) -> QImage:
    """Rasterise SVG source to a transparent size x size image."""
    image = QImage(size, size, QImage.Format.Format_ARGB32_Premultiplied)
    image.fill(Qt.GlobalColor.transparent)
    painter = QPainter(image)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    QSvgRenderer(QByteArray(svg.encode())).render(painter, QRectF(0, 0, size, size))
    painter.end()
    return image


ICON_SIZES = (16, 24, 32, 48, 64, 128, 256)


def app_icon() -> QIcon:
    """The window/taskbar icon, with a hand-picked detail level at every standard size."""
    icon = QIcon()
    for size in ICON_SIZES:
        icon.addPixmap(QPixmap.fromImage(render_svg(icon_svg(size), size)))
    return icon


_ICON_PATHS = {  # 24x24 stroke icons
    "expand": "M4 9V4h5M15 4h5v5M20 15v5h-5M9 20H4v-5",
    "collapse": "M9 4v5H4M20 9h-5V4M15 20v-5h5M4 15h5v5",
}


def ui_icon(name: str) -> QIcon:
    """A small line icon in the text color (expand / collapse for full screen)."""
    svg = (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24"><path d="{_ICON_PATHS[name]}" '
           f'fill="none" stroke="{COLORS["text"]}" stroke-width="2" stroke-linecap="round" '
           'stroke-linejoin="round"/></svg>')
    icon = QIcon()
    for size in (16, 24, 32, 48):
        icon.addPixmap(QPixmap.fromImage(render_svg(svg, size)))
    return icon


def mark_pixmap(size: int, device_pixel_ratio: float = 1.0) -> QPixmap:
    """The bare mark (no tile) for the window header, sharp on high-DPI screens."""
    pixels = round(size * device_pixel_ratio)
    pixmap = QPixmap.fromImage(render_svg(mark_svg(size), pixels))
    pixmap.setDevicePixelRatio(device_pixel_ratio)
    return pixmap


def ui_font(point_size: float = 10.5, weight: QFont.Weight = QFont.Weight.Normal) -> QFont:
    """The UI face (Bahnschrift) at a size and weight."""
    font = QFont()
    font.setFamilies(UI_FONTS)
    font.setPointSizeF(point_size)
    font.setWeight(weight)
    return font


def mono_font(point_size: float = 10.5) -> QFont:
    """The numbers face (Cascadia Mono)."""
    font = QFont()
    font.setFamilies(MONO_FONTS)
    font.setPointSizeF(point_size)
    return font


def label_font(point_size: float = 8.5) -> QFont:
    """Small spaced capitals for card titles and metric names."""
    font = ui_font(point_size, QFont.Weight.DemiBold)
    font.setLetterSpacing(QFont.SpacingType.AbsoluteSpacing, 1.2)
    return font


def stylesheet() -> str:
    """The Qt stylesheet for the whole window, built from COLORS."""
    c = COLORS
    return f"""
    QWidget {{ background: {c["bg"]}; color: {c["text"]}; }}
    QFrame#Header, QFrame#Footer {{ background: {c["surface"]}; }}
    QFrame#Header {{ border-bottom: 1px solid {c["line"]}; }}
    QFrame#Footer {{ border-top: 1px solid {c["line"]}; }}
    QFrame#Card {{ background: {c["surface"]}; border: 1px solid {c["line"]}; border-radius: 14px; }}
    QFrame#Card QLabel, QFrame#Card QWidget {{ background: transparent; }}
    QFrame#Card QFrame#PlateItem {{ background: {c["raised"]}; border: 1px solid {c["line"]};
        border-radius: 10px; }}
    QFrame#Card QFrame#PlateItem[locked="true"] {{ border-left: 3px solid {c["accent"]}; }}
    QFrame#Header QLabel, QFrame#Footer QLabel {{ background: transparent; }}
    QLabel[role="muted"] {{ color: {c["muted"]}; }}
    QFrame#Card QLabel[role="chip"] {{ background: {c["text"]}; color: {c["bg"]}; border-radius: 6px;
        padding: 3px 9px; }}
    QFrame#Card QLabel[role="chipReading"] {{ background: transparent; color: {c["text"]};
        border: 1px solid {c["muted"]}; border-radius: 6px; padding: 2px 8px; }}
    QLabel[role="error"] {{ color: {c["accent_hover"]}; }}
    QComboBox, QSpinBox {{ background: {c["raised"]}; border: 1px solid {c["line"]}; border-radius: 8px;
        padding: 5px 10px; min-height: 24px; }}
    QComboBox:hover, QSpinBox:hover {{ border-color: {c["muted"]}; }}
    QComboBox:disabled, QSpinBox:disabled {{ color: {c["muted"]}; }}
    QComboBox::drop-down {{ border: none; width: 22px; }}
    QComboBox QAbstractItemView {{ background: {c["raised"]}; border: 1px solid {c["line"]};
        selection-background-color: {c["accent_dim"]}; outline: 0; }}
    QPushButton {{ background: transparent; border: 1px solid {c["line"]}; border-radius: 8px;
        padding: 7px 16px; min-height: 24px; }}
    QPushButton:hover {{ border-color: {c["muted"]}; }}
    QPushButton#Primary {{ background: {c["accent"]}; color: #FFFFFF; border: none; font-weight: 600;
        min-width: 84px; }}
    QPushButton#Primary:hover {{ background: {c["accent_hover"]}; }}
    QPushButton#Primary:disabled {{ background: {c["accent_dim"]}; color: {c["muted"]}; }}
    QPushButton#Icon {{ padding: 7px; min-width: 24px; }}
    QTableWidget {{ background: transparent; border: none; gridline-color: transparent; }}
    QTableWidget::item {{ border-top: 1px solid {c["line"]}; padding: 4px; }}
    QHeaderView::section {{ background: transparent; color: {c["muted"]}; border: none; padding: 4px; }}
    QScrollBar:vertical {{ background: transparent; width: 10px; }}
    QScrollBar::handle:vertical {{ background: {c["line"]}; border-radius: 5px; min-height: 24px; }}
    QScrollBar::add-line, QScrollBar::sub-line {{ height: 0; }}
    QToolTip {{ background: {c["raised"]}; color: {c["text"]}; border: 1px solid {c["line"]}; }}
    """


def apply_theme(app: QApplication) -> None:
    """Fusion style + a dark palette (so native arrows and focus rings match) + the stylesheet."""
    app.setStyle("Fusion")
    palette = QPalette()
    for role, key in (
        (QPalette.ColorRole.Window, "bg"),
        (QPalette.ColorRole.Base, "raised"),
        (QPalette.ColorRole.AlternateBase, "surface"),
        (QPalette.ColorRole.Button, "raised"),
        (QPalette.ColorRole.WindowText, "text"),
        (QPalette.ColorRole.Text, "text"),
        (QPalette.ColorRole.ButtonText, "text"),
        (QPalette.ColorRole.Highlight, "accent"),
        (QPalette.ColorRole.PlaceholderText, "muted"),
    ):
        palette.setColor(role, QColor(COLORS[key]))
    palette.setColor(QPalette.ColorRole.HighlightedText, QColor("#FFFFFF"))
    app.setPalette(palette)
    app.setFont(ui_font())
    app.setStyleSheet(stylesheet())
    app.setWindowIcon(app_icon())
    app.setApplicationName(APP_NAME)


def set_app_identity() -> None:
    """Give the process its own taskbar identity, so Windows shows the Visus icon instead
    of grouping the window under python.exe (same approach as Auris). Must run before the
    first window is created; a no-op elsewhere."""
    if sys.platform == "win32":
        try:
            import ctypes

            ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(APP_USER_MODEL_ID)
        except (AttributeError, OSError):
            pass
