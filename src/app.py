"""Visus — the desktop app: real-time licence-plate recognition from a camera or a video file.

Pick a source (the default camera is preselected), a detector and a plate reader, and press
Start. Each tracked plate gets a reticle over the video; its number is read by the CNN+CTC
reader (the whole plate in one pass) and temporal voting locks it once enough reads agree. The number, its
per-digit confidences and its country appear in the side panel — never on the car — and
each locked plate is added to the passages table and the gate log.

The heavy work runs in pipeline.VideoWorker (a QThread); this file is only the window.
The look (colors, fonts, the mark) comes from theme.py.
"""

from __future__ import annotations

import math
import sys
from collections import deque
from pathlib import Path

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QFont, QImage, QKeySequence, QPainter, QPainterPath, QPen, QPolygonF, QShortcut
from PySide6.QtMultimedia import QMediaDevices
from PySide6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QComboBox,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QPushButton,
    QSizePolicy,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

import theme
from common import PROJECT_ROOT, is_camera_source
from pipeline import (
    DEFAULT_DETECTOR,
    DEFAULT_READER,
    GATE_LOG,
    INFER_IMGSZ,
    MAX_PANEL_PLATES,
    READ_EVERY_N_FRAMES,
    FrameInfo,
    FrameStats,
    PlateEvent,
    PlateView,
    TrackView,
    VideoWorker,
    device_label,
    find_detector_models,
    find_reader_models,
)
from plate_origin import format_plate, plate_format_label

C = theme.COLORS
VIDEO_DIR = PROJECT_ROOT / "data" / "raw" / "videos"
MAX_PASSAGE_ROWS = 200
_BROWSE = "__browse__"


def project_relative(path: Path) -> str:
    """A path as the user knows it: relative to the project when it is inside it."""
    try:
        return path.resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return str(path)


def model_label(path: Path) -> str:
    """'best.pt · promoted' for weights/, '<run> · run' for runs/, else the file name."""
    try:
        parts = path.resolve().relative_to(PROJECT_ROOT).parts
    except ValueError:
        return f"{path.name} · external"
    if parts[0] == "weights":
        return f"{path.name} · promoted"
    if parts[0] == "runs" and len(parts) > 2:
        return f"{parts[2]} · run"
    return path.name


def confidence_text(value: float) -> str:
    """Three significant places for a 0-1 confidence: '1.00' at the top, '.997' below it."""
    return "1.00" if value >= 0.9995 else f"{value:.3f}"[1:]


# ---------------------------------------------------------------------------- widgets


class PathCombo(QComboBox):
    """A list of choices that ends with a 'Browse…' entry opening a file dialog."""

    def __init__(self, browse_label: str, dialog_title: str, file_filter: str, start_dir: Path) -> None:
        """Start with only the Browse entry; add_choice() fills the list above it."""
        super().__init__()
        self._dialog = (dialog_title, file_filter, start_dir)
        self._last = -1
        self.addItem(browse_label, _BROWSE)
        self.activated.connect(self._on_activated)
        self.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToContents)

    def add_choice(self, label: str, value: str, tooltip: str = "") -> int:
        """Insert a choice just above 'Browse…' and return its index."""
        index = self.count() - 1
        self.insertItem(index, label, value)
        self.setItemData(index, tooltip or label, Qt.ItemDataRole.ToolTipRole)
        if self._last >= index:
            self._last += 1
        return index

    def clear_choices(self) -> None:
        """Remove everything but 'Browse…'."""
        while self.count() > 1:
            self.removeItem(0)
        self._last = -1

    def choices(self) -> list[tuple[str, str]]:
        """(label, value) of every choice, in order."""
        return [(self.itemText(i), self.itemData(i)) for i in range(self.count() - 1)]

    def select_value(self, value: str) -> bool:
        """Select the choice holding `value`; False if there is none."""
        index = self.findData(value)
        if index < 0 or value == _BROWSE:
            return False
        self.setCurrentIndex(index)
        self._last = index
        return True

    def value(self) -> str | None:
        """The selected choice's value, or None while only 'Browse…' is there."""
        data = self.currentData()
        return None if data in (None, _BROWSE) else str(data)

    def _on_activated(self, index: int) -> None:
        """Remember a real choice; for 'Browse…', ask for a file and add it as a choice."""
        if self.itemData(index) != _BROWSE:
            self._last = index
            return
        title, file_filter, start_dir = self._dialog
        path, _ = QFileDialog.getOpenFileName(self, title, str(start_dir), file_filter)
        if path:
            path = str(Path(path))
            if not self.select_value(path):
                self.select_value(self.itemData(self.add_choice(Path(path).name, path, path)))
        elif self._last >= 0:
            self.setCurrentIndex(self._last)


class Card(QFrame):
    """A rounded surface with a small spaced-capitals title."""

    def __init__(self, title: str = "") -> None:
        """Create the card with an optional title row."""
        super().__init__()
        self.setObjectName("Card")
        self.body = QVBoxLayout(self)
        self.body.setContentsMargins(18, 16, 18, 16)
        self.body.setSpacing(12)
        self.title_row = QHBoxLayout()
        self.title = muted_label(title, theme.label_font())
        self.title_row.addWidget(self.title)
        self.title_row.addStretch(1)
        if title:
            self.body.addLayout(self.title_row)


def muted_label(text: str = "", font: QFont | None = None) -> QLabel:
    """A label in the muted text color."""
    label = QLabel(text)
    label.setProperty("role", "muted")
    if font is not None:
        label.setFont(font)
    return label


def set_role(label: QLabel, role: str) -> None:
    """Switch a label's stylesheet role (chip, muted, error…) and re-apply the style."""
    label.setProperty("role", role)
    label.style().unpolish(label)
    label.style().polish(label)
    label.updateGeometry()  # the new role may add padding: let the layout re-measure it


class VideoView(QWidget):
    """The video with the reticle overlay and the HUD, painted in one pass."""

    def __init__(self) -> None:
        """Start empty, showing a hint until the first frame arrives."""
        super().__init__()
        self.setMinimumSize(560, 315)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self._image: QImage | None = None
        self._tracks: list[TrackView] = []
        self._stats: FrameStats | None = None
        self._message = "Pick a source and press Start"
        self._source_tag = ""
        self._live = False
        self._ambient: QImage | None = None
        self._watermark = theme.mark_pixmap(88, self.devicePixelRatioF())

    def set_frame(self, image: QImage, info: FrameInfo) -> None:
        """Show a new frame and its overlay."""
        self._image, self._tracks, self._stats = image, info.tracks, info.stats
        # A 32x18 thumbnail of the frame, stretched over the letterbox bars and darkened,
        # fills them with the scene's own light instead of flat black (as video players do).
        # Shrinking in two steps keeps it smooth; together they cost well under 1 ms.
        self._ambient = image.scaled(256, 144, Qt.AspectRatioMode.IgnoreAspectRatio).scaled(
            32, 18, Qt.AspectRatioMode.IgnoreAspectRatio, Qt.TransformationMode.SmoothTransformation)
        self._message = ""
        self.update()

    def set_message(self, message: str) -> None:
        """Text shown in the middle while there is no live frame (loading, errors)."""
        self._message = message
        self.update()

    def set_source(self, tag: str, live: bool) -> None:
        """The HUD's source tag ('LIVE · GENERAL WEBCAM' / 'FILE · test1.mp4')."""
        self._source_tag, self._live = tag, live
        self.update()

    def mark_stopped(self) -> None:
        """Keep the last frame on screen, but say the source is no longer live."""
        self._source_tag = self._source_tag.replace("LIVE", "STOPPED", 1).replace("FILE", "ENDED", 1)
        self._live = False
        self.update()

    def clear(self) -> None:
        """Drop the last frame (a new run is starting)."""
        self._image, self._tracks, self._stats, self._ambient = None, [], None, None
        self.update()

    def paintEvent(self, event) -> None:  # noqa: N802 - Qt override name
        """Frame, then overlay, then HUD, then any message."""
        painter = QPainter(self)
        painter.setRenderHints(QPainter.RenderHint.Antialiasing | QPainter.RenderHint.SmoothPixmapTransform)
        bounds = QRectF(self.rect()).adjusted(0.5, 0.5, -0.5, -0.5)
        clip = QPainterPath()
        clip.addRoundedRect(bounds, 14, 14)
        painter.fillPath(clip, QColor("#060607"))
        painter.setClipPath(clip)

        if self._image is not None:
            image_w, image_h = self._image.width(), self._image.height()
            scale = min(bounds.width() / image_w, bounds.height() / image_h)
            target = QRectF(0, 0, image_w * scale, image_h * scale)
            target.moveCenter(bounds.center())
            letterboxed = target.width() < bounds.width() - 2 or target.height() < bounds.height() - 2
            if self._ambient is not None and letterboxed:
                cover = max(bounds.width() / image_w, bounds.height() / image_h)
                fill = QRectF(0, 0, image_w * cover * 1.1, image_h * cover * 1.1)
                fill.moveCenter(bounds.center())
                painter.drawImage(fill, self._ambient)
                painter.fillRect(bounds, QColor(6, 6, 7, 200))
            painter.drawImage(target, self._image)
            for track in self._tracks:
                x1, y1, x2, y2 = track.box
                box = QRectF(target.x() + x1 * scale, target.y() + y1 * scale, (x2 - x1) * scale, (y2 - y1) * scale)
                self._paint_track(painter, box, track)
            self._paint_hud(painter, bounds)

        if self._message:
            text_box = bounds.adjusted(40, 0, -40, 0)
            if self._image is None:  # empty state: a faint mark above the message
                size = self._watermark.deviceIndependentSize()
                painter.setOpacity(0.22)
                painter.drawPixmap(QPointF(bounds.center().x() - size.width() / 2,
                                           bounds.center().y() - size.height() - 12), self._watermark)
                painter.setOpacity(1.0)
                text_box = QRectF(text_box.left(), bounds.center().y(), text_box.width(), bounds.height() / 2)
                align = Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignTop
            else:
                painter.fillRect(bounds, QColor(6, 6, 7, 170))  # keep a message readable over a frame
                align = Qt.AlignmentFlag.AlignCenter
            painter.setFont(theme.ui_font(12))
            painter.setPen(QColor(C["muted"]))
            painter.drawText(text_box, align | Qt.TextFlag.TextWordWrap, self._message)

        painter.setClipping(False)
        painter.setPen(QPen(QColor(C["line"]), 1))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawRoundedRect(bounds, 14, 14)

    @staticmethod
    def _paint_track(painter: QPainter, box: QRectF, track: TrackView) -> None:
        """Locked: red corner brackets + filled tag. Reading: dashed outline + plain tag."""
        painter.setFont(theme.mono_font(9))
        metrics = painter.fontMetrics()
        pad = 6.0
        tag_w = metrics.horizontalAdvance(track.label) + 2 * 8
        tag_h = metrics.height() + 6
        tag = QRectF(box.left() - pad, box.top() - pad - tag_h - 4, tag_w, tag_h)
        if tag.top() < 0:  # no room above: put the tag under the plate
            tag.moveTop(box.bottom() + pad + 4)

        if track.locked:
            outer = box.adjusted(-pad, -pad, pad, pad)
            arm = max(8.0, min(outer.width(), outer.height()) * 0.28)
            painter.setPen(QPen(QColor(C["accent"]), 3, Qt.PenStyle.SolidLine, Qt.PenCapStyle.SquareCap))
            for corner, dx, dy in (
                (outer.topLeft(), 1, 1), (outer.topRight(), -1, 1),
                (outer.bottomRight(), -1, -1), (outer.bottomLeft(), 1, -1),
            ):
                painter.drawPolyline(QPolygonF([
                    QPointF(corner.x(), corner.y() + dy * arm), corner, QPointF(corner.x() + dx * arm, corner.y()),
                ]))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QColor(C["accent"]))
            painter.drawRoundedRect(tag, 5, 5)
            painter.setPen(QColor("#FFFFFF"))
        else:
            dash = Qt.PenStyle.DashLine if track.track_id is not None else Qt.PenStyle.DotLine
            painter.setPen(QPen(QColor(C["text"]), 1.5, dash))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawRect(box.adjusted(-3, -3, 3, 3))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QColor(0, 0, 0, 150))
            painter.drawRoundedRect(tag, 5, 5)
            painter.setPen(QColor(C["text"]))
        painter.drawText(tag, Qt.AlignmentFlag.AlignCenter, track.label)
        painter.setBrush(Qt.BrushStyle.NoBrush)

    def _paint_hud(self, painter: QPainter, bounds: QRectF) -> None:
        """Source tag top-left, per-frame time and fps top-right."""
        painter.setFont(theme.mono_font(9))
        metrics = painter.fontMetrics()
        height = metrics.height() + 10

        if self._source_tag and self._image is not None:
            text = f"{self._source_tag} · {self._image.width()}×{self._image.height()}"
            left = QRectF(bounds.left() + 14, bounds.top() + 14, metrics.horizontalAdvance(text) + 34, height)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QColor(0, 0, 0, 160))
            painter.drawRoundedRect(left, 7, 7)
            painter.setBrush(QColor(C["accent"] if self._live else C["muted"]))
            painter.drawEllipse(QPointF(left.left() + 13, left.center().y()), 4, 4)
            painter.setPen(QColor(C["text"]))
            painter.drawText(left.adjusted(24, 0, -8, 0), Qt.AlignmentFlag.AlignVCenter, text)

        if self._stats is not None and self._stats.fps > 0:
            text = f"{self._stats.frame_ms:.1f} ms · {self._stats.fps:.1f} fps"
            width = metrics.horizontalAdvance(text) + 20
            right = QRectF(bounds.right() - 14 - width, bounds.top() + 14, width, height)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QColor(0, 0, 0, 160))
            painter.drawRoundedRect(right, 7, 7)
            painter.setPen(QColor(C["text"]))
            painter.drawText(right, Qt.AlignmentFlag.AlignCenter, text)


class PlateBadge(QWidget):
    """The read number drawn as a plate: an Israeli one (blue IL strip, yellow body) when
    the origin is IL, a plain light plate otherwise, a dashed outline when empty."""

    def __init__(self, height: int = 76) -> None:
        """Start empty, `height` px tall."""
        super().__init__()
        self.setFixedHeight(height)
        self._number = ""
        self._code = ""
        self._dim = False

    def set_plate(self, number: str, code: str, dim: bool = False) -> None:
        """Show `number`; `dim` while it is only the leading candidate, not locked."""
        self._number, self._code, self._dim = number, code, dim
        self.update()

    def paintEvent(self, event) -> None:  # noqa: N802 - Qt override name
        """Draw the plate and fit the number to it."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        rect = QRectF(self.rect()).adjusted(1.5, 1.5, -1.5, -1.5)
        if not self._number:
            painter.setPen(QPen(QColor(C["line"]), 1.5, Qt.PenStyle.DashLine))
            painter.drawRoundedRect(rect, 8, 8)
            painter.setFont(theme.ui_font(11))
            painter.setPen(QColor(C["muted"]))
            painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, "no plate yet")
            return

        israeli = self._code == "IL"
        painter.setOpacity(0.55 if self._dim else 1.0)
        painter.setPen(QPen(QColor(C["plate_ink"]), 3))
        painter.setBrush(QColor(C["plate"] if israeli else "#ECEAE4"))
        painter.drawRoundedRect(rect, 8, 8)
        body = rect.adjusted(3, 3, -3, -3)
        if israeli:
            strip = QRectF(body.left(), body.top(), max(22.0, body.width() * 0.09), body.height())
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QColor(C["plate_strip"]))
            painter.drawRoundedRect(strip, 4, 4)
            painter.setPen(QColor("#FFFFFF"))
            painter.setFont(theme.ui_font(8, QFont.Weight.Bold))
            painter.drawText(strip.adjusted(0, 0, 0, -6),
                             Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignBottom, "IL")
            body.setLeft(strip.right())

        text = format_plate(self._number, self._code)
        font = theme.ui_font(10, QFont.Weight.Bold)
        font.setPixelSize(max(10, int(body.height() * 0.62)))
        while True:  # largest size that fits the plate body
            painter.setFont(font)
            if font.pixelSize() <= 10 or painter.fontMetrics().horizontalAdvance(text) <= body.width() - 20:
                break
            font.setPixelSize(font.pixelSize() - 1)
        painter.setPen(QColor(C["plate_ink"]))
        painter.drawText(body, Qt.AlignmentFlag.AlignCenter, text)


class FlagIcon(QWidget):
    """The country's flag — Israel's for 'IL', a neutral '?' tile otherwise."""

    def __init__(self, width: int = 44, height: int = 32) -> None:
        """Fixed size, unknown until set_code()."""
        super().__init__()
        self.setFixedSize(width, height)
        self._code = ""

    def set_code(self, code: str) -> None:
        """Switch the flag."""
        self._code = code
        self.update()

    def paintEvent(self, event) -> None:  # noqa: N802 - Qt override name
        """Israel: white field, two blue stripes, Star of David as two triangles."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        rect = QRectF(self.rect()).adjusted(0.5, 0.5, -0.5, -0.5)
        if self._code != "IL":
            painter.setPen(QPen(QColor(C["line"]), 1))
            painter.setBrush(QColor(C["raised"]))
            painter.drawRoundedRect(rect, 3, 3)
            painter.setPen(QColor(C["muted"]))
            painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, "?")
            return
        blue = QColor("#0038B8")
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor("#FFFFFF"))
        painter.drawRoundedRect(rect, 3, 3)
        painter.setBrush(blue)
        h, w = rect.height(), rect.width()
        painter.drawRect(QRectF(rect.left(), rect.top() + h * 0.09, w, h * 0.125))
        painter.drawRect(QRectF(rect.left(), rect.top() + h * 0.785, w, h * 0.125))
        cx, cy, r = rect.center().x(), rect.center().y(), h * 0.21
        painter.setPen(QPen(blue, 1.3))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        for sign in (1, -1):  # one triangle pointing up, one pointing down
            apex = QPointF(cx, cy - sign * r)
            left = QPointF(cx - r * 0.866, cy + sign * r / 2)
            right = QPointF(cx + r * 0.866, cy + sign * r / 2)
            painter.drawPolygon(QPolygonF([apex, left, right]))


class CropView(QWidget):
    """The plate as the camera saw it: the latest crop, fitted and centred, with rounded corners."""

    def __init__(self, height: int) -> None:
        """Fixed height; the width follows the card."""
        super().__init__()
        self.setFixedHeight(height)
        self._image: QImage | None = None

    def set_image(self, image: QImage | None) -> None:
        """Show `image` (None: an empty frame)."""
        self._image = image
        self.update()

    def paintEvent(self, event) -> None:  # noqa: N802 - Qt override name
        """Draw the crop scaled to fit, or a dashed placeholder."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        rect = QRectF(self.rect()).adjusted(0.5, 0.5, -0.5, -0.5)
        if self._image is None or self._image.isNull():
            painter.setPen(QPen(QColor(C["line"]), 1, Qt.PenStyle.DashLine))
            painter.drawRoundedRect(rect, 6, 6)
            return
        scale = min(rect.width() / self._image.width(), rect.height() / self._image.height())
        w, h = self._image.width() * scale, self._image.height() * scale
        target = QRectF(rect.center().x() - w / 2, rect.center().y() - h / 2, w, h)
        clip = QPainterPath()
        clip.addRoundedRect(target, 6, 6)
        painter.setClipPath(clip)
        painter.drawImage(target, self._image)
        painter.setClipping(False)
        painter.setPen(QPen(QColor(C["line"]), 1))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawRoundedRect(target, 6, 6)


class PlateItem(QFrame):
    """One plate in the side panel, on a compact card so several fit:

        ID 3 · 6 reads                         [LOCKED 5/5]
        [the plate as the camera saw it]  ->  [the number read]
        [1][2][3][4][5][6][7][8]   each digit with its confidence
        [flag] Israel · 8 digits              read 0.997 · det 0.90

    A locked plate gets the accent edge; a digit the reader is unsure of (< 0.90) is marked.
    """

    LOW_CONFIDENCE = 0.90

    def __init__(self) -> None:
        """Build the empty card; show_plate() fills it."""
        super().__init__()
        self.setObjectName("PlateItem")
        # A fixed height: the panel shows fewer cards on a short window instead of squeezing them.
        self.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 9, 12, 10)
        layout.setSpacing(7)

        header = QHBoxLayout()
        header.setSpacing(6)
        self.track_label = QLabel("")
        self.track_label.setFont(theme.mono_font(8.5))
        self.reads_label = muted_label("", theme.mono_font(8.5))
        header.addWidget(self.track_label)
        header.addWidget(self.reads_label)
        header.addStretch(1)
        self.chip = QLabel("")
        self.chip.setFont(theme.label_font(7.5))
        self.chip.setContentsMargins(6, 2, 6, 2)  # QSS padding is not applied to this label
        header.addWidget(self.chip)
        layout.addLayout(header)

        plate_row = QHBoxLayout()
        plate_row.setSpacing(8)
        self.crop_view = CropView(40)
        self.crop_view.setToolTip("The plate as the camera saw it")
        plate_row.addWidget(self.crop_view, 9)
        arrow = muted_label("→", theme.ui_font(12))
        plate_row.addWidget(arrow)
        self.badge = PlateBadge(height=40)
        self.badge.setToolTip("The number the reader made of it")
        plate_row.addWidget(self.badge, 11)
        layout.addLayout(plate_row)

        digits = QHBoxLayout()
        digits.setSpacing(3)
        self.digit_cells: list[tuple[QFrame, QLabel, QLabel]] = []
        for _ in range(8):
            cell = QFrame()
            cell.setStyleSheet(f"background: {C['surface']}; border-radius: 5px;")
            cell_layout = QVBoxLayout(cell)
            cell_layout.setContentsMargins(0, 3, 0, 3)
            cell_layout.setSpacing(0)
            digit = QLabel("")
            digit.setFont(theme.mono_font(10.5))
            confidence = muted_label("", theme.mono_font(6.5))
            for label in (digit, confidence):
                label.setAlignment(Qt.AlignmentFlag.AlignCenter)
                label.setStyleSheet("background: transparent;")
                cell_layout.addWidget(label)
            digits.addWidget(cell)
            self.digit_cells.append((cell, digit, confidence))
        layout.addLayout(digits)

        footer = QHBoxLayout()
        footer.setSpacing(8)
        self.flag = FlagIcon(24, 17)
        footer.addWidget(self.flag)
        self.origin_label = QLabel("")
        self.origin_label.setFont(theme.ui_font(9))
        footer.addWidget(self.origin_label, 1)
        self.scores_label = muted_label("", theme.mono_font(8))
        self.scores_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        self.scores_label.setToolTip("read: product of the per-digit confidences, the reader's own score\n"
                                     "for the whole number · det: detector confidence")
        footer.addWidget(self.scores_label)
        layout.addLayout(footer)

    def show_plate(self, plate: PlateView) -> None:
        """Fill the card from one PlateView."""
        if self.property("locked") != plate.locked:
            self.setProperty("locked", plate.locked)  # the accent edge, see theme.stylesheet()
            self.style().unpolish(self)
            self.style().polish(self)
        self.track_label.setText(f"ID {plate.track_id}")
        self.reads_label.setText(f"· {plate.reads} reads" if plate.visible else "· out of view")
        if plate.locked:
            self.chip.setText(f"LOCKED {plate.votes}/{plate.min_votes}")
            set_role(self.chip, "chip")
        else:
            self.chip.setText(f"READING {plate.votes}/{plate.min_votes}")
            set_role(self.chip, "chipReading")
        self.crop_view.set_image(plate.crop)
        self.badge.set_plate(plate.number, plate.origin.code, dim=not plate.locked)
        for index, (cell, digit, confidence) in enumerate(self.digit_cells):
            has_digit = index < len(plate.number)
            cell.setVisible(has_digit)
            if has_digit:
                digit.setText(plate.number[index])
                conf = plate.digit_confs[index] if index < len(plate.digit_confs) else 0.0
                confidence.setText(confidence_text(conf))
                set_role(confidence, "error" if conf < self.LOW_CONFIDENCE else "muted")
        origin = plate.origin
        self.flag.set_code(origin.code)
        country = origin.country if origin.code else "Origin unknown"
        self.origin_label.setText(f"{country} · {len(plate.number)} digits")
        layout = plate_format_label(plate.number, origin.code)
        self.origin_label.setToolTip(f"{layout}\n{origin.method} ({origin.score:.2f})" if origin.code else layout)
        score = f"{math.prod(plate.digit_confs):.3f}" if plate.digit_confs else "—"
        self.scores_label.setText(f"read {score} · det {plate.detector_conf:.2f}")


class Sparkline(QWidget):
    """Per-frame processing time over the last few seconds, against the source's frame budget."""

    def __init__(self, length: int = 90) -> None:
        """Keep the last `length` frames."""
        super().__init__()
        self.setMinimumSize(200, 56)
        self._length = length
        self._values: deque[float] = deque(maxlen=length)
        self._budget = 1000 / 30

    def add(self, value_ms: float, source_fps: float) -> None:
        """Append one frame; the budget follows the source fps (30 when unknown)."""
        self._values.append(value_ms)
        self._budget = 1000 / (source_fps if source_fps > 0 else 30)
        self.update()

    def clear(self) -> None:
        """Forget the history (a new run)."""
        self._values.clear()
        self.update()

    def paintEvent(self, event) -> None:  # noqa: N802 - Qt override name
        """Dashed budget line with its label, then the series."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        rect = QRectF(self.rect()).adjusted(0, 14, 0, -2)
        top = max([self._budget * 1.25, *self._values])

        def y_of(value: float) -> float:
            return rect.bottom() - value / top * rect.height()
        budget_y = y_of(self._budget)
        painter.setPen(QPen(QColor(C["muted"]), 1, Qt.PenStyle.DashLine))
        painter.drawLine(QPointF(rect.left(), budget_y), QPointF(rect.right(), budget_y))
        painter.setFont(theme.mono_font(7.5))
        painter.drawText(QRectF(rect.left(), 0, rect.width(), 13), Qt.AlignmentFlag.AlignRight,
                         f"{self._budget:.1f} ms budget · {1000 / self._budget:.0f} fps")
        if len(self._values) > 1:
            step = rect.width() / (self._length - 1)
            offset = rect.width() - step * (len(self._values) - 1)
            points = [QPointF(rect.left() + offset + i * step, y_of(v)) for i, v in enumerate(self._values)]
            painter.setPen(QPen(QColor(C["text"]), 1.6))
            painter.drawPolyline(QPolygonF(points))


# ------------------------------------------------------------------------------ window


class PassagesTable(QTableWidget):
    """The passages table, with a hint in its empty body instead of a blank box."""

    def __init__(self) -> None:
        """Five read-only columns; the plate column takes the spare width."""
        super().__init__(0, 5)
        self.setHorizontalHeaderLabels(["TIME", "PLATE", "ORIGIN", "DET.", "READ"])
        self.verticalHeader().setVisible(False)
        self.setShowGrid(False)
        self.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        self.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.setMinimumHeight(90)
        header = self.horizontalHeader()
        header.setFont(theme.label_font())
        header.setDefaultAlignment(Qt.AlignmentFlag.AlignLeft)
        header.setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)

    def paintEvent(self, event) -> None:  # noqa: N802 - Qt override name
        """The rows, or a one-line hint while there are none."""
        super().paintEvent(event)
        if self.rowCount() == 0:
            painter = QPainter(self.viewport())
            painter.setPen(QColor(C["muted"]))
            painter.setFont(theme.ui_font(10))
            painter.drawText(self.viewport().rect(), Qt.AlignmentFlag.AlignCenter | Qt.TextFlag.TextWordWrap,
                             "Every plate that locks is listed here,\nnewest first.")


class MainWindow(QWidget):
    """Header controls; video + performance on the left; plate, origin and passages on the right.

    The layout adapts to the window: the side column takes ~25% of the width (360-500 px)
    and always runs to the bottom, the sparkline moves under the numbers when the strip is
    narrow, and F11 / the header button toggle full screen.
    """

    SIDE_MIN, SIDE_MAX, SIDE_SHARE = 360, 500, 0.25
    PASSAGES_MIN = 170  # px the passages card keeps: its title, the header and two rows
    INLINE_SPARKLINE_FROM = 1000  # px of left-column width

    def __init__(self) -> None:
        """Build the window and fill the source and model lists."""
        super().__init__()
        self.setWindowTitle("Visus — Live")
        self.resize(1440, 900)
        self.setMinimumSize(1100, 720)
        self.worker: VideoWorker | None = None
        self._logged = 0
        self._plates: list[PlateView] = []

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)
        root.addWidget(self._build_header())
        body = QHBoxLayout()
        body.setContentsMargins(20, 18, 20, 18)
        body.setSpacing(18)
        body.addLayout(self._build_main_column(), 1)
        body.addWidget(self._build_side_panel())
        root.addLayout(body, 1)
        root.addWidget(self._build_footer())

        QShortcut(QKeySequence(Qt.Key.Key_F11), self, self._toggle_fullscreen)
        QShortcut(QKeySequence(Qt.Key.Key_Escape), self, self._leave_fullscreen)
        self._media = QMediaDevices(self)
        self._media.videoInputsChanged.connect(self._populate_sources)
        self._populate_sources()
        self._populate_models()
        self._set_state("idle")

    def resizeEvent(self, event) -> None:  # noqa: N802 - Qt override name
        """Re-proportion the side column and place the sparkline for the new size."""
        super().resizeEvent(event)
        side = max(self.SIDE_MIN, min(self.SIDE_MAX, int(self.width() * self.SIDE_SHARE)))
        self.side_panel.setFixedWidth(side)
        left_width = self.width() - side - 58  # body margins + spacing
        inline = left_width >= self.INLINE_SPARKLINE_FROM
        if inline != self._sparkline_inline:
            self._sparkline_inline = inline
            for layout in (self.metrics_row, self.spark_row):
                layout.removeWidget(self.sparkline)
            if inline:
                self.metrics_row.addWidget(self.sparkline, 3)
            else:
                self.spark_row.addWidget(self.sparkline)
            self.spark_row_widget.setVisible(not inline)
        self._show_plates(self._plates)  # the height may now hold more or fewer plate cards

    def _toggle_fullscreen(self) -> None:
        """F11 / the header button: full screen and back."""
        if self.isFullScreen():
            self.showNormal()
        else:
            self.showFullScreen()
        self.fullscreen_button.setIcon(theme.ui_icon("collapse" if self.isFullScreen() else "expand"))

    def _leave_fullscreen(self) -> None:
        """Esc leaves full screen (and does nothing otherwise)."""
        if self.isFullScreen():
            self._toggle_fullscreen()

    # --- building

    def _build_header(self) -> QFrame:
        """Logo, source and model pickers, votes, device, Start/Stop."""
        header = QFrame()
        header.setObjectName("Header")
        row = QHBoxLayout(header)
        row.setContentsMargins(22, 12, 22, 12)
        row.setSpacing(16)

        logo = QLabel()
        logo.setPixmap(theme.mark_pixmap(30, self.devicePixelRatioF()))
        wordmark = QLabel("VISUS")
        font = theme.ui_font(15, QFont.Weight.DemiBold)
        font.setLetterSpacing(QFont.SpacingType.AbsoluteSpacing, 5)
        wordmark.setFont(font)
        row.addWidget(logo)
        row.addWidget(wordmark)
        row.addSpacing(10)

        self.source_combo = PathCombo("Video file…", "Open video", "Videos (*.mp4 *.avi *.mkv *.mov);;All files (*)",
                                      VIDEO_DIR if VIDEO_DIR.is_dir() else PROJECT_ROOT)
        self.detector_combo = PathCombo("Browse…", "Choose detector weights", "YOLO weights (*.pt)", PROJECT_ROOT)
        self.reader_combo = PathCombo("Browse…", "Choose plate reader", "Keras model (*.keras)", PROJECT_ROOT)
        for combo, widest in ((self.source_combo, 340), (self.detector_combo, 320), (self.reader_combo, 300)):
            # Shrink to 140 px on a small window, grow to `widest` on a large one.
            combo.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
            combo.setMinimumContentsLength(10)
            combo.setMinimumWidth(140)
            combo.setMaximumWidth(widest)
            combo.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.votes_spin = QSpinBox()
        self.votes_spin.setRange(1, 30)
        self.votes_spin.setValue(5)
        self.votes_spin.setFixedWidth(52)
        self.votes_spin.setAlignment(Qt.AlignmentFlag.AlignCenter)
        # No arrow buttons: the stylesheet hides Fusion's arrows. Wheel and arrow keys still work.
        self.votes_spin.setButtonSymbols(QSpinBox.ButtonSymbols.NoButtons)
        self.votes_spin.setToolTip("Agreeing reads needed before a number locks (scroll or type)")
        for name, widget in (("Source", self.source_combo), ("Detector", self.detector_combo),
                             ("Reader", self.reader_combo), ("Votes", self.votes_spin)):
            label = muted_label(name)
            label.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Preferred)
            row.addWidget(label)
            row.addWidget(widget, 0 if widget is self.votes_spin else 1)  # pickers grow first, up to their max
        row.addStretch(0)  # whatever the pickers cannot use goes before the buttons

        self.fullscreen_button = QPushButton()
        self.fullscreen_button.setObjectName("Icon")
        self.fullscreen_button.setIcon(theme.ui_icon("expand"))
        self.fullscreen_button.setToolTip("Full screen (F11)")
        self.fullscreen_button.setAccessibleName("Full screen")
        self.fullscreen_button.clicked.connect(self._toggle_fullscreen)
        row.addWidget(self.fullscreen_button)
        self.start_button = QPushButton("Start")
        self.start_button.setObjectName("Primary")
        self.start_button.clicked.connect(self._toggle)
        row.addWidget(self.start_button)
        return header

    def _build_main_column(self) -> QVBoxLayout:
        """Video, then the performance strip (numbers, and the sparkline beside or below them)."""
        column = QVBoxLayout()
        column.setSpacing(16)
        self.video = VideoView()
        column.addWidget(self.video, 1)

        perf = Card()
        perf.body.setContentsMargins(20, 14, 20, 14)
        perf.body.setSpacing(10)
        self.metrics_row = QHBoxLayout()
        self.metrics_row.setSpacing(0)
        frame_box = QVBoxLayout()
        frame_box.setSpacing(0)
        frame_box.addWidget(muted_label("FRAME", theme.label_font()))
        self.frame_ms_label = QLabel("—")
        self.frame_ms_label.setFont(theme.mono_font(22))
        self.frame_ms_label.setMinimumWidth(self.frame_ms_label.fontMetrics().horizontalAdvance("000.0 ms"))
        self.frame_ms_label.setToolTip("Processing time per frame: detect + read + conversion,\n"
                                       "averaged over the last 30 frames (capture wait excluded)")
        frame_box.addWidget(self.frame_ms_label)
        self.metrics_row.addLayout(frame_box)
        self.stage_labels: dict[str, QLabel] = {}
        for key, name in (("detect", "DETECT+TRACK"), ("read", "READ"), ("convert", "CONVERT"),
                          ("capture", "CAPTURE"), ("fps", "FPS"), ("tracks", "TRACKS")):
            self.metrics_row.addStretch(1)  # equal gaps: the numbers spread over any width
            box = QVBoxLayout()
            box.setSpacing(2)
            box.addWidget(muted_label(name, theme.label_font()))
            value = QLabel("—")
            value.setFont(theme.mono_font(12))
            box.addWidget(value)
            self.metrics_row.addLayout(box)
            self.stage_labels[key] = value
        self.metrics_row.addSpacing(24)
        self.sparkline = Sparkline()
        self.metrics_row.addWidget(self.sparkline, 3)
        self._sparkline_inline = True
        perf.body.addLayout(self.metrics_row)
        self.spark_row_widget = QWidget()
        self.spark_row = QHBoxLayout(self.spark_row_widget)
        self.spark_row.setContentsMargins(0, 0, 0, 0)
        self.spark_row_widget.setVisible(False)
        perf.body.addWidget(self.spark_row_widget)
        column.addWidget(perf)
        return column

    def _build_side_panel(self) -> QWidget:
        """The plates in view (one compact card each), and the passages table below them,
        which takes the rest of the height."""
        panel = QWidget()
        self.side_panel = panel
        panel.setFixedWidth(self.SIDE_MIN)
        column = QVBoxLayout(panel)
        self.side_column = column
        column.setContentsMargins(0, 0, 0, 0)
        column.setSpacing(16)

        plates = Card("PLATES IN VIEW")
        self.plates_card = plates
        plates.body.setSpacing(10)
        self.plates_count = muted_label("", theme.mono_font(8.5))
        plates.title_row.addWidget(self.plates_count)
        self.no_plate_label = muted_label("No plate read yet")
        self.no_plate_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.no_plate_label.setMinimumHeight(64)
        plates.body.addWidget(self.no_plate_label)
        self.plate_items = [PlateItem() for _ in range(MAX_PANEL_PLATES)]
        for item in self.plate_items:
            item.setVisible(False)
            plates.body.addWidget(item)
        # The plates card keeps its natural height (it grows with the plates shown);
        # the passages card takes whatever is left.
        plates.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
        column.addWidget(plates)

        passages = Card("RECENT PASSAGES")
        self.logged_value = muted_label("0 logged", theme.mono_font(8.5))
        passages.title_row.addWidget(self.logged_value)
        self.passages = PassagesTable()
        passages.body.addWidget(self.passages, 1)
        column.addWidget(passages, 1)
        self._show_plates([])
        return panel

    def _build_footer(self) -> QFrame:
        """A status bar: the current message on the left, the run's fixed settings on the right."""
        footer = QFrame()
        footer.setObjectName("Footer")
        row = QHBoxLayout(footer)
        row.setContentsMargins(22, 7, 22, 7)
        row.setSpacing(24)
        self.status_label = muted_label("Ready", theme.mono_font(8.5))
        row.addWidget(self.status_label, 1)
        for text in (device_label(), f"input {INFER_IMGSZ} · conf 0.25", f"read every {READ_EVERY_N_FRAMES}th frame",
                     f"log → {project_relative(GATE_LOG)}", "F11 full screen"):
            row.addWidget(muted_label(text, theme.mono_font(8.5)))
        return footer

    # --- sources and models

    def _populate_sources(self) -> None:
        """List the cameras by name (the system default preselected), keeping any video
        files already chosen. Re-runs when a camera is plugged in or out.

        A camera's position in Qt's list is used as its OpenCV index. Both enumerate
        through Media Foundation on Windows (OpenCV's default backend there), so the two
        orders match.
        """
        current = self.source_combo.value()
        files = [(label, value) for label, value in self.source_combo.choices() if not is_camera_source(value)]
        self.source_combo.clear_choices()
        devices = QMediaDevices.videoInputs()
        default_id = QMediaDevices.defaultVideoInput().id()
        default_value = "0" if devices else None
        for index, device in enumerate(devices):
            is_default = device.id() == default_id
            label = device.description() + (" · default" if is_default else "")
            self.source_combo.add_choice(label, str(index), f"Camera {index}: {device.description()}")
            if is_default:
                default_value = str(index)
        for label, value in files:
            self.source_combo.add_choice(label, value, value)
        if not (current and self.source_combo.select_value(current)) and default_value is not None:
            self.source_combo.select_value(default_value)

    def _populate_models(self) -> None:
        """Offer every detector and plate reader in weights/ and runs/; preselect the promoted ones."""
        for combo, paths, default in ((self.detector_combo, find_detector_models(), DEFAULT_DETECTOR),
                                      (self.reader_combo, find_reader_models(), DEFAULT_READER)):
            for path in paths:
                combo.add_choice(model_label(path), str(path), project_relative(path))
            if not combo.select_value(str(default)) and paths:
                combo.select_value(str(paths[0]))

    # --- running

    def _toggle(self) -> None:
        """Start when idle, stop when running."""
        if self.worker is None:
            self._start()
        else:
            self.worker.stop()
            self._set_state("stopping")

    def _start(self) -> None:
        """Launch a worker on the chosen source and models."""
        source, detector, reader = self.source_combo.value(), self.detector_combo.value(), self.reader_combo.value()
        if not (source and detector and reader):
            missing = "a source" if not source else "a detector" if not detector else "a plate reader"
            self._status(f"Choose {missing} first.", error=True)
            return
        camera = is_camera_source(source)
        name = self.source_combo.currentText().replace(" · default", "")
        self.video.clear()
        self.video.set_source(f"LIVE · {name}" if camera else f"FILE · {name}", live=True)
        self.sparkline.clear()
        self._show_plates([])

        self.worker = VideoWorker(source, Path(detector), Path(reader), self.votes_spin.value())
        self.worker.frame_ready.connect(self._on_frame)
        self.worker.plate_locked.connect(self._on_plate_locked)
        self.worker.status.connect(self._on_status)
        self.worker.failed.connect(self._on_failed)
        self.worker.finished.connect(self._on_finished)
        self._set_state("running")
        self.worker.start()

    def _on_status(self, message: str) -> None:
        """Startup progress from the worker."""
        self._status(message)
        if message != "Running":
            self.video.set_message(message)

    def _on_failed(self, message: str) -> None:
        """A worker error: show it in the video area and the footer."""
        self.video.set_message(f"Could not run\n\n{message}")
        self._status(message, error=True)

    def _on_finished(self) -> None:
        """The worker thread has really ended (stopped, finished the file, or failed)."""
        worker, self.worker = self.worker, None
        if worker is not None:
            worker.deleteLater()
        self.video.mark_stopped()
        if self.status_label.property("role") != "error":
            self._status("Stopped")
        self._set_state("idle")

    def _set_state(self, state: str) -> None:
        """idle / running / stopping: the button text and which controls are usable."""
        self.start_button.setText({"idle": "Start", "running": "Stop", "stopping": "Stopping…"}[state])
        self.start_button.setEnabled(state != "stopping")
        for widget in (self.source_combo, self.detector_combo, self.reader_combo, self.votes_spin):
            widget.setEnabled(state == "idle")

    def _status(self, message: str, error: bool = False) -> None:
        """Footer message; errors in the accent color."""
        self.status_label.setText(message)
        set_role(self.status_label, "error" if error else "muted")

    # --- per frame

    def _on_frame(self, image: QImage, info: FrameInfo) -> None:
        """New frame from the worker: video, metrics, side panel."""
        self.video.set_frame(image, info)
        stats = info.stats
        if stats.fps > 0:
            self.frame_ms_label.setText(f"{stats.frame_ms:.1f} ms")
            for key, value in (("detect", stats.detect_ms), ("read", stats.read_ms),
                               ("convert", stats.convert_ms), ("capture", stats.capture_ms)):
                self.stage_labels[key].setText(f"{value:.1f} ms")
            self.stage_labels["fps"].setText(f"{stats.fps:.1f}")
            self.sparkline.add(stats.last_frame_ms, stats.source_fps)
        self.stage_labels["tracks"].setText(str(sum(1 for t in info.tracks if t.track_id is not None)))
        self._show_plates(info.plates)

    def _plate_capacity(self) -> int:
        """How many plate cards fit at full size above the passages card (at least one)."""
        item = self.plate_items[0].sizeHint().height()
        spacing = self.plates_card.body.spacing()
        margins = self.plates_card.body.contentsMargins()
        chrome = margins.top() + margins.bottom() + self.plates_card.title.sizeHint().height() + spacing
        free = self.side_panel.height() - self.side_column.spacing() - self.PASSAGES_MIN - chrome
        return max(1, min(len(self.plate_items), (free + spacing) // (item + spacing)))

    def _show_plates(self, plates: list[PlateView]) -> None:
        """One card per plate in view, in the order they appeared, as many as the height holds
        (locked plates first, then the furthest along); unused cards are hidden."""
        self._plates = plates
        capacity = self._plate_capacity()
        shown = sorted(plates, key=lambda p: (p.locked, p.votes), reverse=True)[:capacity]
        shown.sort(key=lambda p: p.track_id)
        for item, plate in zip(self.plate_items, [*shown, *[None] * len(self.plate_items)]):
            if plate is not None:
                item.show_plate(plate)
            item.setVisible(plate is not None)
        self.no_plate_label.setVisible(not plates)
        locked = sum(plate.locked for plate in plates)
        hidden = f" · {len(shown)} shown" if len(shown) < len(plates) else ""
        self.plates_count.setText(f"{len(plates)} · {locked} locked{hidden}" if plates else "")

    def _on_plate_locked(self, event: PlateEvent) -> None:
        """A number locked: add it to the top of the passages table."""
        self._logged += 1
        self.logged_value.setText(f"{self._logged} logged")
        self.passages.insertRow(0)
        cells = (
            event.when.strftime("%H:%M:%S"),
            format_plate(event.number, event.origin.code),
            event.origin.code or "?",
            f"{event.detector_conf:.2f}",
            f"{event.read_score:.3f}",
        )
        for column, text in enumerate(cells):
            item = QTableWidgetItem(text)
            item.setFont(theme.mono_font(10) if column != 2 else theme.ui_font(10.5))
            if column == 0:
                item.setForeground(QColor(C["muted"]))
            self.passages.setItem(0, column, item)
        if self.passages.rowCount() > MAX_PASSAGE_ROWS:
            self.passages.removeRow(MAX_PASSAGE_ROWS)

    def closeEvent(self, event) -> None:  # noqa: N802 - Qt override name
        """Stop the worker and wait until it has really ended, so no thread outlives the
        window (the loop checks the stop flag every frame; only a model load can make
        this take a few seconds)."""
        if self.worker is not None:
            self.worker.stop()
            self.worker.wait()
        event.accept()


def main() -> None:
    """Open the Visus window."""
    theme.set_app_identity()
    app = QApplication(sys.argv)
    theme.apply_theme(app)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
