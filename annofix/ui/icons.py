"""Line icons for the tool bar and header, drawn as small SVGs (24 x 24, stroked) and coloured
at runtime: dark normally, blue when a tool is on, as in CVAT."""
from __future__ import annotations

from PyQt6.QtCore import QByteArray, QRectF, Qt
from PyQt6.QtGui import QIcon, QPainter, QPixmap
from PyQt6.QtSvg import QSvgRenderer

ICON_COLOUR = "#242424"
ACTIVE_COLOUR = "#1890ff"
DISABLED_COLOUR = "#bfbfbf"

PATHS = {
    "select": '<path d="M6 3.5l12.5 7.2-5.6 1.6 3.4 6.1-2.4 1.3-3.4-6.1L6 17.8z"/>',
    "brush": '<path d="M20.5 3.5l-9 9"/><path d="M14.5 4.5l5 5"/>'
             '<path d="M10.2 13.8c-2.6-.4-4.7 1.4-4.7 3.9 0 1.1-.8 1.8-2 1.8 1.2 1.3 3 1.8 4.7 1.8 3.3 0 5.1-2.4 4.4-5.1z"/>',
    "eraser": '<path d="M15.5 4l5 5-10 10H6l-3-3z"/><path d="M8.5 11l5 5"/><path d="M11 20h9.5"/>',
    "points": '<path d="M5 6.5L17.5 4l2.5 12.5-12 3z"/><circle cx="5" cy="6.5" r="1.9"/>'
              '<circle cx="17.5" cy="4" r="1.9"/><circle cx="20" cy="16.5" r="1.9"/><circle cx="8" cy="19.5" r="1.9"/>',
    "sam": '<path d="M4 20l10-10"/><path d="M15.5 3v3"/><path d="M15.5 12v3"/><path d="M9.5 9h3"/>'
           '<path d="M18.5 9h3"/><path d="M11.8 5.3l1.8 1.8"/><path d="M19.2 5.3l-1.8 1.8"/><path d="M17.4 10.9l1.8 1.8"/>',
    "clean": '<path d="M10 3l1.8 4.7L16.5 9.5l-4.7 1.8L10 16l-1.8-4.7L3.5 9.5l4.7-1.8z"/>'
             '<path d="M18 13.5l.9 2.1 2.1.9-2.1.9-.9 2.1-.9-2.1-2.1-.9 2.1-.9z"/>',
    "protect": '<path d="M12 3l7.5 2.8v5.7c0 4.6-3.2 7.9-7.5 9-4.3-1.1-7.5-4.4-7.5-9V5.8z"/><path d="M8.8 12l2.2 2.2 4.2-4.4"/>',
    "outline": '<rect x="4" y="4" width="16" height="16" rx="3" stroke-dasharray="3 2.4"/>',
    "fit": '<path d="M4 9V4h5"/><path d="M20 9V4h-5"/><path d="M4 15v5h5"/><path d="M20 15v5h-5"/>',
    "save": '<path d="M5 3.5h11l3.5 3.5v13.5h-14.5z"/><path d="M8 3.5v5h7.5v-5"/><path d="M8 20.5v-6.5h8v6.5"/>',
    "undo": '<path d="M9 14L4 9l5-5"/><path d="M4 9h10.5a5.5 5.5 0 010 11H11"/>',
    "redo": '<path d="M15 14l5-5-5-5"/><path d="M20 9H9.5a5.5 5.5 0 000 11H13"/>',
    "prev": '<path d="M15 18l-6-6 6-6"/>',
    "next": '<path d="M9 18l6-6-6-6"/>',
    "next_unreviewed": '<path d="M5.5 18l6-6-6-6"/><path d="M12.5 18l6-6-6-6"/>',
    "reviewed": '<circle cx="12" cy="12" r="8.5"/><path d="M8 12.3l2.7 2.7 5.3-5.6"/>',
    "flag": '<path d="M5.5 21V4"/><path d="M5.5 4.5h11l-2.2 4 2.2 4h-11"/>',
    "eye": '<path d="M2.5 12S6 5.5 12 5.5 21.5 12 21.5 12 18 18.5 12 18.5 2.5 12 2.5 12z"/><circle cx="12" cy="12" r="3"/>',
    "eye_off": '<path d="M2.5 12S6 5.5 12 5.5c2 0 3.7.7 5.1 1.7M21.5 12s-3.5 6.5-9.5 6.5c-2 0-3.7-.7-5.1-1.7"/>'
               '<path d="M4 20L20 4"/>',
    "down": '<path d="M6 9l6 6 6-6"/>',
    "help": '<circle cx="12" cy="12" r="8.5"/><path d="M9.6 9.4a2.5 2.5 0 114 2.1c-.9.5-1.6 1-1.6 2.1v.4"/>'
            '<path d="M12 17.2h.01"/>',
}


def _pixmap(name: str, colour: str, size: int, ratio: float) -> QPixmap:
    svg = (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="{colour}" '
           f'stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round">{PATHS[name]}</svg>')
    pix = QPixmap(round(size * ratio), round(size * ratio))
    pix.fill(Qt.GlobalColor.transparent)
    painter = QPainter(pix)
    QSvgRenderer(QByteArray(svg.encode())).render(painter, QRectF(0, 0, pix.width(), pix.height()))
    painter.end()
    pix.setDevicePixelRatio(ratio)
    return pix


def icon(name: str, size: int = 24, colour: str = ICON_COLOUR) -> QIcon:
    """`name` from PATHS: `colour` (dark), blue when checked, grey when disabled."""
    out = QIcon()
    for ratio in (1.0, 2.0):
        out.addPixmap(_pixmap(name, colour, size, ratio), QIcon.Mode.Normal, QIcon.State.Off)
        out.addPixmap(_pixmap(name, ACTIVE_COLOUR, size, ratio), QIcon.Mode.Normal, QIcon.State.On)
        out.addPixmap(_pixmap(name, DISABLED_COLOUR, size, ratio), QIcon.Mode.Disabled, QIcon.State.Off)
    return out


def toggle_icon(on: str, off: str, size: int = 24) -> QIcon:
    """`on` drawn when the button is checked, `off` when not (e.g. eye / eye struck through)."""
    out = QIcon()
    for ratio in (1.0, 2.0):
        out.addPixmap(_pixmap(on, ICON_COLOUR, size, ratio), QIcon.Mode.Normal, QIcon.State.On)
        out.addPixmap(_pixmap(off, DISABLED_COLOUR, size, ratio), QIcon.Mode.Normal, QIcon.State.Off)
    return out
