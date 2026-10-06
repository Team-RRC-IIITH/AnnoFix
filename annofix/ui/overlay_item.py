"""The mask overlay: a QGraphicsItem drawing a uint32 ARGB buffer that numpy recolours in
place. The QImage wraps the buffer (no copy), so a stroke costs one recolour, not a QPixmap
conversion."""
from __future__ import annotations

import numpy as np
from PyQt6.QtCore import QPointF, QRectF
from PyQt6.QtGui import QImage, QPainter
from PyQt6.QtWidgets import QGraphicsItem

from .colors import LUT_SIZE, recolour


class OverlayItem(QGraphicsItem):
    def __init__(self, label_map: np.ndarray):
        super().__init__()
        self._label = label_map  # the Annotation's own array, edited in place
        h, w = label_map.shape
        self._buf = np.zeros((h, w), np.uint32)  # must outlive the QImage that wraps it
        self._edges = np.zeros((h, w), bool)
        self._image = QImage(self._buf.data, w, h, 4 * w, QImage.Format.Format_ARGB32_Premultiplied)
        self._lut = np.zeros(LUT_SIZE, np.uint32)
        self.outline = False

    def boundingRect(self) -> QRectF:
        return QRectF(0, 0, self._buf.shape[1], self._buf.shape[0])

    def set_lut(self, lut: np.ndarray) -> None:
        self._lut = lut
        self.refresh()

    def set_outline(self, on: bool) -> None:
        self.outline = on
        self.refresh()

    def refresh(self) -> None:
        recolour(self._label, self._lut, self._buf, outline=self.outline, edge_buf=self._edges)
        self.update()

    def argb_at(self, x: int, y: int) -> int:
        return int(self._image.pixel(x, y))

    def paint(self, painter: QPainter, option, widget=None) -> None:
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, False)
        painter.drawImage(QPointF(0, 0), self._image)
