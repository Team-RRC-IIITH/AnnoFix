"""Image + mask overlay view. Wheel zooms around the cursor, middle-drag pans. By tool mode:
"select" — left click selects; "paint" — left drag paints; "sam" — left/right click add
include/exclude points and Shift+drag draws a box. Everything is reported in image pixels.
The SAM prompts and the accept preview are drawn on top."""
from __future__ import annotations

import math

import numpy as np
from PyQt6.QtCore import QEvent, QRectF, Qt, pyqtSignal
from PyQt6.QtGui import QBrush, QColor, QImage, QPen, QPixmap
from PyQt6.QtWidgets import (QGraphicsEllipseItem, QGraphicsItem, QGraphicsPixmapItem, QGraphicsRectItem,
                             QGraphicsScene, QGraphicsView)

from .colors import preview_lut
from .overlay_item import OverlayItem


class Canvas(QGraphicsView):
    clicked = pyqtSignal(int, int, bool)        # select: x, y, add to selection (Ctrl held)
    stroke_started = pyqtSignal(int, int)       # paint
    stroke_moved = pyqtSignal(int, int)
    stroke_ended = pyqtSignal()
    sam_clicked = pyqtSignal(int, int, bool)    # sam: x, y, include (left) / exclude (right)
    sam_boxed = pyqtSignal(int, int, int, int)  # sam: Shift+drag box x0, y0, x1, y1

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setScene(QGraphicsScene(self))
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setMouseTracking(True)
        self.setBackgroundBrush(QColor(40, 40, 40))
        self.mode = "select"
        self.brush_radius = 8
        self.overlay: OverlayItem | None = None
        self.sam_preview_item: OverlayItem | None = None
        self._preview_lut = preview_lut()
        self._markers: list[QGraphicsItem] = []
        self._photo = QGraphicsPixmapItem()
        self._photo.setTransformationMode(Qt.TransformationMode.SmoothTransformation)
        self.scene().addItem(self._photo)
        self._cursor = QGraphicsEllipseItem()
        self._cursor.setPen(QPen(QColor(255, 255, 255), 0))
        self._cursor.setZValue(10)
        self._cursor.setVisible(False)
        self.scene().addItem(self._cursor)
        self._band = QGraphicsRectItem()
        band_pen = QPen(QColor(255, 255, 0), 0)
        band_pen.setStyle(Qt.PenStyle.DashLine)
        self._band.setPen(band_pen)
        self._band.setZValue(11)
        self._band.setVisible(False)
        self.scene().addItem(self._band)
        self._painting = False
        self._box_from: tuple[int, int] | None = None
        self._pan_from = None
        self._size = (0, 0)

    def set_image(self, rgb: np.ndarray, label_map: np.ndarray) -> OverlayItem:
        h, w = rgb.shape[:2]
        rgb = np.ascontiguousarray(rgb)
        qimg = QImage(rgb.data, w, h, 3 * w, QImage.Format.Format_RGB888).copy()
        self._photo.setPixmap(QPixmap.fromImage(qimg))
        if self.overlay is not None:
            self.scene().removeItem(self.overlay)
        self.overlay = OverlayItem(label_map)
        self.overlay.setZValue(1)
        self.scene().addItem(self.overlay)
        self.set_sam_preview(None)
        self.set_sam_prompts([], [], None)
        self.scene().setSceneRect(QRectF(0, 0, w, h))
        self._size = (w, h)
        self._painting = False
        self._box_from = None
        self._band.setVisible(False)
        self.resetTransform()
        self.fitInView(self.scene().sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)
        return self.overlay

    def image_pos(self, view_point) -> tuple[int, int]:
        p = self.mapToScene(view_point)
        return math.floor(p.x()), math.floor(p.y())

    def set_masks_visible(self, visible: bool) -> None:
        if self.overlay is not None:
            self.overlay.setVisible(visible)

    def focusNextPrevChild(self, next: bool) -> bool:
        return False  # Tab cycles SAM candidates; it must never move focus off the canvas

    def event(self, ev):
        # An unhandled Tab would propagate to the parent widget, whose focus traversal would
        # move focus anyway; swallow it here (the window's Tab shortcut fires before this).
        if ev.type() == QEvent.Type.KeyPress and ev.key() in (Qt.Key.Key_Tab, Qt.Key.Key_Backtab):
            ev.accept()
            return True
        return super().event(ev)

    # ---- SAM drawing --------------------------------------------------------------------------

    def set_sam_prompts(self, points, labels, box) -> None:
        for item in self._markers:
            self.scene().removeItem(item)
        self._markers = []
        for (x, y), label in zip(points, labels):
            dot = QGraphicsEllipseItem(-5, -5, 10, 10)
            dot.setPos(x + 0.5, y + 0.5)
            dot.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIgnoresTransformations)
            dot.setBrush(QBrush(QColor(40, 220, 40) if label else QColor(230, 30, 60)))
            dot.setPen(QPen(QColor(255, 255, 255), 0))
            dot.setZValue(12)
            self.scene().addItem(dot)
            self._markers.append(dot)
        if box is not None:
            x0, y0, x1, y1 = box
            rect = QGraphicsRectItem(x0, y0, x1 - x0 + 1, y1 - y0 + 1)
            pen = QPen(QColor(255, 255, 0), 0)
            pen.setStyle(Qt.PenStyle.DashLine)
            rect.setPen(pen)
            rect.setZValue(12)
            self.scene().addItem(rect)
            self._markers.append(rect)

    def set_sam_preview(self, preview: np.ndarray | None) -> None:
        if self.sam_preview_item is not None:
            self.scene().removeItem(self.sam_preview_item)
            self.sam_preview_item = None
        if preview is not None:
            item = OverlayItem(preview)
            item.setZValue(2)
            item.setOpacity(0.6)
            item.set_lut(self._preview_lut)
            self.scene().addItem(item)
            self.sam_preview_item = item

    # ---- mouse --------------------------------------------------------------------------------

    def _inside(self, x: int, y: int) -> bool:
        w, h = self._size
        return 0 <= x < w and 0 <= y < h

    def mousePressEvent(self, ev):
        if ev.button() == Qt.MouseButton.MiddleButton:
            self._pan_from = ev.position()
            ev.accept()
            return
        x, y = self.image_pos(ev.position().toPoint())
        if self.mode == "sam" and ev.button() in (Qt.MouseButton.LeftButton, Qt.MouseButton.RightButton):
            if ev.button() == Qt.MouseButton.RightButton:
                self.sam_clicked.emit(x, y, False)
            elif ev.modifiers() & Qt.KeyboardModifier.ShiftModifier:
                self._box_from = (x, y)
                self._band.setRect(x, y, 1, 1)
                self._band.setVisible(True)
            else:
                self.sam_clicked.emit(x, y, True)
            ev.accept()
            return
        if ev.button() == Qt.MouseButton.LeftButton:
            if self.mode == "paint":
                self._painting = True
                self.stroke_started.emit(x, y)
            elif self._inside(x, y):
                add = bool(ev.modifiers() & Qt.KeyboardModifier.ControlModifier)
                self.clicked.emit(x, y, add)
            ev.accept()
            return
        super().mousePressEvent(ev)

    def mouseMoveEvent(self, ev):
        if self._pan_from is not None:
            d = ev.position() - self._pan_from
            self._pan_from = ev.position()
            self.horizontalScrollBar().setValue(self.horizontalScrollBar().value() - int(d.x()))
            self.verticalScrollBar().setValue(self.verticalScrollBar().value() - int(d.y()))
            return
        x, y = self.image_pos(ev.position().toPoint())
        if self._box_from is not None:
            x0, y0 = self._box_from
            self._band.setRect(min(x0, x), min(y0, y), abs(x - x0) + 1, abs(y - y0) + 1)
        r = self.brush_radius
        self._cursor.setRect(x - r, y - r, 2 * r + 1, 2 * r + 1)
        self._cursor.setVisible(self.mode == "paint")
        if self._painting:
            self.stroke_moved.emit(x, y)
        super().mouseMoveEvent(ev)

    def mouseReleaseEvent(self, ev):
        if ev.button() == Qt.MouseButton.MiddleButton:
            self._pan_from = None
            return
        if ev.button() == Qt.MouseButton.LeftButton and self._box_from is not None:
            x, y = self.image_pos(ev.position().toPoint())
            x0, y0 = self._box_from
            self._box_from = None
            self._band.setVisible(False)
            self.sam_boxed.emit(x0, y0, x, y)
            return
        if ev.button() == Qt.MouseButton.LeftButton and self._painting:
            self._painting = False
            self.stroke_ended.emit()
            return
        super().mouseReleaseEvent(ev)

    def wheelEvent(self, ev):
        steps = ev.angleDelta().y() / 120
        if steps:
            factor = 1.25 ** steps
            self.scale(factor, factor)
