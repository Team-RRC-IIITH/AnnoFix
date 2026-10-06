"""Image + mask overlay view. Wheel zooms around the cursor, middle-drag pans. Left clicks
(select tool) and left drags (brush/eraser) are reported in image pixels."""
from __future__ import annotations

import math

import numpy as np
from PyQt6.QtCore import QRectF, Qt, pyqtSignal
from PyQt6.QtGui import QColor, QImage, QPen, QPixmap
from PyQt6.QtWidgets import QGraphicsEllipseItem, QGraphicsPixmapItem, QGraphicsScene, QGraphicsView

from .overlay_item import OverlayItem


class Canvas(QGraphicsView):
    clicked = pyqtSignal(int, int, bool)  # x, y, add to selection (Ctrl held)
    stroke_started = pyqtSignal(int, int)
    stroke_moved = pyqtSignal(int, int)
    stroke_ended = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setScene(QGraphicsScene(self))
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setMouseTracking(True)
        self.setBackgroundBrush(QColor(40, 40, 40))
        self.paint_mode = False
        self.brush_radius = 8
        self.overlay: OverlayItem | None = None
        self._photo = QGraphicsPixmapItem()
        self._photo.setTransformationMode(Qt.TransformationMode.SmoothTransformation)
        self.scene().addItem(self._photo)
        self._cursor = QGraphicsEllipseItem()
        self._cursor.setPen(QPen(QColor(255, 255, 255), 0))
        self._cursor.setZValue(10)
        self._cursor.setVisible(False)
        self.scene().addItem(self._cursor)
        self._painting = False
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
        self.scene().setSceneRect(QRectF(0, 0, w, h))
        self._size = (w, h)
        self._painting = False
        self.resetTransform()
        self.fitInView(self.scene().sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)
        return self.overlay

    def image_pos(self, view_point) -> tuple[int, int]:
        p = self.mapToScene(view_point)
        return math.floor(p.x()), math.floor(p.y())

    def set_masks_visible(self, visible: bool) -> None:
        if self.overlay is not None:
            self.overlay.setVisible(visible)

    def _inside(self, x: int, y: int) -> bool:
        w, h = self._size
        return 0 <= x < w and 0 <= y < h

    def mousePressEvent(self, ev):
        if ev.button() == Qt.MouseButton.MiddleButton:
            self._pan_from = ev.position()
            ev.accept()
            return
        if ev.button() == Qt.MouseButton.LeftButton:
            x, y = self.image_pos(ev.position().toPoint())
            if self.paint_mode:
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
        r = self.brush_radius
        self._cursor.setRect(x - r, y - r, 2 * r + 1, 2 * r + 1)
        self._cursor.setVisible(self.paint_mode)
        if self._painting:
            self.stroke_moved.emit(x, y)
        super().mouseMoveEvent(ev)

    def mouseReleaseEvent(self, ev):
        if ev.button() == Qt.MouseButton.MiddleButton:
            self._pan_from = None
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
