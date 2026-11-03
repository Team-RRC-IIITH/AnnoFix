"""Image + mask overlay view. Wheel zooms around the cursor, middle-drag pans (right-drag too,
outside SAM mode). The whole image stays fitted to the view as it resizes until you zoom or pan;
fit() goes back to that. By tool mode: "select" — left click selects; "paint" — left drag
paints; "sam" — left/right click add include/exclude points and Shift+drag draws a box;
"points" — drag the selected part's outline points, double-click an edge to add one, Ctrl+click
one to delete it, click elsewhere to select. Everything is reported in image pixels. The SAM
prompts, the accept preview and the outline points are drawn on top."""
from __future__ import annotations

import math

import numpy as np
from PyQt6.QtCore import QEvent, QPoint, QPointF, QRectF, Qt, pyqtSignal
from PyQt6.QtGui import QBrush, QColor, QFont, QFontMetricsF, QImage, QPainter, QPainterPath, QPen, QPixmap, QPolygonF
from PyQt6.QtWidgets import (QGraphicsEllipseItem, QGraphicsItem, QGraphicsPathItem, QGraphicsPixmapItem,
                             QGraphicsRectItem, QGraphicsScene, QGraphicsView)

from .colors import preview_lut
from .overlay_item import OverlayItem

HANDLE_PX = 8  # how close (screen pixels) a click must be to an outline point or edge


CLICK_SLOP = 4  # px a right press may move and still count as a click, not a pan


class Tag(QGraphicsItem):
    """A part's name and material written on the canvas, the same size at any zoom: the text CVAT
    shows on its shapes."""

    def __init__(self, x: int, y: int, text: str):
        super().__init__()
        self.text = text
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIgnoresTransformations)
        self.setPos(x + 0.5, y + 0.5)
        self.setZValue(9)
        self._font = QFont()
        self._font.setPixelSize(11)
        metrics = QFontMetricsF(self._font)
        w, h = metrics.horizontalAdvance(text) + 10, metrics.height() + 4
        self._rect = QRectF(-w / 2, -h / 2, w, h)

    def boundingRect(self) -> QRectF:
        return self._rect

    def paint(self, painter, option, widget=None) -> None:
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(0, 0, 0, 150))
        painter.drawRoundedRect(self._rect, 4, 4)
        painter.setPen(QColor(255, 255, 255))
        painter.setFont(self._font)
        painter.drawText(self._rect, Qt.AlignmentFlag.AlignCenter, self.text)


class Canvas(QGraphicsView):
    clicked = pyqtSignal(int, int, bool)        # select: x, y, add to selection (Ctrl held)
    stroke_started = pyqtSignal(int, int)       # paint
    stroke_moved = pyqtSignal(int, int)
    stroke_ended = pyqtSignal()
    sam_clicked = pyqtSignal(int, int, bool)    # sam: x, y, include (left) / exclude (right)
    sam_boxed = pyqtSignal(int, int, int, int)  # sam: Shift+drag box x0, y0, x1, y1
    point_moved = pyqtSignal(int, int, int, int)     # points: ring, index, dropped at x, y
    point_inserted = pyqtSignal(int, int, int, int)  # points: ring, after index, x, y
    point_deleted = pyqtSignal(int, int)             # points: ring, index
    card_requested = pyqtSignal(int, int, QPoint)    # right-click without a drag: x, y, screen pos
    points_boxed = pyqtSignal(int, int, int, int)    # points: a box dragged over outline points

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setScene(QGraphicsScene(self))
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setMouseTracking(True)
        self.setBackgroundBrush(QColor(255, 255, 255))  # CVAT's canvas background
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
        self._cursor_outline = QGraphicsEllipseItem()  # black around the white ring: seen on any pixels
        self._cursor_outline.setPen(QPen(QColor(0, 0, 0), 0))
        self._cursor_outline.setZValue(10)
        self._cursor_outline.setVisible(False)
        self.scene().addItem(self._cursor_outline)
        self._band = QGraphicsRectItem()
        band_pen = QPen(QColor(255, 255, 0), 0)
        band_pen.setStyle(Qt.PenStyle.DashLine)
        self._band.setPen(band_pen)
        self._band.setZValue(11)
        self._band.setVisible(False)
        self.scene().addItem(self._band)
        self._rings: list[np.ndarray] | None = None
        self._paths: list[QGraphicsPathItem] = []        # one outline per ring
        self._handles: list[list[QGraphicsItem]] = []
        self._drag: tuple[int, int, np.ndarray] | None = None  # ring, index, where the point was
        self._painting = False
        self._box_from: tuple[int, int] | None = None
        self._pan_from = None
        self._pan_button = Qt.MouseButton.NoButton
        self._pan_start = None  # where a right press began: released there, it asks for the card
        self.tag_items: list[Tag] = []
        self.selection_items: list[QGraphicsPathItem] = []  # the selected parts' outline
        self._picked: tuple[int, set[int]] | None = None  # points: (ring, indices) drawn blue
        self._box_press = None  # points: where a press on the image began (a click, or a box)
        self._tags_on = True
        self._masks_visible = True
        self._size = (0, 0)
        self._auto_fit = True

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
        self.set_rings(None)
        self.scene().setSceneRect(QRectF(0, 0, w, h))
        self._size = (w, h)
        self._painting = False
        self._box_from = None
        self._band.setVisible(False)
        self.fit()
        return self.overlay

    def fit(self) -> None:
        """Show the whole image; it then stays fitted as the view resizes, until you zoom or pan."""
        self._auto_fit = True
        self.resetTransform()
        self.fitInView(self.scene().sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)

    def resizeEvent(self, ev) -> None:
        super().resizeEvent(ev)
        if self._auto_fit:
            self.fit()

    def image_pos(self, view_point) -> tuple[int, int]:
        p = self.mapToScene(view_point)
        return math.floor(p.x()), math.floor(p.y())

    def set_masks_visible(self, visible: bool) -> None:
        if self.overlay is not None:
            self.overlay.setVisible(visible)
        self._masks_visible = visible
        self._show_tags()
        for item in self.selection_items:
            item.setVisible(visible)

    def set_selection_outline(self, rings) -> None:
        """Outline the selected parts ((N, 2) pixel rings): white under blue, the same width at
        any zoom, so even a sliver of a part stands out."""
        for item in self.selection_items:
            self.scene().removeItem(item)
        self.selection_items = []
        if not rings:
            return
        path = QPainterPath()
        for ring in rings:
            path.addPolygon(QPolygonF([QPointF(x + 0.5, y + 0.5) for x, y in ring]))
            path.closeSubpath()
        for colour, width in ((QColor(255, 255, 255), 4), (QColor(24, 144, 255), 2)):
            pen = QPen(colour, width)
            pen.setCosmetic(True)
            pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
            item = QGraphicsPathItem(path)
            item.setPen(pen)
            item.setZValue(12)
            item.setVisible(self._masks_visible)
            self.scene().addItem(item)
            self.selection_items.append(item)

    def set_tags(self, tags) -> None:
        """(x, y, text) per part, written on the image."""
        for item in self.tag_items:
            self.scene().removeItem(item)
        self.tag_items = [Tag(x, y, text) for x, y, text in tags]
        for item in self.tag_items:
            self.scene().addItem(item)
        self._show_tags()

    def set_tags_visible(self, on: bool) -> None:
        self._tags_on = on
        self._show_tags()

    def _show_tags(self) -> None:
        for item in self.tag_items:
            item.setVisible(self._tags_on and self._masks_visible)

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
            item = OverlayItem(preview, lines=False)  # its colours are a legend: no extra shades
            item.setZValue(2)
            item.setOpacity(0.6)
            item.set_lut(self._preview_lut)
            self.scene().addItem(item)
            self.sam_preview_item = item

    # ---- outline points -----------------------------------------------------------------------

    def set_rings(self, rings) -> None:
        """Show these outline rings ((N, 2) x, y pixel arrays) as draggable points, or none."""
        rings = [np.array(r) for r in rings] if rings else None
        if self._drag is None and self._same_rings(rings):
            return  # nothing to redraw (a big part can have thousands of points)
        if self._drag is None and rings and self._rings and len(rings) == len(self._rings):
            for k, ring in enumerate(rings):  # an edit changes one ring: redraw just that one
                if not np.array_equal(ring, self._rings[k]):
                    self._rings[k] = ring
                    self._redraw_ring(k)
            return
        self._drag = None
        self._rings = rings
        for item in [*self._paths, *(h for row in self._handles for h in row)]:
            self.scene().removeItem(item)
        self._paths, self._handles = [], []
        pen = QPen(QColor(255, 255, 0), 2)
        pen.setCosmetic(True)
        for ring in self._rings or []:
            path = QGraphicsPathItem()
            path.setPen(pen)
            path.setZValue(13)
            self.scene().addItem(path)
            self._paths.append(path)
            self._handles.append([self._handle(x, y) for x, y in ring])
        for k in range(len(self._paths)):
            self._draw_outline(k)
        self._colour_handles()

    def _redraw_ring(self, k: int) -> None:
        for item in self._handles[k]:
            self.scene().removeItem(item)
        self._handles[k] = [self._handle(x, y) for x, y in self._rings[k]]
        self._draw_outline(k)
        self._colour_handles()

    def set_selected_points(self, pick) -> None:
        """(ring, indices) of the outline points picked with a box, drawn blue; None for none."""
        self._picked = None if pick is None else (pick[0], set(pick[1]))
        self._colour_handles()

    def _colour_handles(self) -> None:
        for k, row in enumerate(self._handles):
            for i, handle in enumerate(row):
                picked = self._picked is not None and self._picked[0] == k and i in self._picked[1]
                handle.setBrush(QBrush(QColor(24, 144, 255) if picked else QColor(255, 255, 255)))

    def _handle(self, x: int, y: int) -> QGraphicsRectItem:
        handle = QGraphicsRectItem(-3.5, -3.5, 7, 7)
        handle.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIgnoresTransformations)
        handle.setPos(x + 0.5, y + 0.5)
        handle.setBrush(QBrush(QColor(255, 255, 255)))
        handle.setPen(QPen(QColor(0, 0, 0), 0))
        handle.setZValue(14)
        self.scene().addItem(handle)
        return handle

    def _same_rings(self, rings) -> bool:
        if rings is None or self._rings is None:
            return rings is None and self._rings is None
        return len(rings) == len(self._rings) and all(np.array_equal(a, b) for a, b in zip(rings, self._rings))

    def _draw_outline(self, k: int) -> None:
        path = QPainterPath()
        path.addPolygon(QPolygonF([QPointF(x + 0.5, y + 0.5) for x, y in self._rings[k]]))
        path.closeSubpath()
        self._paths[k].setPath(path)

    def _point_at(self, view_point) -> tuple[int, int] | None:
        """(ring, index) of the outline point under the cursor, if any."""
        p, scale = self.mapToScene(view_point), self.transform().m11() or 1.0
        best = None
        for k, ring in enumerate(self._rings or []):
            d = np.hypot(ring[:, 0] + 0.5 - p.x(), ring[:, 1] + 0.5 - p.y()) * scale
            i = int(d.argmin())
            if d[i] <= HANDLE_PX and (best is None or d[i] < best[0]):
                best = (d[i], k, i)
        return None if best is None else best[1:]

    def _edge_at(self, view_point) -> tuple[int, int, int, int] | None:
        """(ring, index of the edge's first point, x, y) of the outline edge under the cursor,
        x, y being the pixel on that edge nearest the cursor (zoomed out, the cursor can be
        many pixels off the edge)."""
        p, scale = self.mapToScene(view_point), self.transform().m11() or 1.0
        q = np.array([p.x(), p.y()])
        best = None
        for k, ring in enumerate(self._rings or []):
            a = ring + 0.5
            ab = np.roll(a, -1, axis=0) - a
            t = np.clip(((q - a) * ab).sum(1) / np.maximum((ab * ab).sum(1), 1e-9), 0, 1)
            nearest = a + t[:, None] * ab
            d = np.hypot(*(nearest - q).T) * scale
            i = int(d.argmin())
            if d[i] <= HANDLE_PX and (best is None or d[i] < best[0]):
                x, y = np.rint(nearest[i] - 0.5).astype(int)
                best = (d[i], k, i, int(x), int(y))
        return None if best is None else best[1:]

    def _show_drag(self, x: int, y: int) -> None:
        k, i, _ = self._drag
        self._rings[k][i] = (x, y)
        self._handles[k][i].setPos(x + 0.5, y + 0.5)
        self._draw_outline(k)

    def cancel_point_drag(self) -> bool:
        """Esc or a lost release: put the dragged point back; nothing is reported."""
        if self._drag is None:
            return False
        k, i, start = self._drag
        self._show_drag(int(start[0]), int(start[1]))
        self._drag = None
        return True

    def _points_press(self, ev) -> None:
        pos = ev.position().toPoint()
        hit = self._point_at(pos)
        if hit is not None and ev.modifiers() & Qt.KeyboardModifier.ControlModifier:
            self.point_deleted.emit(*hit)
        elif hit is not None:
            k, i = hit
            self._drag = (k, i, self._rings[k][i].copy())
        elif self._edge_at(pos) is None:  # on an edge, wait: it may be a double-click to add a point
            x, y = self.image_pos(pos)
            if self._inside(x, y):  # a click selects; a drag boxes points (released in mouseReleaseEvent)
                self._box_from, self._box_press = (x, y), ev.position()
                self._band.setRect(x, y, 1, 1)
                self._band.setVisible(True)

    def mouseDoubleClickEvent(self, ev):
        if self.mode == "points" and ev.button() == Qt.MouseButton.LeftButton:
            pos = ev.position().toPoint()
            edge = self._edge_at(pos) if self._point_at(pos) is None else None
            if edge is not None:
                self.point_inserted.emit(*edge)
            ev.accept()
            return
        super().mouseDoubleClickEvent(ev)

    # ---- mouse --------------------------------------------------------------------------------

    def _inside(self, x: int, y: int) -> bool:
        w, h = self._size
        return 0 <= x < w and 0 <= y < h

    def _pans(self, button) -> bool:
        if button == Qt.MouseButton.MiddleButton:
            return True
        # Right-drag pans too (touchpads have no middle button), except in SAM mode where the
        # right button places exclude points, and never during a stroke.
        return button == Qt.MouseButton.RightButton and self.mode != "sam" and not self._painting

    def mousePressEvent(self, ev):
        if self._pan_from is None and self._pans(ev.button()):
            self._pan_from = ev.position()
            self._pan_button = ev.button()
            self._pan_start = ev.position() if ev.button() == Qt.MouseButton.RightButton else None
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
        if self.mode == "points" and ev.button() == Qt.MouseButton.LeftButton:
            self._points_press(ev)
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
        if self._pan_from is not None and not ev.buttons() & self._pan_button:
            self._pan_from = None  # the release went to another window
        if self._pan_from is not None:
            d = ev.position() - self._pan_from
            self._pan_from = ev.position()
            self._auto_fit = False
            self.horizontalScrollBar().setValue(self.horizontalScrollBar().value() - int(d.x()))
            self.verticalScrollBar().setValue(self.verticalScrollBar().value() - int(d.y()))
            return
        if not ev.buttons() & Qt.MouseButton.LeftButton:
            self._drop_lost_press()
        x, y = self.image_pos(ev.position().toPoint())
        if self._drag is not None:
            self._show_drag(x, y)
            return
        if self._box_from is not None:
            x0, y0 = self._box_from
            self._band.setRect(min(x0, x), min(y0, y), abs(x - x0) + 1, abs(y - y0) + 1)
        r = self.brush_radius
        self._cursor.setRect(x - r, y - r, 2 * r + 1, 2 * r + 1)
        self._cursor.setVisible(self.mode == "paint")
        self._cursor_outline.setRect(x - r - 1, y - r - 1, 2 * r + 3, 2 * r + 3)
        self._cursor_outline.setVisible(self.mode == "paint")
        if self._painting:
            self.stroke_moved.emit(x, y)
        super().mouseMoveEvent(ev)

    def _drop_lost_press(self) -> None:
        """The left button is up but its release never reached us (it was let go over another
        window, or a dialog took the mouse): end the stroke there, drop a half-drawn box."""
        if self._painting:
            self._painting = False
            self.stroke_ended.emit()
        if self._box_from is not None:
            self._box_from = None
            self._band.setVisible(False)
        self.cancel_point_drag()

    def mouseReleaseEvent(self, ev):
        if self._pan_from is not None and ev.button() == self._pan_button:
            self._pan_from = None
            start, self._pan_start = self._pan_start, None
            if start is not None and (ev.position() - start).manhattanLength() <= CLICK_SLOP:
                x, y = self.image_pos(ev.position().toPoint())
                if self._inside(x, y):  # a right-click, not a pan: CVAT's card for that mask
                    self.card_requested.emit(x, y, ev.globalPosition().toPoint())
            return
        if ev.button() == Qt.MouseButton.LeftButton and self._box_from is not None:
            x, y = self.image_pos(ev.position().toPoint())
            x0, y0 = self._box_from
            self._box_from = None
            self._band.setVisible(False)
            if self.mode == "points":
                press, self._box_press = self._box_press, None
                if press is not None and (ev.position() - press).manhattanLength() <= CLICK_SLOP:
                    self.clicked.emit(x0, y0, False)
                else:
                    self.points_boxed.emit(x0, y0, x, y)
            else:
                self.sam_boxed.emit(x0, y0, x, y)
            return
        if ev.button() == Qt.MouseButton.LeftButton and self._painting:
            self._painting = False
            self.stroke_ended.emit()
            return
        if ev.button() == Qt.MouseButton.LeftButton and self._drag is not None:
            k, i, start = self._drag
            x, y = self.image_pos(ev.position().toPoint())
            self.cancel_point_drag()  # shown as it was until the Session applies the move
            if (x, y) != (int(start[0]), int(start[1])):
                self.point_moved.emit(k, i, x, y)
            return
        super().mouseReleaseEvent(ev)

    def wheelEvent(self, ev):
        steps = ev.angleDelta().y() / 120
        if steps:
            factor = 1.25 ** steps
            self._auto_fit = False
            self.scale(factor, factor)
