"""Shared test data and Qt helpers (tests/ is on sys.path, so `from helpers import ...`)."""
import cv2
import numpy as np

from annofix.annotation import AnnObject, Annotation, Part
from annofix.config import Labels
from annofix.io import save_annotation

LABELS = Labels(materials=("plastic", "metal", "rubber"), objects={"mug": ("body", "handle")})
H, W = 40, 60


def scene_annotation() -> Annotation:
    """mug #1 = body #2 (metal) + handle #3 (no material); object #4 'spaceship' = part #5."""
    lm = np.zeros((H, W), np.uint16)
    lm[5:15, 5:25] = 2
    lm[5:15, 30:40] = 3
    lm[20:30, 5:15] = 5
    objects = [AnnObject(1, "mug", [Part(2, "body", "metal"), Part(3, "handle", None)]),
               AnnObject(4, "spaceship", [Part(5, None, None)])]
    return Annotation(lm, objects, next_id=6)


def make_dataset(tmp_path):
    """images/a.png (40x60, annotated), b.png (40x60, no annotation), c.png (30x50, none)."""
    images, ann_dir = tmp_path / "images", tmp_path / "ann"
    images.mkdir()
    for name, (h, w) in {"a": (H, W), "b": (H, W), "c": (30, 50)}.items():
        cv2.imwrite(str(images / f"{name}.png"), np.full((h, w, 3), 100, np.uint8))
    save_annotation(scene_annotation(), ann_dir, "a")
    return images, ann_dir


def send_move(widget, pos):
    """Deliver a mouse-move with the left button held (QTest.mouseMove is unreliable on
    QGraphicsView viewports)."""
    from PyQt6.QtCore import QEvent, QPointF, Qt
    from PyQt6.QtGui import QMouseEvent
    from PyQt6.QtWidgets import QApplication

    ev = QMouseEvent(QEvent.Type.MouseMove, QPointF(pos), QPointF(widget.mapToGlobal(pos)),
                     Qt.MouseButton.NoButton, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier)
    QApplication.sendEvent(widget, ev)
