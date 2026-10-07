"""Shared test data and Qt helpers (tests/ is on sys.path, so `from helpers import ...`)."""
import json
from pathlib import Path

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


def send_move(widget, pos, buttons=None):
    """Deliver a mouse-move with `buttons` held, the left button by default (QTest.mouseMove
    is unreliable on QGraphicsView viewports)."""
    from PyQt6.QtCore import QEvent, QPointF, Qt
    from PyQt6.QtGui import QMouseEvent
    from PyQt6.QtWidgets import QApplication

    held = Qt.MouseButton.LeftButton if buttons is None else buttons
    ev = QMouseEvent(QEvent.Type.MouseMove, QPointF(pos), QPointF(widget.mapToGlobal(pos)),
                     Qt.MouseButton.NoButton, held, Qt.KeyboardModifier.NoModifier)
    QApplication.sendEvent(widget, ev)


def send_wheel(widget, pos, steps=1):
    """One mouse-wheel turn of `steps` notches over `pos`."""
    from PyQt6.QtCore import QPoint, QPointF, Qt
    from PyQt6.QtGui import QWheelEvent
    from PyQt6.QtWidgets import QApplication

    ev = QWheelEvent(QPointF(pos), QPointF(widget.mapToGlobal(pos)), QPoint(0, 0), QPoint(0, 120 * steps),
                     Qt.MouseButton.NoButton, Qt.KeyboardModifier.NoModifier, Qt.ScrollPhase.NoScrollPhase, False)
    QApplication.sendEvent(widget, ev)


MATERIALS10 = ["fabric", "foliage", "glass", "leather", "metal", "paper", "plastic", "stone", "water", "wood"]


def rect_poly(x0, y0, x1, y1):
    """A pipeline-convention polygon covering pixels x0..x1, y0..y1 (inclusive)."""
    return [x0 + .5, y0 + .5, x1 + .5, y0 + .5, x1 + .5, y1 + .5, x0 + .5, y1 + .5]


def make_cvat_sample(root, name="s1", objects=None, size=(40, 60)):
    """A pipeline-style sample folder: images/<name>.png, annotations/instances_default.json,
    export_manifest.json, cvat_labels.json. Default: mug o0 (plastic, pixels 5..24 square) with
    handle p0_1 (metal, x 20..24, y 10..19), and box o1 (paper, x 30..49, y 10..29)."""
    h, w = size
    d = Path(root) / name
    (d / "images").mkdir(parents=True)
    (d / "annotations").mkdir()
    cv2.imwrite(str(d / "images" / f"{name}.png"), np.full((h, w, 3), 120, np.uint8))
    if objects is None:
        objects = [
            dict(uid="o0", kind="object", object_id=0, object_name="mug", part_name="", material="plastic",
                 conf=90, poly=rect_poly(5, 5, 24, 24)),
            dict(uid="p0_1", kind="part", object_id=0, object_name="mug", part_name="handle", material="metal",
                 conf=40, poly=rect_poly(20, 10, 24, 19)),
            dict(uid="o1", kind="object", object_id=1, object_name="box", part_name="", material="paper",
                 conf=70, poly=rect_poly(30, 10, 49, 29)),
        ]
    anns = []
    for i, o in enumerate(objects, 1):
        anns.append({"id": i, "image_id": 1, "category_id": MATERIALS10.index(o["material"]) + 1,
                     "segmentation": [o["poly"]], "area": 0, "bbox": [0, 0, 0, 0], "iscrowd": 0,
                     "attributes": {"uid": o["uid"], "kind": o["kind"], "object_id": o["object_id"],
                                    "object_name": o["object_name"], "part_name": o["part_name"],
                                    "confidence_pct": o["conf"], "reviewed": False, "occluded": False}})
    coco = {"info": {"description": "test"}, "licenses": [],
            "categories": [{"id": i + 1, "name": m, "supercategory": ""} for i, m in enumerate(MATERIALS10)],
            "images": [{"id": 1, "file_name": f"{name}.png", "width": w, "height": h}], "annotations": anns}
    (d / "annotations" / "instances_default.json").write_text(json.dumps(coco))
    manifest = {"source": "test.npz", "file_name": f"{name}.png", "size": [h, w], "classes": MATERIALS10,
                "annotations": {o["uid"]: {"kind": o["kind"], "object": f"sam3_{o['object_id']:03d}",
                                           "object_id": o["object_id"], "object_name": o["object_name"],
                                           "part_name": o["part_name"], "material": o["material"],
                                           "confidence": o["conf"] / 100} for o in objects}}
    (d / "export_manifest.json").write_text(json.dumps(manifest))
    (d / "cvat_labels.json").write_text(json.dumps(
        [{"name": m, "color": "#000000", "type": "any", "attributes": []} for m in MATERIALS10]))
    return d
