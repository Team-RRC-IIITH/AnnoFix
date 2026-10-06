"""Everything the editor does, without Qt: which image is open, its annotation, the
selection and view options, saving, review and (Task 4) every editing action. The Qt window
only turns mouse/keys into these calls and redraws when notified.

Selection and visibility hold part/object *ids*: undo, redo and rollback rebuild the
Part/AnnObject instances, so holding instances across edits would go stale.

Listeners receive one of:
  "image"       another image (and its annotation) is open
  "annotation"  hierarchy, labels, review status or pixels changed (a finished edit)
  "pixels"      only pixels changed, mid-stroke (cheap: just recolour the overlay)
  "selection"   the selected parts changed
  "view"        colour mode, visibility, solo, tool, brush or protect changed
  "status"      saved state or messages changed
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

from .annotation import Annotation
from .config import Labels
from .io import annotation_paths, has_annotation, list_images, load_annotation, load_image, save_annotation

COLOUR_MODES = ("material", "part", "object")
TOOLS = ("select", "brush", "eraser")


class Session:
    def __init__(self, images_dir: str | Path, ann_dir: str | Path, labels: Labels,
                 annotator: str | None = None):
        self.images = list_images(images_dir)
        if not self.images:
            raise ValueError(f"no colour images in {images_dir}")
        self.ann_dir = Path(ann_dir)
        self.labels = labels
        self.annotator = annotator
        self.index = -1
        self.image: np.ndarray | None = None
        self.ann: Annotation | None = None
        self.selected: list[int] = []
        self.hidden_parts: set[int] = set()
        self.hidden_objects: set[int] = set()
        self.solo_object: int | None = None
        self.colour_mode = "material"
        self.tool = "select"
        self.brush_radius = 8
        self.protect_others = False
        self.messages: list[str] = []
        self._listeners: list[Callable[[str], None]] = []
        self._stroke: tuple[int, int] | None = None  # last point of the open stroke
        self._status_cache: dict[int, str] = {}

    # ---- notifications --------------------------------------------------------------------

    def subscribe(self, fn: Callable[[str], None]) -> None:
        self._listeners.append(fn)

    def _notify(self, kind: str) -> None:
        for fn in list(self._listeners):
            fn(kind)

    def _changed(self) -> None:
        """After any finished edit, undo or redo: drop ids that no longer exist, refresh the
        warnings, tell listeners."""
        part_ids = set(self.ann.part_ids())
        object_ids = {o.id for o in self.ann.objects}
        self.selected = [p for p in self.selected if p in part_ids]
        self.hidden_parts &= part_ids
        self.hidden_objects &= object_ids
        if self.solo_object not in object_ids:
            self.solo_object = None
        self.messages = list(self.ann.warnings) + self.labels.unknown_names(self.ann)
        self._notify("annotation")

    # ---- images ---------------------------------------------------------------------------

    @property
    def stem(self) -> str:
        return self.images[self.index].stem

    @property
    def primary(self) -> int | None:
        return self.selected[0] if self.selected else None

    @property
    def painting(self) -> bool:
        return self._stroke is not None

    def open(self, index: int) -> None:
        """Finish and save the current image, then load image `index` and its annotation (an
        empty one if none exists yet; nothing is written until it is edited). If loading fails
        the current image stays open and the error propagates."""
        if not 0 <= index < len(self.images):
            raise IndexError(f"image index {index} out of range 0..{len(self.images) - 1}")
        self.end_stroke()
        self.save()
        path = self.images[index]
        image = load_image(path)
        h, w = image.shape[:2]
        if has_annotation(self.ann_dir, path.stem):
            ann = load_annotation(self.ann_dir, path.stem, expected_size=(h, w))
        else:
            ann = Annotation.empty(h, w, image=os.path.relpath(path, self.ann_dir), source="manual")
        self.index, self.image, self.ann = index, image, ann
        self.selected, self.hidden_parts, self.hidden_objects = [], set(), set()
        self.solo_object = None
        self.messages = list(ann.warnings) + self.labels.unknown_names(ann)
        self._notify("image")

    def step(self, delta: int) -> bool:
        j = self.index + delta
        if 0 <= j < len(self.images):
            self.open(j)
            return True
        return False

    def save(self) -> bool:
        """Write the open annotation if it has unsaved edits. Never mid-stroke: an open edit is
        left for the next save after it ends."""
        ann = self.ann
        if ann is None or not ann.dirty or ann.in_transaction:
            return False
        if self.annotator and ann.annotator != self.annotator:
            ann.annotator = self.annotator
        save_annotation(ann, self.ann_dir, self.stem)
        self._status_cache[self.index] = ann.review_status
        self._notify("status")
        return True

    def end_stroke(self) -> None:
        """Close an open brush/eraser stroke as one undo step."""
        if self._stroke is None:
            return
        self._stroke = None
        self.ann.commit()
        self._changed()

    # ---- review ---------------------------------------------------------------------------

    def review_status(self, index: int) -> str:
        """'none' (no annotation yet), 'unreviewed', 'reviewed' or 'flagged'."""
        if index == self.index and self.ann is not None:
            if self.ann.dirty or has_annotation(self.ann_dir, self.stem):
                return self.ann.review_status
            return "none"
        if index not in self._status_cache:
            stem = self.images[index].stem
            if not has_annotation(self.ann_dir, stem):
                status = "none"
            else:
                try:
                    meta = json.loads(annotation_paths(self.ann_dir, stem)[1].read_text())
                    status = meta.get("review_status", "unreviewed")
                except (OSError, ValueError):
                    status = "unreviewed"
            self._status_cache[index] = status
        return self._status_cache[index]

    def next_unreviewed(self) -> bool:
        n = len(self.images)
        for k in range(1, n):
            j = (self.index + k) % n
            if self.review_status(j) in ("none", "unreviewed"):
                self.open(j)
                return True
        return False

    def mark_reviewed(self) -> list[str]:
        """Mark reviewed and save, unless validation finds problems (returned; nothing changes)."""
        self.end_stroke()
        problems = self.ann.validate()
        if not problems:
            self.ann.set_review_status("reviewed")
            self._changed()
            self.save()
        return problems

    def toggle_flag(self) -> None:
        self.end_stroke()
        self.ann.set_review_status("unreviewed" if self.ann.review_status == "flagged" else "flagged")
        self._changed()
        self.save()
