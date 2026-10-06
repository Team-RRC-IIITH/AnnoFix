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

    # ---- selection --------------------------------------------------------------------------

    def select_part(self, part_id: int | None, add: bool = False) -> None:
        if part_id is None:
            if not add:
                self.selected = []
        elif add:
            if part_id in self.selected:
                self.selected.remove(part_id)
            else:
                self.selected.append(part_id)
        else:
            self.selected = [part_id]
        self._notify("selection")

    def select_at(self, x: int, y: int, add: bool = False) -> None:
        h, w = self.ann.size
        self.select_part(self.ann.part_at(y, x) if 0 <= x < w and 0 <= y < h else None, add)

    def select_next_object(self, step: int = 1) -> None:
        objects = [o for o in self.ann.objects if o.parts]
        if not objects:
            return
        ids = [o.id for o in objects]
        current = self.ann.object_of(self.primary).id if self.primary is not None else None
        i = (ids.index(current) + step) % len(ids) if current in ids else (0 if step > 0 else len(ids) - 1)
        self.select_part(objects[i].parts[0].id)

    # ---- labels -------------------------------------------------------------------------------

    def set_material(self, material: str) -> None:
        self.end_stroke()
        ids = list(self.selected)
        if not ids:
            return
        with self.ann.transaction("set material"):
            for pid in ids:
                self.ann.set_part_material(pid, material)
        self._changed()

    def set_material_index(self, index: int) -> bool:
        if not self.selected or not 0 <= index < len(self.labels.materials):
            return False
        self.set_material(self.labels.materials[index])
        return True

    def set_part_name(self, name: str) -> None:
        self.end_stroke()
        if self.primary is not None:
            self.ann.set_part_name(self.primary, name.strip() or None)
            self._changed()

    def set_object_class(self, cls: str) -> None:
        self.end_stroke()
        if self.primary is not None:
            self.ann.set_object_class(self.ann.object_of(self.primary).id, cls.strip() or None)
            self._changed()

    # ---- view ---------------------------------------------------------------------------------

    def set_colour_mode(self, mode: str) -> None:
        if mode not in COLOUR_MODES:
            raise ValueError(f"colour mode must be one of {COLOUR_MODES}, got {mode!r}")
        self.colour_mode = mode
        self._notify("view")

    def set_hidden(self, kind: str, item_id: int, hidden: bool) -> None:
        target = self.hidden_parts if kind == "part" else self.hidden_objects
        if hidden:
            target.add(item_id)
        else:
            target.discard(item_id)
        self._notify("view")

    def toggle_part_hidden(self, part_id: int) -> None:
        self.set_hidden("part", part_id, part_id not in self.hidden_parts)

    def toggle_object_hidden(self, object_id: int) -> None:
        self.set_hidden("object", object_id, object_id not in self.hidden_objects)

    def toggle_solo(self) -> None:
        if self.solo_object is not None:
            self.solo_object = None
        elif self.primary is not None:
            self.solo_object = self.ann.object_of(self.primary).id
        self._notify("view")

    def set_tool(self, tool: str) -> None:
        if tool not in TOOLS:
            raise ValueError(f"tool must be one of {TOOLS}, got {tool!r}")
        self.end_stroke()
        self.tool = tool
        self._notify("view")

    def set_brush_radius(self, radius: int) -> None:
        self.brush_radius = int(min(200, max(1, radius)))
        self._notify("view")

    def set_protect(self, on: bool) -> None:
        self.protect_others = bool(on)
        self._notify("view")

    # ---- brush and eraser ---------------------------------------------------------------------

    def begin_stroke(self, x: int, y: int) -> bool:
        """Start a brush (paints the selected part) or eraser (erases the selected part, or any
        part when nothing is selected) stroke. The stroke is one undo step."""
        if self.tool not in ("brush", "eraser") or self.painting:
            return False
        if self.tool == "brush" and self.primary is None:
            self.messages = self.messages + ["select a part (or press N for a new one) before painting"]
            self._notify("status")
            return False
        self.ann.begin(self.tool)
        self._stroke = (x, y)
        self._stamp(x, y, x, y)
        return True

    def stroke_to(self, x: int, y: int) -> None:
        if self._stroke is None:
            return
        x0, y0 = self._stroke
        self._stroke = (x, y)
        self._stamp(x0, y0, x, y)

    def cancel_stroke(self) -> None:
        if self._stroke is None:
            return
        self._stroke = None
        self.ann.rollback()
        self._changed()

    def _stamp(self, x0: int, y0: int, x1: int, y1: int) -> None:
        # A thick line between consecutive mouse positions: mouse events are sparse, so
        # stamping circles per event would leave gaps. cv2 draws round caps and clips.
        canvas = np.zeros(self.ann.size, np.uint8)
        cv2.line(canvas, (int(x0), int(y0)), (int(x1), int(y1)), 1, thickness=2 * self.brush_radius + 1)
        mask = canvas.view(bool)
        if self.tool == "eraser":
            self.ann.erase(mask, self.primary)
        else:
            self.ann.assign(mask, self.primary, self.protect_others)
        self._notify("pixels")

    # ---- structure ----------------------------------------------------------------------------

    def _select_new(self, part_id: int) -> None:
        self.selected = [part_id]
        self.tool = "brush"
        self._changed()
        self._notify("selection")
        self._notify("view")

    def new_object(self) -> int:
        """A new object with one empty part, selected, brush ready."""
        self.end_stroke()
        with self.ann.transaction("new object"):
            pid = self.ann.new_part(self.ann.new_object())
        self._select_new(pid)
        return pid

    def new_part(self) -> int:
        """A new empty part in the selected part's object (or a new object), brush ready."""
        self.end_stroke()
        if self.primary is None:
            return self.new_object()
        pid = self.ann.new_part(self.ann.object_of(self.primary).id)
        self._select_new(pid)
        return pid

    def merge_selected_parts(self) -> bool:
        """The other selected parts' pixels join the first selected part."""
        self.end_stroke()
        if len(self.selected) < 2:
            return False
        keep, others = self.selected[0], self.selected[1:]
        with self.ann.transaction("merge parts"):
            for pid in others:
                self.ann.merge_parts(keep, pid)
        self._changed()
        return True

    def merge_selected_objects(self) -> bool:
        """Every part of the other selected parts' objects moves into the first selected
        part's object (e.g. a robot arm split into links and base)."""
        self.end_stroke()
        if len(self.selected) < 2:
            return False
        keep = self.ann.object_of(self.selected[0]).id
        others: list[int] = []
        for pid in self.selected[1:]:
            oid = self.ann.object_of(pid).id
            if oid != keep and oid not in others:
                others.append(oid)
        if not others:
            return False
        with self.ann.transaction("merge objects"):
            for oid in others:
                for pid in [p.id for p in self.ann.object(oid).parts]:
                    self.ann.move_part(pid, keep)
                self.ann.delete_object(oid)
        self._changed()
        return True

    def delete_parts(self, part_ids) -> None:
        self.end_stroke()
        ids = list(part_ids)
        if not ids:
            return
        with self.ann.transaction("delete part"):
            for pid in ids:
                self.ann.delete_part(pid)
        self._changed()

    def delete_selected_parts(self) -> None:
        self.delete_parts(self.selected)

    def delete_object(self, object_id: int) -> None:
        self.end_stroke()
        self.ann.delete_object(object_id)
        self._changed()

    def delete_selected_object(self) -> None:
        if self.primary is not None:
            self.delete_object(self.ann.object_of(self.primary).id)

    def move_part_to(self, part_id: int, object_id: int) -> None:
        self.end_stroke()
        self.ann.move_part(part_id, object_id)
        self._changed()

    def promote_part(self, part_id: int) -> int:
        self.end_stroke()
        oid = self.ann.promote_part(part_id)
        self._changed()
        return oid

    def clean_part(self, part_id: int) -> None:
        self.end_stroke()
        self.ann.clean_part(part_id)
        self._changed()

    # ---- history ------------------------------------------------------------------------------

    def undo(self) -> str | None:
        if self.painting:
            return None
        label = self.ann.undo()
        if label is not None:
            self._changed()
        return label

    def redo(self) -> str | None:
        if self.painting:
            return None
        label = self.ann.redo()
        if label is not None:
            self._changed()
        return label
