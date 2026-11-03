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
  "saved"       the open image was just written to disk
"""
from __future__ import annotations

import itertools
import json
import os
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

from .annotation import REVIEW_STATUSES, Annotation
from .config import Labels
from .io import annotation_paths, has_annotation, list_images, load_annotation, load_image, save_annotation
from .outline import crosses_itself, edge_crosses, fill, on_segment, rings_of
from . import refine
from .straighten import straighten
from .sam_edit import PREVIEW_ADDED, PREVIEW_REMOVED, PREVIEW_TAKEN, SamEdit, SamRequest, interior_point, preview_map

COLOUR_MODES = ("material", "part", "object")
TOOLS = ("select", "brush", "eraser", "sam", "points")
EDIT_BAND = 3  # px around a points edit where the part is made to match the new outline (ε 1.5 + 1, rounded up)


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
        self._locked: np.ndarray | None = None  # per stroke: part ids the user cannot see
        self.image_token = 0  # bumped by every open(): SAM results for an older token are dropped
        self.sam: SamEdit | None = None
        self.sam_enabled = False  # the window's SAM controller turns this on
        self._sam_request: SamRequest | None = None
        self._sam_counter = itertools.count(1)  # request numbers are unique across all SAM edits
        self.sam_unavailable_reason: str | None = None  # set by the window's SAM controller on failure
        self._last_name: str | None = None      # for '.', repeat the last label
        # CVAT layout (an object is its body plus named parts): a new object's first part, or a
        # part split off into an object of its own, becomes that object's body.
        self.body_parts = False
        self._last_material: str | None = None
        self._status_cache: dict[int, str] = {}
        self._rings_cache: tuple | None = None  # points tool: (part id, its mask, rings, parents)
        self.auto_clean: refine.Proposal | None = None  # Shift+R: cleaned masks waiting for Enter / Esc
        self.clean_pending = False                      # ...or still being worked out by SAM
        self._clean_request: tuple | None = None        # (image token, batch id, jobs) for the SAM worker
        self._clean_batch = 0
        self._carry: list[str] = []  # notes from saving the image being left, shown on the next one

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
        if self.sam is not None and ((self.sam.target is not None and self.sam.target not in part_ids)
                                     or (self.sam.object_id is not None and self.sam.object_id not in object_ids)):
            self.sam = None  # the part or object being fixed is gone (e.g. undone)
        self._drop_clean()  # a proposal made before this edit no longer fits the image
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
        if not ann.image:  # e.g. recovered from a lone PNG
            ann.image = os.path.relpath(path, self.ann_dir)
        self.index, self.image, self.ann = index, image, ann
        self.selected, self.hidden_parts, self.hidden_objects = [], set(), set()
        self.solo_object = None
        self.image_token += 1
        self.sam, self._sam_request = None, None
        self._rings_cache = None
        self._drop_clean()
        self.messages = self._carry + list(ann.warnings) + self.labels.unknown_names(ann)
        self._carry = []
        self._notify("image")

    def _try_open(self, index: int, skipped: list[str]) -> bool:
        try:
            self.open(index)
            return True
        except (ValueError, OSError) as e:
            skipped.append(f"skipped {self.images[index].name}: {e}")
            return False

    def _report_skipped(self, skipped: list[str]) -> None:
        if skipped:
            self.messages = self.messages + skipped
            self._notify("status")

    def open_first(self) -> list[str]:
        """Open the first image that loads (a broken annotation must not lock the whole folder).
        Returns what was skipped; raises ValueError if nothing opens."""
        skipped: list[str] = []
        for i in range(len(self.images)):
            if self._try_open(i, skipped):
                self._report_skipped(skipped)
                return skipped
        raise ValueError("no image in this folder could be opened:\n" + "\n".join(skipped))

    def step(self, delta: int) -> bool:
        """Move by `delta`, skipping images that fail to load (each becomes a message). The current
        image is saved first, so a save failure raises instead of looking like broken images."""
        self.end_stroke()
        self.save()
        skipped: list[str] = []
        direction = 1 if delta > 0 else -1
        j = self.index + delta
        while 0 <= j < len(self.images):
            if self._try_open(j, skipped):
                self._report_skipped(skipped)
                return True
            j += direction
        self._report_skipped(skipped)
        if not skipped:
            self.add_message("already at the last image" if delta > 0 else "already at the first image")
        return False

    def save(self, notify: bool = True) -> bool:
        """Write the open annotation if it has unsaved edits. Never mid-stroke: an open edit is
        left for the next save after it ends. `notify=False` is for the crash handler only."""
        ann = self.ann
        if ann is None or not ann.dirty or ann.in_transaction:
            return False
        if ann.review_status == "reviewed":
            problems = ann.validate()
            if problems:  # "reviewed" on disk must always mean the labels passed validation
                ann.set_review_status("unreviewed")
                note = f"{self.stem} no longer passes review ({problems[0]}); saved as unreviewed"
                self.messages = self.messages + [note]
                self._carry.append(note)
        if self.annotator and ann.annotator != self.annotator:
            ann.annotator = self.annotator
        save_annotation(ann, self.ann_dir, self.stem)
        self._status_cache[self.index] = ann.review_status
        if notify:
            self._notify("saved")  # e.g. the CVAT exporter writes this image's output now
            self._notify("status")
        return True

    def recover_after_error(self) -> bool:
        """Crash handler, step 1: roll back a half-finished edit and write the finished ones,
        without calling any listener (the window that just failed may fail again). True if a
        file was written. Step 2, announce_recovery(), tells the listeners."""
        if self.ann is None:
            return False
        self._stroke = None
        if self.ann.in_transaction:
            self.ann.rollback()
        return self.save(notify=False)

    def announce_recovery(self, saved: bool) -> None:
        """Crash handler, step 2: tell the listeners what recover_after_error() changed."""
        if self.ann is None:
            return
        if saved:
            self._notify("saved")
        self._changed()
        self._notify("pixels")

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
                    status = meta.get("review_status", "unreviewed") if isinstance(meta, dict) else "unreviewed"
                except (OSError, ValueError):
                    status = "unreviewed"
                if status not in REVIEW_STATUSES:  # a pipeline may write null or its own words
                    status = "unreviewed"
            self._status_cache[index] = status
        return self._status_cache[index]

    def next_unreviewed(self) -> bool:
        self.end_stroke()
        self.save()
        n = len(self.images)
        skipped: list[str] = []
        for k in range(1, n):
            j = (self.index + k) % n
            if self.review_status(j) in ("none", "unreviewed") and self._try_open(j, skipped):
                self._report_skipped(skipped)
                return True
        self._report_skipped(skipped)
        self.add_message("no unreviewed images left")
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

    def mark_reviewed_and_next(self) -> list[str]:
        """Ctrl+Enter: mark reviewed, then open the next unreviewed image."""
        problems = self.mark_reviewed()
        if not problems:
            self.next_unreviewed()
        return problems

    def review_progress(self) -> tuple[int, int]:
        """(reviewed images, all images)."""
        return sum(self.review_status(i) == "reviewed" for i in range(len(self.images))), len(self.images)

    def toggle_flag(self) -> None:
        self.end_stroke()
        self.ann.set_review_status("unreviewed" if self.ann.review_status == "flagged" else "flagged")
        self._changed()
        self.save()

    # ---- selection --------------------------------------------------------------------------

    def select_part(self, part_id: int | None, add: bool = False) -> None:
        self.end_stroke()  # a stroke belongs to the part it started on
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
        if self.sam is not None and self.primary != self.sam.target:
            self.sam_cancel()  # that fix was for another part (R starts one for this part)
        self._notify("selection")

    def select_at(self, x: int, y: int, add: bool = False) -> None:
        h, w = self.ann.size
        pid = self.ann.part_at(y, x) if 0 <= x < w and 0 <= y < h else None
        if pid is not None and self._is_locked(pid):
            pid = None  # you can't see it: treat it as background
        self.select_part(pid, add)

    def select_next_object(self, step: int = 1) -> None:
        objects = [o for o in self.ann.objects if o.parts and o.id not in self.hidden_objects]
        if not objects:
            return
        ids = [o.id for o in objects]
        current = self.ann.object_of(self.primary).id if self.primary is not None else None
        i = (ids.index(current) + step) % len(ids) if current in ids else (0 if step > 0 else len(ids) - 1)
        self.select_part(objects[i].parts[0].id)
        if self.solo_object is not None:  # solo follows J/K
            self.solo_object = objects[i].id
            self._notify("view")

    def select_next_problem(self) -> bool:
        """P: select the next part that blocks review (no class, name or material, or no pixels)."""
        counts = np.bincount(self.ann.label_map.ravel(), minlength=self.ann.next_id)
        problems = [p.id for o in self.ann.objects for p in o.parts
                    if not o.cls or not p.material or (not p.name and p.attrs.get("role") != "body")
                    or p.id >= len(counts) or counts[p.id] == 0]
        if not problems:
            self.add_message("no problems left: every object has a class and every part a name and material")
            return False
        order = self.ann.part_ids()
        pos = order.index(self.primary) if self.primary in order else -1
        self.select_part(next((p for p in problems if order.index(p) > pos), problems[0]))
        return True

    # ---- labels -------------------------------------------------------------------------------

    def set_material(self, material: str) -> None:
        self.end_stroke()
        ids = list(self.selected)
        if not ids:
            return
        with self.ann.transaction("set material"):
            for pid in ids:
                self.ann.set_part_material(pid, material)
        self._last_material = material
        self._changed()

    def set_material_index(self, index: int) -> bool:
        if not self.selected or not 0 <= index < len(self.labels.materials):
            return False
        self.set_material(self.labels.materials[index])
        return True

    def set_part_name(self, name: str) -> None:
        """Name every selected part (one undo step)."""
        self.end_stroke()
        ids = list(self.selected)
        if not ids:
            return
        value = name.strip() or None
        with self.ann.transaction("set name"):
            for pid in ids:
                self.ann.set_part_name(pid, value)
        self._last_name = value
        self._changed()

    def set_object_class(self, cls: str) -> None:
        """Set the class of every selected part's object (one undo step)."""
        self.end_stroke()
        oids: list[int] = []
        for pid in self.selected:
            oid = self.ann.object_of(pid).id
            if oid not in oids:
                oids.append(oid)
        if not oids:
            return
        value = cls.strip() or None
        with self.ann.transaction("set class"):
            for oid in oids:
                self.ann.set_object_class(oid, value)
        self._changed()

    def repeat_last_label(self) -> bool:
        """'.': give every selected part the last part name and material you set."""
        self.end_stroke()
        if not self.selected or (self._last_name is None and self._last_material is None):
            return False
        with self.ann.transaction("repeat label"):
            for pid in self.selected:
                if self._last_name is not None:
                    self.ann.set_part_name(pid, self._last_name)
                if self._last_material is not None:
                    self.ann.set_part_material(pid, self._last_material)
        self._changed()
        return True

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
        if tool == "sam" and not self.sam_enabled:
            self.add_message(self._sam_unavailable_message())
            self._notify("view")  # a toolbar button that checked itself must show the real tool
            return
        self.end_stroke()
        if tool != "sam":
            self.sam_cancel()
        self.tool = tool
        self._notify("view")
        if tool == "sam" and self.sam is None and self.primary is not None:
            self.start_sam_fix()

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
        if self.primary is not None and self._is_locked(self.primary):
            self.add_message("that part is hidden (or outside the solo object); show it before painting")
            return False
        if self.tool == "brush" and self.primary is None:
            self.messages = self.messages + ["select a part (or press N for a new one) before painting"]
            self._notify("status")
            return False
        self._locked = self._locked_lut()
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
        if self._locked is not None:  # never change pixels of parts the user cannot see
            mask = mask & ~self._locked[self.ann.label_map]
        if self.tool == "eraser":
            self.ann.erase(mask, self.primary)
        else:
            self.ann.assign(mask, self.primary, self.protect_others)
        self._notify("pixels")

    def part_tags(self) -> list[tuple[int, int, int, str]]:
        """(part id, x, y, "name · material") for every visible part: the text CVAT writes on its
        shapes. It sits at the part's deepest pixel, inside the part and away from its edges."""
        lm, locked = self.ann.label_map, self._locked_lut()
        tags = []
        for o in self.ann.objects:
            for p in o.parts:
                if locked is not None and locked[p.id]:
                    continue
                mask = (lm == p.id).view(np.uint8)
                x0, y0, w, h = cv2.boundingRect(mask)
                if w == 0:
                    continue
                spot = interior_point(mask[y0:y0 + h, x0:x0 + w].view(bool))
                if spot is not None:
                    tags.append((p.id, x0 + spot[0], y0 + spot[1], f"{p.name or '?'} · {p.material or '?'}"))
        return tags

    def _locked_lut(self) -> np.ndarray | None:
        """Bool table over part ids: True for parts hidden directly, in a hidden object, or
        outside the solo object. None when everything is visible."""
        ids = set(self.hidden_parts)
        for o in self.ann.objects:
            if o.id in self.hidden_objects or (self.solo_object is not None and o.id != self.solo_object):
                ids.update(p.id for p in o.parts)
        if not ids:
            return None
        lut = np.zeros(65536, bool)
        lut[list(ids)] = True
        return lut

    # ---- points tool --------------------------------------------------------------------------

    @property
    def rings(self) -> list[np.ndarray] | None:
        """Points tool: the selected part's outline as draggable points (None with another tool,
        nothing selected, or the part hidden). Traced again only when the part's pixels changed
        by other means, so points you placed stay where you put them."""
        outline = self._outline()
        return outline[0] if outline else None

    def _outline(self) -> tuple[list[np.ndarray], list[int]] | None:
        pid = self.primary
        if self.tool != "points" or pid is None or self._is_locked(pid):
            return None
        mask = self.ann.label_map == pid
        cache = self._rings_cache
        if cache is None or cache[0] != pid or not np.array_equal(cache[1], mask):
            cache = self._rings_cache = (pid, mask, *rings_of(mask))
        return (cache[2], cache[3]) if cache[2] else None

    def _rings_to_edit(self, ring: int, index: int) -> list[np.ndarray] | None:
        rings = self.rings
        if rings is None or not 0 <= ring < len(rings) or not 0 <= index < len(rings[ring]):
            return None  # e.g. a click aimed at an outline that has changed since
        return [r.copy() for r in rings]

    def _clamp(self, x: int, y: int) -> tuple[int, int]:
        h, w = self.ann.size
        return min(max(int(x), 0), w - 1), min(max(int(y), 0), h - 1)

    def move_point(self, ring: int, index: int, x: int, y: int) -> bool:
        """Drop an outline point at (x, y): the part changes only around the edges at that
        point. One undo step."""
        rings = self._rings_to_edit(ring, index)
        if rings is None:
            return False
        rings[ring][index] = self._clamp(x, y)
        if crosses_itself(rings[ring], index):
            self.add_message("that would make the outline cross itself; add a point first, or use the brush")
            self._notify("points")
            return False
        return self._apply_rings(rings, ring, "move point")

    def insert_point(self, ring: int, after: int, x: int, y: int) -> bool:
        """Add an outline point on the edge after point `after`, at the place nearest (x, y). It
        changes no pixels until it is moved."""
        rings = self._rings_to_edit(ring, after)
        if rings is None:
            return False
        r = rings[ring]
        rings[ring] = np.insert(r, after + 1, on_segment(r[after], r[(after + 1) % len(r)], x, y), axis=0)
        self._keep_rings(rings)
        self._notify("points")
        return True

    def delete_point(self, ring: int, index: int) -> bool:
        rings = self._rings_to_edit(ring, index)
        if rings is None:
            return False
        if len(rings[ring]) <= 3:
            self.add_message("a shape needs at least 3 points; erase it or delete the part instead")
            return False
        rings[ring] = np.delete(rings[ring], index, axis=0)
        if edge_crosses(rings[ring], (index - 1) % len(rings[ring])):
            self.add_message("that would make the outline cross itself")
            return False
        return self._apply_rings(rings, ring, "delete point")

    def _keep_rings(self, rings: list[np.ndarray]) -> None:
        pid = self.primary
        self._rings_cache = (pid, self.ann.label_map == pid, rings, self._rings_cache[3])

    def _apply_rings(self, rings: list[np.ndarray], ring: int, label: str) -> bool:
        """Make the part follow the edited outline where the outline changed. Outline fills
        are only close to the pixels, so: find where the fill changed, widen that by the
        simplification error, and inside it set the part to exactly the new fill (no gaps of
        background or slivers along the old edge). Pixels further away never change."""
        pid, parents, size = self.primary, self._rings_cache[3], self.ann.size
        piece = {ring if parents[ring] < 0 else parents[ring]}  # only this piece's fill changes
        changed = fill(self.rings, parents, size, piece) ^ fill(rings, parents, size, piece)
        if not changed.any():  # e.g. a point moved along a straight edge
            self._keep_rings(rings)
            self._notify("points")
            return True
        ys, xs = np.nonzero(changed)
        h, w = size
        x0, y0 = max(int(xs.min()) - EDIT_BAND, 0), max(int(ys.min()) - EDIT_BAND, 0)
        x1, y1 = min(int(xs.max()) + EDIT_BAND + 1, w), min(int(ys.max()) + EDIT_BAND + 1, h)
        box = np.s_[y0:y1, x0:x1]
        near = cv2.dilate(changed[box].view(np.uint8), np.ones((2 * EDIT_BAND + 1,) * 2, np.uint8)).view(bool)
        new = fill(rings, parents, size, window=(x0, y0, x1, y1))[box]  # every piece: they may overlap here
        owner = self.ann.label_map[box]
        mine = owner == pid
        add, remove = np.zeros(size, bool), np.zeros(size, bool)
        add[box], remove[box] = new & near & ~mine, mine & near & ~new
        takeable = np.zeros(size, bool)
        takeable[box] = owner == 0 if self.protect_others else True
        locked = self._locked_lut()
        if locked is not None:  # never change pixels of parts the user cannot see
            takeable[box] &= ~locked[owner]
        blocked = bool((add & ~takeable).any())
        add &= takeable
        if not add.any() and not remove.any():
            self.add_message("nothing changed: those pixels belong to protected or hidden parts")
            self._notify("points")  # the outline goes back to where it was
            return False
        with self.ann.transaction(label):
            self.ann.assign(add, pid)
            self.ann.erase(remove, pid)
        if blocked:
            self._rings_cache = None  # show the outline the pixels really have, not the one dragged
        else:
            self._keep_rings(rings)
        self._changed()
        if blocked:
            self.add_message("some of those pixels belong to protected or hidden parts and were left as they were")
        return True

    # ---- structure ----------------------------------------------------------------------------

    def _select_new(self, part_id: int) -> None:
        self.selected = [part_id]
        self.tool = "brush"
        self._changed()
        self._notify("selection")
        self._notify("view")

    def new_object(self) -> int | None:
        """With SAM: start a new object from SAM clicks (returns None). Without: a new object
        with one empty part, selected, brush ready."""
        if self.sam_enabled:
            self.start_sam_new(new_object=True)
            return None
        self.end_stroke()
        with self.ann.transaction("new object"):
            pid = self._first_part(self.ann.new_object())
        self._select_new(pid)
        return pid

    def _first_part(self, object_id: int) -> int:
        """The first part of a new object (call inside a transaction): its body in body mode."""
        pid = self.ann.new_part(object_id)
        if self.body_parts:
            self._make_body(pid)
        return pid

    def _make_body(self, part_id: int) -> None:
        p = self.ann.part(part_id)
        p.name, p.attrs = "body", {**p.attrs, "role": "body"}

    def new_part(self) -> int | None:
        """With SAM: start a new part (in the selected part's object) from SAM clicks (returns
        None). Without: a new empty part there (or a new object), brush ready."""
        if self.sam_enabled:
            self.start_sam_new(new_object=False)
            return None
        self.end_stroke()
        if self.primary is None:
            return self.new_object()
        pid = self.ann.new_part(self.ann.object_of(self.primary).id)
        self._select_new(pid)
        return pid

    def new_empty_part(self) -> int:
        """Ctrl+N: an empty part in the selected part's object (or a new object) with the brush
        ready, also when SAM is on: for things SAM can't segment, like thin cables."""
        self.end_stroke()
        self.sam_cancel()
        if self.primary is None:
            with self.ann.transaction("new object"):
                pid = self._first_part(self.ann.new_object())
        else:
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
            emptied = self._objects_with_parts()
            for pid in others:
                self.ann.merge_parts(keep, pid)
            self._drop_emptied(emptied)
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
            emptied = self._objects_with_parts()
            for pid in ids:
                self.ann.delete_part(pid)
            self._drop_emptied(emptied)
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
        with self.ann.transaction("move part"):
            emptied = self._objects_with_parts()
            self.ann.move_part(part_id, object_id)
            self._drop_emptied(emptied)
        self._changed()

    def promote_part(self, part_id: int) -> int:
        self.end_stroke()
        with self.ann.transaction("promote part"):
            emptied = self._objects_with_parts()
            oid = self.ann.promote_part(part_id)
            if self.body_parts:
                self._make_body(part_id)
            self._drop_emptied(emptied)
        self._changed()
        return oid

    def _objects_with_parts(self) -> set[int]:
        return {o.id for o in self.ann.objects if o.parts}

    def _drop_emptied(self, had_parts: set[int]) -> None:
        """Inside the edit's transaction: remove objects this edit left without any part (an
        empty object blocks review and can't be reached with J/K or Shift+Del)."""
        for o in [o for o in self.ann.objects if o.id in had_parts and not o.parts]:
            self.ann.delete_object(o.id)

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

    # ---- messages -----------------------------------------------------------------------------

    def add_message(self, text: str) -> None:
        self.messages = self.messages + [text]
        self._notify("status")

    # ---- SAM ----------------------------------------------------------------------------------

    def _sam_unavailable_message(self) -> str:
        if self.sam_unavailable_reason:
            return f"SAM is not available ({self.sam_unavailable_reason}); use the brush"
        return "SAM is not available; start AnnoFix without --no-sam to use it, or use the brush"

    def _new_edit(self, target: int | None = None, object_id: int | None = None) -> SamEdit:
        return SamEdit(self.image_token, target, object_id, counter=self._sam_counter)

    def _is_locked(self, part_id: int) -> bool:
        """Hidden, in a hidden object, or outside the solo object: SAM must not touch it."""
        lut = self._locked_lut()
        return lut is not None and bool(lut[part_id])

    def _start_sam(self, edit: SamEdit) -> bool:
        if not self.sam_enabled:
            self.add_message(self._sam_unavailable_message())
            self._notify("view")
            return False
        self.end_stroke()
        self.discard_auto_clean()
        self.sam, self._sam_request = edit, None
        if self.tool != "sam":
            self.tool = "sam"
            self._notify("view")
        self._notify("sam")
        return True

    def _queue(self, request: SamRequest | None) -> None:
        if request is not None:
            self._sam_request = request
            self._notify("sam_request")
        self._notify("sam")

    def start_sam_fix(self) -> bool:
        """Improve the selected part's mask with SAM: prompts come from the mask itself (its box,
        points inside it, points on the neighbours it must not swallow), SAM's answer closest to
        the mask is shown first if it matches the mask well (else the mask smoothed is), and the
        mask smoothed is always one more candidate (Tab). A part that
        is plain wrong is redrawn from clicks instead: Esc, then click it with the SAM tool."""
        if self.primary is None:
            return False
        if self._is_locked(self.primary):
            self.add_message("that part is hidden (or outside the solo object); show it before fixing it with SAM")
            return False
        if not self._start_sam(self._new_edit(target=self.primary)):
            return False
        mask = self.ann.part_mask(self.primary)
        if mask.any():
            others = (self.ann.label_map != 0) & ~mask
            smoothed = refine.smooth(mask) & ~others  # smoothing must not swallow an enclosed part
            self._queue(self.sam.improve(mask, smoothed, *refine.prompts_for(mask, others)))
        return True

    def start_sam_new(self, new_object: bool = False) -> bool:
        """A new part from SAM clicks: in the selected part's object, or in a new object."""
        object_id = None if new_object or self.primary is None else self.ann.object_of(self.primary).id
        return self._start_sam(self._new_edit(object_id=object_id))

    def sam_click(self, x: int, y: int, positive: bool) -> None:
        """SAM tool click. With no edit open, a left click starts one: fixing the part under the
        cursor, or a new object on background. Otherwise it adds an include/exclude point."""
        if self.tool != "sam":
            return
        h, w = self.ann.size
        if not (0 <= x < w and 0 <= y < h):
            return  # outside the image
        if self.sam is None:
            if not positive:
                return
            pid = self.ann.part_at(y, x)
            if pid is not None and self._is_locked(pid):
                pid = None  # an invisible part counts as background: start a new object there
            if pid is not None:
                self.selected = [pid]
                self._notify("selection")
            if not self._start_sam(self._new_edit(target=pid)):
                return
        self._queue(self.sam.add_point(x, y, positive))

    def sam_box(self, x0: int, y0: int, x1: int, y1: int) -> None:
        if self.tool != "sam":
            return
        h, w = self.ann.size
        x0, x1 = (min(max(int(v), 0), w - 1) for v in (x0, x1))
        y0, y1 = (min(max(int(v), 0), h - 1) for v in (y0, y1))
        target = self.primary if self.primary is not None and not self._is_locked(self.primary) else None
        if self.sam is None and not self._start_sam(self._new_edit(target=target)):
            return
        self._queue(self.sam.set_box(x0, y0, x1, y1))

    def take_sam_request(self) -> SamRequest | None:
        request, self._sam_request = self._sam_request, None
        return request

    def sam_receive(self, token: int, seq: int, masks, scores, logits) -> bool:
        if self.sam is None:
            return False
        lm = self.ann.label_map
        fillable = (lm == 0) if self.sam.target is None else (lm == 0) | (lm == self.sam.target)
        if not self.sam.receive(token, seq, masks, scores, logits, fillable=fillable):
            return False
        self._notify("sam")
        return True

    def sam_cycle(self, step: int = 1) -> None:
        if self.sam is not None and self.sam.candidates:
            self.sam.cycle(step)
            self._notify("sam")

    def sam_preview(self) -> np.ndarray | None:
        e = self.sam
        if e is None or e.mask is None:
            return None
        return preview_map(self.ann.label_map, e.target, e.mask, protect=self.protect_others,
                           locked=self._locked_lut())

    def sam_accept(self, union: bool = False) -> bool:
        """Apply the chosen candidate: replace the target's pixels (Enter), add to them
        (Shift+Enter), or create the new part/object. One undo step."""
        e = self.sam
        if e is None:
            return False
        if e.seq == 0:
            self.add_message("click the object for SAM first")
            return False
        if e.mask is None or e.received_seq != e.seq:
            self.add_message("SAM is still working on your last click; press Enter again when the preview updates")
            return False
        if e.target is not None and self._is_locked(e.target):
            self.add_message("that part is hidden (or outside the solo object); show it before accepting")
            return False
        locked = self._locked_lut()
        mask = e.mask if locked is None else e.mask & ~locked[self.ann.label_map]
        if not mask.any():
            self.add_message("SAM's mask is empty there (or covers only hidden parts): click on the object")
            return False
        with self.ann.transaction("sam"):
            if e.target is None:
                if e.object_id is not None:
                    pid = self.ann.new_part(e.object_id)
                else:
                    pid = self._first_part(self.ann.new_object())
                self.ann.assign(mask, pid, self.protect_others)
            elif union:
                pid = e.target
                self.ann.assign(mask, pid, self.protect_others)
            else:
                pid = e.target
                self.ann.replace_part_mask(pid, mask, self.protect_others)
        self.sam, self._sam_request = None, None
        self.selected = [pid]
        self._changed()
        self._notify("selection")
        self._notify("sam")
        return True

    def sam_cancel(self) -> bool:
        if self.sam is None:
            return False
        self.sam, self._sam_request = None, None
        self._notify("sam")
        return True

    # ---- auto-clean (Shift+R) ---------------------------------------------------------------

    def start_auto_clean(self) -> bool:
        """Propose cleaned masks for the whole image: with SAM, each object and part is asked
        again (the answer arrives through refine_receive); without SAM, they are smoothed at
        once. Nothing changes until accept_auto_clean()."""
        self.end_stroke()
        self.sam_cancel()
        self._drop_clean()
        locked = self._locked_lut()
        if not self.sam_enabled:
            self._set_clean(*refine.build_proposal(self.ann, {}, locked))
            return True
        self._clean_batch += 1
        self._clean_request = (self.image_token, self._clean_batch, refine.jobs_for(self.ann, locked))
        self.clean_pending = True
        self._notify("refine_request")
        self._notify("refine")
        return True

    def take_clean_request(self) -> tuple | None:
        request, self._clean_request = self._clean_request, None
        return request

    def clean_receive(self, token: int, batch_id: int, results) -> bool:
        """SAM's answers for the batch asked last on this image; anything older is dropped."""
        if not self.clean_pending or token != self.image_token or batch_id != self._clean_batch:
            return False
        self.clean_pending = False
        self._set_clean(*refine.build_proposal(self.ann, {r.key: r for r in results}, self._locked_lut()))
        return True

    def clean_failed(self) -> bool:
        """SAM could not answer the batch: propose smoothed masks instead."""
        if not self.clean_pending:
            return False
        self.clean_pending = False
        self._set_clean(*refine.build_proposal(self.ann, {}, self._locked_lut()))
        self.add_message("SAM failed, so auto-clean only smoothed the masks")
        return True

    def _set_clean(self, label_map: np.ndarray, report: dict) -> None:
        self.auto_clean = refine.Proposal(label_map, report)
        self._notify("refine")

    def _drop_clean(self) -> None:
        self.auto_clean, self.clean_pending, self._clean_request = None, False, None

    def auto_clean_preview(self) -> np.ndarray | None:
        """What accepting would change: pixels gained, taken from another part, or removed."""
        if self.auto_clean is None:
            return None
        old, new = self.ann.label_map, self.auto_clean.label_map
        out = np.zeros(old.shape, np.uint8)
        out[(old == 0) & (new != 0)] = PREVIEW_ADDED
        out[(old != 0) & (new == 0)] = PREVIEW_REMOVED
        out[(old != 0) & (new != 0) & (old != new)] = PREVIEW_TAKEN
        return out

    def accept_auto_clean(self) -> bool:
        """Apply the proposal as one undo step."""
        proposal = self.auto_clean
        if proposal is None:
            return False
        self.auto_clean = None
        new = proposal.label_map.copy()
        ids = list(refine.held_ids(self.ann, self._locked_lut()))
        if ids:  # objects with a part hidden since the proposal was made: still never changed
            held = np.isin(self.ann.label_map, ids) | np.isin(new, ids)
            new[held] = self.ann.label_map[held]
        with self.ann.transaction(proposal.action):
            self.ann.label_map[...] = new
        self._changed()
        self.add_message(f"{proposal.action}: {proposal.summary()}")
        return True

    def start_straighten(self, whole_image: bool = False) -> bool:
        """Propose the selected parts (or every part) redrawn with straight edges, shown and
        accepted like auto-clean: Enter applies it as one undo step, Esc drops it."""
        self.end_stroke()
        self.sam_cancel()
        self._drop_clean()
        ids = [p.id for o in self.ann.objects for p in o.parts] if whole_image else list(self.selected)
        if not ids:
            self.add_message("select a part to straighten first (Shift+L straightens every part)")
            return False
        new, report = straighten(self.ann.label_map, ids, self._locked_lut(),
                                 protect=self.protect_others, whole_image=whole_image)
        self.auto_clean = refine.Proposal(new, report, "straighten")
        self._notify("refine")
        return True

    def discard_auto_clean(self) -> bool:
        if self.auto_clean is None and not self.clean_pending:
            return False
        self._drop_clean()
        self._notify("refine")
        return True
