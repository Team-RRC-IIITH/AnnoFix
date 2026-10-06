"""In-memory annotation: a uint16 part-id label map plus the object -> part hierarchy.

Every pixel belongs to at most one part (0 = background). Every mutating method runs inside
a transaction, which records a bbox-cropped diff of the label map plus a metadata snapshot,
so any edit can be undone exactly. The label map is only ever modified in place.
"""
from __future__ import annotations

import json
from contextlib import contextmanager
from dataclasses import dataclass, field

import cv2
import numpy as np

SCHEMA_VERSION = 1
REVIEW_STATUSES = ("unreviewed", "reviewed", "flagged")
MAX_PART_ID = int(np.iinfo(np.uint16).max)


@dataclass
class Part:
    id: int
    name: str | None = None
    material: str | None = None


@dataclass
class AnnObject:
    id: int
    cls: str | None = None
    parts: list[Part] = field(default_factory=list)


def objects_from_dicts(items: list[dict]) -> list[AnnObject]:
    return [
        AnnObject(int(o["id"]), o.get("class"),
                  [Part(int(p["id"]), p.get("name"), p.get("material")) for p in o.get("parts", [])])
        for o in items
    ]


@dataclass
class _Entry:
    label: str
    bbox: tuple[int, int, int, int] | None  # y0, y1, x0, x1 of the changed pixels
    before: np.ndarray | None
    after: np.ndarray | None
    meta_before: str  # compact JSON of meta_dict()
    meta_after: str   # the very same str object as meta_before when only pixels changed

    @property
    def nbytes(self) -> int:
        arrays = 0 if self.before is None else self.before.nbytes + self.after.nbytes
        meta = len(self.meta_before) + (0 if self.meta_after is self.meta_before else len(self.meta_after))
        return arrays + meta


class Annotation:
    """Callers that live across edits (the GUI) must hold part/object ids, not Part/AnnObject
    instances: undo, redo and rollback rebuild those instances from a snapshot."""

    def __init__(self, label_map: np.ndarray, objects: list[AnnObject] | None = None, *,
                 next_id: int = 1, image: str = "", source: str = "manual",
                 annotator: str | None = None, review_status: str = "unreviewed",
                 history_bytes: int = 200 * 2**20):
        if label_map.dtype != np.uint16 or label_map.ndim != 2:
            raise ValueError(f"label map must be 2-D uint16, got {label_map.dtype} {label_map.shape}")
        if review_status not in REVIEW_STATUSES:
            raise ValueError(f"review_status must be one of {REVIEW_STATUSES}, got {review_status!r}")
        self.label_map = label_map
        self.objects: list[AnnObject] = objects if objects is not None else []
        self.next_id = next_id
        self.image = image
        self.source = source
        self.annotator = annotator
        self.review_status = review_status
        self.warnings: list[str] = []
        self._history_bytes = history_bytes
        self._undo: list[_Entry] = []
        self._redo: list[_Entry] = []
        self._tx_depth = 0
        self._tx_label = ""
        self._tx_map: np.ndarray | None = None
        self._tx_meta: str | None = None
        self._saved_top: _Entry | None = None

    @classmethod
    def empty(cls, height: int, width: int, **kw) -> Annotation:
        return cls(np.zeros((height, width), np.uint16), **kw)

    @classmethod
    def from_meta(cls, label_map: np.ndarray, meta: dict) -> Annotation:
        return cls(label_map, objects_from_dicts(meta.get("objects", [])),
                   next_id=int(meta.get("next_id", 1)), image=meta.get("image", ""),
                   source=meta.get("source", "manual"), annotator=meta.get("annotator"),
                   review_status=meta.get("review_status", "unreviewed"))

    # ---- lookups -------------------------------------------------------------------------

    @property
    def size(self) -> tuple[int, int]:
        h, w = self.label_map.shape
        return h, w

    def object(self, object_id: int) -> AnnObject:
        for o in self.objects:
            if o.id == object_id:
                return o
        raise KeyError(f"no object {object_id}")

    def _find_part(self, part_id: int) -> tuple[AnnObject, Part]:
        for o in self.objects:
            for p in o.parts:
                if p.id == part_id:
                    return o, p
        raise KeyError(f"no part {part_id}")

    def part(self, part_id: int) -> Part:
        return self._find_part(part_id)[1]

    def object_of(self, part_id: int) -> AnnObject:
        return self._find_part(part_id)[0]

    def part_ids(self) -> list[int]:
        return [p.id for o in self.objects for p in o.parts]

    def part_at(self, y: int, x: int) -> int | None:
        return int(self.label_map[y, x]) or None

    def part_mask(self, part_id: int) -> np.ndarray:
        return self.label_map == part_id

    def part_area(self, part_id: int) -> int:
        return int(np.count_nonzero(self.label_map == part_id))

    # ---- metadata ------------------------------------------------------------------------

    def meta_dict(self) -> dict:
        """The JSON document for this annotation (everything except the label map)."""
        return {
            "schema_version": SCHEMA_VERSION,
            "image": self.image,
            "size": list(self.label_map.shape),
            "next_id": self.next_id,
            "source": self.source,
            "annotator": self.annotator,
            "review_status": self.review_status,
            "objects": [
                {"id": o.id, "class": o.cls,
                 "parts": [{"id": p.id, "name": p.name, "material": p.material} for p in o.parts]}
                for o in self.objects
            ],
        }

    def _restore_meta(self, d: dict) -> None:
        self.next_id = max(self.next_id, d["next_id"])  # ids are never reused, even after undo
        self.image = d["image"]
        self.source = d["source"]
        self.annotator = d["annotator"]
        self.review_status = d["review_status"]
        self.objects = objects_from_dicts(d["objects"])

    def _alloc_id(self) -> int:
        if self.next_id > MAX_PART_ID:
            raise OverflowError("id space exhausted (part ids must fit the uint16 label map)")
        i = self.next_id
        self.next_id += 1
        return i

    # ---- transactions and history -------------------------------------------------------

    @property
    def in_transaction(self) -> bool:
        return self._tx_depth > 0

    def _meta_json(self) -> str:
        return json.dumps(self.meta_dict(), separators=(",", ":"))

    def begin(self, label: str) -> None:
        """Open an edit that can span many calls (a brush stroke over many mouse events).
        Close it with commit() (one undo step) or rollback(). Nested begins merge into the
        outermost one."""
        if self._tx_depth == 0:
            self._tx_label = label
            self._tx_map = self.label_map.copy()
            self._tx_meta = self._meta_json()
        self._tx_depth += 1

    def commit(self) -> None:
        if not self._tx_depth:
            raise RuntimeError("commit() without begin()")
        self._tx_depth -= 1
        if self._tx_depth == 0:
            label, map_before, meta_before = self._tx_label, self._tx_map, self._tx_meta
            self._tx_map = self._tx_meta = None
            self._record(label, map_before, meta_before)

    def rollback(self) -> None:
        """Abandon the whole open edit, however deeply nested: pixels and labels go back to how
        they were at the outermost begin(), and nothing is recorded."""
        if not self._tx_depth:
            raise RuntimeError("rollback() without begin()")
        self.label_map[...] = self._tx_map
        self._restore_meta(json.loads(self._tx_meta))
        self._tx_depth = 0
        self._tx_map = self._tx_meta = None

    @contextmanager
    def transaction(self, label: str):
        """begin/commit around a block. If the block raises, the outermost transaction rolls
        everything back; an inner one only closes its own level and lets the exception through
        (so an op failing inside an open stroke leaves the stroke for its owner to finish)."""
        outermost = self._tx_depth == 0
        self.begin(label)
        try:
            yield
        except BaseException:
            if outermost:
                self.rollback()
            else:
                self._tx_depth -= 1
            raise
        self.commit()

    def _record(self, label: str, map_before: np.ndarray, meta_before: str) -> None:
        meta_after = self._meta_json()
        if meta_after == meta_before:
            meta_after = meta_before  # share one string: pixel-only edits store metadata once
        changed = map_before != self.label_map
        if changed.any():
            rows = np.flatnonzero(changed.any(axis=1))
            cols = np.flatnonzero(changed.any(axis=0))
            y0, y1, x0, x1 = int(rows[0]), int(rows[-1]) + 1, int(cols[0]), int(cols[-1]) + 1
            entry = _Entry(label, (y0, y1, x0, x1), map_before[y0:y1, x0:x1].copy(),
                           self.label_map[y0:y1, x0:x1].copy(), meta_before, meta_after)
        elif meta_after is not meta_before:
            entry = _Entry(label, None, None, None, meta_before, meta_after)
        else:
            return
        self._undo.append(entry)
        self._redo.clear()
        total = sum(e.nbytes for e in self._undo)
        while total > self._history_bytes and len(self._undo) > 1:
            total -= self._undo.pop(0).nbytes

    def _apply(self, bbox, crop, meta: str) -> None:
        if bbox is not None:
            y0, y1, x0, x1 = bbox
            self.label_map[y0:y1, x0:x1] = crop
        self._restore_meta(json.loads(meta))

    def can_undo(self) -> bool:
        return bool(self._undo)

    def can_redo(self) -> bool:
        return bool(self._redo)

    def undo(self) -> str | None:
        if self._tx_depth:
            raise RuntimeError("undo inside a transaction")
        if not self._undo:
            return None
        e = self._undo.pop()
        self._apply(e.bbox, e.before, e.meta_before)
        self._redo.append(e)
        return e.label

    def redo(self) -> str | None:
        if self._tx_depth:
            raise RuntimeError("redo inside a transaction")
        if not self._redo:
            return None
        e = self._redo.pop()
        self._apply(e.bbox, e.after, e.meta_after)
        self._undo.append(e)
        return e.label

    @property
    def dirty(self) -> bool:
        top = self._undo[-1] if self._undo else None
        return top is not self._saved_top

    def mark_saved(self) -> None:
        self._saved_top = self._undo[-1] if self._undo else None

    # ---- hierarchy and labels -----------------------------------------------------------

    def new_object(self, cls: str | None = None) -> int:
        with self.transaction("new object"):
            oid = self._alloc_id()
            self.objects.append(AnnObject(oid, cls or None))
        return oid

    def new_part(self, object_id: int, name: str | None = None, material: str | None = None) -> int:
        with self.transaction("new part"):
            obj = self.object(object_id)
            pid = self._alloc_id()
            obj.parts.append(Part(pid, name or None, material or None))
        return pid

    def set_object_class(self, object_id: int, cls: str | None) -> None:
        with self.transaction("set class"):
            self.object(object_id).cls = cls or None

    def set_part_name(self, part_id: int, name: str | None) -> None:
        with self.transaction("set name"):
            self.part(part_id).name = name or None

    def set_part_material(self, part_id: int, material: str | None) -> None:
        with self.transaction("set material"):
            self.part(part_id).material = material or None

    def set_review_status(self, status: str) -> None:
        if status not in REVIEW_STATUSES:
            raise ValueError(f"review status must be one of {REVIEW_STATUSES}, got {status!r}")
        with self.transaction("review status"):
            self.review_status = status

    # ---- pixel ownership ----------------------------------------------------------------

    def _check_mask(self, mask: np.ndarray) -> None:
        if mask.dtype != bool or mask.shape != self.label_map.shape:
            raise ValueError(f"mask must be bool {self.label_map.shape}, got {mask.dtype} {mask.shape}")

    def assign(self, mask: np.ndarray, part_id: int, protect_others: bool = False) -> int:
        """Give the pixels in `mask` to `part_id`; other parts lose them unless protected.
        Returns the number of pixels that changed owner."""
        self._check_mask(mask)
        self.part(part_id)
        target = mask & (self.label_map == 0) if protect_others else mask & (self.label_map != part_id)
        with self.transaction("assign"):
            self.label_map[target] = part_id
        return int(np.count_nonzero(target))

    def erase(self, mask: np.ndarray, part_id: int | None = None) -> int:
        """Return pixels in `mask` to background (only `part_id`'s pixels if given)."""
        self._check_mask(mask)
        owned = self.label_map == part_id if part_id is not None else self.label_map != 0
        target = mask & owned
        with self.transaction("erase"):
            self.label_map[target] = 0
        return int(np.count_nonzero(target))

    def replace_part_mask(self, part_id: int, mask: np.ndarray, protect_others: bool = False) -> None:
        """Make `part_id` cover exactly `mask` (minus protected pixels). Pixels it loses
        become background."""
        self._check_mask(mask)
        self.part(part_id)
        with self.transaction("replace mask"):
            self.label_map[(self.label_map == part_id) & ~mask] = 0
            self.assign(mask, part_id, protect_others)

    def clean_part(self, part_id: int, min_island_px: int = 64, max_hole_px: int = 2000) -> None:
        """Drop islands smaller than `min_island_px` (the largest piece always stays) and fill
        background holes up to `max_hole_px` that the part fully encloses."""
        self.part(part_id)
        with self.transaction("clean part"):
            mask = self.label_map == part_id
            n, cc, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
            if n > 2:
                areas = stats[1:, cv2.CC_STAT_AREA]
                largest = 1 + int(np.argmax(areas))
                drop = [i for i in range(1, n) if i != largest and stats[i, cv2.CC_STAT_AREA] < min_island_px]
                if drop:
                    self.label_map[np.isin(cc, drop)] = 0
            mask = self.label_map == part_id
            h, w = mask.shape
            n, cc, stats, _ = cv2.connectedComponentsWithStats((~mask).astype(np.uint8), connectivity=4)
            for i in range(1, n):
                x, y, cw, ch, area = (int(v) for v in stats[i])
                if area > max_hole_px or x == 0 or y == 0 or x + cw == w or y + ch == h:
                    continue  # too big, or touches the border so not enclosed
                comp = cc[y:y + ch, x:x + cw] == i
                region = self.label_map[y:y + ch, x:x + cw]  # view: writes go to the label map
                if not region[comp].any():  # only pure-background holes
                    region[comp] = part_id

    # ---- structure ----------------------------------------------------------------------

    def delete_part(self, part_id: int) -> None:
        with self.transaction("delete part"):
            obj, p = self._find_part(part_id)
            self.label_map[self.label_map == part_id] = 0
            obj.parts.remove(p)

    def delete_object(self, object_id: int) -> None:
        with self.transaction("delete object"):
            obj = self.object(object_id)
            for p in obj.parts:
                self.label_map[self.label_map == p.id] = 0
            self.objects.remove(obj)

    def merge_parts(self, keep_id: int, other_id: int) -> None:
        """`other_id`'s pixels join `keep_id`; `other_id` is removed."""
        if keep_id == other_id:
            raise ValueError("cannot merge a part with itself")
        with self.transaction("merge parts"):
            self.part(keep_id)
            obj, other = self._find_part(other_id)
            self.label_map[self.label_map == other_id] = keep_id
            obj.parts.remove(other)

    def move_part(self, part_id: int, object_id: int) -> None:
        with self.transaction("move part"):
            src, p = self._find_part(part_id)
            dst = self.object(object_id)
            if dst is not src:
                src.parts.remove(p)
                dst.parts.append(p)

    def promote_part(self, part_id: int) -> int:
        """Move a part into a new, unlabelled object (two touching objects grouped as one)."""
        self.part(part_id)
        with self.transaction("promote part"):
            oid = self.new_object()
            self.move_part(part_id, oid)
        return oid

    # ---- review -------------------------------------------------------------------------

    def validate(self) -> list[str]:
        """Problems that block marking the image reviewed (empty list = OK)."""
        problems = []
        counts = np.bincount(self.label_map.ravel(), minlength=self.next_id)
        for o in self.objects:
            if not o.cls:
                problems.append(f"object {o.id}: no class")
            if not o.parts:
                problems.append(f"object {o.id}: no parts")
            for p in o.parts:
                where = f"part {p.id} (object {o.id})"
                if not p.name:
                    problems.append(f"{where}: no name")
                if not p.material:
                    problems.append(f"{where}: no material")
                if p.id >= len(counts) or counts[p.id] == 0:
                    problems.append(f"{where}: empty mask")
        stray = sorted(set((np.flatnonzero(counts[1:]) + 1).tolist()) - set(self.part_ids()))
        if stray:
            problems.append(f"pixels with unknown part ids {stray}")
        return problems

    def reconcile(self) -> list[str]:
        """Repair label-map / hierarchy mismatches after loading a file. Not undoable: it
        describes the file, not an edit. Returns (and appends to self.warnings) what it did."""
        warnings = []
        obj_ids = [o.id for o in self.objects]
        part_ids = self.part_ids()
        all_ids = obj_ids + part_ids
        if len(set(all_ids)) != len(all_ids):
            raise ValueError("duplicate object/part ids in annotation JSON")
        bad = sorted(i for i in all_ids if not 1 <= i <= MAX_PART_ID)
        if bad:
            raise ValueError(f"ids must be in 1..{MAX_PART_ID} (0 is background), got {bad}")
        counts = np.bincount(self.label_map.ravel())
        present = (np.flatnonzero(counts[1:]) + 1).tolist()
        clash = sorted(set(present) & set(obj_ids))
        if clash:
            # Adopting these as orphan parts would duplicate the ids and make the next save unloadable.
            raise ValueError(f"label map holds object id(s) {clash}; pixels must hold part ids")
        highest = max(all_ids + present, default=0)
        if self.next_id <= highest:
            warnings.append(f"next_id {self.next_id} <= highest id {highest}; raised to {highest + 1}")
            self.next_id = highest + 1
        orphans = sorted(set(present) - set(part_ids))
        if orphans:
            try:
                oid = self._alloc_id()
            except OverflowError as e:
                raise ValueError(f"no free id for an object holding orphan part ids {orphans}: "
                                 "id space exhausted") from e
            obj = AnnObject(oid, None, [Part(i) for i in orphans])
            self.objects.append(obj)
            warnings.append(f"orphan part ids {orphans} (pixels without a JSON entry) "
                            f"grouped into new object {obj.id}")
        for o in self.objects:
            for p in o.parts:
                if p.id >= len(counts) or counts[p.id] == 0:
                    warnings.append(f"part {p.id} (object {o.id}) has no pixels")
        self.warnings.extend(warnings)
        return warnings
