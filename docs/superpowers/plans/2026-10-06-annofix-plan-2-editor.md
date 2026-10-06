# AnnoFix Plan 2 of 3 — Viewer and Manual Editing Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A usable PyQt6 editor for AnnoFix annotations, without SAM. You open a folder and see coloured masks. You can click parts, set material, name and class, toggle visibility, fix bleeding with the brush and eraser, merge, split and delete parts and objects, and undo. Work autosaves, and each image can be marked reviewed.

**Architecture:** All editor behaviour lives in a Qt-free `Session` (`annofix/session.py`) that holds the open image, its `Annotation`, the selection (ids only), view options, and every action. Listeners are told what changed. The Qt layer (`annofix/ui/`) only turns mouse and keys into `Session` calls and redraws. The overlay is a numpy-recoloured uint32 buffer wrapped by a `QImage` (~2 ms per full 1280x720 recolour). First, the core gets the APIs the GUI needs: explicit begin/commit/rollback for strokes that span mouse events, a save guard, honest history memory accounting, lone-PNG recovery, and unknown-name warnings.

**Tech Stack:** Python 3.12, uv, numpy, opencv-python-headless, PyQt6 6.11, pytest + pytest-qt (offscreen).

**Spec:** `docs/superpowers/specs/2026-10-03-annofix-design.md` (milestones M3–M4). Plan 1 (merged): `docs/superpowers/plans/2026-10-03-annofix-plan-1-foundation.md`. Findings that shaped this plan: `docs/notes/2026-10-03-m0-measurements.md`.

## Global Constraints

- Python `>=3.12,<3.13`, uv-managed `.venv`; `opencv-python-headless` only (never `opencv-python`; its bundled Qt clashes with PyQt6).
- Thread cap stays: `OMP/OPENBLAS/MKL/NUMEXPR_NUM_THREADS = "10"` before numpy (in `annofix/__init__.py` and `tests/conftest.py`).
- Label map: 2-D `uint16`, `0` = background, one owner per pixel, **modified in place only** (the overlay wraps it).
- Ids are shared by objects and parts and never reused; JSON `schema_version: 1`; unlabelled = `null`.
- **GUI code holds part/object ids, never `Part`/`AnnObject` instances across edits** — undo, redo and rollback rebuild those instances.
- **Every brush/eraser stroke is one undo step** (`Annotation.begin` … `commit`); Esc during a stroke rolls it back.
- **Nothing is saved while an edit is open** (`save_annotation` raises; `Session.save` returns False).
- No `torch` import on the GUI path in this plan (`annofix.session`, `annofix.ui.*`, `annofix.__main__`) — SAM arrives in Plan 3.
- Hotkeys are `QShortcut`s only, never also set on a `QAction` (a duplicated sequence becomes ambiguous and fires neither). Panel buttons, lists and the toolbar widgets use `Qt.FocusPolicy.NoFocus`.
- Qt tests run offscreen: `tests/conftest.py` sets `QT_QPA_PLATFORM=offscreen` by default.
- Commits use the configured git author with **no** `Co-Authored-By` trailer (user instruction).

## Review Focus

1. If the autosave timer or navigation fires mid-stroke, it must neither save half a stroke nor lose it. → Task 3 `test_save_only_when_dirty_and_never_mid_edit`; Task 4 `test_autosave_mid_stroke_waits_for_the_stroke`, `test_navigating_mid_stroke_keeps_the_stroke`.
2. Undo or redo after a structural edit must never leave the UI holding ids that no longer exist (selection, hidden sets, solo). → Task 4 `test_undo_prunes_selection_of_parts_that_no_longer_exist`.
3. An image the pipeline hasn't reached yet (no annotation) must open as an empty canvas you can paint, and nothing is written until you edit. → Task 3 `test_image_without_annotation_opens_empty_and_writes_nothing`.
4. An unexpected exception in any handler must not lose work: finished edits are saved, a half stroke is rolled back, and the error is logged and shown. → Task 8 `test_crash_handler_saves_and_logs`.
5. Moving between 1280x720 and 640x480 frames must rebuild the overlay at the new size. An annotation file that doesn't match its image must show an error and keep the current image open. → Task 6 `test_switching_to_a_smaller_image_rebuilds_the_overlay`; Task 8 `test_navigation_saves_and_switches_resolution`; Task 3 `test_bad_annotation_file_leaves_current_image_open`.

## File Structure

| File | Responsibility |
|---|---|
| `annofix/annotation.py` (modify) | `begin/commit/rollback/in_transaction`; history stores metadata as compact JSON and counts it |
| `annofix/io.py` (modify) | save guard, lone-PNG recovery, `.tmp` cleanup, directory fsync |
| `annofix/config.py` (modify) | `Labels.unknown_names(ann)` |
| `annofix/sam_backend.py` (modify) | `device_kind()` so `cuda:0` behaves like `cuda` |
| `annofix/session.py` (new) | Qt-free editor state and actions |
| `annofix/ui/__init__.py` (new, empty) | package marker |
| `annofix/ui/colors.py` (new) | palette, `build_lut`, `recolour`, `edges` |
| `annofix/ui/overlay_item.py` (new) | `OverlayItem`: QImage over a numpy ARGB buffer |
| `annofix/ui/canvas.py` (new) | `Canvas`: image + overlay view, zoom/pan, mouse → image-pixel signals |
| `annofix/ui/tree_panel.py` (new) | `ObjectTree`: objects → parts, visibility checkboxes, context menu |
| `annofix/ui/label_panel.py` (new) | `LabelPanel`: material buttons, part-name suggestions, class box |
| `annofix/ui/main_window.py` (new) | `MainWindow`, hotkeys, toolbar, status, autosave, `install_excepthook` |
| `annofix/__main__.py` (new) | `python -m annofix` CLI |
| `tests/helpers.py` (new) | shared test data + `send_move` |
| `tests/test_session.py`, `tests/test_session_edit.py`, `tests/test_colors.py`, `tests/test_canvas.py`, `tests/test_panels.py`, `tests/test_main_window.py`, `tests/test_sam_helpers.py` (new); `tests/test_annotation_core.py`, `tests/test_io.py`, `tests/test_config.py`, `tests/conftest.py` (modify) | tests |
| `README.md` (modify) | GUI usage and shortcut table |

---

### Task 1: Transactions that span calls, and honest history accounting

**Files:**
- Modify: `annofix/annotation.py` (imports, `_Entry`, `Annotation.__init__`, class docstring, the whole "transactions and history" section)
- Test: `tests/test_annotation_core.py` (append)

**Interfaces:**
- Produces (used by Tasks 2–4 and Plan 3):
  - `Annotation.in_transaction -> bool` (property)
  - `Annotation.begin(label: str) -> None`, `Annotation.commit() -> None`, `Annotation.rollback() -> None` (both raise `RuntimeError` without an open `begin`)
  - `Annotation.transaction(label)` keeps its contract: one undo step; an exception rolls back the outermost transaction only.
  - `_Entry.meta_before` / `meta_after` become compact JSON `str` (internal).

- [ ] **Step 1: Write the failing tests** (append to `tests/test_annotation_core.py`)

```python


def test_begin_commit_spans_calls_as_one_undo_step():
    ann = Annotation.empty(4, 4)
    pid = ann.new_part(ann.new_object())
    ann.begin("stroke")
    ann.label_map[0, 0] = pid      # mouse event 1
    ann.set_part_name(pid, "x")    # an op inside the open stroke
    ann.label_map[1, 1] = pid      # mouse event 2
    assert ann.in_transaction
    ann.commit()
    assert not ann.in_transaction
    assert ann.undo() == "stroke"
    assert not ann.label_map.any() and ann.part(pid).name is None


def test_rollback_abandons_the_open_stroke():
    ann = Annotation.empty(4, 4)
    pid = ann.new_part(ann.new_object())
    ann.begin("stroke")
    ann.label_map[0, 0] = pid
    ann.set_part_material(pid, "metal")
    ann.rollback()
    assert not ann.in_transaction and not ann.label_map.any() and ann.part(pid).material is None
    assert ann.undo() == "new part"


def test_commit_and_rollback_without_begin_are_errors():
    ann = Annotation.empty(2, 2)
    with pytest.raises(RuntimeError):
        ann.commit()
    with pytest.raises(RuntimeError):
        ann.rollback()


def test_failing_op_inside_open_stroke_leaves_it_open_for_the_caller():
    ann = Annotation.empty(4, 4)
    pid = ann.new_part(ann.new_object())
    ann.begin("stroke")
    ann.label_map[0, 0] = pid
    with pytest.raises(KeyError):
        ann.set_part_name(999, "x")
    assert ann.in_transaction and ann.label_map[0, 0] == pid
    ann.rollback()
    assert not ann.label_map.any()


def test_label_only_edits_count_against_the_history_cap():
    ann = Annotation.empty(4, 4, history_bytes=2_000)
    pid = ann.new_part(ann.new_object())
    for i in range(100):
        ann.set_part_name(pid, f"name{i}")
    undos = 0
    while ann.undo():
        undos += 1
    assert 1 <= undos < 100
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_annotation_core.py -v`
Expected: the 5 new tests FAIL (`AttributeError: 'Annotation' object has no attribute 'begin'`; the history test fails `assert 1 <= 100 < 100`); the 15 existing tests PASS.

- [ ] **Step 3: Implement**

In `annofix/annotation.py`:

1. Add `import json` next to `from contextlib import contextmanager`.
2. Replace the `_Entry` dataclass with:

```python
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
```

3. Put this docstring as the first line of `class Annotation:`:

```python
    """Callers that live across edits (the GUI) must hold part/object ids, not Part/AnnObject
    instances: undo, redo and rollback rebuild those instances from a snapshot."""
```

4. In `Annotation.__init__`, after `self._tx_depth = 0`, add:

```python
        self._tx_label = ""
        self._tx_map: np.ndarray | None = None
        self._tx_meta: str | None = None
```

5. Replace everything from `    # ---- transactions and history ---` down to (not including) `    def can_undo(self) -> bool:` with:

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_annotation_core.py tests/test_annotation_ops.py -v`
Expected: all PASS (20 core + 20 ops).
Run: `uv run pytest`
Expected: all PASS, SAM tests skipped.

- [ ] **Step 5: Commit**

```bash
git add annofix/annotation.py tests/test_annotation_core.py
git commit -m "feat: begin/commit/rollback edits spanning calls; count metadata in history cap"
```

---

### Task 2: Core fixes the editor needs (save guard, lone PNG, temp files, unknown names, cuda:N)

**Files:**
- Modify: `annofix/io.py`, `annofix/config.py`, `annofix/sam_backend.py`
- Test: `tests/test_io.py` (append), `tests/test_config.py` (append), `tests/test_sam_helpers.py` (new)

**Interfaces:**
- Consumes: Task 1 `Annotation.in_transaction`, `begin`, `rollback`.
- Produces:
  - `save_annotation(...)` raises `RuntimeError("... open edit ...")` while `ann.in_transaction`.
  - `has_annotation(ann_dir, stem)` is True if either the PNG or the JSON exists.
  - `load_annotation` of a PNG without JSON returns the pixels with all ids in one class-less orphan object, plus a warning containing `"no JSON"`.
  - `Labels.unknown_names(ann) -> list[str]`, exact formats `"object {id}: class {cls!r} is not in labels.yaml"` and `"part {id}: material {m!r} is not in labels.yaml"` (part names are free text, never reported).
  - `sam_backend.device_kind(device: str) -> str` (`"cuda:0"` → `"cuda"`).

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_io.py`:

```python


def test_save_refused_inside_an_open_edit(tmp_path):
    ann = Annotation.empty(4, 4)
    pid = ann.new_part(ann.new_object())
    save_annotation(ann, tmp_path, "img")
    png, _ = annotation_paths(tmp_path, "img")
    before = png.read_bytes()
    ann.begin("stroke")
    ann.label_map[0, 0] = pid
    with pytest.raises(RuntimeError, match="open edit"):
        save_annotation(ann, tmp_path, "img")
    assert png.read_bytes() == before
    ann.rollback()


def test_lone_png_is_recovered_not_hidden(tmp_path):
    lm = np.zeros((4, 4), np.uint16)
    lm[1, 1] = 5
    save_annotation(Annotation(lm, [AnnObject(1, "mug", [Part(5)])], next_id=6), tmp_path, "img")
    _, js = annotation_paths(tmp_path, "img")
    js.unlink()  # a crash between the PNG and the JSON on an image's first save
    assert has_annotation(tmp_path, "img")
    back = load_annotation(tmp_path, "img")
    assert back.part_at(1, 1) == 5 and back.object_of(5).cls is None
    assert any("no JSON" in w for w in back.warnings)


def test_failed_write_leaves_target_and_no_temp_file(tmp_path, monkeypatch):
    ann = big_id_ann()
    save_annotation(ann, tmp_path, "img")
    png, _ = annotation_paths(tmp_path, "img")
    before = png.read_bytes()

    def disk_full(fd):
        raise OSError("disk full")

    monkeypatch.setattr(aio.os, "fsync", disk_full)
    ann.label_map[:] = 0
    with pytest.raises(OSError):
        save_annotation(ann, tmp_path, "img")
    assert png.read_bytes() == before
    assert sorted(p.name for p in tmp_path.iterdir()) == ["img.json", "img.parts.png"]
```

Append to `tests/test_config.py`:

```python


def test_unknown_classes_and_materials_are_reported():
    from annofix.annotation import AnnObject, Annotation, Part

    labels = load_labels(REPO / "labels.yaml")
    ann = Annotation.empty(2, 2)
    ann.objects = [AnnObject(1, "spaceship", [Part(2, "wing", "unobtainium"), Part(3, "free text", "metal")]),
                   AnnObject(4, "mug", [Part(5, None, None)])]
    assert labels.unknown_names(ann) == ["object 1: class 'spaceship' is not in labels.yaml",
                                         "part 2: material 'unobtainium' is not in labels.yaml"]
```

Create `tests/test_sam_helpers.py`:

```python
from annofix.sam_backend import device_kind


def test_device_kind_ignores_the_device_index():
    assert device_kind("cuda") == device_kind("cuda:0") == device_kind("cuda:1") == "cuda"
    assert device_kind("cpu") == "cpu"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_io.py tests/test_config.py tests/test_sam_helpers.py -v`
Expected: FAIL — `test_save_refused…` (DID NOT RAISE), `test_lone_png…` (has_annotation False), `test_failed_write…` (a leftover `img.parts.png.tmp`), `test_unknown_classes…` (no attribute `unknown_names`), `test_sam_helpers.py` (ImportError `device_kind`).

- [ ] **Step 3: Implement**

`annofix/io.py` — replace `has_annotation`, `_atomic_write`, `save_annotation` and `load_annotation` with:

```python
def has_annotation(ann_dir: str | Path, stem: str) -> bool:
    """True if either file exists: a lone PNG (crash between the two writes of a first save)
    must still count, or the next save would overwrite it unseen."""
    return any(p.exists() for p in annotation_paths(ann_dir, stem))


def _fsync_dir(d: Path) -> None:
    """Make the rename itself durable (ext4 can otherwise lose it on power loss)."""
    try:
        fd = os.open(d, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError:
        pass
    finally:
        os.close(fd)


def _atomic_write(path: Path, data: bytes) -> None:
    tmp = path.with_name(path.name + ".tmp")
    try:
        with open(tmp, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    _fsync_dir(path.parent)


def save_annotation(ann: Annotation, ann_dir: str | Path, stem: str) -> None:
    if ann.in_transaction:
        raise RuntimeError(f"cannot save {stem} in the middle of an open edit; commit or roll back first")
    png_path, json_path = annotation_paths(ann_dir, stem)
    meta_bytes = json.dumps(ann.meta_dict(), indent=2).encode()  # fail before touching disk
    ok, buf = cv2.imencode(".png", ann.label_map)
    if not ok:
        raise OSError(f"PNG encode failed for {png_path}")
    png_path.parent.mkdir(parents=True, exist_ok=True)
    _atomic_write(png_path, buf.tobytes())  # PNG first: the JSON never names a missing map
    _atomic_write(json_path, meta_bytes)
    ann.mark_saved()


def load_annotation(ann_dir: str | Path, stem: str,
                    expected_size: tuple[int, int] | None = None) -> Annotation:
    """Load and reconcile. `expected_size` is the image's (H, W); a mismatch is an error.
    A PNG without its JSON is recovered: its ids become one class-less orphan object."""
    png_path, json_path = annotation_paths(ann_dir, stem)
    meta = None
    if json_path.exists():
        meta = json.loads(json_path.read_text())
        version = meta.get("schema_version")
        if version != SCHEMA_VERSION:
            raise ValueError(f"{json_path}: schema_version {version!r}; this AnnoFix reads {SCHEMA_VERSION}")
    elif not png_path.exists():
        raise FileNotFoundError(f"no annotation for {stem} in {ann_dir}")
    label_map = cv2.imread(str(png_path), cv2.IMREAD_UNCHANGED)
    if label_map is None:
        raise ValueError(f"cannot read {png_path}")
    if label_map.dtype != np.uint16 or label_map.ndim != 2:
        raise ValueError(f"{png_path}: expected single-channel uint16, got {label_map.dtype} {label_map.shape}")
    lone_png = meta is None
    if lone_png:
        meta = {"schema_version": SCHEMA_VERSION, "size": list(label_map.shape)}
    if list(label_map.shape) != list(meta.get("size", label_map.shape)):
        raise ValueError(f"{stem}: label map is {list(label_map.shape)} but the JSON says size {meta['size']}")
    if expected_size is not None and tuple(label_map.shape) != tuple(expected_size):
        h, w = label_map.shape
        raise ValueError(f"{stem}: annotation is {w}x{h} but the image is {expected_size[1]}x{expected_size[0]}")
    ann = Annotation.from_meta(label_map, meta)
    if lone_png:
        ann.warnings.append(f"no JSON next to {png_path.name}: part ids recovered from the label map, "
                            "labels lost")
    ann.reconcile()
    return ann
```

`annofix/config.py` — add this method to `class Labels` (after `parts_for`):

```python
    def unknown_names(self, ann) -> list[str]:
        """Classes and materials not in labels.yaml (they are kept, but worth a warning). Part
        names are free text by design, so they are never reported."""
        out = []
        for o in ann.objects:
            if o.cls and o.cls not in self.objects:
                out.append(f"object {o.id}: class {o.cls!r} is not in labels.yaml")
            for p in o.parts:
                if p.material and p.material not in self.materials:
                    out.append(f"part {p.id}: material {p.material!r} is not in labels.yaml")
        return out
```

`annofix/sam_backend.py`:
- Add after `pick_device`:

```python
def device_kind(device: str) -> str:
    """'cuda:0' -> 'cuda'. The default model and bf16 autocast depend on the kind, not the index."""
    return device.split(":", 1)[0]
```

- In `Sam2Backend.__init__`, replace the two lines

```python
        self.device = pick_device(device)
        self.model_name = model or DEFAULT_MODEL["cuda" if self.device == "cuda" else "cpu"]
```

with

```python
        self.device = pick_device(device)
        self._cuda = device_kind(self.device) == "cuda"
        self.model_name = model or DEFAULT_MODEL["cuda" if self._cuda else "cpu"]
```

and replace `        if self.device == "cpu":` with `        if not self._cuda:`.
- In `_context`, replace `        if self.device == "cuda":` with `        if self._cuda:`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_io.py tests/test_config.py tests/test_sam_helpers.py tests/test_bootstrap.py -v`
Expected: all PASS (bootstrap's protection still holds: for a lone PNG its digest is `None`, so it reports `protected`).
Run: `uv run pytest`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add annofix/io.py annofix/config.py annofix/sam_backend.py tests/test_io.py tests/test_config.py tests/test_sam_helpers.py
git commit -m "fix: refuse saves mid-edit, recover lone PNGs, clean temp files, warn on unknown names"
```

---

### Task 3: Session — images, saving and review (no Qt)

**Files:**
- Create: `annofix/session.py`, `tests/helpers.py`, `tests/test_session.py`

**Interfaces:**
- Consumes: `io.list_images`, `io.load_image`, `io.has_annotation`, `io.load_annotation`, `io.save_annotation`, `io.annotation_paths`; `Annotation` (Tasks 1–2); `Labels.unknown_names`.
- Produces (used by Tasks 4, 5, 8):
  - constants `COLOUR_MODES = ("material", "part", "object")`, `TOOLS = ("select", "brush", "eraser")`
  - `class Session(images_dir, ann_dir, labels: Labels, annotator: str | None = None)` with attributes `images: list[Path]`, `ann_dir`, `labels`, `annotator`, `index`, `image` (RGB ndarray), `ann: Annotation`, `selected: list[int]`, `hidden_parts: set[int]`, `hidden_objects: set[int]`, `solo_object: int | None`, `colour_mode`, `tool`, `brush_radius`, `protect_others`, `messages: list[str]`
  - properties `stem`, `primary` (first selected part id or None), `painting`
  - `subscribe(fn: Callable[[str], None])`; kinds `"image" | "annotation" | "pixels" | "selection" | "view" | "status"`
  - `open(index)`, `step(delta) -> bool`, `save() -> bool`, `end_stroke()`, `review_status(index) -> str` (`"none"|"unreviewed"|"reviewed"|"flagged"`), `next_unreviewed() -> bool`, `mark_reviewed() -> list[str]`, `toggle_flag()`
  - internal `_notify(kind)`, `_changed()` (prune ids, refresh messages, notify `"annotation"`)
- `tests/helpers.py` produces `LABELS`, `H`, `W`, `scene_annotation()`, `make_dataset(tmp_path) -> (images_dir, ann_dir)`, `send_move(widget, pos)`.

- [ ] **Step 1: Write the shared test helpers `tests/helpers.py`**

```python
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
```

- [ ] **Step 2: Write the failing test `tests/test_session.py`**

```python
"""Session: images, saving and review."""
import pytest
from helpers import H, LABELS, W, make_dataset

from annofix.annotation import Annotation
from annofix.io import has_annotation, load_annotation, save_annotation
from annofix.session import Session


@pytest.fixture
def dataset(tmp_path):
    return make_dataset(tmp_path)


@pytest.fixture
def s(dataset):
    session = Session(*dataset, LABELS, annotator="tester")
    session.events = []
    session.subscribe(session.events.append)
    session.open(0)
    return session


def test_open_loads_annotation_and_reports_unknown_names(s):
    assert s.stem == "a" and s.ann.part_at(10, 10) == 2
    assert "object 4: class 'spaceship' is not in labels.yaml" in s.messages
    assert s.events == ["image"]


def test_image_without_annotation_opens_empty_and_writes_nothing(s, dataset):
    s.open(1)
    assert s.ann.objects == [] and s.ann.size == (H, W) and s.review_status(1) == "none"
    s.open(0)
    assert not has_annotation(dataset[1], "b")


def test_other_resolution_gets_a_matching_annotation(s):
    s.open(2)
    assert s.ann.size == (30, 50) == s.image.shape[:2]


def test_navigation_saves_edits_with_annotator(s, dataset):
    s.ann.set_part_material(2, "rubber")
    assert s.step(1) and s.stem == "b"
    back = load_annotation(dataset[1], "a")
    assert back.part(2).material == "rubber" and back.annotator == "tester"


def test_step_stops_at_the_ends(s):
    assert not s.step(-1) and s.stem == "a"


def test_save_only_when_dirty_and_never_mid_edit(s, dataset):
    assert s.save() is False
    s.ann.begin("stroke")
    s.ann.label_map[35, 5] = 2
    assert s.save() is False
    assert load_annotation(dataset[1], "a").part_at(35, 5) is None
    s.ann.commit()
    assert s.save() is True
    assert load_annotation(dataset[1], "a").part_at(35, 5) == 2


def test_review_status_and_next_unreviewed(s, dataset):
    save_annotation(Annotation.empty(H, W, review_status="reviewed"), dataset[1], "b")
    assert [s.review_status(i) for i in range(3)] == ["unreviewed", "reviewed", "none"]
    assert s.next_unreviewed() and s.stem == "c"


def test_mark_reviewed_requires_complete_labels(s, dataset):
    problems = s.mark_reviewed()
    assert "part 3 (object 1): no material" in problems and s.ann.review_status == "unreviewed"
    s.ann.set_part_material(3, "metal")
    s.ann.set_object_class(4, "mug")
    s.ann.set_part_name(5, "hull")
    s.ann.set_part_material(5, "metal")
    assert s.mark_reviewed() == []
    assert load_annotation(dataset[1], "a").review_status == "reviewed"
    assert s.review_status(0) == "reviewed"


def test_flag_toggles_and_saves(s, dataset):
    s.toggle_flag()
    assert load_annotation(dataset[1], "a").review_status == "flagged"
    s.toggle_flag()
    assert load_annotation(dataset[1], "a").review_status == "unreviewed"


def test_bad_annotation_file_leaves_current_image_open(s, dataset):
    save_annotation(Annotation.empty(10, 10), dataset[1], "b")  # wrong size for b.png
    with pytest.raises(ValueError, match="image is"):
        s.open(1)
    assert s.stem == "a" and s.ann.part_at(10, 10) == 2


def test_empty_folder_is_an_error(tmp_path):
    (tmp_path / "empty").mkdir()
    with pytest.raises(ValueError, match="no colour images"):
        Session(tmp_path / "empty", tmp_path / "ann", LABELS)
```

- [ ] **Step 3: Run test to verify it fails**

Run: `uv run pytest tests/test_session.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'annofix.session'`.

- [ ] **Step 4: Write `annofix/session.py`**

```python
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
```

- [ ] **Step 5: Run test to verify it passes**

Run: `uv run pytest tests/test_session.py -v`
Expected: PASS (11 tests).

- [ ] **Step 6: Commit**

```bash
git add annofix/session.py tests/helpers.py tests/test_session.py
git commit -m "feat: Qt-free editor session with image navigation, safe saving and review"
```

---

### Task 4: Session — selection, labels, brush/eraser and structural edits

**Files:**
- Modify: `annofix/session.py` (append methods to `class Session`)
- Create: `tests/test_session_edit.py`

**Interfaces:**
- Consumes: Task 3 `Session` (`_changed`, `_notify`, `end_stroke`, `_stroke`), Task 1 `begin/commit/rollback`, Plan 1 `Annotation` ops.
- Produces (used by Task 8 and Plan 3):
  - selection: `select_part(part_id: int | None, add=False)`, `select_at(x, y, add=False)`, `select_next_object(step=1)`
  - labels: `set_material(name)`, `set_material_index(i) -> bool`, `set_part_name(name)`, `set_object_class(cls)`
  - view: `set_colour_mode(mode)`, `set_hidden(kind: str, id: int, hidden: bool)` (`kind` = `"part"|"object"`), `toggle_part_hidden(pid)`, `toggle_object_hidden(oid)`, `toggle_solo()`, `set_tool(tool)`, `set_brush_radius(r)` (clamped 1..200), `set_protect(on)`
  - strokes: `begin_stroke(x, y) -> bool`, `stroke_to(x, y)`, `cancel_stroke()` (+ Task 3 `end_stroke`)
  - structure: `new_object() -> int` (new object with one empty part, selected, tool → brush), `new_part() -> int`, `merge_selected_parts() -> bool`, `merge_selected_objects() -> bool`, `delete_parts(ids)`, `delete_selected_parts()`, `delete_object(oid)`, `delete_selected_object()`, `move_part_to(pid, oid)`, `promote_part(pid) -> int`, `clean_part(pid)`
  - history: `undo() -> str | None`, `redo() -> str | None` (no-ops while painting)
  - Coordinates are image pixels: `x` = column, `y` = row.

- [ ] **Step 1: Write the failing test `tests/test_session_edit.py`**

```python
"""Session: selection, labels, strokes and structural edits."""
import time

import cv2
import numpy as np
import pytest
from helpers import LABELS, make_dataset

from annofix.io import load_annotation
from annofix.session import Session


@pytest.fixture
def dataset(tmp_path):
    return make_dataset(tmp_path)


@pytest.fixture
def s(dataset):
    session = Session(*dataset, LABELS, annotator="tester")
    session.open(0)
    return session


def test_selection_click_and_add(s):
    s.select_at(10, 10)
    assert s.selected == [2]
    s.select_at(35, 10, add=True)
    assert s.selected == [2, 3]
    s.select_at(35, 10, add=True)
    assert s.selected == [2]
    s.select_at(55, 35)  # background clears
    assert s.selected == []
    s.select_at(500, 500)  # outside the image is background too
    assert s.selected == []


def test_material_applies_to_every_selected_part_as_one_undo_step(s):
    s.select_at(10, 10)
    s.select_at(35, 10, add=True)
    assert s.set_material_index(0)
    assert s.ann.part(2).material == s.ann.part(3).material == "plastic"
    s.undo()
    assert s.ann.part(2).material == "metal" and s.ann.part(3).material is None
    assert not s.set_material_index(9)


def test_names_and_classes(s):
    s.select_at(35, 10)
    s.set_part_name("  rim ")
    s.set_object_class("mug")
    assert s.ann.part(3).name == "rim" and s.ann.object(1).cls == "mug"
    s.set_part_name("")
    assert s.ann.part(3).name is None


def test_unknown_class_typed_by_user_is_warned(s):
    s.select_at(10, 10)
    s.set_object_class("teapot")
    assert "object 1: class 'teapot' is not in labels.yaml" in s.messages


def test_brush_stroke_is_continuous_and_one_undo_step(s):
    s.select_at(10, 10)
    s.set_tool("brush")
    s.set_brush_radius(1)
    before = s.ann.label_map.copy()
    assert s.begin_stroke(5, 35)
    s.stroke_to(55, 35)
    s.end_stroke()
    assert (s.ann.label_map[35, 5:56] == 2).all()  # no gaps between the two mouse events
    assert s.undo() == "brush"
    assert np.array_equal(s.ann.label_map, before)


def test_brush_needs_a_selected_part(s):
    s.set_tool("brush")
    assert not s.begin_stroke(5, 35) and not s.ann.in_transaction
    assert "select a part" in s.messages[-1]


def test_protect_others_keeps_neighbours(s):
    s.select_at(10, 10)
    s.set_tool("brush")
    s.set_brush_radius(2)
    s.set_protect(True)
    s.begin_stroke(35, 10)
    s.end_stroke()
    assert s.ann.part_at(10, 35) == 3
    s.set_protect(False)
    s.begin_stroke(35, 10)
    s.end_stroke()
    assert s.ann.part_at(10, 35) == 2


def test_eraser_only_erases_the_selected_part(s):
    s.select_at(10, 10)
    s.set_tool("eraser")
    s.set_brush_radius(30)
    s.begin_stroke(25, 10)
    s.end_stroke()
    assert s.ann.part_area(2) == 0 and s.ann.part_area(3) == 100


def test_cancel_stroke_restores(s):
    s.select_at(10, 10)
    s.set_tool("brush")
    before = s.ann.label_map.copy()
    s.begin_stroke(5, 35)
    s.stroke_to(50, 35)
    s.cancel_stroke()
    assert np.array_equal(s.ann.label_map, before) and not s.ann.in_transaction


def test_autosave_mid_stroke_waits_for_the_stroke(s, dataset):
    s.select_at(10, 10)
    s.set_tool("brush")
    s.set_brush_radius(1)
    s.begin_stroke(5, 35)
    assert s.save() is False
    assert load_annotation(dataset[1], "a").part_at(35, 5) is None
    s.stroke_to(20, 35)
    s.end_stroke()
    assert s.save() is True
    assert load_annotation(dataset[1], "a").part_at(35, 20) == 2


def test_navigating_mid_stroke_keeps_the_stroke(s, dataset):
    s.select_at(10, 10)
    s.set_tool("brush")
    s.set_brush_radius(1)
    s.begin_stroke(5, 35)
    s.stroke_to(20, 35)
    s.step(1)
    assert load_annotation(dataset[1], "a").part_at(35, 20) == 2


def test_undo_prunes_selection_of_parts_that_no_longer_exist(s):
    pid = s.new_part()  # nothing selected -> a new object with one part
    assert s.selected == [pid] and s.tool == "brush"
    s.toggle_part_hidden(pid)
    s.toggle_solo()
    s.undo()
    assert s.selected == [] and pid not in s.hidden_parts and s.solo_object is None
    s.select_at(10, 10)
    assert s.selected == [2]


def test_new_part_goes_into_the_selected_object(s):
    s.select_at(10, 10)
    pid = s.new_part()
    assert s.ann.object_of(pid).id == 1 and s.selected == [pid]


def test_merge_selected_parts(s):
    s.select_at(10, 10)
    s.select_at(35, 10, add=True)
    assert s.merge_selected_parts()
    assert s.ann.part_area(2) == 300 and 3 not in s.ann.part_ids() and s.selected == [2]


def test_merge_selected_objects_like_a_split_robot_arm(s):
    s.select_at(10, 10)            # part 2 of object 1
    s.select_at(10, 25, add=True)  # part 5 of object 4
    assert s.merge_selected_objects()
    assert [o.id for o in s.ann.objects] == [1]
    assert sorted(p.id for p in s.ann.object(1).parts) == [2, 3, 5]
    s.undo()
    assert [o.id for o in s.ann.objects] == [1, 4]


def test_delete_selected_object_and_parts(s):
    s.select_at(35, 10)
    s.delete_selected_object()
    assert [o.id for o in s.ann.objects] == [4] and s.selected == []
    s.select_at(10, 25)
    s.delete_selected_parts()
    assert s.ann.part_ids() == []


def test_move_promote_and_clean(s):
    s.move_part_to(5, 1)
    assert s.ann.object_of(5).id == 1
    oid = s.promote_part(5)
    assert s.ann.object_of(5).id == oid
    s.clean_part(2)
    assert s.ann.part_area(2) == 200


def test_next_object_cycles(s):
    s.select_next_object()
    assert s.selected == [2]
    s.select_next_object()
    assert s.selected == [5]
    s.select_next_object()
    assert s.selected == [2]
    s.select_next_object(-1)
    assert s.selected == [5]


def test_view_options(s):
    s.select_at(10, 10)
    s.toggle_solo()
    assert s.solo_object == 1
    s.toggle_solo()
    assert s.solo_object is None
    s.set_hidden("object", 4, True)
    assert s.hidden_objects == {4}
    s.set_hidden("object", 4, False)
    assert s.hidden_objects == set()
    s.set_colour_mode("part")
    assert s.colour_mode == "part"
    with pytest.raises(ValueError):
        s.set_colour_mode("rainbow")
    s.set_brush_radius(0)
    assert s.brush_radius == 1


def test_stroke_segments_are_fast_enough_for_live_painting(tmp_path):
    images = tmp_path / "images"
    images.mkdir()
    cv2.imwrite(str(images / "big.png"), np.zeros((720, 1280, 3), np.uint8))
    s = Session(images, tmp_path / "ann", LABELS)
    s.open(0)
    s.new_object()
    s.set_brush_radius(10)
    s.begin_stroke(100, 100)
    times = []
    for i in range(30):
        t = time.perf_counter()
        s.stroke_to(100 + 30 * i, 100 + 15 * i)
        times.append(time.perf_counter() - t)
    s.end_stroke()
    assert sorted(times)[15] < 0.015
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_session_edit.py -v`
Expected: FAIL — `AttributeError: 'Session' object has no attribute 'select_at'` (and similar).

- [ ] **Step 3: Append the editing methods to `class Session` in `annofix/session.py`**

```python

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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_session_edit.py tests/test_session.py -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add annofix/session.py tests/test_session_edit.py
git commit -m "feat: session editing actions: selection, labels, strokes, merges, deletes, undo"
```

---

### Task 5: Overlay colours (pure numpy)

**Files:**
- Create: `annofix/ui/__init__.py` (empty), `annofix/ui/colors.py`, `tests/test_colors.py`

**Interfaces:**
- Consumes: `session.COLOUR_MODES`, `Annotation`, `Labels`.
- Produces (used by Tasks 6 and 8):
  - `PALETTE: list[tuple[int, int, int]]`, `UNLABELLED`, `LUT_SIZE = 65536`
  - `argb(rgb) -> int` (`0xFFRRGGBB`), `id_colour(i) -> rgb`, `material_colour(material, labels) -> rgb`, `lighten(rgb, amount=0.5) -> rgb`
  - `build_lut(ann, labels, mode="material", *, hidden_parts=frozenset(), hidden_objects=frozenset(), solo_object=None, selected=frozenset()) -> np.ndarray` (uint32, length 65536, 0 = transparent)
  - `edges(label_map, out=None) -> bool ndarray`
  - `recolour(label_map, lut, out, outline=False, edge_buf=None) -> out` (fills a uint32 `out` in place)

- [ ] **Step 1: Write the failing test `tests/test_colors.py`**

```python
import time

import numpy as np
import pytest
from helpers import LABELS, scene_annotation

from annofix.ui.colors import (UNLABELLED, argb, build_lut, edges, id_colour, lighten, material_colour,
                               recolour)


def test_material_mode_colours_by_material():
    lut = build_lut(scene_annotation(), LABELS, "material")
    assert lut[2] == argb(material_colour("metal", LABELS))
    assert lut[3] == argb(UNLABELLED)
    assert lut[5] == argb(UNLABELLED)
    assert lut[0] == 0
    assert material_colour("plastic", LABELS) != material_colour("metal", LABELS)


def test_object_mode_gives_all_parts_of_an_object_one_colour():
    lut = build_lut(scene_annotation(), LABELS, "object")
    assert lut[2] == lut[3] == argb(id_colour(1)) and lut[5] == argb(id_colour(4))


def test_part_mode_colours_each_part():
    lut = build_lut(scene_annotation(), LABELS, "part")
    assert len({int(lut[2]), int(lut[3]), int(lut[5])}) == 3


def test_hidden_solo_and_selected():
    ann = scene_annotation()
    assert build_lut(ann, LABELS, hidden_parts={3})[3] == 0
    lut = build_lut(ann, LABELS, hidden_objects={1})
    assert lut[2] == lut[3] == 0 and lut[5] != 0
    lut = build_lut(ann, LABELS, solo_object=4)
    assert lut[2] == 0 and lut[5] != 0
    assert build_lut(ann, LABELS, "part", selected={2})[2] == argb(lighten(id_colour(2)))


def test_unknown_mode_rejected():
    with pytest.raises(ValueError):
        build_lut(scene_annotation(), LABELS, "rainbow")


def test_recolour_and_outline():
    ann = scene_annotation()
    lut = build_lut(ann, LABELS, "part")
    out = np.zeros(ann.size, np.uint32)
    recolour(ann.label_map, lut, out)
    assert out[10, 10] == lut[2] and out[0, 0] == 0
    recolour(ann.label_map, lut, out, outline=True)
    assert out[5, 10] == lut[2] and out[10, 10] == 0  # border kept, interior cleared
    lm = np.zeros((5, 5), np.uint16)
    lm[1:4, 1:4] = 2
    e = edges(lm)
    assert e[1, 1] and e[3, 3] and not e[2, 2]


def test_recolour_full_frame_under_5ms():
    lm = np.random.default_rng(0).integers(0, 300, (720, 1280)).astype(np.uint16)
    lut = np.arange(65536, dtype=np.uint32) | np.uint32(0xFF000000)
    out = np.zeros(lm.shape, np.uint32)
    recolour(lm, lut, out)
    times = []
    for _ in range(10):
        t = time.perf_counter()
        recolour(lm, lut, out)
        times.append(time.perf_counter() - t)
    assert sorted(times)[5] < 0.005
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_colors.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'annofix.ui'`.

- [ ] **Step 3: Write `annofix/ui/__init__.py` (empty file) and `annofix/ui/colors.py`**

```python
"""Overlay colours as a lookup table: lut[part_id] -> 0xAARRGGBB (premultiplied; alpha is
255 or 0). Colour mode, hidden parts/objects, solo and the selection highlight are all just a
different table, so a full 1280x720 recolour is one np.take (~2 ms). Opacity is applied by
the Qt item, not here."""
from __future__ import annotations

import numpy as np

from ..config import Labels
from ..session import COLOUR_MODES

PALETTE = [
    (31, 119, 180), (255, 127, 14), (44, 160, 44), (214, 39, 40), (148, 103, 189),
    (140, 86, 75), (227, 119, 194), (188, 189, 34), (23, 190, 207), (174, 199, 232),
    (255, 187, 120), (152, 223, 138), (255, 152, 150), (197, 176, 213), (196, 156, 148),
    (247, 182, 210), (219, 219, 141), (158, 218, 229), (57, 59, 121), (99, 121, 57),
]
UNLABELLED = (128, 128, 128)
LUT_SIZE = 65536


def argb(rgb: tuple[int, int, int]) -> int:
    r, g, b = rgb
    return 0xFF000000 | (r << 16) | (g << 8) | b


def id_colour(i: int) -> tuple[int, int, int]:
    """Stable per id, and neighbouring ids differ (7 is coprime with the palette size)."""
    return PALETTE[(i * 7) % len(PALETTE)]


def material_colour(material: str | None, labels: Labels) -> tuple[int, int, int]:
    if material in labels.materials:
        return PALETTE[labels.materials.index(material) % len(PALETTE)]
    return UNLABELLED


def lighten(rgb: tuple[int, int, int], amount: float = 0.5) -> tuple[int, int, int]:
    return tuple(int(c + (255 - c) * amount) for c in rgb)


def build_lut(ann, labels: Labels, mode: str = "material", *, hidden_parts=frozenset(),
              hidden_objects=frozenset(), solo_object: int | None = None,
              selected=frozenset()) -> np.ndarray:
    if mode not in COLOUR_MODES:
        raise ValueError(f"colour mode must be one of {COLOUR_MODES}, got {mode!r}")
    lut = np.zeros(LUT_SIZE, np.uint32)
    for o in ann.objects:
        if o.id in hidden_objects or (solo_object is not None and o.id != solo_object):
            continue
        for p in o.parts:
            if p.id in hidden_parts:
                continue
            if mode == "material":
                rgb = material_colour(p.material, labels)
            elif mode == "part":
                rgb = id_colour(p.id)
            else:
                rgb = id_colour(o.id)
            if p.id in selected:
                rgb = lighten(rgb)
            lut[p.id] = argb(rgb)
    return lut


def edges(label_map: np.ndarray, out: np.ndarray | None = None) -> np.ndarray:
    """Pixels with a 4-neighbour of a different label, marked on both sides of each border."""
    e = np.zeros(label_map.shape, bool) if out is None else out
    e[...] = False
    dx = label_map[:, :-1] != label_map[:, 1:]
    e[:, :-1] |= dx
    e[:, 1:] |= dx
    dy = label_map[:-1, :] != label_map[1:, :]
    e[:-1, :] |= dy
    e[1:, :] |= dy
    return e


def recolour(label_map: np.ndarray, lut: np.ndarray, out: np.ndarray, outline: bool = False,
             edge_buf: np.ndarray | None = None) -> np.ndarray:
    np.take(lut, label_map, out=out)
    if outline:
        np.multiply(out, edges(label_map, edge_buf), out=out)
    return out
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_colors.py -v`
Expected: PASS (7 tests).

- [ ] **Step 5: Commit**

```bash
git add annofix/ui/__init__.py annofix/ui/colors.py tests/test_colors.py
git commit -m "feat: overlay colour lookup tables for material/part/object modes"
```

---

### Task 6: Overlay item and canvas (Qt)

**Files:**
- Modify: `pyproject.toml` / `uv.lock` (via `uv add --dev pytest-qt`), `tests/conftest.py`
- Create: `annofix/ui/overlay_item.py`, `annofix/ui/canvas.py`, `tests/test_canvas.py`

**Interfaces:**
- Consumes: Task 5 `recolour`, `LUT_SIZE`.
- Produces (used by Task 8):
  - `OverlayItem(label_map)` (QGraphicsItem): `set_lut(lut)`, `set_outline(on)`, `refresh()`, `argb_at(x, y) -> int`, attribute `outline`
  - `Canvas(QGraphicsView)`: signals `clicked(int x, int y, bool add)`, `stroke_started(int, int)`, `stroke_moved(int, int)`, `stroke_ended()`; attributes `paint_mode: bool`, `brush_radius: int`, `overlay: OverlayItem | None`; methods `set_image(rgb, label_map) -> OverlayItem`, `image_pos(view_point) -> (x, y)`, `set_masks_visible(bool)`

- [ ] **Step 1: Add pytest-qt and make Qt tests run offscreen**

Run: `uv add --dev "pytest-qt>=4.4"`
Expected: `pyproject.toml` `[dependency-groups] dev` gains `pytest-qt`; `uv.lock` updated.

Append to `tests/conftest.py`:

```python

# Qt tests run without a display.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
```

- [ ] **Step 2: Write the failing test `tests/test_canvas.py`**

```python
import numpy as np
from helpers import send_move
from PyQt6.QtCore import QPointF, Qt

from annofix.ui.canvas import Canvas
from annofix.ui.overlay_item import OverlayItem


def test_overlay_shows_lut_colour_and_follows_in_place_edits(qtbot):
    lm = np.zeros((4, 6), np.uint16)
    lm[1, 2] = 7
    item = OverlayItem(lm)
    lut = np.zeros(65536, np.uint32)
    lut[7], lut[9] = 0xFF112233, 0xFF445566
    item.set_lut(lut)
    assert item.argb_at(2, 1) == 0xFF112233 and item.argb_at(0, 0) == 0
    lm[0, 0] = 9  # the annotation edits its array in place...
    item.refresh()  # ...and one refresh shows it
    assert item.argb_at(0, 0) == 0xFF445566


def test_outline_mode_hides_interiors(qtbot):
    lm = np.zeros((5, 5), np.uint16)
    lm[1:4, 1:4] = 7
    item = OverlayItem(lm)
    lut = np.zeros(65536, np.uint32)
    lut[7] = 0xFF112233
    item.set_lut(lut)
    item.set_outline(True)
    assert item.argb_at(1, 1) == 0xFF112233 and item.argb_at(2, 2) == 0


def make_canvas(qtbot, h=40, w=60):
    c = Canvas()
    qtbot.addWidget(c)
    c.resize(600, 400)
    c.show()
    qtbot.waitExposed(c)
    c.set_image(np.full((h, w, 3), 90, np.uint8), np.zeros((h, w), np.uint16))
    return c


def at(c, x, y):
    return c.mapFromScene(QPointF(x + 0.5, y + 0.5))


def test_click_reports_image_pixel(qtbot):
    c = make_canvas(qtbot)
    with qtbot.waitSignal(c.clicked) as sig:
        qtbot.mouseClick(c.viewport(), Qt.MouseButton.LeftButton, pos=at(c, 12, 30))
    assert sig.args == [12, 30, False]


def test_ctrl_click_adds_to_selection(qtbot):
    c = make_canvas(qtbot)
    with qtbot.waitSignal(c.clicked) as sig:
        qtbot.mouseClick(c.viewport(), Qt.MouseButton.LeftButton, Qt.KeyboardModifier.ControlModifier,
                         at(c, 5, 5))
    assert sig.args == [5, 5, True]


def test_paint_mode_turns_drags_into_strokes(qtbot):
    c = make_canvas(qtbot)
    c.paint_mode = True
    got = []
    c.stroke_started.connect(lambda x, y: got.append(("start", x, y)))
    c.stroke_moved.connect(lambda x, y: got.append(("move", x, y)))
    c.stroke_ended.connect(lambda: got.append(("end",)))
    vp = c.viewport()
    qtbot.mousePress(vp, Qt.MouseButton.LeftButton, pos=at(c, 10, 10))
    send_move(vp, at(c, 20, 10))
    qtbot.mouseRelease(vp, Qt.MouseButton.LeftButton, pos=at(c, 20, 10))
    assert got[0] == ("start", 10, 10) and ("move", 20, 10) in got and got[-1] == ("end",)


def test_switching_to_a_smaller_image_rebuilds_the_overlay(qtbot):
    c = make_canvas(qtbot)
    item = c.set_image(np.zeros((30, 50, 3), np.uint8), np.zeros((30, 50), np.uint16))
    assert item.boundingRect().width() == 50 and c.scene().sceneRect().height() == 30
    assert sum(isinstance(i, OverlayItem) for i in c.scene().items()) == 1


def test_masks_can_be_hidden(qtbot):
    c = make_canvas(qtbot)
    c.set_masks_visible(False)
    assert not c.overlay.isVisible()
    c.set_masks_visible(True)
    assert c.overlay.isVisible()
```

- [ ] **Step 3: Run test to verify it fails**

Run: `uv run pytest tests/test_canvas.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'annofix.ui.canvas'`.

- [ ] **Step 4: Write `annofix/ui/overlay_item.py`**

```python
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
```

- [ ] **Step 5: Write `annofix/ui/canvas.py`**

```python
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
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `uv run pytest tests/test_canvas.py -v`
Expected: PASS (7 tests).
Run: `uv run pytest`
Expected: all PASS.

- [ ] **Step 7: Commit**

```bash
git add pyproject.toml uv.lock tests/conftest.py annofix/ui/overlay_item.py annofix/ui/canvas.py tests/test_canvas.py
git commit -m "feat: numpy-backed mask overlay and zoomable canvas with click/stroke signals"
```

---

### Task 7: Object tree and label panel

**Files:**
- Create: `annofix/ui/tree_panel.py`, `annofix/ui/label_panel.py`, `tests/test_panels.py`

**Interfaces:**
- Consumes: `Annotation` (read only, per call), `Labels`.
- Produces (used by Task 8):
  - `ObjectTree(QTreeWidget)`: `rebuild(ann, selected, hidden_parts, hidden_objects)`; signals `part_clicked(int, bool)`, `visibility_changed(str, int, bool)` (kind, id, visible), `move_requested(int, int)` (part id, object id), `promote_requested(int)`, `clean_requested(int)`, `delete_part_requested(int)`, `delete_object_requested(int)`; `_build_menu(item) -> QMenu` (also used by tests)
  - `LabelPanel(labels)`: `show_part(part: Part | None, obj: AnnObject | None)` (reads them immediately, keeps no reference); signals `material_chosen(str)`, `part_name_entered(str)`, `class_entered(str)`; attributes `material_buttons`, `name_buttons`, `name_edit`, `class_box`, `title`

- [ ] **Step 1: Write the failing test `tests/test_panels.py`**

```python
from helpers import LABELS, scene_annotation
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QMenu

from annofix.ui.label_panel import LabelPanel
from annofix.ui.tree_panel import ObjectTree


def make_tree(qtbot, selected=(3,), hidden_parts=frozenset({5})):
    t = ObjectTree()
    qtbot.addWidget(t)
    t.rebuild(scene_annotation(), list(selected), set(hidden_parts), set())
    return t


def test_tree_lists_objects_and_parts_with_visibility(qtbot):
    t = make_tree(qtbot)
    assert t.topLevelItemCount() == 2
    mug = t.topLevelItem(0)
    assert mug.childCount() == 2 and mug.isExpanded() and not t.topLevelItem(1).isExpanded()
    assert "handle" in mug.child(1).text(0) and mug.child(1).font(0).bold()
    assert t.topLevelItem(1).child(0).checkState(0) == Qt.CheckState.Unchecked


def test_tree_checkbox_emits_visibility(qtbot):
    t = make_tree(qtbot)
    with qtbot.waitSignal(t.visibility_changed) as sig:
        t.topLevelItem(0).setCheckState(0, Qt.CheckState.Unchecked)
    assert sig.args == ["object", 1, False]


def test_tree_click_selects_part_or_first_part_of_object(qtbot):
    t = make_tree(qtbot)
    got = []
    t.part_clicked.connect(lambda pid, add: got.append((pid, add)))
    t.itemClicked.emit(t.topLevelItem(0).child(1), 0)
    t.itemClicked.emit(t.topLevelItem(1), 0)
    assert got == [(3, False), (5, False)]


def test_tree_context_menu_actions(qtbot):
    t = make_tree(qtbot)
    got = []
    t.move_requested.connect(lambda pid, oid: got.append(("move", pid, oid)))
    t.promote_requested.connect(lambda pid: got.append(("promote", pid)))
    t.delete_object_requested.connect(lambda oid: got.append(("delete object", oid)))
    menu = t._build_menu(t.topLevelItem(0).child(1))
    move_menu = menu.findChildren(QMenu)[0]
    next(a for a in move_menu.actions() if a.text().startswith("#4")).trigger()
    next(a for a in menu.actions() if a.text() == "Make it a new object").trigger()
    t._build_menu(t.topLevelItem(1)).actions()[0].trigger()
    assert got == [("move", 3, 4), ("promote", 3), ("delete object", 4)]


def test_label_panel_emits_material_name_and_class(qtbot):
    p = LabelPanel(LABELS)
    qtbot.addWidget(p)
    ann = scene_annotation()
    p.show_part(ann.part(2), ann.object(1))
    got = []
    p.material_chosen.connect(lambda m: got.append(("material", m)))
    p.part_name_entered.connect(lambda n: got.append(("name", n)))
    p.class_entered.connect(lambda c: got.append(("class", c)))
    qtbot.mouseClick(p.material_buttons[1], Qt.MouseButton.LeftButton)
    assert [b.text() for b in p.name_buttons] == ["body", "handle"]
    qtbot.mouseClick(p.name_buttons[1], Qt.MouseButton.LeftButton)
    p.name_edit.setText("rim")
    qtbot.keyClick(p.name_edit, Qt.Key.Key_Return)
    p.class_box.setEditText("cup")
    qtbot.keyClick(p.class_box.lineEdit(), Qt.Key.Key_Return)
    assert got[:3] == [("material", "metal"), ("name", "handle"), ("name", "rim")]
    assert ("class", "cup") in got[3:]
    assert "Part #2" in p.title.text()


def test_label_panel_disabled_without_selection(qtbot):
    p = LabelPanel(LABELS)
    qtbot.addWidget(p)
    p.show_part(None, None)
    assert not p.material_buttons[0].isEnabled() and p.name_buttons == []


def test_panel_buttons_never_take_keyboard_focus(qtbot):
    p = LabelPanel(LABELS)
    qtbot.addWidget(p)
    ann = scene_annotation()
    p.show_part(ann.part(2), ann.object(1))
    assert all(b.focusPolicy() == Qt.FocusPolicy.NoFocus for b in p.material_buttons + p.name_buttons)
    assert "1" in p.material_buttons[0].text()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_panels.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'annofix.ui.label_panel'`.

- [ ] **Step 3: Write `annofix/ui/tree_panel.py`**

```python
"""Objects -> parts tree: tick boxes show/hide, a click selects, the context menu holds the
structure edits. Rebuilt from the annotation after every change (ids only are kept)."""
from __future__ import annotations

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import QAbstractItemView, QApplication, QMenu, QTreeWidget, QTreeWidgetItem

ROLE = Qt.ItemDataRole.UserRole  # ("object", id) or ("part", id)


class ObjectTree(QTreeWidget):
    part_clicked = pyqtSignal(int, bool)            # part id, add to selection
    visibility_changed = pyqtSignal(str, int, bool)  # "object" | "part", id, visible
    move_requested = pyqtSignal(int, int)           # part id, target object id
    promote_requested = pyqtSignal(int)
    clean_requested = pyqtSignal(int)
    delete_part_requested = pyqtSignal(int)
    delete_object_requested = pyqtSignal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setHeaderLabels(["Objects and parts"])
        self.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        self.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.customContextMenuRequested.connect(self._show_menu)
        self.itemClicked.connect(self._clicked)
        self.itemChanged.connect(self._changed)
        self._objects: list[tuple[int, str]] = []
        self._building = False

    def rebuild(self, ann, selected, hidden_parts, hidden_objects) -> None:
        scroll = self.verticalScrollBar().value()
        self._building = True
        self.clear()
        self._objects = []
        focus_item = None
        for o in ann.objects:
            label = f"#{o.id} {o.cls or '(no class)'}"
            self._objects.append((o.id, label))
            oi = QTreeWidgetItem([f"{label}  ({len(o.parts)} part{'s' if len(o.parts) != 1 else ''})"])
            oi.setData(0, ROLE, ("object", o.id))
            oi.setFlags(oi.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            oi.setCheckState(0, Qt.CheckState.Unchecked if o.id in hidden_objects else Qt.CheckState.Checked)
            for p in o.parts:
                pi = QTreeWidgetItem([f"#{p.id} {p.name or '?'} · {p.material or '?'}"])
                pi.setData(0, ROLE, ("part", p.id))
                pi.setFlags(pi.flags() | Qt.ItemFlag.ItemIsUserCheckable)
                pi.setCheckState(0, Qt.CheckState.Unchecked if p.id in hidden_parts else Qt.CheckState.Checked)
                if p.id in selected:
                    font = pi.font(0)
                    font.setBold(True)
                    pi.setFont(0, font)
                    pi.setBackground(0, QColor(70, 110, 160))
                    focus_item = focus_item or pi
                oi.addChild(pi)
            self.addTopLevelItem(oi)
            oi.setExpanded(any(p.id in selected for p in o.parts))
        self._building = False
        self.verticalScrollBar().setValue(scroll)
        if focus_item is not None:
            self.scrollToItem(focus_item)

    def _clicked(self, item, column) -> None:
        kind, item_id = item.data(0, ROLE)
        if kind == "part":
            add = bool(QApplication.keyboardModifiers() & Qt.KeyboardModifier.ControlModifier)
            self.part_clicked.emit(item_id, add)
        elif item.childCount():
            self.part_clicked.emit(item.child(0).data(0, ROLE)[1], False)

    def _changed(self, item, column) -> None:
        if self._building:
            return
        kind, item_id = item.data(0, ROLE)
        self.visibility_changed.emit(kind, item_id, item.checkState(0) == Qt.CheckState.Checked)

    def _show_menu(self, pos) -> None:
        item = self.itemAt(pos)
        if item is not None:
            self._build_menu(item).exec(self.viewport().mapToGlobal(pos))

    def _build_menu(self, item) -> QMenu:
        kind, item_id = item.data(0, ROLE)
        menu = QMenu(self)

        def add(target_menu, text, fn):
            action = target_menu.addAction(text)
            action.triggered.connect(lambda checked=False: fn())

        if kind == "part":
            move = menu.addMenu("Move to object")
            for oid, label in self._objects:
                add(move, label, lambda oid=oid: self.move_requested.emit(item_id, oid))
            add(menu, "Make it a new object", lambda: self.promote_requested.emit(item_id))
            add(menu, "Clean up (islands, holes)", lambda: self.clean_requested.emit(item_id))
            add(menu, "Delete part", lambda: self.delete_part_requested.emit(item_id))
        else:
            add(menu, "Delete object", lambda: self.delete_object_requested.emit(item_id))
        return menu
```

- [ ] **Step 4: Write `annofix/ui/label_panel.py`**

```python
"""Labels for the selected part: object class (suggestions from labels.yaml or free text),
material buttons (hotkeys 1-9 are window shortcuts), part-name suggestions for the class plus
a free-text field. Nothing here stores Part/AnnObject instances."""
from __future__ import annotations

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (QComboBox, QGridLayout, QHBoxLayout, QLabel, QLineEdit, QPushButton,
                             QVBoxLayout, QWidget)


class LabelPanel(QWidget):
    material_chosen = pyqtSignal(str)
    part_name_entered = pyqtSignal(str)
    class_entered = pyqtSignal(str)

    def __init__(self, labels, parent=None):
        super().__init__(parent)
        self.labels = labels
        layout = QVBoxLayout(self)
        self.title = QLabel()
        layout.addWidget(self.title)

        layout.addWidget(QLabel("Object class"))
        self.class_box = QComboBox()
        self.class_box.setEditable(True)
        self.class_box.setInsertPolicy(QComboBox.InsertPolicy.NoInsert)
        self.class_box.addItems(sorted(labels.objects))
        self.class_box.textActivated.connect(self.class_entered.emit)
        self.class_box.lineEdit().returnPressed.connect(lambda: self.class_entered.emit(self.class_box.currentText()))
        layout.addWidget(self.class_box)

        layout.addWidget(QLabel("Material"))
        grid = QGridLayout()
        self.material_buttons: list[QPushButton] = []
        for i, material in enumerate(labels.materials):
            b = QPushButton(f"{i + 1}  {material}" if i < 9 else material)
            b.setFocusPolicy(Qt.FocusPolicy.NoFocus)
            b.clicked.connect(lambda checked=False, m=material: self.material_chosen.emit(m))
            grid.addWidget(b, i // 3, i % 3)
            self.material_buttons.append(b)
        layout.addLayout(grid)

        layout.addWidget(QLabel("Part name"))
        names = QWidget()
        self._names_layout = QHBoxLayout(names)
        self._names_layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(names)
        self.name_buttons: list[QPushButton] = []
        self.name_edit = QLineEdit()
        self.name_edit.setPlaceholderText("type a part name and press Enter")
        self.name_edit.returnPressed.connect(lambda: self.part_name_entered.emit(self.name_edit.text()))
        layout.addWidget(self.name_edit)
        layout.addStretch(1)
        self.show_part(None, None)

    def show_part(self, part, obj) -> None:
        for b in self.name_buttons:
            self._names_layout.removeWidget(b)
            b.deleteLater()
        self.name_buttons = []
        enabled = part is not None
        for w in (self.class_box, self.name_edit, *self.material_buttons):
            w.setEnabled(enabled)
        if part is None:
            self.title.setText("No part selected: click a mask")
            self.class_box.setEditText("")
            self.name_edit.clear()
            for b in self.material_buttons:
                b.setStyleSheet("")
            return
        self.title.setText(f"Part #{part.id} of object #{obj.id}")
        self.class_box.setEditText(obj.cls or "")
        self.name_edit.setText(part.name or "")
        for name in self.labels.parts_for(obj.cls):
            b = QPushButton(name)
            b.setFocusPolicy(Qt.FocusPolicy.NoFocus)
            b.clicked.connect(lambda checked=False, n=name: self.part_name_entered.emit(n))
            self._names_layout.addWidget(b)
            self.name_buttons.append(b)
        for b, material in zip(self.material_buttons, self.labels.materials):
            b.setStyleSheet("font-weight: bold" if material == part.material else "")
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/test_panels.py -v`
Expected: PASS (7 tests).

- [ ] **Step 6: Commit**

```bash
git add annofix/ui/tree_panel.py annofix/ui/label_panel.py tests/test_panels.py
git commit -m "feat: object/part tree with visibility and context menu; label panel"
```

---

### Task 8: Main window, hotkeys, crash safety and the `python -m annofix` CLI

**Files:**
- Create: `annofix/ui/main_window.py`, `annofix/__main__.py`, `tests/test_main_window.py`

**Interfaces:**
- Consumes: Tasks 3–7 (`Session` API, `build_lut`, `Canvas`, `ObjectTree`, `LabelPanel`).
- Produces:
  - `MainWindow(session)` with attributes `session`, `canvas`, `tree`, `labels_panel`, `files` (QListWidget), `status` (QLabel), `outline_action`, `protect_action`, `tool_actions: dict[str, QAction]`, `mode_box`, `opacity`
  - `install_excepthook(session, log_path) -> hook` (also sets `sys.excepthook`)
  - `STATUS_ICONS = {"none": "○", "unreviewed": "•", "reviewed": "✓", "flagged": "⚑"}`
  - `annofix.__main__.main(argv=None) -> int` (1 on bad paths/labels)
- Hotkeys (all `QShortcut`): `1`–`9` material · `B` brush · `E` eraser · `V` select · `Esc` cancel stroke / back to select · `[` `]` brush size · `S` solo · `O` outline · `N` new part · `Shift+N` new object · `M` merge parts · `Shift+M` merge objects · `Del` delete part(s) · `Shift+Del` delete object · `Ctrl+Z` undo · `Ctrl+Shift+Z`/`Ctrl+Y` redo · `A`/`D` previous/next image · `Ctrl+U` next unreviewed · `Ctrl+Return` mark reviewed · `F` flag · `J`/`K` next/previous object · `Ctrl+S` save · hold `Space` hide masks.

- [ ] **Step 1: Write the failing test `tests/test_main_window.py`**

```python
import os
import subprocess
import sys
from pathlib import Path

import pytest
from helpers import LABELS, make_dataset, send_move
from PyQt6.QtCore import QPointF, Qt
from PyQt6.QtWidgets import QApplication, QMessageBox

from annofix.__main__ import main
from annofix.io import load_annotation
from annofix.session import Session
from annofix.ui.main_window import MainWindow, install_excepthook

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture
def win(tmp_path, qtbot, monkeypatch):
    images, ann_dir = make_dataset(tmp_path)
    monkeypatch.setattr(QMessageBox, "information", lambda *a, **k: None)
    monkeypatch.setattr(QMessageBox, "warning", lambda *a, **k: None)
    w = MainWindow(Session(images, ann_dir, LABELS))
    qtbot.addWidget(w)
    w.resize(1200, 700)
    w.show()
    qtbot.waitExposed(w)
    w.activateWindow()
    qtbot.waitUntil(lambda: QApplication.activeWindow() is w, timeout=2000)
    w.ann_dir = ann_dir
    return w


def at(w, x, y):
    return w.canvas.mapFromScene(QPointF(x + 0.5, y + 0.5))


def test_opens_first_image_with_overlay_and_tree(win):
    assert win.session.stem == "a"
    assert win.canvas.overlay.argb_at(10, 10) != 0 and win.canvas.overlay.argb_at(50, 35) == 0
    assert win.tree.topLevelItemCount() == 2
    assert win.files.item(0).text().startswith("•")


def test_click_then_hotkey_sets_material(win, qtbot):
    qtbot.mouseClick(win.canvas.viewport(), Qt.MouseButton.LeftButton, pos=at(win, 10, 10))
    assert win.session.selected == [2]
    qtbot.keyClick(win.canvas, Qt.Key.Key_3)
    assert win.session.ann.part(2).material == "rubber"
    assert "rubber" in win.tree.topLevelItem(0).child(0).text(0)


def test_brush_stroke_and_undo_through_the_window(win, qtbot):
    qtbot.mouseClick(win.canvas.viewport(), Qt.MouseButton.LeftButton, pos=at(win, 10, 10))
    qtbot.keyClick(win.canvas, Qt.Key.Key_B)
    assert win.canvas.paint_mode and win.tool_actions["brush"].isChecked()
    win.session.set_brush_radius(1)
    vp = win.canvas.viewport()
    qtbot.mousePress(vp, Qt.MouseButton.LeftButton, pos=at(win, 30, 35))
    send_move(vp, at(win, 50, 35))
    qtbot.mouseRelease(vp, Qt.MouseButton.LeftButton, pos=at(win, 50, 35))
    assert (win.session.ann.label_map[35, 30:51] == 2).all()
    assert win.canvas.overlay.argb_at(40, 35) != 0
    qtbot.keyClick(win.canvas, Qt.Key.Key_Z, Qt.KeyboardModifier.ControlModifier)
    assert not win.session.ann.label_map[35, 30:51].any() and win.canvas.overlay.argb_at(40, 35) == 0


def test_navigation_saves_and_switches_resolution(win, qtbot):
    qtbot.mouseClick(win.canvas.viewport(), Qt.MouseButton.LeftButton, pos=at(win, 10, 10))
    qtbot.keyClick(win.canvas, Qt.Key.Key_1)
    qtbot.keyClick(win.canvas, Qt.Key.Key_D)
    qtbot.keyClick(win.canvas, Qt.Key.Key_D)
    assert win.session.stem == "c" and win.canvas.overlay.boundingRect().width() == 50
    assert load_annotation(win.ann_dir, "a").part(2).material == "plastic"
    assert win.files.item(1).text().startswith("○")


def test_space_hides_masks_while_held(win, qtbot):
    qtbot.keyPress(win.canvas, Qt.Key.Key_Space)
    assert not win.canvas.overlay.isVisible()
    qtbot.keyRelease(win.canvas, Qt.Key.Key_Space)
    assert win.canvas.overlay.isVisible()


def test_mark_reviewed_refuses_incomplete_and_lists_problems(win, qtbot, monkeypatch):
    shown = []
    monkeypatch.setattr(QMessageBox, "information", lambda parent, title, text: shown.append(text))
    qtbot.keyClick(win.canvas, Qt.Key.Key_Return, Qt.KeyboardModifier.ControlModifier)
    assert win.session.ann.review_status == "unreviewed" and "no material" in shown[0]


def test_crash_handler_saves_and_logs(win, tmp_path, monkeypatch):
    shown = []
    monkeypatch.setattr(QMessageBox, "critical", lambda *a: shown.append(a[2]))
    monkeypatch.setattr(sys, "excepthook", sys.excepthook)
    log = tmp_path / "errors.log"
    hook = install_excepthook(win.session, log)
    s = win.session
    s.select_at(10, 10)
    s.set_material_index(0)                 # a finished, unsaved edit
    s.set_tool("brush")
    s.begin_stroke(30, 35)
    s.stroke_to(50, 35)                     # crash in the middle of a stroke
    try:
        raise RuntimeError("boom")
    except RuntimeError:
        hook(*sys.exc_info())
    back = load_annotation(win.ann_dir, "a")
    assert back.part(2).material == "plastic"   # the finished edit was saved
    assert back.part_at(35, 40) is None         # the half stroke was rolled back, not saved
    assert "boom" in log.read_text() and "saved" in shown[0]


def test_cli_reports_bad_paths(tmp_path, capsys):
    rc = main(["--images", str(tmp_path / "missing"), "--annotations", str(tmp_path / "ann"),
               "--labels", str(REPO / "labels.yaml")])
    assert rc == 1 and "annofix:" in capsys.readouterr().err


def test_gui_path_does_not_import_torch():
    code = ("import sys, annofix.__main__, annofix.session, annofix.ui.main_window; "
            "assert 'torch' not in sys.modules")
    subprocess.run([sys.executable, "-c", code], check=True,
                   env={**os.environ, "QT_QPA_PLATFORM": "offscreen"})
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_main_window.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'annofix.__main__'` (collection error).

- [ ] **Step 3: Write `annofix/ui/main_window.py`**

```python
"""The AnnoFix window: images | canvas | objects tree + labels. Every action goes through the
Session; this module only maps input to Session calls and redraws on its notifications."""
from __future__ import annotations

import sys
import time
import traceback
from pathlib import Path

from PyQt6.QtCore import QEvent, Qt, QTimer
from PyQt6.QtGui import QAction, QActionGroup, QKeySequence, QShortcut
from PyQt6.QtWidgets import (QApplication, QComboBox, QLabel, QLineEdit, QListWidget, QMainWindow,
                             QMessageBox, QSlider, QSplitter, QToolBar, QVBoxLayout, QWidget)

from ..session import COLOUR_MODES, Session
from .canvas import Canvas
from .colors import build_lut
from .label_panel import LabelPanel
from .tree_panel import ObjectTree

STATUS_ICONS = {"none": "○", "unreviewed": "•", "reviewed": "✓", "flagged": "⚑"}
AUTOSAVE_MS = 30_000


class MainWindow(QMainWindow):
    def __init__(self, session: Session):
        super().__init__()
        self.session = session
        self.setWindowTitle("AnnoFix")
        self.canvas = Canvas()
        self.tree = ObjectTree()
        self.labels_panel = LabelPanel(session.labels)
        self.files = QListWidget()
        self.files.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        for p in session.images:
            self.files.addItem(p.name)
        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.addWidget(self.tree, 3)
        right_layout.addWidget(self.labels_panel, 2)
        split = QSplitter()
        split.addWidget(self.files)
        split.addWidget(self.canvas)
        split.addWidget(right)
        split.setSizes([180, 900, 340])
        self.setCentralWidget(split)
        self.status = QLabel()
        self.statusBar().addWidget(self.status, 1)
        self._build_toolbar()
        self._build_shortcuts()
        self._connect()
        self._autosave = QTimer(self)
        self._autosave.timeout.connect(lambda: self._guard(self.session.save))
        self._autosave.start(AUTOSAVE_MS)
        QApplication.instance().installEventFilter(self)  # hold Space to hide the masks
        session.subscribe(self._on_change)
        if session.index < 0:
            session.open(0)
        else:
            self._on_change("image")
        self.canvas.setFocus()

    # ---- construction ---------------------------------------------------------------------

    def _build_toolbar(self) -> None:
        s = self.session
        tb = QToolBar("Tools")
        tb.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.addToolBar(tb)
        group = QActionGroup(self)
        self.tool_actions: dict[str, QAction] = {}
        for tool, text in (("select", "Select (V)"), ("brush", "Brush (B)"), ("eraser", "Eraser (E)")):
            action = QAction(text, self, checkable=True)
            action.triggered.connect(lambda checked=False, t=tool: s.set_tool(t))
            group.addAction(action)
            tb.addAction(action)
            self.tool_actions[tool] = action
        tb.addSeparator()
        tb.addWidget(QLabel(" Colour by "))
        self.mode_box = QComboBox()
        self.mode_box.addItems(COLOUR_MODES)
        self.mode_box.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.mode_box.currentTextChanged.connect(s.set_colour_mode)
        tb.addWidget(self.mode_box)
        tb.addWidget(QLabel(" Opacity "))
        self.opacity = QSlider(Qt.Orientation.Horizontal)
        self.opacity.setRange(0, 100)
        self.opacity.setValue(55)
        self.opacity.setFixedWidth(120)
        self.opacity.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.opacity.valueChanged.connect(self._apply_opacity)
        tb.addWidget(self.opacity)
        self.outline_action = QAction("Outline (O)", self, checkable=True)
        self.outline_action.toggled.connect(self._apply_outline)
        tb.addAction(self.outline_action)
        self.protect_action = QAction("Protect other parts", self, checkable=True)
        self.protect_action.toggled.connect(s.set_protect)
        tb.addAction(self.protect_action)

    def _build_shortcuts(self) -> None:
        s = self.session

        def key(sequence, fn):
            shortcut = QShortcut(QKeySequence(sequence), self)
            shortcut.activated.connect(lambda: self._guard(fn))

        for i in range(9):
            key(str(i + 1), lambda i=i: s.set_material_index(i))
        key("B", lambda: s.set_tool("brush"))
        key("E", lambda: s.set_tool("eraser"))
        key("V", lambda: s.set_tool("select"))
        key("Esc", self._escape)
        key("[", lambda: s.set_brush_radius(s.brush_radius - max(1, s.brush_radius // 5)))
        key("]", lambda: s.set_brush_radius(s.brush_radius + max(1, s.brush_radius // 5)))
        key("S", s.toggle_solo)
        key("O", self.outline_action.toggle)
        key("N", s.new_part)
        key("Shift+N", s.new_object)
        key("M", s.merge_selected_parts)
        key("Shift+M", s.merge_selected_objects)
        key("Del", s.delete_selected_parts)
        key("Shift+Del", s.delete_selected_object)
        key("Ctrl+Z", s.undo)
        key("Ctrl+Shift+Z", s.redo)
        key("Ctrl+Y", s.redo)
        key("A", lambda: s.step(-1))
        key("D", lambda: s.step(1))
        key("Ctrl+U", s.next_unreviewed)
        key("Ctrl+Return", self._mark_reviewed)
        key("Ctrl+Enter", self._mark_reviewed)
        key("F", s.toggle_flag)
        key("J", lambda: s.select_next_object(1))
        key("K", lambda: s.select_next_object(-1))
        key("Ctrl+S", s.save)

    def _connect(self) -> None:
        s, c, t, lp = self.session, self.canvas, self.tree, self.labels_panel
        c.clicked.connect(lambda x, y, add: s.select_at(x, y, add))
        c.stroke_started.connect(s.begin_stroke)
        c.stroke_moved.connect(s.stroke_to)
        c.stroke_ended.connect(s.end_stroke)
        t.part_clicked.connect(s.select_part)
        t.visibility_changed.connect(lambda kind, i, visible: s.set_hidden(kind, i, not visible))
        t.move_requested.connect(s.move_part_to)
        t.promote_requested.connect(s.promote_part)
        t.clean_requested.connect(s.clean_part)
        t.delete_part_requested.connect(lambda pid: s.delete_parts([pid]))
        t.delete_object_requested.connect(s.delete_object)
        lp.material_chosen.connect(s.set_material)
        lp.part_name_entered.connect(s.set_part_name)
        lp.class_entered.connect(s.set_object_class)
        self.files.itemClicked.connect(lambda item: self._guard(s.open, self.files.row(item)))

    # ---- reacting to the session ----------------------------------------------------------------

    def _on_change(self, kind: str) -> None:
        s = self.session
        if kind == "pixels":
            self.canvas.overlay.refresh()
            return
        if kind == "image":
            self.canvas.set_image(s.image, s.ann.label_map)
            self._apply_opacity()
            self.canvas.overlay.set_outline(self.outline_action.isChecked())
            self.files.blockSignals(True)
            self.files.setCurrentRow(s.index)
            self.files.blockSignals(False)
        if kind in ("image", "annotation", "selection", "view"):
            self._refresh_lut()
        if kind in ("image", "annotation", "selection"):
            self.tree.rebuild(s.ann, s.selected, s.hidden_parts, s.hidden_objects)
            if s.primary is None:
                self.labels_panel.show_part(None, None)
            else:
                self.labels_panel.show_part(s.ann.part(s.primary), s.ann.object_of(s.primary))
        if kind in ("image", "view"):
            self._sync_view()
        if kind in ("image", "annotation", "status"):
            self._refresh_files()
        self._refresh_status()

    def _refresh_lut(self) -> None:
        s = self.session
        self.canvas.overlay.set_lut(build_lut(
            s.ann, s.labels, s.colour_mode, hidden_parts=s.hidden_parts, hidden_objects=s.hidden_objects,
            solo_object=s.solo_object, selected=set(s.selected)))

    def _sync_view(self) -> None:
        s = self.session
        self.tool_actions[s.tool].setChecked(True)
        self.canvas.paint_mode = s.tool in ("brush", "eraser")
        self.canvas.brush_radius = s.brush_radius
        for widget, update in ((self.mode_box, lambda: self.mode_box.setCurrentText(s.colour_mode)),
                               (self.protect_action, lambda: self.protect_action.setChecked(s.protect_others))):
            widget.blockSignals(True)
            update()
            widget.blockSignals(False)

    def _refresh_files(self) -> None:
        s = self.session
        for i, p in enumerate(s.images):
            self.files.item(i).setText(f"{STATUS_ICONS[s.review_status(i)]} {p.name}")

    def _refresh_status(self) -> None:
        s = self.session
        if s.ann is None:
            return
        unsaved = " · unsaved" if s.ann.dirty else ""
        protect = " · protect" if s.protect_others else ""
        note = f"  |  {s.messages[-1]}" if s.messages else ""
        self.status.setText(f"{s.index + 1}/{len(s.images)}  {s.stem}  [{s.ann.review_status}{unsaved}]"
                            f"  tool: {s.tool} r={s.brush_radius}{protect}  "
                            f"objects: {len(s.ann.objects)}{note}")
        self.status.setToolTip("\n".join(s.messages))

    def _apply_opacity(self) -> None:
        if self.canvas.overlay is not None:
            self.canvas.overlay.setOpacity(self.opacity.value() / 100)

    def _apply_outline(self, on: bool) -> None:
        if self.canvas.overlay is not None:
            self.canvas.overlay.set_outline(on)

    # ---- actions --------------------------------------------------------------------------------

    def _guard(self, fn, *args):
        """Run a Session call; show expected failures (bad file, disk full) instead of crashing."""
        try:
            return fn(*args)
        except (ValueError, OSError) as e:
            QMessageBox.warning(self, "AnnoFix", str(e))
            return None

    def _escape(self) -> None:
        s = self.session
        if s.painting:
            s.cancel_stroke()
        else:
            s.set_tool("select")
            s.select_part(None)

    def _mark_reviewed(self) -> None:
        problems = self.session.mark_reviewed()
        if problems:
            more = f"\n… and {len(problems) - 20} more" if len(problems) > 20 else ""
            QMessageBox.information(self, "Not ready to mark reviewed", "\n".join(problems[:20]) + more)

    def eventFilter(self, obj, ev) -> bool:
        if ev.type() in (QEvent.Type.KeyPress, QEvent.Type.KeyRelease) and ev.key() == Qt.Key.Key_Space:
            focus = QApplication.focusWidget()
            typing = isinstance(focus, QLineEdit) or (isinstance(focus, QComboBox) and focus.isEditable())
            if not typing:
                if not ev.isAutoRepeat():
                    self.canvas.set_masks_visible(ev.type() == QEvent.Type.KeyRelease)
                return True
        return super().eventFilter(obj, ev)

    def closeEvent(self, ev) -> None:
        s = self.session
        try:
            s.end_stroke()
            s.save()
        except OSError as e:
            answer = QMessageBox.question(self, "AnnoFix", f"Could not save: {e}\n\nQuit anyway?")
            if answer != QMessageBox.StandardButton.Yes:
                ev.ignore()
                return
        ev.accept()


def install_excepthook(session: Session, log_path: Path):
    """PyQt6 aborts on an unhandled exception in a slot. Instead: log it, roll back a half
    stroke, save the finished edits, and tell the user."""

    def hook(exc_type, exc, tb):
        text = "".join(traceback.format_exception(exc_type, exc, tb))
        try:
            with open(log_path, "a") as f:
                f.write(f"--- {time.strftime('%Y-%m-%d %H:%M:%S')}\n{text}\n")
        except OSError:
            pass
        sys.__stderr__.write(text)
        outcome = "No unsaved work was lost."
        try:
            if session.painting:
                session.cancel_stroke()
            elif session.ann is not None and session.ann.in_transaction:
                session.ann.rollback()
            if session.save():
                outcome = "Your finished edits were saved."
        except Exception as save_error:  # noqa: BLE001 - never raise from the hook itself
            outcome = f"Saving failed too ({save_error}); your last edits may be lost."
        if QApplication.instance() is not None:
            QMessageBox.critical(None, "AnnoFix error",
                                 f"{exc_type.__name__}: {exc}\n\n{outcome}\nDetails: {log_path}")

    sys.excepthook = hook
    return hook
```

- [ ] **Step 4: Write `annofix/__main__.py`**

```python
"""python -m annofix --images DIR --annotations DIR [--labels labels.yaml] [--annotator NAME]"""
from __future__ import annotations

import annofix  # noqa: F401  (thread cap before numpy)

import argparse
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m annofix",
                                 description="Fix object/part/material segmentation annotations.")
    ap.add_argument("--images", required=True, type=Path, help="folder of colour images")
    ap.add_argument("--annotations", required=True, type=Path, help="annotations folder (created if missing)")
    ap.add_argument("--labels", type=Path, default=Path("labels.yaml"), help="materials and object classes")
    ap.add_argument("--annotator", help="your name, recorded in each annotation you save")
    args = ap.parse_args(argv)

    from .config import load_labels
    from .session import Session

    try:
        labels = load_labels(args.labels)
        session = Session(args.images, args.annotations, labels, annotator=args.annotator)
    except (OSError, ValueError) as e:
        print(f"annofix: {e}", file=sys.stderr)
        return 1

    from PyQt6.QtWidgets import QApplication

    from .ui.main_window import MainWindow, install_excepthook

    app = QApplication(sys.argv[:1])
    args.annotations.mkdir(parents=True, exist_ok=True)
    install_excepthook(session, args.annotations / "annofix-errors.log")
    window = MainWindow(session)
    window.resize(1500, 900)
    window.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/test_main_window.py -v`
Expected: PASS (9 tests). If the offscreen platform never reports the window active, the `waitUntil` in the fixture times out and every shortcut test fails. In that case, replace the `activateWindow`/`waitUntil` lines with `QApplication.setActiveWindow(w)` (deprecated but present in PyQt6 6.11) and record the ruling.
Run: `uv run pytest`
Expected: all PASS.

- [ ] **Step 6: Commit**

```bash
git add annofix/ui/main_window.py annofix/__main__.py tests/test_main_window.py
git commit -m "feat: AnnoFix main window with hotkeys, autosave, crash safety and CLI"
```

---

### Task 9: Real-frame check, live run and README

**Files:**
- Modify: `README.md`
- Uses (not committed): `$SCRATCH/gui_tour.py`, where `$SCRATCH` is the session scratchpad directory, and `data/gui_check/` (a copy of the bootstrap output; `data/` is git-ignored)

**Interfaces:**
- Consumes: everything above; `data/sample` + `data/sample_ann` from Plan 1.

- [ ] **Step 1: Make a disposable copy of the bootstrap annotations**

```bash
rm -rf data/gui_check && cp -r data/sample_ann data/gui_check && rm -rf data/gui_check/.amg_cache
ls data/gui_check | head -3
```
Expected: `cam0_000_color.json`, `cam0_000_color.parts.png`, … (no cache).

- [ ] **Step 2: Write the scripted tour (scratchpad, not committed): `$SCRATCH/gui_tour.py`**

```python
"""Drive the real window offscreen on real frames and save screenshots + timings."""
import os
import sys
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
import annofix  # noqa: E402,F401
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
from PyQt6.QtWidgets import QApplication  # noqa: E402

from annofix.config import load_labels  # noqa: E402
from annofix.session import Session  # noqa: E402
from annofix.ui.main_window import MainWindow  # noqa: E402

out = Path(sys.argv[1])
out.mkdir(parents=True, exist_ok=True)
app = QApplication([])
s = Session("data/sample", "data/gui_check", load_labels("labels.yaml"), annotator="tour")
w = MainWindow(s)
w.resize(1600, 900)
w.show()


def snap(name):
    app.processEvents()
    w.grab().save(str(out / f"{name}.png"))
    print(name, "| selected", s.selected, "| objects", len(s.ann.objects), "|", w.status.text())


s.open(3)                                    # cam0_120_color
snap("1_open")
s.select_at(627, 192)                        # robot arm upper link
s.select_at(500, 290, add=True)              # robot arm base
snap("2_arm_two_parts_selected")
s.merge_selected_objects()
s.set_material_index(1)                      # metal
s.set_object_class("robot_arm")
snap("3_arm_merged_metal")
s.set_colour_mode("object")
w.outline_action.setChecked(True)
snap("4_by_object_outline")
w.outline_action.setChecked(False)
s.set_colour_mode("material")
s.select_at(260, 560)                        # laptop keyboard
s.set_tool("brush")
s.set_brush_radius(12)
s.begin_stroke(200, 600)
times = []
for i in range(40):
    t = time.perf_counter()
    s.stroke_to(200 + 5 * i, 600 - 2 * i)
    w.canvas.overlay.refresh()
    times.append(time.perf_counter() - t)
s.end_stroke()
print(f"stroke segment + overlay refresh: median {np.median(times) * 1000:.1f} ms, max {max(times) * 1000:.1f} ms")
snap("5_brush_stroke")
s.undo()
snap("6_after_undo")
s.step(1)
s.open(7)                                    # a 640x480 cam1 frame
snap("7_cam1_640x480")
```

- [ ] **Step 3: Run the tour and inspect the screenshots**

Run: `uv run python $SCRATCH/gui_tour.py $SCRATCH/tour`
Expected:
- 7 lines of output; `objects` drops by one after `3_arm_merged_metal`.
- The timing line shows a median well under 16 ms.

Open each PNG and check:
- The masks line up with the image at 1280x720 and at 640x480.
- After the merge, the robot arm is one object with a lighter selection highlight and the metal colour.
- The tree shows `#… robot_arm`.
- The brush stroke appears in screenshot 5 and is gone in 6.
- The status bar shows the right image, tool and object count.

If any of these fails, fix it test-first (write a failing Session or window test that reproduces it) and record a ruling.

- [ ] **Step 4: Launch the real window for the user to try**

Run (background): `uv run python -m annofix --images data/sample --annotations data/gui_check --annotator <name>`
Expected: the window opens on the user's Wayland desktop. Ask the user to try:
- click a part, press `2`
- `B`, drag to paint; `Ctrl+Z`
- Ctrl-click a second object, `Shift+M`
- `D` to the next image, `Space`
- `Ctrl+Return`

Record their feedback in the ledger. Fixing UX issues they raise is a new request, not part of this task.

- [ ] **Step 5: Add the GUI section to `README.md`** (insert after the "Bootstrap starting annotations" section)

````markdown
## Fixing annotations (GUI)

```bash
uv run python -m annofix --images IMAGES_DIR --annotations ANNOTATIONS_DIR [--labels labels.yaml] [--annotator NAME]
```
Left: images (○ no annotation yet · • unreviewed · ✓ reviewed · ⚑ flagged). Centre: the image with
its masks. Right: objects → parts (tick boxes show/hide; right-click to move, split off, clean or
delete) and the label panel. Work autosaves every 30 s, when you change image, and on close.

| To do this | Press / click |
|---|---|
| Select a part · add/remove one | click · Ctrl+click (canvas or tree) |
| Set material | `1`–`9` or a material button |
| Part name · object class | suggestion buttons or type + Enter |
| Brush (paint the selected part) · eraser · select | `B` · `E` · `V` |
| Brush size | `[` `]` |
| Protect other parts while painting | toolbar toggle |
| Cancel a stroke | `Esc` |
| New part in this object · new object | `N` · `Shift+N` |
| Merge selected parts · merge their objects | `M` · `Shift+M` |
| Delete part(s) · delete object | `Del` · `Shift+Del` |
| Undo · redo | `Ctrl+Z` · `Ctrl+Shift+Z` |
| Hide masks while held · solo object · outlines | `Space` · `S` · `O` |
| Next/previous object | `J` · `K` |
| Previous/next image · next unreviewed | `A` · `D` · `Ctrl+U` |
| Mark reviewed (checks every label) · flag | `Ctrl+Enter` · `F` |
| Zoom · pan | mouse wheel · middle-drag |

Errors are logged to `ANNOTATIONS_DIR/annofix-errors.log`; on a crash, finished edits are saved
and a half-finished stroke is rolled back.
````

- [ ] **Step 6: Full test run and commit**

Run: `uv run pytest`
Expected: all PASS (SAM tests skipped).

```bash
git add README.md
git commit -m "docs: GUI usage and shortcuts"
```

---

## What Plan 3 will build on

- `Session` gains SAM actions: include/exclude clicks, Tab to cycle candidates, Enter to replace and Shift+Enter to add. They call `Sam2Backend.predict` from a QThread worker and accept results through `Annotation.replace_part_mask` / `assign` inside a transaction. `N`/`Shift+N` switch from "new empty part + brush" to "new part from a SAM click".
- `predict(prev_logits=...)` must receive a single `(1, 256, 256)` slice of the previous result (Plan 1 review note).
- `Canvas` needs right-click (exclude) and Shift+drag (box) in a new `sam` tool, and `focusNextPrevChild` overridden so Tab reaches it.
