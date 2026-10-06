# AnnoFix Plan 3 of 3 — SAM Click-to-Fix Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix masks with SAM 2.1 in the editor:
- **R** on a selected part re-segments it from a point SAM picks deep inside it.
- **Left click** adds an include point, **right click** an exclude point (e.g. on the region a mask bled into), and **Shift+drag** draws a box.
- **Tab** cycles SAM's candidate masks. **Enter** replaces the part, **Shift+Enter** adds to it, **Esc** cancels.
- **N / Shift+N** create a new part or object from clicks.
- A colour preview shows exactly which pixels would be kept, added, taken from another part, or removed.

**Architecture:**
- A pure, Qt-free `SamEdit` (`annofix/sam_edit.py`) holds one fix in progress: prompts, the request it implies, the candidates received, and the chosen candidate. Requests and results are tagged `(image token, seq)`, so stale results are dropped.
- `Session` gains the SAM actions and stays Qt-free. Its tests drive it synchronously with a deterministic `FakeBackend`.
- In the window, a `SamController` runs SAM in a `SamWorker` on its own `QThread`. The model loads lazily (first SAM use), the image is embedded on demand, and only the newest request is computed. Every worker exception becomes a signal.
- The canvas gains a `sam` mode with prompt markers and a preview overlay.

**Tech Stack:** Python 3.12, uv, numpy, opencv-python-headless, PyQt6 6.11, SAM 2.1 (`sam-2` @ 2b90b9f) on PyTorch 2.14 + CUDA, pytest + pytest-qt.

**Spec:** `docs/superpowers/specs/2026-10-03-annofix-design.md` ("SAM fix" row of the UI table, "SAM backend" section, milestones M5–M6). Earlier plans: `docs/superpowers/plans/2026-10-03-annofix-plan-1-foundation.md`, `docs/superpowers/plans/2026-10-06-annofix-plan-2-editor.md`. Measurements: `docs/notes/2026-10-03-m0-measurements.md`.

## Global Constraints

- Everything from Plans 1–2 still holds:
  - Python `>=3.12,<3.13`, uv; `opencv-python-headless` only.
  - Thread cap of 10.
  - Label map edited in place; ids never reused.
  - The GUI holds ids only.
  - A stroke is one undo step; no saves during an open edit.
  - Hotkeys are `QShortcut`s only.
  - Qt tests run offscreen.
  - No `Co-Authored-By` trailer in commits.
- `torch`/`sam2` are imported **only inside the SAM worker thread**, through the backend factory. Importing `annofix.__main__`, `annofix.session` or `annofix.ui.main_window` must not import torch.
- The GUI thread never waits for SAM. The `SamWorker` never touches the `Session` or any widget: it only receives and emits Qt signals. Every exception inside it becomes a `failed(message, fatal)` signal.
- **The model loads on first SAM use, not at startup.** An image is embedded only when SAM needs it: the SAM tool is active, or a request is sent.
- SAM is asked for 3 candidates only for a single include point with no box. Every other prompt asks for 1, and refinements feed back the chosen candidate's logits as `prev_logits` of shape `(1, 256, 256)`.
- A request needs at least one include point or a box.
- Accepting a SAM mask follows the same pixel rules as the brush: "Protect other parts", and hidden or non-solo parts are never changed. One accept is one undo step labelled `"sam"`.
- Without SAM (`--no-sam`, model failed to load, or no factory): `R` reports that SAM is unavailable, and `N`/`Shift+N` keep Plan 2's behaviour (empty part + brush).

## Review Focus

1. **Clicking faster than SAM answers, or changing image while SAM is busy, must never apply a result to the wrong image or an older set of prompts.** → Task 2 `test_stale_results_are_dropped`; Task 3 `test_stale_result_after_a_newer_click_is_ignored`, `test_result_for_an_image_you_left_is_ignored`; Task 4 `test_result_for_an_image_you_left_is_dropped`, `test_only_the_newest_request_is_computed`.
2. **The SAM model failing to load must leave the editor fully usable**, for example with no network for the checkpoint download, or CUDA out of memory: a clear message, `N` falls back to the brush, and no hang. → Task 4 `test_model_that_fails_to_load_disables_sam_but_not_the_editor`.
3. **Accepting a SAM mask must never change hidden, non-solo or protected parts.** → Task 3 `test_accept_never_changes_hidden_parts`, `test_accept_respects_protect`.
4. **Closing the window while SAM is mid-prediction must neither hang nor crash.** → Task 4 `test_shutdown_while_predicting_does_not_hang`; Task 6 (closeEvent shuts the controller down).
5. **Undo after a SAM accept must restore exactly. An undo that removes the part being fixed must cancel that fix.** → Task 3 `test_accept_is_one_undo_step`, `test_undo_that_removes_the_target_cancels_the_edit`.

## File Structure

| File | Responsibility |
|---|---|
| `annofix/sam_backend.py` (modify) | `SegmentationBackend` protocol, `check_prev_logits`, `FakeBackend`; `Sam2Backend.predict` validates `prev_logits` |
| `annofix/sam_edit.py` (new) | `SamRequest`, `SamEdit`, `interior_point`, `preview_map` + `PREVIEW_*` codes |
| `annofix/session.py` (modify) | tool `"sam"`, image token, SAM actions, `add_message`, N/Shift+N routing |
| `annofix/ui/sam_worker.py` (new) | `SamWorker` (worker thread) and `SamController` (GUI-thread glue) |
| `annofix/ui/colors.py` (modify) | `preview_lut()` |
| `annofix/ui/canvas.py` (rewrite) | `mode` (`select`/`paint`/`sam`), SAM click/box signals, prompt markers, preview overlay, Tab stays on canvas |
| `annofix/ui/main_window.py` (modify) | `sam_factory`, R/Tab/Enter/Shift+Enter/Esc, SAM status, shutdown on close |
| `annofix/__main__.py` (modify) | `--sam-model`, `--device`, `--checkpoint`, `--no-sam` |
| `README.md` (modify) | SAM usage |
| `docs/notes/2026-10-03-m0-measurements.md` (modify) | live SAM latency and the M6 (zoomed-crop) decision |
| tests: `tests/test_sam_helpers.py`, `tests/test_canvas.py`, `tests/test_main_window.py`, `tests/test_sam_backend.py` (modify); `tests/test_sam_edit.py`, `tests/test_session_sam.py`, `tests/test_sam_worker.py`, `tests/test_main_window_sam.py` (new) | |

---

### Task 1: Backend protocol, prompt check and a deterministic FakeBackend

**Files:**
- Modify: `annofix/sam_backend.py`, `tests/test_sam_helpers.py`

**Interfaces:**
- Produces (used by every later task):
  - `SegmentationBackend` (`typing.Protocol`): attributes `device: str`, `model_name: str`; `set_image(image)`; `predict(points=None, labels=None, box=None, prev_logits=None, multimask=True) -> (bool masks (C,H,W), float32 scores (C,), float32 logits (C,256,256))`
  - `check_prev_logits(prev_logits) -> None` (raises `ValueError("... one candidate ...")` unless `None` or shape `(1, 256, 256)`)
  - `FakeBackend(delay=0.0)`: `calls: list[tuple]`, where `("set_image", (H, W))` and `("predict", points, labels, box, has_prev_logits, multimask)` are appended **before** any delay. `RADII = (4, 8, 16)`, `SCORES = (0.5, 0.9, 0.7)`. One include point with `multimask=True` → 3 discs; otherwise → the radius-8 disc; a box → its inclusive rectangle; each exclude point cuts out a radius-6 disc. Logits are filled with the candidate's radius.

- [ ] **Step 1: Write the failing tests** — in `tests/test_sam_helpers.py` replace the import line with the first block below, then append the rest

```python
import numpy as np
import pytest

from annofix.sam_backend import FakeBackend, check_prev_logits, device_kind
```

```python


def blank():
    b = FakeBackend()
    b.set_image(np.zeros((40, 60, 3), np.uint8))
    return b


def test_fake_backend_gives_three_nested_candidates_for_one_click():
    masks, scores, logits = blank().predict(points=[[30, 20]], labels=[1], multimask=True)
    assert masks.shape == (3, 40, 60) and masks.dtype == bool and logits.shape == (3, 256, 256)
    areas = masks.sum((1, 2))
    assert areas[0] < areas[1] < areas[2] and masks[:, 20, 30].all()
    assert scores.tolist() == pytest.approx([0.5, 0.9, 0.7]) and logits[2, 0, 0] == 16


def test_fake_backend_respects_exclude_points_and_boxes():
    b = blank()
    masks, _, _ = b.predict(points=[[30, 20], [37, 20]], labels=[1, 0], multimask=False)
    assert masks.shape == (1, 40, 60) and masks[0, 20, 30] and not masks[0, 20, 36]
    masks, _, _ = b.predict(box=[10, 5, 20, 15], multimask=False)
    assert masks[0].sum() == 11 * 11
    assert [c[0] for c in b.calls] == ["set_image", "predict", "predict"]


def test_prev_logits_must_be_a_single_candidate():
    check_prev_logits(None)
    check_prev_logits(np.zeros((1, 256, 256), np.float32))
    with pytest.raises(ValueError, match="one candidate"):
        check_prev_logits(np.zeros((3, 256, 256), np.float32))
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_sam_helpers.py -v`
Expected: FAIL — `ImportError: cannot import name 'FakeBackend'`.

- [ ] **Step 3: Implement in `annofix/sam_backend.py`**

1. Imports: add `import time` after `import contextlib`, and `from typing import Protocol` after `from pathlib import Path`.
2. After `device_kind` add:

```python
class SegmentationBackend(Protocol):
    """What the editor needs from a promptable segmenter (SAM 2.1, or FakeBackend in tests)."""

    device: str
    model_name: str

    def set_image(self, image: np.ndarray) -> None: ...

    def predict(self, points=None, labels=None, box=None, prev_logits=None,
                multimask: bool = True) -> tuple[np.ndarray, np.ndarray, np.ndarray]: ...


def check_prev_logits(prev_logits) -> None:
    """SAM's mask_input is ONE candidate's low-res logits, shape (1, 256, 256): pass
    logits[i:i+1] from the previous call, never the whole (C, 256, 256) stack (sam2 would read
    it as a single 3-channel mask and fail)."""
    if prev_logits is not None and np.asarray(prev_logits).shape != (1, 256, 256):
        raise ValueError(f"prev_logits must be one candidate, shape (1, 256, 256); "
                         f"got {np.asarray(prev_logits).shape}")
```

3. In `Sam2Backend.predict`, replace

```python
    def predict(self, points=None, labels=None, box=None, prev_logits=None, multimask=True):
        with self._context():
```

with

```python
    def predict(self, points=None, labels=None, box=None, prev_logits=None, multimask=True):
        """points: (N, 2) x, y pixels; labels: 1 include / 0 exclude; box: x0, y0, x1, y1;
        prev_logits: (1, 256, 256), see check_prev_logits. Returns bool masks (C, H, W),
        scores (C,), low-res logits (C, 256, 256); C is 3 with multimask, else 1."""
        check_prev_logits(prev_logits)
        with self._context():
```

4. Append at the end of the file:

```python


class FakeBackend:
    """Deterministic, torch-free stand-in for tests (and for UI work without a GPU).

    One include point with multimask gives three nested discs (radius 4, 8, 16; scores 0.5,
    0.9, 0.7); any other prompt gives the radius-8 disc. Each exclude point cuts out a radius-6
    disc; a box gives its inclusive rectangle. Logits are filled with the candidate's radius so
    tests can tell which one was fed back. `delay` seconds make predict slow (threading tests);
    calls are recorded before the delay."""

    device = "fake"
    model_name = "fake"
    RADII = (4, 8, 16)
    SCORES = (0.5, 0.9, 0.7)

    def __init__(self, delay: float = 0.0):
        self.delay = delay
        self.calls: list[tuple] = []
        self._shape: tuple[int, int] | None = None

    def set_image(self, image: np.ndarray) -> None:
        self._shape = tuple(image.shape[:2])
        self.calls.append(("set_image", self._shape))

    def predict(self, points=None, labels=None, box=None, prev_logits=None, multimask=True):
        check_prev_logits(prev_logits)
        self.calls.append(("predict", None if points is None else np.asarray(points).tolist(),
                           None if labels is None else np.asarray(labels).tolist(),
                           None if box is None else np.asarray(box).tolist(),
                           prev_logits is not None, multimask))
        if self.delay:
            time.sleep(self.delay)
        if self._shape is None:
            raise RuntimeError("predict() before set_image()")
        h, w = self._shape
        yy, xx = np.mgrid[:h, :w]

        def disc(x, y, r):
            return (xx - x) ** 2 + (yy - y) ** 2 <= r * r

        pts = [] if points is None else [tuple(p) for p in np.asarray(points)]
        lbl = [] if labels is None else [int(v) for v in np.asarray(labels)]
        include = [p for p, v in zip(pts, lbl) if v == 1]
        exclude = [p for p, v in zip(pts, lbl) if v == 0]
        if box is not None:
            x0, y0, x1, y1 = (int(v) for v in box)
            radii = [0]
            masks = [(xx >= x0) & (xx <= x1) & (yy >= y0) & (yy <= y1)]
        else:
            if not include:
                raise ValueError("FakeBackend needs an include point or a box")
            px, py = include[0]
            radii = list(self.RADII) if multimask else [self.RADII[1]]
            masks = [disc(px, py, r) for r in radii]
        for ex, ey in exclude:
            masks = [m & ~disc(ex, ey, 6) for m in masks]
        scores = [self.SCORES[self.RADII.index(r)] if r in self.RADII else 0.9 for r in radii]
        logits = np.stack([np.full((256, 256), r, np.float32) for r in radii])
        return np.stack(masks), np.array(scores, np.float32), logits
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_sam_helpers.py -v`
Expected: PASS (4 tests). Run: `uv run pytest` — all PASS.

- [ ] **Step 5: Commit**

```bash
git add annofix/sam_backend.py tests/test_sam_helpers.py
git commit -m "feat: segmentation backend protocol, prev_logits check and deterministic FakeBackend"
```

---

### Task 2: `SamEdit` — prompts, requests, candidates and the accept preview (pure)

**Files:**
- Create: `annofix/sam_edit.py`, `tests/test_sam_edit.py`

**Interfaces:**
- Consumes: `FakeBackend` (tests only).
- Produces (used by Tasks 3, 5, 6):
  - `SamRequest` (frozen dataclass): `token: int`, `seq: int`, `points: float32 (N,2) | None`, `labels: int32 (N,) | None`, `box: float32 (4,) | None`, `prev_logits: (1,256,256) | None`, `multimask: bool`
  - `SamEdit(token, target=None, object_id=None)`:
    - attributes `token`, `target` (part being fixed, or None for a new part), `object_id` (object for a new part; None with `target` None = a new object), `points: list[(x, y)]`, `labels: list[int]`, `box: (x0, y0, x1, y1) | None`, `seq`, `received_seq`, `candidates: list[bool HxW]` (ascending area), `scores`, `index`
    - properties `mask`, `score`
    - methods `add_point(x, y, positive) -> SamRequest | None`, `set_box(x0, y0, x1, y1) -> SamRequest | None`, `receive(token, seq, masks, scores, logits) -> bool`, `cycle(step=1)`
  - `interior_point(mask) -> (x, y) | None`
  - `preview_map(label_map, target, mask, *, union=False, protect=False, locked=None) -> uint8 HxW` with codes `PREVIEW_KEPT=1`, `PREVIEW_ADDED=2`, `PREVIEW_TAKEN=3`, `PREVIEW_REMOVED=4` (0 = unchanged)

- [ ] **Step 1: Write the failing test `tests/test_sam_edit.py`**

```python
import numpy as np

from annofix.sam_backend import FakeBackend
from annofix.sam_edit import (PREVIEW_ADDED, PREVIEW_KEPT, PREVIEW_REMOVED, PREVIEW_TAKEN, SamEdit,
                              interior_point, preview_map)


def backend():
    b = FakeBackend()
    b.set_image(np.zeros((40, 60, 3), np.uint8))
    return b


def answer(edit, req, b):
    masks, scores, logits = b.predict(req.points, req.labels, req.box, req.prev_logits, req.multimask)
    return edit.receive(req.token, req.seq, masks, scores, logits)


def test_first_include_point_asks_for_three_candidates():
    req = SamEdit(token=7, target=2).add_point(30, 20, True)
    assert (req.token, req.seq, req.multimask, req.prev_logits) == (7, 1, True, None)
    assert req.points.tolist() == [[30.0, 20.0]] and req.labels.tolist() == [1] and req.box is None


def test_exclude_only_waits_for_an_include_point():
    e = SamEdit(1)
    assert e.add_point(5, 5, False) is None and e.seq == 0
    req = e.add_point(10, 10, True)
    assert req.labels.tolist() == [0, 1] and not req.multimask


def test_candidates_sorted_by_area_best_score_chosen_and_cycled():
    b, e = backend(), SamEdit(1, 2)
    assert answer(e, e.add_point(30, 20, True), b)
    areas = [int(m.sum()) for m in e.candidates]
    assert areas == sorted(areas) and e.score == 0.9 and e.index == 1 and e.received_seq == 1
    e.cycle()
    assert e.index == 2
    e.cycle()
    assert e.index == 0
    e.cycle(-1)
    assert e.index == 2


def test_refinement_feeds_back_the_chosen_candidates_logits():
    b, e = backend(), SamEdit(1, 2)
    answer(e, e.add_point(30, 20, True), b)
    e.cycle()  # the radius-16 candidate
    req = e.add_point(37, 20, False)
    assert not req.multimask and req.prev_logits.shape == (1, 256, 256) and req.prev_logits[0, 0, 0] == 16
    assert answer(e, req, b) and len(e.candidates) == 1 and not e.mask[20, 36]


def test_stale_results_are_dropped():
    b, e = backend(), SamEdit(1, 2)
    r1 = e.add_point(30, 20, True)
    r2 = e.add_point(37, 20, False)
    out1 = b.predict(r1.points, r1.labels, r1.box, r1.prev_logits, r1.multimask)
    out2 = b.predict(r2.points, r2.labels, r2.box, r2.prev_logits, r2.multimask)
    assert not e.receive(r1.token, r1.seq, *out1)  # an older prompt
    assert not e.receive(99, r2.seq, *out2)        # another image
    assert e.receive(r2.token, r2.seq, *out2) and e.mask is not None


def test_box_prompt_is_normalised():
    req = SamEdit(1).set_box(20, 15, 10, 5)
    assert req.box.tolist() == [10, 5, 20, 15] and not req.multimask and req.points is None


def test_interior_point_is_the_deepest_pixel():
    m = np.zeros((20, 30), bool)
    m[2:18, 2:12] = True    # a fat block...
    m[8:12, 12:28] = True   # ...with a thin arm
    x, y = interior_point(m)
    assert m[y, x] and 4 <= x <= 9 and 6 <= y <= 13
    assert interior_point(np.zeros((5, 5), bool)) is None


def test_preview_shows_kept_added_taken_removed():
    lm = np.zeros((6, 6), np.uint16)
    lm[0:3, 0:3] = 2
    lm[3:6, 3:6] = 3
    mask = np.zeros((6, 6), bool)
    mask[1:5, 1:5] = True
    p = preview_map(lm, 2, mask)
    assert (p[1, 1], p[0, 0], p[1, 4], p[4, 4]) == (PREVIEW_KEPT, PREVIEW_REMOVED, PREVIEW_ADDED, PREVIEW_TAKEN)
    assert preview_map(lm, 2, mask, protect=True)[4, 4] == 0
    assert preview_map(lm, 2, mask, union=True)[0, 0] == 0
    locked = np.zeros(65536, bool)
    locked[3] = True
    assert preview_map(lm, 2, mask, locked=locked)[4, 4] == 0
    assert preview_map(lm, None, mask)[1, 1] == PREVIEW_TAKEN  # a new part would take from part 2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_sam_edit.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'annofix.sam_edit'`.

- [ ] **Step 3: Write `annofix/sam_edit.py`**

```python
"""One SAM fix in progress, without Qt or torch: the prompts (include/exclude points, a box),
the request they imply, the candidate masks SAM returned, which one is chosen, and a preview
of what accepting it would change. Requests and results carry (image token, seq), so a result
for an older prompt or another image is dropped."""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

PREVIEW_KEPT, PREVIEW_ADDED, PREVIEW_TAKEN, PREVIEW_REMOVED = 1, 2, 3, 4


@dataclass(frozen=True)
class SamRequest:
    token: int
    seq: int
    points: np.ndarray | None       # (N, 2) float32 x, y
    labels: np.ndarray | None       # (N,) int32, 1 include / 0 exclude
    box: np.ndarray | None          # (4,) float32 x0, y0, x1, y1
    prev_logits: np.ndarray | None  # (1, 256, 256): the chosen candidate of the previous answer
    multimask: bool


class SamEdit:
    def __init__(self, token: int, target: int | None = None, object_id: int | None = None):
        """`target`: the part being fixed, or None for a new part, which goes into `object_id`,
        or into a new object when that is None too."""
        self.token = token
        self.target = target
        self.object_id = object_id
        self.points: list[tuple[int, int]] = []
        self.labels: list[int] = []
        self.box: tuple[int, int, int, int] | None = None
        self.seq = 0
        self.received_seq = 0
        self.candidates: list[np.ndarray] = []
        self.scores: list[float] = []
        self.index = 0
        self._logits: np.ndarray | None = None  # (C, 256, 256), aligned with candidates

    @property
    def mask(self) -> np.ndarray | None:
        return self.candidates[self.index] if self.candidates else None

    @property
    def score(self) -> float | None:
        return self.scores[self.index] if self.scores else None

    def add_point(self, x: int, y: int, positive: bool) -> SamRequest | None:
        self.points.append((int(x), int(y)))
        self.labels.append(1 if positive else 0)
        return self._request()

    def set_box(self, x0: int, y0: int, x1: int, y1: int) -> SamRequest | None:
        self.box = (min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1))
        return self._request()

    def _request(self) -> SamRequest | None:
        if 1 not in self.labels and self.box is None:
            return None  # nothing to segment yet
        self.seq += 1
        multimask = self.box is None and len(self.points) == 1  # only a lone click is ambiguous
        prev = None if multimask or self._logits is None else self._logits[self.index:self.index + 1]
        return SamRequest(
            self.token, self.seq,
            np.array(self.points, np.float32) if self.points else None,
            np.array(self.labels, np.int32) if self.points else None,
            None if self.box is None else np.array(self.box, np.float32),
            prev, multimask)

    def receive(self, token: int, seq: int, masks, scores, logits) -> bool:
        """Take SAM's answer if it is for this image and the newest prompt; False otherwise."""
        if token != self.token or seq != self.seq or len(masks) == 0:
            return False
        order = np.argsort(np.asarray(masks).reshape(len(masks), -1).sum(1), kind="stable")
        self.candidates = [np.asarray(masks[i], bool) for i in order]
        self.scores = [float(scores[i]) for i in order]
        self._logits = np.asarray(logits)[order]
        self.index = int(np.argmax(self.scores))
        self.received_seq = seq
        return True

    def cycle(self, step: int = 1) -> None:
        if self.candidates:
            self.index = (self.index + step) % len(self.candidates)


def interior_point(mask: np.ndarray) -> tuple[int, int] | None:
    """The pixel deepest inside `mask` (distance-transform maximum) as (x, y). A click there is
    unambiguous even when the mask's outline, and so its bbox, is wrong."""
    if not mask.any():
        return None
    rows = np.flatnonzero(mask.any(1))
    cols = np.flatnonzero(mask.any(0))
    y0, x0 = int(rows[0]), int(cols[0])
    crop = np.pad(mask[y0:int(rows[-1]) + 1, x0:int(cols[-1]) + 1], 1).astype(np.uint8)
    dist = cv2.distanceTransform(crop, cv2.DIST_L2, 5)
    y, x = np.unravel_index(int(np.argmax(dist)), dist.shape)
    return x0 + int(x) - 1, y0 + int(y) - 1


def preview_map(label_map: np.ndarray, target: int | None, mask: np.ndarray, *, union: bool = False,
                protect: bool = False, locked: np.ndarray | None = None) -> np.ndarray:
    """uint8 per pixel: what accepting `mask` for `target` (None = a new part) would do:
    KEPT (already the target's), ADDED (from background), TAKEN (from another part), REMOVED
    (the target loses it; replace only). Locked (hidden/non-solo) parts are never changed."""
    effective = mask if locked is None else mask & ~locked[label_map]
    own = label_map == target if target is not None else np.zeros(label_map.shape, bool)
    background = label_map == 0
    out = np.zeros(label_map.shape, np.uint8)
    out[effective & own] = PREVIEW_KEPT
    out[effective & background] = PREVIEW_ADDED
    if not protect:
        out[effective & ~background & ~own] = PREVIEW_TAKEN
    if target is not None and not union:
        out[own & ~mask] = PREVIEW_REMOVED
    return out
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_sam_edit.py -v`
Expected: PASS (8 tests).

- [ ] **Step 5: Commit**

```bash
git add annofix/sam_edit.py tests/test_sam_edit.py
git commit -m "feat: SamEdit prompt/candidate state with stale-result guard and accept preview"
```

---

### Task 3: Session — SAM actions (still no Qt)

**Files:**
- Modify: `annofix/session.py`
- Create: `tests/test_session_sam.py`

**Interfaces:**
- Consumes: Task 2 `SamEdit`, `SamRequest`, `interior_point`, `preview_map`; Plan 2 `Session` internals (`_changed`, `_notify`, `end_stroke`, `_locked_lut`, `primary`).
- Produces (used by Tasks 4 and 6):
  - `TOOLS` gains `"sam"`
  - attributes `image_token: int` (incremented by every successful `open`), `sam: SamEdit | None`, `sam_enabled: bool` (default False; the window's controller sets it)
  - `add_message(text)` (appends + notifies `"status"`)
  - `start_sam_fix() -> bool`, `start_sam_new(new_object=False) -> bool`, `sam_click(x, y, positive)`, `sam_box(x0, y0, x1, y1)`, `take_sam_request() -> SamRequest | None`, `sam_receive(token, seq, masks, scores, logits) -> bool`, `sam_cycle(step=1)`, `sam_preview() -> uint8 HxW | None`, `sam_accept(union=False) -> bool`, `sam_cancel() -> bool`
  - New notification kinds: `"sam"` (the edit or its candidates changed), `"sam_request"` (a request is waiting in `take_sam_request()`).
  - `new_part()` / `new_object()` start a SAM edit and return `None` when `sam_enabled`; otherwise unchanged (return the new part id).

- [ ] **Step 1: Write the failing test `tests/test_session_sam.py`**

```python
"""Session SAM actions, driven synchronously by FakeBackend (what the window's controller does
asynchronously)."""
import numpy as np
import pytest
from helpers import LABELS, make_dataset

from annofix.sam_backend import FakeBackend
from annofix.session import Session


@pytest.fixture
def dataset(tmp_path):
    return make_dataset(tmp_path)


def make_session(dataset, drive=True):
    session = Session(*dataset, LABELS)
    session.sam_enabled = True
    backend = FakeBackend()

    def controller(kind):
        if kind == "image":
            backend.set_image(session.image)
        elif kind == "sam_request" and drive:
            req = session.take_sam_request()
            out = backend.predict(req.points, req.labels, req.box, req.prev_logits, req.multimask)
            session.sam_receive(req.token, req.seq, *out)

    session.subscribe(controller)
    session.open(0)
    session.backend = backend
    return session


@pytest.fixture
def s(dataset):
    return make_session(dataset)


def test_r_on_a_selected_part_seeds_inside_it_and_shows_candidates(s):
    s.select_at(10, 10)
    s.set_tool("sam")
    e = s.sam
    assert s.tool == "sam" and e.target == 2 and e.labels == [1]
    x, y = e.points[0]
    assert s.ann.part_at(y, x) == 2
    assert len(e.candidates) == 3 and e.score == 0.9 and s.sam_preview() is not None


def test_exclude_click_refines_and_enter_replaces_the_part(s):
    s.select_at(10, 10)
    s.set_tool("sam")
    x, y = s.sam.points[0]
    s.sam_click(x + 7, y, False)
    assert s.sam.labels == [1, 0] and len(s.sam.candidates) == 1
    expected = s.sam.mask.copy()
    assert s.sam_accept()
    assert s.sam is None and s.selected == [2] and s.tool == "sam"
    assert np.array_equal(s.ann.part_mask(2), expected)


def test_accept_is_one_undo_step(s):
    before = s.ann.label_map.copy()
    s.select_at(10, 10)
    s.set_tool("sam")
    s.sam_accept()
    assert s.undo() == "sam"
    assert np.array_equal(s.ann.label_map, before)


def test_shift_enter_adds_instead_of_replacing(s):
    s.select_at(10, 10)
    s.set_tool("sam")
    before = s.ann.part_area(2)
    s.sam_accept(union=True)
    assert s.ann.part_area(2) > before and s.ann.part_at(5, 5) == 2  # the old corner is kept


def test_tab_cycles_candidates_and_the_preview_follows(s):
    s.select_at(10, 10)
    s.set_tool("sam")
    first = s.sam_preview().copy()
    s.sam_cycle()
    assert s.sam.index == 2 and not np.array_equal(s.sam_preview(), first)


def test_click_on_a_part_in_the_sam_tool_starts_fixing_it(s):
    s.set_tool("sam")
    assert s.sam is None
    s.sam_click(35, 10, True)  # the handle, part 3
    assert s.selected == [3] and s.sam.target == 3 and s.sam.points == [(35, 10)]
    s.sam_click(36, 11, False)  # right-click with no edit open would be ignored; here it refines
    assert s.sam.labels == [1, 0]


def test_right_click_without_an_edit_is_ignored(s):
    s.set_tool("sam")
    s.sam_click(35, 10, False)
    assert s.sam is None


def test_click_on_background_makes_a_new_object(s):
    s.set_tool("sam")
    s.sam_click(50, 35, True)
    assert s.sam.target is None and s.sam.object_id is None
    n = len(s.ann.objects)
    assert s.sam_accept()
    assert len(s.ann.objects) == n + 1 and s.ann.part_at(35, 50) == s.selected[0]


def test_box_prompt(s):
    s.set_tool("sam")
    s.sam_box(45, 30, 55, 38)
    assert s.sam.box == (45, 30, 55, 38) and s.sam.mask.sum() == 11 * 9


def test_n_with_sam_adds_a_part_to_the_selected_object(s):
    s.select_at(10, 10)
    assert s.new_part() is None
    assert s.tool == "sam" and s.sam.target is None and s.sam.object_id == 1 and s.sam.points == []
    s.sam_click(50, 35, True)
    s.sam_accept()
    assert s.ann.object_of(s.selected[0]).id == 1


def test_shift_n_with_sam_makes_a_new_object(s):
    s.select_at(10, 10)
    assert s.new_object() is None and s.sam.object_id is None and s.sam.target is None


def test_n_without_sam_still_makes_an_empty_part_for_the_brush(s):
    s.sam_enabled = False
    s.select_at(10, 10)
    pid = s.new_part()
    assert s.tool == "brush" and s.ann.object_of(pid).id == 1


def test_sam_tool_unavailable_says_so(s):
    s.sam_enabled = False
    s.select_at(10, 10)
    s.set_tool("sam")
    assert s.tool == "select" and s.sam is None and "SAM is not available" in s.messages[-1]


def test_accept_never_changes_hidden_parts(s):
    s.select_at(10, 25)                 # part 5
    s.set_hidden("object", 1, True)     # hide the mug (parts 2 and 3)
    s.set_tool("sam")
    s.sam_cycle()                       # the radius-16 disc reaches into the mug
    assert s.sam_accept()
    assert s.ann.part_area(2) == 200 and s.ann.part_area(3) == 100


def test_accept_respects_protect(s):
    s.select_at(10, 25)
    s.set_protect(True)
    s.set_tool("sam")
    s.sam_cycle()
    s.sam_accept()
    assert s.ann.part_area(2) == 200


def test_switching_tool_or_image_drops_the_edit(s):
    s.select_at(10, 10)
    s.set_tool("sam")
    s.set_tool("brush")
    assert s.sam is None
    s.set_tool("sam")
    token = s.image_token
    s.step(1)
    assert s.sam is None and s.image_token == token + 1


def test_undo_that_removes_the_target_cancels_the_edit(s):
    s.set_tool("sam")
    s.sam_click(50, 35, True)
    s.sam_accept()                      # a new object + part
    new = s.selected[0]
    s.set_tool("sam")                   # R again: fix the new part
    assert s.sam.target == new
    s.undo()                            # removes the new object
    assert s.sam is None


def test_stale_result_after_a_newer_click_is_ignored(dataset):
    s = make_session(dataset, drive=False)
    s.select_at(10, 10)
    s.set_tool("sam")
    r1 = s.take_sam_request()
    x, y = s.sam.points[0]
    s.sam_click(x + 7, y, False)
    r2 = s.take_sam_request()
    b = s.backend
    assert not s.sam_receive(r1.token, r1.seq, *b.predict(r1.points, r1.labels, r1.box, r1.prev_logits, r1.multimask))
    assert s.sam_receive(r2.token, r2.seq, *b.predict(r2.points, r2.labels, r2.box, r2.prev_logits, r2.multimask))


def test_result_for_an_image_you_left_is_ignored(dataset):
    s = make_session(dataset, drive=False)
    s.select_at(10, 10)
    s.set_tool("sam")
    r1 = s.take_sam_request()
    out = s.backend.predict(r1.points, r1.labels, r1.box, r1.prev_logits, r1.multimask)
    s.step(1)
    assert not s.sam_receive(r1.token, r1.seq, *out) and s.sam is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_session_sam.py -v`
Expected: FAIL. Most tests error with `ValueError: tool must be one of ('select', 'brush', 'eraser')` or `AttributeError` (`sam_click`, `sam_box`, …).

- [ ] **Step 3: Modify `annofix/session.py`**

1. Imports: after `from .io import …` add `from .sam_edit import SamEdit, SamRequest, interior_point, preview_map`.
2. Replace `TOOLS = ("select", "brush", "eraser")` with `TOOLS = ("select", "brush", "eraser", "sam")`.
3. In `__init__`, after the line `        self._locked: np.ndarray | None = None  # per stroke: part ids the user cannot see` add:

```python
        self.image_token = 0  # bumped by every open(): SAM results for an older token are dropped
        self.sam: SamEdit | None = None
        self.sam_enabled = False  # the window's SAM controller turns this on
        self._sam_request: SamRequest | None = None
```

4. In `open`, replace

```python
        self.solo_object = None
        self.messages = list(ann.warnings) + self.labels.unknown_names(ann)
        self._notify("image")
```

with

```python
        self.solo_object = None
        self.image_token += 1
        self.sam, self._sam_request = None, None
        self.messages = list(ann.warnings) + self.labels.unknown_names(ann)
        self._notify("image")
```

5. In `_changed`, replace

```python
            self.solo_object = None
        self.messages = list(self.ann.warnings) + self.labels.unknown_names(self.ann)
```

with

```python
            self.solo_object = None
        if self.sam is not None and ((self.sam.target is not None and self.sam.target not in part_ids)
                                     or (self.sam.object_id is not None and self.sam.object_id not in object_ids)):
            self.sam = None  # the part or object being fixed is gone (e.g. undone)
        self.messages = list(self.ann.warnings) + self.labels.unknown_names(self.ann)
```

6. Replace the whole `set_tool` method with:

```python
    def set_tool(self, tool: str) -> None:
        if tool not in TOOLS:
            raise ValueError(f"tool must be one of {TOOLS}, got {tool!r}")
        if tool == "sam" and not self.sam_enabled:
            self.add_message("SAM is not available (see the status bar); use the brush")
            return
        self.end_stroke()
        if tool != "sam":
            self.sam_cancel()
        self.tool = tool
        self._notify("view")
        if tool == "sam" and self.sam is None and self.primary is not None:
            self.start_sam_fix()
```

7. In `new_object`, replace

```python
    def new_object(self) -> int:
        """A new object with one empty part, selected, brush ready."""
        self.end_stroke()
```

with

```python
    def new_object(self) -> int | None:
        """With SAM: start a new object from SAM clicks (returns None). Without: a new object
        with one empty part, selected, brush ready."""
        if self.sam_enabled:
            self.start_sam_new(new_object=True)
            return None
        self.end_stroke()
```

and in `new_part`, replace

```python
    def new_part(self) -> int:
        """A new empty part in the selected part's object (or a new object), brush ready."""
        self.end_stroke()
```

with

```python
    def new_part(self) -> int | None:
        """With SAM: start a new part (in the selected part's object) from SAM clicks (returns
        None). Without: a new empty part there (or a new object), brush ready."""
        if self.sam_enabled:
            self.start_sam_new(new_object=False)
            return None
        self.end_stroke()
```

8. Append to the end of `class Session`:

```python

    # ---- messages -----------------------------------------------------------------------------

    def add_message(self, text: str) -> None:
        self.messages = self.messages + [text]
        self._notify("status")

    # ---- SAM ----------------------------------------------------------------------------------

    def _start_sam(self, edit: SamEdit) -> bool:
        if not self.sam_enabled:
            self.add_message("SAM is not available (see the status bar); use the brush")
            return False
        self.end_stroke()
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
        """Re-segment the selected part, seeded with one include point deep inside it (not its
        bbox: the part being fixed is wrong, so its bbox is too)."""
        if self.primary is None or not self._start_sam(SamEdit(self.image_token, target=self.primary)):
            return False
        seed = interior_point(self.ann.part_mask(self.primary))
        if seed is not None:
            self._queue(self.sam.add_point(seed[0], seed[1], True))
        return True

    def start_sam_new(self, new_object: bool = False) -> bool:
        """A new part from SAM clicks: in the selected part's object, or in a new object."""
        object_id = None if new_object or self.primary is None else self.ann.object_of(self.primary).id
        return self._start_sam(SamEdit(self.image_token, object_id=object_id))

    def sam_click(self, x: int, y: int, positive: bool) -> None:
        """SAM tool click. With no edit open, a left click starts one: fixing the part under the
        cursor, or a new object on background. Otherwise it adds an include/exclude point."""
        if self.tool != "sam":
            return
        if self.sam is None:
            if not positive:
                return
            h, w = self.ann.size
            pid = self.ann.part_at(y, x) if 0 <= x < w and 0 <= y < h else None
            if pid is not None:
                self.selected = [pid]
                self._notify("selection")
            if not self._start_sam(SamEdit(self.image_token, target=pid)):
                return
        self._queue(self.sam.add_point(x, y, positive))

    def sam_box(self, x0: int, y0: int, x1: int, y1: int) -> None:
        if self.tool != "sam":
            return
        if self.sam is None and not self._start_sam(SamEdit(self.image_token, target=self.primary)):
            return
        self._queue(self.sam.set_box(x0, y0, x1, y1))

    def take_sam_request(self) -> SamRequest | None:
        request, self._sam_request = self._sam_request, None
        return request

    def sam_receive(self, token: int, seq: int, masks, scores, logits) -> bool:
        if self.sam is None or not self.sam.receive(token, seq, masks, scores, logits):
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
        if e is None or e.mask is None:
            return False
        locked = self._locked_lut()
        mask = e.mask if locked is None else e.mask & ~locked[self.ann.label_map]
        with self.ann.transaction("sam"):
            if e.target is None:
                oid = e.object_id if e.object_id is not None else self.ann.new_object()
                pid = self.ann.new_part(oid)
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_session_sam.py -v`
Expected: PASS (19 tests).
Run: `uv run pytest`
Expected: all PASS. Plan 2's tests keep passing because `sam_enabled` defaults to False.

- [ ] **Step 5: Commit**

```bash
git add annofix/session.py tests/test_session_sam.py
git commit -m "feat: session SAM actions: seeded fix, include/exclude/box prompts, cycle, accept, cancel"
```

---

### Task 4: SAM worker thread and controller

**Files:**
- Create: `annofix/ui/sam_worker.py`, `tests/test_sam_worker.py`

**Interfaces:**
- Consumes: Task 3 `Session` (`image_token`, `image`, `tool`, `sam`, `sam_enabled`, `take_sam_request`, `sam_receive`, `sam_cancel`, `set_tool`, `add_message`, `subscribe`), a backend factory `Callable[[], SegmentationBackend]`.
- Produces (used by Task 6):
  - `SamWorker(factory)` (lives in the worker thread): slots `set_image(int token, object image)`, `predict(object request)`; signals `loaded(str)`, `result(int token, int seq, object masks, object scores, object logits, float seconds)`, `failed(str message, bool fatal)`
  - `SamController(session, factory, parent=None)` (GUI thread): sets `session.sam_enabled = True`; attribute `status: str`; signal `status_changed(str)`; `shutdown()` (idempotent; never hangs more than ~3 s)

- [ ] **Step 1: Write the failing test `tests/test_sam_worker.py`**

```python
import time

import pytest
from helpers import LABELS, make_dataset

from annofix.sam_backend import FakeBackend
from annofix.session import Session
from annofix.ui.sam_worker import SamController


@pytest.fixture
def make(tmp_path):
    controllers = []

    def build(backend=None, factory=None):
        images, ann_dir = make_dataset(tmp_path)
        s = Session(images, ann_dir, LABELS)
        made = []

        def default_factory():
            made.append(1)
            return backend

        c = SamController(s, factory or default_factory)
        controllers.append(c)
        s.open(0)
        return s, c, made

    yield build
    for c in controllers:
        c.shutdown()


def predicts(backend):
    return [call for call in backend.calls if call[0] == "predict"]


def test_window_opens_without_loading_the_model(make, qtbot):
    backend = FakeBackend()
    s, c, made = make(backend)
    qtbot.wait(150)
    assert s.sam_enabled and made == [] and backend.calls == [] and c.status.startswith("SAM: not loaded")


def test_sam_round_trip_through_the_worker_thread(make, qtbot):
    backend = FakeBackend()
    s, c, made = make(backend)
    s.select_at(10, 10)
    s.set_tool("sam")
    qtbot.waitUntil(lambda: s.sam is not None and s.sam.mask is not None, timeout=3000)
    assert made == [1] and backend.calls[0] == ("set_image", (40, 60))
    assert c.status.startswith("SAM ready")


def test_only_the_newest_request_is_computed(make, qtbot):
    backend = FakeBackend(delay=0.3)
    s, c, _ = make(backend)
    s.select_at(10, 10)
    s.set_tool("sam")
    qtbot.waitUntil(lambda: len(predicts(backend)) == 1, timeout=3000)  # busy with the seed
    for x in (20, 21, 22):
        s.sam_click(x, 12, False)
    qtbot.waitUntil(lambda: s.sam.received_seq == 4, timeout=5000)
    assert len(predicts(backend)) == 2  # the seed, then only the newest refinement


def test_result_for_an_image_you_left_is_dropped(make, qtbot):
    backend = FakeBackend(delay=0.3)
    s, c, _ = make(backend)
    s.select_at(10, 10)
    s.set_tool("sam")
    qtbot.waitUntil(lambda: len(predicts(backend)) == 1, timeout=3000)
    s.step(1)
    qtbot.wait(600)
    assert s.stem == "b" and s.sam is None


def test_model_that_fails_to_load_disables_sam_but_not_the_editor(make, qtbot):
    def broken():
        raise RuntimeError("no network for the checkpoint download")

    s, c, _ = make(factory=broken)
    s.select_at(10, 10)
    s.set_tool("sam")
    qtbot.waitUntil(lambda: not s.sam_enabled, timeout=3000)
    assert s.tool == "select" and s.sam is None and "no network" in c.status
    assert s.new_part() is not None and s.tool == "brush"  # N falls back to the brush


def test_prediction_error_keeps_sam_enabled(make, qtbot):
    class Flaky(FakeBackend):
        def predict(self, *a, **k):
            raise RuntimeError("CUDA out of memory")

    s, c, _ = make(Flaky())
    s.select_at(10, 10)
    s.set_tool("sam")
    qtbot.waitUntil(lambda: "out of memory" in c.status, timeout=3000)
    assert s.sam_enabled and s.tool == "sam"


def test_shutdown_while_predicting_does_not_hang(make, qtbot):
    backend = FakeBackend(delay=0.5)
    s, c, _ = make(backend)
    s.select_at(10, 10)
    s.set_tool("sam")
    qtbot.waitUntil(lambda: len(predicts(backend)) == 1, timeout=3000)
    t = time.perf_counter()
    c.shutdown()
    assert time.perf_counter() - t < 2.5
    c.shutdown()  # idempotent
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_sam_worker.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'annofix.ui.sam_worker'`.

- [ ] **Step 3: Write `annofix/ui/sam_worker.py`**

```python
"""SAM off the GUI thread. The SamWorker lives in its own QThread: it builds the backend on
first use (importing torch there, not at startup), embeds an image only when asked, and
computes only the newest prediction request. It never touches the Session or widgets, and
every exception becomes a `failed` signal (an exception escaping a worker slot would reach
the GUI excepthook from the wrong thread). The SamController is the GUI-thread glue: it
listens to the Session, sends images/requests, and hands results back."""
from __future__ import annotations

import time
from typing import Callable

from PyQt6.QtCore import QObject, QThread, QTimer, pyqtSignal, pyqtSlot


class SamWorker(QObject):
    loaded = pyqtSignal(str)                                       # "<model> on <device>"
    result = pyqtSignal(int, int, object, object, object, float)  # token, seq, masks, scores, logits, s
    failed = pyqtSignal(str, bool)                                 # message, fatal (model never loaded)

    def __init__(self, factory: Callable):
        super().__init__()
        self._factory = factory
        self._backend = None
        self._image_token: int | None = None
        self._pending_image = None
        self._latest = None
        self._scheduled = False
        self._dead = False

    @pyqtSlot(int, object)
    def set_image(self, token: int, image) -> None:
        self._pending_image = (token, image)
        self._schedule()

    @pyqtSlot(object)
    def predict(self, request) -> None:
        self._latest = request  # a newer request replaces one still waiting
        self._schedule()

    def _schedule(self) -> None:
        if not self._scheduled:
            self._scheduled = True
            QTimer.singleShot(0, self._run)

    def _run(self) -> None:
        self._scheduled = False
        if self._dead:
            return
        try:
            if self._backend is None:
                self._backend = self._factory()
                self.loaded.emit(f"{self._backend.model_name} on {self._backend.device}")
            if self._pending_image is not None:
                token, image = self._pending_image
                self._pending_image = None
                self._backend.set_image(image)
                self._image_token = token
            if self._latest is not None:
                req, self._latest = self._latest, None
                if req.token != self._image_token:
                    return  # the image changed after this request was made
                t = time.perf_counter()
                masks, scores, logits = self._backend.predict(req.points, req.labels, req.box,
                                                              req.prev_logits, req.multimask)
                self.result.emit(req.token, req.seq, masks, scores, logits, time.perf_counter() - t)
        except Exception as e:  # noqa: BLE001 - must never escape the worker thread
            fatal = self._backend is None
            self._dead = fatal
            self.failed.emit(f"{type(e).__name__}: {e}", fatal)


class SamController(QObject):
    status_changed = pyqtSignal(str)
    _send_image = pyqtSignal(int, object)
    _send_request = pyqtSignal(object)

    def __init__(self, session, factory: Callable, parent=None):
        super().__init__(parent)
        self.session = session
        self.status = "SAM: not loaded (press R to use it)"
        self._sent_token: int | None = None
        self._thread = QThread()
        self._worker = SamWorker(factory)
        self._worker.moveToThread(self._thread)
        self._send_image.connect(self._worker.set_image)
        self._send_request.connect(self._worker.predict)
        self._worker.loaded.connect(lambda what: self._set_status(f"SAM ready: {what}"))
        self._worker.result.connect(self._on_result)
        self._worker.failed.connect(self._on_failed)
        self._thread.start()
        session.sam_enabled = True
        session.subscribe(self._on_change)

    def _set_status(self, text: str) -> None:
        self.status = text
        self.status_changed.emit(text)

    def _image(self) -> None:
        s = self.session
        if self.status.startswith("SAM: not loaded"):
            self._set_status("SAM: loading the model (the first time can take a while)…")
        self._send_image.emit(s.image_token, s.image)
        self._sent_token = s.image_token

    def _on_change(self, kind: str) -> None:
        s = self.session
        if not s.sam_enabled:
            return
        if kind == "image":
            self._sent_token = None
            if s.tool == "sam":
                self._image()  # keep the embedding ready while working with SAM
        elif kind == "view" and s.tool == "sam" and self._sent_token != s.image_token:
            self._image()      # warm up as soon as the SAM tool is chosen
        elif kind == "sam_request":
            request = s.take_sam_request()
            if request is None:
                return
            if self._sent_token != request.token:
                self._image()
            if not self.status.startswith("SAM: loading"):
                self._set_status("SAM: thinking…")
            self._send_request.emit(request)

    def _on_result(self, token, seq, masks, scores, logits, seconds) -> None:
        if self.session.sam_receive(token, seq, masks, scores, logits):
            self._set_status(f"SAM ready ({seconds * 1000:.0f} ms)")

    def _on_failed(self, message: str, fatal: bool) -> None:
        s = self.session
        if fatal:
            s.sam_enabled = False
            s.sam_cancel()
            if s.tool == "sam":
                s.set_tool("select")
            self._set_status(f"SAM unavailable: {message}")
        else:
            self._set_status(f"SAM error: {message}")
        s.add_message(self.status)

    def shutdown(self) -> None:
        if not self._thread.isRunning():
            return
        self._thread.quit()
        if not self._thread.wait(3000):  # e.g. a first-time checkpoint download in progress
            self._thread.terminate()
            self._thread.wait()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_sam_worker.py -v`
Expected: PASS (7 tests).
Run: `uv run pytest`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add annofix/ui/sam_worker.py tests/test_sam_worker.py
git commit -m "feat: SAM worker thread with lazy model, on-demand embedding and newest-request-only"
```

---

### Task 5: Canvas SAM mode, prompt markers and the preview overlay

**Files:**
- Modify: `annofix/ui/colors.py` (add `preview_lut`), `annofix/ui/main_window.py` (one line in `_sync_view`), `tests/test_canvas.py`, `tests/test_main_window.py`
- Rewrite: `annofix/ui/canvas.py`

**Interfaces:**
- Consumes: Task 2 `PREVIEW_*` codes; Plan 2 `OverlayItem`.
- Produces (used by Task 6):
  - `colors.preview_lut() -> uint32 LUT` (kept white, added green, taken orange, removed red)
  - `Canvas.mode: str` (`"select" | "paint" | "sam"`, **replaces `paint_mode`**); new signals `sam_clicked(int x, int y, bool include)`, `sam_boxed(int x0, int y0, int x1, int y1)`; methods `set_sam_prompts(points, labels, box)`, `set_sam_preview(preview | None)`; attribute `sam_preview_item: OverlayItem | None`; `focusNextPrevChild` always returns False

- [ ] **Step 1: Update existing tests and write the new failing tests**

In `tests/test_canvas.py` replace `    c.paint_mode = True` with `    c.mode = "paint"`.
In `tests/test_main_window.py` replace `    assert win.canvas.paint_mode and win.tool_actions["brush"].isChecked()` with `    assert win.canvas.mode == "paint" and win.tool_actions["brush"].isChecked()`.
Append to `tests/test_canvas.py`:

```python


def test_sam_mode_left_is_include_right_is_exclude(qtbot):
    c = make_canvas(qtbot)
    c.mode = "sam"
    got = []
    c.sam_clicked.connect(lambda x, y, include: got.append((x, y, include)))
    c.clicked.connect(lambda *a: got.append("select click"))
    qtbot.mouseClick(c.viewport(), Qt.MouseButton.LeftButton, pos=at(c, 12, 30))
    qtbot.mouseClick(c.viewport(), Qt.MouseButton.RightButton, pos=at(c, 13, 31))
    assert got == [(12, 30, True), (13, 31, False)]


def test_shift_drag_draws_a_box(qtbot):
    c = make_canvas(qtbot)
    c.mode = "sam"
    got = []
    c.sam_boxed.connect(lambda *box: got.append(box))
    c.sam_clicked.connect(lambda *a: got.append("click"))
    vp = c.viewport()
    qtbot.mousePress(vp, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.ShiftModifier, at(c, 10, 10))
    send_move(vp, at(c, 20, 15))
    qtbot.mouseRelease(vp, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.ShiftModifier, at(c, 20, 15))
    assert got == [(10, 10, 20, 15)]


def test_prompt_markers_and_preview(qtbot):
    from annofix.sam_edit import PREVIEW_ADDED
    from annofix.ui.colors import preview_lut

    c = make_canvas(qtbot)
    c.set_sam_prompts([(5, 5), (10, 10)], [1, 0], (1, 1, 20, 20))
    assert len(c._markers) == 3
    preview = np.zeros((40, 60), np.uint8)
    preview[5, 5] = PREVIEW_ADDED
    c.set_sam_preview(preview)
    assert c.sam_preview_item.argb_at(5, 5) == preview_lut()[PREVIEW_ADDED]
    c.set_sam_preview(None)
    c.set_sam_prompts([], [], None)
    assert c.sam_preview_item is None and c._markers == []


def test_tab_never_moves_focus_off_the_canvas(qtbot):
    c = make_canvas(qtbot)
    assert c.focusNextPrevChild(True) is False
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_canvas.py tests/test_main_window.py -v`
Expected: FAIL — the 4 new tests (no `sam_clicked`, `sam_boxed`, `set_sam_prompts`, `preview_lut`; `focusNextPrevChild` returns True). The updated paint tests also fail, because `mode` isn't used yet and nothing paints.

- [ ] **Step 3: Implement**

Append to `annofix/ui/colors.py`:

```python


def preview_lut() -> np.ndarray:
    """Colours for the SAM accept preview (sam_edit.preview_map codes)."""
    from ..sam_edit import PREVIEW_ADDED, PREVIEW_KEPT, PREVIEW_REMOVED, PREVIEW_TAKEN

    lut = np.zeros(LUT_SIZE, np.uint32)
    lut[PREVIEW_KEPT] = argb((255, 255, 255))
    lut[PREVIEW_ADDED] = argb((40, 220, 40))
    lut[PREVIEW_TAKEN] = argb((255, 150, 0))
    lut[PREVIEW_REMOVED] = argb((230, 30, 60))
    return lut
```

In `annofix/ui/main_window.py` `_sync_view`, replace `        self.canvas.paint_mode = s.tool in ("brush", "eraser")` with
`        self.canvas.mode = {"brush": "paint", "eraser": "paint", "sam": "sam"}.get(s.tool, "select")`.

Replace `annofix/ui/canvas.py` with:

```python
"""Image + mask overlay view. Wheel zooms around the cursor, middle-drag pans. By tool mode:
"select" — left click selects; "paint" — left drag paints; "sam" — left/right click add
include/exclude points and Shift+drag draws a box. Everything is reported in image pixels.
The SAM prompts and the accept preview are drawn on top."""
from __future__ import annotations

import math

import numpy as np
from PyQt6.QtCore import QRectF, Qt, pyqtSignal
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_canvas.py tests/test_main_window.py -v`
Expected: PASS (11 canvas + 12 window tests).
Run: `uv run pytest`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add annofix/ui/colors.py annofix/ui/canvas.py annofix/ui/main_window.py tests/test_canvas.py tests/test_main_window.py
git commit -m "feat: canvas tool modes with SAM include/exclude clicks, box drag, prompt markers and preview"
```

---

### Task 6: Window and CLI wiring

**Files:**
- Modify: `annofix/ui/main_window.py`, `annofix/__main__.py`
- Create: `tests/test_main_window_sam.py`

**Interfaces:**
- Consumes: Task 3 Session SAM API, Task 4 `SamController`, Task 5 canvas API.
- Produces:
  - `MainWindow(session, sam_factory=None)`, with attribute `sam: SamController | None` and `tool_actions["sam"]` (visible only with a factory)
  - Hotkeys:
    - `R`: SAM tool; with a part selected it starts a seeded fix
    - `Tab` / `Backtab` (Shift+Tab): next / previous candidate
    - `Return`/`Enter`: replace; `Shift+Return`/`Shift+Enter`: add
    - `Esc`: cancel the stroke, then the SAM edit, then return to select
  - CLI flags `--sam-model {tiny,small,base_plus,large}`, `--device`, `--checkpoint`, `--no-sam`

- [ ] **Step 1: Write the failing test `tests/test_main_window_sam.py`**

```python
import numpy as np
import pytest
from helpers import LABELS, make_dataset
from PyQt6.QtCore import QPointF, Qt
from PyQt6.QtWidgets import QApplication, QMessageBox

from annofix.__main__ import main
from annofix.sam_backend import FakeBackend
from annofix.session import Session
from annofix.ui.main_window import MainWindow


@pytest.fixture
def win(tmp_path, qtbot, monkeypatch):
    images, ann_dir = make_dataset(tmp_path)
    monkeypatch.setattr(QMessageBox, "information", lambda *a, **k: None)
    monkeypatch.setattr(QMessageBox, "warning", lambda *a, **k: None)
    w = MainWindow(Session(images, ann_dir, LABELS), sam_factory=FakeBackend)
    qtbot.addWidget(w)
    w.resize(1200, 700)
    w.show()
    qtbot.waitExposed(w)
    w.activateWindow()
    qtbot.waitUntil(lambda: QApplication.activeWindow() is w, timeout=2000)
    yield w
    w.sam.shutdown()


def at(w, x, y):
    return w.canvas.mapFromScene(QPointF(x + 0.5, y + 0.5))


def fix_part_2(w, qtbot):
    qtbot.mouseClick(w.canvas.viewport(), Qt.MouseButton.LeftButton, pos=at(w, 10, 10))
    qtbot.keyClick(w.canvas, Qt.Key.Key_R)
    qtbot.waitUntil(lambda: w.session.sam is not None and w.session.sam.mask is not None, timeout=3000)


def test_r_exclude_click_enter_fixes_a_part(win, qtbot):
    s = win.session
    fix_part_2(win, qtbot)
    assert win.canvas.mode == "sam" and win.canvas.sam_preview_item is not None
    assert win.tool_actions["sam"].isChecked() and "candidate" in win.status.text()
    x, y = s.sam.points[0]
    qtbot.mouseClick(win.canvas.viewport(), Qt.MouseButton.RightButton, pos=at(win, x + 7, y))
    qtbot.waitUntil(lambda: s.sam.received_seq == 2, timeout=3000)
    expected = s.sam.mask.copy()
    qtbot.keyClick(win.canvas, Qt.Key.Key_Return)
    assert s.sam is None and np.array_equal(s.ann.part_mask(2), expected)
    assert win.canvas.sam_preview_item is None


def test_tab_cycles_and_esc_cancels(win, qtbot):
    s = win.session
    fix_part_2(win, qtbot)
    index = s.sam.index
    qtbot.keyClick(win.canvas, Qt.Key.Key_Tab)
    assert s.sam.index == (index + 1) % 3 and QApplication.focusWidget() is win.canvas
    qtbot.keyClick(win.canvas, Qt.Key.Key_Escape)
    assert s.sam is None and s.tool == "sam"
    qtbot.keyClick(win.canvas, Qt.Key.Key_Escape)
    assert s.tool == "select"


def test_shift_enter_adds(win, qtbot):
    s = win.session
    fix_part_2(win, qtbot)
    before = s.ann.part_area(2)
    qtbot.keyClick(win.canvas, Qt.Key.Key_Return, Qt.KeyboardModifier.ShiftModifier)
    assert s.ann.part_area(2) > before and s.ann.part_at(5, 5) == 2


def test_window_without_sam_hides_the_tool(tmp_path, qtbot):
    images, ann_dir = make_dataset(tmp_path)
    w = MainWindow(Session(images, ann_dir, LABELS))
    qtbot.addWidget(w)
    assert w.sam is None and not w.tool_actions["sam"].isVisible() and not w.session.sam_enabled


def test_cli_rejects_unknown_sam_model(tmp_path):
    with pytest.raises(SystemExit):
        main(["--images", str(tmp_path), "--annotations", str(tmp_path), "--sam-model", "huge"])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_main_window_sam.py -v`
Expected: FAIL — `TypeError: MainWindow.__init__() got an unexpected keyword argument 'sam_factory'` (and the CLI test fails because `--sam-model` doesn't exist yet: argparse exits with SystemExit, so that one may already pass. If so, it is a pinning test).

- [ ] **Step 3: Modify `annofix/ui/main_window.py`**

1. Add `from .sam_worker import SamController` after `from .label_panel import LabelPanel`.
2. Replace `    def __init__(self, session: Session):` with `    def __init__(self, session: Session, sam_factory=None):`.
3. Replace

```python
        self._connect()
        self._autosave = QTimer(self)
```

with

```python
        self._connect()
        self.sam: SamController | None = None
        if sam_factory is not None:
            self.sam = SamController(session, sam_factory, self)
            self.sam.status_changed.connect(lambda _text: self._refresh_status())
        self.tool_actions["sam"].setVisible(self.sam is not None)
        self._autosave = QTimer(self)
```

4. In `_build_toolbar`, replace
`        for tool, text in (("select", "Select (V)"), ("brush", "Brush (B)"), ("eraser", "Eraser (E)")):`
with
`        for tool, text in (("select", "Select (V)"), ("brush", "Brush (B)"), ("eraser", "Eraser (E)"), ("sam", "SAM (R)")):`
5. In `_build_shortcuts`, after `        key("Ctrl+S", s.save)` add:

```python
        key("R", lambda: s.set_tool("sam"))
        key("Tab", lambda: s.sam_cycle(1))
        key("Backtab", lambda: s.sam_cycle(-1))
        key("Return", lambda: s.sam_accept(union=False))
        key("Enter", lambda: s.sam_accept(union=False))
        key("Shift+Return", lambda: s.sam_accept(union=True))
        key("Shift+Enter", lambda: s.sam_accept(union=True))
```

6. In `_connect`, after `        c.stroke_ended.connect(s.end_stroke)` add:

```python
        c.sam_clicked.connect(s.sam_click)
        c.sam_boxed.connect(s.sam_box)
```

7. In `_on_change`, replace the final `        self._refresh_status()` of that method (the line right after `            self._refresh_files()`) with:

```python
        if kind in ("image", "annotation", "view", "sam"):
            self._refresh_sam()
        self._refresh_status()
```

8. Add after `_refresh_lut`:

```python
    def _refresh_sam(self) -> None:
        s = self.session
        if s.sam is None:
            self.canvas.set_sam_prompts([], [], None)
            self.canvas.set_sam_preview(None)
            return
        self.canvas.set_sam_prompts(s.sam.points, s.sam.labels, s.sam.box)
        self.canvas.set_sam_preview(s.sam_preview())
```

9. In `_refresh_status`, replace

```python
        self.status.setText(f"{s.index + 1}/{len(s.images)}  {s.stem}  [{s.ann.review_status}{unsaved}]"
                            f"  tool: {s.tool} r={s.brush_radius}{protect}  "
                            f"objects: {len(s.ann.objects)}{note}")
```

with

```python
        sam = ""
        if s.sam is not None and s.sam.candidates:
            sam = (f"  |  candidate {s.sam.index + 1}/{len(s.sam.candidates)} IoU {s.sam.score:.2f}: "
                   "Tab next · Enter replace · Shift+Enter add · Esc cancel")
        elif s.tool == "sam" and self.sam is not None:
            sam = f"  |  {self.sam.status}"
        self.status.setText(f"{s.index + 1}/{len(s.images)}  {s.stem}  [{s.ann.review_status}{unsaved}]"
                            f"  tool: {s.tool} r={s.brush_radius}{protect}  "
                            f"objects: {len(s.ann.objects)}{sam}{note}")
```

10. Replace `_escape` with:

```python
    def _escape(self) -> None:
        s = self.session
        if s.painting:
            s.cancel_stroke()
        elif s.sam is not None:
            s.sam_cancel()
        else:
            s.set_tool("select")
            s.select_part(None)
```

11. In `closeEvent`, replace the final `        ev.accept()` with:

```python
        if self.sam is not None:
            self.sam.shutdown()
        ev.accept()
```

`annofix/__main__.py`:
- After `    args = ap.parse_args(argv)`'s preceding argument definitions (i.e. right before `    args = ap.parse_args(argv)`), insert:

```python
    from .sam_backend import MODELS  # a plain dict; torch is only imported by Sam2Backend()

    ap.add_argument("--sam-model", choices=sorted(MODELS),
                    help="SAM 2.1 size (default: base_plus on a GPU, small on CPU)")
    ap.add_argument("--device", help="cuda or cpu (default: cuda if available)")
    ap.add_argument("--checkpoint", type=Path, help="local SAM 2.1 .pt instead of a Hugging Face download")
    ap.add_argument("--no-sam", action="store_true", help="start without SAM (brush and eraser only)")
```

- Replace `        window = MainWindow(session)` with:

```python
        sam_factory = None
        if not args.no_sam:
            def sam_factory():  # runs in the SAM worker thread, on first SAM use
                from .sam_backend import Sam2Backend

                return Sam2Backend(args.sam_model, args.device, args.checkpoint)
        window = MainWindow(session, sam_factory=sam_factory)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_main_window_sam.py -v`
Expected: PASS (5 tests).
Run: `uv run pytest`
Expected: all PASS, including `test_gui_path_does_not_import_torch`.

- [ ] **Step 5: Commit**

```bash
git add annofix/ui/main_window.py annofix/__main__.py tests/test_main_window_sam.py
git commit -m "feat: SAM in the editor: R, include/exclude clicks, Tab, Enter/Shift+Enter, status; CLI flags"
```

---

### Task 7: Real SAM on real frames, the M6 decision, README

**Files:**
- Modify: `tests/test_sam_backend.py` (gated real-SAM Session test), `README.md`, `docs/notes/2026-10-03-m0-measurements.md`
- Uses (not committed): `$SCRATCH/sam_tour.py` (`$SCRATCH` = the session scratchpad), `data/sam_check/` (a fresh copy of `data/sample_ann`; `data/` is git-ignored)

**Interfaces:**
- Consumes: everything above; real `Sam2Backend`; frames in `data/sample`.

- [ ] **Step 1: Write the gated real-SAM test** (append to `tests/test_sam_backend.py`)

```python


@pytest.mark.skipif(not REAL_FRAME.exists(), reason="needs the CamCalibration lab frame")
def test_session_sam_fix_on_a_real_frame(tmp_path):
    import shutil

    from annofix.config import Labels
    from annofix.sam_backend import Sam2Backend
    from annofix.session import Session

    images = tmp_path / "images"
    images.mkdir()
    shutil.copy(REAL_FRAME, images / REAL_FRAME.name)
    s = Session(images, tmp_path / "ann", Labels(materials=("metal",), objects={}))
    s.sam_enabled = True
    backend = Sam2Backend("base_plus")

    def drive(kind):
        if kind == "image":
            backend.set_image(s.image)
        elif kind == "sam_request":
            req = s.take_sam_request()
            s.sam_receive(req.token, req.seq,
                          *backend.predict(req.points, req.labels, req.box, req.prev_logits, req.multimask))

    s.subscribe(drive)
    s.open(0)
    s.set_tool("sam")
    s.sam_click(627, 192, True)  # robot arm, upper link
    assert s.sam.mask is not None and s.sam.mask[192, 627]
    assert 2_000 < int(s.sam.mask.sum()) < 0.25 * 1280 * 720
    assert s.sam_accept() and s.ann.part_at(192, 627) == s.selected[0]
```

Run: `ANNOFIX_SAM_TESTS=1 uv run pytest tests/test_sam_backend.py -v`
Expected: PASS (6 tests).

- [ ] **Step 2: Write the tour `$SCRATCH/sam_tour.py`** (drives the real window offscreen with real SAM)

```python
"""Real SAM through the real window, offscreen: latency per step and screenshots."""
import os
import sys
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
import annofix  # noqa: E402,F401
from pathlib import Path  # noqa: E402

from PyQt6.QtWidgets import QApplication  # noqa: E402

from annofix.config import load_labels  # noqa: E402
from annofix.sam_backend import Sam2Backend  # noqa: E402
from annofix.session import Session  # noqa: E402
from annofix.ui.main_window import MainWindow  # noqa: E402

out = Path(sys.argv[1])
out.mkdir(parents=True, exist_ok=True)
app = QApplication([])
s = Session("data/sample", "data/sam_check", load_labels("labels.yaml"), annotator="tour")
w = MainWindow(s, sam_factory=lambda: Sam2Backend("base_plus"))
w.resize(1600, 900)
w.show()


def wait_for(cond, label, timeout=300):
    t = time.perf_counter()
    while not cond():
        app.processEvents()
        if time.perf_counter() - t > timeout:
            raise TimeoutError(label)
        time.sleep(0.005)
    dt = time.perf_counter() - t
    print(f"{label}: {dt * 1000:.0f} ms")
    return dt


def snap(name):
    app.processEvents()
    w.grab().save(str(out / f"{name}.png"))
    print("  saved", name, "|", w.status.text()[-110:])


def fix_at(x, y, name):
    s.set_tool("select")
    s.select_at(x, y)
    s.set_tool("sam")
    if s.sam is None:  # the point is on background: start a new-object fix from a click there
        s.sam_click(x, y, True)
    seq = s.sam.seq
    wait_for(lambda: s.sam is not None and s.sam.received_seq == seq, f"{name}: first answer")
    for i in range(len(s.sam.candidates)):
        snap(f"{name}_candidate{i + 1}")
        s.sam_cycle()


s.open(3)                                                   # cam0_120
fix_at(627, 192, "arm")                                     # first use: model load + embed + predict
x, y = s.sam.points[0]
s.sam_click(x + 40, y + 120, False)                         # exclude below the seed
wait_for(lambda: s.sam.received_seq == 2, "arm: refinement click")
snap("arm_refined")
s.sam_accept()
snap("arm_accepted")
fix_at(830, 330, "cable")                                   # thin: cable on the table
fix_at(610, 530, "tripod_leg")                              # thin: tripod leg
s.sam_cancel()
t = time.perf_counter()
s.step(1)
s.set_tool("sam")
s.sam_click(400, 300, True)
seq = s.sam.seq
wait_for(lambda: s.sam.received_seq == seq, "next image: embed + first click")
snap("next_image_click")
w.sam.shutdown()
```

- [ ] **Step 3: Run the tour and inspect**

```bash
rm -rf data/sam_check && cp -r data/sample_ann data/sam_check && rm -rf data/sam_check/.amg_cache
uv run python $SCRATCH/sam_tour.py $SCRATCH/sam_tour
```
Expected: timing lines plus saved screenshots. The first seeded fix includes the model load (a few seconds; longer only if the checkpoint isn't cached). Later clicks should be well under 500 ms, and a new image's embed plus first click around 150–300 ms on the RTX 3050. Watch `sensors | grep Package`; if the CPU passes 95 °C, wait for it to cool before re-running.

Open the screenshots and check:
- the arm candidates show the include (green) and exclude (red) markers, with the preview colours: white kept, green added, orange taken, red removed
- after accepting, the arm part matches the chosen candidate
- for the cable and the tripod leg, whether **any** candidate follows the thin structure closely, or all of them are blobby (spilling more than a few pixels past the edge)

- [ ] **Step 4: Record measurements and the M6 decision** — append to `docs/notes/2026-10-03-m0-measurements.md`

```markdown
## Interactive SAM in the editor (Plan 3)

| step | ms |
|---|---|
| first SAM use (model load + embed + predict) | <from the tour> |
| refinement click | <from the tour> |
| next image: embed + first click | <from the tour> |

Thin parts (1280x720, base_plus): cable — <follows / blobby>; tripod leg — <follows / blobby>.
**M6 decision:** <zoomed-crop SAM needed: yes/no>, because <one sentence from the screenshots>.
If yes, it is a follow-up plan (crop the target's bbox x1.5-2, at least 256 px, offset the prompts);
not part of Plan 3.
```

Replace every `<…>` with the measured value or observation from Step 3. The template above is a form to fill in, not the final text.

- [ ] **Step 5: Add SAM to `README.md`** — in the "Fixing annotations (GUI)" section:

a) Replace the first command line block with:

````markdown
```bash
uv run python -m annofix --images IMAGES_DIR --annotations ANNOTATIONS_DIR [--labels labels.yaml] [--annotator NAME]
#   SAM options: [--sam-model tiny|small|base_plus|large] [--device cuda|cpu] [--checkpoint FILE.pt] [--no-sam]
```
````

b) Add these rows to the shortcut table, right after the "Undo · redo" row:

```markdown
| **SAM fix** the selected part (seeded inside it) | `R` |
| SAM: include · exclude · box | left click · right click · Shift+drag |
| SAM: next/previous candidate · replace · add · cancel | `Tab`/`Shift+Tab` · `Enter` · `Shift+Enter` · `Esc` |
| New part · new object (with SAM: from clicks) | `N` · `Shift+N` |
```

and delete the old `| New part in this object · new object | \`N\` · \`Shift+N\` |` row.

c) After the table, add:

```markdown
SAM loads on first use (`R`), in the background; the status bar shows its state. The preview
colours what Enter would do: white kept, green added, orange taken from another part, red
removed. Hidden parts (and parts outside a solo object) are never changed, and "Protect other
parts" applies to SAM as it does to the brush.
```

- [ ] **Step 6: Launch the real window for the user**

Run (background): `uv run python -m annofix --images data/sample --annotations data/sam_check --annotator Shabaj`
Ask the user to try `R` on a bled mask, then a right-click on the bleed, then `Tab`, then `Enter`. Record their feedback in the ledger; fixes it prompts are a new request.

- [ ] **Step 7: Full test run and commit**

Run: `uv run pytest && ANNOFIX_SAM_TESTS=1 uv run pytest -m sam`
Expected: all PASS.

```bash
git add tests/test_sam_backend.py README.md docs/notes/2026-10-03-m0-measurements.md
git commit -m "docs: SAM usage, live SAM latency and the zoomed-crop decision; real-frame SAM test"
```
