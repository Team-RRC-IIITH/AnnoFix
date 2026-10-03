# AnnoFix Plan 1 of 3 — Foundation (environment, data core, bootstrap) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A tested, GUI-less core for AnnoFix: a uv environment with SAM 2.1 running on the GPU (measured), the object → part → material annotation model with single pixel ownership and exact undo, the on-disk format, and a `bootstrap` CLI that turns SAM automatic masks into starting annotations.

**Architecture:** Pure-numpy core (`annotation.py`, `io.py`, `overlaps.py`, `config.py`) with no torch dependency; SAM 2.1 isolated behind `sam_backend.py` (heavy imports inside functions); `bootstrap.py` composes them and caches raw SAM masks so grouping can be re-tuned without the GPU. Plans 2 (viewer + manual editing) and 3 (interactive SAM) build on these interfaces and are written after this plan's M0 measurements are in.

**Tech Stack:** Python 3.12, uv, numpy, opencv-python-headless, PyYAML, PyTorch (CUDA), SAM 2.1 (`facebookresearch/sam2`), huggingface-hub, pytest. PyQt6 is installed now (Qt coexistence is checked in M0) but used from Plan 2.

**Spec:** `docs/superpowers/specs/2026-10-03-annofix-design.md`

## Global Constraints

- Python `>=3.12,<3.13`; environment managed by `uv` in `.venv`; never `--system-site-packages`.
- `opencv-python-headless` only — never `opencv-python` (its bundled Qt clashes with PyQt6).
- sam2 comes from git `https://github.com/facebookresearch/sam2.git` at `2b90b9f5ceec907a1c18123530e92e794ad901a4`, built with `SAM2_BUILD_CUDA=0`. Never the PyPI package `sam2` (third-party fork). Never clone the sam2 repo into the project root.
- Thread cap (machine rule): `OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, `MKL_NUM_THREADS`, `NUMEXPR_NUM_THREADS` = `"10"` set before numpy is imported — in `annofix/__init__.py` and `tests/conftest.py`; `torch.set_num_threads(10)` when SAM runs on CPU.
- Label map: 2-D `uint16`, `0` = background, pixel value = part id; one owner per pixel; always modified **in place** (the UI in Plan 2 holds references to it).
- JSON `schema_version: 1`; unlabelled values are `null`; objects and parts share one id counter; ids are never reused (`next_id` only grows, including across undo).
- Files per image `<stem>`: `<stem>.parts.png` + `<stem>.json`; JSON is serialised before anything is written; PNG is written before JSON; each write is atomic (tmp + fsync + `os.replace`).
- `torch` / `sam2` are imported only inside `annofix/sam_backend.py` functions; every other module imports without them.
- Commits use the configured git author and end with the trailer `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.

### Spec corrections made while planning (flag to the user in the hand-off)

- **Containment `parent_area_ratio` is 1.05, not 1.25.** With 1.25 a part can be at most 80% of its parent, so a mug body (≈85% of the mug) would become a separate overlapping *object* instead of a part. Near-duplicates are already removed by the IoU-0.9 dedupe, so 1.05 only needs to exclude near-equal masks.
- **Orphan pixels** (ids in the PNG but not the JSON) go into a new object with class `null` and a load warning, rather than an object literally named "orphans" (which would be an unknown class).
- **Bootstrap `--overwrite` never replaces annotations whose `review_status` is `reviewed` or `flagged`** (added safety; the spec only said bootstrap writes unreviewed annotations).

## Review Focus

1. Re-running bootstrap on a folder someone is already fixing must not replace their work: an existing annotation is skipped by default, and `--overwrite` still never replaces a `reviewed`/`flagged` one. → Task 8 tests `test_existing_annotation_is_never_overwritten_by_default`, `test_overwrite_never_touches_reviewed_work`.
2. A crash or full disk during save must leave something loadable with no silent loss: serialise first, PNG then JSON, and a new-PNG/old-JSON pair loads with the new part recovered as an orphan. → Task 6 tests `test_unserialisable_meta_leaves_existing_files_untouched`, `test_failed_json_write_still_loads`.
3. Mixed 1280x720 / 640x480 frames, or a stale mask cache, must fail loudly instead of producing misaligned masks. → Task 6 `test_size_mismatch_with_image_is_an_error`, Task 8 `test_cache_of_wrong_size_is_an_error`.
4. Image folders as recorded (colour and depth PNGs side by side) must just work: depth frames are skipped, not crashed on. → Task 6 `test_list_images_skips_depth_and_outputs`, `test_load_image_accepts_grey_rejects_depth`; Task 8 `test_main_regroup_skips_depth_and_counts`.
5. An edit that fails halfway must leave the annotation exactly as before and record nothing; part ids ≥ 256 must survive save/load. → Task 4 `test_exception_rolls_back_and_records_nothing`; Task 6 `test_roundtrip_preserves_uint16_ids_and_meta`.

## File Structure

| File | Responsibility |
|---|---|
| `pyproject.toml` | uv project, dependencies, sam2 git source + build variable, pytest config |
| `labels.yaml` | example materials list and object classes → suggested part names |
| `README.md` | install, bootstrap usage, file format, tests |
| `annofix/__init__.py` | thread cap before numpy |
| `annofix/config.py` | `Labels`, `load_labels()` |
| `annofix/annotation.py` | `Part`, `AnnObject`, `Annotation` (ops, ownership rules, transactions, undo/redo, validate, reconcile) |
| `annofix/io.py` | image listing/loading, annotation save/load, `load()` helper |
| `annofix/overlaps.py` | candidate masks → single-ownership object/part `Annotation` |
| `annofix/sam_backend.py` | `Sam2Backend` (set_image / predict / generate), model tables |
| `annofix/bootstrap.py` | `python -m annofix.bootstrap` CLI, mask cache |
| `scripts/spike_sam.py` | M0 measurements |
| `scripts/render_overlay.py` | write overlay JPEGs to eyeball annotations before the GUI exists |
| `docs/notes/2026-10-03-m0-measurements.md` | measured numbers + model decision |
| `tests/…` | one test file per module (see tasks) |

---

### Task 1: Project scaffold and environment

**Files:**
- Create: `pyproject.toml`, `labels.yaml`, `annofix/__init__.py`, `tests/conftest.py`, `tests/test_package.py`
- Modify: none (`.gitignore` already exists and covers `.venv/`, `checkpoints/`, `data/`, `annotations/`, `*.npz`)

**Interfaces:**
- Produces: importable package `annofix` that sets the thread-cap env vars on import; `labels.yaml` at repo root (read by Task 3 tests).

- [ ] **Step 1: Write `pyproject.toml`**

```toml
[project]
name = "annofix"
version = "0.1.0"
description = "Fix object/part/material segmentation annotations with SAM 2.1"
requires-python = ">=3.12,<3.13"
dependencies = [
    "numpy>=1.26",
    "opencv-python-headless>=4.10",
    "pyyaml>=6.0",
    "pyqt6>=6.7",
    "torch>=2.5.1",
    "torchvision>=0.20.1",
    "sam-2",
    "huggingface-hub>=0.24",
]

[dependency-groups]
dev = ["pytest>=8"]

[tool.uv.sources]
sam-2 = { git = "https://github.com/facebookresearch/sam2.git", rev = "2b90b9f5ceec907a1c18123530e92e794ad901a4" }

[tool.uv.extra-build-variables]
sam-2 = { SAM2_BUILD_CUDA = "0" }

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["annofix"]

[tool.pytest.ini_options]
testpaths = ["tests"]
markers = ["sam: needs torch + a SAM 2.1 checkpoint download (run with ANNOFIX_SAM_TESTS=1)"]
```

- [ ] **Step 2: Write `labels.yaml`**

```yaml
# Materials: list order sets the editor hotkeys 1-9.
materials:
  - plastic
  - metal
  - rubber
  - glass
  - ceramic
  - wood
  - paper
  - fabric
  - other

# Object classes -> suggested part names (the editor also accepts free text).
objects:
  screwdriver: [handle, shaft, tip]
  mug: [body, handle]
  bottle: [body, cap, label]
  scissors: [blade, handle, pivot]
  camera: [body, lens, mount]
  tripod: [legs, head]
  cable: [wire, connector]
  laptop: [lid, screen, keyboard, base]
  robot_arm: [link, joint, gripper, base]
```

- [ ] **Step 3: Write the failing test `tests/test_package.py`**

```python
import os
import subprocess
import sys


def test_import_caps_blas_threads():
    env = {k: v for k, v in os.environ.items()
           if k not in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")}
    code = ("import os, annofix; "
            "print(*(os.environ[v] for v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', "
            "'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS')))")
    out = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, check=True)
    assert out.stdout.split() == ["10", "10", "10", "10"]
```

And `tests/conftest.py`:

```python
import os

# Machine rule: cap BLAS/OpenMP pools before any test module imports numpy.
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_var, "10")
```

- [ ] **Step 4: Install the dependencies (not the project yet — `annofix/` doesn't exist, so building it would fail)**

Run: `SAM2_BUILD_CUDA=0 uv sync --no-install-project`
Expected: resolves and installs (several GB: torch + CUDA wheels); `sam-2` builds from git without trying to compile CUDA. If the resolve fails on `extra-build-variables`, delete that table and rely on the env var, and note it in the README.

- [ ] **Step 5: Run the test to verify it fails**

Run: `uv run --no-sync pytest tests/test_package.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'annofix'` in the subprocess (package not created yet).

- [ ] **Step 6: Write `annofix/__init__.py`**

```python
"""AnnoFix: fix object/part/material segmentation annotations with SAM 2.1."""
import os

# Machine rule: cap BLAS/OpenMP pools before numpy is imported anywhere. On this 24-core
# laptop the default (all cores) is several times slower and much hotter for our array sizes.
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_var, "10")

__version__ = "0.1.0"
```

Then run `SAM2_BUILD_CUDA=0 uv sync` (the project itself is now installable).

- [ ] **Step 7: Run tests and environment checks**

Run: `uv run pytest tests/test_package.py -v`
Expected: PASS.

Run: `uv run python -c "import torch, sam2, cv2, PyQt6.QtCore as q; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0), cv2.__version__, q.PYQT_VERSION_STR)"`
Expected: a torch version, `True`, `NVIDIA GeForce RTX 3050 6GB Laptop GPU`, a cv2 version, a PyQt version.

Run: `uv pip list | grep -i -E "opencv|sam"`
Expected: exactly `opencv-python-headless` and `sam-2` — no plain `opencv-python`.

If `import sam2` or the CUDA check fails with the resolved torch: pin `"torch==2.7.1"`, `"torchvision==0.22.1"` in `pyproject.toml`, re-run `SAM2_BUILD_CUDA=0 uv sync`, repeat this step, and record the pin and reason in the README.

- [ ] **Step 8: Commit**

```bash
git add pyproject.toml uv.lock labels.yaml annofix/__init__.py tests/conftest.py tests/test_package.py
git commit -m "chore: uv project with SAM 2.1, thread cap, example labels" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 2: SAM 2.1 backend and M0 measurements

**Files:**
- Create: `annofix/sam_backend.py`, `tests/test_sam_backend.py`, `scripts/spike_sam.py`, `docs/notes/2026-10-03-m0-measurements.md`

**Interfaces:**
- Consumes: Task 1 environment.
- Produces (used by Task 8 and Plan 3):
  - `MODELS: dict[str, str]` (`"tiny"|"small"|"base_plus"|"large"` → HF id), `CONFIGS: dict[str, str]`, `DEFAULT_MODEL: dict[str, str]` (`{"cuda": ..., "cpu": ...}`)
  - `pick_device(device: str | None) -> str`
  - `class Sam2Backend(model: str | None = None, device: str | None = None, checkpoint: str | Path | None = None)` with attributes `.device: str`, `.model_name: str` and methods
    - `set_image(image: np.ndarray) -> None` (RGB uint8 HxWx3)
    - `predict(points=None, labels=None, box=None, prev_logits=None, multimask=True) -> tuple[np.ndarray, np.ndarray, np.ndarray]` = (bool masks `(C,H,W)`, scores `(C,)`, low-res logits `(C,256,256)`); points are `(x, y)` pixels, labels 1 = include / 0 = exclude, box is XYXY pixels
    - `generate(image, points_per_side=32, points_per_batch=64) -> tuple[np.ndarray, np.ndarray]` = (bool masks `(N,H,W)`, predicted-IoU scores `(N,)`)

- [ ] **Step 1: Write the failing test `tests/test_sam_backend.py`**

```python
import os

import numpy as np
import pytest

pytestmark = [
    pytest.mark.sam,
    pytest.mark.skipif(os.environ.get("ANNOFIX_SAM_TESTS") != "1",
                       reason="set ANNOFIX_SAM_TESTS=1 (downloads a SAM 2.1 checkpoint)"),
]


@pytest.fixture(scope="module")
def backend():
    from annofix.sam_backend import Sam2Backend
    return Sam2Backend("tiny")


def square_image():
    img = np.full((240, 320, 3), 30, np.uint8)
    img[60:180, 100:220] = (230, 200, 40)
    return img


def iou(a, b):
    return (a & b).sum() / (a | b).sum()


def test_click_segments_the_square(backend):
    backend.set_image(square_image())
    masks, scores, logits = backend.predict(points=[[160, 120]], labels=[1], multimask=True)
    assert masks.shape == (3, 240, 320) and masks.dtype == bool
    assert scores.shape == (3,) and logits.shape == (3, 256, 256)
    truth = np.zeros((240, 320), bool)
    truth[60:180, 100:220] = True
    assert max(iou(m, truth) for m in masks) > 0.85


def test_generate_returns_aligned_masks(backend):
    masks, scores = backend.generate(square_image(), points_per_side=8)
    assert masks.ndim == 3 and masks.shape[1:] == (240, 320) and masks.dtype == bool
    assert len(scores) == len(masks) > 0


def test_local_checkpoint_loads_with_matching_config():
    """The offline --checkpoint path: CONFIGS must pair with the 2.1 .pt files."""
    from huggingface_hub import hf_hub_download

    from annofix.sam_backend import Sam2Backend
    path = hf_hub_download("facebook/sam2.1-hiera-tiny", "sam2.1_hiera_tiny.pt")
    backend = Sam2Backend("tiny", checkpoint=path)
    backend.set_image(square_image())
    masks, _, _ = backend.predict(points=[[160, 120]], labels=[1])
    assert masks.shape == (3, 240, 320)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `ANNOFIX_SAM_TESTS=1 uv run pytest tests/test_sam_backend.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'annofix.sam_backend'`.

- [ ] **Step 3: Write `annofix/sam_backend.py`**

```python
"""SAM 2.1 wrapper. torch and sam2 are imported inside functions so the rest of AnnoFix
imports (and its tests run) without them."""
from __future__ import annotations

import contextlib
from pathlib import Path

import numpy as np

MODELS = {
    "tiny": "facebook/sam2.1-hiera-tiny",
    "small": "facebook/sam2.1-hiera-small",
    "base_plus": "facebook/sam2.1-hiera-base-plus",
    "large": "facebook/sam2.1-hiera-large",
}
# Hydra config names inside the installed sam2 package, for --checkpoint (offline) loading.
CONFIGS = {
    "tiny": "configs/sam2.1/sam2.1_hiera_t.yaml",
    "small": "configs/sam2.1/sam2.1_hiera_s.yaml",
    "base_plus": "configs/sam2.1/sam2.1_hiera_b+.yaml",
    "large": "configs/sam2.1/sam2.1_hiera_l.yaml",
}
# Set from the M0 measurements (docs/notes/2026-10-03-m0-measurements.md).
DEFAULT_MODEL = {"cuda": "base_plus", "cpu": "small"}


def pick_device(device: str | None) -> str:
    import torch

    if device:
        return device
    return "cuda" if torch.cuda.is_available() else "cpu"


class Sam2Backend:
    def __init__(self, model: str | None = None, device: str | None = None,
                 checkpoint: str | Path | None = None):
        import torch
        from sam2.build_sam import build_sam2, build_sam2_hf
        from sam2.sam2_image_predictor import SAM2ImagePredictor

        self.device = pick_device(device)
        self.model_name = model or DEFAULT_MODEL["cuda" if self.device == "cuda" else "cpu"]
        if self.model_name not in MODELS:
            raise ValueError(f"unknown SAM model {self.model_name!r}; choose from {sorted(MODELS)}")
        if self.device == "cpu":
            torch.set_num_threads(10)  # machine rule
        else:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        if checkpoint:
            self.model = build_sam2(CONFIGS[self.model_name], str(checkpoint), device=self.device)
        else:
            self.model = build_sam2_hf(MODELS[self.model_name], device=self.device)
        self.predictor = SAM2ImagePredictor(self.model)
        self._amg = None
        self._amg_grid = None

    def _context(self):
        import torch

        stack = contextlib.ExitStack()
        stack.enter_context(torch.inference_mode())
        if self.device == "cuda":
            stack.enter_context(torch.autocast("cuda", dtype=torch.bfloat16))
        return stack

    def set_image(self, image: np.ndarray) -> None:
        with self._context():
            self.predictor.set_image(image)

    def predict(self, points=None, labels=None, box=None, prev_logits=None, multimask=True):
        with self._context():
            masks, scores, logits = self.predictor.predict(
                point_coords=None if points is None else np.asarray(points, np.float32),
                point_labels=None if labels is None else np.asarray(labels, np.int32),
                box=None if box is None else np.asarray(box, np.float32),
                mask_input=prev_logits,
                multimask_output=multimask,
            )
        return masks > 0, np.asarray(scores, np.float32), np.asarray(logits, np.float32)

    def generate(self, image: np.ndarray, points_per_side: int = 32, points_per_batch: int = 64):
        from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

        if self._amg is None or self._amg_grid != (points_per_side, points_per_batch):
            # min_mask_region_area=0: hole/sprinkle cleanup needs the CUDA extension we don't
            # build; overlaps.remove_small_components does that job instead.
            self._amg = SAM2AutomaticMaskGenerator(
                self.model, points_per_side=points_per_side, points_per_batch=points_per_batch,
                box_nms_thresh=0.9, min_mask_region_area=0, output_mode="binary_mask",
                multimask_output=True)
            self._amg_grid = (points_per_side, points_per_batch)
        with self._context():
            records = self._amg.generate(image)
        if not records:
            return np.zeros((0, *image.shape[:2]), bool), np.zeros(0, np.float32)
        masks = np.stack([r["segmentation"] for r in records]).astype(bool)
        scores = np.array([r["predicted_iou"] for r in records], np.float32)
        return masks, scores
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `ANNOFIX_SAM_TESTS=1 uv run pytest tests/test_sam_backend.py -v`
Expected: PASS, 3 tests (first run downloads the tiny checkpoint, ~150 MB).
Also run: `uv run pytest -v` — the SAM tests show as SKIPPED without the env var, everything else passes.

- [ ] **Step 5: Write `scripts/spike_sam.py`**

```python
"""M0 spike: does SAM 2.1 run on this machine, how fast, how much memory?

uv run python scripts/spike_sam.py --image PATH [--models tiny small base_plus] [--devices cuda cpu] [--amg] [--qt]
"""
import annofix  # noqa: F401  (thread cap before numpy)

import argparse
import statistics
import time

import cv2


def median_time(fn, repeat, sync):
    fn()
    sync()  # warm-up
    times = []
    for _ in range(repeat):
        t = time.perf_counter()
        fn()
        sync()
        times.append(time.perf_counter() - t)
    return statistics.median(times)


def measure(model, device, image, amg):
    import torch

    from annofix.sam_backend import Sam2Backend

    sync = torch.cuda.synchronize if device == "cuda" else (lambda: None)
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    t = time.perf_counter()
    backend = Sam2Backend(model, device)
    load_s = time.perf_counter() - t
    h, w = image.shape[:2]
    encode = median_time(lambda: backend.set_image(image), 5 if device == "cuda" else 2, sync)
    decode = median_time(lambda: backend.predict(points=[[w // 2, h // 2]], labels=[1]), 20, sync)
    row = {"model": model, "device": device, "load_s": round(load_s, 1),
           "encode_ms": round(encode * 1000), "decode_ms": round(decode * 1000, 1)}
    if amg:
        t = time.perf_counter()
        masks, _ = backend.generate(image)
        sync()
        row["amg_s"] = round(time.perf_counter() - t, 1)
        row["amg_masks"] = len(masks)
    if device == "cuda":
        row["peak_vram_mb"] = round(torch.cuda.max_memory_allocated() / 2**20)
        del backend
        torch.cuda.empty_cache()
    return row


def qt_check():
    import torch  # noqa: F401  (same import order as the app: torch + cv2 before Qt)
    from PyQt6.QtCore import QTimer
    from PyQt6.QtWidgets import QApplication, QLabel

    app = QApplication([])
    label = QLabel("AnnoFix: Qt + torch + cv2 in one process")
    label.show()
    QTimer.singleShot(1500, app.quit)
    app.exec()
    print(f"Qt OK on platform '{app.platformName()}'")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--image", required=True)
    ap.add_argument("--models", nargs="+", default=["tiny", "small", "base_plus"])
    ap.add_argument("--devices", nargs="+", default=["cuda", "cpu"])
    ap.add_argument("--amg", action="store_true", help="also time automatic mask generation (CUDA only)")
    ap.add_argument("--qt", action="store_true", help="open a Qt window with torch + cv2 loaded")
    args = ap.parse_args()
    image = cv2.cvtColor(cv2.imread(args.image), cv2.COLOR_BGR2RGB)
    print(f"image {image.shape[1]}x{image.shape[0]}")
    for device in args.devices:
        for model in args.models:
            print(measure(model, device, image, args.amg and device == "cuda"), flush=True)
    if args.qt:
        qt_check()


if __name__ == "__main__":
    main()
```

- [ ] **Step 6: Run the spike**

Run (GPU, all models, with AMG): `uv run python scripts/spike_sam.py --image ../CamCalibration/MC-Calib/data/rosbag_mccalib_moving/frames/cam0_100_color.png --devices cuda --amg`
Run (CPU, interactive only): `uv run python scripts/spike_sam.py --image ../CamCalibration/MC-Calib/data/rosbag_mccalib_moving/frames/cam0_100_color.png --devices cpu --models tiny small base_plus`
Run (Qt coexistence on the real Wayland session): `uv run python scripts/spike_sam.py --image ../CamCalibration/MC-Calib/data/rosbag_mccalib_moving/frames/cam0_100_color.png --devices cuda --models tiny --qt`
Expected: one dict per model/device, then `Qt OK on platform 'wayland'` (or `xcb`). A crash or `OutOfMemoryError` is a finding — record it.

- [ ] **Step 7: Record measurements and decide defaults**

Write `docs/notes/2026-10-03-m0-measurements.md` with the printed rows copied verbatim into this table, plus the torch/CUDA versions from Task 1 Step 7:

```markdown
# M0 measurements (2026-10-03)

Machine: RTX 3050 Laptop 6 GB, driver 595.84; torch <version>, CUDA <version>; image 1280x720.

| model | device | load s | encode ms | decode ms | AMG s | AMG masks | peak VRAM MB |
|---|---|---|---|---|---|---|---|
| (one row per printed dict) |

Decision rule: CUDA default = base_plus if its encode < 500 ms and peak VRAM < 3000 MB, else small.
CPU default = small if its encode < 5000 ms, else tiny.
Decision: cuda -> <model>, cpu -> <model>. Qt check: <platform / problem>.
```

If the rule picks different models, update `DEFAULT_MODEL` in `annofix/sam_backend.py` to match.

- [ ] **Step 8: Commit**

```bash
git add annofix/sam_backend.py tests/test_sam_backend.py scripts/spike_sam.py docs/notes/2026-10-03-m0-measurements.md
git commit -m "feat: SAM 2.1 backend and M0 performance measurements" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 3: Label configuration

**Files:**
- Create: `annofix/config.py`, `tests/test_config.py`

**Interfaces:**
- Produces: `class Labels` (frozen dataclass: `materials: tuple[str, ...]`, `objects: dict[str, tuple[str, ...]]`, method `parts_for(cls: str | None) -> tuple[str, ...]`), `class LabelsError(ValueError)`, `load_labels(path: str | Path) -> Labels`.

- [ ] **Step 1: Write the failing test `tests/test_config.py`**

```python
from pathlib import Path

import pytest

from annofix.config import LabelsError, load_labels

REPO = Path(__file__).resolve().parents[1]


def write(tmp_path, text):
    p = tmp_path / "labels.yaml"
    p.write_text(text)
    return p


def test_example_labels_load():
    labels = load_labels(REPO / "labels.yaml")
    assert labels.materials[0] == "plastic"
    assert labels.parts_for("mug") == ("body", "handle")
    assert labels.parts_for(None) == () and labels.parts_for("unknown") == ()


def test_object_without_part_suggestions(tmp_path):
    labels = load_labels(write(tmp_path, "materials: [metal]\nobjects:\n  widget:\n"))
    assert labels.parts_for("widget") == ()


@pytest.mark.parametrize("text, message", [
    ("objects: {}\n", "materials"),
    ("materials: []\n", "at least one"),
    ("materials: [metal, metal]\n", "duplicate"),
    ("materials: [metal, 3]\n", "non-empty strings"),
    ("materials: [metal]\nobjects: [mug]\n", "mapping"),
    ("- just a list\n", "top level"),
])
def test_invalid_files_are_rejected(tmp_path, text, message):
    with pytest.raises(LabelsError, match=message):
        load_labels(write(tmp_path, text))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_config.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'annofix.config'`.

- [ ] **Step 3: Write `annofix/config.py`**

```python
"""labels.yaml: the material list and object classes with suggested part names."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


class LabelsError(ValueError):
    pass


@dataclass(frozen=True)
class Labels:
    materials: tuple[str, ...]
    objects: dict[str, tuple[str, ...]]  # class -> suggested part names

    def parts_for(self, cls: str | None) -> tuple[str, ...]:
        return self.objects.get(cls, ()) if cls else ()


def _names(value, where: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not all(isinstance(v, str) and v.strip() for v in value):
        raise LabelsError(f"{where}: expected a list of non-empty strings, got {value!r}")
    names = tuple(v.strip() for v in value)
    dupes = sorted({n for n in names if names.count(n) > 1})
    if dupes:
        raise LabelsError(f"{where}: duplicate names {dupes}")
    return names


def load_labels(path: str | Path) -> Labels:
    data = yaml.safe_load(Path(path).read_text())
    if not isinstance(data, dict):
        raise LabelsError(f"{path}: top level must be a mapping with 'materials' and 'objects'")
    materials = _names(data.get("materials"), "materials")
    if not materials:
        raise LabelsError("materials: need at least one material")
    raw_objects = data.get("objects") or {}
    if not isinstance(raw_objects, dict):
        raise LabelsError("objects: expected a mapping of class -> list of part names")
    objects = {str(cls): _names(parts or [], f"objects.{cls}") for cls, parts in raw_objects.items()}
    return Labels(materials=materials, objects=objects)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_config.py -v`
Expected: PASS (8 tests).

- [ ] **Step 5: Commit**

```bash
git add annofix/config.py tests/test_config.py
git commit -m "feat: labels.yaml loading and validation" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 4: Annotation model — hierarchy, labels, transactions, undo/redo

**Files:**
- Create: `annofix/annotation.py`, `tests/test_annotation_core.py`

**Interfaces:**
- Produces (used by every later task and Plans 2–3):
  - constants `SCHEMA_VERSION = 1`, `REVIEW_STATUSES = ("unreviewed", "reviewed", "flagged")`, `MAX_PART_ID = 65535`
  - dataclasses `Part(id: int, name: str | None = None, material: str | None = None)`, `AnnObject(id: int, cls: str | None = None, parts: list[Part] = [])`
  - `objects_from_dicts(items: list[dict]) -> list[AnnObject]`
  - `class Annotation(label_map, objects=None, *, next_id=1, image="", source="manual", annotator=None, review_status="unreviewed", history_bytes=200 * 2**20)` with
    - fields `label_map`, `objects`, `next_id`, `image`, `source`, `annotator`, `review_status`, `warnings: list[str]`
    - `Annotation.empty(height, width, **kw)`, `Annotation.from_meta(label_map, meta: dict)`
    - `size -> tuple[int, int]` (H, W), `object(id)`, `part(id)`, `object_of(part_id)`, `part_ids()`, `part_at(y, x) -> int | None`, `part_mask(id)`, `part_area(id)`
    - `meta_dict() -> dict` (the JSON document minus nothing — see spec format)
    - `new_object(cls=None) -> int`, `new_part(object_id, name=None, material=None) -> int`
    - `set_object_class`, `set_part_name`, `set_part_material`, `set_review_status`
    - `transaction(label: str)` context manager; `undo() -> str | None`, `redo() -> str | None`, `can_undo()`, `can_redo()`, `dirty: bool`, `mark_saved()`
  - Task 5 adds the pixel/structural ops to this class.

- [ ] **Step 1: Write the failing test `tests/test_annotation_core.py`**

```python
import numpy as np
import pytest

from annofix.annotation import MAX_PART_ID, Annotation, Part


def meta_wo_next_id(ann):
    d = ann.meta_dict()
    d.pop("next_id")
    return d


def test_rejects_non_uint16_maps():
    with pytest.raises(ValueError):
        Annotation(np.zeros((4, 4), np.int32))


def test_ids_are_shared_and_sequential():
    ann = Annotation.empty(4, 4)
    oid = ann.new_object("mug")
    a = ann.new_part(oid, "body")
    b = ann.new_part(oid)
    assert (oid, a, b, ann.next_id) == (1, 2, 3, 4)
    assert ann.part(a) == Part(2, "body", None)
    assert ann.object_of(b).id == oid
    assert ann.size == (4, 4)


def test_meta_dict_is_json_shaped():
    ann = Annotation.empty(3, 5, image="img.png")
    ann.new_part(ann.new_object())
    d = ann.meta_dict()
    assert d["schema_version"] == 1 and d["size"] == [3, 5] and d["image"] == "img.png"
    assert d["objects"] == [{"id": 1, "class": None, "parts": [{"id": 2, "name": None, "material": None}]}]
    assert d["review_status"] == "unreviewed" and d["next_id"] == 3


def test_label_setters_and_review_status():
    ann = Annotation.empty(4, 4)
    oid = ann.new_object()
    pid = ann.new_part(oid)
    ann.set_object_class(oid, "mug")
    ann.set_part_name(pid, "handle")
    ann.set_part_material(pid, "ceramic")
    assert ann.object(oid).cls == "mug" and ann.part(pid) == Part(pid, "handle", "ceramic")
    ann.set_part_name(pid, "")
    assert ann.part(pid).name is None
    ann.set_review_status("reviewed")
    assert ann.review_status == "reviewed"
    with pytest.raises(ValueError):
        ann.set_review_status("done")


def test_undo_redo_restore_exact_state():
    ann = Annotation.empty(6, 6)
    oid = ann.new_object("mug")
    pid = ann.new_part(oid, "body", "ceramic")
    states = []

    def snap():
        states.append((ann.label_map.copy(), meta_wo_next_id(ann)))

    snap()
    with ann.transaction("paint"):
        ann.label_map[1:4, 1:4] = pid
    snap()
    ann.set_part_material(pid, "glass")
    snap()
    ann.set_object_class(oid, "cup")
    snap()
    for i in (2, 1, 0):
        assert ann.undo() is not None
        m, d = states[i]
        assert np.array_equal(ann.label_map, m) and meta_wo_next_id(ann) == d
    for i in (1, 2, 3):
        assert ann.redo() is not None
        m, d = states[i]
        assert np.array_equal(ann.label_map, m) and meta_wo_next_id(ann) == d


def test_nested_transactions_are_one_undo_step():
    ann = Annotation.empty(4, 4)
    pid = ann.new_part(ann.new_object())
    with ann.transaction("stroke"):
        with ann.transaction("dab"):
            ann.label_map[0, 0] = pid
        ann.set_part_name(pid, "x")
        ann.label_map[1, 1] = pid
    assert ann.undo() == "stroke"
    assert not ann.label_map.any() and ann.part(pid).name is None


def test_exception_rolls_back_and_records_nothing():
    ann = Annotation.empty(4, 4)
    pid = ann.new_part(ann.new_object())
    with pytest.raises(RuntimeError):
        with ann.transaction("bad"):
            ann.label_map[0, 0] = pid
            ann.set_part_material(pid, "metal")
            raise RuntimeError("boom")
    assert not ann.label_map.any() and ann.part(pid).material is None
    assert ann.undo() == "new part"  # the failed transaction left no undo entry


def test_new_edit_clears_redo():
    ann = Annotation.empty(4, 4)
    pid = ann.new_part(ann.new_object())
    ann.set_part_name(pid, "a")
    ann.undo()
    assert ann.can_redo()
    ann.set_part_name(pid, "b")
    assert not ann.can_redo()


def test_noop_edit_is_not_recorded():
    ann = Annotation.empty(4, 4)
    pid = ann.new_part(ann.new_object())
    ann.set_part_name(pid, "a")
    ann.set_part_name(pid, "a")
    assert ann.undo() == "set name"
    assert ann.part(pid).name is None


def test_ids_never_reused_after_undo():
    ann = Annotation.empty(4, 4)
    oid = ann.new_object()
    first = ann.new_part(oid)
    ann.undo()
    assert ann.new_part(oid) == first + 1


def test_dirty_tracking():
    ann = Annotation.empty(4, 4)
    assert not ann.dirty
    ann.new_object()
    assert ann.dirty
    ann.mark_saved()
    assert not ann.dirty
    ann.undo()
    assert ann.dirty
    ann.redo()
    assert not ann.dirty


def test_history_is_capped_by_bytes():
    ann = Annotation.empty(100, 100, history_bytes=50_000)
    pid = ann.new_part(ann.new_object())
    for v in (pid, 0, pid):  # each full-map change is a 40 kB entry
        with ann.transaction("fill"):
            ann.label_map[:] = v
    undos = 0
    while ann.undo():
        undos += 1
    assert undos == 1


def test_undo_inside_transaction_rejected():
    ann = Annotation.empty(4, 4)
    ann.new_object()
    with pytest.raises(RuntimeError):
        with ann.transaction("t"):
            ann.undo()


def test_id_space_exhaustion_is_an_error():
    ann = Annotation.empty(4, 4, next_id=MAX_PART_ID + 1)
    with pytest.raises(OverflowError):
        ann.new_object()


def test_from_meta_rejects_bad_review_status():
    with pytest.raises(ValueError):
        Annotation.from_meta(np.zeros((2, 2), np.uint16), {"review_status": "done"})
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_annotation_core.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'annofix.annotation'`.

- [ ] **Step 3: Write `annofix/annotation.py`**

```python
"""In-memory annotation: a uint16 part-id label map plus the object -> part hierarchy.

Every pixel belongs to at most one part (0 = background). Every mutating method runs inside
a transaction, which records a bbox-cropped diff of the label map plus a metadata snapshot,
so any edit can be undone exactly. The label map is only ever modified in place.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field

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
    meta_before: dict
    meta_after: dict

    @property
    def nbytes(self) -> int:
        return 0 if self.before is None else self.before.nbytes + self.after.nbytes


class Annotation:
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

    @contextmanager
    def transaction(self, label: str):
        """Group edits into one undo step. Nested transactions merge into the outermost.
        If the block raises, the label map and metadata are restored and nothing is recorded."""
        if self._tx_depth:
            self._tx_depth += 1
            try:
                yield
            finally:
                self._tx_depth -= 1
            return
        map_before = self.label_map.copy()
        meta_before = self.meta_dict()
        self._tx_depth = 1
        try:
            yield
        except BaseException:
            self.label_map[...] = map_before
            self._restore_meta(meta_before)
            raise
        finally:
            self._tx_depth = 0
        self._record(label, map_before, meta_before)

    def _record(self, label: str, map_before: np.ndarray, meta_before: dict) -> None:
        meta_after = self.meta_dict()
        changed = map_before != self.label_map
        if changed.any():
            rows = np.flatnonzero(changed.any(axis=1))
            cols = np.flatnonzero(changed.any(axis=0))
            y0, y1, x0, x1 = int(rows[0]), int(rows[-1]) + 1, int(cols[0]), int(cols[-1]) + 1
            entry = _Entry(label, (y0, y1, x0, x1), map_before[y0:y1, x0:x1].copy(),
                           self.label_map[y0:y1, x0:x1].copy(), meta_before, meta_after)
        elif meta_after != meta_before:
            entry = _Entry(label, None, None, None, meta_before, meta_after)
        else:
            return
        self._undo.append(entry)
        self._redo.clear()
        total = sum(e.nbytes for e in self._undo)
        while total > self._history_bytes and len(self._undo) > 1:
            total -= self._undo.pop(0).nbytes

    def _apply(self, bbox, crop, meta) -> None:
        if bbox is not None:
            y0, y1, x0, x1 = bbox
            self.label_map[y0:y1, x0:x1] = crop
        self._restore_meta(meta)

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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_annotation_core.py -v`
Expected: PASS (15 tests).

- [ ] **Step 5: Commit**

```bash
git add annofix/annotation.py tests/test_annotation_core.py
git commit -m "feat: annotation model with hierarchy, transactions and exact undo" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 5: Annotation model — pixel ownership and structural edits

**Files:**
- Modify: `annofix/annotation.py` (add `import cv2` at the top and the methods below to `class Annotation`)
- Create: `tests/test_annotation_ops.py`

**Interfaces:**
- Consumes: Task 4 `Annotation`.
- Produces (used by Task 6 reconcile, Plan 2 tools, Plan 3 SAM accept):
  - `assign(mask: bool[H,W], part_id, protect_others=False) -> int` (pixels changed)
  - `erase(mask, part_id=None) -> int` (pixels erased)
  - `replace_part_mask(part_id, mask, protect_others=False) -> None`
  - `delete_part(part_id)`, `delete_object(object_id)`, `merge_parts(keep_id, other_id)`, `move_part(part_id, object_id)`, `promote_part(part_id) -> int`
  - `clean_part(part_id, min_island_px=64, max_hole_px=2000) -> None`
  - `validate() -> list[str]` (exact message formats below)

- [ ] **Step 1: Write the failing test `tests/test_annotation_ops.py`**

```python
import numpy as np
import pytest

from annofix.annotation import Annotation


def rect(y0, y1, x0, x1, shape=(10, 10)):
    m = np.zeros(shape, bool)
    m[y0:y1, x0:x1] = True
    return m


@pytest.fixture
def mug():
    ann = Annotation.empty(10, 10)
    oid = ann.new_object("mug")
    a = ann.new_part(oid, "body", "ceramic")
    b = ann.new_part(oid, "handle", "ceramic")
    ann.assign(rect(0, 5, 0, 5), a)
    ann.assign(rect(5, 10, 5, 10), b)
    return ann, oid, a, b


def test_assign_steals_from_other_parts_by_default(mug):
    ann, _, a, b = mug
    assert ann.assign(rect(3, 7, 3, 7), a) == 12
    assert ann.part_area(a) == 37 and ann.part_area(b) == 21
    assert ann.part_at(6, 6) == a


def test_assign_protect_only_fills_background(mug):
    ann, _, a, b = mug
    assert ann.assign(rect(3, 7, 3, 7), a, protect_others=True) == 8
    assert ann.part_area(a) == 33 and ann.part_area(b) == 25
    assert ann.part_at(6, 6) == b


def test_erase_only_touches_given_part(mug):
    ann, _, a, b = mug
    assert ann.erase(np.ones((10, 10), bool), a) == 25
    assert ann.part_area(a) == 0 and ann.part_area(b) == 25


def test_erase_all_parts_in_mask(mug):
    ann, _, a, b = mug
    assert ann.erase(rect(4, 6, 4, 6)) == 2
    assert ann.part_area(a) == 24 and ann.part_area(b) == 24


def test_replace_part_mask_moves_the_part(mug):
    ann, _, a, b = mug
    ann.replace_part_mask(a, rect(0, 2, 0, 10))
    assert ann.part_area(a) == 20 and ann.part_at(3, 3) is None and ann.part_area(b) == 25


def test_replace_part_mask_protect_keeps_neighbours(mug):
    ann, _, a, b = mug
    ann.replace_part_mask(a, np.ones((10, 10), bool), protect_others=True)
    assert ann.part_area(a) == 75 and ann.part_area(b) == 25


def test_pixel_op_is_one_undo_step(mug):
    ann, _, a, _ = mug
    before = ann.label_map.copy()
    ann.replace_part_mask(a, rect(0, 2, 0, 10))
    ann.undo()
    assert np.array_equal(ann.label_map, before)


def test_merge_parts(mug):
    ann, _, a, b = mug
    ann.merge_parts(a, b)
    assert ann.part_area(a) == 50 and ann.part_ids() == [a]


def test_merge_with_itself_rejected(mug):
    ann, _, a, _ = mug
    with pytest.raises(ValueError):
        ann.merge_parts(a, a)


def test_delete_part_clears_pixels(mug):
    ann, _, a, b = mug
    ann.delete_part(a)
    assert ann.part_area(a) == 0 and ann.part_ids() == [b]


def test_delete_object_clears_all_its_parts(mug):
    ann, oid, _, _ = mug
    ann.delete_object(oid)
    assert ann.objects == [] and not ann.label_map.any()


def test_move_and_promote(mug):
    ann, oid, a, b = mug
    other = ann.new_object("cup")
    ann.move_part(b, other)
    assert ann.object_of(b).id == other
    new = ann.promote_part(a)
    assert ann.object_of(a).id == new and ann.object(new).cls is None
    assert ann.object(oid).parts == []
    assert ann.part_area(a) == 25  # pixels untouched


def test_unknown_part_rejected(mug):
    ann, *_ = mug
    with pytest.raises(KeyError):
        ann.assign(rect(0, 1, 0, 1), 999)


def test_bad_mask_rejected(mug):
    ann, _, a, _ = mug
    with pytest.raises(ValueError):
        ann.assign(np.ones((10, 10), np.uint8), a)
    with pytest.raises(ValueError):
        ann.assign(np.ones((5, 5), bool), a)


def test_clean_part_drops_islands_and_fills_holes():
    ann = Annotation.empty(20, 20)
    a = ann.new_part(ann.new_object())
    m = rect(2, 12, 2, 12, (20, 20))
    m[6, 6] = False   # 1 px hole
    m[17, 17] = True  # 1 px island
    ann.assign(m, a)
    ann.clean_part(a, min_island_px=4, max_hole_px=10)
    assert ann.part_at(6, 6) == a and ann.part_at(17, 17) is None
    assert ann.part_area(a) == 100


def test_clean_part_leaves_enclosed_other_parts():
    ann = Annotation.empty(20, 20)
    oid = ann.new_object()
    a, b = ann.new_part(oid), ann.new_part(oid)
    ann.assign(rect(2, 12, 2, 12, (20, 20)), a)
    ann.assign(rect(6, 7, 6, 7, (20, 20)), b)
    ann.clean_part(a, min_island_px=4, max_hole_px=10)
    assert ann.part_at(6, 6) == b


def test_clean_part_keeps_large_holes():
    ann = Annotation.empty(20, 20)
    a = ann.new_part(ann.new_object())
    m = rect(2, 18, 2, 18, (20, 20))
    m[5:15, 5:15] = False  # 100 px hole
    ann.assign(m, a)
    ann.clean_part(a, min_island_px=4, max_hole_px=10)
    assert ann.part_at(10, 10) is None


def test_validate_lists_missing_labels():
    ann = Annotation.empty(10, 10)
    oid = ann.new_object()
    pid = ann.new_part(oid)
    problems = ann.validate()
    assert f"object {oid}: no class" in problems
    assert f"part {pid} (object {oid}): no name" in problems
    assert f"part {pid} (object {oid}): no material" in problems
    assert f"part {pid} (object {oid}): empty mask" in problems


def test_validate_ok_when_complete(mug):
    ann, *_ = mug
    assert ann.validate() == []


def test_validate_flags_object_without_parts():
    ann = Annotation.empty(4, 4)
    oid = ann.new_object("mug")
    assert ann.validate() == [f"object {oid}: no parts"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_annotation_ops.py -v`
Expected: FAIL — `AttributeError: 'Annotation' object has no attribute 'assign'`.

- [ ] **Step 3: Add the methods to `class Annotation` in `annofix/annotation.py`** (and `import cv2` next to `import numpy as np`)

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_annotation_ops.py tests/test_annotation_core.py -v`
Expected: PASS (all).

- [ ] **Step 5: Commit**

```bash
git add annofix/annotation.py tests/test_annotation_ops.py
git commit -m "feat: pixel ownership, structural edits, clean-up and validation" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 6: On-disk format (save, load, reconcile, images)

**Files:**
- Create: `annofix/io.py`, `tests/test_io.py`
- Modify: `annofix/annotation.py` (add `reconcile()` to `class Annotation`)

**Interfaces:**
- Consumes: Tasks 4–5 `Annotation`, `AnnObject`, `Part`, `SCHEMA_VERSION`.
- Produces (used by Task 8, scripts, Plans 2–3):
  - `Annotation.reconcile() -> list[str]`
  - `IMAGE_EXTS`, `list_images(images_dir) -> list[Path]`, `load_image(path) -> np.ndarray` (RGB uint8 HxWx3)
  - `annotation_paths(ann_dir, stem) -> tuple[Path, Path]` (png, json), `has_annotation(ann_dir, stem) -> bool`
  - `save_annotation(ann, ann_dir, stem) -> None` (marks `ann` saved)
  - `load_annotation(ann_dir, stem, expected_size: tuple[int, int] | None = None) -> Annotation` (`expected_size` is (H, W))
  - `load(ann_dir, stem) -> tuple[np.ndarray, dict]` (downstream helper)
  - private `_atomic_write(path: Path, data: bytes)` (tests monkeypatch it)

- [ ] **Step 1: Write the failing test `tests/test_io.py`**

```python
import json

import cv2
import numpy as np
import pytest

import annofix.io as aio
from annofix.annotation import AnnObject, Annotation, Part
from annofix.io import (annotation_paths, has_annotation, list_images, load, load_annotation,
                        load_image, save_annotation)


def big_id_ann():
    lm = np.zeros((4, 6), np.uint16)
    lm[0, :2], lm[1, :2], lm[2, :2], lm[3, :2] = 255, 256, 32768, 65535
    objs = [AnnObject(1, "mug", [Part(255, "body", "ceramic"), Part(256)]),
            AnnObject(2, None, [Part(32768), Part(65535)])]
    return Annotation(lm, objs, next_id=65536, image="img.png")


def test_roundtrip_preserves_uint16_ids_and_meta(tmp_path):
    ann = big_id_ann()
    save_annotation(ann, tmp_path, "img")
    back = load_annotation(tmp_path, "img")
    assert back.label_map.dtype == np.uint16
    assert np.array_equal(back.label_map, ann.label_map)
    assert back.meta_dict() == ann.meta_dict()
    assert back.warnings == []


def test_save_writes_both_files_and_no_temp_files(tmp_path):
    save_annotation(big_id_ann(), tmp_path, "img")
    png, js = annotation_paths(tmp_path, "img")
    assert (png.name, js.name) == ("img.parts.png", "img.json")
    assert sorted(p.name for p in tmp_path.iterdir()) == ["img.json", "img.parts.png"]
    d = json.loads(js.read_text())
    assert d["schema_version"] == 1 and d["objects"][1]["class"] is None
    assert has_annotation(tmp_path, "img") and not has_annotation(tmp_path, "other")


def test_save_marks_clean(tmp_path):
    ann = Annotation.empty(4, 4)
    ann.new_object()
    assert ann.dirty
    save_annotation(ann, tmp_path, "x")
    assert not ann.dirty


def test_unserialisable_meta_leaves_existing_files_untouched(tmp_path):
    ann = big_id_ann()
    save_annotation(ann, tmp_path, "img")
    png, _ = annotation_paths(tmp_path, "img")
    before = png.read_bytes()
    ann.label_map[:] = 0
    ann.annotator = object()
    with pytest.raises(TypeError):
        save_annotation(ann, tmp_path, "img")
    assert png.read_bytes() == before


def test_failed_json_write_still_loads(tmp_path, monkeypatch):
    ann = Annotation.empty(4, 4)
    oid = ann.new_object("mug")
    save_annotation(ann, tmp_path, "img")
    pid = ann.new_part(oid)
    m = np.zeros((4, 4), bool)
    m[0, 0] = True
    ann.assign(m, pid)
    real = aio._atomic_write

    def flaky(path, data):
        if path.suffix == ".json":
            raise OSError("disk full")
        real(path, data)

    monkeypatch.setattr(aio, "_atomic_write", flaky)
    with pytest.raises(OSError):
        save_annotation(ann, tmp_path, "img")
    assert ann.dirty
    monkeypatch.undo()
    back = load_annotation(tmp_path, "img")  # new PNG + old JSON
    assert back.part_at(0, 0) == pid
    assert any("orphan" in w for w in back.warnings)


def test_size_mismatch_with_image_is_an_error(tmp_path):
    save_annotation(Annotation.empty(4, 6), tmp_path, "img")
    with pytest.raises(ValueError, match="image is"):
        load_annotation(tmp_path, "img", expected_size=(6, 4))


def test_json_size_must_match_png(tmp_path):
    save_annotation(Annotation.empty(4, 6), tmp_path, "img")
    _, js = annotation_paths(tmp_path, "img")
    d = json.loads(js.read_text())
    d["size"] = [5, 6]
    js.write_text(json.dumps(d))
    with pytest.raises(ValueError, match="JSON says"):
        load_annotation(tmp_path, "img")


def test_unknown_schema_version_rejected(tmp_path):
    save_annotation(Annotation.empty(2, 2), tmp_path, "img")
    _, js = annotation_paths(tmp_path, "img")
    d = json.loads(js.read_text())
    d["schema_version"] = 2
    js.write_text(json.dumps(d))
    with pytest.raises(ValueError, match="schema_version"):
        load_annotation(tmp_path, "img")


def test_orphans_and_empty_parts_reconciled(tmp_path):
    lm = np.zeros((4, 4), np.uint16)
    lm[0, 0], lm[3, 3] = 2, 7
    save_annotation(Annotation(lm, [AnnObject(1, "mug", [Part(2), Part(3)])], next_id=4), tmp_path, "img")
    back = load_annotation(tmp_path, "img")
    assert back.object_of(7).cls is None and back.object_of(7).id == 8
    assert back.next_id == 9
    assert any("orphan" in w for w in back.warnings)
    assert any("part 3" in w and "no pixels" in w for w in back.warnings)
    assert not back.dirty


def test_duplicate_ids_rejected(tmp_path):
    save_annotation(Annotation(np.zeros((2, 2), np.uint16), [AnnObject(1, None, [Part(1)])], next_id=2),
                    tmp_path, "img")
    with pytest.raises(ValueError, match="duplicate"):
        load_annotation(tmp_path, "img")


def test_load_helper(tmp_path):
    save_annotation(big_id_ann(), tmp_path, "img")
    lm, meta = load(tmp_path, "img")
    assert lm.dtype == np.uint16 and meta["objects"][0]["parts"][0]["id"] == 255


def test_list_images_skips_depth_and_outputs(tmp_path):
    for name in ["cam0_000_color.png", "cam0_000_depth.png", "Depth-01.png", "notes.txt",
                 "x.parts.png", "mug.JPG"]:
        (tmp_path / name).write_bytes(b"")
    (tmp_path / "sub.png").mkdir()
    assert [p.name for p in list_images(tmp_path)] == ["cam0_000_color.png", "mug.JPG"]


def test_load_image_returns_rgb(tmp_path):
    bgr = np.zeros((2, 3, 3), np.uint8)
    bgr[..., 0] = 255  # pure blue, stored BGR
    cv2.imwrite(str(tmp_path / "c.png"), bgr)
    rgb = load_image(tmp_path / "c.png")
    assert rgb.shape == (2, 3, 3) and tuple(rgb[0, 0]) == (0, 0, 255)


def test_load_image_accepts_grey_rejects_depth(tmp_path):
    cv2.imwrite(str(tmp_path / "g.png"), np.full((2, 2), 9, np.uint8))
    assert load_image(tmp_path / "g.png").shape == (2, 2, 3)
    cv2.imwrite(str(tmp_path / "d.png"), np.full((2, 2), 900, np.uint16))
    with pytest.raises(ValueError, match="depth"):
        load_image(tmp_path / "d.png")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_io.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'annofix.io'`.

- [ ] **Step 3: Add `reconcile()` to `class Annotation`** (in `annofix/annotation.py`, after `validate`)

```python
    def reconcile(self) -> list[str]:
        """Repair label-map / hierarchy mismatches after loading a file. Not undoable: it
        describes the file, not an edit. Returns (and appends to self.warnings) what it did."""
        warnings = []
        obj_ids = [o.id for o in self.objects]
        part_ids = self.part_ids()
        all_ids = obj_ids + part_ids
        if len(set(all_ids)) != len(all_ids):
            raise ValueError("duplicate object/part ids in annotation JSON")
        counts = np.bincount(self.label_map.ravel())
        present = (np.flatnonzero(counts[1:]) + 1).tolist()
        highest = max(all_ids + present, default=0)
        if self.next_id <= highest:
            warnings.append(f"next_id {self.next_id} <= highest id {highest}; raised to {highest + 1}")
            self.next_id = highest + 1
        orphans = sorted(set(present) - set(part_ids))
        if orphans:
            obj = AnnObject(self._alloc_id(), None, [Part(i) for i in orphans])
            self.objects.append(obj)
            warnings.append(f"orphan part ids {orphans} (pixels without a JSON entry) "
                            f"grouped into new object {obj.id}")
        for o in self.objects:
            for p in o.parts:
                if p.id >= len(counts) or counts[p.id] == 0:
                    warnings.append(f"part {p.id} (object {o.id}) has no pixels")
        self.warnings.extend(warnings)
        return warnings
```

- [ ] **Step 4: Write `annofix/io.py`**

```python
"""Images and annotation files.

For image `<stem>` the annotation is `<stem>.parts.png` (uint16 part-id map, 0 = background)
plus `<stem>.json` (hierarchy and labels) in the annotations directory.
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path

import cv2
import numpy as np

from .annotation import SCHEMA_VERSION, Annotation

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def list_images(images_dir: str | Path) -> list[Path]:
    """Colour images in `images_dir` (not recursive), sorted by name. Skips depth frames
    (a 'depth' token in the name, e.g. cam0_000_depth.png) and AnnoFix's own outputs."""
    out = []
    for p in sorted(Path(images_dir).iterdir()):
        if not p.is_file() or p.suffix.lower() not in IMAGE_EXTS or p.name.endswith(".parts.png"):
            continue
        if "depth" in re.split(r"[_\-. ]+", p.stem.lower()):
            continue
        out.append(p)
    return out


def load_image(path: str | Path) -> np.ndarray:
    """RGB uint8 HxWx3. 8-bit grey is expanded; 16-bit (depth) images are rejected."""
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(f"cannot read image {path}")
    if img.dtype != np.uint8:
        raise ValueError(f"{path}: {img.dtype} image, expected 8-bit colour (a depth map?)")
    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    code = {3: cv2.COLOR_BGR2RGB, 4: cv2.COLOR_BGRA2RGB}.get(img.shape[2])
    if code is None:
        raise ValueError(f"{path}: {img.shape[2]} channels, expected 3 or 4")
    return cv2.cvtColor(img, code)


def annotation_paths(ann_dir: str | Path, stem: str) -> tuple[Path, Path]:
    d = Path(ann_dir)
    return d / f"{stem}.parts.png", d / f"{stem}.json"


def has_annotation(ann_dir: str | Path, stem: str) -> bool:
    return annotation_paths(ann_dir, stem)[1].exists()  # JSON is written last


def _atomic_write(path: Path, data: bytes) -> None:
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def save_annotation(ann: Annotation, ann_dir: str | Path, stem: str) -> None:
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
    """Load and reconcile. `expected_size` is the image's (H, W); a mismatch is an error."""
    png_path, json_path = annotation_paths(ann_dir, stem)
    meta = json.loads(json_path.read_text())
    version = meta.get("schema_version")
    if version != SCHEMA_VERSION:
        raise ValueError(f"{json_path}: schema_version {version!r}; this AnnoFix reads {SCHEMA_VERSION}")
    label_map = cv2.imread(str(png_path), cv2.IMREAD_UNCHANGED)
    if label_map is None:
        raise ValueError(f"cannot read {png_path}")
    if label_map.dtype != np.uint16 or label_map.ndim != 2:
        raise ValueError(f"{png_path}: expected single-channel uint16, got {label_map.dtype} {label_map.shape}")
    if list(label_map.shape) != list(meta.get("size", label_map.shape)):
        raise ValueError(f"{stem}: label map is {list(label_map.shape)} but the JSON says size {meta['size']}")
    if expected_size is not None and tuple(label_map.shape) != tuple(expected_size):
        h, w = label_map.shape
        raise ValueError(f"{stem}: annotation is {w}x{h} but the image is {expected_size[1]}x{expected_size[0]}")
    ann = Annotation.from_meta(label_map, meta)
    ann.reconcile()
    return ann


def load(ann_dir: str | Path, stem: str) -> tuple[np.ndarray, dict]:
    """Downstream helper: (uint16 part-id map, metadata dict) for one image."""
    ann = load_annotation(ann_dir, stem)
    return ann.label_map, ann.meta_dict()
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/test_io.py tests/test_annotation_core.py tests/test_annotation_ops.py -v`
Expected: PASS (all).

- [ ] **Step 6: Commit**

```bash
git add annofix/io.py annofix/annotation.py tests/test_io.py
git commit -m "feat: atomic annotation save/load with reconcile, image listing" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 7: Overlapping candidates → single-ownership objects and parts

**Files:**
- Create: `annofix/overlaps.py`, `tests/test_overlaps.py`

**Interfaces:**
- Consumes: Task 4 `Annotation`, `AnnObject`, `Part`.
- Produces (used by Task 8 and a future pipeline importer):
  - `GroupingParams` (frozen dataclass; fields and defaults below)
  - `resolve_overlaps(masks: bool[N,H,W]) -> int32[H,W]` (mask index + 1, 0 = none; smaller masks win)
  - `remove_small_components(mask: bool[H,W], min_px: int) -> bool[H,W]`
  - `filter_candidates(masks, p) -> int ndarray of kept indices`
  - `pairwise_intersections(masks) -> int64[N,N]` (diagonal = areas)
  - `dedupe(scores, inter, iou_thresh) -> list[int]` (sorted kept indices)
  - `find_parents(inter, p) -> int ndarray` (-1 = root)
  - `group_objects(parent, areas, p) -> list[tuple[int, list[int]]]`
  - `paint_groups(masks, groups, p) -> Annotation`
  - `candidates_to_annotation(masks, scores, p=GroupingParams(), image="") -> Annotation`

- [ ] **Step 1: Write the failing test `tests/test_overlaps.py`**

```python
import numpy as np

from annofix.overlaps import (GroupingParams, candidates_to_annotation, dedupe, filter_candidates,
                              find_parents, group_objects, pairwise_intersections,
                              remove_small_components, resolve_overlaps)


def rect(y0, y1, x0, x1, shape=(100, 100)):
    m = np.zeros(shape, bool)
    m[y0:y1, x0:x1] = True
    return m


def mug_scene():
    body = rect(20, 60, 20, 52)    # 1280 px
    handle = rect(25, 55, 52, 62)  # 300 px
    mug = body | handle            # 1580 px: the whole object
    grip = rect(30, 35, 25, 45)    # 100 px inside the body: a sub-part
    box = rect(70, 90, 70, 90)     # 400 px: a separate object
    table = rect(60, 100, 0, 100)  # 4000 px: background
    speck = rect(5, 8, 90, 93)     # 9 px
    return np.stack([table, mug, body, handle, grip, box, speck])


def test_resolve_overlaps_smaller_wins():
    out = resolve_overlaps(np.stack([rect(0, 60, 0, 60), rect(10, 20, 10, 20)]))
    assert out[15, 15] == 2 and out[50, 50] == 1 and out[80, 80] == 0


def test_filter_drops_tiny_huge_and_border_spanning():
    masks = np.stack([rect(40, 45, 40, 45),   # 25 px: too small
                      rect(0, 60, 20, 80),    # 3600 px > 25% of image: background
                      rect(80, 90, 0, 100),   # 1000 px spanning left+right: table edge
                      rect(40, 60, 40, 60),   # normal object
                      rect(0, 20, 0, 20)])    # small object in a corner: kept
    assert filter_candidates(masks, GroupingParams(min_area=100)).tolist() == [3, 4]


def test_pairwise_intersections_match_brute_force():
    rng = np.random.default_rng(0)
    ms = []
    for _ in range(12):
        y, x = rng.integers(0, 80, 2)
        h, w = rng.integers(5, 20, 2)
        ms.append(rect(y, y + h, x, x + w))
    masks = np.stack(ms)
    flat = masks.reshape(12, -1).astype(np.int64)
    assert np.array_equal(pairwise_intersections(masks), flat @ flat.T)


def test_dedupe_keeps_best_scoring_duplicate():
    masks = np.stack([rect(10, 30, 10, 30), rect(10, 30, 10, 31), rect(50, 70, 50, 70)])
    assert dedupe(np.array([0.8, 0.9, 0.5]), pairwise_intersections(masks), 0.9) == [1, 2]


def test_find_parents_picks_smallest_container():
    masks = mug_scene()[1:5]  # mug, body, handle, grip
    assert find_parents(pairwise_intersections(masks), GroupingParams()).tolist() == [-1, 0, 0, 1]


def test_group_objects_keeps_depth_one_parts_only():
    masks = mug_scene()[1:5]
    inter = pairwise_intersections(masks)
    p = GroupingParams()
    assert group_objects(find_parents(inter, p), np.diag(inter), p) == [(0, [1, 2])]


def test_lone_near_complete_child_collapses():
    assert group_objects(np.array([-1, 0]), np.array([100, 95]), GroupingParams()) == [(0, [])]


def test_candidates_to_annotation_builds_objects_and_parts():
    masks = mug_scene()
    p = GroupingParams(min_area=20, min_component_px=10)
    ann = candidates_to_annotation(masks, np.full(len(masks), 0.9), p, image="scene.png")
    areas = sorted(sorted(ann.part_area(q.id) for q in o.parts) for o in ann.objects)
    assert areas == [[300, 1280], [400]]
    assert ann.part_at(6, 91) is None and ann.part_at(95, 5) is None  # speck and table
    assert ann.source == "amg" and ann.image == "scene.png"
    assert all(o.cls is None and all(q.name is None and q.material is None for q in o.parts)
               for o in ann.objects)


def test_partial_overlap_smaller_object_wins():
    masks = np.stack([rect(10, 40, 10, 40), rect(30, 50, 30, 50)])
    ann = candidates_to_annotation(masks, np.array([0.9, 0.9]), GroupingParams(min_area=20))
    assert sorted(ann.part_area(o.parts[0].id) for o in ann.objects) == [400, 800]
    assert ann.part_area(ann.part_at(35, 35)) == 400


def test_leftover_object_pixels_become_a_rest_part():
    masks = np.stack([rect(10, 50, 10, 50), rect(10, 50, 10, 30)])
    ann = candidates_to_annotation(masks, np.array([0.9, 0.9]), GroupingParams(min_area=20))
    assert len(ann.objects) == 1
    assert sorted(ann.part_area(q.id) for q in ann.objects[0].parts) == [800, 800]


def test_remove_small_components():
    m = rect(0, 10, 0, 10, (20, 20)) | rect(15, 17, 15, 17, (20, 20))
    out = remove_small_components(m, 10)
    assert out.sum() == 100 and not out[16, 16]


def test_no_candidates_gives_empty_annotation():
    ann = candidates_to_annotation(np.zeros((0, 30, 40), bool), np.zeros(0))
    assert ann.size == (30, 40) and ann.objects == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_overlaps.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'annofix.overlaps'`.

- [ ] **Step 3: Write `annofix/overlaps.py`**

```python
"""Overlapping candidate masks (SAM automatic masks, or a pipeline's instance masks) ->
a single-ownership object -> part Annotation.

Candidates nest roughly (whole object > part > sub-part). Each candidate's parent is the
smallest candidate that mostly contains it; roots become objects and their direct children
become parts. Overlaps are resolved by painting large -> small so smaller masks win.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .annotation import AnnObject, Annotation, Part


@dataclass(frozen=True)
class GroupingParams:
    min_area: int = 100              # px; smaller candidates are dropped
    max_area_frac: float = 0.25      # larger candidates are background (table, wall)
    border_large_frac: float = 0.05  # "large" for the spans-two-borders rule
    dedupe_iou: float = 0.9          # near-duplicates keep the higher score
    contain_frac: float = 0.85       # |child ∩ parent| / |child| to count as contained
    parent_area_ratio: float = 1.05  # parent must be at least this much bigger
    collapse_frac: float = 0.9       # a lone child this big is the object itself
    min_component_px: int = 64       # final speckle removal per part


def resolve_overlaps(masks: np.ndarray) -> np.ndarray:
    """int32 map of (mask index + 1), 0 where no mask; smaller masks win overlaps."""
    areas = masks.sum(axis=(1, 2))
    out = np.zeros(masks.shape[1:], np.int32)
    for i in np.argsort(-areas, kind="stable"):
        out[masks[i]] = i + 1
    return out


def remove_small_components(mask: np.ndarray, min_px: int) -> np.ndarray:
    if min_px <= 1 or not mask.any():
        return mask
    _, cc, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    small = np.flatnonzero(stats[:, cv2.CC_STAT_AREA] < min_px)
    small = small[small != 0]
    if small.size == 0:
        return mask
    return mask & ~np.isin(cc, small)


def _borders_touched(m: np.ndarray) -> int:
    return int(m[0].any()) + int(m[-1].any()) + int(m[:, 0].any()) + int(m[:, -1].any())


def filter_candidates(masks: np.ndarray, p: GroupingParams) -> np.ndarray:
    n, h, w = masks.shape
    image_area = h * w
    areas = masks.sum(axis=(1, 2))
    keep = []
    for i in range(n):
        a = areas[i]
        if a < p.min_area or a > p.max_area_frac * image_area:
            continue
        if a > p.border_large_frac * image_area and _borders_touched(masks[i]) >= 2:
            continue
        keep.append(i)
    return np.array(keep, dtype=int)


def _bboxes(masks: np.ndarray) -> np.ndarray:
    out = np.zeros((len(masks), 4), int)  # y0, y1, x0, x1 (exclusive)
    for i, m in enumerate(masks):
        rows = np.flatnonzero(m.any(1))
        cols = np.flatnonzero(m.any(0))
        if rows.size:
            out[i] = rows[0], rows[-1] + 1, cols[0], cols[-1] + 1
    return out


def pairwise_intersections(masks: np.ndarray) -> np.ndarray:
    """(N,N) int64 |mi ∩ mj|, exact, computed only where bounding boxes overlap."""
    n = len(masks)
    inter = np.diag(masks.sum(axis=(1, 2)).astype(np.int64))  # sum, not reshape: works for N=0
    if n < 2:
        return inter
    bb = _bboxes(masks)
    y0 = np.maximum(bb[:, None, 0], bb[None, :, 0])
    y1 = np.minimum(bb[:, None, 1], bb[None, :, 1])
    x0 = np.maximum(bb[:, None, 2], bb[None, :, 2])
    x1 = np.minimum(bb[:, None, 3], bb[None, :, 3])
    overlap = np.triu((y0 < y1) & (x0 < x1), k=1)
    for i, j in zip(*np.nonzero(overlap)):
        sy, sx = slice(y0[i, j], y1[i, j]), slice(x0[i, j], x1[i, j])
        inter[i, j] = inter[j, i] = np.count_nonzero(masks[i, sy, sx] & masks[j, sy, sx])
    return inter


def dedupe(scores: np.ndarray, inter: np.ndarray, iou_thresh: float) -> list[int]:
    areas = np.diag(inter)
    kept: list[int] = []
    for i in np.argsort(-np.asarray(scores), kind="stable"):
        i = int(i)
        if all(inter[i, k] / (areas[i] + areas[k] - inter[i, k]) <= iou_thresh for k in kept):
            kept.append(i)
    return sorted(kept)


def find_parents(inter: np.ndarray, p: GroupingParams) -> np.ndarray:
    """parent[i] = index of the smallest candidate that contains candidate i, else -1."""
    areas = np.diag(inter).astype(np.float64)
    n = len(areas)
    if n == 0:
        return np.zeros(0, int)
    contained = inter / np.maximum(areas[:, None], 1) >= p.contain_frac
    bigger = areas[None, :] >= p.parent_area_ratio * areas[:, None]
    ok = contained & bigger
    np.fill_diagonal(ok, False)
    parent = np.argmin(np.where(ok, areas[None, :], np.inf), axis=1)
    parent[~ok.any(axis=1)] = -1
    return parent


def group_objects(parent: np.ndarray, areas: np.ndarray, p: GroupingParams) -> list[tuple[int, list[int]]]:
    """[(root, depth-1 children)]. Deeper candidates are dropped; their pixels stay with
    their depth-1 ancestor."""
    n = len(parent)
    children: dict[int, list[int]] = {i: [] for i in range(n)}
    for i in range(n):
        par = int(parent[i])
        if par != -1 and parent[par] == -1:
            children[par].append(i)
    groups = []
    for r in range(n):
        if parent[r] != -1:
            continue
        kids = children[r]
        if len(kids) == 1 and areas[kids[0]] >= p.collapse_frac * areas[r]:
            kids = []
        groups.append((r, kids))
    return groups


def paint_groups(masks: np.ndarray, groups: list[tuple[int, list[int]]], p: GroupingParams) -> Annotation:
    shape = masks.shape[1:]
    label = np.zeros(shape, np.uint16)
    objects: list[AnnObject] = []
    next_id = 1
    if groups:
        obj_map = resolve_overlaps(masks[[root for root, _ in groups]])
        for g, (_, kids) in enumerate(groups):
            region = obj_map == g + 1
            if not region.any():
                continue  # fully covered by smaller objects
            if kids:
                part_map = resolve_overlaps(masks[kids])
                pieces = [region & (part_map == k + 1) for k in range(len(kids))]
                pieces.append(region & (part_map == 0))  # leftover -> "rest" part
            else:
                pieces = [region]
            obj = AnnObject(next_id)
            next_id += 1
            for piece in pieces:
                piece = remove_small_components(piece, p.min_component_px)
                if piece.any():
                    label[piece] = next_id
                    obj.parts.append(Part(next_id))
                    next_id += 1
            if obj.parts:
                objects.append(obj)
    return Annotation(label, objects, next_id=next_id, source="amg")


def candidates_to_annotation(masks: np.ndarray, scores: np.ndarray,
                             p: GroupingParams = GroupingParams(), image: str = "") -> Annotation:
    keep = filter_candidates(masks, p)
    masks, scores = masks[keep], np.asarray(scores)[keep]
    inter = pairwise_intersections(masks)
    kept = dedupe(scores, inter, p.dedupe_iou)
    masks, inter = masks[kept], inter[np.ix_(kept, kept)]
    groups = group_objects(find_parents(inter, p), np.diag(inter), p)
    ann = paint_groups(masks, groups, p)
    ann.image = image
    return ann
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_overlaps.py -v`
Expected: PASS (12 tests).

- [ ] **Step 5: Commit**

```bash
git add annofix/overlaps.py tests/test_overlaps.py
git commit -m "feat: group overlapping candidate masks into objects and parts" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 8: Bootstrap CLI with mask cache

**Files:**
- Create: `annofix/bootstrap.py`, `tests/test_bootstrap.py`

**Interfaces:**
- Consumes: `io.list_images`, `io.load_image`, `io.has_annotation`, `io.load_annotation`, `io.save_annotation`, `overlaps.GroupingParams`, `overlaps.candidates_to_annotation`, `sam_backend.Sam2Backend`, `sam_backend.MODELS`.
- Produces:
  - `cache_path(out_dir, stem) -> Path` (`<out>/.amg_cache/<stem>.npz`)
  - `save_cache(path, masks, scores)`, `load_cache(path) -> (bool[N,H,W], float32[N])`
  - `bootstrap_image(image_path, out_dir, generate, params, *, overwrite=False, use_cache=True) -> str` returning one of `"generated" | "cached" | "skipped" | "protected" | "no-cache"`; `generate` is `Callable[[np.ndarray], tuple[masks, scores]]` or `None` (regroup only)
  - `main(argv: list[str] | None = None) -> int` (0 ok, 1 no images, 2 some images errored)

- [ ] **Step 1: Write the failing test `tests/test_bootstrap.py`**

```python
import subprocess
import sys

import cv2
import numpy as np
import pytest

from annofix.bootstrap import bootstrap_image, cache_path, load_cache, main, save_cache
from annofix.io import load_annotation, save_annotation
from annofix.overlaps import GroupingParams

H, W = 60, 80


def object_masks(image):
    m = np.zeros((1, H, W), bool)
    m[0, 10:30, 10:40] = True  # 600 px object
    return m, np.array([0.9], np.float32)


def must_not_run(image):
    raise AssertionError("generator called although a cache exists")


@pytest.fixture
def images(tmp_path):
    d = tmp_path / "images"
    d.mkdir()
    for name in ("a_color.png", "b_color.png"):
        cv2.imwrite(str(d / name), np.full((H, W, 3), 120, np.uint8))
    cv2.imwrite(str(d / "a_depth.png"), np.full((H, W), 900, np.uint16))
    return d


def test_generates_annotation_and_cache(images, tmp_path):
    out = tmp_path / "ann"
    assert bootstrap_image(images / "a_color.png", out, object_masks, GroupingParams()) == "generated"
    ann = load_annotation(out, "a_color", expected_size=(H, W))
    assert [len(o.parts) for o in ann.objects] == [1]
    assert ann.part_area(ann.part_ids()[0]) == 600
    assert ann.source == "amg" and ann.image == "../images/a_color.png"
    assert cache_path(out, "a_color").exists()


def test_existing_annotation_is_never_overwritten_by_default(images, tmp_path):
    out = tmp_path / "ann"
    bootstrap_image(images / "a_color.png", out, object_masks, GroupingParams())
    assert bootstrap_image(images / "a_color.png", out, must_not_run, GroupingParams()) == "skipped"


def test_overwrite_reuses_cache_without_generator(images, tmp_path):
    out = tmp_path / "ann"
    bootstrap_image(images / "a_color.png", out, object_masks, GroupingParams())
    assert bootstrap_image(images / "a_color.png", out, must_not_run, GroupingParams(),
                           overwrite=True) == "cached"


def test_overwrite_never_touches_reviewed_work(images, tmp_path):
    out = tmp_path / "ann"
    bootstrap_image(images / "a_color.png", out, object_masks, GroupingParams())
    ann = load_annotation(out, "a_color")
    ann.set_review_status("reviewed")
    save_annotation(ann, out, "a_color")
    assert bootstrap_image(images / "a_color.png", out, object_masks, GroupingParams(),
                           overwrite=True) == "protected"
    assert load_annotation(out, "a_color").review_status == "reviewed"


def test_regroup_without_cache_reports_it(images, tmp_path):
    assert bootstrap_image(images / "a_color.png", tmp_path / "ann", None, GroupingParams()) == "no-cache"


def test_cache_roundtrip(tmp_path):
    masks = np.random.default_rng(1).random((3, 7, 13)) > 0.5
    save_cache(tmp_path / "c.npz", masks, np.array([0.1, 0.2, 0.3], np.float32))
    back, scores = load_cache(tmp_path / "c.npz")
    assert np.array_equal(back, masks) and scores.tolist() == pytest.approx([0.1, 0.2, 0.3])


def test_cache_of_wrong_size_is_an_error(images, tmp_path):
    out = tmp_path / "ann"
    save_cache(cache_path(out, "a_color"), np.zeros((1, H + 1, W), bool), np.array([0.9]))
    with pytest.raises(ValueError, match="cache"):
        bootstrap_image(images / "a_color.png", out, None, GroupingParams())


def test_main_regroup_skips_depth_and_counts(images, tmp_path, capsys):
    out = tmp_path / "ann"
    for stem in ("a_color", "b_color"):
        save_cache(cache_path(out, stem), *object_masks(None))
    assert main(["--images", str(images), "--out", str(out), "--regroup"]) == 0
    text = capsys.readouterr().out
    assert "a_depth" not in text and "a_color.png: cached" in text and "b_color.png: cached" in text
    assert (out / "b_color.json").exists()


def test_main_returns_2_on_errors(images, tmp_path):
    out = tmp_path / "ann"
    save_cache(cache_path(out, "a_color"), np.zeros((1, H + 1, W), bool), np.array([0.9]))
    assert main(["--images", str(images), "--out", str(out), "--regroup"]) == 2


def test_core_modules_import_without_torch():
    code = ("import sys, annofix.bootstrap, annofix.io, annofix.overlaps, annofix.config, "
            "annofix.annotation; assert 'torch' not in sys.modules")
    subprocess.run([sys.executable, "-c", code], check=True)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_bootstrap.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'annofix.bootstrap'`.

- [ ] **Step 3: Write `annofix/bootstrap.py`**

```python
"""python -m annofix.bootstrap --images DIR --out DIR

Generate starting annotations from SAM 2.1 automatic masks. An existing annotation is never
replaced unless --overwrite, and even then never one marked reviewed or flagged. Raw masks
are cached in <out>/.amg_cache so --regroup can re-run the grouping without the GPU.
"""
from __future__ import annotations

import annofix  # noqa: F401  (thread cap before numpy)

import argparse
import os
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

from .io import has_annotation, list_images, load_annotation, load_image, save_annotation
from .overlaps import GroupingParams, candidates_to_annotation


def cache_path(out_dir: str | Path, stem: str) -> Path:
    return Path(out_dir) / ".amg_cache" / f"{stem}.npz"


def save_cache(path: Path, masks: np.ndarray, scores: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, packed=np.packbits(masks, axis=-1), shape=np.array(masks.shape),
                        scores=np.asarray(scores, np.float32))


def load_cache(path: Path) -> tuple[np.ndarray, np.ndarray]:
    d = np.load(path)
    _, _, w = d["shape"]
    masks = np.unpackbits(d["packed"], axis=-1, count=int(w)).astype(bool)
    return masks, d["scores"]


def bootstrap_image(image_path, out_dir, generate, params: GroupingParams, *,
                    overwrite: bool = False, use_cache: bool = True) -> str:
    image_path, out_dir = Path(image_path), Path(out_dir)
    stem = image_path.stem
    if has_annotation(out_dir, stem):
        if not overwrite:
            return "skipped"
        if load_annotation(out_dir, stem).review_status != "unreviewed":
            return "protected"
    image = load_image(image_path)
    cpath = cache_path(out_dir, stem)
    if use_cache and cpath.exists():
        masks, scores = load_cache(cpath)
        status = "cached"
    elif generate is None:
        return "no-cache"
    else:
        masks, scores = generate(image)
        save_cache(cpath, masks, scores)
        status = "generated"
    if masks.shape[1:] != image.shape[:2]:
        raise ValueError(f"{cpath.name}: cached masks are {masks.shape[2]}x{masks.shape[1]} but the "
                         f"image is {image.shape[1]}x{image.shape[0]}; delete the cache file")
    ann = candidates_to_annotation(masks, scores, params, image=os.path.relpath(image_path, out_dir))
    save_annotation(ann, out_dir, stem)
    return status


def main(argv: list[str] | None = None) -> int:
    from .sam_backend import MODELS  # plain dict; torch is only imported by Sam2Backend()

    ap = argparse.ArgumentParser(prog="python -m annofix.bootstrap", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--images", required=True, type=Path, help="folder of colour images")
    ap.add_argument("--out", required=True, type=Path, help="annotations folder")
    ap.add_argument("--sam-model", choices=sorted(MODELS), help="default depends on device")
    ap.add_argument("--checkpoint", type=Path, help="local .pt instead of a Hugging Face download")
    ap.add_argument("--device", help="cuda or cpu (default: cuda if available)")
    ap.add_argument("--points-per-side", type=int, default=32)
    ap.add_argument("--limit", type=int, help="only the first N images")
    ap.add_argument("--overwrite", action="store_true",
                    help="replace existing unreviewed annotations (reviewed/flagged are never replaced)")
    ap.add_argument("--regroup", action="store_true", help="only re-group cached masks; no SAM, no GPU")
    ap.add_argument("--no-cache", action="store_true", help="ignore cached masks and run SAM again")
    args = ap.parse_args(argv)

    images = list_images(args.images)
    if args.limit:
        images = images[: args.limit]
    if not images:
        print(f"no colour images in {args.images}", file=sys.stderr)
        return 1

    generate = None
    if not args.regroup:
        from .sam_backend import Sam2Backend

        backend = Sam2Backend(args.sam_model, args.device, args.checkpoint)
        print(f"SAM 2.1 {backend.model_name} on {backend.device}")
        if backend.device == "cpu":
            print("warning: automatic masks on CPU are very slow; run bootstrap on a GPU machine",
                  file=sys.stderr)

        def generate(image):
            return backend.generate(image, points_per_side=args.points_per_side)

    counts: Counter[str] = Counter()
    for i, path in enumerate(images, 1):
        t = time.perf_counter()
        try:
            status = bootstrap_image(path, args.out, generate, GroupingParams(),
                                     overwrite=args.overwrite, use_cache=not args.no_cache)
        except ValueError as e:
            status = "error"
            print(f"  {path.name}: {e}", file=sys.stderr)
        counts[status] += 1
        print(f"[{i}/{len(images)}] {path.name}: {status} ({time.perf_counter() - t:.1f}s)", flush=True)
    print("done: " + ", ".join(f"{k} {v}" for k, v in sorted(counts.items())))
    return 2 if counts["error"] else 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest -v`
Expected: PASS for every test file; `tests/test_sam_backend.py` SKIPPED (no env var).

- [ ] **Step 5: Commit**

```bash
git add annofix/bootstrap.py tests/test_bootstrap.py
git commit -m "feat: bootstrap CLI turning SAM automatic masks into starting annotations" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 9: Real-frame bootstrap, overlay check, README

**Files:**
- Create: `scripts/render_overlay.py`, `README.md`
- Modify: `docs/notes/2026-10-03-m0-measurements.md` (append bootstrap results); possibly `annofix/overlaps.py` `GroupingParams` defaults (only if Step 4 shows a clear problem)

**Interfaces:**
- Consumes: `io.list_images`, `io.load_image`, `io.has_annotation`, `io.load_annotation`, the bootstrap CLI.

- [ ] **Step 1: Write `scripts/render_overlay.py`**

```python
"""Eyeball annotations before the GUI exists: writes <out>/<stem>.overlay.jpg showing the
image next to a copy with each part tinted and each object outlined in white.

uv run python scripts/render_overlay.py --images DIR --annotations DIR --out DIR
"""
import annofix  # noqa: F401  (thread cap before numpy)

import argparse
from pathlib import Path

import cv2
import numpy as np

from annofix.io import has_annotation, list_images, load_annotation, load_image


def render(image: np.ndarray, ann) -> np.ndarray:
    lut = np.random.default_rng(0).integers(40, 256, size=(int(ann.label_map.max()) + 1, 3), dtype=np.uint8)
    colour = lut[ann.label_map]
    fg = ann.label_map > 0
    tinted = image.copy()
    tinted[fg] = (0.45 * image[fg] + 0.55 * colour[fg]).astype(np.uint8)
    for o in ann.objects:
        m = np.isin(ann.label_map, [p.id for p in o.parts]).astype(np.uint8)
        contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(tinted, contours, -1, (255, 255, 255), 2)
    return np.hstack([image, tinted])


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--images", required=True, type=Path)
    ap.add_argument("--annotations", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    for path in list_images(args.images):
        if not has_annotation(args.annotations, path.stem):
            continue
        image = load_image(path)
        ann = load_annotation(args.annotations, path.stem, expected_size=image.shape[:2])
        cv2.imwrite(str(args.out / f"{path.stem}.overlay.jpg"),
                    cv2.cvtColor(render(image, ann), cv2.COLOR_RGB2BGR))
        print(f"{path.name}: {len(ann.objects)} objects, {len(ann.part_ids())} parts")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Build a 10-frame sample with both resolutions**

```bash
F=../CamCalibration/MC-Calib/data/rosbag_mccalib_moving/frames
mkdir -p data/sample
for i in 000 040 080 120 160 195; do cp "$F/cam0_${i}_color.png" data/sample/; done
for i in 000 060 120 180; do cp "$F/cam1_${i}_color.png" data/sample/; done
ls data/sample | wc -l
```
Expected: `10` (cam0 = 1280x720, cam1 = 640x480; `data/` is git-ignored).

- [ ] **Step 3: Run bootstrap on the GPU, then regroup from cache**

Run: `uv run python -m annofix.bootstrap --images data/sample --out data/sample_ann`
Expected: `[n/10] <name>: generated (<t>s)` lines, ending `done: generated 10`.
Run: `uv run python -m annofix.bootstrap --images data/sample --out data/sample_ann --regroup --overwrite`
Expected: `done: cached 10` in seconds, with no GPU model load line.

- [ ] **Step 4: Render and inspect overlays**

Run: `uv run python scripts/render_overlay.py --images data/sample --annotations data/sample_ann --out data/sample_overlay`
Open 3–4 overlays (one cam1 640x480). Check: the robot arm, tripod camera, laptop and cable each come out as one outlined object with a few parts; the table, floor and walls are not objects; masks line up with the image at both resolutions. If a defect is clear and systematic (for example, the table appears as an object, or every object is a single part), change the matching `GroupingParams` default, re-run Step 3's `--regroup --overwrite` command, re-render, and run `uv run pytest tests/test_overlaps.py` to confirm the synthetic tests still pass. Record what changed and why.

- [ ] **Step 5: Append results to `docs/notes/2026-10-03-m0-measurements.md`**

```markdown
## Bootstrap on 10 real frames (6 x 1280x720 cam0, 4 x 640x480 cam1)

| frame | seconds | objects | parts |
|---|---|---|---|
| (one row per printed line from Steps 3-4) |

GroupingParams changes: <none | what changed and why>.
Observed problems to carry into Plans 2-3: <e.g. thin cable fragmented, people split into many parts>.
```

- [ ] **Step 6: Write `README.md`**

````markdown
# AnnoFix

Fix object → part → material segmentation annotations, with SAM 2.1 doing the heavy lifting.
Design: `docs/superpowers/specs/2026-10-03-annofix-design.md`.

## Install

```bash
SAM2_BUILD_CUDA=0 uv sync      # PyTorch with CUDA, SAM 2.1 from GitHub (pinned commit)
```
SAM 2.1 checkpoints download from Hugging Face on first use (~150–900 MB depending on model);
pass `--checkpoint path/to/sam2.1_hiera_*.pt` to work offline.

## Bootstrap starting annotations (GPU)

```bash
uv run python -m annofix.bootstrap --images IMAGES_DIR --out ANNOTATIONS_DIR
uv run python -m annofix.bootstrap --images IMAGES_DIR --out ANNOTATIONS_DIR --regroup --overwrite  # re-group cached masks, no GPU
uv run python scripts/render_overlay.py --images IMAGES_DIR --annotations ANNOTATIONS_DIR --out OVERLAY_DIR
```
Existing annotations are skipped; `--overwrite` replaces unreviewed ones only.
Depth frames (`*_depth.png`) in the image folder are ignored.

## File format (contract with the annotation pipeline)

Per image `<stem>`: `<stem>.parts.png` — single-channel **uint16**, pixel = part id, 0 = background
(one owner per pixel) — and `<stem>.json`:

```json
{"schema_version": 1, "image": "../images/x.png", "size": [720, 1280], "next_id": 17,
 "source": "amg", "annotator": null, "review_status": "unreviewed",
 "objects": [{"id": 1, "class": "screwdriver",
              "parts": [{"id": 3, "name": "handle", "material": "rubber"}]}]}
```
Unlabelled values are `null`. Ids are shared by objects and parts and never reused.
Read one from Python with `annofix.io.load(ann_dir, stem) -> (label_map, meta)`.

## Tests

```bash
uv run pytest                                   # core + bootstrap
ANNOFIX_SAM_TESTS=1 uv run pytest -m sam        # real SAM 2.1 (downloads the tiny model)
```
````

- [ ] **Step 7: Full test run and commit**

Run: `uv run pytest -v && ANNOFIX_SAM_TESTS=1 uv run pytest -m sam -v`
Expected: all PASS.

```bash
git add scripts/render_overlay.py README.md docs/notes/2026-10-03-m0-measurements.md annofix/overlaps.py
git commit -m "docs: README, overlay renderer, bootstrap results on real frames" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

## What Plans 2 and 3 will build on

- Plan 2 (viewer + manual editing, spec M3–M4) consumes `Annotation` (all ops, `transaction`, `undo/redo`, `dirty`, `validate`), `io.save_annotation/load_annotation/list_images/load_image`, `config.load_labels`.
- Plan 3 (interactive SAM, spec M5–M6) consumes `Sam2Backend.set_image/predict` and `Annotation.replace_part_mask/assign/new_part/new_object`, and uses the M0 numbers to decide whether embedding prefetch and zoomed-crop SAM are needed.
