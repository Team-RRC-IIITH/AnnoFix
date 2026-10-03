# AnnoFix — Segmentation Annotation Fixing Tool (design)

## Context

The team's auto-annotation pipeline (in progress, separate) will label RGB frames of
tabletop / robot-grasping scenes (RealSense D455, 1280x720 and 640x480). Automatic labels
are always partly wrong: **wrong class/material**, **bleeding masks** (mask spills past an
object's edge onto the table or a neighbour), **missing objects**, **merged or junk masks**.

AnnoFix is a PyQt6 desktop app where a person reviews each image and fixes these quickly.
A **SAM 2.1** model makes fixes cheap: one click inside the object plus one click on the bled
region, and SAM redraws the mask. Until the pipeline exists, AnnoFix bootstraps its own
starting masks with SAM 2.1 automatic mask generation (AMG), so the tool is testable now.

Project dir is empty, not a git repo; no prior design exists on record.

### Decisions made with the user
| Topic | Decision |
|---|---|
| Label structure | **Object → parts → material.** Object has a class (screwdriver); split into parts (handle, shaft, tip); each part has a part name + a material. One part = one mask. |
| Overlap | **Single ownership** — each pixel belongs to ≤1 part (0 = background). Visible-only (modal) masks. |
| Model | **SAM 2.1 via PyTorch on the RTX 3050**, CPU fallback (slower) for teammates. ~3 GB install, accepted. |
| Mask source now | Tool bootstraps its own (SAM AMG); pipeline writes the same format later. |
| Build vs adopt | Custom tool (X-AnyLabeling exists; custom chosen for hierarchy/material/format control). |

## On-disk format (the contract with the pipeline)

Per image `<stem>` in `--annotations DIR`:
- `<stem>.parts.png` — **uint16** label map, pixel = part id, 0 = background. Written with cv2.
- `<stem>.json`:
  ```json
  {"schema_version": 1, "image": "rel/path.png", "size": [H, W], "next_id": 17,
   "source": "amg|pipeline|manual", "annotator": "...", "review_status": "unreviewed|reviewed|flagged",
   "objects": [{"id": 1, "class": "screwdriver",
                "parts": [{"id": 3, "name": "handle", "material": "rubber"}]}]}
  ```
- Unlabelled = `null`. IDs are never reused (`next_id`). Unknown names (not in labels.yaml) are kept + warned.
- Save order: PNG then JSON, each atomic (`cv2.imencode` → write → fsync → `os.replace`).
- Load reconciles: PNG ids missing from JSON → "orphans" object; JSON parts with 0 px → flagged; size mismatch → error.
- `annofix.io.load(stem) -> (label_map, meta)` is the public helper for downstream consumers.
- `resolve_overlaps()` is reusable, so a pipeline emitting overlapping instance masks can convert.

`labels.yaml` (user-edited): `materials: [...]` (order = hotkeys 1–9, colours auto) and
`objects: {screwdriver: [handle, shaft, tip], mug: [body, handle], ...}` — part names are
*suggestions* per class; free text allowed.

## Editing model (pixel ownership rules)

- **Assign** pixels to a part: steals from other parts unless the global **"Protect other parts"** toggle is on (then only fills background).
- **Erase / shrink**: removed pixels → background.
- Fixing a bleed works from either side: shrink the bleeding part (SAM or eraser), or regrow the neighbour (it takes its pixels back).
- Every edit runs in a **transaction**: copy label map before, diff after, store bbox-cropped before/after + metadata → undo/redo. History capped by bytes (~200 MB), cleared on image switch (after save). Tracks dirty state.

## UI

Layout: left = file list (status icon per image) · centre = canvas · right = object/part tree + label panel.

| Feature | Control |
|---|---|
| Show/hide a part or object | checkbox in tree |
| Hide all masks while held | **Space** |
| Solo selected object | **S** |
| Colour by material / part / object | toolbar toggle; opacity slider; outline mode **O** |
| Select part | click on canvas or tree |
| Set material | material buttons, hotkeys **1–9** |
| Set part name / object class | buttons from labels.yaml + free-text field |
| **SAM fix** | **R**: left-click = include, right-click = exclude, Shift+drag = box. Seeded with one positive point at the part's interior (distance-transform max), not its (wrong) bbox. Single click → 3 candidates, **Tab** cycles (sorted by area, IoU shown). Feeds back low-res logits between clicks. **Enter** = replace part, **Shift+Enter** = add (union), **Esc** = cancel. Diff preview shows added / removed / stolen pixels. |
| Brush / eraser | **B** / **E**, radius **[ ]**, strokes interpolated with `cv2.line` |
| New part (in selected object) | **N** then SAM click |
| New object (missed object) | **Shift+N** then SAM click on background |
| Merge parts / delete part | Ctrl-click a second part, then **M** (merged into the first) / **Del** |
| Move part to object / promote part → new object | tree context menu |
| Clean selected part | remove islands + fill holes (context menu) |
| Undo / redo | **Ctrl+Z / Ctrl+Shift+Z** |
| Navigate | **A / D** prev/next, **Ctrl+U** next unreviewed; autosave on navigate, dirty timer, close |
| Review | **Ctrl+Enter** mark reviewed (validates: every object has class, every part name + material, no empty parts); **F** flag |

Overlay rendering: persistent uint32 buffer + `QImage(Format_ARGB32_Premultiplied)` wrapped in a
custom `QGraphicsItem`; recolour via `np.take(lut32, label_map)` (~1.3 ms at 1280x720, measured).
Visibility, colour mode, solo and selection highlight are all just LUT changes. Opacity via
`item.setOpacity`. Colours derived from a hash of the id (stable across edits).

Robustness: `sys.excepthook` logs + autosaves + shows a dialog (PyQt6 otherwise aborts);
panel buttons `NoFocus`, hotkeys via `QShortcut`; canvas overrides `focusNextPrevChild` so Tab
reaches it; Space ignores auto-repeat.

## SAM backend

- `SegmentationBackend` protocol (`set_image`, `predict(points, labels, box, prev_logits, multimask)`, `generate`) with `Sam2Backend` and a `FakeBackend` for UI tests.
- Runs in a `QThread` worker; model loaded lazily so the window opens instantly; requests tagged `(image_key, seq)`, stale results dropped, pending predicts collapsed to latest; thread stopped in `closeEvent`.
- Embedding computed on image load (prefetch/LRU deferred to M6 unless CPU users need it).
- Models: default `sam2.1_hiera_base_plus` on CUDA (config `configs/sam2.1/sam2.1_hiera_b+.yaml`), `--sam-model small|tiny` for CPU; `--checkpoint PATH` (offline) or HF id. bf16 autocast + TF32 on CUDA.
- `multimask_output=True` only for a single click; False with ≥2 points or a box.
- Known limit: decoder is 256x256, so thin parts come out blobby at 1280x720 → brush is essential; **zoomed-crop SAM** is the M6 remedy if measurements show it's needed.

## Bootstrap (`python -m annofix.bootstrap --images DIR --out DIR`)

1. AMG (`binary_mask`, `box_nms_thresh≈0.9`); cache raw masks per image (`npz` packbits) to re-tune grouping without GPU.
2. Filter: min area, max area (~25% of image), drop large masks touching ≥2 borders (table/wall).
3. Dedupe: IoU > 0.9 keeps higher score.
4. Parent = smallest mask j with area_j ≥ 1.25·area_i and |i∩j|/|i| ≥ 0.85. Roots = objects, depth-1 children = parts; deeper levels folded into depth-1 ancestor; a child covering ≥90% of parent collapses.
5. Paint objects large→small; within each object's final pixels paint its parts large→small; leftover object pixels → "rest" part (or single part if no children).
6. Drop connected components < 64 px (tunable) per part. Write with all labels `null`, `source="amg"`, unreviewed.
GPU batch job; CPU AMG not supported in v1.

## Project layout

```
AnnotationFixingTool/
├── pyproject.toml          uv project: torch, torchvision, sam-2 (git, pinned sha, SAM2_BUILD_CUDA=0),
│                           huggingface-hub, pyqt6, opencv-python-headless, numpy, pyyaml; dev: pytest, pytest-qt
├── labels.yaml             example materials + object classes/parts
├── README.md
├── annofix/
│   ├── __init__.py         sets OMP/OPENBLAS/MKL/NUMEXPR_NUM_THREADS=10 before numpy (machine rule)
│   ├── __main__.py         CLI: --images --annotations --labels --sam-model --checkpoint --device
│   ├── config.py           labels.yaml loading/validation
│   ├── annotation.py       pure-numpy model: ops, ownership rules, transactions, undo/redo
│   ├── io.py               load/save/reconcile, atomic writes, image loading (skip depth PNGs, BGR→RGB)
│   ├── overlaps.py         resolve_overlaps(), containment grouping (shared by bootstrap + pipeline import)
│   ├── sam_backend.py      protocol, Sam2Backend, FakeBackend
│   ├── bootstrap.py        AMG CLI
│   └── ui/  main_window.py, canvas.py, overlay_item.py, tree_panel.py, label_panel.py, tools.py, worker.py
├── tests/                  conftest.py (thread cap, offscreen Qt), unit + pytest-qt tests
└── checkpoints/            git-ignored
```
Venv is clean (no `--system-site-packages`); never clone the sam2 repo into the project root.

## Milestones (each independently testable)

- **M0 spike** — `git init`, `uv sync`; load base+/small via `.pt` and HF on CUDA and CPU; measure encode/decode time and VRAM; AMG on a 1280x720 frame; Qt window with torch + cv2 in one process (Wayland). Decide torch pin (sam2 untested on torch 2.14 — pin older if it breaks) and default models.
- **M1 core (TDD)** — `annotation.py`, `io.py`, `config.py`. Tests: steal vs protect, erase, merge, move, promote, delete; undo/redo byte-identical incl. metadata; ids never reused; uint16 round-trip (ids 255/256/32768/65535); atomic write; orphan reconcile; size check.
- **M2 bootstrap** — synthetic-mask tests (nested, duplicate, table-size, partial overlap, speckle), then eyeball ~10 real frames.
- **M3 viewer** — LUT overlay, colour modes, opacity, outline, Space, zoom/pan, file list. Test: rendered pixel = LUT colour; recolour < 5 ms.
- **M4 manual editing** — select, tree, label panel, hotkeys, brush/eraser, structural ops, undo, autosave, review + validation, excepthook. Tool is useful without SAM after this.
- **M5 interactive SAM** — worker, backend protocol, clicks, Tab cycling, logit feedback, diff preview, accept modes. Tests with FakeBackend; real-SAM test skipped without checkpoint.
- **M6 (only if measured need)** — zoomed-crop SAM, embedding prefetch/LRU, shortcut help overlay.

Out of scope v1: COCO/other exporters, amodal masks, depth display, multi-user locking, cross-image undo, CPU AMG.

## Verification

- `uv run pytest` — unit tests (M1/M2), pytest-qt offscreen UI tests with FakeBackend (M3–M5); real-SAM integration test runs when a checkpoint is present.
- M0 timings reported as measured numbers (encode/decode ms, peak VRAM, AMG s/image).
- End-to-end on real frames: bootstrap ~10 frames from `CamCalibration/MC-Calib/data/rosbag_mccalib_moving/frames/cam0_*_color.png` (tripod camera, cable, laptop, robot arm = good multi-part objects), launch `uv run python -m annofix ...`, then: relabel a part, fix a bleed with SAM (include + exclude click), brush touch-up, undo/redo, mark reviewed, reopen and confirm the PNG/JSON round-trip. Screenshots of each step.

## Process after approval

Per the brainstorming workflow: save this design to
`docs/superpowers/specs/2026-10-03-annofix-design.md` in the new repo, then produce the
step-by-step implementation plan (writing-plans) and you choose how it's executed.
