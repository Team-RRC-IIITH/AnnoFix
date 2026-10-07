# AnnoFix

Fix object → part → material segmentation annotations, with SAM 2.1 doing the heavy lifting.
Design: `docs/superpowers/specs/2026-10-03-annofix-design.md`.
Measurements and bootstrap findings: `docs/notes/2026-10-03-m0-measurements.md`.

## Install

```bash
SAM2_BUILD_CUDA=0 uv sync      # PyTorch with CUDA, SAM 2.1 from GitHub (pinned commit)
```
This is large: the `.venv` is ~5.7 GB (torch 2.14.1+cu130 plus NVIDIA libraries) and uv's cache
holds a similar amount. On a slow or flaky connection, download one file at a time:
```bash
UV_CONCURRENT_DOWNLOADS=1 UV_HTTP_RETRIES=10 UV_HTTP_TIMEOUT=600 SAM2_BUILD_CUDA=0 uv sync
```
SAM 2.1 checkpoints download from Hugging Face on first use (~150–900 MB depending on model);
pass `--checkpoint path/to/sam2.1_hiera_*.pt` to work offline.

## Bootstrap starting annotations (GPU)

```bash
uv run python -m annofix.bootstrap --images IMAGES_DIR --out ANNOTATIONS_DIR
uv run python -m annofix.bootstrap --images IMAGES_DIR --out ANNOTATIONS_DIR --regroup --overwrite  # re-group cached masks, no GPU
uv run python scripts/render_overlay.py --images IMAGES_DIR --annotations ANNOTATIONS_DIR --out OVERLAY_DIR
```
Existing annotations are skipped. `--overwrite` only replaces annotations that bootstrap itself
wrote and that nobody has changed since (checked again right before each write) — anything
edited (even unreviewed), reviewed, flagged, written by the pipeline, or unreadable is reported
as `protected` and left alone. Close the editor on that folder first, though: edits it has not
saved yet are not on disk, so bootstrap cannot see them.

The mask cache records which image and which SAM settings produced it: a cache from a
re-recorded frame or older settings is regenerated (or, under `--regroup`, reported as an error
for that image) instead of being reused silently. Depth frames (`*_depth.png`) are ignored.
On the RTX 3050 a 1280x720 frame takes ~5 s with the default `base_plus` model.

## Fixing pipeline masks (CVAT/COCO)

The pipeline's output is CVAT-style COCO, one folder per sample (`<sample>/images/…`,
`<sample>/annotations/instances_default.json`, `export_manifest.json`, `cvat_labels.json`). Fix it
with one command:

```bash
uv run python -m annofix.cvat Test --out Test_fixed [--annotator NAME] [SAM options]
uv run python -m annofix.cvat Test --out Test_fixed --export-only   # rewrite every output, no window
```
- `Test/` is never written. Every save writes that sample to `Test_fixed/<sample>/` in the same
  layout and files, so the output folder is always current (no export step).
- The working copy lives in `Test_fixed/.annofix/` (AnnoFix's own format). A sample already
  there is never re-imported, so re-running the command keeps your edits; new samples in
  `Test/` are added.
- The 10 materials are the COCO categories (keys `1`–`9`, `0`); object classes and part-name
  suggestions are collected from the data into `Test_fixed/.annofix/labels.yaml`.
- **Object = its parts + a `body` part.** The editor gives each pixel one owner, so on import an
  object's pixels not covered by its parts become a part named `body`, carrying the object's
  material. On export the object mask is again the union of its pieces and the object's
  category is the body's material. Overlap slivers between different objects go to the smaller
  object.
- The tree shows the pipeline's confidence after each label (`#12 lid · plastic · 49%`), and
  `P` jumps to the next part with a missing label.
- uids, object ids and confidences are kept; new objects get the next `object_id`, uid `o<N>`
  and confidence 100, new parts uid `p<object>_<k>`. Marking an image reviewed sets
  `reviewed: true` on all its annotations.
- COCO polygons cannot hold holes, and a part's polygons may differ from the pipeline's by
  about a pixel at the boundary.

## Fixing annotations (GUI)

```bash
uv run python -m annofix --images IMAGES_DIR --annotations ANNOTATIONS_DIR [--labels labels.yaml] [--annotator NAME]
#   SAM options: [--sam-model tiny|small|base_plus|large] [--device cuda|cpu] [--checkpoint FILE.pt] [--no-sam]
```
Left: images (○ no annotation yet · • unreviewed · ✓ reviewed · ⚑ flagged). Centre: the image with
its masks; part borders are drawn darker, so neighbouring parts of the same colour stay distinct.
Right: objects → parts (tick boxes show/hide; right-click to move, split off, clean or delete) and
the label panel. The title shows the file and how many images are reviewed.

Work saves 2 s after your last finished edit (never in the middle of a stroke), every 30 s, when
you change image, and on close. Ctrl+C in the terminal also saves and closes. `F1` or `?` lists
every shortcut.

| To do this | Press / click |
|---|---|
| Select a part · add/remove one | click · Ctrl+click (canvas or tree) |
| Set material | `1`–`9`, `0` (10th) — keypad too — or a material button |
| Part name · object class | suggestion buttons, or type (applied on Enter or when you click away; `Esc` drops it) — for every selected part |
| Next part with a missing label | `P` |
| Give the selected parts the last name + material again | `.` |
| Brush (paint the selected part) · eraser · select | `B` · `E` · `V` |
| Brush size | `[` `]` |
| Protect other parts while painting | toolbar toggle |
| Cancel a stroke | `Esc` |
| Merge selected parts · merge their objects | `M` · `Shift+M` |
| Delete part(s) · delete object | `Del` · `Shift+Del` |
| Undo · redo | `Ctrl+Z` · `Ctrl+Shift+Z` |
| **SAM fix** the selected part (seeded inside it) | `R` |
| SAM: include · exclude · box | left click · right click · Shift+drag |
| SAM: next/previous candidate · replace · add · cancel | `Tab`/`Shift+Tab` · `Enter` · `Shift+Enter` · `Esc` |
| New part · new object (with SAM: from clicks) | `N` · `Shift+N` |
| New empty part, brush ready (also with SAM) | `Ctrl+N` |
| Hide masks while held · solo object · outlines | `Space` · `S` · `O` |
| Next/previous object | `J` · `K` |
| Previous/next image · next unreviewed | `A` · `D` · `Ctrl+U` |
| Mark reviewed (checks every label) and open the next unreviewed · flag | `Ctrl+Enter` · `F` |
| Save now | `Ctrl+S` |
| Zoom · pan · fit the image | mouse wheel · middle- or right-drag (right-drag not in SAM mode) · `Home` |
| All shortcuts | `F1` · `?` |

SAM loads on first use (`R`), in the background; the status bar shows its state. The preview
colours what Enter would do: white kept, green added, orange taken from another part, red
removed. Hidden parts (and parts outside a solo object) are never changed — by SAM, the brush or
the eraser — and "Protect other parts" applies to SAM as it does to the brush. If SAM fails to
load, its button disappears and the status bar says why; the brush keeps working.

Errors are logged to `ANNOTATIONS_DIR/annofix-errors.log`; on a crash, finished edits are saved
and a half-finished stroke is rolled back. An annotation file that cannot be read (broken JSON,
an unknown review status) is reported and skipped, or treated as unreviewed; it never stops
the tool from starting.

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

Materials and object classes (with suggested part names) live in `labels.yaml`.

## Tests

```bash
uv run pytest                                   # core + bootstrap
ANNOFIX_SAM_TESTS=1 uv run pytest -m sam        # real SAM 2.1 (downloads models; two tests use a CamCalibration frame)
```
