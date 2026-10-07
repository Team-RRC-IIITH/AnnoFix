# AnnoFix round 4: fix your pipeline's CVAT/COCO masks, plus bug fixes and improvements

## Context

The real masks to fix come from the user's pipeline (SegObjectMaterial: "part+material predictions")
in a CVAT-style COCO export, one folder per sample (`Test/000000_000001/`):
- `images/000000_000001.png`: a 1920x1080 RGB photo
- `annotations/instances_default.json`: COCO, polygons only.
  - **categories = 10 materials** (fabric, foliage, glass, leather, metal, paper, plastic, stone, water, wood).
  - Each annotation's `attributes` holds `uid` (`o9`, `p9_1`), `kind` (object/part), `object_id`, `object_name`, `part_name`, `confidence_pct`, `reviewed`, `occluded`.
- `export_manifest.json`: source npz, `file_name`, `size`, `classes`, and `annotations` (uid → kind, object handle `sam3_###`, object_id, object_name, part_name, material, confidence 0–1).
- `cvat_labels.json`: CVAT label definitions (the 10 materials with those attributes).

Measured on the sample:
- 16 object masks and 14 part masks. Each part sits 99.5–100% inside its object.
- Each object with parts also has an uncovered "rest" region (5–96% of the object), carrying the object's own material.
- Different objects overlap only in thin boundary slivers (≤ 475 px), i.e. bleed errors.
- Many materials are wrong (a cup lid as `water`, a granola box as `water`, an inhaler as `glass`), and confidences are 21–96%.

The user wants to fix these now; part-based extras and grasp annotation come later.

At the same time, fix every known bug and add the agreed improvements:
- 14 bugs deferred by the Plan 2/3 reviews
- 12 new bugs from the read-only bug hunt, each confirmed by a probe
- all four improvement groups the user picked

### Decisions (asked and answered)
| Topic | Decision |
|---|---|
| Output | **New folder, same format** (`Test_fixed/<sample>/…` with the same 3 JSON files + image); the pipeline output in `Test/` is never modified |
| Objects vs parts | **Object = its parts + a "body" part.** The body is the object minus its parts and carries the object's material. On export the object mask is rebuilt as the union of its pieces, so the output again has overlapping object + part masks |
| Extras | Faster labelling, review flow, never lose edits, easier viewing (all four) |

## Part 1 — CVAT/COCO format support (`annofix/cvat.py`, new)

**Working store stays AnnoFix's native, lossless format** (uint16 label map + JSON, already tested:
undo, safe saves, review). The CVAT files are read once on import and written on every save:
- **Import** (`import_dataset(src_root, work_dir)`):
  - Find every `<sample>/annotations/instances_default.json` under the root, plus each JSON `images[]` entry. Stem = sample name, or `<sample>__<file stem>` if a sample has several images.
  - Link (symlink, falling back to copy) the image into `work/images/<stem>.png`.
  - Rasterize polygons with `cv2.fillPoly`.
  - Paint objects large→small (smaller wins the bleed slivers). Inside each object's final region, paint its parts large→small (parts clipped to the object). What's left is the **body** part (material = the object's category).
  - Write `work/annotations/<stem>.{parts.png,json}`. **Never re-import a stem that already exists in `work/`** (human edits are kept).
  - Also write `work/labels.yaml`: the dataset's 10 materials in category order; object classes and part-name suggestions collected from the data.
- **Provenance travels in an `attrs` dict.** `Part` and `AnnObject` get an optional `attrs: dict` (default `{}`), written into the JSON (schema stays v1; older files load with `{}`):
  - part `{"uid", "confidence_pct", "role": "part"|"body"}`
  - object `{"uid", "object_id", "object": "sam3_009", "confidence_pct"}`
  - image-level `{"source_sample", "source_file", "source_json"}`

  These survive undo (they are in `meta_dict`) and are what export uses to keep uids stable. They also make room for future grasp / part attributes.
- **Export** (`export_image(ann, src_sample_dir, out_root)`), writing `out_root/<sample>/` in the same layout:
  - `images/<file>`: a copy of the image.
  - `annotations/instances_default.json`: same `info`/`licenses`/`categories`/`images` as the source. Annotations:
    - **Object** annotation (kind object): mask = union of the object's parts; category = the body's material (or the original object category if there's no body); `object_name` = class; uid / `object_id` / confidence kept, and new objects get the next free `object_id`, `uid o<N>` and confidence 100.
    - **Part** annotation (kind part) for every non-body part: `part_name`, category = material, uid kept or new `p<objid>_<k>`.
    - Polygons come from `cv2.findContours` (RETR_EXTERNAL, one polygon per connected piece), plus `bbox` and `area`.
    - `reviewed` = True for every annotation when the image is marked reviewed.
  - `export_manifest.json` (same keys, annotations rebuilt from the above, confidence as 0–1) and `cvat_labels.json` (copied).
  - Atomic writes (reuse `io._atomic_write`).
  - Known format limits, kept as-is: COCO polygons can't hold holes, and boundaries may move by about 1 px through polygonization.
- **One command**: `uv run python -m annofix.cvat Test --out Test_fixed [--annotator NAME] [--no-sam …]`
  - Imports any new samples into `Test_fixed/.annofix/` (the working store) and opens the editor on it, with `work/labels.yaml`.
  - **Exports that image to `Test_fixed/<sample>/` every time it is saved**, so the output folder is always current and there is no separate export step.
  - `--export-only` regenerates every output from the working store without opening the window.
- Hooks needed in existing code:
  - `Session` gets a `"saved"` notification (the stem is known), which the CVAT command connects to the exporter.
  - Labels allow up to 10 material hotkeys (`1`–`9`, `0`).
  - `validate()` accepts the body part without a typed name (its role is "body").

## Part 2 — Bug fixes (all 26)

**Core / data safety (`io.py`, `session.py`, `bootstrap.py`, `config.py`)**
- A malformed annotation JSON (missing `id`, `objects: null`, …) now becomes a ValueError, so it is skipped with a message instead of crashing startup or trapping A/D.
- An unknown `review_status` (`null`, `"done"`, non-dict JSON) falls back to "unreviewed", so the window always starts.
- Duplicate stems (`x.png` + `x.jpg`) are refused with a message naming them.
- A save failure during A/D/Ctrl+U surfaces as a save error dialog, not as "skipped" images.
- The "no longer passes review → saved unreviewed" note survives the image change.
- Objects emptied by Del, M across objects, move or promote are dropped in the same undo step.
- A recovered lone PNG gets its image path filled in.
- `bootstrap --overwrite` re-checks "untouched" right before writing; README wording softened.
- CLI errors: broken YAML → readable error; missing `--images` → message, rc 1; `--regroup` with nothing cached → rc 1; `--limit` ≥ 1.
- Running out of part ids → warning dialog, not the crash dialog.

**Editing (`session.py`)**
- The brush/eraser refuse when the selected part is hidden or outside the solo object. This was the worst data bug: visible parts were losing pixels unseen.
- Canvas clicks don't select invisible parts.
- J/K end an open stroke first; with solo on, J/K move the solo to the new object.
- SAM: clicks outside the image are ignored, boxes are clamped, and an empty mask can't be accepted.
- SAM: changing the selection cancels the open fix; R restarts on the selected part.
- SAM status: the failure reason is shown, and the status is reset when the image changes.

**UI (`canvas.py`, `main_window.py`, `label_panel.py`, `sam_worker.py`, `__main__.py`)**
- A missed mouse release no longer keeps painting.
- Clicking the already-open file no longer reloads it.
- Fit to view at startup and on resize; `Home` refits.
- Space only acts inside the main window, not in dialogs.
- Ctrl+C in the terminal saves and quits.
- Crash handler: roll back, save, then notify the UI.
- `Shift+Backtab` registered.
- Label-panel buttons and Esc return focus to the canvas.
- The SAM toolbar button stays in sync, and is hidden after a fatal load failure.
- The name field also commits on focus-out.
- README gaps filled.

## Part 3 — Improvements (all four groups)

- **Faster labelling:**
  - name/class apply to all selected parts; `.` repeats the last name+material
  - `0` = 10th material
  - keypad digits work
  - the tree shows confidence, e.g. `printed label · paper · 49%`
- **Review flow:**
  - `P` selects the next part/object with a missing label
  - `Ctrl+Enter` marks reviewed and opens the next unreviewed image
  - "reviewed 12/200" in the title and status bar
  - messages at the list ends and "saved" on Ctrl+S
- **Never lose edits:** save 2 s after the last finished edit (the 30 s timer stays as a backstop), never mid-stroke. In the CVAT command, every save also exports.
- **Easier viewing:**
  - part boundaries drawn over the fills, so same-colour parts are distinct; `O` still gives outlines only
  - right-drag pans in select/paint modes, for the touchpad
  - `Home` fits the view
  - `F1`/`?` shortcut help
  - the window title shows the file name
- **Empty part for brushing:** `Ctrl+N` makes an empty part with the brush ready, even when SAM is on.

**Not in this round:** grasp annotation and richer part-based labels (the `attrs` design leaves room for both), copying labels between frames, zoomed-crop SAM, and a CPU-only install.

## Process and files

- One branch `round-4`, Native execution by me, test-first for every item: a failing test first, then the fix, then the full suite.
- Commits go in groups: CVAT import/export, then bugs, then improvements.
- One fresh whole-branch reviewer at the end and a single fix pass, then merge to `main`.
- No co-author trailer on commits.

**Files:**
- `annofix/cvat.py` (new)
- `annofix/annotation.py` (`attrs`; body-role validation)
- `annofix/io.py`, `annofix/session.py` (`"saved"` notification + the fixes)
- `annofix/config.py`
- `annofix/bootstrap.py`
- `annofix/ui/{canvas,main_window,label_panel,tree_panel,colors,sam_worker}.py`
- `annofix/__main__.py`
- `README.md` (new "Fixing pipeline masks (CVAT/COCO)" section)

**Reused:**
- `overlaps.resolve_overlaps` (paint order)
- `io._atomic_write`, `io.save_annotation/load_annotation`
- `Session._is_locked`, `_try_open`
- `Annotation.transaction`, `validate`
- `colors.edges`
- `tests/helpers.py`
- `FakeBackend`

## Verification

- **Unit tests for the format:**
  - Import of `Test/000000_000001` gives 16 objects, 14 non-body parts plus bodies, materials as in the file, and stems/uids recorded.
  - Import → export with no edits gives the same uids, kinds, object_ids, names and materials, with mask IoU ≥ 0.97 for every annotation against the source polygons, and a JSON that parses as COCO with the same categories.
  - Editing a material or part exports the change with uids kept; a new SAM part gets a new `p<obj>_<k>` uid.
  - A reviewed image exports `reviewed: true`.
  - Re-running the command doesn't re-import edited samples.
- Every bug and improvement gets a test that fails before its change (RED) and passes after it; threaded tests are run 3× for flakiness.
- **End-to-end:** run `python -m annofix.cvat Test --out Test_fixed` offscreen with `FakeBackend`. Fix a material, merge, mark reviewed, and check that `Test_fixed/000000_000001/` is written; screenshots check the 1920x1080 fit and the new part outlines.
- Then launch it on the user's desktop on `Test` → `Test_fixed`, with real SAM, for them to try.
