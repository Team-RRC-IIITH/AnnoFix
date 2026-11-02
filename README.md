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
- `Test/` is never written (`--out` must be outside it). Every save writes that sample to
  `Test_fixed/<sample>/` in the same layout and files, so there is no export step; on start, any
  output older than its working copy (say, an export that failed) is written again.
- The working copy lives in `Test_fixed/.annofix/` (AnnoFix's own format). A sample already
  there is never re-imported, so re-running the command keeps your edits; new samples in
  `Test/` are added.
- The 10 materials are the COCO categories (keys `1`–`9`, `0`); object classes and part-name
  suggestions are collected from the data into `Test_fixed/.annofix/labels.yaml`.
- **Object = its parts + a `body` part.** The editor gives each pixel one owner, so on import an
  object's pixels not covered by its parts become a part named `body`, carrying the object's
  material. On export the object mask is again the union of its pieces and the object's
  category is the body's material (with no body: the material covering most of the object). A
  new object's first part, and a part split off into an object of its own, is its body; rename
  a body and it is exported as a part. Overlap slivers between different objects go to the
  smaller object; anything lost that way is printed when the sample is imported.
- The tree shows the pipeline's confidence after each label (`#12 lid · plastic · 49%`), and
  `P` jumps to the next part with a missing label.
- uids, object ids, confidences and per-annotation `reviewed` flags are kept. New objects get
  an `object_id` after the source's (uid `o<N>`, confidence 100) and new parts uid
  `p<object>_<k>` with `k` after the source's; these never change between saves and a deleted
  annotation's uid is never handed out again (the numbers can have gaps).
- Anything without a material yet is left out of the output (the status bar says what) rather
  than given a guessed one.
- An image comes in as reviewed only if the source marks every annotation reviewed and its
  labels pass the checks. Marking it reviewed in AnnoFix sets `reviewed: true` on all its
  annotations.
- COCO polygons cannot hold holes, and a part's polygons may differ from the pipeline's by
  about a pixel at the boundary.

## Fixing annotations (GUI)

```bash
uv run python -m annofix --images IMAGES_DIR --annotations ANNOTATIONS_DIR [--labels labels.yaml] [--annotator NAME]
#   SAM options: [--sam-model tiny|small|base_plus|large] [--device cuda|cpu] [--checkpoint FILE.pt] [--no-sam]
```
The window is laid out like CVAT's annotation view. Top: save, undo/redo, previous/next image
and next unreviewed, the image and how many are reviewed, flag, **Mark reviewed**, help. Far left:
the tools (select, brush, eraser, points, SAM) and auto-clean, outlines, protect, fit; every
button's tooltip gives its key. Then the images (○ no annotation yet · • unreviewed · ✓ reviewed
· ⚑ flagged) and the image with its masks; part borders are drawn darker, so neighbouring parts of
the same colour stay distinct. Right: objects → parts (tick boxes show/hide; right-click to move,
split off, clean or delete), the selected part's class, material and name, and appearance (colour
by, opacity). The keys are unchanged.

Work saves 2 s after your last finished edit (never in the middle of a stroke), every 30 s, when
you change image, and on close. Ctrl+C in the terminal also saves and closes. `F1` or `?` lists
every shortcut.

| To do this | Press / click |
|---|---|
| Select a part · add/remove one | click · Ctrl+click (canvas or tree) |
| Set material | `1`–`9`, `0` (10th) — keypad too — or a material button |
| Part name · object class | suggestion buttons, or type (applied on Enter or as soon as you do anything else; `Esc` drops it) — for every selected part |
| Next part with a missing label | `P` |
| Give the selected parts the last name + material again | `.` |
| Brush (paint the selected part) · eraser · select | `B` · `E` · `V` |
| Points: drag the selected part's outline points · add one · delete one | `G`, then drag · double-click an edge · Ctrl+click a point |
| Brush size | `[` `]` |
| Protect other parts while painting | toolbar toggle |
| Cancel a stroke | `Esc` |
| Merge selected parts · merge their objects | `M` · `Shift+M` |
| Delete part(s) · delete object | `Del` · `Shift+Del` |
| Undo · redo | `Ctrl+Z` · `Ctrl+Shift+Z` |
| **Improve** the selected part with SAM (from its own mask; `Tab`: other answers, the mask smoothed) | `R` |
| Redraw a part that is plain wrong from clicks | `Esc`, then click it with the SAM tool |
| SAM: include · exclude · box | left click · right click · Shift+drag |
| SAM: next/previous candidate · replace · add · cancel | `Tab`/`Shift+Tab` · `Enter` · `Shift+Enter` · `Esc` |
| Auto-clean every mask (SAM, else smoothing) · accept · discard | `Shift+R` · `Enter` · `Esc` |
| New part · new object (with SAM: from clicks) | `N` · `Shift+N` |
| New empty part, brush ready (also with SAM) | `Ctrl+N` |
| Hide masks while held · solo object · outlines | `Space` · `S` · `O` |
| Next/previous object | `J` · `K` |
| Previous/next image · next unreviewed | `A` · `D` · `Ctrl+U` |
| Mark reviewed (checks every label) and open the next unreviewed · flag | `Ctrl+Enter` · `F` |
| Save now | `Ctrl+S` |
| Zoom · pan · fit the image | mouse wheel · middle- or right-drag (right-drag not in SAM mode) · `Home` |
| All shortcuts | `F1` · `?` |

**Improve one part (`R`)** keeps the part and makes its mask better: SAM is asked with prompts
taken from the mask itself (its box, points inside it, points on the neighbours to keep out), so
no clicks are needed. If SAM's closest answer matches the mask well, it is shown first; if not
(SAM found something else), the mask smoothed is shown first. `Tab` steps through SAM's answers
and the smoothed mask; clicks refine SAM's answer; `Enter` writes the chosen mask into the same
part (name, material, object and CVAT uid stay), one undo step; `Esc` leaves it as it was.

**Auto-clean (`Shift+R`)** cleans every mask in the image at once. SAM is asked again for each
object and part, prompted from the mask it has (its box, points inside it, points on its
neighbours to keep out), and its answer is taken where it matches the old mask well: edges then
follow the image. Where it doesn't match, the mask is only smoothed (staircase edges, specks and
pinholes). The changes show as a preview; `Enter` accepts them as one undo step, `Esc` drops
them. Hidden parts are never changed. Without SAM, `Shift+R` smooths only.

Points (`G`) suit straight edges such as boxes and lids: drag a corner and only the pixels
along the edges at that corner change (the part follows the new edges exactly there); the rest
of the part stays as it was. Holes stay holes, other pieces of the part are never cut, a move
that would make the outline cross itself is refused, and protected or hidden parts keep their
pixels. Tiny specks get no points. For ragged or bleeding edges, use the brush or SAM.

SAM loads on first use (`R`), in the background; the status bar shows its state. The preview
colours what Enter would do: white kept, green added, orange taken from another part, red
removed. Hidden parts (and parts outside a solo object) are never changed — by SAM, the brush or
the eraser — and "Protect other parts" applies to SAM as it does to the brush. If SAM fails to
load, its button disappears and the status bar says why; the brush keeps working. If the GPU
is full (CVAT's SAM, say, holds it), SAM loads on the CPU instead and says so; `--device cpu`
never touches the GPU. SAM's pinholes and stray specks are cleaned up as smoothing does.

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
