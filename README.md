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
Existing annotations are skipped; `--overwrite` replaces unreviewed ones only (reviewed or flagged
ones are never replaced). Depth frames (`*_depth.png`) in the image folder are ignored.
On the RTX 3050 a 1280x720 frame takes ~5 s with the default `base_plus` model.

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
