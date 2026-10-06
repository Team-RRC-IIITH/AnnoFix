# M0 measurements (2026-10-03, run 2026-10-05)

Machine: RTX 3050 Laptop 6 GB, driver 595.84; torch 2.14.1+cu130 (CUDA 13.0), sam-2 @ 2b90b9f,
opencv-python-headless 5.0.0.93, PyQt6 6.11.0; image `cam0_100_color.png`, 1280x720.
BLAS/OpenMP threads capped at 10 (`torch.set_num_threads(10)` on CPU).

| model | device | load s | encode ms | decode ms | AMG s | AMG masks | peak VRAM MB |
|---|---|---|---|---|---|---|---|
| tiny | cuda | 2.0 | 70 | 7.0 | 3.4 | 55 | 1890 (502 without AMG) |
| small | cuda | 11.6* | 78 | 7.2 | 3.3 | 59 | 1924 |
| base_plus | cuda | 138.2* | 130 | 7.0 | 3.5 | 87 | 2070 |
| tiny | cpu | 2.0 | 1080 | 23.2 | – | – | – |
| small | cpu | 0.6 | 1327 | 24.3 | – | – | – |
| base_plus | cpu | 0.8 | 2283 | 23.8 | – | – | – |

\* includes the one-time Hugging Face checkpoint download; cached loads take ~1–2 s.
Encode = `set_image` (once per image); decode = one click (`predict`, multimask), median of 20.
AMG = `generate` with defaults (32x32 points, batch 64), one run, bf16 autocast.

Decision rule: CUDA default = base_plus if its encode < 500 ms and peak VRAM < 3000 MB, else small.
CPU default = small if its encode < 5000 ms, else tiny.

**Decision: cuda -> base_plus, cpu -> small** (unchanged from `DEFAULT_MODEL`).
Qt check: window opened on platform `wayland` with torch + cv2 loaded in the same process.

Implications for Plans 2-3:
- GPU encode is 130 ms, so computing the embedding on image load is enough; prefetch/LRU is not needed on GPU.
- CPU encode is ~1.3 s (small): show a "preparing SAM" state on navigate for CPU users; clicks (24 ms) stay interactive.
- base_plus finds more masks (87 vs 55-59) at the same AMG time, so it is the right bootstrap model.
- sam2 emits a `torch.jit.script` FutureWarning on torch 2.14; harmless.

## Bootstrap on 10 real frames (6 x 1280x720 cam0, 4 x 640x480 cam1), base_plus on CUDA

| frame | seconds | objects | parts |
|---|---|---|---|
| cam0_000 | 5.1 | 56 | 101 |
| cam0_040 | 4.9 | 63 | 104 |
| cam0_080 | 4.9 | 65 | 108 |
| cam0_120 | 4.8 | 64 | 103 |
| cam0_160 | 4.8 | 59 | 104 |
| cam0_195 | 4.9 | 64 | 109 |
| cam1_000 | 3.8 | 50 | 66 |
| cam1_060 | 3.8 | 52 | 68 |
| cam1_120 | 3.8 | 50 | 68 |
| cam1_180 | 3.8 | 51 | 65 |

`--regroup --overwrite` over all 10 from the mask cache: 1.8 s total, no GPU. Masks line up with
the image at both resolutions.

**AMG settings changed (sam_backend.generate defaults):**
- `pred_iou_thresh 0.8 -> 0.7`, `stability_score_thresh 0.95 -> 0.88`. With SAM's defaults
  the robot arm, laptop and tripod camera had *no candidate mask at all* (83 masks on cam0_120),
  so no grouping parameter could recover them. Loosened: 170 masks, all three found, 4.1 s.
  Tried and rejected: `use_m2m` (13.3 s, 3.4 GB, lost the arm) and 48x48 points (8.9 s, 3.4 GB,
  marginal gain). Pinned by `test_generate_recalls_lab_objects`.
- `points_per_batch 64 -> 32`: same 170 masks, same speed (4.2 s), but 2.8 GB reserved instead
  of 4.6 GB; at 64 the run hit CUDA OOM retries. Pinned by `test_generate_leaves_gpu_headroom`.

GroupingParams changes: none. Diagnosis on cam0_120: the missing objects had zero candidates
(an AMG issue, fixed above); the teal-shirt person (10-13%, touching top+left borders) is dropped
by the border rule, which is acceptable since people are not grasp targets.

Observed problems to carry into Plans 2-3:
- Background surfaces still become objects: the wooden table top (7.9% of the frame, no border
  contact) and wall/whiteboard patches. One Delete each; not worth risking real objects with a
  tighter max-area filter.
- The robot arm splits into two objects (upper links vs. base) -> annotator merges (Plan 2 needs a
  quick "merge objects" path, not only merge parts).
- The black cable on the black mat is not found; the cable on the wooden table is.
- ~50-65 objects per frame is a lot to review: Plan 2 should make "delete object" and
  "next object" one keystroke each.
- Thermals: AMG + SAM tests push the CPU package to 95-96 C on this laptop; batch bootstrap runs
  should be spaced out or run with the lid open / on a cooling pad.

## Interactive SAM in the editor (Plan 3)

Measured through the real window (offscreen) with `base_plus` on the RTX 3050, frame cam0_120 (1280x720):

| step | ms |
|---|---|
| first SAM use (model load from cache + embed + predict) | 3647 |
| refinement click (exclude point, prev logits fed back) | 49 |
| another first click on the same image | 25–27 |
| next image: embed + first click | 165 |

Peak memory of the SAM-enabled editor: 1.9 GB RSS (the gated real-SAM test suite: 2.5 GB).
The CPU package reaches ~88–92 °C after a burst of SAM work; let it cool between batch runs.

Thin parts, checked on 4x zoomed crops at native resolution (exact pixels picked from the image):
- **tripod leg** (~5–6 px wide): the smallest candidate (Tab to it; the default is the
  highest-scoring whole tripod) follows the single leg within ~1 px — clean.
- **cable on the table** (~2–3 px wide): the middle candidate follows most of the cable, but is
  blobby around the click (~10–15 px spill onto the table) and misses part of the loop.

**M6 decision: zoomed-crop SAM is not needed now.** Parts a few pixels wide and up segment cleanly
at 1280x720; only hair-thin structures like cables are blobby, and those are fixable with an exclude
click plus the eraser and matter less for a grasping dataset. If thin objects become important,
the follow-up is: crop the target's bbox x1.5–2 (at least 256 px), offset the prompts, and paste
the mask back. Embedding prefetch/LRU is also unnecessary on the GPU (165 ms per new image).
