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
