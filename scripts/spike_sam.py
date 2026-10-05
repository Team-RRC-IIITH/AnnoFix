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
