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
