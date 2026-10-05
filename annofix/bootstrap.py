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
