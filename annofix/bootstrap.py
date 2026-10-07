"""python -m annofix.bootstrap --images DIR --out DIR

Generate starting annotations from SAM 2.1 automatic masks. An existing annotation is never
replaced unless --overwrite, and even then only if bootstrap wrote it and nobody has changed it
since (so in-progress, reviewed, flagged and pipeline-written work is always kept). Raw masks
are cached in <out>/.amg_cache, tagged with the image they came from and the SAM settings, so
--regroup can re-run the grouping without the GPU and a stale cache is never used silently.
"""
from __future__ import annotations

import annofix  # noqa: F401  (thread cap before numpy)

import argparse
import hashlib
import os
import sys
import time
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np

from .io import annotation_paths, has_annotation, list_images, load_image, save_annotation
from .overlaps import GroupingParams, candidates_to_annotation


def image_fingerprint(image: np.ndarray) -> str:
    """Identifies the decoded pixels, so a cache built from another recording is detected."""
    return hashlib.sha1(np.ascontiguousarray(image).tobytes()).hexdigest()


def cache_path(out_dir: str | Path, stem: str) -> Path:
    return Path(out_dir) / ".amg_cache" / f"{stem}.npz"


def written_marker(out_dir: str | Path, stem: str) -> Path:
    return Path(out_dir) / ".amg_cache" / f"{stem}.written"


def save_cache(path: Path, masks: np.ndarray, scores: np.ndarray,
               image_hash: str = "", settings: str = "") -> None:
    """Atomic: an interrupted write (Ctrl+C mid-compression) leaves no cache file behind."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    try:
        with open(tmp, "wb") as f:
            np.savez_compressed(f, packed=np.packbits(masks, axis=-1), shape=np.array(masks.shape),
                                scores=np.asarray(scores, np.float32),
                                image_hash=np.array(image_hash), settings=np.array(settings))
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def load_cache(path: Path) -> tuple[np.ndarray, np.ndarray, str, str]:
    """(masks, scores, image_hash, settings); hash/settings are "" for untagged caches."""
    with np.load(path) as d:
        _, _, w = d["shape"]
        masks = np.unpackbits(d["packed"], axis=-1, count=int(w)).astype(bool)
        image_hash = str(d["image_hash"]) if "image_hash" in d.files else ""
        settings = str(d["settings"]) if "settings" in d.files else ""
        return masks, d["scores"], image_hash, settings


def _files_digest(out_dir: Path, stem: str) -> str | None:
    try:
        return hashlib.sha1(b"\0".join(p.read_bytes() for p in annotation_paths(out_dir, stem))).hexdigest()
    except OSError:
        return None


def _untouched_since_bootstrap(out_dir: Path, stem: str) -> bool:
    marker = written_marker(out_dir, stem)
    return marker.exists() and marker.read_text().strip() == _files_digest(out_dir, stem)


def _usable_cache(cpath: Path, image_hash: str, generate, settings: str):
    """Cached (masks, scores), or None when SAM should (re)generate. Raises ValueError in
    regroup mode (generate is None) when the cache is unreadable or from another image."""
    try:
        masks, scores, cached_hash, cached_settings = load_cache(cpath)
    except (OSError, ValueError, KeyError, EOFError, zipfile.BadZipFile) as e:
        if generate is None:
            raise ValueError(f"{cpath}: unreadable mask cache ({e}); delete it or run without --regroup") from e
        return None
    if cached_hash != image_hash:
        if generate is None:
            raise ValueError(f"{cpath}: cached masks were made from a different image (re-recorded "
                             "frame?); delete the cache or run without --regroup")
        return None
    if generate is not None and cached_settings != settings:
        return None
    return masks, scores


def bootstrap_image(image_path, out_dir, generate, params: GroupingParams, *,
                    overwrite: bool = False, use_cache: bool = True, settings: str = "") -> str:
    """`generate` is None in regroup mode (cached masks only). `settings` identifies the SAM
    model/parameters; a cache made with other settings is regenerated."""
    image_path, out_dir = Path(image_path), Path(out_dir)
    stem = image_path.stem
    if has_annotation(out_dir, stem):
        if not overwrite:
            return "skipped"
        if not _untouched_since_bootstrap(out_dir, stem):
            return "protected"
    image = load_image(image_path)
    image_hash = image_fingerprint(image)
    cpath = cache_path(out_dir, stem)
    cached = _usable_cache(cpath, image_hash, generate, settings) if use_cache and cpath.exists() else None
    if cached is not None:
        masks, scores = cached
        status = "cached"
    elif generate is None:
        return "no-cache"
    else:
        masks, scores = generate(image)
        save_cache(cpath, masks, scores, image_hash, settings)
        status = "generated"
    if masks.shape[1:] != image.shape[:2]:
        raise ValueError(f"{cpath.name}: cached masks are {masks.shape[2]}x{masks.shape[1]} but the "
                         f"image is {image.shape[1]}x{image.shape[0]}; delete the cache file")
    ann = candidates_to_annotation(masks, scores, params, image=os.path.relpath(image_path, out_dir))
    if has_annotation(out_dir, stem) and not _untouched_since_bootstrap(out_dir, stem):
        return "protected"  # someone saved this image while SAM was working: keep their version
    save_annotation(ann, out_dir, stem)
    written_marker(out_dir, stem).write_text(_files_digest(out_dir, stem))
    return status


def _positive_int(text: str) -> int:
    value = int(text)
    if value < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return value


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
    ap.add_argument("--limit", type=_positive_int, help="only the first N images (N >= 1)")
    ap.add_argument("--overwrite", action="store_true",
                    help="replace annotations bootstrap wrote that nobody has changed since "
                         "(edited, reviewed, flagged or pipeline-written ones are never replaced)")
    ap.add_argument("--regroup", action="store_true", help="only re-group cached masks; no SAM, no GPU")
    ap.add_argument("--no-cache", action="store_true", help="ignore cached masks and run SAM again")
    args = ap.parse_args(argv)

    if not args.images.is_dir():
        print(f"no such folder: {args.images}", file=sys.stderr)
        return 1
    try:
        images = list_images(args.images)
    except ValueError as e:
        print(e, file=sys.stderr)
        return 1
    if args.limit:
        images = images[: args.limit]
    if not images:
        print(f"no colour images in {args.images}", file=sys.stderr)
        return 1

    generate = None
    settings = ""
    if not args.regroup:
        from .sam_backend import Sam2Backend, amg_settings

        backend = Sam2Backend(args.sam_model, args.device, args.checkpoint)
        settings = amg_settings(backend.model_name, args.points_per_side)
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
                                     overwrite=args.overwrite, use_cache=not args.no_cache,
                                     settings=settings)
        except ValueError as e:
            status = "error"
            print(f"  {path.name}: {e}", file=sys.stderr)
        counts[status] += 1
        print(f"[{i}/{len(images)}] {path.name}: {status} ({time.perf_counter() - t:.1f}s)", flush=True)
    print("done: " + ", ".join(f"{k} {v}" for k, v in sorted(counts.items())))
    if counts["error"]:
        return 2
    if counts["no-cache"] == len(images):
        print("nothing is cached for these images yet: run once without --regroup", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
