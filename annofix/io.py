"""Images and annotation files.

For image `<stem>` the annotation is `<stem>.parts.png` (uint16 part-id map, 0 = background)
plus `<stem>.json` (hierarchy and labels) in the annotations directory.
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path

import cv2
import numpy as np

from .annotation import REVIEW_STATUSES, SCHEMA_VERSION, Annotation

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def list_images(images_dir: str | Path) -> list[Path]:
    """Colour images in `images_dir` (not recursive), sorted by name. Skips depth frames
    (a 'depth' token in the name, e.g. cam0_000_depth.png) and AnnoFix's own outputs."""
    out = []
    for p in sorted(Path(images_dir).iterdir()):
        if not p.is_file() or p.suffix.lower() not in IMAGE_EXTS or p.name.endswith(".parts.png"):
            continue
        if "depth" in re.split(r"[_\-. ]+", p.stem.lower()):
            continue
        out.append(p)
    counts: dict[str, int] = {}
    for p in out:
        counts[p.stem] = counts.get(p.stem, 0) + 1
    dupes = [p.name for p in out if counts[p.stem] > 1]
    if dupes:  # x.png and x.jpg would silently share one annotation file
        raise ValueError(f"{images_dir}: images share a name with different extensions "
                         f"(each name gets one annotation): {', '.join(dupes)}")
    return out


def load_image(path: str | Path) -> np.ndarray:
    """RGB uint8 HxWx3. 8-bit grey is expanded; 16-bit (depth) images are rejected."""
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(f"cannot read image {path}")
    if img.dtype != np.uint8:
        raise ValueError(f"{path}: {img.dtype} image, expected 8-bit colour (a depth map?)")
    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    code = {3: cv2.COLOR_BGR2RGB, 4: cv2.COLOR_BGRA2RGB}.get(img.shape[2])
    if code is None:
        raise ValueError(f"{path}: {img.shape[2]} channels, expected 3 or 4")
    return cv2.cvtColor(img, code)


def annotation_paths(ann_dir: str | Path, stem: str) -> tuple[Path, Path]:
    d = Path(ann_dir)
    return d / f"{stem}.parts.png", d / f"{stem}.json"


def has_annotation(ann_dir: str | Path, stem: str) -> bool:
    """True if either file exists: a lone PNG (crash between the two writes of a first save)
    must still count, or the next save would overwrite it unseen."""
    return any(p.exists() for p in annotation_paths(ann_dir, stem))


def _fsync_dir(d: Path) -> None:
    """Make the rename itself durable (ext4 can otherwise lose it on power loss)."""
    try:
        fd = os.open(d, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError:
        pass
    finally:
        os.close(fd)


def _atomic_write(path: Path, data: bytes) -> None:
    tmp = path.with_name(path.name + ".tmp")
    try:
        with open(tmp, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    _fsync_dir(path.parent)


def save_annotation(ann: Annotation, ann_dir: str | Path, stem: str) -> None:
    if ann.in_transaction:
        raise RuntimeError(f"cannot save {stem} in the middle of an open edit; commit or roll back first")
    png_path, json_path = annotation_paths(ann_dir, stem)
    meta_bytes = json.dumps(ann.meta_dict(), indent=2).encode()  # fail before touching disk
    ok, buf = cv2.imencode(".png", ann.label_map)
    if not ok:
        raise OSError(f"PNG encode failed for {png_path}")
    png_path.parent.mkdir(parents=True, exist_ok=True)
    _atomic_write(png_path, buf.tobytes())  # PNG first: the JSON never names a missing map
    _atomic_write(json_path, meta_bytes)
    ann.mark_saved()


def load_annotation(ann_dir: str | Path, stem: str,
                    expected_size: tuple[int, int] | None = None) -> Annotation:
    """Load and reconcile. `expected_size` is the image's (H, W); a mismatch is an error.
    A PNG without its JSON is recovered: its ids become one class-less orphan object."""
    png_path, json_path = annotation_paths(ann_dir, stem)
    meta = None
    if json_path.exists():
        try:
            meta = json.loads(json_path.read_text())
        except json.JSONDecodeError as e:
            raise ValueError(f"{json_path}: malformed annotation JSON: {e}") from e
        if not isinstance(meta, dict):
            raise ValueError(f"{json_path}: malformed annotation JSON: expected an object, got {type(meta).__name__}")
        version = meta.get("schema_version")
        if version != SCHEMA_VERSION:
            raise ValueError(f"{json_path}: schema_version {version!r}; this AnnoFix reads {SCHEMA_VERSION}")
    elif not png_path.exists():
        raise FileNotFoundError(f"no annotation for {stem} in {ann_dir}")
    label_map = cv2.imread(str(png_path), cv2.IMREAD_UNCHANGED)
    if label_map is None:
        raise ValueError(f"cannot read {png_path}")
    if label_map.dtype != np.uint16 or label_map.ndim != 2:
        raise ValueError(f"{png_path}: expected single-channel uint16, got {label_map.dtype} {label_map.shape}")
    lone_png = meta is None
    if lone_png:
        meta = {"schema_version": SCHEMA_VERSION, "size": list(label_map.shape)}
    if list(label_map.shape) != list(meta.get("size", label_map.shape)):
        raise ValueError(f"{stem}: label map is {list(label_map.shape)} but the JSON says size {meta['size']}")
    if expected_size is not None and tuple(label_map.shape) != tuple(expected_size):
        h, w = label_map.shape
        raise ValueError(f"{stem}: annotation is {w}x{h} but the image is {expected_size[1]}x{expected_size[0]}")
    warnings = []
    if meta.get("review_status", "unreviewed") not in REVIEW_STATUSES:
        warnings.append(f"review_status {meta.get('review_status')!r} is not one of {REVIEW_STATUSES}; "
                        "treated as unreviewed")
        meta = {**meta, "review_status": "unreviewed"}
    if lone_png:
        warnings.append(f"no JSON next to {png_path.name}: part ids recovered from the label map, labels lost")
    try:
        ann = Annotation.from_meta(label_map, meta)
        ann.warnings.extend(warnings)
        ann.reconcile()
    except (KeyError, TypeError, AttributeError) as e:  # e.g. a part without "id", "objects": null
        raise ValueError(f"{json_path}: malformed annotation JSON: {type(e).__name__}: {e}") from e
    return ann


def load(ann_dir: str | Path, stem: str) -> tuple[np.ndarray, dict]:
    """Downstream helper: (uint16 part-id map, metadata dict) for one image."""
    ann = load_annotation(ann_dir, stem)
    return ann.label_map, ann.meta_dict()
