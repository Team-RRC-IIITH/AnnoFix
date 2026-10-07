"""Pipeline masks in CVAT-style COCO ("COCO 1.0" export) <-> AnnoFix.

    uv run python -m annofix.cvat SRC --out OUT [--annotator NAME] [--export-only] [SAM options]

Input layout, one folder per sample:
    SRC/<sample>/images/<file>.png
    SRC/<sample>/annotations/instances_default.json  categories = materials; polygons; attributes:
        uid, kind (object|part), object_id, object_name, part_name, confidence_pct, reviewed, occluded
    SRC/<sample>/export_manifest.json, SRC/<sample>/cvat_labels.json

AnnoFix edits with one owner per pixel, so on import an object becomes its parts plus a "body"
part: the object minus its parts, carrying the object's material. Export rebuilds the object
mask as the union of its pieces, so OUT/<sample>/ holds the same overlapping object + part masks
in the same files. The working copy lives in OUT/.annofix/ (AnnoFix's own lossless format); a
sample already there is never re-imported, so edits are never overwritten. SRC is never written.

Polygon convention (measured on pipeline output): pixel (x, y) is the unit square centred on
(x + 0.5, y + 0.5). Import rasterizes with fillPoly(round(p - 0.5)); export writes boundary-pixel
centres + 0.5, so importing an export is exact. COCO polygons cannot hold holes.
"""
from __future__ import annotations

import annofix  # noqa: F401  (thread cap before numpy)

import argparse
import json
import os
import re
import shutil
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import yaml

from .annotation import AnnObject, Annotation, Part
from .io import _atomic_write, has_annotation, load_annotation, save_annotation
from .overlaps import resolve_overlaps

ANN_JSON = Path("annotations") / "instances_default.json"
MANIFEST = "export_manifest.json"
CVAT_LABELS = "cvat_labels.json"
WORK_DIR = ".annofix"
LABELS = "labels.yaml"
PART_UID = re.compile(r"^p(\d+)_(\d+)$")


@dataclass(frozen=True)
class SampleImage:
    stem: str          # AnnoFix's name for this image
    sample_dir: Path   # SRC/<sample>
    entry: dict        # the COCO images[] entry

    @property
    def image_path(self) -> Path:
        return self.sample_dir / "images" / self.entry["file_name"]


@dataclass
class ImportReport:
    imported: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)  # already in the working copy (kept as edited)


def find_samples(src_root) -> list[SampleImage]:
    out = []
    for js in sorted(Path(src_root).glob(f"*/{ANN_JSON.as_posix()}")):
        sample_dir = js.parent.parent
        entries = json.loads(js.read_text()).get("images", [])
        for e in entries:
            stem = sample_dir.name if len(entries) == 1 else f"{sample_dir.name}__{Path(e['file_name']).stem}"
            out.append(SampleImage(stem, sample_dir, e))
    return out


# ---- polygons ------------------------------------------------------------------------------------

def rasterize(segmentation, height: int, width: int) -> np.ndarray:
    mask = np.zeros((height, width), np.uint8)
    for poly in segmentation or []:
        pts = np.round(np.asarray(poly, np.float64).reshape(-1, 2) - 0.5).astype(np.int32)
        if len(pts):
            cv2.fillPoly(mask, [pts], 1)  # one polygon at a time: overlapping pieces must union
    return mask.astype(bool)


def polygons(mask: np.ndarray, offset: tuple[int, int] = (0, 0)) -> list[list[float]]:
    """One polygon per connected piece (outer boundary), in the pipeline's coordinates. `offset`
    (x, y) places a cropped mask back in the image."""
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE,
                                   offset=offset)
    out = []
    for c in contours:
        pts = c.reshape(-1, 2).astype(np.float64) + 0.5
        while len(pts) < 3:  # 1-2 pixel pieces: COCO wants at least 3 points
            pts = np.vstack([pts, pts[-1:]])
        out.append([float(v) for v in pts.ravel()])
    return out


def _bbox(mask: np.ndarray) -> list[float]:
    rows, cols = np.flatnonzero(mask.any(1)), np.flatnonzero(mask.any(0))
    return [float(cols[0]), float(rows[0]), float(cols[-1] - cols[0] + 1), float(rows[-1] - rows[0] + 1)]


# ---- import --------------------------------------------------------------------------------------

def _read_json(path: Path, default=None):
    return json.loads(path.read_text()) if path.exists() else default


def import_image(sample: SampleImage, coco: dict, manifest: dict | None = None) -> Annotation:
    e = sample.entry
    h, w = int(e["height"]), int(e["width"])
    cats = {c["id"]: c["name"] for c in coco.get("categories", [])}
    handles = {uid: v.get("object") for uid, v in (manifest or {}).get("annotations", {}).items()}
    anns = [a for a in coco.get("annotations", []) if a.get("image_id") == e["id"]]
    objects = [a for a in anns if a.get("attributes", {}).get("kind") != "part"]
    parts_of = defaultdict(list)
    for a in anns:
        if a.get("attributes", {}).get("kind") == "part":
            parts_of[a["attributes"].get("object_id")].append(a)
    known_ids = {a.get("attributes", {}).get("object_id") for a in objects}
    for oid in [k for k in parts_of if k not in known_ids]:  # parts whose object has no mask
        first = parts_of[oid][0]
        objects.append({"category_id": first.get("category_id"), "segmentation": [], "_parts_only": True,
                        "attributes": {"object_id": oid, "object_name": first["attributes"].get("object_name")}})

    warnings = []
    obj_masks = [rasterize(a.get("segmentation"), h, w) for a in objects]
    for i, a in enumerate(objects):  # an object is at least the union of its parts
        for p in parts_of.get(a.get("attributes", {}).get("object_id"), []):
            obj_masks[i] |= rasterize(p.get("segmentation"), h, w)
    owner = resolve_overlaps(np.stack(obj_masks)) if obj_masks else np.zeros((h, w), np.int32)

    label = np.zeros((h, w), np.uint16)
    out_objects: list[AnnObject] = []
    next_id = 1
    for i, a in enumerate(objects):
        at = a.get("attributes", {})
        region = owner == i + 1
        material = cats.get(a.get("category_id"))
        obj = AnnObject(next_id, at.get("object_name") or None, [], {
            k: v for k, v in {"uid": at.get("uid"), "object_id": at.get("object_id"),
                              "object": handles.get(at.get("uid")), "material": material,
                              "confidence_pct": at.get("confidence_pct"),
                              "occluded": at.get("occluded")}.items() if v is not None})
        next_id += 1
        plist = parts_of.get(at.get("object_id"), [])
        part_owner = (resolve_overlaps(np.stack([rasterize(p.get("segmentation"), h, w) & region for p in plist]))
                      if plist else np.zeros((h, w), np.int32))
        body = region & (part_owner == 0)
        if body.any() and not a.get("_parts_only"):
            label[body] = next_id
            obj.parts.append(Part(next_id, "body", material, {
                k: v for k, v in {"role": "body", "uid": at.get("uid"),
                                  "confidence_pct": at.get("confidence_pct")}.items() if v is not None}))
            next_id += 1
        for k, p in enumerate(plist):
            pat = p.get("attributes", {})
            pixels = part_owner == k + 1
            if not pixels.any():
                warnings.append(f"part {pat.get('uid')} has no pixels left after overlaps were resolved")
                continue
            label[pixels] = next_id
            obj.parts.append(Part(next_id, pat.get("part_name") or None, cats.get(p.get("category_id")), {
                kk: v for kk, v in {"role": "part", "uid": pat.get("uid"),
                                    "confidence_pct": pat.get("confidence_pct"),
                                    "occluded": pat.get("occluded")}.items() if v is not None}))
            next_id += 1
        if obj.parts:
            out_objects.append(obj)
        else:
            warnings.append(f"object {at.get('uid')} has no pixels left after overlaps were resolved")

    reviewed = bool(anns) and all(a.get("attributes", {}).get("reviewed") for a in anns)
    ann = Annotation(label, out_objects, next_id=next_id, source="pipeline",
                     review_status="reviewed" if reviewed else "unreviewed",
                     attrs={"cvat_sample": sample.sample_dir.name, "cvat_image_id": e["id"],
                            "cvat_file": e["file_name"]})
    ann.warnings.extend(warnings)
    return ann


def _labels_from(coco_files: list[dict]) -> dict:
    materials: list[str] = []
    parts: dict[str, set] = defaultdict(set)
    for coco in coco_files:
        for c in sorted(coco.get("categories", []), key=lambda c: c["id"]):
            if c["name"] not in materials:
                materials.append(c["name"])
        for a in coco.get("annotations", []):
            at = a.get("attributes", {})
            name = at.get("object_name")
            if name:
                parts[name]
                if at.get("kind") == "part" and at.get("part_name"):
                    parts[name].add(at["part_name"])
    return {"materials": materials, "objects": {k: sorted(v) for k, v in sorted(parts.items())}}


def _link_image(src: Path, dst: Path) -> None:
    if dst.exists() or dst.is_symlink():
        return
    try:
        os.symlink(src.resolve(), dst)
    except OSError:
        shutil.copy2(src, dst)


def import_dataset(src_root, work_dir) -> ImportReport:
    """Bring every sample under `src_root` into the working copy. Samples already there are
    skipped (their edits are kept); labels.yaml is regenerated from all samples."""
    work = Path(work_dir)
    img_dir, ann_dir = work / "images", work / "annotations"
    img_dir.mkdir(parents=True, exist_ok=True)
    ann_dir.mkdir(parents=True, exist_ok=True)
    report = ImportReport()
    cocos: dict[Path, dict] = {}
    for s in find_samples(src_root):
        js = s.sample_dir / ANN_JSON
        coco = cocos.setdefault(js, json.loads(js.read_text()))
        dst = img_dir / f"{s.stem}{s.image_path.suffix}"
        _link_image(s.image_path, dst)
        if has_annotation(ann_dir, s.stem):
            report.skipped.append(s.stem)
            continue
        ann = import_image(s, coco, _read_json(s.sample_dir / MANIFEST))
        ann.image = os.path.relpath(dst, ann_dir)
        save_annotation(ann, ann_dir, s.stem)
        report.imported.append(s.stem)
    if cocos:
        _atomic_write(work / LABELS, yaml.safe_dump(_labels_from(list(cocos.values())), sort_keys=False,
                                                    allow_unicode=True).encode())
    return report


# ---- export --------------------------------------------------------------------------------------

def _material_of_object(obj: AnnObject) -> str | None:
    body = next((p for p in obj.parts if p.attrs.get("role") == "body" and p.material), None)
    if body is not None:
        return body.material
    if obj.attrs.get("material"):
        return obj.attrs["material"]
    counts = Counter(p.material for p in obj.parts if p.material)
    return counts.most_common(1)[0][0] if counts else None


def export_image(ann: Annotation, sample: SampleImage, out_root) -> Path:
    """Write OUT/<sample>/ for this image, in the source's layout. Returns the sample folder."""
    out_dir = Path(out_root) / sample.sample_dir.name
    (out_dir / "images").mkdir(parents=True, exist_ok=True)
    (out_dir / "annotations").mkdir(parents=True, exist_ok=True)
    dst_img = out_dir / "images" / sample.entry["file_name"]
    if not dst_img.exists():
        shutil.copy2(sample.image_path, dst_img)

    src_coco = json.loads((sample.sample_dir / ANN_JSON).read_text())
    single_image = len(src_coco.get("images", [])) <= 1
    base = _read_json(out_dir / ANN_JSON) or src_coco
    categories = list(base.get("categories", []))
    cat_id = {c["name"]: c["id"] for c in categories}

    def category(material: str | None) -> int:
        material = material or (categories[0]["name"] if categories else "unknown")
        if material not in cat_id:  # a material the source didn't have
            cat_id[material] = max(cat_id.values(), default=0) + 1
            categories.append({"id": cat_id[material], "name": material, "supercategory": ""})
        return cat_id[material]

    image_id = sample.entry["id"]
    reviewed = ann.review_status == "reviewed"
    lm = ann.label_map
    new_anns, manifest_anns = [], {}
    next_object_id = max((o.attrs["object_id"] for o in ann.objects
                          if isinstance(o.attrs.get("object_id"), int)), default=-1) + 1

    def record(mask, material, attrs, handle):
        x, y, w, h = (int(v) for v in _bbox(mask))
        crop = mask[y:y + h, x:x + w]  # trace only around the mask, not the whole frame
        new_anns.append({"id": 0, "image_id": image_id, "category_id": category(material),
                         "segmentation": polygons(crop, (x, y)), "area": int(np.count_nonzero(crop)),
                         "bbox": [float(x), float(y), float(w), float(h)], "iscrowd": 0, "attributes": attrs})
        manifest_anns[attrs["uid"]] = {
            "kind": attrs["kind"], "object": handle, "object_id": attrs["object_id"],
            "object_name": attrs["object_name"], "part_name": attrs["part_name"], "material": material,
            "confidence": attrs["confidence_pct"] / 100}

    for obj in ann.objects:
        part_masks = {p.id: lm == p.id for p in obj.parts}
        mask = np.logical_or.reduce(list(part_masks.values())) if part_masks else np.zeros(lm.shape, bool)
        if not mask.any():
            continue
        oid = obj.attrs.get("object_id")
        if not isinstance(oid, int):
            oid, next_object_id = next_object_id, next_object_id + 1
        material = _material_of_object(obj)
        name = obj.cls or ""
        handle = obj.attrs.get("object") or f"annofix_{oid:03d}"
        record(mask, material, {"uid": obj.attrs.get("uid") or f"o{oid}", "kind": "object", "object_id": oid,
                                "object_name": name, "part_name": "",
                                "confidence_pct": int(obj.attrs.get("confidence_pct", 100)),
                                "reviewed": reviewed, "occluded": bool(obj.attrs.get("occluded", False))}, handle)
        taken = [int(m.group(2)) for p in obj.parts if (m := PART_UID.match(str(p.attrs.get("uid", ""))))]
        next_k = max(taken, default=0) + 1
        for p in obj.parts:
            if p.attrs.get("role") == "body":
                continue
            pmask = part_masks[p.id]
            if not pmask.any():
                continue
            uid = p.attrs.get("uid")
            if not uid:
                uid, next_k = f"p{oid}_{next_k}", next_k + 1
            record(pmask, p.material or material,
                   {"uid": uid, "kind": "part", "object_id": oid, "object_name": name, "part_name": p.name or "",
                    "confidence_pct": int(p.attrs.get("confidence_pct", 100)), "reviewed": reviewed,
                    "occluded": bool(p.attrs.get("occluded", False))}, handle)

    others = [a for a in base.get("annotations", []) if a.get("image_id") != image_id]
    for i, a in enumerate(others + new_anns, 1):
        a["id"] = i
    coco_out = dict(base)
    coco_out["categories"] = categories
    coco_out["annotations"] = others + new_anns
    _atomic_write(out_dir / ANN_JSON, json.dumps(coco_out).encode())

    src_manifest = _read_json(sample.sample_dir / MANIFEST) or {"file_name": sample.entry["file_name"]}
    if single_image:
        manifest = {**src_manifest, "annotations": manifest_anns}
    else:  # one manifest for several images: merge this image's entries into what is there
        prev = _read_json(out_dir / MANIFEST) or src_manifest
        manifest = {**prev, "annotations": {**prev.get("annotations", {}), **manifest_anns}}
    if "classes" in manifest:
        manifest["classes"] = [c["name"] for c in sorted(categories, key=lambda c: c["id"])]
    _atomic_write(out_dir / MANIFEST, json.dumps(manifest, indent=1).encode())

    src_labels = sample.sample_dir / CVAT_LABELS
    if src_labels.exists():
        labels = json.loads(src_labels.read_text())
        known = {lab["name"] for lab in labels}
        for c in categories:  # keep CVAT's label list in step with any new material
            if c["name"] not in known and labels:
                labels.append({**labels[0], "name": c["name"]})
        _atomic_write(out_dir / CVAT_LABELS, json.dumps(labels, indent=1).encode())
    return out_dir


def export_dataset(src_root, work_dir, out_root, only_missing: bool = False) -> list[str]:
    ann_dir = Path(work_dir) / "annotations"
    written = []
    for s in find_samples(src_root):
        if not has_annotation(ann_dir, s.stem):
            continue
        if only_missing and (Path(out_root) / s.sample_dir.name / ANN_JSON).exists():
            continue
        export_image(load_annotation(ann_dir, s.stem), s, out_root)
        written.append(s.stem)
    return written


def attach_auto_export(session, src_root, out_root) -> None:
    """Export the open image every time the editor saves it."""
    samples = {s.stem: s for s in find_samples(src_root)}

    def on_change(kind: str) -> None:
        if kind != "saved" or session.stem not in samples:
            return
        try:
            export_image(session.ann, samples[session.stem], out_root)
        except Exception as e:  # noqa: BLE001 - an export problem must never break editing
            session.add_message(f"export of {session.stem} failed: {type(e).__name__}: {e}")

    session.subscribe(on_change)


# ---- command line --------------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    from .__main__ import add_sam_arguments, run_window, sam_factory_from_args

    ap = argparse.ArgumentParser(prog="python -m annofix.cvat",
                                 description="Fix pipeline masks stored as CVAT-style COCO, one folder per sample.")
    ap.add_argument("src", type=Path, help="folder of samples: <sample>/annotations/instances_default.json")
    ap.add_argument("--out", required=True, type=Path,
                    help="where corrected samples are written, same layout (the source is never changed)")
    ap.add_argument("--annotator", help="your name, recorded in each annotation you save")
    ap.add_argument("--export-only", action="store_true",
                    help="(re)write every output from the working copy without opening the editor")
    add_sam_arguments(ap)
    args = ap.parse_args(argv)

    if not args.src.is_dir():
        print(f"annofix: no such folder: {args.src}", file=sys.stderr)
        return 1
    if args.out.resolve() == args.src.resolve():
        print("annofix: --out must be a different folder from the source", file=sys.stderr)
        return 1
    work = args.out / WORK_DIR
    try:
        report = import_dataset(args.src, work)
    except (OSError, ValueError, KeyError) as e:
        print(f"annofix: could not read the source: {type(e).__name__}: {e}", file=sys.stderr)
        return 1
    if not report.imported and not report.skipped:
        print(f"annofix: no samples under {args.src} (expected <sample>/{ANN_JSON.as_posix()})", file=sys.stderr)
        return 1
    print(f"imported {len(report.imported)} new sample(s); kept {len(report.skipped)} already in {work}")
    written = export_dataset(args.src, work, args.out, only_missing=not args.export_only)
    if args.export_only:
        print(f"exported {len(written)} sample(s) to {args.out}")
        return 0

    from .config import load_labels
    from .session import Session

    try:
        session = Session(work / "images", work / "annotations", load_labels(work / LABELS),
                          annotator=args.annotator)
    except (OSError, ValueError) as e:
        print(f"annofix: {e}", file=sys.stderr)
        return 1
    attach_auto_export(session, args.src, args.out)
    return run_window(session, work / "annotations", sam_factory_from_args(args))


if __name__ == "__main__":
    sys.exit(main())
