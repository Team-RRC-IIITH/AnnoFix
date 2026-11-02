"""Automatic clean-up of noisy masks (Shift+R). Each object, then each of its parts, is asked
of SAM again, prompted from the mask it already has: its box, a few points inside it, and a
couple of points on its neighbours to exclude. SAM's answer that best matches the old mask is
taken when it matches well enough (its edges follow the image); otherwise the old mask is only
smoothed (the staircase edges of an upscaled mask, specks and pinholes go). Then the image is
put back together with one owner per pixel: objects first (smaller wins an overlap, as on
import), parts clipped to their object, and the rest of each object to its body. Hidden parts
keep every pixel. Qt-free and torch-free: `predict` is any SAM-like callable."""
from __future__ import annotations

import annofix  # noqa: F401  (thread cap before numpy)

from dataclasses import dataclass

import cv2
import numpy as np

from .annotation import Annotation, is_body
from .overlaps import resolve_overlaps

MIN_IOU = 0.70             # SAM's answer must overlap the old mask this much to be taken
AREA_RATIO = (0.6, 1.6)    # ...and be within this factor of its size
SMOOTH_SIGMA = 3.0         # px; the pipeline's staircase steps are ~2.5 px (median) to ~10 px
MIN_ISLAND_PX = 64         # smaller loose pieces are dropped when smoothing
MAX_HOLE_PX = 64           # only pinholes are filled; a handle's loop is a real hole
MAX_SMOOTH_LOSS = 0.10     # smoothing that removes more than this share of a mask is too much
MIN_JOB_PX = 50            # smaller masks are left exactly as they are
NEIGHBOUR_PX = 5           # how far a new edge may push into pixels another object still has


@dataclass(frozen=True)
class Proposal:
    label_map: np.ndarray  # the cleaned label map
    report: dict           # (kind, id) -> "sam" | "smoothed" | "kept" | "hidden"

    def summary(self) -> str:
        words = {"sam": "snapped to edges by SAM", "smoothed": "smoothed", "kept": "too small, kept",
                 "hidden": "hidden, untouched"}
        counts = [(list(self.report.values()).count(k), w) for k, w in words.items()]
        return ", ".join(f"{n} {w}" for n, w in counts if n) or "nothing to clean"


@dataclass(frozen=True)
class Job:
    key: tuple[str, int]            # ("object", id) or ("part", id)
    points: np.ndarray              # (N, 2) float32 x, y
    labels: np.ndarray              # (N,) int32, 1 include / 0 exclude
    box: np.ndarray                 # (4,) float32 x0, y0, x1, y1
    window: tuple[int, int, int, int]  # x0, y0, x1, y1: where SAM's answer is looked at
    original: np.ndarray            # the old mask inside `window`


@dataclass(frozen=True)
class Result:
    key: tuple[str, int]
    window: tuple[int, int, int, int]
    mask: np.ndarray | None         # the taken answer inside `window`, or None (smooth instead)
    iou: float


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    rows, cols = np.flatnonzero(mask.any(1)), np.flatnonzero(mask.any(0))
    return int(cols[0]), int(rows[0]), int(cols[-1]), int(rows[-1])


def spread_points(mask: np.ndarray, n: int) -> list[tuple[int, int]]:
    """Up to n points well inside the mask and apart from each other: distance-transform
    maxima, each one blanking a disc around it before the next is taken."""
    if n <= 0 or not mask.any():
        return []
    x0, y0, x1, y1 = _bbox(mask)
    crop = np.pad(mask[y0:y1 + 1, x0:x1 + 1], 1).astype(np.uint8)
    dist = cv2.distanceTransform(crop, cv2.DIST_L2, 5)
    deepest = float(dist.max())
    points = []
    for _ in range(n):
        y, x = np.unravel_index(int(np.argmax(dist)), dist.shape)
        d = float(dist[y, x])
        if d <= 0 or (points and d < 0.3 * deepest):
            break
        points.append((x0 + int(x) - 1, y0 + int(y) - 1))
        cv2.circle(dist, (int(x), int(y)), max(2, int(0.5 * d) + 1), 0, -1)
    return points


def prompts_for(mask: np.ndarray, others: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(points, labels, box): the mask's box padded a little, 3 points inside the mask, and up
    to 2 on `others` (the neighbours SAM must not swallow) inside that box."""
    h, w = mask.shape
    x0, y0, x1, y1 = _bbox(mask)
    pad = max(4, int(round(0.03 * max(x1 - x0, y1 - y0))))
    x0, y0, x1, y1 = max(x0 - pad, 0), max(y0 - pad, 0), min(x1 + pad, w - 1), min(y1 + pad, h - 1)
    near = np.zeros_like(mask)
    near[y0:y1 + 1, x0:x1 + 1] = others[y0:y1 + 1, x0:x1 + 1]
    inside, outside = spread_points(mask, 3), spread_points(near, 2)
    points = np.array(inside + outside, np.float32).reshape(-1, 2)
    labels = np.array([1] * len(inside) + [0] * len(outside), np.int32)
    return points, labels, np.array([x0, y0, x1, y1], np.float32)


def make_job(key: tuple[str, int], mask: np.ndarray, others: np.ndarray) -> Job:
    points, labels, box = prompts_for(mask, others)
    h, w = mask.shape
    x0, y0, x1, y1 = (int(v) for v in box)
    mx, my = (x1 - x0) // 4 + 8, (y1 - y0) // 4 + 8  # room for SAM to grow the mask a little
    window = (max(x0 - mx, 0), max(y0 - my, 0), min(x1 + mx + 1, w), min(y1 + my + 1, h))
    wx0, wy0, wx1, wy1 = window
    return Job(key, points, labels, box, window, mask[wy0:wy1, wx0:wx1].copy())


def run_job(predict, job: Job) -> Result:
    """Ask SAM, keep the candidate that best matches the old mask, if it matches well."""
    masks, _scores, _logits = predict(job.points, job.labels, job.box, None, True)
    x0, y0, x1, y1 = job.window
    area = int(job.original.sum())
    best, best_iou, best_ratio = None, 0.0, 0.0
    for m in masks:
        m = tidy(np.asarray(m, bool))
        whole = int(m.sum())  # judged on the whole answer: a leak past the window counts too
        crop = m[y0:y1, x0:x1]
        inter = int((crop & job.original).sum())
        union = whole + area - inter
        score = inter / union if union else 0.0
        if score > best_iou:
            best, best_iou, best_ratio = crop, score, whole / max(area, 1)
    if best is None or best_iou < MIN_IOU or not AREA_RATIO[0] <= best_ratio <= AREA_RATIO[1]:
        return Result(job.key, job.window, None, best_iou)
    return Result(job.key, job.window, best.copy(), best_iou)


def place(result: Result, shape: tuple[int, int]) -> np.ndarray:
    out = np.zeros(shape, bool)
    x0, y0, x1, y1 = result.window
    out[y0:y1, x0:x1] = result.mask
    return out


def smooth(mask: np.ndarray, sigma: float = SMOOTH_SIGMA) -> np.ndarray:
    """Round off staircase edges, drop loose specks, fill pinholes. Thin shapes (a pen, a
    handle) would melt away at full strength: then half the strength, and if even that removes
    too much, the mask is left as it was."""
    for s in (sigma, sigma / 2):
        out = _smooth(mask, s)
        if (mask & ~out).sum() <= MAX_SMOOTH_LOSS * mask.sum():
            return out
    return mask.copy()


def _smooth(mask: np.ndarray, sigma: float) -> np.ndarray:
    if not mask.any():
        return mask.copy()
    h, w = mask.shape
    pad = int(3 * sigma) + 2
    x0, y0, x1, y1 = _bbox(mask)
    x0, y0, x1, y1 = max(x0 - pad, 0), max(y0 - pad, 0), min(x1 + pad + 1, w), min(y1 + pad + 1, h)
    crop = _tidy(cv2.GaussianBlur(mask[y0:y1, x0:x1].astype(np.float32), (0, 0), sigma) > 0.5)
    out = np.zeros_like(mask)
    out[y0:y1, x0:x1] = crop
    return out


def tidy(mask: np.ndarray, fillable: np.ndarray | None = None) -> np.ndarray:
    """Drop loose specks (the largest piece always stays) and fill pinholes, as smoothing does;
    SAM leaves both, e.g. pinholes on a shiny cap. A real hole, like a handle's loop, is bigger
    than MAX_HOLE_PX and stays open. Holes are filled only where `fillable`, if given."""
    if not mask.any():
        return mask.copy()
    h, w = mask.shape
    x0, y0, x1, y1 = _bbox(mask)
    x0, y0, x1, y1 = max(x0 - 1, 0), max(y0 - 1, 0), min(x1 + 2, w), min(y1 + 2, h)
    out = np.zeros_like(mask)
    out[y0:y1, x0:x1] = _tidy(mask[y0:y1, x0:x1].copy())  # _tidy fills holes in place
    if fillable is not None:
        out &= mask | fillable
    return out


def _tidy(crop: np.ndarray) -> np.ndarray:
    n, cc, stats, _ = cv2.connectedComponentsWithStats(crop.astype(np.uint8), connectivity=8)
    if n > 2:
        areas = stats[1:, cv2.CC_STAT_AREA]
        keep = [i for i in range(1, n) if areas[i - 1] >= MIN_ISLAND_PX or i == 1 + int(np.argmax(areas))]
        crop = np.isin(cc, keep)
    n, cc, stats, _ = cv2.connectedComponentsWithStats((~crop).astype(np.uint8), connectivity=4)
    ch, cw = crop.shape
    for i in range(1, n):
        x, y, bw, bh, area = (int(v) for v in stats[i])
        if area <= MAX_HOLE_PX and x > 0 and y > 0 and x + bw < cw and y + bh < ch:
            crop[cc == i] = True  # an enclosed pinhole
    return crop


def held_ids(ann: Annotation, locked) -> set[int]:
    """Every part of every object that has a hidden (locked) part: such objects are left
    exactly as they are, all their parts."""
    if locked is None:
        return set()
    return {p.id for o in ann.objects if any(locked[q.id] for q in o.parts) for p in o.parts}


def _limit(origs: list[np.ndarray], refined: list[np.ndarray], pool: np.ndarray) -> None:
    """Keep new edges from pushing more than NEIGHBOUR_PX into pixels a neighbour still has:
    each mask may grow into `pool` (free pixels), into what other masks gave up, and a few px
    beyond its old edge, nowhere else. In place."""
    freed = np.zeros(pool.shape, bool)
    for o, r in zip(origs, refined):
        freed |= o & ~r
    kernel = np.ones((2 * NEIGHBOUR_PX + 1,) * 2, np.uint8)
    for o, r in zip(origs, refined):
        grown = r & ~o
        if not grown.any():
            continue
        near = cv2.dilate(o.view(np.uint8), kernel).view(bool)
        r &= near | pool | freed


def jobs_for(ann: Annotation, locked=None) -> list[Job]:
    """One SAM job per object (the union of its parts) and per real part of a multi-part
    object. Objects with a hidden part, bodies, and tiny masks get none."""
    lm, hidden = ann.label_map, held_ids(ann, locked)
    jobs = []
    for o in ann.objects:
        ids = [p.id for p in o.parts]
        if not ids or hidden & set(ids):
            continue
        union = np.isin(lm, ids)
        if union.sum() >= MIN_JOB_PX:
            jobs.append(make_job(("object", o.id), union, (lm != 0) & ~union))
        if len(o.parts) > 1:
            for p in o.parts:
                part = lm == p.id
                if not is_body(p) and part.sum() >= MIN_JOB_PX:
                    jobs.append(make_job(("part", p.id), part, union & ~part))
    return jobs


def _refined(original: np.ndarray, result: Result | None) -> tuple[np.ndarray, str]:
    if original.sum() < MIN_JOB_PX:
        return original, "kept"
    if result is not None and result.mask is not None:
        return place(result, original.shape), "sam"
    return smooth(original), "smoothed"


def build_proposal(ann: Annotation, results: dict, locked=None) -> tuple[np.ndarray, dict]:
    """The cleaned label map, and how each object and part was cleaned:
    "sam", "smoothed", "kept" (too small to touch) or "hidden"."""
    lm, hidden = ann.label_map, held_ids(ann, locked)
    held = np.isin(lm, list(hidden)) if hidden else np.zeros(lm.shape, bool)
    proposal = np.zeros_like(lm)
    proposal[held] = lm[held]
    report: dict = {}
    free = [o for o in ann.objects if o.parts and not hidden & {p.id for p in o.parts}]
    for o in ann.objects:
        if o not in free and o.parts:
            report[("object", o.id)] = "hidden"
    origs, regions = [], []
    for o in free:
        orig = np.isin(lm, [p.id for p in o.parts])
        mask, how = _refined(orig, results.get(("object", o.id)))
        origs.append(orig)
        regions.append(mask & ~held)
        report[("object", o.id)] = how
    _limit(origs, regions, lm == 0)
    owner = resolve_overlaps(np.stack(regions)) if regions else np.zeros(lm.shape, np.int32)
    for k, o in enumerate(free):
        region = owner == k + 1
        if len(o.parts) == 1:
            proposal[region] = o.parts[0].id
            continue
        real = [p for p in o.parts if not is_body(p)]
        part_origs, masks = [], []
        for p in real:
            mask, how = _refined(lm == p.id, results.get(("part", p.id)))
            part_origs.append(lm == p.id)
            masks.append(mask & region)
            report[("part", p.id)] = how
        bodies = [p.id for p in o.parts if is_body(p)]
        _limit(part_origs, masks, region & ((lm == 0) | np.isin(lm, bodies) | ~np.isin(lm, [p.id for p in o.parts])))
        for i, p in enumerate(real):
            if not masks[i].any():  # never lose a part
                masks[i] = (lm == p.id) & region
        covered = np.zeros(lm.shape, bool)
        if masks:
            part_owner = resolve_overlaps(np.stack(masks))
            for i, p in enumerate(real):
                here = (part_owner == i + 1) & region
                proposal[here] = p.id
                covered |= here
        rest = region & ~covered
        body = next((p for p in o.parts if is_body(p)), None)
        if body is not None:
            proposal[rest] = body.id
        elif rest.any():
            _nearest_part(proposal, rest, covered, o, lm)
    return proposal, report


def _nearest_part(proposal: np.ndarray, rest: np.ndarray, covered: np.ndarray, obj, lm) -> None:
    """Give the object's leftover pixels to the nearest of its parts (no body to take them)."""
    if not covered.any():
        largest = max(obj.parts, key=lambda p: int((lm == p.id).sum()))
        proposal[rest] = largest.id
        return
    x0, y0, x1, y1 = _bbox(rest | covered)
    box = np.s_[y0:y1 + 1, x0:x1 + 1]
    src = (~covered[box]).astype(np.uint8)  # distance to the nearest covered pixel
    _, labels = cv2.distanceTransformWithLabels(src, cv2.DIST_L2, 5, labelType=cv2.DIST_LABEL_PIXEL)
    lut = np.zeros(int(labels.max()) + 1, proposal.dtype)
    lut[labels[covered[box]]] = proposal[box][covered[box]]
    target = proposal[box]
    target[rest[box]] = lut[labels[rest[box]]]
