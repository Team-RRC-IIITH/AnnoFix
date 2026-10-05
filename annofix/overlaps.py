"""Overlapping candidate masks (SAM automatic masks, or a pipeline's instance masks) ->
a single-ownership object -> part Annotation.

Candidates nest roughly (whole object > part > sub-part). Each candidate's parent is the
smallest candidate that mostly contains it; roots become objects and their direct children
become parts. Overlaps are resolved by painting large -> small so smaller masks win.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .annotation import AnnObject, Annotation, Part


@dataclass(frozen=True)
class GroupingParams:
    min_area: int = 100              # px; smaller candidates are dropped
    max_area_frac: float = 0.25      # larger candidates are background (table, wall)
    border_large_frac: float = 0.05  # "large" for the spans-two-borders rule
    dedupe_iou: float = 0.9          # near-duplicates keep the higher score
    contain_frac: float = 0.85       # |child ∩ parent| / |child| to count as contained
    parent_area_ratio: float = 1.05  # parent must be at least this much bigger
    collapse_frac: float = 0.9       # a lone child this big is the object itself
    min_component_px: int = 64       # final speckle removal per part


def resolve_overlaps(masks: np.ndarray) -> np.ndarray:
    """int32 map of (mask index + 1), 0 where no mask; smaller masks win overlaps."""
    areas = masks.sum(axis=(1, 2))
    out = np.zeros(masks.shape[1:], np.int32)
    for i in np.argsort(-areas, kind="stable"):
        out[masks[i]] = i + 1
    return out


def remove_small_components(mask: np.ndarray, min_px: int) -> np.ndarray:
    if min_px <= 1 or not mask.any():
        return mask
    _, cc, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    small = np.flatnonzero(stats[:, cv2.CC_STAT_AREA] < min_px)
    small = small[small != 0]
    if small.size == 0:
        return mask
    return mask & ~np.isin(cc, small)


def _borders_touched(m: np.ndarray) -> int:
    return int(m[0].any()) + int(m[-1].any()) + int(m[:, 0].any()) + int(m[:, -1].any())


def filter_candidates(masks: np.ndarray, p: GroupingParams) -> np.ndarray:
    n, h, w = masks.shape
    image_area = h * w
    areas = masks.sum(axis=(1, 2))
    keep = []
    for i in range(n):
        a = areas[i]
        if a < p.min_area or a > p.max_area_frac * image_area:
            continue
        if a > p.border_large_frac * image_area and _borders_touched(masks[i]) >= 2:
            continue
        keep.append(i)
    return np.array(keep, dtype=int)


def _bboxes(masks: np.ndarray) -> np.ndarray:
    out = np.zeros((len(masks), 4), int)  # y0, y1, x0, x1 (exclusive)
    for i, m in enumerate(masks):
        rows = np.flatnonzero(m.any(1))
        cols = np.flatnonzero(m.any(0))
        if rows.size:
            out[i] = rows[0], rows[-1] + 1, cols[0], cols[-1] + 1
    return out


def pairwise_intersections(masks: np.ndarray) -> np.ndarray:
    """(N,N) int64 |mi ∩ mj|, exact, computed only where bounding boxes overlap."""
    n = len(masks)
    inter = np.diag(masks.sum(axis=(1, 2)).astype(np.int64))  # sum, not reshape: works for N=0
    if n < 2:
        return inter
    bb = _bboxes(masks)
    y0 = np.maximum(bb[:, None, 0], bb[None, :, 0])
    y1 = np.minimum(bb[:, None, 1], bb[None, :, 1])
    x0 = np.maximum(bb[:, None, 2], bb[None, :, 2])
    x1 = np.minimum(bb[:, None, 3], bb[None, :, 3])
    overlap = np.triu((y0 < y1) & (x0 < x1), k=1)
    for i, j in zip(*np.nonzero(overlap)):
        sy, sx = slice(y0[i, j], y1[i, j]), slice(x0[i, j], x1[i, j])
        inter[i, j] = inter[j, i] = np.count_nonzero(masks[i, sy, sx] & masks[j, sy, sx])
    return inter


def dedupe(scores: np.ndarray, inter: np.ndarray, iou_thresh: float) -> list[int]:
    areas = np.diag(inter)
    kept: list[int] = []
    for i in np.argsort(-np.asarray(scores), kind="stable"):
        i = int(i)
        if all(inter[i, k] / (areas[i] + areas[k] - inter[i, k]) <= iou_thresh for k in kept):
            kept.append(i)
    return sorted(kept)


def find_parents(inter: np.ndarray, p: GroupingParams) -> np.ndarray:
    """parent[i] = index of the smallest candidate that contains candidate i, else -1."""
    areas = np.diag(inter).astype(np.float64)
    n = len(areas)
    if n == 0:
        return np.zeros(0, int)
    contained = inter / np.maximum(areas[:, None], 1) >= p.contain_frac
    bigger = areas[None, :] >= p.parent_area_ratio * areas[:, None]
    ok = contained & bigger
    np.fill_diagonal(ok, False)
    parent = np.argmin(np.where(ok, areas[None, :], np.inf), axis=1)
    parent[~ok.any(axis=1)] = -1
    return parent


def group_objects(parent: np.ndarray, areas: np.ndarray, p: GroupingParams) -> list[tuple[int, list[int]]]:
    """[(root, depth-1 children)]. Deeper candidates are dropped; their pixels stay with
    their depth-1 ancestor."""
    n = len(parent)
    children: dict[int, list[int]] = {i: [] for i in range(n)}
    for i in range(n):
        par = int(parent[i])
        if par != -1 and parent[par] == -1:
            children[par].append(i)
    groups = []
    for r in range(n):
        if parent[r] != -1:
            continue
        kids = children[r]
        if len(kids) == 1 and areas[kids[0]] >= p.collapse_frac * areas[r]:
            kids = []
        groups.append((r, kids))
    return groups


def paint_groups(masks: np.ndarray, groups: list[tuple[int, list[int]]], p: GroupingParams) -> Annotation:
    shape = masks.shape[1:]
    label = np.zeros(shape, np.uint16)
    objects: list[AnnObject] = []
    next_id = 1
    if groups:
        obj_map = resolve_overlaps(masks[[root for root, _ in groups]])
        for g, (_, kids) in enumerate(groups):
            region = obj_map == g + 1
            if not region.any():
                continue  # fully covered by smaller objects
            if kids:
                part_map = resolve_overlaps(masks[kids])
                pieces = [region & (part_map == k + 1) for k in range(len(kids))]
                pieces.append(region & (part_map == 0))  # leftover -> "rest" part
            else:
                pieces = [region]
            obj = AnnObject(next_id)
            next_id += 1
            for piece in pieces:
                piece = remove_small_components(piece, p.min_component_px)
                if piece.any():
                    label[piece] = next_id
                    obj.parts.append(Part(next_id))
                    next_id += 1
            if obj.parts:
                objects.append(obj)
    return Annotation(label, objects, next_id=next_id, source="amg")


def candidates_to_annotation(masks: np.ndarray, scores: np.ndarray,
                             p: GroupingParams = GroupingParams(), image: str = "") -> Annotation:
    keep = filter_candidates(masks, p)
    masks, scores = masks[keep], np.asarray(scores)[keep]
    inter = pairwise_intersections(masks)
    kept = dedupe(scores, inter, p.dedupe_iou)
    masks, inter = masks[kept], inter[np.ix_(kept, kept)]
    groups = group_objects(find_parents(inter, p), np.diag(inter), p)
    ann = paint_groups(masks, groups, p)
    ann.image = image
    return ann
