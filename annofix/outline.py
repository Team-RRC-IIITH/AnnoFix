"""Editable outlines for the points tool. A part's outline is a list of rings, each (N, 2) x, y
pixel centres, with `parents`: -1 for an outer boundary, else the index of the outer ring a
hole belongs to. Traced exactly, the rings fill back to exactly the part's pixels. Simplified
(approxPolyDP) they have few enough points to drag by hand, but no longer fill back exactly, so
the Session applies an edit only near where the fill changed (see Session._apply_rings)."""
from __future__ import annotations

import annofix  # noqa: F401  (thread cap before numpy)

import cv2
import numpy as np

MIN_RING_AREA = 12.0  # px²: smaller pieces and holes get no points (they stay plain pixels)


def rings_of(mask: np.ndarray, epsilon: float = 1.5,
             min_area: float = MIN_RING_AREA) -> tuple[list[np.ndarray], list[int]]:
    """(rings, parents) of the mask, simplified to within `epsilon` pixels (0: exact)."""
    contours, hierarchy = cv2.findContours(mask.astype(np.uint8), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if hierarchy is None:
        return [], []
    parent_of = hierarchy[0][:, 3]
    outers_first = sorted(range(len(contours)), key=lambda i: parent_of[i] >= 0)
    rings: list[np.ndarray] = []
    parents: list[int] = []
    index: dict[int, int] = {}
    for i in outers_first:
        c, parent = contours[i], int(parent_of[i])
        if (parent >= 0 and parent not in index) or abs(cv2.contourArea(c)) < min_area:
            continue  # too small, or a hole in a piece that was
        if epsilon > 0:
            c = cv2.approxPolyDP(c, epsilon, True)
        ring = c.reshape(-1, 2).astype(np.int64)
        if len(ring) < 3:
            continue
        index[i] = len(rings)
        rings.append(ring)
        parents.append(-1 if parent < 0 else index[parent])
    return rings, parents


def densify(ring: np.ndarray, step: float) -> np.ndarray:
    """The ring with extra points along each edge, at most `step` px apart (rounded to pixel
    centres on the edge), so part of a long straight edge can be grabbed. Its fill is the same."""
    out = []
    n = len(ring)
    for i in range(n):
        a, b = ring[i].astype(float), ring[(i + 1) % n].astype(float)
        out.append(ring[i])
        pieces = int(np.ceil(np.hypot(*(b - a)) / step))
        for j in range(1, pieces):
            p = np.round(a + (b - a) * j / pieces).astype(ring.dtype)
            if not (p == out[-1]).all() and not (p == ring[(i + 1) % n]).all():
                out.append(p)
    return np.array(out)


def fill(rings: list[np.ndarray], parents: list[int], shape: tuple[int, int],
         outers=None, window: tuple[int, int, int, int] | None = None) -> np.ndarray:
    """The pixels the rings enclose: each outer ring minus the inside of its own holes, and the
    union of those. A hole's ring runs through the part's pixels around it, so only what is
    strictly inside it is taken away. Overlapping outer rings simply merge. `outers` limits it
    to those outer rings; `window` (x0, y0, x1, y1) to that box, for speed on big parts."""
    h, w = shape
    wx0, wy0, wx1, wy1 = window if window is not None else (0, 0, w, h)
    out = np.zeros(shape, bool)
    for k, ring in enumerate(rings):
        if parents[k] >= 0 or (outers is not None and k not in outers):
            continue
        x0, y0 = np.maximum(ring.min(0), (wx0, wy0))
        x1, y1 = np.minimum(ring.max(0) + 1, (wx1, wy1))
        if x0 >= x1 or y0 >= y1:
            continue
        offset = np.array([x0, y0])
        piece = np.zeros((y1 - y0, x1 - x0), np.uint8)
        cv2.fillPoly(piece, [(ring - offset).reshape(-1, 1, 2).astype(np.int32)], 1)
        for j, hole in enumerate(rings):
            if parents[j] == k:
                pts = [(hole - offset).reshape(-1, 1, 2).astype(np.int32)]
                inside = np.zeros_like(piece)
                cv2.fillPoly(inside, pts, 1)
                cv2.polylines(inside, pts, True, 0)
                piece[inside.view(bool)] = 0
        out[y0:y1, x0:x1] |= piece.view(bool)
    return out


def _orient(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    return (b[..., 0] - a[..., 0]) * (c[..., 1] - a[..., 1]) - (b[..., 1] - a[..., 1]) * (c[..., 0] - a[..., 0])


def edge_crosses(ring: np.ndarray, e: int) -> bool:
    """Does edge e (point e to point e+1) properly cross another edge of the same ring?"""
    n = len(ring)
    if n < 4:
        return False
    starts = ring.astype(np.int64)
    ends = np.roll(starts, -1, axis=0)
    others = np.array([j for j in range(n) if j not in ((e - 1) % n, e, (e + 1) % n)])
    p, q = starts[e], ends[e]
    a, b = starts[others], ends[others]
    return bool(((_orient(a, b, p) * _orient(a, b, q) < 0) & (_orient(p, q, a) * _orient(p, q, b) < 0)).any())


def crosses_itself(ring: np.ndarray, i: int) -> bool:
    """Do the two edges at point i cross the rest of the ring?"""
    n = len(ring)
    return edge_crosses(ring, (i - 1) % n) or edge_crosses(ring, i)


def on_segment(a: np.ndarray, b: np.ndarray, x: int, y: int) -> tuple[int, int]:
    """The point of segment a-b nearest (x, y), rounded to a pixel."""
    a, b = a.astype(np.float64), b.astype(np.float64)
    ab = b - a
    t = float(np.clip(np.dot((x, y) - a, ab) / max(np.dot(ab, ab), 1e-9), 0, 1))
    q = np.rint(a + t * ab)
    return int(q[0]), int(q[1])
