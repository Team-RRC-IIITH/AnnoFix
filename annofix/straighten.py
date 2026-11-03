"""Straighten: a part's zigzag edges redrawn as straight lines. Its outline is cut into
stretches where it bends (Douglas-Peucker), a line is fitted through each stretch, and the
corners go where neighbouring lines meet; the part is then filled from those lines. Unlike the
points tool's outline, whose points sit on the zigzag's bumps, fitted lines run through the
middle of the zigzag. A pixel mask still steps along a slanted line, but evenly and on it.

An outline is only straightened where every fitted line really runs through its stretch: its
pixels within MAX_RESIDUAL of it (RMS), scattered evenly (a curve's middle bulges to one side by
more than MAX_BEND). So round parts are left as they are instead of being cut into facets. For the whole image only the coarsest fit is tried, so only
clearly straight-edged parts change.

Where a straightened edge is shared with another part, the pixels it lets go of go to that
part (never background, so no gaps open) and the neighbour's pixels past the line come to it.
Holes stay holes. Hidden (locked) parts are never changed or taken; with `protect`, a part only
grows into background. Qt-free: the Session shows the result as a proposal, as auto-clean does."""
from __future__ import annotations

import annofix  # noqa: F401  (thread cap before numpy)

import cv2
import numpy as np

from .outline import MIN_RING_AREA, fill

TOLERANCES = (6.0, 4.0, 2.5)   # px: coarse first (fewest corners), finer if the lines do not fit
WHOLE_IMAGE_TOLERANCES = (6.0,)
MAX_RESIDUAL = 1.5             # px RMS: a stretch's pixels around its line; pixel zigzag, not a curve
MAX_BEND = 1.0                 # px the middle of a stretch may bulge from its line vs its ends: a curve does
MAX_CORNER_SHIFT = 6.0         # px a corner may move from the outline to where its lines meet
NEIGHBOURS = ((0, 1), (1, 0), (0, -1), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1))


def straight_ring(contour: np.ndarray, tolerances=TOLERANCES) -> np.ndarray | None:
    """The contour (N, 2) as straight lines with corners where they meet, or None if no
    tolerance gives lines that fit (a curve)."""
    pts = contour.astype(np.float32)
    for eps in tolerances:
        corners = cv2.approxPolyDP(pts.reshape(-1, 1, 2), eps, True)[:, 0, :]
        if len(corners) < 3:
            continue
        idx = sorted({int(np.argmin(((pts - c) ** 2).sum(1))) for c in corners})
        if len(idx) < 3:
            continue
        lines, fits = [], True
        for i, a in enumerate(idx):
            b = idx[(i + 1) % len(idx)]
            stretch = pts[a:b + 1] if b > a else np.vstack([pts[a:], pts[:b + 1]])
            vx, vy, x0, y0 = cv2.fitLine(stretch, cv2.DIST_HUBER, 0, 0.01, 0.01).ravel()
            off = (stretch - (x0, y0)) @ np.array([-vy, vx])  # signed distance from the line
            third = len(off) // 3
            bend = abs(off[third:len(off) - third].mean() - np.r_[off[:third], off[len(off) - third:]].mean()) \
                if third else 0.0
            if float(np.sqrt((off ** 2).mean())) > MAX_RESIDUAL or bend > MAX_BEND:
                fits = False
                break
            lines.append((np.array([x0, y0]), np.array([vx, vy])))
        if not fits:
            continue
        ring = []
        for i in range(len(lines)):
            (p1, d1), (p2, d2) = lines[i - 1], lines[i]
            original = pts[idx[i]]
            m = np.array([d1, -d2]).T
            if abs(np.linalg.det(m)) < 0.15:  # nearly the same direction: keep the outline's corner
                ring.append(original)
                continue
            t = np.linalg.solve(m, p2 - p1)
            meet = p1 + t[0] * d1
            ring.append(meet if np.linalg.norm(meet - original) <= MAX_CORNER_SHIFT else original)
        return np.round(np.array(ring)).astype(np.int32)
    return None


def straighten(label_map: np.ndarray, part_ids, locked=None, protect: bool = False,
               whole_image: bool = False) -> tuple[np.ndarray, dict]:
    """(new label map, report): report[("part", id)] is "straightened", "left as is" (no
    straight edges found), "kept" (too small to have an outline) or "hidden"."""
    lm = label_map
    new = lm.copy()
    tolerances = WHOLE_IMAGE_TOLERANCES if whole_image else TOLERANCES
    report, fills = {}, {}
    for pid in part_ids:
        if locked is not None and locked[pid]:
            report[("part", pid)] = "hidden"
            continue
        mask = lm == pid
        if not mask.any():
            continue
        contours, hierarchy = cv2.findContours(mask.view(np.uint8), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
        rings, parents, straightened = [], [], False
        index = {}
        for k, c in enumerate(contours):
            if cv2.contourArea(c) < MIN_RING_AREA:
                continue
            index[k] = len(rings)
            ring = straight_ring(c[:, 0, :], tolerances)
            straightened |= ring is not None
            rings.append(c[:, 0, :] if ring is None else ring)  # exact if no straight lines fit
            parents.append(int(hierarchy[0][k][3]))
        if not rings:
            report[("part", pid)] = "kept"
            continue
        if not straightened:
            report[("part", pid)] = "left as is"
            continue
        parents = [index.get(p, -1) if p >= 0 else -1 for p in parents]
        filled = fill(rings, parents, lm.shape)
        filled |= mask & ~_outline_reach(rings, parents, lm.shape)  # specks without an outline stay
        fills[pid] = filled
        report[("part", pid)] = "straightened"
    if not fills:
        return new, report

    ids = list(fills)
    mine = np.isin(lm, ids)
    new[mine] = 0
    takeable = (lm == 0) | mine if protect else np.ones(lm.shape, bool)
    if locked is not None:
        takeable &= ~locked[lm]
    for pid in sorted(ids, key=lambda i: -int(fills[i].sum())):  # where two claim a pixel, the smaller wins
        new[fills[pid] & takeable] = pid
    for pid in ids:  # ...unless it was one of theirs: a part keeps what it had inside its lines
        new[fills[pid] & (lm == pid)] = pid

    handed = hand_over(new, mine & (new == 0), lm, locked)
    new[handed != 0] = handed[handed != 0]
    return new, report


def hand_over(new: np.ndarray, released: np.ndarray, original: np.ndarray, locked=None) -> np.ndarray:
    """Who gets the pixels a part let go of (`released`, 0 in `new`): a neighbouring other part,
    so a shared edge never opens a gap; 0 (background) where none touches them. Labels per
    pixel, 0 elsewhere. Teeth up to 3 px deep are reached; hidden parts never receive any."""
    h, w = new.shape
    work, out, pending = new.copy(), np.zeros_like(new), released.copy()
    for _ in range(3):
        if not pending.any():
            break
        padded = np.pad(work, 1)
        got = np.zeros_like(new)
        for dy, dx in NEIGHBOURS:
            nb = padded[1 + dy:1 + dy + h, 1 + dx:1 + dx + w]
            ok = pending & (got == 0) & (nb != 0) & (nb != original)
            if locked is not None:
                ok &= ~locked[nb]
            got[ok] = nb[ok]
        if not got.any():
            break
        work[got != 0] = out[got != 0] = got[got != 0]
        pending &= got == 0
    return out


def _outline_reach(rings, parents, shape) -> np.ndarray:
    """Pixels inside or near any outer ring (dilated a little): a part's pixels outside this are
    pieces too small to have an outline, kept as they were."""
    reach = fill(rings, parents, shape, outers=None)
    for k, ring in enumerate(rings):
        if parents[k] < 0:
            cv2.fillPoly(reach.view(np.uint8), [ring.reshape(-1, 1, 2)], 1)
    return cv2.dilate(reach.view(np.uint8), np.ones((9, 9), np.uint8)).view(bool)
