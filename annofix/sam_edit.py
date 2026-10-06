"""One SAM fix in progress, without Qt or torch: the prompts (include/exclude points, a box),
the request they imply, the candidate masks SAM returned, which one is chosen, and a preview
of what accepting it would change. Requests and results carry (image token, seq), so a result
for an older prompt or another image is dropped."""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

PREVIEW_KEPT, PREVIEW_ADDED, PREVIEW_TAKEN, PREVIEW_REMOVED = 1, 2, 3, 4


@dataclass(frozen=True)
class SamRequest:
    token: int
    seq: int
    points: np.ndarray | None       # (N, 2) float32 x, y
    labels: np.ndarray | None       # (N,) int32, 1 include / 0 exclude
    box: np.ndarray | None          # (4,) float32 x0, y0, x1, y1
    prev_logits: np.ndarray | None  # (1, 256, 256): the chosen candidate of the previous answer
    multimask: bool


class SamEdit:
    def __init__(self, token: int, target: int | None = None, object_id: int | None = None):
        """`target`: the part being fixed, or None for a new part, which goes into `object_id`,
        or into a new object when that is None too."""
        self.token = token
        self.target = target
        self.object_id = object_id
        self.points: list[tuple[int, int]] = []
        self.labels: list[int] = []
        self.box: tuple[int, int, int, int] | None = None
        self.seq = 0
        self.received_seq = 0
        self.candidates: list[np.ndarray] = []
        self.scores: list[float] = []
        self.index = 0
        self._logits: np.ndarray | None = None  # (C, 256, 256), aligned with candidates

    @property
    def mask(self) -> np.ndarray | None:
        return self.candidates[self.index] if self.candidates else None

    @property
    def score(self) -> float | None:
        return self.scores[self.index] if self.scores else None

    def add_point(self, x: int, y: int, positive: bool) -> SamRequest | None:
        self.points.append((int(x), int(y)))
        self.labels.append(1 if positive else 0)
        return self._request()

    def set_box(self, x0: int, y0: int, x1: int, y1: int) -> SamRequest | None:
        self.box = (min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1))
        return self._request()

    def _request(self) -> SamRequest | None:
        if 1 not in self.labels and self.box is None:
            return None  # nothing to segment yet
        self.seq += 1
        multimask = self.box is None and len(self.points) == 1  # only a lone click is ambiguous
        prev = None if multimask or self._logits is None else self._logits[self.index:self.index + 1]
        return SamRequest(
            self.token, self.seq,
            np.array(self.points, np.float32) if self.points else None,
            np.array(self.labels, np.int32) if self.points else None,
            None if self.box is None else np.array(self.box, np.float32),
            prev, multimask)

    def receive(self, token: int, seq: int, masks, scores, logits) -> bool:
        """Take SAM's answer if it is for this image and the newest prompt; False otherwise."""
        if token != self.token or seq != self.seq or len(masks) == 0:
            return False
        order = np.argsort(np.asarray(masks).reshape(len(masks), -1).sum(1), kind="stable")
        self.candidates = [np.asarray(masks[i], bool) for i in order]
        self.scores = [float(scores[i]) for i in order]
        self._logits = np.asarray(logits)[order]
        self.index = int(np.argmax(self.scores))
        self.received_seq = seq
        return True

    def cycle(self, step: int = 1) -> None:
        if self.candidates:
            self.index = (self.index + step) % len(self.candidates)


def interior_point(mask: np.ndarray) -> tuple[int, int] | None:
    """The pixel deepest inside `mask` (distance-transform maximum) as (x, y). A click there is
    unambiguous even when the mask's outline, and so its bbox, is wrong."""
    if not mask.any():
        return None
    rows = np.flatnonzero(mask.any(1))
    cols = np.flatnonzero(mask.any(0))
    y0, x0 = int(rows[0]), int(cols[0])
    crop = np.pad(mask[y0:int(rows[-1]) + 1, x0:int(cols[-1]) + 1], 1).astype(np.uint8)
    dist = cv2.distanceTransform(crop, cv2.DIST_L2, 5)
    y, x = np.unravel_index(int(np.argmax(dist)), dist.shape)
    return x0 + int(x) - 1, y0 + int(y) - 1


def preview_map(label_map: np.ndarray, target: int | None, mask: np.ndarray, *, union: bool = False,
                protect: bool = False, locked: np.ndarray | None = None) -> np.ndarray:
    """uint8 per pixel: what accepting `mask` for `target` (None = a new part) would do:
    KEPT (already the target's), ADDED (from background), TAKEN (from another part), REMOVED
    (the target loses it; replace only). Locked (hidden/non-solo) parts are never changed."""
    effective = mask if locked is None else mask & ~locked[label_map]
    own = label_map == target if target is not None else np.zeros(label_map.shape, bool)
    background = label_map == 0
    out = np.zeros(label_map.shape, np.uint8)
    out[effective & own] = PREVIEW_KEPT
    out[effective & background] = PREVIEW_ADDED
    if not protect:
        out[effective & ~background & ~own] = PREVIEW_TAKEN
    if target is not None and not union:
        out[own & ~mask] = PREVIEW_REMOVED
    return out
