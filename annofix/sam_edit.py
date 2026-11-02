"""One SAM fix in progress, without Qt or torch: the prompts (include/exclude points, a box),
the request they imply, the candidate masks SAM returned, which one is chosen, and a preview
of what accepting it would change. Requests and results carry (image token, seq), so a result
for an older prompt or another image is dropped."""
from __future__ import annotations

import itertools
from dataclasses import dataclass

import cv2
import numpy as np

from .refine import AREA_RATIO, MIN_IOU, tidy

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
    def __init__(self, token: int, target: int | None = None, object_id: int | None = None,
                 counter=None):
        """`target`: the part being fixed, or None for a new part, which goes into `object_id`,
        or into a new object when that is None too. `counter` numbers requests; the Session
        shares one across all its edits, so an answer for a cancelled edit can never match a
        later edit's request."""
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
        self._logits: np.ndarray | None = None  # (C, 256, 256), aligned with SAM's candidates
        self.reference: np.ndarray | None = None    # improve(): the target's mask before the fix
        self.alternative: np.ndarray | None = None  # improve(): that mask smoothed, the last candidate
        self._improve_seq = 0                       # improve(): its request, judged against `reference`
        self._counter = counter if counter is not None else itertools.count(1)

    @property
    def mask(self) -> np.ndarray | None:
        return self.candidates[self.index] if self.candidates else None

    @property
    def score(self) -> float | None:
        return self.scores[self.index] if self.scores else None

    @property
    def is_alternative(self) -> bool:
        """The smoothed mask is chosen, not one of SAM's answers."""
        return self.alternative is not None and bool(self.candidates) and self.index == len(self.candidates) - 1

    def improve(self, reference: np.ndarray, alternative: np.ndarray, points, labels, box) -> SamRequest:
        """Improve the target's current mask (`reference`) instead of drawing it anew: SAM is
        asked with prompts taken from that mask and several answers, the one closest to the mask
        is shown first, and `alternative` (the mask smoothed) is always one more candidate."""
        self.reference, self.alternative = reference, alternative
        self.points = [(int(x), int(y)) for x, y in np.asarray(points).reshape(-1, 2)]
        self.labels = [int(v) for v in np.asarray(labels).reshape(-1)]
        x0, y0, x1, y1 = (int(v) for v in box)
        self.box = (x0, y0, x1, y1)
        request = self._request(multimask=True)
        self._improve_seq = self.seq
        return request

    def add_point(self, x: int, y: int, positive: bool) -> SamRequest | None:
        self.points.append((int(x), int(y)))
        self.labels.append(1 if positive else 0)
        return self._request()

    def set_box(self, x0: int, y0: int, x1: int, y1: int) -> SamRequest | None:
        self.box = (min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1))
        return self._request()

    def _request(self, multimask: bool | None = None) -> SamRequest | None:
        if 1 not in self.labels and self.box is None:
            return None  # nothing to segment yet
        self.seq = next(self._counter)
        if multimask is None:
            multimask = self.box is None and len(self.points) == 1  # only a lone click is ambiguous
        sam_chosen = self._logits is not None and self.index < len(self._logits)  # not the smoothed mask
        prev = self._logits[self.index:self.index + 1] if sam_chosen and not multimask else None
        return SamRequest(
            self.token, self.seq,
            np.array(self.points, np.float32) if self.points else None,
            np.array(self.labels, np.int32) if self.points else None,
            None if self.box is None else np.array(self.box, np.float32),
            prev, multimask)

    def receive(self, token: int, seq: int, masks, scores, logits, fillable=None) -> bool:
        """Take SAM's answer if it is for this image and the newest prompt; False otherwise.
        Pinholes are filled only where `fillable` (default: anywhere), so a small part that SAM
        rightly left out is never swallowed."""
        if token != self.token or seq != self.seq or len(masks) == 0:
            return False
        order = np.argsort(np.asarray(masks).reshape(len(masks), -1).sum(1), kind="stable")
        candidates = [tidy(np.asarray(masks[i], bool), fillable) for i in order]
        self.scores = [float(scores[i]) for i in order]
        self._logits = np.asarray(logits)[order]
        weak = False
        if self.reference is not None and seq == self._improve_seq:  # the answer to improve()
            ious = [_iou(c, self.reference) for c in candidates]
            self.index = int(np.argmax(ious))
            ratio = candidates[self.index].sum() / max(int(self.reference.sum()), 1)
            weak = ious[self.index] < MIN_IOU or not AREA_RATIO[0] <= ratio <= AREA_RATIO[1]  # as auto-clean judges
        else:
            self.index = int(np.argmax(self.scores))
        if self.alternative is not None:
            candidates.append(self.alternative)
            self.scores.append(float("nan"))
            if weak:  # SAM found something else: the smoothed mask first, SAM's answers one Tab away
                self.index = len(candidates) - 1
        self.candidates = candidates
        self.received_seq = seq
        return True

    def cycle(self, step: int = 1) -> None:
        if self.candidates:
            self.index = (self.index + step) % len(self.candidates)


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    union = int((a | b).sum())
    return int((a & b).sum()) / union if union else 0.0


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
        out[own & ~effective] = PREVIEW_REMOVED  # exactly what replace_part_mask removes
    return out
