"""Editable outlines for the points tool. A part's outline, traced exactly, fills back to exactly
its pixels (holes too: rings are filled even-odd). Simplified (approxPolyDP) it has few enough
points to drag by hand, but no longer fills back exactly, so an edit is applied as the
difference between filling the rings before and after it: pixels away from the edit stay as
they are."""
from __future__ import annotations

import annofix  # noqa: F401  (thread cap before numpy)

import cv2
import numpy as np


def rings_of(mask: np.ndarray, epsilon: float = 1.5) -> list[np.ndarray]:
    """Outer boundaries and holes as (N, 2) integer x, y pixel centres, simplified to within
    `epsilon` pixels (0: exact). Pieces too small for 3 points stay plain pixels."""
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    rings = []
    for c in contours:
        if epsilon > 0:
            c = cv2.approxPolyDP(c, epsilon, True)
        ring = c.reshape(-1, 2).astype(np.int64)
        if len(ring) >= 3:
            rings.append(ring)
    return rings


def fill(rings: list[np.ndarray], shape: tuple[int, int]) -> np.ndarray:
    out = np.zeros(shape, np.uint8)
    if rings:
        cv2.fillPoly(out, [r.reshape(-1, 1, 2).astype(np.int32) for r in rings], 1)
    return out.view(bool)
