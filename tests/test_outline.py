"""Editable part outlines (annofix/outline.py)."""
import cv2
import numpy as np

from annofix.outline import fill, rings_of


def test_outline_round_trip_is_exact_including_holes():
    mask = np.zeros((40, 60), bool)
    mask[5:30, 10:50] = True
    mask[12:20, 20:30] = False  # a hole
    mask[35, 2] = True          # a one-pixel speck
    rings = rings_of(mask, epsilon=0)
    assert np.array_equal(fill(rings, mask.shape), mask & ~speck(mask.shape))
    assert all(r.dtype.kind == "i" and r.shape[1] == 2 and len(r) >= 3 for r in rings)


def speck(shape):
    s = np.zeros(shape, bool)
    s[35, 2] = True  # too small to have 3 points: left as plain pixels, not an editable ring
    return s


def test_simplifying_keeps_few_points():
    mask = np.zeros((200, 200), np.uint8)
    cv2.circle(mask, (100, 100), 70, 1, -1)
    exact = sum(len(r) for r in rings_of(mask.astype(bool), epsilon=0))
    simple = sum(len(r) for r in rings_of(mask.astype(bool)))
    assert simple < exact / 3 and simple >= 8
