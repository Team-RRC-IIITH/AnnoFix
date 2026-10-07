"""Editable part outlines (annofix/outline.py)."""
import cv2
import numpy as np

from annofix.outline import crosses_itself, fill, rings_of


def shape_with_hole_and_island():
    mask = np.zeros((60, 80), bool)
    mask[5:50, 10:70] = True
    mask[12:40, 20:60] = False  # a hole...
    mask[20:30, 30:45] = True   # ...with an island in it
    return mask


def test_outline_round_trip_is_exact_with_holes_and_islands_in_holes():
    mask = shape_with_hole_and_island()
    rings, parents = rings_of(mask, epsilon=0)
    holes = [k for k, parent in enumerate(parents) if parent >= 0]
    assert len(rings) == 3 and len(holes) == 1 and parents[parents[holes[0]]] == -1  # a hole of an outer ring
    assert np.array_equal(fill(rings, parents, mask.shape), mask)
    assert all(r.dtype.kind == "i" and r.shape[1] == 2 and len(r) >= 3 for r in rings)


def test_specks_stay_plain_pixels():
    mask = np.zeros((40, 60), bool)
    mask[5:30, 10:50] = True
    mask[35, 2] = True             # a speck
    mask[34:36, 55:57] = True      # a 2x2 crumb
    rings, parents = rings_of(mask)
    assert len(rings) == 1 and parents == [-1]


def test_overlapping_outlines_are_a_union_not_a_hole():
    a = np.array([[10, 10], [10, 30], [30, 30], [30, 10]])
    b = np.array([[20, 20], [20, 40], [40, 40], [40, 20]])
    filled = fill([a, b], [-1, -1], (50, 50))
    assert filled[25, 25] and filled[35, 35] and filled[15, 15]


def test_a_hole_never_adds_pixels_outside_its_outer_outline():
    outer = np.array([[10, 10], [10, 30], [30, 30], [30, 10]])
    hole = np.array([[15, 15], [15, 25], [45, 25], [45, 15]])  # dragged past the outer edge
    filled = fill([outer, hole], [-1, 0], (50, 50))
    assert not filled[20, 40] and not filled[20, 20] and filled[12, 12]


def test_simplifying_keeps_few_points():
    mask = np.zeros((200, 200), np.uint8)
    cv2.circle(mask, (100, 100), 70, 1, -1)
    exact = sum(len(r) for r in rings_of(mask.astype(bool), epsilon=0)[0])
    simple = sum(len(r) for r in rings_of(mask.astype(bool))[0])
    assert simple < exact / 3 and simple >= 8


def test_self_crossing_is_detected_at_the_moved_point():
    ring = np.array([[5, 5], [5, 14], [24, 14], [24, 5]])
    assert not crosses_itself(ring, 2)
    bow = ring.copy()
    bow[2] = (24, 0)  # the edge from (5,14) now crosses the top edge
    assert crosses_itself(bow, 2)
