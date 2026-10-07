"""Points tool: drag, add and delete a part's outline points (Session)."""
import numpy as np
import pytest
from helpers import LABELS, make_dataset

from annofix.session import Session


@pytest.fixture
def s(tmp_path):
    s = Session(*make_dataset(tmp_path), LABELS)
    s.open(0)
    s.select_part(2)  # part 2: rows 5..14, columns 5..24
    s.set_tool("points")
    return s


def corner(s, x, y):
    ring = s.rings[0]
    return 0, next(i for i, (px, py) in enumerate(ring) if (px, py) == (x, y))


def test_points_tool_shows_the_selected_parts_outline(s):
    assert len(s.rings) == 1 and {tuple(p) for p in s.rings[0]} == {(5, 5), (24, 5), (24, 14), (5, 14)}
    s.select_part(3)
    assert {tuple(p) for p in s.rings[0]} == {(30, 5), (39, 5), (39, 14), (30, 14)}
    s.set_tool("select")
    assert s.rings is None


def test_dragging_a_corner_changes_only_the_pixels_between_it_and_its_neighbours(s):
    before = s.ann.label_map.copy()
    assert s.move_point(*corner(s, 24, 14), 28, 14)
    changed = np.argwhere(s.ann.label_map != before)
    assert changed.size and changed[:, 0].min() >= 5 and changed[:, 0].max() <= 14   # rows
    assert changed[:, 1].min() >= 25 and changed[:, 1].max() <= 28                    # columns
    assert s.ann.part_at(13, 27) == 2 and s.ann.part_at(6, 26) is None  # inside / outside the new edge
    assert {tuple(p) for p in s.rings[0]} == {(5, 5), (24, 5), (28, 14), (5, 14)}  # your edit stays as drawn
    s.undo()
    assert np.array_equal(s.ann.label_map, before)
    assert {tuple(p) for p in s.rings[0]} == {(5, 5), (24, 5), (24, 14), (5, 14)}


def test_adding_a_point_changes_no_pixels_and_deleting_one_does(s):
    before = s.ann.label_map.copy()
    kinds = []
    s.subscribe(kinds.append)
    ring, i = corner(s, 5, 14)
    j = (i + 1) % 4
    a, b = s.rings[0][i], s.rings[0][j]
    mid = (a + b) // 2
    assert s.insert_point(0, i, int(mid[0]), int(mid[1]))
    assert len(s.rings[0]) == 5 and np.array_equal(s.ann.label_map, before) and "points" in kinds
    assert not s.ann.dirty
    assert s.delete_point(*corner(s, 24, 14))  # the shape loses that corner's triangle
    assert s.ann.part_at(13, 23) is None and s.ann.part_at(6, 6) == 2
    while len(s.rings[0]) > 3:
        s.delete_point(0, 0)
    assert not s.delete_point(0, 0) and "3 points" in s.messages[-1]


def test_other_parts_lose_pixels_unless_protected_or_hidden(s):
    s.move_point(*corner(s, 24, 14), 34, 14)          # the new edge crosses part 3 (columns 30..39)
    assert s.ann.part_at(13, 31) == 2
    s.undo()
    s.set_protect(True)
    s.move_point(*corner(s, 24, 14), 34, 14)
    assert s.ann.part_at(13, 31) == 3 and s.ann.part_at(13, 27) == 2
    s.undo()
    s.set_protect(False)
    s.set_hidden("part", 3, True)                      # a part you cannot see is never changed
    s.move_point(*corner(s, 24, 14), 34, 14)
    assert s.ann.part_at(13, 31) == 3


def test_a_hidden_part_offers_no_points(s):
    s.set_hidden("part", 2, True)
    assert s.rings is None and not s.move_point(0, 0, 0, 0)


def test_stale_or_outside_points_are_handled(s):
    before = s.ann.label_map.copy()
    assert not s.move_point(3, 0, 1, 1) and not s.move_point(0, 9, 1, 1)   # no such ring / point
    assert np.array_equal(s.ann.label_map, before)
    s.move_point(*corner(s, 24, 14), 500, 500)                                 # clamped to the image
    assert s.ann.part_at(39, 59) == 2
