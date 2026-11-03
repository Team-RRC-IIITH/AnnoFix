"""Points tool: drag, add and delete a part's outline points (Session)."""
import cv2
import numpy as np
import pytest
from helpers import LABELS, make_dataset

from annofix.annotation import AnnObject, Annotation, Part
from annofix.io import save_annotation
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


def session_on(tmp_path, label_map):
    """A one-image session whose part 2 (object 1) is wherever `label_map` is 2."""
    h, w = label_map.shape
    images, ann_dir = tmp_path / "images", tmp_path / "ann"
    images.mkdir()
    cv2.imwrite(str(images / "x.png"), np.full((h, w, 3), 100, np.uint8))
    parts = [Part(int(i), "p", "metal") for i in np.unique(label_map) if i]
    save_annotation(Annotation(label_map.astype(np.uint16), [AnnObject(1, "thing", parts)], next_id=10),
                    ann_dir, "x")
    s = Session(images, ann_dir, LABELS)
    s.open(0)
    s.select_part(2)
    s.set_tool("points")
    return s


def enclosed_background(mask):
    """Background pixels not connected to the image border (holes in the part)."""
    outside = np.zeros((mask.shape[0] + 2, mask.shape[1] + 2), np.uint8)
    outside[1:-1, 1:-1] = mask
    cv2.floodFill(outside, None, (0, 0), 2)
    return int(np.count_nonzero(outside[1:-1, 1:-1] == 0))


def pieces(mask):
    return cv2.connectedComponents(mask.astype(np.uint8))[0] - 1


def test_dragging_a_curved_outline_leaves_no_gaps_or_slivers(tmp_path):
    lm = np.zeros((100, 100), np.uint8)
    cv2.circle(lm, (50, 50), 35, 2, -1)
    s = session_on(tmp_path, lm)
    k = int(np.argmax(s.rings[0][:, 0]))
    x, y = s.rings[0][k]
    assert s.move_point(0, k, int(x) + 12, int(y))
    mask = s.ann.label_map == 2
    assert enclosed_background(mask) == 0 and pieces(mask) == 1
    s.undo()
    k = int(np.argmax(s.rings[0][:, 0]))
    x, y = s.rings[0][k]
    assert s.move_point(0, k, int(x) - 12, int(y))
    mask = s.ann.label_map == 2
    assert enclosed_background(mask) == 0 and pieces(mask) == 1


def test_sweeping_over_a_fragment_of_the_same_part_keeps_all_of_it(tmp_path):
    lm = np.zeros((40, 60), np.uint8)
    lm[10:30, 10:30] = 2
    lm[22:27, 32:37] = 2  # a detached 5x5 fragment
    s = session_on(tmp_path, lm)
    ring = next(i for i, r in enumerate(s.rings) if (r == (29, 29)).all(1).any())
    k = int(np.flatnonzero((s.rings[ring] == (29, 29)).all(1))[0])
    assert s.move_point(ring, k, 45, 29)
    assert (s.ann.label_map[22:27, 32:37] == 2).all()


def test_dragging_across_a_hole_keeps_the_hole(tmp_path):
    lm = np.zeros((30, 50), np.uint8)
    lm[5:25, 5:40] = 2
    lm[10:20, 20:30] = 0  # a hole
    s = session_on(tmp_path, lm)
    k = next(i for i, r in enumerate(s.rings) if (r == (39, 24)).all(1).any())
    i = int(np.flatnonzero((s.rings[k] == (39, 24)).all(1))[0])
    assert s.move_point(k, i, 15, 8)  # the outer corner sweeps right across the hole
    assert not (s.ann.label_map[10:20, 20:30] == 2).any()  # hole pixels stay background


def test_a_hole_dragged_out_of_the_part_adds_nothing_outside(tmp_path):
    lm = np.zeros((50, 50), np.uint8)
    lm[10:40, 10:40] = 2
    lm[15:25, 15:25] = 0
    s = session_on(tmp_path, lm)
    hi = next(i for i, r in enumerate(s.rings) if (r[:, 0] > 10).all() and (r[:, 1] > 10).all())
    k = int(np.argmin(s.rings[hi].sum(1)))
    s.move_point(hi, k, 2, 2)
    outside = np.ones((50, 50), bool)
    outside[10:40, 10:40] = False
    assert not (s.ann.label_map[outside] == 2).any()


def test_a_move_that_crosses_its_own_outline_is_refused(s):
    before = s.ann.label_map.copy()
    assert not s.move_point(*corner(s, 24, 14), 24, 0)
    assert np.array_equal(s.ann.label_map, before) and "cross" in s.messages[-1]
    assert {tuple(p) for p in s.rings[0]} == {(5, 5), (24, 5), (24, 14), (5, 14)}


def test_a_move_blocked_by_protected_parts_keeps_the_outline(s):
    m = np.zeros(s.ann.size, bool)
    m[5:15, 25:30] = True
    s.ann.assign(m, 3)          # part 3 now touches part 2's right edge
    s.set_protect(True)
    before = s.ann.label_map.copy()
    assert not s.move_point(*corner(s, 24, 14), 29, 14)
    assert np.array_equal(s.ann.label_map, before) and "protected" in s.messages[-1]
    assert {tuple(p) for p in s.rings[0]} == {(5, 5), (24, 5), (24, 14), (5, 14)}


def test_an_inserted_point_lands_on_the_edge(s):
    ring, i = corner(s, 5, 14)
    j = (i + 1) % 4
    a, b = s.rings[0][i], s.rings[0][j]
    mid = (a + b) // 2
    off = (int(mid[0]), int(mid[1]) + 3) if a[1] == b[1] else (int(mid[0]) + 3, int(mid[1]))
    assert s.insert_point(0, i, *off)
    assert tuple(s.rings[0][i + 1]) == (int(mid[0]), int(mid[1]))


def test_a_partly_protected_move_shows_the_outline_the_pixels_really_have(s):
    m = np.zeros(s.ann.size, bool)
    m[5:15, 25:40] = True
    s.ann.assign(m, 3)          # part 3 touches part 2's right edge
    s.set_protect(True)
    s.move_point(*corner(s, 24, 14), 34, 18)  # part of the new triangle is part 3's (protected)
    assert s.ann.part_at(15, 30) == 2 and s.ann.part_at(13, 30) == 3
    assert s.ann.part_at(13, 30) == 3 and "protected" in s.messages[-1]
    from annofix.outline import fill
    shown = fill(s.rings, [-1] * len(s.rings), s.ann.size)
    assert (shown & (s.ann.label_map == 3)).sum() < 5  # re-traced from the pixels: not over part 3


def zigzag_bottom(s):
    """Part 2's bottom edge (row 14) given a 1-px zigzag: every other column reaches row 15."""
    for x in range(6, 24, 2):
        s.ann.label_map[15, x] = 2


def test_shift_g_shows_every_corner_of_the_real_edge(s):
    zigzag_bottom(s)
    few = len(s.rings[0])
    s.toggle_exact_points()
    assert s.points_exact and len(s.rings[0]) > few + 15      # each step of the zigzag has its corners
    s.toggle_exact_points()
    assert len(s.rings[0]) == few


def test_a_box_selects_the_points_inside_it(s):
    s.select_points_in(3, 12, 26, 16)                         # the bottom corners (5,14) and (24,14)
    ring, picked = s.point_selection
    assert ring == 0 and sorted(tuple(s.rings[0][i]) for i in picked) == [(5, 14), (24, 14)]
    s.select_points_in(40, 30, 50, 35)                        # nothing there
    assert s.point_selection is None


def test_l_makes_the_selected_stretch_one_straight_line_pixel_for_pixel(s):
    zigzag_bottom(s)
    s.toggle_exact_points()
    before = s.ann.label_map.copy()
    s.select_points_in(3, 13, 26, 16)                         # every corner along the bottom edge
    assert s.straighten_points()
    lm = s.ann.label_map
    assert not (lm[15] == 2).any() and (lm[14, 5:25] == 2).all()   # the zigzag is gone, on one line
    assert (lm[5, 5:25] == 2).all()                                  # the rest of the part untouched
    assert len(s.rings[0]) < 10 and s.point_selection is None
    assert s.undo() == "straighten points" and np.array_equal(s.ann.label_map, before)


def test_l_works_on_the_simplified_points_too(s):
    # The case in the screenshot: the simplified outline is already straight there, but the
    # pixels under it still zigzag. Straightening that stretch makes them follow it.
    zigzag_bottom(s)
    s.select_points_in(3, 12, 26, 17)
    assert s.straighten_points()
    lm = s.ann.label_map
    rows = [int(np.nonzero(lm[:, x] == 2)[0].max()) for x in range(6, 24)]
    assert max(rows) - min(rows) <= 1 and (rows == sorted(rows) or rows == sorted(rows, reverse=True))


def test_straightening_needs_two_points(s):
    s.select_points_in(3, 3, 7, 7)                            # one corner
    assert not s.straighten_points() and "two points" in s.messages[-1]


def test_a_straightened_stretch_against_a_neighbour_opens_no_gap(s):
    zigzag_bottom(s)
    s.ann.label_map[16:25, 5:25][s.ann.label_map[16:25, 5:25] == 0] = 5   # part 5 right below
    s.ann.label_map[15, 5:25][s.ann.label_map[15, 5:25] == 0] = 5
    s.toggle_exact_points()
    s.select_points_in(3, 13, 26, 16)
    assert s.straighten_points()
    lm = s.ann.label_map
    assert (lm[15, 5:25] == 5).all()                          # the teeth went to the neighbour


def test_more_and_fewer_points_step_through_the_levels(s):
    zigzag_bottom(s)
    counts = []
    s.set_point_level(0)
    for _ in range(len(s.POINT_LEVELS)):
        counts.append(len(s.rings[0]))
        s.more_points()
    assert counts == sorted(counts) and counts[-1] > counts[1] > counts[0] - 1
    assert s.point_level == len(s.POINT_LEVELS) - 1
    s.more_points()                                     # already the most: stays
    assert s.point_level == len(s.POINT_LEVELS) - 1
    s.fewer_points()
    assert s.point_level == len(s.POINT_LEVELS) - 2


def test_the_densest_level_has_a_point_every_few_pixels_and_changes_no_pixels(s):
    before = s.ann.label_map.copy()
    s.set_point_level(len(s.POINT_LEVELS) - 1)
    ring = s.rings[0]
    gaps = np.hypot(*np.diff(np.vstack([ring, ring[:1]]), axis=0).T)
    assert gaps.max() <= 5 and len(ring) > 10
    assert np.array_equal(s.ann.label_map, before)


def test_shift_g_jumps_to_every_corner_and_back_to_the_level_before(s):
    s.set_point_level(0)
    s.toggle_exact_points()
    assert s.points_exact and s.POINT_LEVELS[s.point_level][0] == "every corner"
    s.toggle_exact_points()
    assert s.point_level == 0
