"""Straighten (L, Shift+L): a part redrawn as the fill of its straight outline, the one the
points tool shows, so its zigzag edges become straight lines."""
import cv2
import numpy as np

from annofix.straighten import straighten


def edge_error(mask, a, b, c):
    """Largest distance (px) from the line a*x + b*y + c = 0 of the mask's top edge (its first
    pixel in each column), away from the wedge's corners."""
    cols = range(25, mask.shape[1] - 25)
    tops = [(x, int(np.argmax(mask[:, x]))) for x in cols if mask[:, x].any()]
    return max(abs(a * x + b * y + c) / np.hypot(a, b) for x, y in tops)


def jagged_wedge(h=120, w=160, seed=0):
    """A part below the line y = 0.5 x + 20, its edge zigzagging by 1-2 px like the pipeline's."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[:h, :w]
    noise = rng.integers(-2, 3, size=w)[None, :]
    lm = np.zeros((h, w), np.uint16)
    lm[(yy > 0.5 * xx + 20 + noise) & (yy < h - 10) & (xx > 10) & (xx < w - 10)] = 2
    return lm


def test_a_zigzag_edge_becomes_straight():
    lm = jagged_wedge()
    new, report = straighten(lm, [2])
    assert report == {("part", 2): "straightened"}
    before = edge_error(lm == 2, 0.5, -1, 20)
    after = edge_error(new == 2, 0.5, -1, 20)
    assert before >= 2 and after <= 1.5


def test_a_square_hole_is_straightened_and_stays_a_hole():
    rng = np.random.default_rng(1)
    lm = np.zeros((120, 120), np.uint16)
    lm[10:110, 10:110] = 2
    lm[40:80, 40:80] = 0                               # a window in a frame
    for x in range(40, 80):                             # its top edge zigzags by a pixel
        lm[40 + int(rng.integers(-1, 2)) if rng.random() < 0.5 else 40, x] ^= 2
    new, report = straighten(lm, [2])
    assert report[("part", 2)] == "straightened"
    assert new[60, 60] == 0 and (new[40:80, 40:80] == 0).mean() > 0.95


def test_round_parts_are_left_as_they_are():
    lm = np.zeros((160, 160), np.uint16)
    cv2.circle(lm, (80, 80), 60, 2, -1)
    new, report = straighten(lm, [2], whole_image=True)
    assert report[("part", 2)] == "left as is" and np.array_equal(new, lm)   # no facets on a bowl


def test_a_shared_edge_never_opens_a_gap():
    # Two parts fill the whole image along a zigzag border; only one is straightened.
    lm = jagged_wedge()
    lm[lm == 0] = 3
    new, _ = straighten(lm, [2])
    assert not (new == 0).any()                      # its teeth went to the neighbour, not to background
    assert edge_error(new == 2, 0.5, -1, 20) <= 1.5


def test_hidden_parts_are_never_changed_or_taken():
    lm = jagged_wedge()
    lm[lm == 0] = 3
    locked = np.zeros(65536, bool)
    locked[3] = True
    new, report = straighten(lm, [2, 3], locked)
    assert report[("part", 3)] == "hidden"
    assert np.array_equal(new == 3, lm == 3)


def test_protect_only_grows_into_background():
    lm = jagged_wedge()
    lm[lm == 0] = 3
    new, _ = straighten(lm, [2], protect=True)
    assert not ((lm == 3) & (new == 2)).any()


def test_specks_too_small_for_an_outline_are_kept():
    lm = np.zeros((40, 40), np.uint16)
    lm[5:7, 5:7] = 2                                    # 4 px: no outline points
    new, report = straighten(lm, [2])
    assert report == {("part", 2): "kept"} and np.array_equal(new, lm)


def test_small_holes_stay_holes_and_a_part_inside_one_is_never_swallowed():
    lm = np.zeros((40, 40), np.uint16)
    lm[5:35, 5:35] = 2
    lm[10:12, 10] = 0                                   # a 2 px hole
    lm[20, 20] = 3                                      # a 1 px part inside part 2
    new, _ = straighten(lm, [2])
    assert (new[10:12, 10] == 0).all() and new[20, 20] == 3
