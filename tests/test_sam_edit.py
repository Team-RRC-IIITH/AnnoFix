import numpy as np
import pytest

from annofix.sam_backend import FakeBackend
from annofix.sam_edit import (PREVIEW_ADDED, PREVIEW_KEPT, PREVIEW_REMOVED, PREVIEW_TAKEN, SamEdit,
                              interior_point, preview_map)


def backend():
    b = FakeBackend()
    b.set_image(np.zeros((40, 60, 3), np.uint8))
    return b


def answer(edit, req, b):
    masks, scores, logits = b.predict(req.points, req.labels, req.box, req.prev_logits, req.multimask)
    return edit.receive(req.token, req.seq, masks, scores, logits)


def test_first_include_point_asks_for_three_candidates():
    req = SamEdit(token=7, target=2).add_point(30, 20, True)
    assert (req.token, req.seq, req.multimask, req.prev_logits) == (7, 1, True, None)
    assert req.points.tolist() == [[30.0, 20.0]] and req.labels.tolist() == [1] and req.box is None


def test_exclude_only_waits_for_an_include_point():
    e = SamEdit(1)
    assert e.add_point(5, 5, False) is None and e.seq == 0
    req = e.add_point(10, 10, True)
    assert req.labels.tolist() == [0, 1] and not req.multimask


def test_candidates_sorted_by_area_best_score_chosen_and_cycled():
    b, e = backend(), SamEdit(1, 2)
    assert answer(e, e.add_point(30, 20, True), b)
    areas = [int(m.sum()) for m in e.candidates]
    assert areas == sorted(areas) and e.score == pytest.approx(0.9) and e.index == 1 and e.received_seq == 1
    e.cycle()
    assert e.index == 2
    e.cycle()
    assert e.index == 0
    e.cycle(-1)
    assert e.index == 2


def test_refinement_feeds_back_the_chosen_candidates_logits():
    b, e = backend(), SamEdit(1, 2)
    answer(e, e.add_point(30, 20, True), b)
    e.cycle()  # the radius-16 candidate
    req = e.add_point(37, 20, False)
    assert not req.multimask and req.prev_logits.shape == (1, 256, 256) and req.prev_logits[0, 0, 0] == 16
    assert answer(e, req, b) and len(e.candidates) == 1 and not e.mask[20, 36]


def test_stale_results_are_dropped():
    b, e = backend(), SamEdit(1, 2)
    r1 = e.add_point(30, 20, True)
    r2 = e.add_point(37, 20, False)
    out1 = b.predict(r1.points, r1.labels, r1.box, r1.prev_logits, r1.multimask)
    out2 = b.predict(r2.points, r2.labels, r2.box, r2.prev_logits, r2.multimask)
    assert not e.receive(r1.token, r1.seq, *out1)  # an older prompt
    assert not e.receive(99, r2.seq, *out2)        # another image
    assert e.receive(r2.token, r2.seq, *out2) and e.mask is not None


def test_box_prompt_is_normalised():
    req = SamEdit(1).set_box(20, 15, 10, 5)
    assert req.box.tolist() == [10, 5, 20, 15] and not req.multimask and req.points is None


def test_interior_point_is_the_deepest_pixel():
    m = np.zeros((20, 30), bool)
    m[2:18, 2:12] = True    # a fat block...
    m[8:12, 12:28] = True   # ...with a thin arm
    x, y = interior_point(m)
    assert m[y, x] and 4 <= x <= 9 and 6 <= y <= 13
    assert interior_point(np.zeros((5, 5), bool)) is None


def test_preview_shows_kept_added_taken_removed():
    lm = np.zeros((6, 6), np.uint16)
    lm[0:3, 0:3] = 2
    lm[3:6, 3:6] = 3
    mask = np.zeros((6, 6), bool)
    mask[1:5, 1:5] = True
    p = preview_map(lm, 2, mask)
    assert (p[1, 1], p[0, 0], p[1, 4], p[4, 4]) == (PREVIEW_KEPT, PREVIEW_REMOVED, PREVIEW_ADDED, PREVIEW_TAKEN)
    assert preview_map(lm, 2, mask, protect=True)[4, 4] == 0
    assert preview_map(lm, 2, mask, union=True)[0, 0] == 0
    locked = np.zeros(65536, bool)
    locked[3] = True
    assert preview_map(lm, 2, mask, locked=locked)[4, 4] == 0
    assert preview_map(lm, None, mask)[1, 1] == PREVIEW_TAKEN  # a new part would take from part 2


def test_requests_from_a_shared_counter_never_repeat():
    import itertools

    counter = itertools.count(1)
    b = backend()
    a = SamEdit(1, 2, counter=counter)
    r1 = a.add_point(30, 20, True)
    second = SamEdit(1, 5, counter=counter)  # a new fix on the same image
    r2 = second.add_point(10, 25, True)
    assert r2.seq != r1.seq
    assert not second.receive(r1.token, r1.seq, *b.predict(r1.points, r1.labels, r1.box, r1.prev_logits, r1.multimask))


def test_preview_marks_removed_from_what_replace_really_removes():
    lm = np.zeros((6, 6), np.uint16)
    lm[0:3, 0:3] = 2
    mask = np.zeros((6, 6), bool)
    mask[0:2, 0:2] = True
    locked = np.zeros(65536, bool)
    locked[2] = True  # the target itself is locked: replace would remove all of it
    assert (preview_map(lm, 2, mask, locked=locked)[lm == 2] == PREVIEW_REMOVED).all()
