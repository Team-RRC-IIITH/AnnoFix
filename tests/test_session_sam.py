"""Session SAM actions, driven synchronously by FakeBackend (what the window's controller does
asynchronously)."""
import numpy as np
import pytest
from helpers import LABELS, make_dataset

from annofix.sam_backend import FakeBackend
from annofix.session import Session


@pytest.fixture
def dataset(tmp_path):
    return make_dataset(tmp_path)


def make_session(dataset, drive=True):
    session = Session(*dataset, LABELS)
    session.sam_enabled = True
    backend = FakeBackend()

    def controller(kind):
        if kind == "image":
            backend.set_image(session.image)
        elif kind == "sam_request" and drive:
            req = session.take_sam_request()
            out = backend.predict(req.points, req.labels, req.box, req.prev_logits, req.multimask)
            session.sam_receive(req.token, req.seq, *out)

    session.subscribe(controller)
    session.open(0)
    session.backend = backend
    return session


@pytest.fixture
def s(dataset):
    return make_session(dataset)


def test_exclude_click_refines_and_enter_replaces_the_part(s):
    s.select_at(10, 10)
    s.set_tool("sam")
    n = len(s.sam.labels)
    s.sam_click(22, 10, False)
    assert s.sam.labels[n:] == [0] and len(s.sam.candidates) == 2   # SAM's one answer, then smoothed
    assert not s.sam.is_alternative
    expected = s.sam.mask.copy()
    assert not expected[10, 22]
    assert s.sam_accept()
    assert s.sam is None and s.selected == [2] and s.tool == "sam"
    assert np.array_equal(s.ann.part_mask(2), expected)


def test_accept_is_one_undo_step(s):
    before = s.ann.label_map.copy()
    s.select_at(10, 10)
    s.set_tool("sam")
    s.sam_accept()
    assert s.undo() == "sam"
    assert np.array_equal(s.ann.label_map, before)


def test_shift_enter_adds_instead_of_replacing(s):
    s.select_at(10, 10)
    s.set_tool("sam")
    s.sam_cycle()                                      # SAM's answer (the box), bigger than the part
    before = s.ann.part_area(2)
    s.sam_accept(union=True)
    assert s.ann.part_area(2) > before and s.ann.part_at(5, 5) == 2  # the old corner is kept


def test_tab_cycles_candidates_and_the_preview_follows(s):
    s.select_at(10, 10)
    s.set_tool("sam")
    first = s.sam_preview().copy()
    s.sam_cycle()
    assert s.sam.index == 0 and not s.sam.is_alternative and not np.array_equal(s.sam_preview(), first)


def test_click_on_a_part_in_the_sam_tool_starts_fixing_it(s):
    s.set_tool("sam")
    assert s.sam is None
    s.sam_click(35, 10, True)  # the handle, part 3
    assert s.selected == [3] and s.sam.target == 3 and s.sam.points == [(35, 10)]
    s.sam_click(36, 11, False)  # right-click with no edit open would be ignored; here it refines
    assert s.sam.labels == [1, 0]


def test_right_click_without_an_edit_is_ignored(s):
    s.set_tool("sam")
    s.sam_click(35, 10, False)
    assert s.sam is None


def test_click_on_background_makes_a_new_object(s):
    s.set_tool("sam")
    s.sam_click(50, 35, True)
    assert s.sam.target is None and s.sam.object_id is None
    n = len(s.ann.objects)
    assert s.sam_accept()
    assert len(s.ann.objects) == n + 1 and s.ann.part_at(35, 50) == s.selected[0]


def test_box_prompt(s):
    s.set_tool("sam")
    s.sam_box(45, 30, 55, 38)
    assert s.sam.box == (45, 30, 55, 38) and s.sam.mask.sum() == 11 * 9


def test_n_with_sam_adds_a_part_to_the_selected_object(s):
    s.select_at(10, 10)
    assert s.new_part() is None
    assert s.tool == "sam" and s.sam.target is None and s.sam.object_id == 1 and s.sam.points == []
    s.sam_click(50, 35, True)
    s.sam_accept()
    assert s.ann.object_of(s.selected[0]).id == 1


def test_shift_n_with_sam_makes_a_new_object(s):
    s.select_at(10, 10)
    assert s.new_object() is None and s.sam.object_id is None and s.sam.target is None


def test_n_without_sam_still_makes_an_empty_part_for_the_brush(s):
    s.sam_enabled = False
    s.select_at(10, 10)
    pid = s.new_part()
    assert s.tool == "brush" and s.ann.object_of(pid).id == 1


def test_sam_tool_unavailable_says_so(s):
    s.sam_enabled = False
    s.select_at(10, 10)
    s.set_tool("sam")
    assert s.tool == "select" and s.sam is None and "SAM is not available" in s.messages[-1]


def test_accept_never_changes_hidden_parts(s):
    s.select_at(10, 25)                 # part 5
    s.set_hidden("object", 1, True)     # hide the mug (parts 2 and 3)
    s.set_tool("sam")
    s.sam_cycle()                       # the radius-16 disc reaches into the mug
    assert s.sam_accept()
    assert s.ann.part_area(2) == 200 and s.ann.part_area(3) == 100


def test_accept_respects_protect(s):
    s.select_at(10, 25)
    s.set_protect(True)
    s.set_tool("sam")
    s.sam_cycle()
    s.sam_accept()
    assert s.ann.part_area(2) == 200


def test_switching_tool_or_image_drops_the_edit(s):
    s.select_at(10, 10)
    s.set_tool("sam")
    s.set_tool("brush")
    assert s.sam is None
    s.set_tool("sam")
    token = s.image_token
    s.step(1)
    assert s.sam is None and s.image_token == token + 1


def test_undo_that_removes_the_target_cancels_the_edit(s):
    s.set_tool("sam")
    s.sam_click(50, 35, True)
    s.sam_accept()                      # a new object + part
    new = s.selected[0]
    s.set_tool("sam")                   # R again: fix the new part
    assert s.sam.target == new
    s.undo()                            # removes the new object
    assert s.sam is None


def test_stale_result_after_a_newer_click_is_ignored(dataset):
    s = make_session(dataset, drive=False)
    s.select_at(10, 10)
    s.set_tool("sam")
    r1 = s.take_sam_request()
    x, y = s.sam.points[0]
    s.sam_click(x + 7, y, False)
    r2 = s.take_sam_request()
    b = s.backend
    assert not s.sam_receive(r1.token, r1.seq, *b.predict(r1.points, r1.labels, r1.box, r1.prev_logits, r1.multimask))
    assert s.sam_receive(r2.token, r2.seq, *b.predict(r2.points, r2.labels, r2.box, r2.prev_logits, r2.multimask))


def test_result_for_an_image_you_left_is_ignored(dataset):
    s = make_session(dataset, drive=False)
    s.select_at(10, 10)
    s.set_tool("sam")
    r1 = s.take_sam_request()
    out = s.backend.predict(r1.points, r1.labels, r1.box, r1.prev_logits, r1.multimask)
    s.step(1)
    assert not s.sam_receive(r1.token, r1.seq, *out) and s.sam is None


def test_answer_for_a_cancelled_fix_is_not_applied_to_the_next_one(dataset):
    s = make_session(dataset, drive=False)
    s.select_at(10, 10)
    s.set_tool("sam")
    r_a = s.take_sam_request()          # fix A (part 2) is asked...
    s.sam_cancel()                      # ...Esc while SAM is busy...
    s.sam_click(10, 25, True)           # ...and fix B starts on part 5
    r_b = s.take_sam_request()
    b = s.backend
    assert not s.sam_receive(r_a.token, r_a.seq, *b.predict(r_a.points, r_a.labels, r_a.box, r_a.prev_logits, r_a.multimask))
    assert s.sam.mask is None
    assert s.sam_receive(r_b.token, r_b.seq, *b.predict(r_b.points, r_b.labels, r_b.box, r_b.prev_logits, r_b.multimask))


def test_r_on_a_hidden_part_is_refused(s):
    s.select_at(10, 10)
    s.set_hidden("part", 2, True)
    s.set_tool("sam")
    assert s.sam is None and "hidden" in s.messages[-1]
    assert s.ann.part_area(2) == 200


def test_click_where_a_non_solo_part_sits_starts_a_new_object(s):
    s.select_at(10, 10)
    s.toggle_solo()                     # only the mug is visible; part 5 looks like background
    s.set_tool("sam")
    s.sam_cancel()
    s.sam_click(10, 25, True)
    assert s.sam.target is None
    s.sam_accept()
    assert s.ann.part_area(5) == 100


def test_accept_refuses_a_target_hidden_mid_fix(s):
    s.select_at(10, 10)
    s.set_tool("sam")
    s.set_hidden("part", 2, True)
    assert not s.sam_accept() and "hidden" in s.messages[-1]
    assert s.ann.part_area(2) == 200


def test_enter_waits_for_the_answer_to_the_newest_click(dataset):
    s = make_session(dataset, drive=False)
    b = s.backend

    def answer(r):
        return s.sam_receive(r.token, r.seq, *b.predict(r.points, r.labels, r.box, r.prev_logits, r.multimask))

    s.select_at(10, 10)
    s.set_tool("sam")
    assert not s.sam_accept() and "still working" in s.messages[-1]   # no candidate yet
    answer(s.take_sam_request())
    x, y = s.sam.points[0]
    s.sam_click(x + 7, y, False)
    r2 = s.take_sam_request()
    assert not s.sam_accept() and "still working" in s.messages[-1]   # the exclude click isn't answered
    answer(r2)
    assert s.sam_accept()
    assert s.ann.part_at(y, x + 7) != 2


def test_sam_click_outside_the_image_is_ignored(s):
    s.set_tool("sam")
    s.sam_click(-40, -40, True)
    s.sam_click(500, 10, True)
    assert s.sam is None


def test_sam_box_is_clamped_to_the_image(s):
    s.set_tool("sam")
    s.sam_box(-10, -5, 70, 50)
    assert s.sam.box == (0, 0, 59, 39)


def test_an_empty_sam_mask_cannot_be_accepted(dataset):
    s = make_session(dataset, drive=False)
    s.set_tool("sam")
    s.sam_click(50, 35, True)
    r = s.take_sam_request()
    s.sam_receive(r.token, r.seq, np.zeros((1, 40, 60), bool), np.array([0.9], np.float32),
                  np.zeros((1, 256, 256), np.float32))
    n = len(s.ann.objects)
    assert not s.sam_accept() and "empty" in s.messages[-1] and len(s.ann.objects) == n


def test_selecting_another_part_cancels_the_fix_and_r_restarts(s):
    s.select_at(10, 10)
    s.set_tool("sam")
    assert s.sam.target == 2
    s.select_next_object()
    assert s.sam is None and s.selected == [5]
    s.set_tool("sam")
    assert s.sam.target == 5


def test_a_refused_sam_tool_resyncs_the_toolbar(dataset):
    # The toolbar's SAM button checks itself when clicked; if the Session refuses, the window
    # must hear "view" to check the real tool again.
    s = Session(*dataset, LABELS)
    s.open(0)
    kinds = []
    s.subscribe(kinds.append)
    s.set_tool("sam")
    assert s.tool == "select" and "view" in kinds and "not available" in s.messages[-1]


def test_r_improves_the_selected_part_from_its_own_mask(s):
    s.select_at(10, 10)                                # body #2 = rows 5-14, columns 5-24
    s.set_tool("sam")
    e = s.sam
    assert e.target == 2 and e.box is not None
    x0, y0, x1, y1 = e.box
    assert x0 <= 5 and y0 <= 5 and x1 >= 24 and y1 >= 14 and x1 < 30   # the part's own box, padded
    assert e.labels.count(1) >= 1
    assert all(s.ann.part_at(y, x) == 2 for (x, y), v in zip(e.points, e.labels) if v)
    assert np.array_equal(e.reference, s.ann.part_mask(2))
    # FakeBackend answers with the padded box (504 px for a 200 px part): too unlike the part,
    # so the part smoothed comes first and SAM's answer is one Tab away
    assert e.is_alternative and len(e.candidates) == 2
    s.sam_cycle(1)
    assert not e.is_alternative and s.sam_preview() is not None


def test_enter_keeps_the_part_itself(s):
    s.select_at(10, 10)
    s.set_tool("sam")
    assert s.sam.is_alternative                        # the smoothed mask (see above)
    expected = s.sam.mask.copy()
    obj = s.ann.object_of(2)
    assert s.sam_accept()
    part = next(p for p in obj.parts if p.id == 2)
    assert np.array_equal(s.ann.part_mask(2), expected)
    assert (s.ann.object_of(2).id, part.name, part.material) == (1, "body", "metal")


def test_a_wrong_part_can_still_be_redrawn_from_a_click(s):
    s.select_at(10, 10)
    s.set_tool("sam")
    s.sam_cancel()                                     # Esc: no fix open, still the SAM tool
    s.sam_click(10, 10, True)
    assert s.sam.target == 2 and s.sam.labels == [1] and s.sam.box is None and len(s.sam.candidates) == 3
