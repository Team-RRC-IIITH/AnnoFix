"""Auto-clean (Shift+R) in the Session: propose, review, accept as one undo step or discard."""
import numpy as np
import pytest
from helpers import LABELS, make_dataset

from annofix import refine
from annofix.sam_backend import FakeBackend
from annofix.session import Session


@pytest.fixture
def s(tmp_path):
    s = Session(*make_dataset(tmp_path), LABELS)
    s.open(0)
    return s


def answer(s):
    """Run the Session's SAM batch the way the worker does, with the fake SAM."""
    token, batch, jobs = s.take_clean_request()
    backend = FakeBackend()
    backend.set_image(s.image)
    return s.clean_receive(token, batch, [refine.run_job(backend.predict, j) for j in jobs])


def test_without_sam_the_proposal_is_ready_at_once_and_accept_is_one_undo_step(s):
    before = s.ann.label_map.copy()
    kinds = []
    s.subscribe(kinds.append)
    assert s.start_auto_clean()
    assert s.auto_clean is not None and "refine" in kinds and not s.ann.dirty
    preview = s.auto_clean_preview()
    assert preview is not None and preview.shape == before.shape
    assert np.array_equal(s.ann.label_map, before)  # nothing changes before you accept
    assert s.accept_auto_clean()
    assert s.auto_clean is None and not np.array_equal(s.ann.label_map, before)
    assert "smoothed" in s.messages[-1]
    s.undo()
    assert np.array_equal(s.ann.label_map, before)


def test_with_sam_the_masks_go_to_sam_as_one_batch(s):
    s.sam_enabled = True
    kinds = []
    s.subscribe(kinds.append)
    assert s.start_auto_clean()
    assert "refine_request" in kinds and s.clean_pending and s.auto_clean is None
    token, batch, jobs = s.take_clean_request()
    assert {j.key for j in jobs} == {("object", 1), ("object", 4), ("part", 2), ("part", 3)}
    perfect = {j.key: j for j in jobs}

    def exact(job):  # a SAM that answers with exactly the old mask
        return lambda *a, **k: (refine.place(refine.Result(job.key, job.window, job.original, 1.0),
                                             s.ann.size)[None], np.ones(1), np.zeros((1, 256, 256)))

    results = [refine.run_job(exact(j), j) for j in perfect.values()]
    assert s.clean_receive(token, batch, results)
    assert not s.clean_pending and s.auto_clean is not None
    assert s.auto_clean.report[("object", 1)] == "sam" and "snapped" in s.auto_clean.summary()


def test_a_sam_failure_falls_back_to_smoothing(s):
    s.sam_enabled = True
    s.start_auto_clean()
    s.take_clean_request()
    assert s.clean_failed()
    assert not s.clean_pending and set(s.auto_clean.report.values()) == {"smoothed"}


def test_an_old_answer_is_ignored(s):
    s.sam_enabled = True
    s.start_auto_clean()
    token, batch, jobs = s.take_clean_request()
    s.start_auto_clean()  # asked again: the first answer is stale
    assert not s.clean_receive(token, batch, [])
    assert s.clean_pending


def test_esc_or_any_edit_drops_the_proposal(s):
    s.start_auto_clean()
    assert s.discard_auto_clean() and s.auto_clean is None
    s.start_auto_clean()
    s.select_part(2)
    s.set_material("metal")       # the image changed under the proposal
    assert s.auto_clean is None
    s.start_auto_clean()
    s.step(1)
    assert s.auto_clean is None


def test_hiding_part_of_an_object_keeps_the_whole_object(s):
    s.set_hidden("part", 3, True)     # part 3 is one of object 1's two parts
    before = s.ann.label_map.copy()
    s.start_auto_clean()
    s.accept_auto_clean()
    mug = np.isin(before, [2, 3])
    assert np.array_equal(s.ann.label_map[mug], before[mug])


def test_hidden_parts_are_left_alone(s):
    s.set_hidden("part", 5, True)
    before = s.ann.label_map == 5
    s.start_auto_clean()
    s.accept_auto_clean()
    assert np.array_equal(s.ann.label_map == 5, before)


def test_l_proposes_the_selected_part_straightened_and_enter_applies_it_as_one_undo_step(tmp_path):
    from helpers import LABELS, make_dataset
    from annofix.session import Session
    s = Session(*make_dataset(tmp_path), LABELS)
    s.open(0)
    s.ann.label_map[14, 5:25:3] = 0                     # a zigzag bottom edge on body #2
    before = s.ann.label_map.copy()
    s.select_at(10, 10)
    assert s.start_straighten()
    assert s.auto_clean.action == "straighten" and "1 straightened" in s.auto_clean.summary()
    assert s.auto_clean_preview() is not None
    assert s.accept_auto_clean()
    assert s.undo() == "straighten" and np.array_equal(s.ann.label_map, before)


def test_shift_l_straightens_every_part_and_esc_drops_it(tmp_path):
    from helpers import LABELS, make_dataset
    from annofix.session import Session
    s = Session(*make_dataset(tmp_path), LABELS)
    s.open(0)
    assert s.start_straighten(whole_image=True)
    assert set(s.auto_clean.report) == {("part", 2), ("part", 3), ("part", 5)}
    assert s.discard_auto_clean() and s.auto_clean is None


def test_l_with_nothing_selected_says_what_to_do(tmp_path):
    from helpers import LABELS, make_dataset
    from annofix.session import Session
    s = Session(*make_dataset(tmp_path), LABELS)
    s.open(0)
    assert not s.start_straighten() and "select a part" in s.messages[-1]
