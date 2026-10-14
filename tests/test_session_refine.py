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
    assert answer(s)
    assert not s.clean_pending and s.auto_clean is not None
    assert set(s.auto_clean.report) >= {("object", 1), ("object", 4)}


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


def test_hidden_parts_are_left_alone(s):
    s.set_hidden("part", 5, True)
    before = s.ann.label_map == 5
    s.start_auto_clean()
    s.accept_auto_clean()
    assert np.array_equal(s.ann.label_map == 5, before)
