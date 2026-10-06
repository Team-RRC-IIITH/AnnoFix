"""Session: images, saving and review."""
import pytest
from helpers import H, LABELS, W, make_dataset

from annofix.annotation import Annotation
from annofix.io import has_annotation, load_annotation, save_annotation
from annofix.session import Session


@pytest.fixture
def dataset(tmp_path):
    return make_dataset(tmp_path)


@pytest.fixture
def s(dataset):
    session = Session(*dataset, LABELS, annotator="tester")
    session.events = []
    session.subscribe(session.events.append)
    session.open(0)
    return session


def test_open_loads_annotation_and_reports_unknown_names(s):
    assert s.stem == "a" and s.ann.part_at(10, 10) == 2
    assert "object 4: class 'spaceship' is not in labels.yaml" in s.messages
    assert s.events == ["image"]


def test_image_without_annotation_opens_empty_and_writes_nothing(s, dataset):
    s.open(1)
    assert s.ann.objects == [] and s.ann.size == (H, W) and s.review_status(1) == "none"
    s.open(0)
    assert not has_annotation(dataset[1], "b")


def test_other_resolution_gets_a_matching_annotation(s):
    s.open(2)
    assert s.ann.size == (30, 50) == s.image.shape[:2]


def test_navigation_saves_edits_with_annotator(s, dataset):
    s.ann.set_part_material(2, "rubber")
    assert s.step(1) and s.stem == "b"
    back = load_annotation(dataset[1], "a")
    assert back.part(2).material == "rubber" and back.annotator == "tester"


def test_step_stops_at_the_ends(s):
    assert not s.step(-1) and s.stem == "a"


def test_save_only_when_dirty_and_never_mid_edit(s, dataset):
    assert s.save() is False
    s.ann.begin("stroke")
    s.ann.label_map[35, 5] = 2
    assert s.save() is False
    assert load_annotation(dataset[1], "a").part_at(35, 5) is None
    s.ann.commit()
    assert s.save() is True
    assert load_annotation(dataset[1], "a").part_at(35, 5) == 2


def test_review_status_and_next_unreviewed(s, dataset):
    save_annotation(Annotation.empty(H, W, review_status="reviewed"), dataset[1], "b")
    assert [s.review_status(i) for i in range(3)] == ["unreviewed", "reviewed", "none"]
    assert s.next_unreviewed() and s.stem == "c"


def test_mark_reviewed_requires_complete_labels(s, dataset):
    problems = s.mark_reviewed()
    assert "part 3 (object 1): no material" in problems and s.ann.review_status == "unreviewed"
    s.ann.set_part_material(3, "metal")
    s.ann.set_object_class(4, "mug")
    s.ann.set_part_name(5, "hull")
    s.ann.set_part_material(5, "metal")
    assert s.mark_reviewed() == []
    assert load_annotation(dataset[1], "a").review_status == "reviewed"
    assert s.review_status(0) == "reviewed"


def test_flag_toggles_and_saves(s, dataset):
    s.toggle_flag()
    assert load_annotation(dataset[1], "a").review_status == "flagged"
    s.toggle_flag()
    assert load_annotation(dataset[1], "a").review_status == "unreviewed"


def test_bad_annotation_file_leaves_current_image_open(s, dataset):
    save_annotation(Annotation.empty(10, 10), dataset[1], "b")  # wrong size for b.png
    with pytest.raises(ValueError, match="image is"):
        s.open(1)
    assert s.stem == "a" and s.ann.part_at(10, 10) == 2


def test_empty_folder_is_an_error(tmp_path):
    (tmp_path / "empty").mkdir()
    with pytest.raises(ValueError, match="no colour images"):
        Session(tmp_path / "empty", tmp_path / "ann", LABELS)
