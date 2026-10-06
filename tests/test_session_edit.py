"""Session: selection, labels, strokes and structural edits."""
import time

import cv2
import numpy as np
import pytest
from helpers import LABELS, make_dataset

from annofix.io import load_annotation
from annofix.session import Session


@pytest.fixture
def dataset(tmp_path):
    return make_dataset(tmp_path)


@pytest.fixture
def s(dataset):
    session = Session(*dataset, LABELS, annotator="tester")
    session.open(0)
    return session


def test_selection_click_and_add(s):
    s.select_at(10, 10)
    assert s.selected == [2]
    s.select_at(35, 10, add=True)
    assert s.selected == [2, 3]
    s.select_at(35, 10, add=True)
    assert s.selected == [2]
    s.select_at(55, 35)  # background clears
    assert s.selected == []
    s.select_at(500, 500)  # outside the image is background too
    assert s.selected == []


def test_material_applies_to_every_selected_part_as_one_undo_step(s):
    s.select_at(10, 10)
    s.select_at(35, 10, add=True)
    assert s.set_material_index(0)
    assert s.ann.part(2).material == s.ann.part(3).material == "plastic"
    s.undo()
    assert s.ann.part(2).material == "metal" and s.ann.part(3).material is None
    assert not s.set_material_index(9)


def test_names_and_classes(s):
    s.select_at(35, 10)
    s.set_part_name("  rim ")
    s.set_object_class("mug")
    assert s.ann.part(3).name == "rim" and s.ann.object(1).cls == "mug"
    s.set_part_name("")
    assert s.ann.part(3).name is None


def test_unknown_class_typed_by_user_is_warned(s):
    s.select_at(10, 10)
    s.set_object_class("teapot")
    assert "object 1: class 'teapot' is not in labels.yaml" in s.messages


def test_brush_stroke_is_continuous_and_one_undo_step(s):
    s.select_at(10, 10)
    s.set_tool("brush")
    s.set_brush_radius(1)
    before = s.ann.label_map.copy()
    assert s.begin_stroke(5, 35)
    s.stroke_to(55, 35)
    s.end_stroke()
    assert (s.ann.label_map[35, 5:56] == 2).all()  # no gaps between the two mouse events
    assert s.undo() == "brush"
    assert np.array_equal(s.ann.label_map, before)


def test_brush_needs_a_selected_part(s):
    s.set_tool("brush")
    assert not s.begin_stroke(5, 35) and not s.ann.in_transaction
    assert "select a part" in s.messages[-1]


def test_protect_others_keeps_neighbours(s):
    s.select_at(10, 10)
    s.set_tool("brush")
    s.set_brush_radius(2)
    s.set_protect(True)
    s.begin_stroke(35, 10)
    s.end_stroke()
    assert s.ann.part_at(10, 35) == 3
    s.set_protect(False)
    s.begin_stroke(35, 10)
    s.end_stroke()
    assert s.ann.part_at(10, 35) == 2


def test_eraser_only_erases_the_selected_part(s):
    s.select_at(10, 10)
    s.set_tool("eraser")
    s.set_brush_radius(30)
    s.begin_stroke(25, 10)
    s.end_stroke()
    assert s.ann.part_area(2) == 0 and s.ann.part_area(3) == 100


def test_cancel_stroke_restores(s):
    s.select_at(10, 10)
    s.set_tool("brush")
    before = s.ann.label_map.copy()
    s.begin_stroke(5, 35)
    s.stroke_to(50, 35)
    s.cancel_stroke()
    assert np.array_equal(s.ann.label_map, before) and not s.ann.in_transaction


def test_autosave_mid_stroke_waits_for_the_stroke(s, dataset):
    s.select_at(10, 10)
    s.set_tool("brush")
    s.set_brush_radius(1)
    s.begin_stroke(5, 35)
    assert s.save() is False
    assert load_annotation(dataset[1], "a").part_at(35, 5) is None
    s.stroke_to(20, 35)
    s.end_stroke()
    assert s.save() is True
    assert load_annotation(dataset[1], "a").part_at(35, 20) == 2


def test_navigating_mid_stroke_keeps_the_stroke(s, dataset):
    s.select_at(10, 10)
    s.set_tool("brush")
    s.set_brush_radius(1)
    s.begin_stroke(5, 35)
    s.stroke_to(20, 35)
    s.step(1)
    assert load_annotation(dataset[1], "a").part_at(35, 20) == 2


def test_undo_prunes_selection_of_parts_that_no_longer_exist(s):
    pid = s.new_part()  # nothing selected -> a new object with one part
    assert s.selected == [pid] and s.tool == "brush"
    s.toggle_part_hidden(pid)
    s.toggle_solo()
    s.undo()
    assert s.selected == [] and pid not in s.hidden_parts and s.solo_object is None
    s.select_at(10, 10)
    assert s.selected == [2]


def test_new_part_goes_into_the_selected_object(s):
    s.select_at(10, 10)
    pid = s.new_part()
    assert s.ann.object_of(pid).id == 1 and s.selected == [pid]


def test_merge_selected_parts(s):
    s.select_at(10, 10)
    s.select_at(35, 10, add=True)
    assert s.merge_selected_parts()
    assert s.ann.part_area(2) == 300 and 3 not in s.ann.part_ids() and s.selected == [2]


def test_merge_selected_objects_like_a_split_robot_arm(s):
    s.select_at(10, 10)            # part 2 of object 1
    s.select_at(10, 25, add=True)  # part 5 of object 4
    assert s.merge_selected_objects()
    assert [o.id for o in s.ann.objects] == [1]
    assert sorted(p.id for p in s.ann.object(1).parts) == [2, 3, 5]
    s.undo()
    assert [o.id for o in s.ann.objects] == [1, 4]


def test_delete_selected_object_and_parts(s):
    s.select_at(35, 10)
    s.delete_selected_object()
    assert [o.id for o in s.ann.objects] == [4] and s.selected == []
    s.select_at(10, 25)
    s.delete_selected_parts()
    assert s.ann.part_ids() == []


def test_move_promote_and_clean(s):
    s.move_part_to(5, 1)
    assert s.ann.object_of(5).id == 1
    oid = s.promote_part(5)
    assert s.ann.object_of(5).id == oid
    s.clean_part(2)
    assert s.ann.part_area(2) == 200


def test_next_object_cycles(s):
    s.select_next_object()
    assert s.selected == [2]
    s.select_next_object()
    assert s.selected == [5]
    s.select_next_object()
    assert s.selected == [2]
    s.select_next_object(-1)
    assert s.selected == [5]


def test_view_options(s):
    s.select_at(10, 10)
    s.toggle_solo()
    assert s.solo_object == 1
    s.toggle_solo()
    assert s.solo_object is None
    s.set_hidden("object", 4, True)
    assert s.hidden_objects == {4}
    s.set_hidden("object", 4, False)
    assert s.hidden_objects == set()
    s.set_colour_mode("part")
    assert s.colour_mode == "part"
    with pytest.raises(ValueError):
        s.set_colour_mode("rainbow")
    s.set_brush_radius(0)
    assert s.brush_radius == 1


def test_stroke_segments_are_fast_enough_for_live_painting(tmp_path):
    images = tmp_path / "images"
    images.mkdir()
    cv2.imwrite(str(images / "big.png"), np.zeros((720, 1280, 3), np.uint8))
    s = Session(images, tmp_path / "ann", LABELS)
    s.open(0)
    s.new_object()
    s.set_brush_radius(10)
    s.begin_stroke(100, 100)
    times = []
    for i in range(30):
        t = time.perf_counter()
        s.stroke_to(100 + 30 * i, 100 + 15 * i)
        times.append(time.perf_counter() - t)
    s.end_stroke()
    assert sorted(times)[15] < 0.015


def complete_labels(s):
    s.ann.set_part_material(3, "metal")
    s.ann.set_object_class(4, "mug")
    s.ann.set_part_name(5, "hull")
    s.ann.set_part_material(5, "metal")


def test_reviewed_image_that_no_longer_validates_is_saved_unreviewed(s, dataset):
    complete_labels(s)
    assert s.mark_reviewed() == []
    s.new_part()  # an accidental N: an empty, unlabelled part
    s.step(1)
    assert load_annotation(dataset[1], "a").review_status == "unreviewed"
    assert s.review_status(0) == "unreviewed"


def test_edits_that_keep_labels_complete_keep_reviewed(s, dataset):
    complete_labels(s)
    assert s.mark_reviewed() == []
    s.select_at(10, 10)
    s.set_tool("brush")
    s.set_brush_radius(1)
    s.begin_stroke(5, 35)
    s.end_stroke()
    assert s.save()
    assert load_annotation(dataset[1], "a").review_status == "reviewed"


def test_eraser_with_nothing_selected_spares_hidden_parts(s):
    s.set_hidden("object", 4, True)
    s.set_tool("eraser")
    s.set_brush_radius(40)
    s.begin_stroke(20, 20)
    s.end_stroke()
    assert s.ann.part_area(2) == 0 and s.ann.part_area(5) == 100


def test_brush_never_steals_from_hidden_parts(s):
    s.select_at(10, 10)
    s.set_hidden("object", 4, True)
    s.set_tool("brush")
    s.set_brush_radius(40)
    s.begin_stroke(20, 20)
    s.end_stroke()
    assert s.ann.part_area(5) == 100 and s.ann.part_area(3) == 0  # visible neighbour still taken


def test_solo_protects_other_objects_from_strokes(s):
    s.select_at(10, 10)
    s.toggle_solo()
    s.set_tool("brush")
    s.set_brush_radius(40)
    s.begin_stroke(20, 20)
    s.end_stroke()
    assert s.ann.part_area(5) == 100 and s.ann.part_area(3) == 0
