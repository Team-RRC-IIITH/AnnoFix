import numpy as np
import pytest

from annofix.annotation import Annotation


def rect(y0, y1, x0, x1, shape=(10, 10)):
    m = np.zeros(shape, bool)
    m[y0:y1, x0:x1] = True
    return m


@pytest.fixture
def mug():
    ann = Annotation.empty(10, 10)
    oid = ann.new_object("mug")
    a = ann.new_part(oid, "body", "ceramic")
    b = ann.new_part(oid, "handle", "ceramic")
    ann.assign(rect(0, 5, 0, 5), a)
    ann.assign(rect(5, 10, 5, 10), b)
    return ann, oid, a, b


def test_assign_steals_from_other_parts_by_default(mug):
    ann, _, a, b = mug
    assert ann.assign(rect(3, 7, 3, 7), a) == 12
    assert ann.part_area(a) == 37 and ann.part_area(b) == 21
    assert ann.part_at(6, 6) == a


def test_assign_protect_only_fills_background(mug):
    ann, _, a, b = mug
    assert ann.assign(rect(3, 7, 3, 7), a, protect_others=True) == 8
    assert ann.part_area(a) == 33 and ann.part_area(b) == 25
    assert ann.part_at(6, 6) == b


def test_erase_only_touches_given_part(mug):
    ann, _, a, b = mug
    assert ann.erase(np.ones((10, 10), bool), a) == 25
    assert ann.part_area(a) == 0 and ann.part_area(b) == 25


def test_erase_all_parts_in_mask(mug):
    ann, _, a, b = mug
    assert ann.erase(rect(4, 6, 4, 6)) == 2
    assert ann.part_area(a) == 24 and ann.part_area(b) == 24


def test_replace_part_mask_moves_the_part(mug):
    ann, _, a, b = mug
    ann.replace_part_mask(a, rect(0, 2, 0, 10))
    assert ann.part_area(a) == 20 and ann.part_at(3, 3) is None and ann.part_area(b) == 25


def test_replace_part_mask_protect_keeps_neighbours(mug):
    ann, _, a, b = mug
    ann.replace_part_mask(a, np.ones((10, 10), bool), protect_others=True)
    assert ann.part_area(a) == 75 and ann.part_area(b) == 25


def test_pixel_op_is_one_undo_step(mug):
    ann, _, a, _ = mug
    before = ann.label_map.copy()
    ann.replace_part_mask(a, rect(0, 2, 0, 10))
    ann.undo()
    assert np.array_equal(ann.label_map, before)


def test_merge_parts(mug):
    ann, _, a, b = mug
    ann.merge_parts(a, b)
    assert ann.part_area(a) == 50 and ann.part_ids() == [a]


def test_merge_with_itself_rejected(mug):
    ann, _, a, _ = mug
    with pytest.raises(ValueError):
        ann.merge_parts(a, a)


def test_delete_part_clears_pixels(mug):
    ann, _, a, b = mug
    ann.delete_part(a)
    assert ann.part_area(a) == 0 and ann.part_ids() == [b]


def test_delete_object_clears_all_its_parts(mug):
    ann, oid, _, _ = mug
    ann.delete_object(oid)
    assert ann.objects == [] and not ann.label_map.any()


def test_move_and_promote(mug):
    ann, oid, a, b = mug
    other = ann.new_object("cup")
    ann.move_part(b, other)
    assert ann.object_of(b).id == other
    new = ann.promote_part(a)
    assert ann.object_of(a).id == new and ann.object(new).cls is None
    assert ann.object(oid).parts == []
    assert ann.part_area(a) == 25  # pixels untouched


def test_unknown_part_rejected(mug):
    ann, *_ = mug
    with pytest.raises(KeyError):
        ann.assign(rect(0, 1, 0, 1), 999)


def test_bad_mask_rejected(mug):
    ann, _, a, _ = mug
    with pytest.raises(ValueError):
        ann.assign(np.ones((10, 10), np.uint8), a)
    with pytest.raises(ValueError):
        ann.assign(np.ones((5, 5), bool), a)


def test_clean_part_drops_islands_and_fills_holes():
    ann = Annotation.empty(20, 20)
    a = ann.new_part(ann.new_object())
    m = rect(2, 12, 2, 12, (20, 20))
    m[6, 6] = False   # 1 px hole
    m[17, 17] = True  # 1 px island
    ann.assign(m, a)
    ann.clean_part(a, min_island_px=4, max_hole_px=10)
    assert ann.part_at(6, 6) == a and ann.part_at(17, 17) is None
    assert ann.part_area(a) == 100


def test_clean_part_leaves_enclosed_other_parts():
    ann = Annotation.empty(20, 20)
    oid = ann.new_object()
    a, b = ann.new_part(oid), ann.new_part(oid)
    ann.assign(rect(2, 12, 2, 12, (20, 20)), a)
    ann.assign(rect(6, 7, 6, 7, (20, 20)), b)
    ann.clean_part(a, min_island_px=4, max_hole_px=10)
    assert ann.part_at(6, 6) == b


def test_clean_part_keeps_large_holes():
    ann = Annotation.empty(20, 20)
    a = ann.new_part(ann.new_object())
    m = rect(2, 18, 2, 18, (20, 20))
    m[5:15, 5:15] = False  # 100 px hole
    ann.assign(m, a)
    ann.clean_part(a, min_island_px=4, max_hole_px=10)
    assert ann.part_at(10, 10) is None


def test_validate_lists_missing_labels():
    ann = Annotation.empty(10, 10)
    oid = ann.new_object()
    pid = ann.new_part(oid)
    problems = ann.validate()
    assert f"object {oid}: no class" in problems
    assert f"part {pid} (object {oid}): no name" in problems
    assert f"part {pid} (object {oid}): no material" in problems
    assert f"part {pid} (object {oid}): empty mask" in problems


def test_validate_ok_when_complete(mug):
    ann, *_ = mug
    assert ann.validate() == []


def test_validate_flags_object_without_parts():
    ann = Annotation.empty(4, 4)
    oid = ann.new_object("mug")
    assert ann.validate() == [f"object {oid}: no parts"]
