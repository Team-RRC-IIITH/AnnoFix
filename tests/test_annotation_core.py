import numpy as np
import pytest

from annofix.annotation import MAX_PART_ID, Annotation, Part


def meta_wo_next_id(ann):
    d = ann.meta_dict()
    d.pop("next_id")
    return d


def test_rejects_non_uint16_maps():
    with pytest.raises(ValueError):
        Annotation(np.zeros((4, 4), np.int32))


def test_ids_are_shared_and_sequential():
    ann = Annotation.empty(4, 4)
    oid = ann.new_object("mug")
    a = ann.new_part(oid, "body")
    b = ann.new_part(oid)
    assert (oid, a, b, ann.next_id) == (1, 2, 3, 4)
    assert ann.part(a) == Part(2, "body", None)
    assert ann.object_of(b).id == oid
    assert ann.size == (4, 4)


def test_meta_dict_is_json_shaped():
    ann = Annotation.empty(3, 5, image="img.png")
    ann.new_part(ann.new_object())
    d = ann.meta_dict()
    assert d["schema_version"] == 1 and d["size"] == [3, 5] and d["image"] == "img.png"
    assert d["objects"] == [{"id": 1, "class": None, "parts": [{"id": 2, "name": None, "material": None}]}]
    assert d["review_status"] == "unreviewed" and d["next_id"] == 3


def test_label_setters_and_review_status():
    ann = Annotation.empty(4, 4)
    oid = ann.new_object()
    pid = ann.new_part(oid)
    ann.set_object_class(oid, "mug")
    ann.set_part_name(pid, "handle")
    ann.set_part_material(pid, "ceramic")
    assert ann.object(oid).cls == "mug" and ann.part(pid) == Part(pid, "handle", "ceramic")
    ann.set_part_name(pid, "")
    assert ann.part(pid).name is None
    ann.set_review_status("reviewed")
    assert ann.review_status == "reviewed"
    with pytest.raises(ValueError):
        ann.set_review_status("done")


def test_undo_redo_restore_exact_state():
    ann = Annotation.empty(6, 6)
    oid = ann.new_object("mug")
    pid = ann.new_part(oid, "body", "ceramic")
    states = []

    def snap():
        states.append((ann.label_map.copy(), meta_wo_next_id(ann)))

    snap()
    with ann.transaction("paint"):
        ann.label_map[1:4, 1:4] = pid
    snap()
    ann.set_part_material(pid, "glass")
    snap()
    ann.set_object_class(oid, "cup")
    snap()
    for i in (2, 1, 0):
        assert ann.undo() is not None
        m, d = states[i]
        assert np.array_equal(ann.label_map, m) and meta_wo_next_id(ann) == d
    for i in (1, 2, 3):
        assert ann.redo() is not None
        m, d = states[i]
        assert np.array_equal(ann.label_map, m) and meta_wo_next_id(ann) == d


def test_nested_transactions_are_one_undo_step():
    ann = Annotation.empty(4, 4)
    pid = ann.new_part(ann.new_object())
    with ann.transaction("stroke"):
        with ann.transaction("dab"):
            ann.label_map[0, 0] = pid
        ann.set_part_name(pid, "x")
        ann.label_map[1, 1] = pid
    assert ann.undo() == "stroke"
    assert not ann.label_map.any() and ann.part(pid).name is None


def test_exception_rolls_back_and_records_nothing():
    ann = Annotation.empty(4, 4)
    pid = ann.new_part(ann.new_object())
    with pytest.raises(RuntimeError):
        with ann.transaction("bad"):
            ann.label_map[0, 0] = pid
            ann.set_part_material(pid, "metal")
            raise RuntimeError("boom")
    assert not ann.label_map.any() and ann.part(pid).material is None
    assert ann.undo() == "new part"  # the failed transaction left no undo entry


def test_new_edit_clears_redo():
    ann = Annotation.empty(4, 4)
    pid = ann.new_part(ann.new_object())
    ann.set_part_name(pid, "a")
    ann.undo()
    assert ann.can_redo()
    ann.set_part_name(pid, "b")
    assert not ann.can_redo()


def test_noop_edit_is_not_recorded():
    ann = Annotation.empty(4, 4)
    pid = ann.new_part(ann.new_object())
    ann.set_part_name(pid, "a")
    ann.set_part_name(pid, "a")
    assert ann.undo() == "set name"
    assert ann.part(pid).name is None


def test_ids_never_reused_after_undo():
    ann = Annotation.empty(4, 4)
    oid = ann.new_object()
    first = ann.new_part(oid)
    ann.undo()
    assert ann.new_part(oid) == first + 1


def test_dirty_tracking():
    ann = Annotation.empty(4, 4)
    assert not ann.dirty
    ann.new_object()
    assert ann.dirty
    ann.mark_saved()
    assert not ann.dirty
    ann.undo()
    assert ann.dirty
    ann.redo()
    assert not ann.dirty


def test_history_is_capped_by_bytes():
    ann = Annotation.empty(100, 100, history_bytes=50_000)
    pid = ann.new_part(ann.new_object())
    for v in (pid, 0, pid):  # each full-map change is a 40 kB entry
        with ann.transaction("fill"):
            ann.label_map[:] = v
    undos = 0
    while ann.undo():
        undos += 1
    assert undos == 1


def test_undo_inside_transaction_rejected():
    ann = Annotation.empty(4, 4)
    ann.new_object()
    with pytest.raises(RuntimeError):
        with ann.transaction("t"):
            ann.undo()


def test_id_space_exhaustion_is_an_error():
    ann = Annotation.empty(4, 4, next_id=MAX_PART_ID + 1)
    with pytest.raises(OverflowError):
        ann.new_object()


def test_from_meta_rejects_bad_review_status():
    with pytest.raises(ValueError):
        Annotation.from_meta(np.zeros((2, 2), np.uint16), {"review_status": "done"})


def test_begin_commit_spans_calls_as_one_undo_step():
    ann = Annotation.empty(4, 4)
    pid = ann.new_part(ann.new_object())
    ann.begin("stroke")
    ann.label_map[0, 0] = pid      # mouse event 1
    ann.set_part_name(pid, "x")    # an op inside the open stroke
    ann.label_map[1, 1] = pid      # mouse event 2
    assert ann.in_transaction
    ann.commit()
    assert not ann.in_transaction
    assert ann.undo() == "stroke"
    assert not ann.label_map.any() and ann.part(pid).name is None


def test_rollback_abandons_the_open_stroke():
    ann = Annotation.empty(4, 4)
    pid = ann.new_part(ann.new_object())
    ann.begin("stroke")
    ann.label_map[0, 0] = pid
    ann.set_part_material(pid, "metal")
    ann.rollback()
    assert not ann.in_transaction and not ann.label_map.any() and ann.part(pid).material is None
    assert ann.undo() == "new part"


def test_commit_and_rollback_without_begin_are_errors():
    ann = Annotation.empty(2, 2)
    with pytest.raises(RuntimeError):
        ann.commit()
    with pytest.raises(RuntimeError):
        ann.rollback()


def test_failing_op_inside_open_stroke_leaves_it_open_for_the_caller():
    ann = Annotation.empty(4, 4)
    pid = ann.new_part(ann.new_object())
    ann.begin("stroke")
    ann.label_map[0, 0] = pid
    with pytest.raises(KeyError):
        ann.set_part_name(999, "x")
    assert ann.in_transaction and ann.label_map[0, 0] == pid
    ann.rollback()
    assert not ann.label_map.any()


def test_label_only_edits_count_against_the_history_cap():
    ann = Annotation.empty(4, 4, history_bytes=2_000)
    pid = ann.new_part(ann.new_object())
    for i in range(100):
        ann.set_part_name(pid, f"name{i}")
    undos = 0
    while ann.undo():
        undos += 1
    assert 1 <= undos < 100


def test_attrs_ride_along_in_meta_and_survive_undo():
    ann = Annotation.empty(4, 4, attrs={"cvat_sample": "s1"})
    oid = ann.new_object("mug")
    pid = ann.new_part(oid, "handle", "metal")
    ann.object(oid).attrs.update({"uid": "o0", "object_id": 0})
    ann.part(pid).attrs.update({"uid": "p0_1", "role": "part", "confidence_pct": 40})
    d = ann.meta_dict()
    assert d["attrs"] == {"cvat_sample": "s1"}
    assert d["objects"][0]["attrs"] == {"uid": "o0", "object_id": 0}
    assert d["objects"][0]["parts"][0]["attrs"]["uid"] == "p0_1"
    ann.set_part_material(pid, "plastic")
    ann.undo()
    assert ann.part(pid).attrs["uid"] == "p0_1" and ann.object(oid).attrs["uid"] == "o0"


def test_attrs_are_omitted_when_empty():
    ann = Annotation.empty(2, 2)
    ann.new_part(ann.new_object())
    d = ann.meta_dict()
    assert "attrs" not in d and "attrs" not in d["objects"][0] and "attrs" not in d["objects"][0]["parts"][0]


def test_body_part_needs_no_name_to_validate():
    ann = Annotation.empty(4, 4)
    oid = ann.new_object("mug")
    pid = ann.new_part(oid, None, "plastic")
    ann.part(pid).attrs["role"] = "body"
    ann.label_map[0, 0] = pid
    assert ann.validate() == []
