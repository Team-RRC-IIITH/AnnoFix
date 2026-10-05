import json

import cv2
import numpy as np
import pytest

import annofix.io as aio
from annofix.annotation import AnnObject, Annotation, Part
from annofix.io import (annotation_paths, has_annotation, list_images, load, load_annotation,
                        load_image, save_annotation)


def big_id_ann():
    lm = np.zeros((4, 6), np.uint16)
    lm[0, :2], lm[1, :2], lm[2, :2], lm[3, :2] = 255, 256, 32768, 65535
    objs = [AnnObject(1, "mug", [Part(255, "body", "ceramic"), Part(256)]),
            AnnObject(2, None, [Part(32768), Part(65535)])]
    return Annotation(lm, objs, next_id=65536, image="img.png")


def test_roundtrip_preserves_uint16_ids_and_meta(tmp_path):
    ann = big_id_ann()
    save_annotation(ann, tmp_path, "img")
    back = load_annotation(tmp_path, "img")
    assert back.label_map.dtype == np.uint16
    assert np.array_equal(back.label_map, ann.label_map)
    assert back.meta_dict() == ann.meta_dict()
    assert back.warnings == []


def test_save_writes_both_files_and_no_temp_files(tmp_path):
    save_annotation(big_id_ann(), tmp_path, "img")
    png, js = annotation_paths(tmp_path, "img")
    assert (png.name, js.name) == ("img.parts.png", "img.json")
    assert sorted(p.name for p in tmp_path.iterdir()) == ["img.json", "img.parts.png"]
    d = json.loads(js.read_text())
    assert d["schema_version"] == 1 and d["objects"][1]["class"] is None
    assert has_annotation(tmp_path, "img") and not has_annotation(tmp_path, "other")


def test_save_marks_clean(tmp_path):
    ann = Annotation.empty(4, 4)
    ann.new_object()
    assert ann.dirty
    save_annotation(ann, tmp_path, "x")
    assert not ann.dirty


def test_unserialisable_meta_leaves_existing_files_untouched(tmp_path):
    ann = big_id_ann()
    save_annotation(ann, tmp_path, "img")
    png, _ = annotation_paths(tmp_path, "img")
    before = png.read_bytes()
    ann.label_map[:] = 0
    ann.annotator = object()
    with pytest.raises(TypeError):
        save_annotation(ann, tmp_path, "img")
    assert png.read_bytes() == before


def test_failed_json_write_still_loads(tmp_path, monkeypatch):
    ann = Annotation.empty(4, 4)
    oid = ann.new_object("mug")
    save_annotation(ann, tmp_path, "img")
    pid = ann.new_part(oid)
    m = np.zeros((4, 4), bool)
    m[0, 0] = True
    ann.assign(m, pid)
    real = aio._atomic_write

    def flaky(path, data):
        if path.suffix == ".json":
            raise OSError("disk full")
        real(path, data)

    monkeypatch.setattr(aio, "_atomic_write", flaky)
    with pytest.raises(OSError):
        save_annotation(ann, tmp_path, "img")
    assert ann.dirty
    monkeypatch.undo()
    back = load_annotation(tmp_path, "img")  # new PNG + old JSON
    assert back.part_at(0, 0) == pid
    assert any("orphan" in w for w in back.warnings)


def test_size_mismatch_with_image_is_an_error(tmp_path):
    save_annotation(Annotation.empty(4, 6), tmp_path, "img")
    with pytest.raises(ValueError, match="image is"):
        load_annotation(tmp_path, "img", expected_size=(6, 4))


def test_json_size_must_match_png(tmp_path):
    save_annotation(Annotation.empty(4, 6), tmp_path, "img")
    _, js = annotation_paths(tmp_path, "img")
    d = json.loads(js.read_text())
    d["size"] = [5, 6]
    js.write_text(json.dumps(d))
    with pytest.raises(ValueError, match="JSON says"):
        load_annotation(tmp_path, "img")


def test_unknown_schema_version_rejected(tmp_path):
    save_annotation(Annotation.empty(2, 2), tmp_path, "img")
    _, js = annotation_paths(tmp_path, "img")
    d = json.loads(js.read_text())
    d["schema_version"] = 2
    js.write_text(json.dumps(d))
    with pytest.raises(ValueError, match="schema_version"):
        load_annotation(tmp_path, "img")


def test_orphans_and_empty_parts_reconciled(tmp_path):
    lm = np.zeros((4, 4), np.uint16)
    lm[0, 0], lm[3, 3] = 2, 7
    save_annotation(Annotation(lm, [AnnObject(1, "mug", [Part(2), Part(3)])], next_id=4), tmp_path, "img")
    back = load_annotation(tmp_path, "img")
    assert back.object_of(7).cls is None and back.object_of(7).id == 8
    assert back.next_id == 9
    assert any("orphan" in w for w in back.warnings)
    assert any("part 3" in w and "no pixels" in w for w in back.warnings)
    assert not back.dirty


def test_duplicate_ids_rejected(tmp_path):
    save_annotation(Annotation(np.zeros((2, 2), np.uint16), [AnnObject(1, None, [Part(1)])], next_id=2),
                    tmp_path, "img")
    with pytest.raises(ValueError, match="duplicate"):
        load_annotation(tmp_path, "img")


def test_load_helper(tmp_path):
    save_annotation(big_id_ann(), tmp_path, "img")
    lm, meta = load(tmp_path, "img")
    assert lm.dtype == np.uint16 and meta["objects"][0]["parts"][0]["id"] == 255


def test_list_images_skips_depth_and_outputs(tmp_path):
    for name in ["cam0_000_color.png", "cam0_000_depth.png", "Depth-01.png", "notes.txt",
                 "x.parts.png", "mug.JPG"]:
        (tmp_path / name).write_bytes(b"")
    (tmp_path / "sub.png").mkdir()
    assert [p.name for p in list_images(tmp_path)] == ["cam0_000_color.png", "mug.JPG"]


def test_load_image_returns_rgb(tmp_path):
    bgr = np.zeros((2, 3, 3), np.uint8)
    bgr[..., 0] = 255  # pure blue, stored BGR
    cv2.imwrite(str(tmp_path / "c.png"), bgr)
    rgb = load_image(tmp_path / "c.png")
    assert rgb.shape == (2, 3, 3) and tuple(rgb[0, 0]) == (0, 0, 255)


def test_load_image_accepts_grey_rejects_depth(tmp_path):
    cv2.imwrite(str(tmp_path / "g.png"), np.full((2, 2), 9, np.uint8))
    assert load_image(tmp_path / "g.png").shape == (2, 2, 3)
    cv2.imwrite(str(tmp_path / "d.png"), np.full((2, 2), 900, np.uint16))
    with pytest.raises(ValueError, match="depth"):
        load_image(tmp_path / "d.png")
