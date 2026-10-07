"""Pipeline masks in CVAT-style COCO <-> AnnoFix (annofix/cvat.py)."""
import json
import shutil
from pathlib import Path

import numpy as np
import pytest
import yaml
from helpers import LABELS, make_cvat_sample, rect_poly

from annofix import cvat
from annofix.io import load_annotation, save_annotation

REAL = Path(__file__).resolve().parents[1] / "Test"


@pytest.fixture
def src(tmp_path):
    root = tmp_path / "src"
    make_cvat_sample(root, "s1")
    return root


def imported(src, tmp_path):
    work = tmp_path / "work"
    cvat.import_dataset(src, work)
    return work, load_annotation(work / "annotations", "s1")


def coco_by_uid(root, sample="s1"):
    coco = json.loads((root / sample / "annotations" / "instances_default.json").read_text())
    return coco, {a["attributes"]["uid"]: a for a in coco["annotations"]}


def test_polygons_round_trip_exactly():
    mask = np.zeros((30, 40), bool)
    mask[5:12, 3:20] = True
    mask[20:22, 30:31] = True  # a 2-pixel sliver
    mask[25, 5] = True         # a single pixel
    assert np.array_equal(cvat.rasterize(cvat.polygons(mask), 30, 40), mask)


def test_find_samples(src):
    make_cvat_sample(src, "s2")
    found = cvat.find_samples(src)
    assert [f.stem for f in found] == ["s1", "s2"] and found[0].image_path.exists()


def test_import_builds_objects_parts_and_a_body(src, tmp_path):
    work, ann = imported(src, tmp_path)
    mug = next(o for o in ann.objects if o.cls == "mug")
    body = next(p for p in mug.parts if p.attrs["role"] == "body")
    handle = next(p for p in mug.parts if p.attrs["role"] == "part")
    assert (handle.name, handle.material, handle.attrs["uid"], handle.attrs["confidence_pct"]) == \
        ("handle", "metal", "p0_1", 40)
    assert (body.material, ann.part_area(body.id), ann.part_area(handle.id)) == ("plastic", 350, 50)
    assert (mug.attrs["uid"], mug.attrs["object_id"], mug.attrs["object"]) == ("o0", 0, "sam3_000")
    box = next(o for o in ann.objects if o.cls == "box")
    assert [p.attrs["role"] for p in box.parts] == ["body"] and ann.part_area(box.parts[0].id) == 400
    assert ann.attrs["cvat_sample"] == "s1" and ann.source == "pipeline"
    assert (work / "images" / "s1.png").exists()
    labels = yaml.safe_load((work / "labels.yaml").read_text())
    assert labels["materials"][:3] == ["fabric", "foliage", "glass"] and len(labels["materials"]) == 10
    assert labels["objects"]["mug"] == ["handle"] and labels["objects"]["box"] == []


def test_overlapping_objects_smaller_one_wins(tmp_path):
    root = tmp_path / "src"
    make_cvat_sample(root, "s1", objects=[
        dict(uid="o0", kind="object", object_id=0, object_name="big", part_name="", material="paper",
             conf=50, poly=rect_poly(0, 0, 29, 29)),
        dict(uid="o1", kind="object", object_id=1, object_name="small", part_name="", material="metal",
             conf=50, poly=rect_poly(25, 25, 34, 34)),
    ])
    _, ann = imported(root, tmp_path)
    small = next(o for o in ann.objects if o.cls == "small")
    assert ann.part_at(27, 27) == small.parts[0].id


def test_import_never_overwrites_existing_work(src, tmp_path):
    work, ann = imported(src, tmp_path)
    handle = next(p for o in ann.objects for p in o.parts if p.name == "handle")
    ann.set_part_material(handle.id, "plastic")
    save_annotation(ann, work / "annotations", "s1")
    report = cvat.import_dataset(src, work)
    assert report.skipped == ["s1"] and report.imported == []
    assert load_annotation(work / "annotations", "s1").part(handle.id).material == "plastic"


def test_export_without_edits_reproduces_the_source(src, tmp_path):
    _, ann = imported(src, tmp_path)
    out = tmp_path / "out"
    cvat.export_image(ann, cvat.find_samples(src)[0], out)
    src_coco, src_by_uid = coco_by_uid(src)
    coco, by_uid = coco_by_uid(out)
    assert set(by_uid) == set(src_by_uid) == {"o0", "p0_1", "o1"}
    assert coco["categories"] == src_coco["categories"] and coco["images"] == src_coco["images"]
    for uid, a in src_by_uid.items():
        b = by_uid[uid]
        assert b["category_id"] == a["category_id"]
        for key in ("kind", "object_id", "object_name", "part_name", "confidence_pct"):
            assert b["attributes"][key] == a["attributes"][key]
        source_mask = cvat.rasterize(a["segmentation"], 40, 60)
        assert np.array_equal(cvat.rasterize(b["segmentation"], 40, 60), source_mask)
        assert b["area"] == int(source_mask.sum())
    manifest = json.loads((out / "s1" / "export_manifest.json").read_text())
    assert set(manifest["annotations"]) == {"o0", "p0_1", "o1"}
    assert manifest["annotations"]["o0"]["object"] == "sam3_000"
    assert (out / "s1" / "cvat_labels.json").exists() and (out / "s1" / "images" / "s1.png").exists()


def test_export_reflects_edits_new_parts_and_review(src, tmp_path):
    _, ann = imported(src, tmp_path)
    mug = next(o for o in ann.objects if o.cls == "mug")
    handle = next(p for p in mug.parts if p.name == "handle")
    ann.set_part_material(handle.id, "plastic")
    lid = ann.new_part(mug.id, "lid", "metal")
    m = np.zeros((40, 60), bool)
    m[5:7, 5:10] = True
    ann.assign(m, lid)
    cup = ann.new_part(ann.new_object("cup"), None, "glass")
    m2 = np.zeros((40, 60), bool)
    m2[32:38, 2:8] = True
    ann.assign(m2, cup)
    ann.set_review_status("reviewed")
    out = tmp_path / "out"
    cvat.export_image(ann, cvat.find_samples(src)[0], out)
    _, by_uid = coco_by_uid(out)
    cup_oid = next(o.id for o in ann.objects if o.cls == "cup")
    new_object_id = 2 + cup_oid  # after the source's ids, from the object's own never-reused id
    assert by_uid["p0_1"]["category_id"] == 7                               # now plastic
    assert by_uid[f"p0_{2 + lid}"]["attributes"]["part_name"] == "lid"      # a new part: a new uid
    assert by_uid[f"o{new_object_id}"]["attributes"]["object_name"] == "cup"
    assert by_uid[f"o{new_object_id}"]["attributes"]["object_id"] == new_object_id
    assert by_uid[f"o{new_object_id}"]["attributes"]["confidence_pct"] == 100
    assert by_uid[f"o{new_object_id}"]["category_id"] == 3                  # glass, from its only part
    assert all(a["attributes"]["reviewed"] for a in by_uid.values())


def test_object_mask_is_the_union_of_its_pieces(src, tmp_path):
    _, ann = imported(src, tmp_path)
    mug = next(o for o in ann.objects if o.cls == "mug")
    body = next(p for p in mug.parts if p.attrs["role"] == "body")
    m = np.zeros((40, 60), bool)
    m[5:10, 5:25] = True
    ann.erase(m, body.id)  # trim the top of the mug
    out = tmp_path / "out"
    cvat.export_image(ann, cvat.find_samples(src)[0], out)
    _, by_uid = coco_by_uid(out)
    mask = cvat.rasterize(by_uid["o0"]["segmentation"], 40, 60)
    assert mask.sum() == 300 and not mask[6, 10] and mask[15, 22]


def test_cli_export_only(src, tmp_path):
    out = tmp_path / "out"
    assert cvat.main([str(src), "--out", str(out), "--export-only"]) == 0
    assert (out / "s1" / "annotations" / "instances_default.json").exists()
    assert (out / ".annofix" / "annotations" / "s1.json").exists()


def test_cli_rejects_a_missing_source(tmp_path, capsys):
    assert cvat.main([str(tmp_path / "nope"), "--out", str(tmp_path / "out"), "--export-only"]) == 1
    assert "annofix" in capsys.readouterr().err


def test_saving_in_the_editor_exports(src, tmp_path):
    from annofix.config import load_labels
    from annofix.session import Session

    out = tmp_path / "out"
    work = out / ".annofix"
    cvat.import_dataset(src, work)
    s = Session(work / "images", work / "annotations", load_labels(work / "labels.yaml"))
    cvat.attach_auto_export(s, src, out)
    s.open(0)
    handle = next(p for o in s.ann.objects for p in o.parts if p.name == "handle")
    s.select_part(handle.id)
    s.set_material("plastic")
    assert s.save()
    _, by_uid = coco_by_uid(out)
    assert by_uid["p0_1"]["category_id"] == 7


@pytest.mark.skipif(not REAL.exists(), reason="needs the pipeline sample in Test/")
def test_real_pipeline_sample_round_trip(tmp_path):
    src = tmp_path / "src"
    shutil.copytree(REAL, src)
    work = tmp_path / "work"
    cvat.import_dataset(src, work)
    sample = cvat.find_samples(src)[0]
    ann = load_annotation(work / "annotations", sample.stem)
    roles = [p.attrs.get("role") for o in ann.objects for p in o.parts]
    assert len(ann.objects) == 16 and roles.count("part") == 14
    out = tmp_path / "out"
    cvat.export_image(ann, sample, out)
    _, src_by_uid = coco_by_uid(src, sample.sample_dir.name)
    _, by_uid = coco_by_uid(out, sample.sample_dir.name)
    assert set(by_uid) == set(src_by_uid)
    worst = 1.0
    for uid, a in src_by_uid.items():
        b = by_uid[uid]
        assert b["category_id"] == a["category_id"]
        assert b["attributes"]["object_name"] == a["attributes"]["object_name"]
        ma, mb = cvat.rasterize(a["segmentation"], 1080, 1920), cvat.rasterize(b["segmentation"], 1080, 1920)
        worst = min(worst, (ma & mb).sum() / (ma | mb).sum())
    assert worst >= 0.97


def edit_source(src, fn, sample="s1"):
    """Change the source COCO JSON of a test sample in place (before import)."""
    path = src / sample / "annotations" / "instances_default.json"
    coco = json.loads(path.read_text())
    fn(coco)
    path.write_text(json.dumps(coco))


def export(ann, src, out, notes=None):
    cvat.export_image(ann, cvat.find_samples(src)[0], out, notes)
    coco, by_uid = coco_by_uid(out)
    manifest = json.loads((out / "s1" / "export_manifest.json").read_text())
    return coco, by_uid, manifest


def test_new_uids_are_unique_stable_and_never_reused(src, tmp_path):
    _, ann = imported(src, tmp_path)
    mug = next(o for o in ann.objects if o.cls == "mug")
    handle = next(p for p in mug.parts if p.name == "handle")
    split = ann.promote_part(handle.id)           # p0_1 now belongs to a new object...
    ann.set_object_class(split, "handle thing")
    rim = ann.new_part(mug.id, "rim", "metal")    # ...and the mug gets a new part
    m = np.zeros((40, 60), bool)
    m[5:7, 5:10] = True
    ann.assign(m, rim)
    cup = ann.new_part(ann.new_object("cup"), "cup", "glass")
    m2 = np.zeros((40, 60), bool)
    m2[32:38, 2:8] = True
    ann.assign(m2, cup)
    out = tmp_path / "out"
    coco, by_uid, manifest = export(ann, src, out)
    uids = [a["attributes"]["uid"] for a in coco["annotations"]]
    assert len(uids) == len(set(uids)) == len(manifest["annotations"])
    cup_uid = next(u for u, a in by_uid.items() if a["attributes"]["object_name"] == "cup"
                   and a["attributes"]["kind"] == "object")
    box = next(o for o in ann.objects if o.cls == "box")
    ann.delete_object(box.id)                     # o1 is gone: its number must not be handed out
    coco2, by_uid2, _ = export(ann, src, out)
    assert set(by_uid2) == set(by_uid) - {"o1"} and cup_uid != "o1"
    assert by_uid2[cup_uid]["attributes"]["object_name"] == "cup"


def test_import_marks_reviewed_only_when_the_labels_pass(src, tmp_path):
    def all_reviewed_but_one_part_unnamed(coco):
        for a in coco["annotations"]:
            a["attributes"]["reviewed"] = True
        coco["annotations"][1]["attributes"]["part_name"] = ""

    edit_source(src, all_reviewed_but_one_part_unnamed)
    work = tmp_path / "work"
    report = cvat.import_dataset(src, work)
    assert load_annotation(work / "annotations", "s1").review_status == "unreviewed"
    assert any("reviewed" in w for w in report.warnings)
    src2 = tmp_path / "src2"
    make_cvat_sample(src2, "s1")
    edit_source(src2, lambda coco: [a["attributes"].update(reviewed=True) for a in coco["annotations"]])
    cvat.import_dataset(src2, tmp_path / "work2")
    assert load_annotation(tmp_path / "work2" / "annotations", "s1").review_status == "reviewed"


def test_object_category_follows_its_parts_once_the_body_is_gone(src, tmp_path):
    _, ann = imported(src, tmp_path)
    mug = next(o for o in ann.objects if o.cls == "mug")
    body = next(p for p in mug.parts if p.attrs["role"] == "body")
    handle = next(p for p in mug.parts if p.name == "handle")
    ann.merge_parts(handle.id, body.id)     # the whole mug is now its "handle" part...
    ann.set_part_material(handle.id, "wood")  # ...made of wood
    _, by_uid, _ = export(ann, src, tmp_path / "out")
    assert by_uid["o0"]["category_id"] == 10  # wood, not the imported plastic


def test_a_renamed_body_is_exported_as_a_part(src, tmp_path):
    _, ann = imported(src, tmp_path)
    box = next(o for o in ann.objects if o.cls == "box")
    ann.set_part_name(box.parts[0].id, "lid")
    coco, by_uid, _ = export(ann, src, tmp_path / "out")
    lid = [a for a in coco["annotations"] if a["attributes"]["part_name"] == "lid"]
    assert len(lid) == 1 and lid[0]["attributes"]["kind"] == "part" and lid[0]["attributes"]["uid"] != "o1"
    assert by_uid["o1"]["attributes"]["kind"] == "object"


def test_missing_materials_are_left_out_with_a_note(src, tmp_path):
    _, ann = imported(src, tmp_path)
    mug = next(o for o in ann.objects if o.cls == "mug")
    handle = next(p for p in mug.parts if p.name == "handle")
    ann.set_part_material(handle.id, None)
    new = ann.new_part(ann.new_object("cup"))
    m = np.zeros((40, 60), bool)
    m[32:38, 2:8] = True
    ann.assign(m, new)
    notes = []
    coco, by_uid, _ = export(ann, src, tmp_path / "out", notes)
    assert "p0_1" not in by_uid and set(by_uid) == {"o0", "o1"}  # nothing exported as "fabric"
    assert all(a["category_id"] != 1 for a in coco["annotations"])
    assert len(notes) == 2 and any("handle" in n for n in notes) and any("cup" in n for n in notes)


def test_annotations_lost_on_import_are_reported(tmp_path, capsys):
    root = tmp_path / "src"
    make_cvat_sample(root, "s1", objects=[
        dict(uid="o0", kind="object", object_id=0, object_name="mug", part_name="", material="plastic",
             conf=90, poly=rect_poly(5, 5, 24, 24)),
        dict(uid="p0_1", kind="part", object_id=0, object_name="mug", part_name="a", material="metal",
             conf=40, poly=rect_poly(20, 10, 24, 19)),
        dict(uid="p0_2", kind="part", object_id=0, object_name="mug", part_name="b", material="metal",
             conf=40, poly=rect_poly(20, 10, 24, 19)),  # the same pixels: one of the two loses
    ])
    report = cvat.import_dataset(root, tmp_path / "work")
    assert any("p0_" in w and "s1" in w for w in report.warnings)
    assert cvat.main([str(root), "--out", str(tmp_path / "out"), "--export-only"]) == 0
    assert "p0_" in capsys.readouterr().err


def test_out_inside_the_source_is_refused(src, capsys):
    assert cvat.main([str(src), "--out", str(src / "fixed"), "--export-only"]) == 1
    assert not (src / "fixed").exists() and "annofix" in capsys.readouterr().err


def test_pipeline_confidences_and_reviewed_flags_are_kept(src, tmp_path):
    manifest_path = src / "s1" / "export_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["annotations"]["o0"]["confidence"] = 0.9227409958839417
    manifest_path.write_text(json.dumps(manifest))
    edit_source(src, lambda coco: coco["annotations"][1]["attributes"].update(reviewed=True))  # p0_1
    _, ann = imported(src, tmp_path)
    _, by_uid, manifest = export(ann, src, tmp_path / "out")
    assert manifest["annotations"]["o0"]["confidence"] == 0.9227409958839417
    assert by_uid["p0_1"]["attributes"]["reviewed"] and not by_uid["o1"]["attributes"]["reviewed"]


def test_startup_export_repairs_stale_outputs(src, tmp_path):
    import os

    out = tmp_path / "out"
    work = out / ".annofix"
    cvat.import_dataset(src, work)
    assert cvat.export_dataset(src, work, out, only_stale=True) == ["s1"]   # missing
    assert cvat.export_dataset(src, work, out, only_stale=True) == []       # current
    ann = load_annotation(work / "annotations", "s1")
    handle = next(p for o in ann.objects for p in o.parts if p.name == "handle")
    ann.set_part_material(handle.id, "plastic")
    save_annotation(ann, work / "annotations", "s1")                         # an export that failed
    out_json = out / "s1" / "annotations" / "instances_default.json"
    os.utime(out_json, ns=(out_json.stat().st_mtime_ns - 10**9,) * 2)
    assert cvat.export_dataset(src, work, out, only_stale=True) == ["s1"]   # stale
    assert coco_by_uid(out)[1]["p0_1"]["category_id"] == 7


def test_multi_image_manifest_drops_deleted_entries(tmp_path):
    root = tmp_path / "src"
    d = make_cvat_sample(root, "s1")
    shutil.copy(d / "images" / "s1.png", d / "images" / "s2.png")

    def add_second_image(coco):
        coco["images"].append({"id": 2, "file_name": "s2.png", "width": 60, "height": 40})
        coco["annotations"].append({**coco["annotations"][2], "id": 4, "image_id": 2,
                                    "attributes": {**coco["annotations"][2]["attributes"], "uid": "o7",
                                                   "object_id": 7}})

    edit_source(root, add_second_image)
    manifest_path = d / "export_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["annotations"]["o7"] = {**manifest["annotations"]["o1"], "object_id": 7}
    manifest_path.write_text(json.dumps(manifest))
    work = tmp_path / "work"
    cvat.import_dataset(root, work)
    first = next(s for s in cvat.find_samples(root) if s.entry["id"] == 1)
    ann = load_annotation(work / "annotations", first.stem)
    ann.delete_object(next(o.id for o in ann.objects if o.cls == "box"))
    out = tmp_path / "out"
    cvat.export_image(ann, first, out)
    entries = json.loads((out / "s1" / "export_manifest.json").read_text())["annotations"]
    assert "o1" not in entries and {"o0", "p0_1", "o7"} <= set(entries)


def test_body_mode_new_objects_are_a_body_not_an_unnamed_part(src, tmp_path):
    from annofix.config import load_labels
    from annofix.session import Session

    out = tmp_path / "out"
    work = out / ".annofix"
    cvat.import_dataset(src, work)
    s = Session(work / "images", work / "annotations", load_labels(work / "labels.yaml"))
    s.body_parts = True
    s.open(0)
    s.new_object()                         # brush ready, no SAM
    new = s.primary
    assert s.ann.part(new).name == "body" and s.ann.part(new).attrs["role"] == "body"
    m = np.zeros((40, 60), bool)
    m[32:38, 2:8] = True
    s.ann.assign(m, new)
    s.set_material("glass")
    s.set_object_class("cup")
    assert not [p for p in s.ann.validate() if f"part {new}" in p]  # a body needs no name
    handle = next(p for o in s.ann.objects for p in o.parts if p.name == "handle")
    s.promote_part(handle.id)              # split off into an object of its own: its body
    assert s.ann.part(handle.id).attrs["role"] == "body"
    s.undo()
    assert s.ann.part(handle.id).attrs["role"] == "part"
    cvat.export_image(s.ann, cvat.find_samples(src)[0], out)
    _, by_uid = coco_by_uid(out)
    cup = [a for a in by_uid.values() if a["attributes"]["object_name"] == "cup"]
    assert [a["attributes"]["kind"] for a in cup] == ["object"]  # no duplicate part with the same mask
