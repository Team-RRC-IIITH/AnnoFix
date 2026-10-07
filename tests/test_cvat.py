"""Pipeline masks in CVAT-style COCO <-> AnnoFix (annofix/cvat.py)."""
import json
import shutil
from pathlib import Path

import numpy as np
import pytest
import yaml
from helpers import make_cvat_sample, rect_poly

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
    assert by_uid["p0_1"]["category_id"] == 7                   # now plastic
    assert by_uid["p0_2"]["attributes"]["part_name"] == "lid"   # a new part gets the next uid
    assert by_uid["o2"]["attributes"]["object_name"] == "cup"   # a new object gets the next id
    assert by_uid["o2"]["attributes"]["object_id"] == 2 and by_uid["o2"]["attributes"]["confidence_pct"] == 100
    assert by_uid["o2"]["category_id"] == 3                     # glass, from its only part
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
