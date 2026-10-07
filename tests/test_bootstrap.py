import subprocess
import sys

import cv2
import numpy as np
import pytest

from annofix.annotation import Annotation
from annofix.bootstrap import (bootstrap_image, cache_path, image_fingerprint, load_cache, main,
                               save_cache)
from annofix.io import annotation_paths, load_annotation, load_image, save_annotation
from annofix.overlaps import GroupingParams

H, W = 60, 80


def object_masks(image):
    m = np.zeros((1, H, W), bool)
    m[0, 10:30, 10:40] = True  # 600 px object
    return m, np.array([0.9], np.float32)


def must_not_run(image):
    raise AssertionError("generator called although a cache exists")


@pytest.fixture
def images(tmp_path):
    d = tmp_path / "images"
    d.mkdir()
    for name in ("a_color.png", "b_color.png"):
        cv2.imwrite(str(d / name), np.full((H, W, 3), 120, np.uint8))
    cv2.imwrite(str(d / "a_depth.png"), np.full((H, W), 900, np.uint16))
    return d


def test_generates_annotation_and_cache(images, tmp_path):
    out = tmp_path / "ann"
    assert bootstrap_image(images / "a_color.png", out, object_masks, GroupingParams()) == "generated"
    ann = load_annotation(out, "a_color", expected_size=(H, W))
    assert [len(o.parts) for o in ann.objects] == [1]
    assert ann.part_area(ann.part_ids()[0]) == 600
    assert ann.source == "amg" and ann.image == "../images/a_color.png"
    assert cache_path(out, "a_color").exists()


def test_existing_annotation_is_never_overwritten_by_default(images, tmp_path):
    out = tmp_path / "ann"
    bootstrap_image(images / "a_color.png", out, object_masks, GroupingParams())
    assert bootstrap_image(images / "a_color.png", out, must_not_run, GroupingParams()) == "skipped"


def test_overwrite_reuses_cache_without_generator(images, tmp_path):
    out = tmp_path / "ann"
    bootstrap_image(images / "a_color.png", out, object_masks, GroupingParams())
    assert bootstrap_image(images / "a_color.png", out, must_not_run, GroupingParams(),
                           overwrite=True) == "cached"


def test_overwrite_never_touches_reviewed_work(images, tmp_path):
    out = tmp_path / "ann"
    bootstrap_image(images / "a_color.png", out, object_masks, GroupingParams())
    ann = load_annotation(out, "a_color")
    ann.set_review_status("reviewed")
    save_annotation(ann, out, "a_color")
    assert bootstrap_image(images / "a_color.png", out, object_masks, GroupingParams(),
                           overwrite=True) == "protected"
    assert load_annotation(out, "a_color").review_status == "reviewed"


def test_regroup_without_cache_reports_it(images, tmp_path):
    assert bootstrap_image(images / "a_color.png", tmp_path / "ann", None, GroupingParams()) == "no-cache"


def test_cache_roundtrip(tmp_path):
    masks = np.random.default_rng(1).random((3, 7, 13)) > 0.5
    save_cache(tmp_path / "c.npz", masks, np.array([0.1, 0.2, 0.3], np.float32), "abc123", "sam|pps=32")
    back, scores, image_hash, settings = load_cache(tmp_path / "c.npz")
    assert np.array_equal(back, masks) and scores.tolist() == pytest.approx([0.1, 0.2, 0.3])
    assert (image_hash, settings) == ("abc123", "sam|pps=32")


def test_cache_of_wrong_size_is_an_error(images, tmp_path):
    out = tmp_path / "ann"
    save_cache(cache_path(out, "a_color"), np.zeros((1, H + 1, W), bool), np.array([0.9]),
               image_fingerprint(load_image(images / "a_color.png")))
    with pytest.raises(ValueError, match="cached masks are"):
        bootstrap_image(images / "a_color.png", out, None, GroupingParams())


def test_main_regroup_skips_depth_and_counts(images, tmp_path, capsys):
    out = tmp_path / "ann"
    for stem in ("a_color", "b_color"):
        save_cache(cache_path(out, stem), *object_masks(None),
                   image_fingerprint(load_image(images / f"{stem}.png")))
    assert main(["--images", str(images), "--out", str(out), "--regroup"]) == 0
    text = capsys.readouterr().out
    assert "a_depth" not in text and "a_color.png: cached" in text and "b_color.png: cached" in text
    assert (out / "b_color.json").exists()


def test_main_returns_2_on_errors(images, tmp_path):
    out = tmp_path / "ann"
    save_cache(cache_path(out, "a_color"), np.zeros((1, H + 1, W), bool), np.array([0.9]))
    assert main(["--images", str(images), "--out", str(out), "--regroup"]) == 2


def test_core_modules_import_without_torch():
    code = ("import sys, annofix.bootstrap, annofix.io, annofix.overlaps, annofix.config, "
            "annofix.annotation; assert 'torch' not in sys.modules")
    subprocess.run([sys.executable, "-c", code], check=True)


def moved_object(image):
    m = np.zeros((1, H, W), bool)
    m[0, 30:50, 45:75] = True
    return m, np.array([0.9], np.float32)


def rerecord(images):
    """Same file name and size, different pixels: a re-extracted frame."""
    cv2.imwrite(str(images / "a_color.png"), np.full((H, W, 3), 30, np.uint8))


def test_regroup_refuses_cache_made_from_a_different_image(images, tmp_path):
    out = tmp_path / "ann"
    bootstrap_image(images / "a_color.png", out, object_masks, GroupingParams())
    rerecord(images)
    with pytest.raises(ValueError, match="different image"):
        bootstrap_image(images / "a_color.png", out, None, GroupingParams(), overwrite=True)


def test_stale_cache_is_regenerated_when_sam_is_available(images, tmp_path):
    out = tmp_path / "ann"
    bootstrap_image(images / "a_color.png", out, object_masks, GroupingParams())
    rerecord(images)
    assert bootstrap_image(images / "a_color.png", out, moved_object, GroupingParams(),
                           overwrite=True) == "generated"
    ann = load_annotation(out, "a_color")
    assert ann.part_at(40, 60) is not None and ann.part_at(20, 20) is None


def test_changed_sam_settings_regenerate_the_cache(images, tmp_path):
    out = tmp_path / "ann"
    a = images / "a_color.png"
    bootstrap_image(a, out, object_masks, GroupingParams(), settings="s1")
    assert bootstrap_image(a, out, object_masks, GroupingParams(), overwrite=True, settings="s1") == "cached"
    assert bootstrap_image(a, out, object_masks, GroupingParams(), overwrite=True, settings="s2") == "generated"


def test_overwrite_never_touches_in_progress_edits(images, tmp_path):
    out = tmp_path / "ann"
    bootstrap_image(images / "a_color.png", out, object_masks, GroupingParams())
    ann = load_annotation(out, "a_color")
    pid = ann.part_ids()[0]
    ann.set_part_material(pid, "ceramic")
    save_annotation(ann, out, "a_color")
    assert bootstrap_image(images / "a_color.png", out, object_masks, GroupingParams(),
                           overwrite=True) == "protected"
    assert load_annotation(out, "a_color").part(pid).material == "ceramic"


def test_overwrite_never_touches_flagged_work(images, tmp_path):
    out = tmp_path / "ann"
    bootstrap_image(images / "a_color.png", out, object_masks, GroupingParams())
    ann = load_annotation(out, "a_color")
    ann.set_review_status("flagged")
    save_annotation(ann, out, "a_color")
    assert bootstrap_image(images / "a_color.png", out, object_masks, GroupingParams(),
                           overwrite=True) == "protected"


def test_overwrite_never_touches_annotations_bootstrap_did_not_write(images, tmp_path):
    out = tmp_path / "ann"
    save_annotation(Annotation.empty(H, W, source="pipeline"), out, "a_color")
    assert bootstrap_image(images / "a_color.png", out, object_masks, GroupingParams(),
                           overwrite=True) == "protected"


def test_overwrite_leaves_a_corrupt_annotation_alone(images, tmp_path):
    out = tmp_path / "ann"
    bootstrap_image(images / "a_color.png", out, object_masks, GroupingParams())
    _, js = annotation_paths(out, "a_color")
    js.write_text("{ truncated")
    assert bootstrap_image(images / "a_color.png", out, object_masks, GroupingParams(),
                           overwrite=True) == "protected"
    assert js.read_text() == "{ truncated"


def test_truncated_cache_is_a_cache_miss_when_sam_is_available(images, tmp_path):
    out = tmp_path / "ann"
    cpath = cache_path(out, "a_color")
    cpath.parent.mkdir(parents=True)
    cpath.write_bytes(b"PK\x03\x04 truncated")
    assert bootstrap_image(images / "a_color.png", out, object_masks, GroupingParams()) == "generated"


def test_truncated_cache_in_regroup_is_a_per_image_error(images, tmp_path):
    out = tmp_path / "ann"
    cpath = cache_path(out, "a_color")
    cpath.parent.mkdir(parents=True)
    cpath.write_bytes(b"garbage")
    save_cache(cache_path(out, "b_color"), *object_masks(None),
               image_fingerprint(load_image(images / "b_color.png")))
    assert main(["--images", str(images), "--out", str(out), "--regroup"]) == 2
    assert (out / "b_color.json").exists()


def test_interrupted_cache_write_leaves_no_cache(tmp_path, monkeypatch):
    def interrupted(f, **arrays):
        f.write(b"partial")
        raise KeyboardInterrupt

    monkeypatch.setattr(np, "savez_compressed", interrupted)
    path = tmp_path / "c.npz"
    with pytest.raises(KeyboardInterrupt):
        save_cache(path, np.zeros((1, 4, 4), bool), np.array([0.9]))
    assert list(tmp_path.iterdir()) == []


def test_amg_settings_name_everything_that_changes_the_masks():
    from annofix.sam_backend import AMG_PRED_IOU, AMG_STABILITY, amg_settings

    s = amg_settings("base_plus", 32)
    assert all(str(v) in s for v in ("base_plus", 32, AMG_PRED_IOU, AMG_STABILITY))
    assert amg_settings("small", 32) != s and amg_settings("base_plus", 48) != s


def test_overwrite_rechecks_just_before_writing(images, tmp_path, monkeypatch):
    import annofix.bootstrap as bootstrap_module

    out = tmp_path / "ann"
    bootstrap_image(images / "a_color.png", out, object_masks, GroupingParams())
    real_load = bootstrap_module.load_image

    def editor_saves_meanwhile(path):
        ann = load_annotation(out, "a_color")
        ann.set_part_material(ann.part_ids()[0], "metal")
        save_annotation(ann, out, "a_color")
        return real_load(path)

    monkeypatch.setattr(bootstrap_module, "load_image", editor_saves_meanwhile)
    assert bootstrap_image(images / "a_color.png", out, object_masks, GroupingParams(), overwrite=True) == "protected"
    back = load_annotation(out, "a_color")
    assert back.part(back.part_ids()[0]).material == "metal"


def test_cli_errors_are_reported(images, tmp_path, capsys):
    assert main(["--images", str(tmp_path / "missing"), "--out", str(tmp_path / "ann"), "--regroup"]) == 1
    assert "no such folder" in capsys.readouterr().err
    assert main(["--images", str(images), "--out", str(tmp_path / "ann"), "--regroup"]) == 1  # nothing cached
    with pytest.raises(SystemExit):
        main(["--images", str(images), "--out", str(tmp_path / "ann"), "--regroup", "--limit", "0"])
    cv2.imwrite(str(images / "a_color.jpg"), np.zeros((H, W, 3), np.uint8))
    assert main(["--images", str(images), "--out", str(tmp_path / "ann"), "--regroup"]) == 1
    assert "a_color" in capsys.readouterr().err
