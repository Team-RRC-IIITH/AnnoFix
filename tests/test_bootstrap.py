import subprocess
import sys

import cv2
import numpy as np
import pytest

from annofix.bootstrap import bootstrap_image, cache_path, load_cache, main, save_cache
from annofix.io import load_annotation, save_annotation
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
    save_cache(tmp_path / "c.npz", masks, np.array([0.1, 0.2, 0.3], np.float32))
    back, scores = load_cache(tmp_path / "c.npz")
    assert np.array_equal(back, masks) and scores.tolist() == pytest.approx([0.1, 0.2, 0.3])


def test_cache_of_wrong_size_is_an_error(images, tmp_path):
    out = tmp_path / "ann"
    save_cache(cache_path(out, "a_color"), np.zeros((1, H + 1, W), bool), np.array([0.9]))
    with pytest.raises(ValueError, match="cache"):
        bootstrap_image(images / "a_color.png", out, None, GroupingParams())


def test_main_regroup_skips_depth_and_counts(images, tmp_path, capsys):
    out = tmp_path / "ann"
    for stem in ("a_color", "b_color"):
        save_cache(cache_path(out, stem), *object_masks(None))
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
