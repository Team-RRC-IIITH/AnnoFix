import numpy as np
import pytest

from annofix.sam_backend import FakeBackend, check_prev_logits, device_kind


def test_device_kind_ignores_the_device_index():
    assert device_kind("cuda") == device_kind("cuda:0") == device_kind("cuda:1") == "cuda"
    assert device_kind("cpu") == "cpu"


def blank():
    b = FakeBackend()
    b.set_image(np.zeros((40, 60, 3), np.uint8))
    return b


def test_fake_backend_gives_three_nested_candidates_for_one_click():
    masks, scores, logits = blank().predict(points=[[30, 20]], labels=[1], multimask=True)
    assert masks.shape == (3, 40, 60) and masks.dtype == bool and logits.shape == (3, 256, 256)
    areas = masks.sum((1, 2))
    assert areas[0] < areas[1] < areas[2] and masks[:, 20, 30].all()
    assert scores.tolist() == pytest.approx([0.5, 0.9, 0.7]) and logits[2, 0, 0] == 16


def test_fake_backend_respects_exclude_points_and_boxes():
    b = blank()
    masks, _, _ = b.predict(points=[[30, 20], [37, 20]], labels=[1, 0], multimask=False)
    assert masks.shape == (1, 40, 60) and masks[0, 20, 30] and not masks[0, 20, 36]
    masks, _, _ = b.predict(box=[10, 5, 20, 15], multimask=False)
    assert masks[0].sum() == 11 * 11
    assert [c[0] for c in b.calls] == ["set_image", "predict", "predict"]


def test_prev_logits_must_be_a_single_candidate():
    check_prev_logits(None)
    check_prev_logits(np.zeros((1, 256, 256), np.float32))
    with pytest.raises(ValueError, match="one candidate"):
        check_prev_logits(np.zeros((3, 256, 256), np.float32))
