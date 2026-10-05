import os

import numpy as np
import pytest

pytestmark = [
    pytest.mark.sam,
    pytest.mark.skipif(os.environ.get("ANNOFIX_SAM_TESTS") != "1",
                       reason="set ANNOFIX_SAM_TESTS=1 (downloads a SAM 2.1 checkpoint)"),
]


@pytest.fixture(scope="module")
def backend():
    from annofix.sam_backend import Sam2Backend
    return Sam2Backend("tiny")


def square_image():
    img = np.full((240, 320, 3), 30, np.uint8)
    img[60:180, 100:220] = (230, 200, 40)
    return img


def iou(a, b):
    return (a & b).sum() / (a | b).sum()


def test_click_segments_the_square(backend):
    backend.set_image(square_image())
    masks, scores, logits = backend.predict(points=[[160, 120]], labels=[1], multimask=True)
    assert masks.shape == (3, 240, 320) and masks.dtype == bool
    assert scores.shape == (3,) and logits.shape == (3, 256, 256)
    truth = np.zeros((240, 320), bool)
    truth[60:180, 100:220] = True
    assert max(iou(m, truth) for m in masks) > 0.85


def test_generate_returns_aligned_masks(backend):
    masks, scores = backend.generate(square_image(), points_per_side=8)
    assert masks.ndim == 3 and masks.shape[1:] == (240, 320) and masks.dtype == bool
    assert len(scores) == len(masks) > 0


def test_local_checkpoint_loads_with_matching_config():
    """The offline --checkpoint path: CONFIGS must pair with the 2.1 .pt files."""
    from huggingface_hub import hf_hub_download

    from annofix.sam_backend import Sam2Backend
    path = hf_hub_download("facebook/sam2.1-hiera-tiny", "sam2.1_hiera_tiny.pt")
    backend = Sam2Backend("tiny", checkpoint=path)
    backend.set_image(square_image())
    masks, _, _ = backend.predict(points=[[160, 120]], labels=[1])
    assert masks.shape == (3, 240, 320)
