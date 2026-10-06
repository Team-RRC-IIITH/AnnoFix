import os
from pathlib import Path

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


REAL_FRAME = (Path(__file__).resolve().parents[1].parent
              / "CamCalibration/MC-Calib/data/rosbag_mccalib_moving/frames/cam0_120_color.png")


@pytest.mark.skipif(not REAL_FRAME.exists(), reason="needs the CamCalibration lab frame")
def test_generate_recalls_lab_objects():
    """SAM's default AMG thresholds (0.8 / 0.95) produced no candidate at all on these."""
    from annofix.io import load_image
    from annofix.sam_backend import Sam2Backend

    masks, _ = Sam2Backend("base_plus").generate(load_image(REAL_FRAME))
    for name, (y, x) in {"robot arm": (192, 627), "laptop": (560, 260), "tripod camera": (474, 621)}.items():
        assert masks[:, y, x].any(), f"no candidate mask on the {name}"


@pytest.mark.skipif(not REAL_FRAME.exists(), reason="needs the CamCalibration lab frame")
def test_generate_leaves_gpu_headroom():
    """With 64 points per batch AMG reserved 4.6 of the RTX 3050's 6 GB and hit OOM retries
    while the desktop shared the GPU; 32 gives the same masks at the same speed in 2.8 GB."""
    import torch

    from annofix.io import load_image
    from annofix.sam_backend import Sam2Backend

    if not torch.cuda.is_available():
        pytest.skip("needs CUDA")
    backend = Sam2Backend("base_plus", "cuda")
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    backend.generate(load_image(REAL_FRAME))
    assert torch.cuda.max_memory_reserved() < 3.5 * 2**30


@pytest.mark.skipif(not REAL_FRAME.exists(), reason="needs the CamCalibration lab frame")
def test_session_sam_fix_on_a_real_frame(tmp_path):
    import shutil

    from annofix.config import Labels
    from annofix.sam_backend import Sam2Backend
    from annofix.session import Session

    images = tmp_path / "images"
    images.mkdir()
    shutil.copy(REAL_FRAME, images / REAL_FRAME.name)
    s = Session(images, tmp_path / "ann", Labels(materials=("metal",), objects={}))
    s.sam_enabled = True
    backend = Sam2Backend("base_plus")

    def drive(kind):
        if kind == "image":
            backend.set_image(s.image)
        elif kind == "sam_request":
            req = s.take_sam_request()
            s.sam_receive(req.token, req.seq,
                          *backend.predict(req.points, req.labels, req.box, req.prev_logits, req.multimask))

    s.subscribe(drive)
    s.open(0)
    s.set_tool("sam")
    s.sam_click(627, 192, True)  # robot arm, upper link
    assert s.sam.mask is not None and s.sam.mask[192, 627]
    assert 2_000 < int(s.sam.mask.sum()) < 0.25 * 1280 * 720
    assert s.sam_accept() and s.ann.part_at(192, 627) == s.selected[0]
