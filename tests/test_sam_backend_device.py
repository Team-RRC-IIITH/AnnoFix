"""Which device SAM loads on, without loading a model: SAM 2.1's builder is stubbed."""
import pytest

pytest.importorskip("sam2")

import sam2.build_sam
import sam2.sam2_image_predictor

from annofix import sam_backend
from annofix.sam_backend import Sam2Backend

NO_WARMUP = ["++model.image_encoder.neck.position_encoding.warmup_cache=false",
             "++model.memory_encoder.position_encoding.warmup_cache=false"]


@pytest.fixture
def builds(monkeypatch):
    """Every build_sam2_hf call as (model id, device, hydra overrides); a device listed in
    `full` fails the way a full GPU does."""
    calls, full = [], set()

    def fake_build(model_id, device="cuda", hydra_overrides_extra=(), **kwargs):
        calls.append((model_id, device, list(hydra_overrides_extra)))
        if device in full:
            raise RuntimeError("CUDA error: out of memory")
        return object()

    monkeypatch.setattr(sam2.build_sam, "build_sam2_hf", fake_build)
    monkeypatch.setattr(sam2.sam2_image_predictor, "SAM2ImagePredictor", lambda model: object())
    return calls, full


def test_cpu_never_warms_up_on_the_gpu(builds):
    # SAM 2.1 builds a small cache on the GPU whenever one exists; with the GPU full (CVAT's SAM
    # holding it) that crashed even with --device cpu.
    calls, _ = builds
    b = Sam2Backend("small", "cpu")
    assert b.device == "cpu" and calls[0][1] == "cpu"
    assert all(o in calls[0][2] for o in NO_WARMUP)


def test_a_full_gpu_falls_back_to_the_cpu_and_says_so(builds, monkeypatch):
    calls, full = builds
    full.add("cuda")
    monkeypatch.setattr(sam_backend, "pick_device", lambda device: device or "cuda")
    b = Sam2Backend()
    assert [c[1] for c in calls] == ["cuda", "cpu"]
    assert b.device == "cpu" and b.model_name == "small" and "GPU" in b.note


def test_an_explicit_gpu_choice_is_not_overridden(builds, monkeypatch):
    _, full = builds
    full.add("cuda")
    with pytest.raises(RuntimeError, match="out of memory"):
        Sam2Backend(device="cuda")
