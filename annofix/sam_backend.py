"""SAM 2.1 wrapper. torch and sam2 are imported inside functions so the rest of AnnoFix
imports (and its tests run) without them."""
from __future__ import annotations

import contextlib
from pathlib import Path

import numpy as np

MODELS = {
    "tiny": "facebook/sam2.1-hiera-tiny",
    "small": "facebook/sam2.1-hiera-small",
    "base_plus": "facebook/sam2.1-hiera-base-plus",
    "large": "facebook/sam2.1-hiera-large",
}
# Hydra config names inside the installed sam2 package, for --checkpoint (offline) loading.
CONFIGS = {
    "tiny": "configs/sam2.1/sam2.1_hiera_t.yaml",
    "small": "configs/sam2.1/sam2.1_hiera_s.yaml",
    "base_plus": "configs/sam2.1/sam2.1_hiera_b+.yaml",
    "large": "configs/sam2.1/sam2.1_hiera_l.yaml",
}
# Set from the M0 measurements (docs/notes/2026-10-03-m0-measurements.md).
DEFAULT_MODEL = {"cuda": "base_plus", "cpu": "small"}


def pick_device(device: str | None) -> str:
    import torch

    if device:
        return device
    return "cuda" if torch.cuda.is_available() else "cpu"


class Sam2Backend:
    def __init__(self, model: str | None = None, device: str | None = None,
                 checkpoint: str | Path | None = None):
        import torch
        from sam2.build_sam import build_sam2, build_sam2_hf
        from sam2.sam2_image_predictor import SAM2ImagePredictor

        self.device = pick_device(device)
        self.model_name = model or DEFAULT_MODEL["cuda" if self.device == "cuda" else "cpu"]
        if self.model_name not in MODELS:
            raise ValueError(f"unknown SAM model {self.model_name!r}; choose from {sorted(MODELS)}")
        if self.device == "cpu":
            torch.set_num_threads(10)  # machine rule
        else:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        if checkpoint:
            self.model = build_sam2(CONFIGS[self.model_name], str(checkpoint), device=self.device)
        else:
            self.model = build_sam2_hf(MODELS[self.model_name], device=self.device)
        self.predictor = SAM2ImagePredictor(self.model)
        self._amg = None
        self._amg_grid = None

    def _context(self):
        import torch

        stack = contextlib.ExitStack()
        stack.enter_context(torch.inference_mode())
        if self.device == "cuda":
            stack.enter_context(torch.autocast("cuda", dtype=torch.bfloat16))
        return stack

    def set_image(self, image: np.ndarray) -> None:
        with self._context():
            self.predictor.set_image(image)

    def predict(self, points=None, labels=None, box=None, prev_logits=None, multimask=True):
        with self._context():
            masks, scores, logits = self.predictor.predict(
                point_coords=None if points is None else np.asarray(points, np.float32),
                point_labels=None if labels is None else np.asarray(labels, np.int32),
                box=None if box is None else np.asarray(box, np.float32),
                mask_input=prev_logits,
                multimask_output=multimask,
            )
        return masks > 0, np.asarray(scores, np.float32), np.asarray(logits, np.float32)

    def generate(self, image: np.ndarray, points_per_side: int = 32, points_per_batch: int = 64):
        from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

        if self._amg is None or self._amg_grid != (points_per_side, points_per_batch):
            # min_mask_region_area=0: hole/sprinkle cleanup needs the CUDA extension we don't
            # build; overlaps.remove_small_components does that job instead.
            self._amg = SAM2AutomaticMaskGenerator(
                self.model, points_per_side=points_per_side, points_per_batch=points_per_batch,
                box_nms_thresh=0.9, min_mask_region_area=0, output_mode="binary_mask",
                multimask_output=True)
            self._amg_grid = (points_per_side, points_per_batch)
        with self._context():
            records = self._amg.generate(image)
        if not records:
            return np.zeros((0, *image.shape[:2]), bool), np.zeros(0, np.float32)
        masks = np.stack([r["segmentation"] for r in records]).astype(bool)
        scores = np.array([r["predicted_iou"] for r in records], np.float32)
        return masks, scores
