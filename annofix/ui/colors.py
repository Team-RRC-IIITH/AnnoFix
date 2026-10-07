"""Overlay colours as a lookup table: lut[part_id] -> 0xAARRGGBB (premultiplied; alpha is
255 or 0). Colour mode, hidden parts/objects, solo and the selection highlight are all just a
different table, so a full 1280x720 recolour is one np.take (~2 ms). Part borders are drawn at
half brightness, so neighbouring parts of the same colour stay distinct (~1 ms more). Opacity is
applied by the Qt item, not here."""
from __future__ import annotations

import numpy as np

from ..config import Labels
from ..session import COLOUR_MODES

PALETTE = [
    (31, 119, 180), (255, 127, 14), (44, 160, 44), (214, 39, 40), (148, 103, 189),
    (140, 86, 75), (227, 119, 194), (188, 189, 34), (23, 190, 207), (174, 199, 232),
    (255, 187, 120), (152, 223, 138), (255, 152, 150), (197, 176, 213), (196, 156, 148),
    (247, 182, 210), (219, 219, 141), (158, 218, 229), (57, 59, 121), (99, 121, 57),
]
UNLABELLED = (128, 128, 128)
LUT_SIZE = 65536


def argb(rgb: tuple[int, int, int]) -> int:
    r, g, b = rgb
    return 0xFF000000 | (r << 16) | (g << 8) | b


def id_colour(i: int) -> tuple[int, int, int]:
    """Stable per id, and neighbouring ids differ (7 is coprime with the palette size)."""
    return PALETTE[(i * 7) % len(PALETTE)]


def material_colour(material: str | None, labels: Labels) -> tuple[int, int, int]:
    if material in labels.materials:
        return PALETTE[labels.materials.index(material) % len(PALETTE)]
    return UNLABELLED


def lighten(rgb: tuple[int, int, int], amount: float = 0.5) -> tuple[int, int, int]:
    return tuple(int(c + (255 - c) * amount) for c in rgb)


def build_lut(ann, labels: Labels, mode: str = "material", *, hidden_parts=frozenset(),
              hidden_objects=frozenset(), solo_object: int | None = None,
              selected=frozenset()) -> np.ndarray:
    if mode not in COLOUR_MODES:
        raise ValueError(f"colour mode must be one of {COLOUR_MODES}, got {mode!r}")
    lut = np.zeros(LUT_SIZE, np.uint32)
    for o in ann.objects:
        if o.id in hidden_objects or (solo_object is not None and o.id != solo_object):
            continue
        for p in o.parts:
            if p.id in hidden_parts:
                continue
            if mode == "material":
                rgb = material_colour(p.material, labels)
            elif mode == "part":
                rgb = id_colour(p.id)
            else:
                rgb = id_colour(o.id)
            if p.id in selected:
                rgb = lighten(rgb)
            lut[p.id] = argb(rgb)
    return lut


def edges(label_map: np.ndarray, out: np.ndarray | None = None) -> np.ndarray:
    """Pixels with a 4-neighbour of a different label, marked on both sides of each border."""
    e = np.zeros(label_map.shape, bool) if out is None else out
    e[...] = False
    dx = label_map[:, :-1] != label_map[:, 1:]
    e[:, :-1] |= dx
    e[:, 1:] |= dx
    dy = label_map[:-1, :] != label_map[1:, :]
    e[:-1, :] |= dy
    e[1:, :] |= dy
    return e


def recolour(label_map: np.ndarray, lut: np.ndarray, out: np.ndarray, outline: bool = False,
             edge_buf: np.ndarray | None = None, lines: bool = True) -> np.ndarray:
    """Colour `out` from the LUT. `outline` keeps only the part borders; otherwise, with
    `lines`, the borders are drawn darker over the fills."""
    np.take(lut, label_map, out=out)
    if outline:
        np.multiply(out, edges(label_map, edge_buf), out=out)
    elif lines:
        idx = np.flatnonzero(edges(label_map, edge_buf))  # only border pixels: far fewer than all
        flat = out.reshape(-1)
        v = flat[idx]
        # Halve RGB, keep alpha (premultiplied with alpha 255, or all zero for background).
        flat[idx] = (v & np.uint32(0xFF000000)) | ((v >> np.uint32(1)) & np.uint32(0x007F7F7F))
    return out


def preview_lut() -> np.ndarray:
    """Colours for the SAM accept preview (sam_edit.preview_map codes)."""
    from ..sam_edit import PREVIEW_ADDED, PREVIEW_KEPT, PREVIEW_REMOVED, PREVIEW_TAKEN

    lut = np.zeros(LUT_SIZE, np.uint32)
    lut[PREVIEW_KEPT] = argb((255, 255, 255))
    lut[PREVIEW_ADDED] = argb((40, 220, 40))
    lut[PREVIEW_TAKEN] = argb((255, 150, 0))
    lut[PREVIEW_REMOVED] = argb((230, 30, 60))
    return lut
