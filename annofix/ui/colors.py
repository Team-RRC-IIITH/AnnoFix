"""Overlay colours as a lookup table: lut[part_id] -> 0xAARRGGBB (premultiplied: 255 for a part,
less for the parts faded while another is selected, 0 for background and hidden parts). Colour
mode, hidden parts/objects, solo and the selection highlight are all just a different table, so a full 1280x720 recolour is one np.take (~2 ms). Part borders are drawn at
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


def _rgb(obj, part, labels: Labels, mode: str) -> tuple[int, int, int]:
    """A part's colour in `mode`: by material, by part id or by object id."""
    if mode == "material":
        return material_colour(part.material, labels)
    return id_colour(part.id if mode == "part" else obj.id)


def part_colours(ann, labels: Labels, mode: str = "material") -> dict[int, str]:
    """Part id -> "#rrggbb" as the masks are coloured in `mode` (not faded, not hidden): the
    dots and bars of the objects list."""
    return {p.id: "#%02x%02x%02x" % _rgb(o, p, labels, mode) for o in ann.objects for p in o.parts}


def argb_faded(rgb: tuple[int, int, int], alpha: float) -> int:
    """Premultiplied ARGB at `alpha` (0-1): the overlay image is premultiplied."""
    a = int(round(255 * alpha))
    r, g, b = (int(round(c * a / 255)) for c in rgb)
    return (a << 24) | (r << 16) | (g << 8) | b


def build_lut(ann, labels: Labels, mode: str = "material", *, hidden_parts=frozenset(),
              hidden_objects=frozenset(), solo_object: int | None = None,
              selected=frozenset(), others: float = 1.0) -> np.ndarray:
    """Part id -> overlay colour. With a selection, the other parts fade to `others` (0-1);
    the selected parts keep their true colour (the canvas outlines them)."""
    if mode not in COLOUR_MODES:
        raise ValueError(f"colour mode must be one of {COLOUR_MODES}, got {mode!r}")
    lut = np.zeros(LUT_SIZE, np.uint32)
    for o in ann.objects:
        if o.id in hidden_objects or (solo_object is not None and o.id != solo_object):
            continue
        for p in o.parts:
            if p.id in hidden_parts:
                continue
            rgb = _rgb(o, p, labels, mode)
            if selected and p.id not in selected and others < 1.0:
                lut[p.id] = argb_faded(rgb, others)
            else:
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
        # Halve RGB, keep alpha (premultiplied, so RGB stays within alpha).
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
