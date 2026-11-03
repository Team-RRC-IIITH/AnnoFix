import time

import numpy as np
import pytest
from helpers import LABELS, scene_annotation

from annofix.ui.colors import (UNLABELLED, argb, build_lut, edges, id_colour, material_colour,
                               recolour)


def test_material_mode_colours_by_material():
    lut = build_lut(scene_annotation(), LABELS, "material")
    assert lut[2] == argb(material_colour("metal", LABELS))
    assert lut[3] == argb(UNLABELLED)
    assert lut[5] == argb(UNLABELLED)
    assert lut[0] == 0
    assert material_colour("plastic", LABELS) != material_colour("metal", LABELS)


def test_object_mode_gives_all_parts_of_an_object_one_colour():
    lut = build_lut(scene_annotation(), LABELS, "object")
    assert lut[2] == lut[3] == argb(id_colour(1)) and lut[5] == argb(id_colour(4))


def test_part_mode_colours_each_part():
    lut = build_lut(scene_annotation(), LABELS, "part")
    assert len({int(lut[2]), int(lut[3]), int(lut[5])}) == 3


def test_hidden_solo_and_selected():
    ann = scene_annotation()
    assert build_lut(ann, LABELS, hidden_parts={3})[3] == 0
    lut = build_lut(ann, LABELS, hidden_objects={1})
    assert lut[2] == lut[3] == 0 and lut[5] != 0
    lut = build_lut(ann, LABELS, solo_object=4)
    assert lut[2] == 0 and lut[5] != 0
    # the selection keeps its true colour; the other parts fade (premultiplied alpha)
    lut = build_lut(ann, LABELS, "part", selected={2}, others=0.4)
    assert lut[2] == argb(id_colour(2))
    a5, r5 = int(lut[5]) >> 24, (int(lut[5]) >> 16) & 0xFF
    assert a5 == round(0.4 * 255) and r5 == round(id_colour(5)[0] * a5 / 255)
    assert build_lut(ann, LABELS, "part", others=0.4)[5] == argb(id_colour(5))   # nothing selected: no fading


def test_unknown_mode_rejected():
    with pytest.raises(ValueError):
        build_lut(scene_annotation(), LABELS, "rainbow")


def test_recolour_and_outline():
    ann = scene_annotation()
    lut = build_lut(ann, LABELS, "part")
    out = np.zeros(ann.size, np.uint32)
    recolour(ann.label_map, lut, out)
    assert out[10, 10] == lut[2] and out[0, 0] == 0
    recolour(ann.label_map, lut, out, outline=True)
    assert out[5, 10] == lut[2] and out[10, 10] == 0  # border kept, interior cleared
    lm = np.zeros((5, 5), np.uint16)
    lm[1:4, 1:4] = 2
    e = edges(lm)
    assert e[1, 1] and e[3, 3] and not e[2, 2]


def test_part_borders_are_drawn_darker():
    lm = np.zeros((6, 8), np.uint16)
    lm[1:5, 1:4] = 2
    lm[1:5, 4:7] = 3
    lut = np.zeros(65536, np.uint32)
    lut[2] = lut[3] = argb((200, 100, 50))  # same colour: only the border tells them apart
    out = np.zeros(lm.shape, np.uint32)
    recolour(lm, lut, out)
    dark = argb((100, 50, 25))
    assert out[2, 2] == lut[2]                    # inside a part: its own colour
    assert out[2, 3] == dark and out[2, 4] == dark  # both sides of the shared border
    assert out[1, 1] == dark and out[0, 0] == 0     # the outer border too; background stays clear
    recolour(lm, lut, out, lines=False)
    assert out[2, 3] == lut[2]


def test_recolour_full_frame_under_5ms():
    # A realistic map (a few dozen regions, not per-pixel noise): border lines cost per border pixel.
    rng = np.random.default_rng(0)
    lm = np.zeros((720, 1280), np.uint16)
    for i in range(1, 40):
        y, x = rng.integers(0, 520), rng.integers(0, 980)
        lm[y:y + rng.integers(20, 200), x:x + rng.integers(20, 300)] = i
    lut = np.arange(65536, dtype=np.uint32) | np.uint32(0xFF000000)
    out = np.zeros(lm.shape, np.uint32)
    recolour(lm, lut, out)
    times = []
    for _ in range(10):
        t = time.perf_counter()
        recolour(lm, lut, out)
        times.append(time.perf_counter() - t)
    assert sorted(times)[5] < 0.005
