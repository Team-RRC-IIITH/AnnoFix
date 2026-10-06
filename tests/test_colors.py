import time

import numpy as np
import pytest
from helpers import LABELS, scene_annotation

from annofix.ui.colors import (UNLABELLED, argb, build_lut, edges, id_colour, lighten, material_colour,
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
    assert build_lut(ann, LABELS, "part", selected={2})[2] == argb(lighten(id_colour(2)))


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


def test_recolour_full_frame_under_5ms():
    lm = np.random.default_rng(0).integers(0, 300, (720, 1280)).astype(np.uint16)
    lut = np.arange(65536, dtype=np.uint32) | np.uint32(0xFF000000)
    out = np.zeros(lm.shape, np.uint32)
    recolour(lm, lut, out)
    times = []
    for _ in range(10):
        t = time.perf_counter()
        recolour(lm, lut, out)
        times.append(time.perf_counter() - t)
    assert sorted(times)[5] < 0.005
