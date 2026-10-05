import numpy as np

from annofix.overlaps import (GroupingParams, candidates_to_annotation, dedupe, filter_candidates,
                              find_parents, group_objects, pairwise_intersections,
                              remove_small_components, resolve_overlaps)


def rect(y0, y1, x0, x1, shape=(100, 100)):
    m = np.zeros(shape, bool)
    m[y0:y1, x0:x1] = True
    return m


def mug_scene():
    body = rect(20, 60, 20, 52)    # 1280 px
    handle = rect(25, 55, 52, 62)  # 300 px
    mug = body | handle            # 1580 px: the whole object
    grip = rect(30, 35, 25, 45)    # 100 px inside the body: a sub-part
    box = rect(70, 90, 70, 90)     # 400 px: a separate object
    table = rect(60, 100, 0, 100)  # 4000 px: background
    speck = rect(5, 8, 90, 93)     # 9 px
    return np.stack([table, mug, body, handle, grip, box, speck])


def test_resolve_overlaps_smaller_wins():
    out = resolve_overlaps(np.stack([rect(0, 60, 0, 60), rect(10, 20, 10, 20)]))
    assert out[15, 15] == 2 and out[50, 50] == 1 and out[80, 80] == 0


def test_filter_drops_tiny_huge_and_border_spanning():
    masks = np.stack([rect(40, 45, 40, 45),   # 25 px: too small
                      rect(0, 60, 20, 80),    # 3600 px > 25% of image: background
                      rect(80, 90, 0, 100),   # 1000 px spanning left+right: table edge
                      rect(40, 60, 40, 60),   # normal object
                      rect(0, 20, 0, 20)])    # small object in a corner: kept
    assert filter_candidates(masks, GroupingParams(min_area=100)).tolist() == [3, 4]


def test_pairwise_intersections_match_brute_force():
    rng = np.random.default_rng(0)
    ms = []
    for _ in range(12):
        y, x = rng.integers(0, 80, 2)
        h, w = rng.integers(5, 20, 2)
        ms.append(rect(y, y + h, x, x + w))
    masks = np.stack(ms)
    flat = masks.reshape(12, -1).astype(np.int64)
    assert np.array_equal(pairwise_intersections(masks), flat @ flat.T)


def test_dedupe_keeps_best_scoring_duplicate():
    masks = np.stack([rect(10, 30, 10, 30), rect(10, 30, 10, 31), rect(50, 70, 50, 70)])
    assert dedupe(np.array([0.8, 0.9, 0.5]), pairwise_intersections(masks), 0.9) == [1, 2]


def test_find_parents_picks_smallest_container():
    masks = mug_scene()[1:5]  # mug, body, handle, grip
    assert find_parents(pairwise_intersections(masks), GroupingParams()).tolist() == [-1, 0, 0, 1]


def test_group_objects_keeps_depth_one_parts_only():
    masks = mug_scene()[1:5]
    inter = pairwise_intersections(masks)
    p = GroupingParams()
    assert group_objects(find_parents(inter, p), np.diag(inter), p) == [(0, [1, 2])]


def test_lone_near_complete_child_collapses():
    assert group_objects(np.array([-1, 0]), np.array([100, 95]), GroupingParams()) == [(0, [])]


def test_candidates_to_annotation_builds_objects_and_parts():
    masks = mug_scene()
    p = GroupingParams(min_area=20, min_component_px=10)
    ann = candidates_to_annotation(masks, np.full(len(masks), 0.9), p, image="scene.png")
    areas = sorted(sorted(ann.part_area(q.id) for q in o.parts) for o in ann.objects)
    assert areas == [[300, 1280], [400]]
    assert ann.part_at(6, 91) is None and ann.part_at(95, 5) is None  # speck and table
    assert ann.source == "amg" and ann.image == "scene.png"
    assert all(o.cls is None and all(q.name is None and q.material is None for q in o.parts)
               for o in ann.objects)


def test_partial_overlap_smaller_object_wins():
    masks = np.stack([rect(10, 40, 10, 40), rect(30, 50, 30, 50)])
    ann = candidates_to_annotation(masks, np.array([0.9, 0.9]), GroupingParams(min_area=20))
    assert sorted(ann.part_area(o.parts[0].id) for o in ann.objects) == [400, 800]
    assert ann.part_area(ann.part_at(35, 35)) == 400


def test_leftover_object_pixels_become_a_rest_part():
    masks = np.stack([rect(10, 50, 10, 50), rect(10, 50, 10, 30)])
    ann = candidates_to_annotation(masks, np.array([0.9, 0.9]), GroupingParams(min_area=20))
    assert len(ann.objects) == 1
    assert sorted(ann.part_area(q.id) for q in ann.objects[0].parts) == [800, 800]


def test_remove_small_components():
    m = rect(0, 10, 0, 10, (20, 20)) | rect(15, 17, 15, 17, (20, 20))
    out = remove_small_components(m, 10)
    assert out.sum() == 100 and not out[16, 16]


def test_no_candidates_gives_empty_annotation():
    ann = candidates_to_annotation(np.zeros((0, 30, 40), bool), np.zeros(0))
    assert ann.size == (30, 40) and ann.objects == []
