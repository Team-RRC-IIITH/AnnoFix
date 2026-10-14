"""Automatic mask clean-up (annofix/refine.py): SAM refinement with a smoothing fallback."""
import cv2
import numpy as np

from annofix import refine
from annofix.annotation import AnnObject, Annotation, Part


def disc(h, w, cx, cy, r):
    m = np.zeros((h, w), np.uint8)
    cv2.circle(m, (cx, cy), r, 1, -1)
    return m.astype(bool)


def staircase(mask, step=8):
    """What the pipeline's masks look like: computed small, scaled up with nearest-neighbour."""
    h, w = mask.shape
    small = cv2.resize(mask.astype(np.uint8), (w // step, h // step), interpolation=cv2.INTER_AREA)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)


def iou(a, b):
    return (a & b).sum() / max((a | b).sum(), 1)


def test_smoothing_takes_a_staircase_closer_to_the_true_shape():
    true = disc(200, 200, 100, 100, 60)
    noisy = staircase(true)
    noisy[10:13, 10:13] = True        # a speck
    noisy[100:104, 100:104] = False   # a pinhole
    smooth = refine.smooth(noisy)
    assert iou(smooth, true) > iou(noisy, true)
    assert not smooth[11, 11] and smooth[102, 102]
    perimeter = lambda m: cv2.arcLength(max(cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL,
                                                              cv2.CHAIN_APPROX_NONE)[0], key=len), True)
    assert perimeter(smooth) < perimeter(noisy)


def test_prompts_box_the_mask_and_point_inside_it_and_at_neighbours():
    mask = disc(100, 120, 40, 50, 20)
    others = disc(100, 120, 75, 50, 18) & ~mask
    pts, labels, box = refine.prompts_for(mask, others)
    x0, y0, x1, y1 = box
    assert x0 <= 20 and x1 >= 60 and y0 <= 30 and y1 >= 70 and x1 < 70  # the mask's box, a little padded
    for (x, y), label in zip(pts.astype(int), labels):
        assert (mask if label else others)[y, x]
    assert labels.tolist().count(1) == 3 and labels.tolist().count(0) >= 1


def predictor(*candidates):
    """A stand-in for SAM: always answers with these full-image candidates."""
    def predict(points, labels, box, prev_logits=None, multimask=True):
        masks = np.stack(candidates)
        return masks, np.full(len(masks), 0.9, np.float32), np.zeros((len(masks), 256, 256), np.float32)
    return predict


def test_the_candidate_closest_to_the_pipeline_mask_is_taken():
    true = disc(100, 100, 50, 50, 30)
    noisy = staircase(true, 6)
    job = refine.make_job(("object", 1), noisy, np.zeros_like(noisy))
    result = refine.run_job(predictor(disc(100, 100, 50, 50, 10), true, np.ones_like(true)), job)
    assert result.mask is not None and result.iou > 0.8  # a staircase vs the true disc: ~0.88
    assert np.array_equal(refine.place(result, true.shape), true)


def test_an_answer_unlike_the_pipeline_mask_is_refused():
    noisy = staircase(disc(100, 100, 50, 50, 30), 6)
    job = refine.make_job(("object", 1), noisy, np.zeros_like(noisy))
    result = refine.run_job(predictor(disc(100, 100, 50, 50, 8)), job)  # SAM found something else
    assert result.mask is None and result.iou < refine.MIN_IOU


def scene():
    """Object 1: body 2 + handle 3 (a disc and a bar sticking out); object 4: one part 5."""
    lm = np.zeros((120, 160), np.uint16)
    body = staircase(disc(120, 160, 50, 60, 30), 6)
    handle = np.zeros_like(body)
    handle[50:70, 75:95] = True
    lm[body] = 2
    lm[handle] = 3
    lm[staircase(disc(120, 160, 125, 60, 22), 6)] = 5
    objects = [AnnObject(1, "mug", [Part(2, "body", "plastic", {"role": "body"}), Part(3, "handle", "metal")]),
               AnnObject(4, "ball", [Part(5, "ball", "plastic")])]
    return Annotation(lm, objects, next_id=6)


def test_without_sam_every_mask_is_smoothed_and_the_structure_holds():
    ann = scene()
    proposal, report = refine.build_proposal(ann, {}, None)
    assert set(report.values()) == {"smoothed"}
    assert set(np.unique(proposal)) == {0, 2, 3, 5}
    mug = np.isin(proposal, [2, 3])
    assert not (mug & (proposal == 5)).any()           # one owner per pixel
    assert (proposal[55:65, 80:90] == 3).all()         # the handle survives inside the mug


def test_parts_stay_inside_their_object_and_the_body_is_the_rest():
    ann = scene()
    mug_sam = disc(120, 160, 50, 60, 30)
    mug_sam[50:70, 75:95] = True
    handle_sam = np.zeros_like(mug_sam)
    handle_sam[48:72, 75:110] = True                   # SAM's handle reaches beyond the mug
    jobs = {j.key: j for j in refine.jobs_for(ann, None)}
    assert set(jobs) == {("object", 1), ("object", 4), ("part", 3)}  # no job for the body or a lone part
    results = {("object", 1): refine.run_job(predictor(mug_sam), jobs[("object", 1)]),
               ("part", 3): refine.run_job(predictor(handle_sam), jobs[("part", 3)])}
    proposal, report = refine.build_proposal(ann, results, None)
    assert report[("object", 1)] == "sam" and report[("object", 4)] == "smoothed"
    assert not (proposal[48:72, 96:110] == 3).any()     # clipped to the mug
    assert set(np.unique(proposal[np.isin(proposal, [2, 3]) | mug_sam])) <= {2, 3}  # the mug is body + handle
    assert (proposal[mug_sam] != 0).all()


def test_hidden_parts_keep_every_pixel():
    ann = scene()
    locked = np.zeros(65536, bool)
    locked[5] = True
    before = ann.label_map == 5
    proposal, report = refine.build_proposal(ann, {}, locked)
    assert np.array_equal(proposal == 5, before) and report[("object", 4)] == "hidden"
    assert all(k[1] != 4 for k in (j.key for j in refine.jobs_for(ann, locked)))


def test_hiding_one_part_keeps_its_whole_object_as_it_is():
    ann = scene()
    locked = np.zeros(65536, bool)
    locked[3] = True                              # only the mug's handle is hidden
    before = ann.label_map.copy()
    proposal, report = refine.build_proposal(ann, {}, locked)
    mug = np.isin(before, [2, 3])
    assert np.array_equal(proposal[mug], before[mug]) and report[("object", 1)] == "hidden"
    assert not any(j.key in (("object", 1), ("part", 3)) for j in refine.jobs_for(ann, locked))


def test_smoothing_never_wipes_out_thin_shapes_or_fills_real_holes():
    pen = np.zeros((100, 200), bool)
    pen[50:53, 20:180] = True                     # a 3 px wide pen
    assert refine.smooth(pen).sum() >= 0.9 * pen.sum()
    ring = disc(120, 120, 60, 60, 40) & ~disc(120, 120, 60, 60, 25)  # a handle's loop: a real hole
    assert not refine.smooth(ring)[60, 60]
    pinhole = disc(120, 120, 60, 60, 40)
    pinhole[58:62, 58:62] = False
    assert refine.smooth(pinhole)[60, 60]


def test_an_answer_that_leaks_beyond_the_window_is_refused():
    h, w = 200, 300
    old = disc(h, w, 60, 100, 30)
    leak = old.copy()
    leak[95:105, 60:300] = True                   # runs along an edge to the image border
    job = refine.make_job(("object", 1), old, np.zeros_like(old))
    assert refine.run_job(predictor(leak), job).mask is None


def test_an_object_cannot_take_pixels_its_neighbour_still_has():
    lm = np.zeros((100, 160), np.uint16)
    lm[20:80, 10:60] = 2                          # small object
    lm[10:90, 60:150] = 3                         # big neighbour
    ann = Annotation(lm, [AnnObject(1, "a", [Part(2, "a", "metal")]), AnnObject(4, "b", [Part(3, "b", "paper")])],
                     next_id=5)
    greedy = (lm == 2).copy()
    greedy[20:80, 60:80] = True                   # SAM's answer for the small object runs into the big one
    big = lm == 3                                 # the big one keeps its mask
    jobs = {j.key: j for j in refine.jobs_for(ann, None)}
    results = {("object", 1): refine.Result(("object", 1), (0, 0, 160, 100), greedy, 0.75),
               ("object", 4): refine.Result(("object", 4), (0, 0, 160, 100), big, 1.0)}
    assert set(jobs) == {("object", 1), ("object", 4)}
    proposal, _ = refine.build_proposal(ann, results, None)
    assert (proposal[20:80, 66:80] == 3).all()    # beyond a few px, the neighbour's pixels stay its own
