import numpy as np
import pytest
from helpers import send_move, send_wheel
from PyQt6.QtCore import QPoint, QPointF, Qt

from annofix.ui.canvas import Canvas
from annofix.ui.overlay_item import OverlayItem


def test_overlay_shows_lut_colour_and_follows_in_place_edits(qtbot):
    lm = np.zeros((4, 6), np.uint16)
    lm[1, 2] = 7
    item = OverlayItem(lm, lines=False)  # border lines are tested in test_colors
    lut = np.zeros(65536, np.uint32)
    lut[7], lut[9] = 0xFF112233, 0xFF445566
    item.set_lut(lut)
    assert item.argb_at(2, 1) == 0xFF112233 and item.argb_at(0, 0) == 0
    lm[0, 0] = 9  # the annotation edits its array in place...
    item.refresh()  # ...and one refresh shows it
    assert item.argb_at(0, 0) == 0xFF445566


def test_outline_mode_hides_interiors(qtbot):
    lm = np.zeros((5, 5), np.uint16)
    lm[1:4, 1:4] = 7
    item = OverlayItem(lm)
    lut = np.zeros(65536, np.uint32)
    lut[7] = 0xFF112233
    item.set_lut(lut)
    item.set_outline(True)
    assert item.argb_at(1, 1) == 0xFF112233 and item.argb_at(2, 2) == 0


def make_canvas(qtbot, h=40, w=60):
    c = Canvas()
    qtbot.addWidget(c)
    c.resize(600, 400)
    c.show()
    qtbot.waitExposed(c)
    c.set_image(np.full((h, w, 3), 90, np.uint8), np.zeros((h, w), np.uint16))
    return c


def at(c, x, y):
    return c.mapFromScene(QPointF(x + 0.5, y + 0.5))


def test_click_reports_image_pixel(qtbot):
    c = make_canvas(qtbot)
    with qtbot.waitSignal(c.clicked) as sig:
        qtbot.mouseClick(c.viewport(), Qt.MouseButton.LeftButton, pos=at(c, 12, 30))
    assert sig.args == [12, 30, False]


def test_ctrl_click_adds_to_selection(qtbot):
    c = make_canvas(qtbot)
    with qtbot.waitSignal(c.clicked) as sig:
        qtbot.mouseClick(c.viewport(), Qt.MouseButton.LeftButton, Qt.KeyboardModifier.ControlModifier,
                         at(c, 5, 5))
    assert sig.args == [5, 5, True]


def test_paint_mode_turns_drags_into_strokes(qtbot):
    c = make_canvas(qtbot)
    c.mode = "paint"
    got = []
    c.stroke_started.connect(lambda x, y: got.append(("start", x, y)))
    c.stroke_moved.connect(lambda x, y: got.append(("move", x, y)))
    c.stroke_ended.connect(lambda: got.append(("end",)))
    vp = c.viewport()
    qtbot.mousePress(vp, Qt.MouseButton.LeftButton, pos=at(c, 10, 10))
    send_move(vp, at(c, 20, 10))
    qtbot.mouseRelease(vp, Qt.MouseButton.LeftButton, pos=at(c, 20, 10))
    assert got[0] == ("start", 10, 10) and ("move", 20, 10) in got and got[-1] == ("end",)


def test_switching_to_a_smaller_image_rebuilds_the_overlay(qtbot):
    c = make_canvas(qtbot)
    item = c.set_image(np.zeros((30, 50, 3), np.uint8), np.zeros((30, 50), np.uint16))
    assert item.boundingRect().width() == 50 and c.scene().sceneRect().height() == 30
    assert sum(isinstance(i, OverlayItem) for i in c.scene().items()) == 1


def test_masks_can_be_hidden(qtbot):
    c = make_canvas(qtbot)
    c.set_masks_visible(False)
    assert not c.overlay.isVisible()
    c.set_masks_visible(True)
    assert c.overlay.isVisible()


def test_sam_mode_left_is_include_right_is_exclude(qtbot):
    c = make_canvas(qtbot)
    c.mode = "sam"
    got = []
    c.sam_clicked.connect(lambda x, y, include: got.append((x, y, include)))
    c.clicked.connect(lambda *a: got.append("select click"))
    qtbot.mouseClick(c.viewport(), Qt.MouseButton.LeftButton, pos=at(c, 12, 30))
    qtbot.mouseClick(c.viewport(), Qt.MouseButton.RightButton, pos=at(c, 13, 31))
    assert got == [(12, 30, True), (13, 31, False)]


def test_shift_drag_draws_a_box(qtbot):
    c = make_canvas(qtbot)
    c.mode = "sam"
    got = []
    c.sam_boxed.connect(lambda *box: got.append(box))
    c.sam_clicked.connect(lambda *a: got.append("click"))
    vp = c.viewport()
    qtbot.mousePress(vp, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.ShiftModifier, at(c, 10, 10))
    send_move(vp, at(c, 20, 15))
    qtbot.mouseRelease(vp, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.ShiftModifier, at(c, 20, 15))
    assert got == [(10, 10, 20, 15)]


def test_prompt_markers_and_preview(qtbot):
    from annofix.sam_edit import PREVIEW_ADDED
    from annofix.ui.colors import preview_lut

    c = make_canvas(qtbot)
    c.set_sam_prompts([(5, 5), (10, 10)], [1, 0], (1, 1, 20, 20))
    assert len(c._markers) == 3
    preview = np.zeros((40, 60), np.uint8)
    preview[5, 5] = PREVIEW_ADDED
    c.set_sam_preview(preview)
    assert c.sam_preview_item.argb_at(5, 5) == preview_lut()[PREVIEW_ADDED]
    c.set_sam_preview(None)
    c.set_sam_prompts([], [], None)
    assert c.sam_preview_item is None and c._markers == []


def test_tab_never_moves_focus_off_the_canvas(qtbot):
    from PyQt6.QtWidgets import QApplication, QHBoxLayout, QLineEdit, QWidget

    box = QWidget()
    qtbot.addWidget(box)
    layout = QHBoxLayout(box)
    c = Canvas()
    layout.addWidget(c)
    layout.addWidget(QLineEdit())  # somewhere Tab could move focus to
    box.show()
    qtbot.waitExposed(box)
    box.activateWindow()
    c.setFocus()
    qtbot.waitUntil(lambda: QApplication.focusWidget() is c, timeout=2000)
    qtbot.keyClick(c, Qt.Key.Key_Tab)
    assert QApplication.focusWidget() is c


def test_a_missed_release_ends_the_stroke(qtbot):
    # The button went up over another window (or a dialog grabbed the mouse): the next move
    # arrives with no button held and must not keep painting.
    c = make_canvas(qtbot)
    c.mode = "paint"
    got = []
    c.stroke_moved.connect(lambda x, y: got.append("move"))
    c.stroke_ended.connect(lambda: got.append("end"))
    qtbot.mousePress(c.viewport(), Qt.MouseButton.LeftButton, pos=at(c, 10, 10))
    send_move(c.viewport(), at(c, 20, 10), buttons=Qt.MouseButton.NoButton)
    send_move(c.viewport(), at(c, 30, 10), buttons=Qt.MouseButton.NoButton)
    assert got == ["end"]


def fitted_scale(c):
    vp = c.viewport()
    return min((vp.width() - 4) / 60, (vp.height() - 4) / 40)  # fitInView keeps a 2 px margin


def test_view_fits_on_resize_until_you_zoom_and_home_fits_again(qtbot):
    c = make_canvas(qtbot)
    c.resize(1200, 800)
    qtbot.waitUntil(lambda: c.transform().m11() == pytest.approx(fitted_scale(c), rel=0.02), timeout=2000)
    send_wheel(c.viewport(), QPoint(100, 100), 2)
    zoomed = c.transform().m11()
    assert zoomed > fitted_scale(c) * 1.2
    c.resize(700, 500)
    qtbot.waitUntil(lambda: c.viewport().width() < 700, timeout=2000)
    assert c.transform().m11() == pytest.approx(zoomed)  # your zoom is kept
    c.fit()
    assert c.transform().m11() == pytest.approx(fitted_scale(c), rel=0.02)
    c.resize(900, 600)
    qtbot.waitUntil(lambda: c.transform().m11() == pytest.approx(fitted_scale(c), rel=0.02), timeout=2000)


@pytest.mark.parametrize("mode", ["select", "paint"])
def test_right_drag_pans_outside_sam_mode(qtbot, mode):
    c = make_canvas(qtbot)
    c.mode = mode
    c.scale(8, 8)  # zoomed in, so there is room to pan
    got = []
    for signal in (c.clicked, c.stroke_started, c.sam_clicked, c.card_requested):
        signal.connect(lambda *a: got.append(a))
    h0, v0 = c.horizontalScrollBar().value(), c.verticalScrollBar().value()
    vp = c.viewport()
    qtbot.mousePress(vp, Qt.MouseButton.RightButton, pos=QPoint(300, 200))
    send_move(vp, QPoint(260, 170), buttons=Qt.MouseButton.RightButton)
    qtbot.mouseRelease(vp, Qt.MouseButton.RightButton, pos=QPoint(260, 170))
    assert (c.horizontalScrollBar().value(), c.verticalScrollBar().value()) == (h0 + 40, v0 + 30)
    assert got == []


RECT = np.array([[5, 5], [5, 14], [24, 14], [24, 5]])


def points_canvas(qtbot):
    c = make_canvas(qtbot)
    c.mode = "points"
    c.set_rings([RECT])
    got = []
    c.point_moved.connect(lambda *a: got.append(("move", *a)))
    c.point_inserted.connect(lambda *a: got.append(("insert", *a)))
    c.point_deleted.connect(lambda *a: got.append(("delete", *a)))
    c.clicked.connect(lambda *a: got.append(("click", *a)))
    return c, got


def test_dragging_a_point_reports_where_it_was_dropped(qtbot):
    c, got = points_canvas(qtbot)
    vp = c.viewport()
    qtbot.mousePress(vp, Qt.MouseButton.LeftButton, pos=at(c, 24, 14))
    send_move(vp, at(c, 26, 14))
    send_move(vp, at(c, 28, 15))
    qtbot.mouseRelease(vp, Qt.MouseButton.LeftButton, pos=at(c, 28, 15))
    assert got == [("move", 0, 2, 28, 15)]
    assert (c._rings[0] == RECT).all()  # the shown outline waits for the Session's answer


def test_ctrl_click_deletes_a_point_and_double_click_adds_one(qtbot):
    c, got = points_canvas(qtbot)
    vp = c.viewport()
    qtbot.mouseClick(vp, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.ControlModifier, at(c, 5, 5))
    qtbot.mouseDClick(vp, Qt.MouseButton.LeftButton, pos=at(c, 14, 14))
    assert got == [("delete", 0, 0), ("insert", 0, 1, 14, 14)]


def test_clicks_away_from_the_outline_select(qtbot):
    c, got = points_canvas(qtbot)
    qtbot.mouseClick(c.viewport(), Qt.MouseButton.LeftButton, pos=at(c, 40, 30))
    assert got == [("click", 40, 30, False)]


def test_a_lost_release_drops_the_drag(qtbot):
    c, got = points_canvas(qtbot)
    vp = c.viewport()
    qtbot.mousePress(vp, Qt.MouseButton.LeftButton, pos=at(c, 24, 14))
    send_move(vp, at(c, 30, 20))
    send_move(vp, at(c, 31, 21), buttons=Qt.MouseButton.NoButton)
    qtbot.mouseRelease(vp, Qt.MouseButton.LeftButton, pos=at(c, 31, 21))
    assert got == [] and (c._rings[0] == RECT).all()


def test_double_click_beside_an_edge_puts_the_point_on_it_when_zoomed_out(qtbot):
    c = Canvas()
    qtbot.addWidget(c)
    c.resize(700, 420)
    c.show()
    qtbot.waitExposed(c)
    c.set_image(np.full((1080, 1920, 3), 90, np.uint8), np.zeros((1080, 1920), np.uint16))
    assert c.transform().m11() < 0.5
    c.mode = "points"
    c.set_rings([np.array([[100, 100], [100, 600], [900, 600], [900, 100]])])
    got = []
    c.point_inserted.connect(lambda *a: got.append(a))
    p = c.mapFromScene(QPointF(500.5, 600.5))
    p.setY(p.y() + 6)  # 6 screen px below the bottom edge: ~17 image px at this zoom
    qtbot.mouseDClick(c.viewport(), Qt.MouseButton.LeftButton, pos=p)
    assert len(got) == 1 and got[0][:2] == (0, 1) and got[0][3] == 600 and abs(got[0][2] - 500) <= 1


def test_unchanged_outlines_are_not_redrawn(qtbot):
    c, _ = points_canvas(qtbot)
    handle = c._handles[0][0]
    c.set_rings([RECT.copy()])
    assert c._handles[0][0] is handle
    c.set_rings([RECT + 1])
    assert c._handles[0][0] is not handle


@pytest.mark.parametrize("mode", ["select", "paint", "points"])
def test_a_right_click_without_a_drag_asks_for_the_part_card(qtbot, mode):
    c = make_canvas(qtbot)
    c.mode = mode
    with qtbot.waitSignal(c.card_requested) as sig:
        qtbot.mouseClick(c.viewport(), Qt.MouseButton.RightButton, pos=at(c, 12, 30))
    assert sig.args[:2] == [12, 30]


def test_in_sam_mode_a_right_click_stays_an_exclude_point(qtbot):
    c = make_canvas(qtbot)
    c.mode = "sam"
    got = []
    c.card_requested.connect(lambda *a: got.append(a))
    qtbot.mouseClick(c.viewport(), Qt.MouseButton.RightButton, pos=at(c, 12, 30))
    assert got == []
