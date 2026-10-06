import numpy as np
from helpers import send_move
from PyQt6.QtCore import QPointF, Qt

from annofix.ui.canvas import Canvas
from annofix.ui.overlay_item import OverlayItem


def test_overlay_shows_lut_colour_and_follows_in_place_edits(qtbot):
    lm = np.zeros((4, 6), np.uint16)
    lm[1, 2] = 7
    item = OverlayItem(lm)
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
