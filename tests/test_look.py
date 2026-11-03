"""The CVAT-style window: header bar, icon tool bar, side panels. Behaviour only, not pixels."""
import pytest
from helpers import LABELS, make_dataset
from PyQt6.QtCore import QPoint, Qt
from PyQt6.QtWidgets import QApplication

from annofix.session import Session
from annofix.ui.main_window import MainWindow


@pytest.fixture
def win(tmp_path, qtbot):
    images, ann_dir = make_dataset(tmp_path)
    w = MainWindow(Session(images, ann_dir, LABELS))
    qtbot.addWidget(w)
    w.resize(1200, 800)
    w.show()
    qtbot.waitExposed(w)
    return w


def test_header_buttons_step_through_the_images(win):
    assert win.session.stem == "a"
    win.header_actions["next"].trigger()
    assert win.session.stem == "b"
    win.header_actions["prev"].trigger()
    assert win.session.stem == "a"
    assert "1/3" in win.image_label.text() and "a" in win.image_label.text()


def test_header_undo_and_redo(win):
    s = win.session
    s.select_at(10, 10)
    s.set_material_index(0)                            # plastic
    assert s.ann.part(2).material == "plastic"
    win.header_actions["undo"].trigger()
    assert s.ann.part(2).material == "metal"
    win.header_actions["redo"].trigger()
    assert s.ann.part(2).material == "plastic"


def test_header_marks_the_image_reviewed(win, monkeypatch):
    called = []
    monkeypatch.setattr(win, "_mark_reviewed", lambda: called.append(1))
    win.header_actions["reviewed"].trigger()
    assert called == [1]


def test_header_undo_while_typing_a_name_drops_the_typing_and_undoes_the_last_edit(win, qtbot):
    s = win.session
    s.select_at(10, 10)
    s.set_material_index(0)                            # plastic: the edit to undo
    win.labels_panel.name_edit.setText("TYPED")        # typed, not entered
    win.header_actions["undo"].trigger()
    assert s.ann.part(2).material == "metal" and s.ann.part(2).name == "body"


def test_clicking_header_and_tool_buttons_keeps_the_keyboard_on_the_canvas(win, qtbot):
    from PyQt6.QtCore import Qt
    from PyQt6.QtWidgets import QToolBar
    win.canvas.setFocus()
    for bar, action in ((win.findChild(QToolBar, "header"), win.header_actions["next"]),
                        (win.findChild(QToolBar, "tools"), win.tool_actions["brush"])):
        qtbot.mouseClick(bar.widgetForAction(action), Qt.MouseButton.LeftButton)
        assert QApplication.focusWidget() is win.canvas


def test_the_bars_cannot_be_hidden_by_a_right_click(win):
    from PyQt6.QtWidgets import QToolBar
    for name in ("header", "tools"):
        assert not win.findChild(QToolBar, name).toggleViewAction().isVisible()


def test_every_tool_has_an_icon_and_its_key_in_the_tooltip(win):
    keys = {"select": "V", "brush": "B", "eraser": "E", "points": "G", "sam": "R"}
    for tool, action in win.tool_actions.items():
        assert not action.icon().isNull() and f"({keys[tool]})" in action.toolTip()
    for action in win.header_actions.values():
        assert not action.icon().isNull() and action.toolTip()


def test_the_material_of_the_selected_part_is_marked_active(win):
    win.session.select_at(10, 10)                      # part 2: metal
    active = [b.text() for b in win.labels_panel.material_buttons if b.property("active")]
    assert len(active) == 1 and "metal" in active[0]
    assert all(not b.styleSheet() for b in win.labels_panel.material_buttons)  # the window's style rules


def test_the_canvas_cursor_shows_on_white_and_on_dark_pixels(win):
    pens = [win.canvas._cursor.pen()]
    outline = getattr(win.canvas, "_cursor_outline", None)
    assert outline is not None
    colours = {pens[0].color().name(), outline.pen().color().name()}
    assert "#000000" in colours and "#ffffff" in colours


def open_card(win, x, y):
    win.canvas.card_requested.emit(x, y, win.canvas.mapToGlobal(QPoint(40, 40)))
    return win.part_card


def test_a_right_click_on_a_mask_opens_its_card(win):
    card = open_card(win, 10, 10)
    assert card.isVisible() and win.session.selected == [2]
    assert card.panel.name_edit.text() == "body" and "#2" in card.panel.title.text()


def test_a_name_typed_in_the_card_renames_the_part_on_enter_and_closes_it(win, qtbot):
    card = open_card(win, 10, 10)
    card.panel.name_edit.setText("lid")
    qtbot.keyClick(card.panel.name_edit, Qt.Key.Key_Return)
    assert win.session.ann.part(2).name == "lid" and not card.isVisible()


def test_a_material_chosen_in_the_card_applies_and_the_card_stays_open(win):
    card = open_card(win, 10, 10)
    card.panel.material_buttons[2].click()             # rubber
    assert win.session.ann.part(2).material == "rubber" and card.isVisible()
    assert [b.property("active") for b in card.panel.material_buttons] == [False, False, True]


def test_esc_closes_the_card_and_drops_the_typing(win, qtbot):
    card = open_card(win, 10, 10)
    card.panel.name_edit.setText("zzz")
    qtbot.keyClick(card.panel.name_edit, Qt.Key.Key_Escape)
    assert not card.isVisible() and win.session.ann.part(2).name == "body"


def test_a_right_click_on_background_opens_no_card(win):
    assert not open_card(win, 55, 35).isVisible()


def shown_tags(win):
    return sorted(t.text for t in win.canvas.tag_items if t.isVisible())


def test_t_cycles_the_text_on_the_masks_all_then_the_selected_object_then_off(win, qtbot):
    win.session.select_at(10, 10)                      # the mug: body #2 and handle #3
    assert shown_tags(win) == ["? · ?", "body · metal", "handle · ?"]
    qtbot.keyClick(win.canvas, Qt.Key.Key_T)
    assert win.tags_box.currentText() == "selected object" and shown_tags(win) == ["body · metal", "handle · ?"]
    qtbot.keyClick(win.canvas, Qt.Key.Key_T)
    assert win.tags_box.currentText() == "off" and shown_tags(win) == []
    qtbot.keyClick(win.canvas, Qt.Key.Key_T)
    assert win.tags_box.currentText() == "all parts" and len(shown_tags(win)) == 3


def test_selecting_another_object_moves_the_text_with_it(win):
    win.tags_box.setCurrentText("selected object")
    win.session.select_at(10, 25)                      # object 4's only part
    assert shown_tags(win) == ["? · ?"]


def test_the_text_follows_a_rename(win):
    s = win.session
    s.select_at(10, 10)
    s.set_part_name("lid")
    assert "lid · metal" in [t.text for t in win.canvas.tag_items]


def test_closing_stops_the_save_timers(win):
    win.session.select_at(10, 10)
    win.session.set_material_index(0)                  # an edit: "save soon" is running
    assert win._save_soon.isActive()
    win.close()
    assert not win._save_soon.isActive() and not win._autosave.isActive()


def test_l_previews_straightening_the_selected_part_and_enter_applies_it(win, qtbot):
    s = win.session
    s.ann.label_map[14, 5:25:3] = 0                     # a zigzag edge on body #2
    win.canvas.setFocus()
    s.select_at(10, 10)
    qtbot.keyClick(win.canvas, Qt.Key.Key_L)
    assert s.auto_clean is not None and "straighten ready" in win.status.text()
    assert win.canvas.sam_preview_item is not None
    qtbot.keyClick(win.canvas, Qt.Key.Key_Return)
    assert s.auto_clean is None and "straighten:" in win.status.text()


def test_shift_l_previews_every_part_and_esc_drops_it(win, qtbot):
    s = win.session
    win.canvas.setFocus()
    qtbot.keyClick(win.canvas, Qt.Key.Key_L, Qt.KeyboardModifier.ShiftModifier)
    assert s.auto_clean is not None and len(s.auto_clean.report) == 3
    qtbot.keyClick(win.canvas, Qt.Key.Key_Escape)
    assert s.auto_clean is None


def test_in_the_points_tool_a_box_then_l_straightens_that_stretch(win, qtbot):
    s = win.session
    for x in range(6, 24, 2):
        s.ann.label_map[15, x] = 2                     # body #2's bottom edge zigzags
    s.select_at(10, 10)
    s.set_tool("points")
    win.canvas.setFocus()
    qtbot.keyClick(win.canvas, Qt.Key.Key_G, Qt.KeyboardModifier.ShiftModifier)
    assert s.points_exact and len(s.rings[0]) > 15
    win.canvas.points_boxed.emit(3, 13, 26, 16)
    assert s.point_selection is not None and win.canvas._picked is not None
    qtbot.keyClick(win.canvas, Qt.Key.Key_L)
    assert not (s.ann.label_map[15] == 2).any() and (s.ann.label_map[14, 5:25] == 2).all()


def test_esc_in_the_points_tool_first_drops_the_picked_points(win, qtbot):
    s = win.session
    s.select_at(10, 10)
    s.set_tool("points")
    win.canvas.setFocus()
    win.canvas.points_boxed.emit(3, 12, 26, 16)
    qtbot.keyClick(win.canvas, Qt.Key.Key_Escape)
    assert s.point_selection is None and s.tool == "points"


def test_brackets_change_the_point_detail_in_the_points_tool_and_the_brush_elsewhere(win, qtbot):
    s = win.session
    s.select_at(10, 10)
    s.set_tool("points")
    win.canvas.setFocus()
    level, radius = s.point_level, s.brush_radius
    qtbot.keyClick(win.canvas, Qt.Key.Key_BracketRight)
    assert s.point_level == level + 1 and s.brush_radius == radius
    assert s.POINT_LEVELS[s.point_level][0] in win.status.text()
    qtbot.keyClick(win.canvas, Qt.Key.Key_BracketLeft)
    assert s.point_level == level
    s.set_tool("brush")
    qtbot.keyClick(win.canvas, Qt.Key.Key_BracketRight)
    assert s.brush_radius > radius and s.point_level == level


def alpha_at(win, x, y):
    return win.canvas.overlay.argb_at(x, y) >> 24


def test_a_selected_part_is_outlined_and_the_others_fade(win):
    s = win.session
    assert alpha_at(win, 10, 25) == 255 and not win.canvas.selection_items   # nothing selected
    s.select_at(10, 10)                                                      # body #2
    assert win.canvas.selection_items                                        # its outline
    assert alpha_at(win, 10, 10) == 255 and alpha_at(win, 10, 25) < 200      # part #5 fades
    win.others_slider.setValue(100)
    assert alpha_at(win, 10, 25) == 255                                      # 100%: no fading
    s.select_part(None)
    assert not win.canvas.selection_items and alpha_at(win, 10, 25) == 255


def test_the_outline_follows_an_edit(win):
    s = win.session
    s.select_at(10, 10)
    before = win.canvas.selection_items[0].path().boundingRect()
    s.ann.label_map[5:15, 25:29] = 2          # body #2 grows right
    s._changed()
    after = win.canvas.selection_items[0].path().boundingRect()
    assert after.right() > before.right()


def test_moving_the_others_slider_does_not_trace_the_outline_again(win):
    win.session.select_at(10, 10)
    items = list(win.canvas.selection_items)
    win.others_slider.setValue(70)
    assert win.canvas.selection_items == items                 # same outline items: not re-traced
    assert alpha_at(win, 10, 25) == round(0.7 * 255)


def test_opened_cards_are_forgotten_on_another_image(win):
    win.tree.cards[1].chevron.click()                         # open object 4's card on image a
    assert win.tree.cards[1].expanded
    win.session.step(1)                                       # image b: ids there are other objects
    win.session.step(-1)
    assert not any(c.expanded for c in win.tree.cards)
