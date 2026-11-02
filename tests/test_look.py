"""The CVAT-style window: header bar, icon tool bar, side panels. Behaviour only, not pixels."""
import pytest
from helpers import LABELS, make_dataset
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
    s = win.session
    monkeypatch.setattr(s, "validate", lambda: [], raising=False)
    called = []
    monkeypatch.setattr(win, "_mark_reviewed", lambda: called.append(1))
    win.header_actions["reviewed"].trigger()
    assert called == [1]


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


def test_the_window_uses_the_light_cvat_style(win):
    assert "#1890ff" in win.styleSheet()
    assert QApplication.instance() is not None
