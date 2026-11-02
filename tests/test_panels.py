from helpers import LABELS, scene_annotation
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QMenu

from annofix.ui.label_panel import LabelPanel


def test_label_panel_emits_material_name_and_class(qtbot):
    p = LabelPanel(LABELS)
    qtbot.addWidget(p)
    ann = scene_annotation()
    p.show_part(ann.part(2), ann.object(1))
    got = []
    p.material_chosen.connect(lambda m: got.append(("material", m)))
    p.part_name_entered.connect(lambda n: got.append(("name", n)))
    p.class_entered.connect(lambda c: got.append(("class", c)))
    qtbot.mouseClick(p.material_buttons[1], Qt.MouseButton.LeftButton)
    assert [b.text() for b in p.name_buttons] == ["body", "handle"]
    qtbot.mouseClick(p.name_buttons[1], Qt.MouseButton.LeftButton)
    p.name_edit.setText("rim")
    qtbot.keyClick(p.name_edit, Qt.Key.Key_Return)
    p.class_box.setEditText("cup")
    qtbot.keyClick(p.class_box.lineEdit(), Qt.Key.Key_Return)
    assert got[:3] == [("material", "metal"), ("name", "handle"), ("name", "rim")]
    assert ("class", "cup") in got[3:]
    assert "Part #2" in p.title.text()


def test_label_panel_disabled_without_selection(qtbot):
    p = LabelPanel(LABELS)
    qtbot.addWidget(p)
    p.show_part(None, None)
    assert not p.material_buttons[0].isEnabled() and p.name_buttons == []


def test_panel_buttons_never_take_keyboard_focus(qtbot):
    p = LabelPanel(LABELS)
    qtbot.addWidget(p)
    ann = scene_annotation()
    p.show_part(ann.part(2), ann.object(1))
    assert all(b.focusPolicy() == Qt.FocusPolicy.NoFocus for b in p.material_buttons + p.name_buttons)
    assert "1" in p.material_buttons[0].text()
