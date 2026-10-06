from helpers import LABELS, scene_annotation
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QMenu

from annofix.ui.label_panel import LabelPanel
from annofix.ui.tree_panel import ObjectTree


def make_tree(qtbot, selected=(3,), hidden_parts=frozenset({5})):
    t = ObjectTree()
    qtbot.addWidget(t)
    t.rebuild(scene_annotation(), list(selected), set(hidden_parts), set())
    return t


def test_tree_lists_objects_and_parts_with_visibility(qtbot):
    t = make_tree(qtbot)
    assert t.topLevelItemCount() == 2
    mug = t.topLevelItem(0)
    assert mug.childCount() == 2 and mug.isExpanded() and not t.topLevelItem(1).isExpanded()
    assert "handle" in mug.child(1).text(0) and mug.child(1).font(0).bold()
    assert t.topLevelItem(1).child(0).checkState(0) == Qt.CheckState.Unchecked


def test_tree_checkbox_emits_visibility(qtbot):
    t = make_tree(qtbot)
    with qtbot.waitSignal(t.visibility_changed) as sig:
        t.topLevelItem(0).setCheckState(0, Qt.CheckState.Unchecked)
    assert sig.args == ["object", 1, False]


def test_tree_click_selects_part_or_first_part_of_object(qtbot):
    t = make_tree(qtbot)
    got = []
    t.part_clicked.connect(lambda pid, add: got.append((pid, add)))
    t.itemClicked.emit(t.topLevelItem(0).child(1), 0)
    t.itemClicked.emit(t.topLevelItem(1), 0)
    assert got == [(3, False), (5, False)]


def test_tree_context_menu_actions(qtbot):
    t = make_tree(qtbot)
    got = []
    t.move_requested.connect(lambda pid, oid: got.append(("move", pid, oid)))
    t.promote_requested.connect(lambda pid: got.append(("promote", pid)))
    t.delete_object_requested.connect(lambda oid: got.append(("delete object", oid)))
    menu = t._build_menu(t.topLevelItem(0).child(1))
    move_menu = menu.findChildren(QMenu)[0]
    next(a for a in move_menu.actions() if a.text().startswith("#4")).trigger()
    next(a for a in menu.actions() if a.text() == "Make it a new object").trigger()
    t._build_menu(t.topLevelItem(1)).actions()[0].trigger()
    assert got == [("move", 3, 4), ("promote", 3), ("delete object", 4)]


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


def test_tree_ctrl_click_adds_to_selection(qtbot):
    t = make_tree(qtbot)
    t.resize(300, 300)
    t.show()
    qtbot.waitExposed(t)
    got = []
    t.part_clicked.connect(lambda pid, add: got.append((pid, add)))
    pos = t.visualItemRect(t.topLevelItem(0).child(1)).center()
    qtbot.mouseClick(t.viewport(), Qt.MouseButton.LeftButton, Qt.KeyboardModifier.ControlModifier, pos)
    qtbot.mouseClick(t.viewport(), Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, pos)
    assert got == [(3, True), (3, False)]


def check_box_pos(tree, item):
    from PyQt6.QtWidgets import QStyle, QStyleOptionViewItem

    opt = QStyleOptionViewItem()
    tree.initViewItemOption(opt)
    opt.rect = tree.visualItemRect(item)
    opt.features |= QStyleOptionViewItem.ViewItemFeature.HasCheckIndicator
    return tree.style().subElementRect(QStyle.SubElement.SE_ItemViewItemCheckIndicator, opt, tree).center()


def test_ticking_a_box_hides_without_selecting(qtbot):
    t = make_tree(qtbot)
    t.resize(300, 300)
    t.show()
    qtbot.waitExposed(t)
    selected, visibility = [], []
    t.part_clicked.connect(lambda pid, add: selected.append(pid))
    t.visibility_changed.connect(lambda kind, i, visible: visibility.append((kind, i, visible)))
    qtbot.mouseClick(t.viewport(), Qt.MouseButton.LeftButton, pos=check_box_pos(t, t.topLevelItem(1)))
    assert visibility == [("object", 4, False)] and selected == []
    qtbot.mouseClick(t.viewport(), Qt.MouseButton.LeftButton,
                     pos=t.visualItemRect(t.topLevelItem(0).child(1)).center())
    assert selected == [3]  # an ordinary click right after still selects
