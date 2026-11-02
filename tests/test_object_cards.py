"""The objects list as CVAT-style cards: one card per object, one row per part."""
from helpers import LABELS, scene_annotation
from PyQt6.QtCore import QPoint, QPointF, Qt
from PyQt6.QtGui import QWheelEvent
from PyQt6.QtWidgets import QApplication, QMenu

from annofix.ui.object_cards import ObjectCards


def make_cards(qtbot, selected=(3,), hidden_parts=frozenset({5}), ann=None):
    c = ObjectCards(LABELS)
    qtbot.addWidget(c)
    c.resize(360, 500)
    c.show()
    qtbot.waitExposed(c)
    c.rebuild(ann or scene_annotation(), list(selected), set(hidden_parts), set())
    return c


def test_one_card_per_object_and_one_row_per_part(qtbot):
    c = make_cards(qtbot)
    assert [card.object_id for card in c.cards] == [1, 4]
    mug, ship = c.cards
    assert [r.part_id for r in mug.rows] == [2, 3] and "mug" in mug.title.text()
    assert "handle" in mug.rows[1].name.text() and mug.rows[1].material.currentText() == "?"
    assert mug.rows[0].material.currentText() == "metal"
    assert mug.expanded and not ship.expanded               # the selection's object is open
    assert mug.rows[1].property("selected") and not mug.rows[0].property("selected")
    assert not ship.rows[0].eye.isChecked() and mug.eye.isChecked()   # part 5 is hidden


def test_clicking_a_row_selects_its_part_and_ctrl_adds(qtbot):
    c = make_cards(qtbot)
    got = []
    c.part_clicked.connect(lambda pid, add: got.append((pid, add)))
    row = c.cards[0].rows[1]
    qtbot.mouseClick(row, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.ControlModifier, QPoint(5, 5))
    qtbot.mouseClick(row, Qt.MouseButton.LeftButton, pos=QPoint(5, 5))
    qtbot.mouseClick(c.cards[1].header, Qt.MouseButton.LeftButton, pos=QPoint(5, 5))
    assert got == [(3, True), (3, False), (5, False)]       # a header selects the object's first part


def test_the_eye_hides_without_selecting(qtbot):
    c = make_cards(qtbot)
    selected, visibility = [], []
    c.part_clicked.connect(lambda pid, add: selected.append(pid))
    c.visibility_changed.connect(lambda kind, i, visible: visibility.append((kind, i, visible)))
    c.cards[0].eye.click()
    c.cards[0].rows[0].eye.click()
    assert visibility == [("object", 1, False), ("part", 2, False)] and selected == []


def test_choosing_a_material_in_a_row_sets_it_for_that_part(qtbot):
    c = make_cards(qtbot)
    got = []
    c.material_set.connect(lambda pid, m: got.append((pid, m)))
    combo = c.cards[0].rows[1].material
    combo.setCurrentText("rubber")
    combo.activated.emit(combo.currentIndex())
    assert got == [(3, "rubber")]


def test_the_wheel_scrolls_the_list_and_never_changes_a_material(qtbot):
    c = make_cards(qtbot)
    combo = c.cards[0].rows[0].material
    before = combo.currentText()
    ev = QWheelEvent(QPointF(5, 5), QPointF(combo.mapToGlobal(QPoint(5, 5))), QPoint(0, 0), QPoint(0, -120),
                     Qt.MouseButton.NoButton, Qt.KeyboardModifier.NoModifier, Qt.ScrollPhase.NoScrollPhase, False)
    QApplication.sendEvent(combo, ev)
    assert combo.currentText() == before


def test_the_chevron_opens_a_card_and_it_stays_open_after_a_rebuild(qtbot):
    c = make_cards(qtbot)
    c.cards[1].chevron.click()
    assert c.cards[1].expanded
    c.rebuild(scene_annotation(), [3], {5}, set())
    assert c.cards[1].expanded


def test_the_context_menu_moves_promotes_and_deletes(qtbot):
    c = make_cards(qtbot)
    got = []
    c.move_requested.connect(lambda pid, oid: got.append(("move", pid, oid)))
    c.promote_requested.connect(lambda pid: got.append(("promote", pid)))
    c.delete_object_requested.connect(lambda oid: got.append(("delete object", oid)))
    menu = c._build_menu("part", 3)
    move_menu = menu.findChildren(QMenu)[0]
    next(a for a in move_menu.actions() if a.text().startswith("#4")).trigger()
    next(a for a in menu.actions() if a.text() == "Make it a new object").trigger()
    c._build_menu("object", 4).actions()[0].trigger()
    assert got == [("move", 3, 4), ("promote", 3), ("delete object", 4)]


def test_cards_show_the_pipeline_confidence(qtbot):
    ann = scene_annotation()
    ann.part(3).attrs["confidence_pct"] = 49
    ann.object(1).attrs["confidence_pct"] = 72
    c = make_cards(qtbot, ann=ann)
    mug = c.cards[0]
    assert mug.confidence.text() == "72%" and mug.rows[1].confidence.text() == "49%"
    assert mug.rows[0].confidence.text() == ""


def test_nothing_in_the_cards_takes_the_keyboard(qtbot):
    from PyQt6.QtWidgets import QWidget
    c = make_cards(qtbot)
    assert all(w.focusPolicy() == Qt.FocusPolicy.NoFocus for w in c.findChildren(QWidget) if w.isVisible())


def test_a_click_that_rebuilds_the_list_does_not_crash(qtbot):
    c = make_cards(qtbot)
    ann = scene_annotation()
    c.part_clicked.connect(lambda pid, add: c.rebuild(ann, [pid], set(), set()))  # what the window does
    qtbot.mouseClick(c.cards[0].rows[0], Qt.MouseButton.LeftButton, pos=QPoint(5, 5))
    qtbot.mouseClick(c.cards[0].rows[1], Qt.MouseButton.LeftButton, pos=QPoint(5, 5))
    assert c.cards[0].rows[1].property("selected")
