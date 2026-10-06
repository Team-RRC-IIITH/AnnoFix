import numpy as np
import pytest
from helpers import LABELS, make_dataset
from PyQt6.QtCore import QPointF, Qt
from PyQt6.QtWidgets import QApplication, QMessageBox

from annofix.__main__ import main
from annofix.sam_backend import FakeBackend
from annofix.session import Session
from annofix.ui.main_window import MainWindow


@pytest.fixture
def win(tmp_path, qtbot, monkeypatch):
    images, ann_dir = make_dataset(tmp_path)
    monkeypatch.setattr(QMessageBox, "information", lambda *a, **k: None)
    monkeypatch.setattr(QMessageBox, "warning", lambda *a, **k: None)
    w = MainWindow(Session(images, ann_dir, LABELS), sam_factory=FakeBackend)
    qtbot.addWidget(w)
    w.resize(1200, 700)
    w.show()
    qtbot.waitExposed(w)
    w.activateWindow()
    qtbot.waitUntil(lambda: QApplication.activeWindow() is w, timeout=2000)
    yield w
    w.sam.shutdown()


def at(w, x, y):
    return w.canvas.mapFromScene(QPointF(x + 0.5, y + 0.5))


def fix_part_2(w, qtbot):
    qtbot.mouseClick(w.canvas.viewport(), Qt.MouseButton.LeftButton, pos=at(w, 10, 10))
    qtbot.keyClick(w.canvas, Qt.Key.Key_R)
    qtbot.waitUntil(lambda: w.session.sam is not None and w.session.sam.mask is not None, timeout=3000)


def test_r_exclude_click_enter_fixes_a_part(win, qtbot):
    s = win.session
    fix_part_2(win, qtbot)
    assert win.canvas.mode == "sam" and win.canvas.sam_preview_item is not None
    assert win.tool_actions["sam"].isChecked() and "candidate" in win.status.text()
    x, y = s.sam.points[0]
    qtbot.mouseClick(win.canvas.viewport(), Qt.MouseButton.RightButton, pos=at(win, x + 7, y))
    qtbot.waitUntil(lambda: s.sam.received_seq == 2, timeout=3000)
    expected = s.sam.mask.copy()
    qtbot.keyClick(win.canvas, Qt.Key.Key_Return)
    assert s.sam is None and np.array_equal(s.ann.part_mask(2), expected)
    assert win.canvas.sam_preview_item is None


def test_tab_cycles_and_esc_cancels(win, qtbot):
    s = win.session
    fix_part_2(win, qtbot)
    index = s.sam.index
    qtbot.keyClick(win.canvas, Qt.Key.Key_Tab)
    assert s.sam.index == (index + 1) % 3 and QApplication.focusWidget() is win.canvas
    qtbot.keyClick(win.canvas, Qt.Key.Key_Escape)
    assert s.sam is None and s.tool == "sam"
    qtbot.keyClick(win.canvas, Qt.Key.Key_Escape)
    assert s.tool == "select"


def test_shift_enter_adds(win, qtbot):
    s = win.session
    fix_part_2(win, qtbot)
    before = s.ann.part_area(2)
    qtbot.keyClick(win.canvas, Qt.Key.Key_Return, Qt.KeyboardModifier.ShiftModifier)
    assert s.ann.part_area(2) > before and s.ann.part_at(5, 5) == 2


def test_window_without_sam_hides_the_tool(tmp_path, qtbot):
    images, ann_dir = make_dataset(tmp_path)
    w = MainWindow(Session(images, ann_dir, LABELS))
    qtbot.addWidget(w)
    assert w.sam is None and not w.tool_actions["sam"].isVisible() and not w.session.sam_enabled


def test_cli_rejects_unknown_sam_model(tmp_path):
    with pytest.raises(SystemExit):
        main(["--images", str(tmp_path), "--annotations", str(tmp_path), "--sam-model", "huge"])
