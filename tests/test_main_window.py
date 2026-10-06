import os
import subprocess
import sys
from pathlib import Path

import pytest
from helpers import LABELS, make_dataset, send_move
from PyQt6.QtCore import QPointF, Qt
from PyQt6.QtWidgets import QApplication, QMessageBox

from annofix.__main__ import main
from annofix.io import load_annotation
from annofix.session import Session
from annofix.ui.main_window import MainWindow, install_excepthook

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture
def win(tmp_path, qtbot, monkeypatch):
    images, ann_dir = make_dataset(tmp_path)
    monkeypatch.setattr(QMessageBox, "information", lambda *a, **k: None)
    monkeypatch.setattr(QMessageBox, "warning", lambda *a, **k: None)
    w = MainWindow(Session(images, ann_dir, LABELS))
    qtbot.addWidget(w)
    w.resize(1200, 700)
    w.show()
    qtbot.waitExposed(w)
    w.activateWindow()
    qtbot.waitUntil(lambda: QApplication.activeWindow() is w, timeout=2000)
    w.ann_dir = ann_dir
    return w


def at(w, x, y):
    return w.canvas.mapFromScene(QPointF(x + 0.5, y + 0.5))


def test_opens_first_image_with_overlay_and_tree(win):
    assert win.session.stem == "a"
    assert win.canvas.overlay.argb_at(10, 10) != 0 and win.canvas.overlay.argb_at(50, 35) == 0
    assert win.tree.topLevelItemCount() == 2
    assert win.files.item(0).text().startswith("•")


def test_click_then_hotkey_sets_material(win, qtbot):
    qtbot.mouseClick(win.canvas.viewport(), Qt.MouseButton.LeftButton, pos=at(win, 10, 10))
    assert win.session.selected == [2]
    qtbot.keyClick(win.canvas, Qt.Key.Key_3)
    assert win.session.ann.part(2).material == "rubber"
    assert "rubber" in win.tree.topLevelItem(0).child(0).text(0)


def test_brush_stroke_and_undo_through_the_window(win, qtbot):
    qtbot.mouseClick(win.canvas.viewport(), Qt.MouseButton.LeftButton, pos=at(win, 10, 10))
    qtbot.keyClick(win.canvas, Qt.Key.Key_B)
    assert win.canvas.paint_mode and win.tool_actions["brush"].isChecked()
    win.session.set_brush_radius(1)
    vp = win.canvas.viewport()
    qtbot.mousePress(vp, Qt.MouseButton.LeftButton, pos=at(win, 30, 35))
    send_move(vp, at(win, 50, 35))
    qtbot.mouseRelease(vp, Qt.MouseButton.LeftButton, pos=at(win, 50, 35))
    assert (win.session.ann.label_map[35, 30:51] == 2).all()
    assert win.canvas.overlay.argb_at(40, 35) != 0
    qtbot.keyClick(win.canvas, Qt.Key.Key_Z, Qt.KeyboardModifier.ControlModifier)
    assert not win.session.ann.label_map[35, 30:51].any() and win.canvas.overlay.argb_at(40, 35) == 0


def test_navigation_saves_and_switches_resolution(win, qtbot):
    qtbot.mouseClick(win.canvas.viewport(), Qt.MouseButton.LeftButton, pos=at(win, 10, 10))
    qtbot.keyClick(win.canvas, Qt.Key.Key_1)
    qtbot.keyClick(win.canvas, Qt.Key.Key_D)
    qtbot.keyClick(win.canvas, Qt.Key.Key_D)
    assert win.session.stem == "c" and win.canvas.overlay.boundingRect().width() == 50
    assert load_annotation(win.ann_dir, "a").part(2).material == "plastic"
    assert win.files.item(1).text().startswith("○")


def test_space_hides_masks_while_held(win, qtbot):
    qtbot.keyPress(win.canvas, Qt.Key.Key_Space)
    assert not win.canvas.overlay.isVisible()
    qtbot.keyRelease(win.canvas, Qt.Key.Key_Space)
    assert win.canvas.overlay.isVisible()


def test_mark_reviewed_refuses_incomplete_and_lists_problems(win, qtbot, monkeypatch):
    shown = []
    monkeypatch.setattr(QMessageBox, "information", lambda parent, title, text: shown.append(text))
    qtbot.keyClick(win.canvas, Qt.Key.Key_Return, Qt.KeyboardModifier.ControlModifier)
    assert win.session.ann.review_status == "unreviewed" and "no material" in shown[0]


def test_crash_handler_saves_and_logs(win, tmp_path, monkeypatch):
    shown = []
    monkeypatch.setattr(QMessageBox, "critical", lambda *a: shown.append(a[2]))
    monkeypatch.setattr(sys, "excepthook", sys.excepthook)
    log = tmp_path / "errors.log"
    hook = install_excepthook(win.session, log)
    s = win.session
    s.select_at(10, 10)
    s.set_material_index(0)                 # a finished, unsaved edit
    s.set_tool("brush")
    s.begin_stroke(30, 35)
    s.stroke_to(50, 35)                     # crash in the middle of a stroke
    try:
        raise RuntimeError("boom")
    except RuntimeError:
        hook(*sys.exc_info())
    back = load_annotation(win.ann_dir, "a")
    assert back.part(2).material == "plastic"   # the finished edit was saved
    assert back.part_at(35, 40) is None         # the half stroke was rolled back, not saved
    assert "boom" in log.read_text() and "saved" in shown[0]


def test_cli_reports_bad_paths(tmp_path, capsys):
    rc = main(["--images", str(tmp_path / "missing"), "--annotations", str(tmp_path / "ann"),
               "--labels", str(REPO / "labels.yaml")])
    assert rc == 1 and "annofix:" in capsys.readouterr().err


def test_gui_path_does_not_import_torch():
    code = ("import sys, annofix.__main__, annofix.session, annofix.ui.main_window; "
            "assert 'torch' not in sys.modules")
    subprocess.run([sys.executable, "-c", code], check=True,
                   env={**os.environ, "QT_QPA_PLATFORM": "offscreen"})
