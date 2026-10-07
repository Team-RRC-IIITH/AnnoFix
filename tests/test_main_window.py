import os
import signal
import subprocess
import sys
from pathlib import Path

import pytest
from helpers import LABELS, make_dataset, send_move
from PyQt6.QtCore import QPointF, Qt, QTimer
from PyQt6.QtWidgets import QApplication, QDialog, QMessageBox, QPushButton

from annofix.__main__ import main, run_window
from annofix.io import load_annotation
from annofix.session import Session
from annofix.ui import main_window
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
    assert win.canvas.mode == "paint" and win.tool_actions["brush"].isChecked()
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


def test_hotkeys_work_again_after_typing_a_name_or_class(win, qtbot):
    qtbot.mouseClick(win.canvas.viewport(), Qt.MouseButton.LeftButton, pos=at(win, 10, 10))
    edit = win.labels_panel.name_edit
    qtbot.mouseClick(edit, Qt.MouseButton.LeftButton)
    edit.clear()
    qtbot.keyClicks(edit, "rim")
    qtbot.keyClick(edit, Qt.Key.Key_Return)
    qtbot.keyClick(QApplication.focusWidget(), Qt.Key.Key_3)
    assert win.session.ann.part(2).name == "rim" and win.session.ann.part(2).material == "rubber"
    box = win.labels_panel.class_box.lineEdit()
    qtbot.mouseClick(box, Qt.MouseButton.LeftButton)
    box.clear()
    qtbot.keyClicks(box, "cup")
    qtbot.keyClick(box, Qt.Key.Key_Return)
    qtbot.keyClick(QApplication.focusWidget(), Qt.Key.Key_1)
    assert win.session.ann.object(1).cls == "cup" and win.session.ann.part(2).material == "plastic"


def test_bad_first_annotation_does_not_stop_the_tool(tmp_path, qtbot, monkeypatch):
    from annofix.annotation import Annotation
    from annofix.io import save_annotation

    images, ann_dir = make_dataset(tmp_path)
    save_annotation(Annotation.empty(10, 10), ann_dir, "a")
    warned = []
    monkeypatch.setattr(QMessageBox, "warning", lambda parent, title, text: warned.append(text))
    w = MainWindow(Session(images, ann_dir, LABELS))
    qtbot.addWidget(w)
    w.show()
    qtbot.waitUntil(lambda: bool(warned), timeout=2000)
    assert w.session.stem == "b" and "a.png" in warned[0]


def test_failed_list_click_keeps_the_list_on_the_open_image(win, qtbot, monkeypatch):
    from annofix.annotation import Annotation
    from annofix.io import save_annotation

    save_annotation(Annotation.empty(10, 10), win.ann_dir, "b")
    monkeypatch.setattr(QMessageBox, "warning", lambda *a: None)
    qtbot.mouseClick(win.files.viewport(), Qt.MouseButton.LeftButton,
                     pos=win.files.visualItemRect(win.files.item(1)).center())
    assert win.session.stem == "a" and win.files.currentRow() == 0


def test_zero_key_sets_the_tenth_material(tmp_path, qtbot, monkeypatch):
    from helpers import MATERIALS10

    from annofix.config import Labels

    images, ann_dir = make_dataset(tmp_path)
    labels = Labels(materials=tuple(MATERIALS10), objects={"mug": ("body", "handle")})
    w = MainWindow(Session(images, ann_dir, labels))
    qtbot.addWidget(w)
    w.show()
    qtbot.waitExposed(w)
    w.activateWindow()
    qtbot.waitUntil(lambda: QApplication.activeWindow() is w, timeout=2000)
    qtbot.mouseClick(w.canvas.viewport(), Qt.MouseButton.LeftButton, pos=at(w, 10, 10))
    qtbot.keyClick(w.canvas, Qt.Key.Key_0)
    assert w.session.ann.part(2).material == "wood"
    assert w.labels_panel.material_buttons[9].text().startswith("0")


def click(w, qtbot, x, y):
    qtbot.mouseClick(w.canvas.viewport(), Qt.MouseButton.LeftButton, pos=at(w, x, y))


def complete_labels(s):
    s.ann.set_part_material(3, "metal")
    s.ann.set_object_class(4, "mug")
    s.ann.set_part_name(5, "hull")
    s.ann.set_part_material(5, "metal")


def test_running_out_of_ids_is_a_warning_not_a_crash(win, qtbot, monkeypatch):
    warned = []
    monkeypatch.setattr(QMessageBox, "warning", lambda parent, title, text: warned.append(text))
    win.session.ann.next_id = 65536  # every uint16 part id is taken
    click(win, qtbot, 10, 10)
    qtbot.keyClick(win.canvas, Qt.Key.Key_N)
    win.tree.promote_requested.emit(3)
    assert len(warned) == 2 and all("id space exhausted" in t for t in warned)


def test_clicking_the_open_image_in_the_list_does_not_reload_it(win, qtbot):
    click(win, qtbot, 10, 10)
    win.session.set_hidden("part", 5, True)
    qtbot.mouseClick(win.files.viewport(), Qt.MouseButton.LeftButton,
                     pos=win.files.visualItemRect(win.files.item(0)).center())
    assert win.session.selected == [2] and win.session.hidden_parts == {5}


def test_space_in_a_dialog_is_left_to_the_dialog(win, qtbot):
    dialog = QDialog(win)
    button = QPushButton("OK", dialog)
    pressed = []
    button.pressed.connect(lambda: pressed.append(True))
    dialog.show()
    qtbot.waitExposed(dialog)
    qtbot.keyPress(button, Qt.Key.Key_Space)
    assert pressed and win.canvas.overlay.isVisible()
    qtbot.keyRelease(button, Qt.Key.Key_Space)
    dialog.close()


def test_ctrl_c_in_the_terminal_saves_and_quits(tmp_path, qtbot, monkeypatch):
    images, ann_dir = make_dataset(tmp_path)
    monkeypatch.setattr(sys, "excepthook", sys.excepthook)
    s = Session(images, ann_dir, LABELS)
    s.open(0)
    s.select_at(10, 10)
    s.set_material_index(0)  # unsaved
    fallback = []  # if run_window had no Ctrl+C handler, this one would run instead

    def not_handled(*_):
        fallback.append(True)
        QApplication.closeAllWindows()

    previous = signal.signal(signal.SIGINT, not_handled)
    try:
        QTimer.singleShot(300, lambda: os.kill(os.getpid(), signal.SIGINT))
        QTimer.singleShot(8000, QApplication.closeAllWindows)  # never hang the test run
        rc = run_window(s, ann_dir, None)
    finally:
        signal.signal(signal.SIGINT, previous)
    assert rc == 0 and not fallback
    assert load_annotation(ann_dir, "a").part(2).material == "plastic"
    assert signal.getsignal(signal.SIGINT) is previous


def test_ctrl_c_during_a_slot_closes_the_window_the_normal_way(win, tmp_path, monkeypatch):
    shown = []
    monkeypatch.setattr(QMessageBox, "critical", lambda *a: shown.append(a[2]))
    monkeypatch.setattr(sys, "excepthook", sys.excepthook)
    hook = install_excepthook(win.session, tmp_path / "errors.log")
    win.session.select_at(10, 10)
    win.session.set_material_index(0)
    try:
        raise KeyboardInterrupt
    except KeyboardInterrupt:
        hook(*sys.exc_info())
    assert not shown and not win.isVisible()
    assert load_annotation(win.ann_dir, "a").part(2).material == "plastic"


def test_crash_handler_saves_even_if_the_window_cannot_redraw(win, tmp_path, monkeypatch):
    shown = []
    monkeypatch.setattr(QMessageBox, "critical", lambda *a: shown.append(a[2]))
    monkeypatch.setattr(sys, "excepthook", sys.excepthook)
    hook = install_excepthook(win.session, tmp_path / "errors.log")
    s = win.session
    s.select_at(10, 10)
    s.set_material_index(0)
    s.set_tool("brush")
    s.begin_stroke(30, 35)
    s.stroke_to(50, 35)

    def broken_redraw(kind):
        raise RuntimeError("redraw failed")

    s.subscribe(broken_redraw)  # whatever crashed may crash again on the next notification
    try:
        raise RuntimeError("boom")
    except RuntimeError:
        hook(*sys.exc_info())
    back = load_annotation(win.ann_dir, "a")
    assert back.part(2).material == "plastic" and back.part_at(35, 40) is None
    assert "saved" in shown[0]


def test_buttons_and_esc_give_the_keyboard_back_to_the_canvas(win, qtbot):
    lp = win.labels_panel
    click(win, qtbot, 10, 10)
    qtbot.mouseClick(lp.name_edit, Qt.MouseButton.LeftButton)
    qtbot.mouseClick(lp.material_buttons[1], Qt.MouseButton.LeftButton)
    assert QApplication.focusWidget() is win.canvas and win.session.ann.part(2).material == "metal"
    qtbot.mouseClick(lp.name_edit, Qt.MouseButton.LeftButton)
    qtbot.mouseClick(lp.name_buttons[1], Qt.MouseButton.LeftButton)
    assert QApplication.focusWidget() is win.canvas and win.session.ann.part(2).name == "handle"
    qtbot.mouseClick(lp.name_edit, Qt.MouseButton.LeftButton)
    qtbot.keyClicks(lp.name_edit, "zzz")
    qtbot.keyClick(lp.name_edit, Qt.Key.Key_Escape)  # only leaves the field; the typing is dropped
    assert QApplication.focusWidget() is win.canvas and win.session.selected == [2]
    assert win.session.ann.part(2).name == "handle" and lp.name_edit.text() == "handle"
    qtbot.mouseClick(lp.class_box.lineEdit(), Qt.MouseButton.LeftButton)
    lp.class_box.showPopup()
    lp.class_box.hidePopup()
    assert QApplication.focusWidget() is win.canvas


def test_typed_name_and_class_are_kept_when_focus_moves_away(win, qtbot):
    lp = win.labels_panel
    click(win, qtbot, 10, 10)
    qtbot.mouseClick(lp.name_edit, Qt.MouseButton.LeftButton)
    lp.name_edit.clear()
    qtbot.keyClicks(lp.name_edit, "rim")
    box = lp.class_box.lineEdit()
    qtbot.mouseClick(box, Qt.MouseButton.LeftButton)  # focus moves on without Enter
    assert win.session.ann.part(2).name == "rim"
    box.clear()
    qtbot.keyClicks(box, "cup")
    win.canvas.setFocus()
    assert win.session.ann.object(1).cls == "cup"


def test_edits_are_saved_soon_after_the_last_one_but_never_mid_stroke(win, qtbot):
    assert win._save_soon.interval() == main_window.SAVE_SOON_MS == 2000
    win._save_soon.setInterval(50)
    s = win.session
    s.select_at(10, 10)
    s.set_material_index(0)
    qtbot.waitUntil(lambda: not s.ann.dirty, timeout=2000)
    assert load_annotation(win.ann_dir, "a").part(2).material == "plastic"
    s.set_tool("brush")
    s.begin_stroke(30, 35)
    s.stroke_to(40, 35)
    qtbot.wait(200)
    assert s.ann.in_transaction and load_annotation(win.ann_dir, "a").part_at(35, 35) is None
    s.end_stroke()
    qtbot.waitUntil(lambda: not s.ann.dirty, timeout=2000)
    assert load_annotation(win.ann_dir, "a").part_at(35, 35) == 2


def test_f1_and_question_mark_show_the_shortcuts(win, qtbot, monkeypatch):
    shown = []
    monkeypatch.setattr(QMessageBox, "information", lambda parent, title, text: shown.append(text))
    qtbot.keyClick(win.canvas, Qt.Key.Key_F1)
    qtbot.keyClick(win.canvas, "?")
    assert len(shown) == 2 and "Brush" in shown[0] and "Ctrl+N" in shown[0]


def test_title_shows_the_file_and_review_progress(win, qtbot):
    assert "a.png" in win.windowTitle() and "reviewed 0/3" in win.windowTitle()
    complete_labels(win.session)
    qtbot.keyClick(win.canvas, Qt.Key.Key_Return, Qt.KeyboardModifier.ControlModifier)
    assert "reviewed 1/3" in win.windowTitle() and "reviewed 1/3" in win.status.text()


def test_ctrl_enter_marks_reviewed_and_opens_the_next_unreviewed(win, qtbot):
    complete_labels(win.session)
    qtbot.keyClick(win.canvas, Qt.Key.Key_Return, Qt.KeyboardModifier.ControlModifier)
    assert win.session.stem == "b" and load_annotation(win.ann_dir, "a").review_status == "reviewed"


def test_keypad_digits_set_materials(win, qtbot):
    click(win, qtbot, 10, 10)
    qtbot.keyClick(win.canvas, Qt.Key.Key_3, Qt.KeyboardModifier.KeypadModifier)
    assert win.session.ann.part(2).material == "rubber"


def test_labelling_hotkeys(win, qtbot):
    s = win.session
    qtbot.keyClick(win.canvas, Qt.Key.Key_P)  # P: the first part missing a label
    assert s.selected == [3]
    qtbot.keyClick(win.canvas, Qt.Key.Key_2)
    s.select_part(5)
    s.set_part_name("hull")
    qtbot.keyClick(win.canvas, Qt.Key.Key_3)
    s.select_part(3)
    qtbot.keyClick(win.canvas, Qt.Key.Key_Period)  # '.': the last name and material again
    assert (s.ann.part(3).name, s.ann.part(3).material) == ("hull", "rubber")
    qtbot.keyClick(win.canvas, Qt.Key.Key_N, Qt.KeyboardModifier.ControlModifier)
    new = s.primary
    assert new not in (2, 3, 5) and s.ann.object_of(new).id == 1 and s.tool == "brush"
    qtbot.keyClick(win.canvas, Qt.Key.Key_S, Qt.KeyboardModifier.ControlModifier)
    assert s.messages[-1] == "saved" and not s.ann.dirty
