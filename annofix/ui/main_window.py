"""The AnnoFix window: images | canvas | objects tree + labels. Every action goes through the
Session; this module only maps input to Session calls and redraws on its notifications."""
from __future__ import annotations

import sys
import time
import traceback
from pathlib import Path

from PyQt6.QtCore import QEvent, Qt, QTimer
from PyQt6.QtGui import QAction, QActionGroup, QKeySequence, QShortcut
from PyQt6.QtWidgets import (QApplication, QComboBox, QLabel, QLineEdit, QListWidget, QMainWindow,
                             QMessageBox, QSlider, QSplitter, QToolBar, QVBoxLayout, QWidget)

from ..session import COLOUR_MODES, Session
from .canvas import Canvas
from .colors import build_lut
from .label_panel import LabelPanel
from .sam_worker import SamController
from .tree_panel import ObjectTree

STATUS_ICONS = {"none": "○", "unreviewed": "•", "reviewed": "✓", "flagged": "⚑"}
AUTOSAVE_MS = 30_000  # backstop
SAVE_SOON_MS = 2_000  # after the last finished edit

SHORTCUTS = (  # F1 / ? help, in the order you need them
    ("click · Ctrl+click", "select a part · add/remove one (canvas or tree)"),
    ("1–9, 0", "set the material (or a material button)"),
    ("P", "select the next part with a missing label"),
    (".", "give the selected parts the last name and material again"),
    ("type + Enter", "part name or object class (applies to every selected part)"),
    ("B · E · V", "Brush (paint the selected part) · eraser · select"),
    ("G", "Points: drag the part's outline points (double-click an edge adds one, Ctrl+click deletes)"),
    ("[ · ]", "brush size"),
    ("Esc", "cancel a stroke or SAM fix · leave a text field · deselect"),
    ("R", "improve the selected part with SAM (from its own mask; Tab: other answers, then smoothed)"),
    ("Esc, then SAM click", "redraw a part that is plain wrong from your clicks"),
    ("Shift+R", "auto-clean every mask in the image (SAM, else smoothing); Enter accepts, Esc discards"),
    ("SAM: left · right click · Shift+drag", "include · exclude · box"),
    ("SAM: Tab · Shift+Tab", "next · previous candidate"),
    ("SAM: Enter · Shift+Enter", "replace the part · add to it"),
    ("N · Shift+N", "new part · new object (with SAM: from clicks)"),
    ("Ctrl+N", "new empty part, brush ready (also with SAM)"),
    ("M · Shift+M", "merge selected parts · merge their objects"),
    ("Del · Shift+Del", "delete part(s) · delete object"),
    ("Ctrl+Z · Ctrl+Shift+Z", "undo · redo"),
    ("Space (hold) · S · O", "hide masks · solo object · outlines only"),
    ("J · K", "next · previous object"),
    ("A · D · Ctrl+U", "previous · next image · next unreviewed"),
    ("Ctrl+Enter", "mark reviewed and open the next unreviewed image"),
    ("F", "flag the image"),
    ("Ctrl+S", "save now (it also saves by itself)"),
    ("wheel · middle- or right-drag · Home", "zoom · pan · fit the image"),
    ("F1 · ?", "this list"),
)


class MainWindow(QMainWindow):
    def __init__(self, session: Session, sam_factory=None):
        super().__init__()
        self.session = session
        self.setWindowTitle("AnnoFix")
        self.canvas = Canvas()
        self.tree = ObjectTree()
        self.labels_panel = LabelPanel(session.labels)
        self.files = QListWidget()
        self.files.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        for p in session.images:
            self.files.addItem(p.name)
        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.addWidget(self.tree, 3)
        right_layout.addWidget(self.labels_panel, 2)
        split = QSplitter()
        split.addWidget(self.files)
        split.addWidget(self.canvas)
        split.addWidget(right)
        split.setSizes([180, 900, 340])
        self.setCentralWidget(split)
        self.status = QLabel()
        self.statusBar().addWidget(self.status, 1)
        self._build_toolbar()
        self._build_shortcuts()
        self._connect()
        self.sam: SamController | None = None
        if sam_factory is not None:
            self.sam = SamController(session, sam_factory, self)
            self.sam.status_changed.connect(lambda _text: self._on_sam_status())
        self._sync_sam_action()
        self.sam_still_running = False  # set on close if the SAM thread could not stop in time
        self._autosave = QTimer(self)
        self._autosave.timeout.connect(lambda: self._guard(self.session.save))
        self._autosave.start(AUTOSAVE_MS)
        self._save_soon = QTimer(self)  # restarted by every finished edit
        self._save_soon.setSingleShot(True)
        self._save_soon.setInterval(SAVE_SOON_MS)
        self._save_soon.timeout.connect(lambda: self._guard(self.session.save))
        QApplication.instance().installEventFilter(self)  # hold Space to hide the masks
        session.subscribe(self._on_change)
        if session.index < 0:
            skipped = session.open_first()  # raises ValueError if no image opens at all
            if skipped:
                QTimer.singleShot(0, lambda: QMessageBox.warning(
                    self, "AnnoFix", "Some images could not be opened and were skipped:\n" + "\n".join(skipped)))
        else:
            self._on_change("image")
        self.canvas.setFocus()

    # ---- construction ---------------------------------------------------------------------

    def _build_toolbar(self) -> None:
        s = self.session
        tb = QToolBar("Tools")
        tb.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.addToolBar(tb)
        group = QActionGroup(self)
        self.tool_actions: dict[str, QAction] = {}
        for tool, text in (("select", "Select (V)"), ("brush", "Brush (B)"), ("eraser", "Eraser (E)"),
                           ("points", "Points (G)"), ("sam", "SAM (R)")):
            action = QAction(text, self, checkable=True)
            action.triggered.connect(lambda checked=False, t=tool: s.set_tool(t))
            group.addAction(action)
            tb.addAction(action)
            self.tool_actions[tool] = action
        tb.addSeparator()
        tb.addWidget(QLabel(" Colour by "))
        self.mode_box = QComboBox()
        self.mode_box.addItems(COLOUR_MODES)
        self.mode_box.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.mode_box.currentTextChanged.connect(s.set_colour_mode)
        tb.addWidget(self.mode_box)
        tb.addWidget(QLabel(" Opacity "))
        self.opacity = QSlider(Qt.Orientation.Horizontal)
        self.opacity.setRange(0, 100)
        self.opacity.setValue(55)
        self.opacity.setFixedWidth(120)
        self.opacity.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.opacity.valueChanged.connect(self._apply_opacity)
        tb.addWidget(self.opacity)
        self.outline_action = QAction("Outline (O)", self, checkable=True)
        self.outline_action.toggled.connect(self._apply_outline)
        tb.addAction(self.outline_action)
        clean_action = QAction("Auto-clean (Shift+R)", self)
        clean_action.triggered.connect(lambda: self._guard(s.start_auto_clean))
        tb.addAction(clean_action)
        self.protect_action = QAction("Protect other parts", self, checkable=True)
        self.protect_action.toggled.connect(s.set_protect)
        tb.addAction(self.protect_action)

    def _build_shortcuts(self) -> None:
        s = self.session

        def key(sequence, fn):
            shortcut = QShortcut(QKeySequence(sequence), self)
            shortcut.activated.connect(lambda: self._guard(self._typed_then, fn))

        for i in range(10):  # 1-9, then 0 for the 10th material
            key(str((i + 1) % 10), lambda i=i: s.set_material_index(i))
        key("B", lambda: s.set_tool("brush"))
        key("E", lambda: s.set_tool("eraser"))
        key("V", lambda: s.set_tool("select"))
        key("G", lambda: s.set_tool("points"))
        QShortcut(QKeySequence("Esc"), self).activated.connect(lambda: self._guard(self._escape))  # Esc drops typing
        key("[", lambda: s.set_brush_radius(s.brush_radius - max(1, s.brush_radius // 5)))
        key("]", lambda: s.set_brush_radius(s.brush_radius + max(1, s.brush_radius // 5)))
        key("S", s.toggle_solo)
        key("O", self.outline_action.toggle)
        key("N", s.new_part)
        key("Shift+N", s.new_object)
        key("M", s.merge_selected_parts)
        key("Shift+M", s.merge_selected_objects)
        key("Del", s.delete_selected_parts)
        key("Shift+Del", s.delete_selected_object)
        key("Ctrl+Z", s.undo)
        key("Ctrl+Shift+Z", s.redo)
        key("Ctrl+Y", s.redo)
        key("A", lambda: s.step(-1))
        key("D", lambda: s.step(1))
        key("Ctrl+U", s.next_unreviewed)
        key("Ctrl+Return", self._mark_reviewed)
        key("Ctrl+Enter", self._mark_reviewed)
        key("F", s.toggle_flag)
        key("J", lambda: s.select_next_object(1))
        key("K", lambda: s.select_next_object(-1))
        key("Ctrl+S", self._save_now)
        key("R", lambda: s.set_tool("sam"))
        key("Shift+R", s.start_auto_clean)
        key("P", s.select_next_problem)
        key(".", s.repeat_last_label)
        key("Ctrl+N", s.new_empty_part)
        key("Home", self.canvas.fit)
        key("F1", self._show_help)
        key("?", self._show_help)

        def canvas_key(sequence, fn):
            # Only while the canvas has focus: the name and class fields need their own Enter.
            shortcut = QShortcut(QKeySequence(sequence), self.canvas)
            shortcut.setContext(Qt.ShortcutContext.WidgetShortcut)
            shortcut.activated.connect(lambda: self._guard(fn))

        canvas_key("Tab", lambda: s.sam_cycle(1))
        canvas_key("Backtab", lambda: s.sam_cycle(-1))
        canvas_key("Shift+Backtab", lambda: s.sam_cycle(-1))  # how some platforms report Shift+Tab
        canvas_key("Return", lambda: self._enter(union=False))
        canvas_key("Enter", lambda: self._enter(union=False))
        canvas_key("Shift+Return", lambda: s.sam_accept(union=True))
        canvas_key("Shift+Enter", lambda: s.sam_accept(union=True))

    def _connect(self) -> None:
        s, c, t, lp = self.session, self.canvas, self.tree, self.labels_panel

        def guarded(fn):
            return lambda *args: self._guard(fn, *args)

        c.clicked.connect(guarded(s.select_at))
        c.stroke_started.connect(guarded(s.begin_stroke))
        c.stroke_moved.connect(guarded(s.stroke_to))
        c.stroke_ended.connect(guarded(s.end_stroke))
        c.sam_clicked.connect(guarded(s.sam_click))
        c.sam_boxed.connect(guarded(s.sam_box))
        c.point_moved.connect(guarded(s.move_point))
        c.point_inserted.connect(guarded(s.insert_point))
        c.point_deleted.connect(guarded(s.delete_point))
        t.part_clicked.connect(guarded(lambda pid, add: self._typed_then(s.select_part, pid, add)))
        t.visibility_changed.connect(lambda kind, i, visible: s.set_hidden(kind, i, not visible))
        t.move_requested.connect(guarded(s.move_part_to))
        t.promote_requested.connect(guarded(s.promote_part))
        t.clean_requested.connect(guarded(s.clean_part))
        t.delete_part_requested.connect(guarded(lambda pid: s.delete_parts([pid])))
        t.delete_object_requested.connect(guarded(s.delete_object))
        lp.material_chosen.connect(guarded(s.set_material))
        lp.part_name_entered.connect(guarded(s.set_part_name))
        lp.class_entered.connect(guarded(s.set_object_class))
        # After a label button, Enter in a text field or closing the class list, give the
        # keyboard back to the canvas, or the next hotkeys (2, D, Del...) would be typed there.
        lp.done.connect(self.canvas.setFocus)
        self.files.itemClicked.connect(lambda item: self._open_row(self.files.row(item)))

    # ---- reacting to the session ----------------------------------------------------------------

    def _on_change(self, kind: str) -> None:
        s = self.session
        if kind == "pixels":
            self.canvas.overlay.refresh()
            return
        if kind == "image":
            self.canvas.set_image(s.image, s.ann.label_map)
            self._apply_opacity()
            self.canvas.overlay.set_outline(self.outline_action.isChecked())
            self.files.blockSignals(True)
            self.files.setCurrentRow(s.index)
            self.files.blockSignals(False)
        if kind in ("image", "annotation", "selection", "view"):
            self._refresh_lut()
        if kind in ("image", "annotation", "selection"):
            self.tree.rebuild(s.ann, s.selected, s.hidden_parts, s.hidden_objects)
            if s.primary is None:
                self.labels_panel.show_part(None, None)
            else:
                self.labels_panel.show_part(s.ann.part(s.primary), s.ann.object_of(s.primary))
        if kind in ("image", "view"):
            self._sync_view()
        if kind in ("image", "annotation", "selection", "view", "points"):
            self.canvas.set_rings(s.rings)
        if kind in ("image", "annotation", "status"):
            self._refresh_files()
        if kind in ("image", "annotation", "view", "sam", "refine"):
            self._refresh_sam()
        if kind == "annotation":
            self._save_soon.start()  # (re)start: saves once the edits pause
        self._refresh_status()

    def _refresh_lut(self) -> None:
        s = self.session
        self.canvas.overlay.set_lut(build_lut(
            s.ann, s.labels, s.colour_mode, hidden_parts=s.hidden_parts, hidden_objects=s.hidden_objects,
            solo_object=s.solo_object, selected=set(s.selected)))

    def _refresh_sam(self) -> None:
        s = self.session
        if s.auto_clean is not None:
            self.canvas.set_sam_prompts([], [], None)
            self.canvas.set_sam_preview(s.auto_clean_preview())
            return
        if s.sam is None:
            self.canvas.set_sam_prompts([], [], None)
            self.canvas.set_sam_preview(None)
            return
        self.canvas.set_sam_prompts(s.sam.points, s.sam.labels, s.sam.box)
        self.canvas.set_sam_preview(s.sam_preview())

    def _sync_sam_action(self) -> None:
        # Hidden without SAM, and after SAM failed for good (e.g. the model could not load).
        self.tool_actions["sam"].setVisible(self.sam is not None and self.session.sam_enabled)

    def _on_sam_status(self) -> None:
        self._sync_sam_action()
        self._refresh_status()

    def _sync_view(self) -> None:
        s = self.session
        self._sync_sam_action()
        self.tool_actions[s.tool].setChecked(True)
        self.canvas.mode = {"brush": "paint", "eraser": "paint", "sam": "sam", "points": "points"}.get(s.tool, "select")
        self.canvas.brush_radius = s.brush_radius
        for widget, update in ((self.mode_box, lambda: self.mode_box.setCurrentText(s.colour_mode)),
                               (self.protect_action, lambda: self.protect_action.setChecked(s.protect_others))):
            widget.blockSignals(True)
            update()
            widget.blockSignals(False)

    def _refresh_files(self) -> None:
        s = self.session
        for i, p in enumerate(s.images):
            self.files.item(i).setText(f"{STATUS_ICONS[s.review_status(i)]} {p.name}")

    def _refresh_status(self) -> None:
        s = self.session
        if s.ann is None:
            return
        unsaved = " · unsaved" if s.ann.dirty else ""
        protect = " · protect" if s.protect_others else ""
        note = f"  |  {s.messages[-1]}" if s.messages else ""
        sam = ""
        if s.auto_clean is not None:
            sam = f"  |  auto-clean ready: {s.auto_clean.summary()} — Enter accept · Esc discard"
        elif s.clean_pending:
            sam = f"  |  auto-clean: {self.sam.status if self.sam is not None else 'working'}"
        elif s.sam is not None and s.sam.candidates:
            which = ("your mask smoothed" if s.sam.is_alternative
                     else f"SAM answer, IoU {s.sam.score:.2f}")
            sam = (f"  |  candidate {s.sam.index + 1}/{len(s.sam.candidates)}: {which} — "
                   "Tab next · Enter replace · Shift+Enter add · Esc cancel")
        elif s.tool == "sam" and self.sam is not None:
            sam = f"  |  {self.sam.status}"
        elif s.tool == "points":
            sam = ("  |  points: drag a point · double-click an edge to add one · Ctrl+click a point to delete it"
                   if s.rings else "  |  points: click a part to edit its outline")
        reviewed, total = s.review_progress()
        self.setWindowTitle(f"AnnoFix — {s.images[s.index].name} — reviewed {reviewed}/{total}")
        self.status.setText(f"{s.index + 1}/{len(s.images)}  {s.stem}  [{s.ann.review_status}{unsaved}]"
                            f"  reviewed {reviewed}/{total}  tool: {s.tool} r={s.brush_radius}{protect}  "
                            f"objects: {len(s.ann.objects)}{sam}{note}")
        self.status.setToolTip("\n".join(s.messages))

    def _apply_opacity(self) -> None:
        if self.canvas.overlay is not None:
            self.canvas.overlay.setOpacity(self.opacity.value() / 100)

    def _apply_outline(self, on: bool) -> None:
        if self.canvas.overlay is not None:
            self.canvas.overlay.set_outline(on)

    # ---- actions --------------------------------------------------------------------------------

    def _guard(self, fn, *args):
        """Run a Session call; show expected failures (bad file, disk full, no part ids left)
        instead of crashing."""
        try:
            return fn(*args)
        except (ValueError, OSError, OverflowError) as e:
            QMessageBox.warning(self, "AnnoFix", str(e))
            return None

    def _typed_then(self, fn, *args):
        """Apply a name or class typed but not entered, then act (it is for the current part)."""
        self.labels_panel.commit_pending()
        return fn(*args)

    def _open_row(self, row: int) -> None:
        if row == self.session.index:
            return  # already open: reloading would only drop the selection and hidden parts
        self._guard(self._typed_then, self.session.open, row)
        self.files.blockSignals(True)
        self.files.setCurrentRow(self.session.index)  # stay in step if that image failed to load
        self.files.blockSignals(False)

    def _escape(self) -> None:
        s = self.session
        if self.labels_panel.isAncestorOf(QApplication.focusWidget()):
            self.labels_panel.revert()  # Esc in a text field: drop the typing, nothing else
            self.canvas.setFocus()
        elif self.canvas.cancel_point_drag():
            pass
        elif s.discard_auto_clean():
            pass
        elif s.painting:
            s.cancel_stroke()
        elif s.sam is not None:
            s.sam_cancel()
        else:
            s.set_tool("select")
            s.select_part(None)

    def _mark_reviewed(self) -> None:
        problems = self.session.mark_reviewed_and_next()
        if problems:
            more = f"\n… and {len(problems) - 20} more" if len(problems) > 20 else ""
            QMessageBox.information(self, "Not ready to mark reviewed", "\n".join(problems[:20]) + more)

    def _enter(self, union: bool) -> None:
        if self.session.auto_clean is not None:
            self.session.accept_auto_clean()
        else:
            self.session.sam_accept(union=union)

    def _save_now(self) -> None:
        s = self.session
        s.end_stroke()
        s.add_message("saved" if s.save() else "no unsaved changes")

    def _show_help(self) -> None:
        rows = "".join(f"<tr><td><b>{keys}</b>&nbsp;&nbsp;</td><td>{what}</td></tr>" for keys, what in SHORTCUTS)
        QMessageBox.information(self, "AnnoFix shortcuts", f"<table>{rows}</table>")

    def eventFilter(self, obj, ev) -> bool:
        # Only Space sent to a widget in this window (a dialog's buttons need their Space).
        if (ev.type() in (QEvent.Type.KeyPress, QEvent.Type.KeyRelease) and ev.key() == Qt.Key.Key_Space
                and isinstance(obj, QWidget) and obj.window() is self):
            focus = QApplication.focusWidget()
            typing = isinstance(focus, QLineEdit) or (isinstance(focus, QComboBox) and focus.isEditable())
            if not typing:
                if not ev.isAutoRepeat():
                    self.canvas.set_masks_visible(ev.type() == QEvent.Type.KeyRelease)
                return True
        return super().eventFilter(obj, ev)

    def closeEvent(self, ev) -> None:
        s = self.session
        try:
            self.labels_panel.commit_pending()
            s.end_stroke()
            s.save()
        except OSError as e:
            answer = QMessageBox.question(self, "AnnoFix", f"Could not save: {e}\n\nQuit anyway?")
            if answer != QMessageBox.StandardButton.Yes:
                ev.ignore()
                return
        if self.sam is not None and not self.sam.shutdown():
            self.sam_still_running = True  # __main__ ends the process with os._exit()
        ev.accept()


def close_all_soon() -> None:
    """Close every window (which saves) once the code running now has returned to the event
    loop: Ctrl+C can arrive in the middle of an edit, and a save then would skip that edit."""
    QTimer.singleShot(0, QApplication.closeAllWindows)


def install_excepthook(session: Session, log_path: Path):
    """PyQt6 aborts on an unhandled exception in a slot. Instead: log it, roll back a half
    stroke, save the finished edits, and tell the user. Ctrl+C in the terminal while a slot
    runs closes the window the normal way (which saves)."""

    def hook(exc_type, exc, tb):
        if issubclass(exc_type, KeyboardInterrupt) and QApplication.instance() is not None:
            close_all_soon()
            return
        text = "".join(traceback.format_exception(exc_type, exc, tb))
        try:
            with open(log_path, "a") as f:
                f.write(f"--- {time.strftime('%Y-%m-%d %H:%M:%S')}\n{text}\n")
        except OSError:
            pass
        sys.__stderr__.write(text)
        outcome = "No unsaved work was lost."
        saved = False
        try:  # roll back and save first, without the window: it may be what failed
            saved = session.recover_after_error()
            if saved:
                outcome = "Your finished edits were saved."
        except Exception as save_error:  # noqa: BLE001 - never raise from the hook itself
            outcome = f"Saving failed too ({save_error}); your last edits may be lost."
        try:  # then redraw (and export) what was rolled back and saved
            session.announce_recovery(saved)
        except Exception as redraw_error:  # noqa: BLE001
            sys.__stderr__.write(f"redraw after the error failed too: {redraw_error!r}\n")
        if QApplication.instance() is not None:
            QMessageBox.critical(None, "AnnoFix error",
                                 f"{exc_type.__name__}: {exc}\n\n{outcome}\nDetails: {log_path}")

    sys.excepthook = hook
    return hook
