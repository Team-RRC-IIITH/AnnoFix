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
from .tree_panel import ObjectTree

STATUS_ICONS = {"none": "○", "unreviewed": "•", "reviewed": "✓", "flagged": "⚑"}
AUTOSAVE_MS = 30_000


class MainWindow(QMainWindow):
    def __init__(self, session: Session):
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
        self._autosave = QTimer(self)
        self._autosave.timeout.connect(lambda: self._guard(self.session.save))
        self._autosave.start(AUTOSAVE_MS)
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
        for tool, text in (("select", "Select (V)"), ("brush", "Brush (B)"), ("eraser", "Eraser (E)")):
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
        self.protect_action = QAction("Protect other parts", self, checkable=True)
        self.protect_action.toggled.connect(s.set_protect)
        tb.addAction(self.protect_action)

    def _build_shortcuts(self) -> None:
        s = self.session

        def key(sequence, fn):
            shortcut = QShortcut(QKeySequence(sequence), self)
            shortcut.activated.connect(lambda: self._guard(fn))

        for i in range(9):
            key(str(i + 1), lambda i=i: s.set_material_index(i))
        key("B", lambda: s.set_tool("brush"))
        key("E", lambda: s.set_tool("eraser"))
        key("V", lambda: s.set_tool("select"))
        key("Esc", self._escape)
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
        key("Ctrl+S", s.save)

    def _connect(self) -> None:
        s, c, t, lp = self.session, self.canvas, self.tree, self.labels_panel
        c.clicked.connect(lambda x, y, add: s.select_at(x, y, add))
        c.stroke_started.connect(s.begin_stroke)
        c.stroke_moved.connect(s.stroke_to)
        c.stroke_ended.connect(s.end_stroke)
        t.part_clicked.connect(s.select_part)
        t.visibility_changed.connect(lambda kind, i, visible: s.set_hidden(kind, i, not visible))
        t.move_requested.connect(s.move_part_to)
        t.promote_requested.connect(s.promote_part)
        t.clean_requested.connect(s.clean_part)
        t.delete_part_requested.connect(lambda pid: s.delete_parts([pid]))
        t.delete_object_requested.connect(s.delete_object)
        lp.material_chosen.connect(s.set_material)
        lp.part_name_entered.connect(s.set_part_name)
        lp.class_entered.connect(s.set_object_class)
        # After Enter in a text field, give the keyboard back to the canvas, or the next
        # hotkeys (2, D, Del...) would be typed into the field instead.
        lp.part_name_entered.connect(lambda _name: self.canvas.setFocus())
        lp.class_entered.connect(lambda _cls: self.canvas.setFocus())
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
        if kind in ("image", "annotation", "status"):
            self._refresh_files()
        self._refresh_status()

    def _refresh_lut(self) -> None:
        s = self.session
        self.canvas.overlay.set_lut(build_lut(
            s.ann, s.labels, s.colour_mode, hidden_parts=s.hidden_parts, hidden_objects=s.hidden_objects,
            solo_object=s.solo_object, selected=set(s.selected)))

    def _sync_view(self) -> None:
        s = self.session
        self.tool_actions[s.tool].setChecked(True)
        self.canvas.paint_mode = s.tool in ("brush", "eraser")
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
        self.status.setText(f"{s.index + 1}/{len(s.images)}  {s.stem}  [{s.ann.review_status}{unsaved}]"
                            f"  tool: {s.tool} r={s.brush_radius}{protect}  "
                            f"objects: {len(s.ann.objects)}{note}")
        self.status.setToolTip("\n".join(s.messages))

    def _apply_opacity(self) -> None:
        if self.canvas.overlay is not None:
            self.canvas.overlay.setOpacity(self.opacity.value() / 100)

    def _apply_outline(self, on: bool) -> None:
        if self.canvas.overlay is not None:
            self.canvas.overlay.set_outline(on)

    # ---- actions --------------------------------------------------------------------------------

    def _guard(self, fn, *args):
        """Run a Session call; show expected failures (bad file, disk full) instead of crashing."""
        try:
            return fn(*args)
        except (ValueError, OSError) as e:
            QMessageBox.warning(self, "AnnoFix", str(e))
            return None

    def _open_row(self, row: int) -> None:
        self._guard(self.session.open, row)
        self.files.blockSignals(True)
        self.files.setCurrentRow(self.session.index)  # stay in step if that image failed to load
        self.files.blockSignals(False)

    def _escape(self) -> None:
        s = self.session
        if s.painting:
            s.cancel_stroke()
        else:
            s.set_tool("select")
            s.select_part(None)

    def _mark_reviewed(self) -> None:
        problems = self.session.mark_reviewed()
        if problems:
            more = f"\n… and {len(problems) - 20} more" if len(problems) > 20 else ""
            QMessageBox.information(self, "Not ready to mark reviewed", "\n".join(problems[:20]) + more)

    def eventFilter(self, obj, ev) -> bool:
        if ev.type() in (QEvent.Type.KeyPress, QEvent.Type.KeyRelease) and ev.key() == Qt.Key.Key_Space:
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
            s.end_stroke()
            s.save()
        except OSError as e:
            answer = QMessageBox.question(self, "AnnoFix", f"Could not save: {e}\n\nQuit anyway?")
            if answer != QMessageBox.StandardButton.Yes:
                ev.ignore()
                return
        ev.accept()


def install_excepthook(session: Session, log_path: Path):
    """PyQt6 aborts on an unhandled exception in a slot. Instead: log it, roll back a half
    stroke, save the finished edits, and tell the user."""

    def hook(exc_type, exc, tb):
        text = "".join(traceback.format_exception(exc_type, exc, tb))
        try:
            with open(log_path, "a") as f:
                f.write(f"--- {time.strftime('%Y-%m-%d %H:%M:%S')}\n{text}\n")
        except OSError:
            pass
        sys.__stderr__.write(text)
        outcome = "No unsaved work was lost."
        try:
            if session.painting:
                session.cancel_stroke()
            elif session.ann is not None and session.ann.in_transaction:
                session.ann.rollback()
            if session.save():
                outcome = "Your finished edits were saved."
        except Exception as save_error:  # noqa: BLE001 - never raise from the hook itself
            outcome = f"Saving failed too ({save_error}); your last edits may be lost."
        if QApplication.instance() is not None:
            QMessageBox.critical(None, "AnnoFix error",
                                 f"{exc_type.__name__}: {exc}\n\n{outcome}\nDetails: {log_path}")

    sys.excepthook = hook
    return hook
