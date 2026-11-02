"""Labels for the selected part: object class (suggestions from labels.yaml or free text),
material buttons (hotkeys 1-9, 0 are window shortcuts), part-name suggestions for the class plus
a free-text field. A typed name or class is applied on Enter, when the field loses focus, and by
commit_pending() before anything else acts (buttons, the tree, the file list and shortcuts take
no focus, so they would otherwise skip it). After a button, Enter or closing the class list,
`done` asks the window to give the keyboard back to the canvas. Nothing here stores
Part/AnnObject instances."""
from __future__ import annotations

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (QComboBox, QGridLayout, QHBoxLayout, QLabel, QLineEdit, QPushButton,
                             QVBoxLayout, QWidget)


def _heading(text: str) -> QLabel:
    label = QLabel(text)
    label.setObjectName("secondary")  # grey, as CVAT's field labels
    return label


def _set_active(button: QPushButton, on: bool) -> None:
    """The window's style sheet draws an active button blue (QPushButton[active="true"])."""
    if button.property("active") != on:
        button.setProperty("active", on)
        button.style().unpolish(button)
        button.style().polish(button)


class _ClassBox(QComboBox):
    popup_closed = pyqtSignal()

    def hidePopup(self) -> None:
        super().hidePopup()
        self.popup_closed.emit()


class LabelPanel(QWidget):
    material_chosen = pyqtSignal(str)
    part_name_entered = pyqtSignal(str)
    class_entered = pyqtSignal(str)
    done = pyqtSignal()  # give the keyboard back to the canvas

    def __init__(self, labels, parent=None):
        super().__init__(parent)
        self.labels = labels
        layout = QVBoxLayout(self)
        self.title = QLabel()
        self.title.setObjectName("panelTitle")
        layout.addWidget(self.title)

        layout.addWidget(_heading("Object class"))
        self._shown_name = self._shown_class = ""  # what the selection has: only changes are sent
        self.class_box = _ClassBox()
        self.class_box.setEditable(True)
        self.class_box.setInsertPolicy(QComboBox.InsertPolicy.NoInsert)
        self.class_box.addItems(sorted(labels.objects))
        self.class_box.textActivated.connect(self._commit_class)
        self.class_box.popup_closed.connect(self.done.emit)
        self.class_box.lineEdit().returnPressed.connect(self._commit_class)
        self.class_box.lineEdit().returnPressed.connect(self.done.emit)
        self.class_box.lineEdit().editingFinished.connect(self._commit_class)
        layout.addWidget(self.class_box)

        layout.addWidget(_heading("Material"))
        grid = QGridLayout()
        self.material_buttons: list[QPushButton] = []
        for i, material in enumerate(labels.materials):
            b = QPushButton(f"{(i + 1) % 10}  {material}" if i < 10 else material)
            b.setFocusPolicy(Qt.FocusPolicy.NoFocus)
            b.clicked.connect(self.commit_pending)
            b.clicked.connect(lambda checked=False, m=material: self.material_chosen.emit(m))
            b.clicked.connect(self.done.emit)
            grid.addWidget(b, i // 3, i % 3)
            self.material_buttons.append(b)
        layout.addLayout(grid)

        layout.addWidget(_heading("Part name"))
        names = QWidget()
        self._names_layout = QHBoxLayout(names)
        self._names_layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(names)
        self.name_buttons: list[QPushButton] = []
        self.name_edit = QLineEdit()
        self.name_edit.setPlaceholderText("type a part name and press Enter")
        self.name_edit.returnPressed.connect(self._commit_name)
        self.name_edit.returnPressed.connect(self.done.emit)
        self.name_edit.editingFinished.connect(self._commit_name)
        layout.addWidget(self.name_edit)
        layout.addStretch(1)
        self.show_part(None, None)

    def show_part(self, part, obj) -> None:
        for b in self.name_buttons:
            self._names_layout.removeWidget(b)
            b.deleteLater()
        self.name_buttons = []
        enabled = part is not None
        for w in (self.class_box, self.name_edit, *self.material_buttons):
            w.setEnabled(enabled)
        if part is None:
            self._shown_name = self._shown_class = ""
            self.title.setText("No part selected: click a mask")
            self.class_box.setEditText("")
            self.name_edit.clear()
            for b in self.material_buttons:
                _set_active(b, False)
            return
        self._shown_name, self._shown_class = part.name or "", obj.cls or ""
        self.title.setText(f"Part #{part.id} of object #{obj.id}")
        self.class_box.setEditText(self._shown_class)
        self.name_edit.setText(self._shown_name)
        for name in self.labels.parts_for(obj.cls):
            b = QPushButton(name)
            b.setFocusPolicy(Qt.FocusPolicy.NoFocus)
            b.clicked.connect(self._commit_class)
            b.clicked.connect(lambda checked=False, n=name: self.part_name_entered.emit(n))
            b.clicked.connect(self.done.emit)
            self._names_layout.addWidget(b)
            self.name_buttons.append(b)
        for b, material in zip(self.material_buttons, self.labels.materials):
            _set_active(b, material == part.material)

    def _commit_name(self, *_) -> None:
        text = self.name_edit.text().strip()
        if self.name_edit.isEnabled() and text != self._shown_name:
            self.part_name_entered.emit(text)

    def _commit_class(self, *_) -> None:
        text = self.class_box.currentText().strip()
        if self.class_box.isEnabled() and text != self._shown_class:
            self.class_entered.emit(text)

    def commit_pending(self, *_) -> None:
        """Apply what was typed but not yet entered, before another action replaces the view."""
        self._commit_class()
        self._commit_name()

    def revert(self) -> None:
        """Esc in a field: drop what was typed, show the selection's labels again."""
        self.name_edit.setText(self._shown_name)
        self.class_box.setEditText(self._shown_class)
