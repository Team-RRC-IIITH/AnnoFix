"""Labels for the selected part: object class (suggestions from labels.yaml or free text),
material buttons (hotkeys 1-9 are window shortcuts), part-name suggestions for the class plus
a free-text field. Nothing here stores Part/AnnObject instances."""
from __future__ import annotations

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (QComboBox, QGridLayout, QHBoxLayout, QLabel, QLineEdit, QPushButton,
                             QVBoxLayout, QWidget)


class LabelPanel(QWidget):
    material_chosen = pyqtSignal(str)
    part_name_entered = pyqtSignal(str)
    class_entered = pyqtSignal(str)

    def __init__(self, labels, parent=None):
        super().__init__(parent)
        self.labels = labels
        layout = QVBoxLayout(self)
        self.title = QLabel()
        layout.addWidget(self.title)

        layout.addWidget(QLabel("Object class"))
        self.class_box = QComboBox()
        self.class_box.setEditable(True)
        self.class_box.setInsertPolicy(QComboBox.InsertPolicy.NoInsert)
        self.class_box.addItems(sorted(labels.objects))
        self.class_box.textActivated.connect(self.class_entered.emit)
        self.class_box.lineEdit().returnPressed.connect(lambda: self.class_entered.emit(self.class_box.currentText()))
        layout.addWidget(self.class_box)

        layout.addWidget(QLabel("Material"))
        grid = QGridLayout()
        self.material_buttons: list[QPushButton] = []
        for i, material in enumerate(labels.materials):
            b = QPushButton(f"{(i + 1) % 10}  {material}" if i < 10 else material)
            b.setFocusPolicy(Qt.FocusPolicy.NoFocus)
            b.clicked.connect(lambda checked=False, m=material: self.material_chosen.emit(m))
            grid.addWidget(b, i // 3, i % 3)
            self.material_buttons.append(b)
        layout.addLayout(grid)

        layout.addWidget(QLabel("Part name"))
        names = QWidget()
        self._names_layout = QHBoxLayout(names)
        self._names_layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(names)
        self.name_buttons: list[QPushButton] = []
        self.name_edit = QLineEdit()
        self.name_edit.setPlaceholderText("type a part name and press Enter")
        self.name_edit.returnPressed.connect(lambda: self.part_name_entered.emit(self.name_edit.text()))
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
            self.title.setText("No part selected: click a mask")
            self.class_box.setEditText("")
            self.name_edit.clear()
            for b in self.material_buttons:
                b.setStyleSheet("")
            return
        self.title.setText(f"Part #{part.id} of object #{obj.id}")
        self.class_box.setEditText(obj.cls or "")
        self.name_edit.setText(part.name or "")
        for name in self.labels.parts_for(obj.cls):
            b = QPushButton(name)
            b.setFocusPolicy(Qt.FocusPolicy.NoFocus)
            b.clicked.connect(lambda checked=False, n=name: self.part_name_entered.emit(n))
            self._names_layout.addWidget(b)
            self.name_buttons.append(b)
        for b, material in zip(self.material_buttons, self.labels.materials):
            b.setStyleSheet("font-weight: bold" if material == part.material else "")
