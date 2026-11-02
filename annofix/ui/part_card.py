"""CVAT's right-click card: the clicked part's object class, material and part name, edited right
at the mask. It holds a LabelPanel, so it behaves like the one in the side bar (suggestions,
1-9 / 0 for the materials). The name field has the keyboard, ready to type. Enter in a text
field applies it and closes the card; a material applies and leaves it open; a click elsewhere
applies what was typed and closes it; Esc closes it and drops the typing."""
from __future__ import annotations

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QKeySequence, QShortcut
from PyQt6.QtWidgets import QApplication, QFrame, QVBoxLayout

from .label_panel import LabelPanel


class PartCard(QFrame):
    closed = pyqtSignal()

    def __init__(self, labels, parent=None):
        super().__init__(parent, Qt.WindowType.Popup)
        self.setObjectName("partCard")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        self.panel = LabelPanel(labels)
        self.panel.setMinimumWidth(320)
        layout.addWidget(self.panel)
        self.panel.name_edit.returnPressed.connect(self.close)
        self.panel.class_box.lineEdit().returnPressed.connect(self.close)
        for i, button in enumerate(self.panel.material_buttons[:10]):
            QShortcut(QKeySequence(str((i + 1) % 10)), self).activated.connect(button.click)

    def show_for(self, part, obj, screen_pos) -> None:
        self.panel.show_part(part, obj)
        self.adjustSize()
        screen = (QApplication.screenAt(screen_pos) or QApplication.primaryScreen()).availableGeometry()
        x = min(max(screen_pos.x() + 12, screen.left()), screen.right() - self.width())
        y = min(max(screen_pos.y() + 12, screen.top()), screen.bottom() - self.height())
        self.move(x, y)
        self.show()
        self.panel.name_edit.setFocus()
        self.panel.name_edit.selectAll()

    def refresh(self, part, obj) -> None:
        """The part changed while the card is open (a material was chosen): show it again."""
        if self.isVisible() and part is not None:
            self.panel.show_part(part, obj)

    def keyPressEvent(self, ev) -> None:
        if ev.key() == Qt.Key.Key_Escape:
            self.panel.revert()
            self.close()
            return
        super().keyPressEvent(ev)

    def hideEvent(self, ev) -> None:
        self.panel.commit_pending()  # clicked away: keep what was typed, as the side bar does
        super().hideEvent(ev)
        self.closed.emit()
