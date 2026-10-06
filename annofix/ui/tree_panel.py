"""Objects -> parts tree: tick boxes show/hide, a click selects, the context menu holds the
structure edits. Rebuilt from the annotation after every change (ids only are kept)."""
from __future__ import annotations

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import QAbstractItemView, QApplication, QMenu, QTreeWidget, QTreeWidgetItem

ROLE = Qt.ItemDataRole.UserRole  # ("object", id) or ("part", id)


class ObjectTree(QTreeWidget):
    part_clicked = pyqtSignal(int, bool)            # part id, add to selection
    visibility_changed = pyqtSignal(str, int, bool)  # "object" | "part", id, visible
    move_requested = pyqtSignal(int, int)           # part id, target object id
    promote_requested = pyqtSignal(int)
    clean_requested = pyqtSignal(int)
    delete_part_requested = pyqtSignal(int)
    delete_object_requested = pyqtSignal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setHeaderLabels(["Objects and parts"])
        self.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        self.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.customContextMenuRequested.connect(self._show_menu)
        self.itemClicked.connect(self._clicked)
        self.itemChanged.connect(self._changed)
        self._objects: list[tuple[int, str]] = []
        self._building = False

    def rebuild(self, ann, selected, hidden_parts, hidden_objects) -> None:
        scroll = self.verticalScrollBar().value()
        self._building = True
        self.clear()
        self._objects = []
        focus_item = None
        for o in ann.objects:
            label = f"#{o.id} {o.cls or '(no class)'}"
            self._objects.append((o.id, label))
            oi = QTreeWidgetItem([f"{label}  ({len(o.parts)} part{'s' if len(o.parts) != 1 else ''})"])
            oi.setData(0, ROLE, ("object", o.id))
            oi.setFlags(oi.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            oi.setCheckState(0, Qt.CheckState.Unchecked if o.id in hidden_objects else Qt.CheckState.Checked)
            for p in o.parts:
                pi = QTreeWidgetItem([f"#{p.id} {p.name or '?'} · {p.material or '?'}"])
                pi.setData(0, ROLE, ("part", p.id))
                pi.setFlags(pi.flags() | Qt.ItemFlag.ItemIsUserCheckable)
                pi.setCheckState(0, Qt.CheckState.Unchecked if p.id in hidden_parts else Qt.CheckState.Checked)
                if p.id in selected:
                    font = pi.font(0)
                    font.setBold(True)
                    pi.setFont(0, font)
                    pi.setBackground(0, QColor(70, 110, 160))
                    focus_item = focus_item or pi
                oi.addChild(pi)
            self.addTopLevelItem(oi)
            oi.setExpanded(any(p.id in selected for p in o.parts))
        self._building = False
        self.verticalScrollBar().setValue(scroll)
        if focus_item is not None:
            self.scrollToItem(focus_item)

    def _clicked(self, item, column) -> None:
        kind, item_id = item.data(0, ROLE)
        if kind == "part":
            add = bool(QApplication.keyboardModifiers() & Qt.KeyboardModifier.ControlModifier)
            self.part_clicked.emit(item_id, add)
        elif item.childCount():
            self.part_clicked.emit(item.child(0).data(0, ROLE)[1], False)

    def _changed(self, item, column) -> None:
        if self._building:
            return
        kind, item_id = item.data(0, ROLE)
        self.visibility_changed.emit(kind, item_id, item.checkState(0) == Qt.CheckState.Checked)

    def _show_menu(self, pos) -> None:
        item = self.itemAt(pos)
        if item is not None:
            self._build_menu(item).exec(self.viewport().mapToGlobal(pos))

    def _build_menu(self, item) -> QMenu:
        kind, item_id = item.data(0, ROLE)
        menu = QMenu(self)

        def add(target_menu, text, fn):
            action = target_menu.addAction(text)
            action.triggered.connect(lambda checked=False: fn())

        if kind == "part":
            move = menu.addMenu("Move to object")
            for oid, label in self._objects:
                add(move, label, lambda oid=oid: self.move_requested.emit(item_id, oid))
            add(menu, "Make it a new object", lambda: self.promote_requested.emit(item_id))
            add(menu, "Clean up (islands, holes)", lambda: self.clean_requested.emit(item_id))
            add(menu, "Delete part", lambda: self.delete_part_requested.emit(item_id))
        else:
            add(menu, "Delete object", lambda: self.delete_object_requested.emit(item_id))
        return menu
