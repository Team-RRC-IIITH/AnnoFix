"""Objects -> parts as CVAT-style cards: a card per object (colour bar, class, confidence, eye,
open/close), a row per part (colour dot, name, material list, confidence, eye). A click
selects (Ctrl adds), the eye shows/hides, the material list sets that part's material, and
right-click holds the structure edits. Rebuilt from the annotation after every change; which
cards you opened is remembered by object id. Nothing here takes the keyboard: it stays on the
canvas for the shortcuts."""
from __future__ import annotations

from PyQt6.QtCore import QSize, Qt, pyqtSignal
from PyQt6.QtWidgets import (QComboBox, QFrame, QHBoxLayout, QLabel, QMenu, QScrollArea, QToolButton,
                             QVBoxLayout, QWidget)

from .icons import icon, toggle_icon

NO_MATERIAL = "?"


def confidence(attrs: dict) -> str:
    pct = attrs.get("confidence_pct")
    return f"{pct:g}%" if isinstance(pct, (int, float)) and not isinstance(pct, bool) else ""


def _flag(button: QWidget, name: str, on: bool) -> None:
    """A style-sheet property (e.g. selected="true"), re-applied at once."""
    if button.property(name) != on:
        button.setProperty(name, on)
        button.style().unpolish(button)
        button.style().polish(button)


class _MaterialBox(QComboBox):
    def wheelEvent(self, ev) -> None:
        ev.ignore()  # the wheel scrolls the list; it must never change a material by accident


class _Clickable(QFrame):
    clicked = pyqtSignal(bool)   # Ctrl held
    menu_requested = pyqtSignal(object)  # global position

    def mousePressEvent(self, ev) -> None:
        if ev.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit(bool(ev.modifiers() & Qt.KeyboardModifier.ControlModifier))
            ev.accept()
            return
        super().mousePressEvent(ev)

    def contextMenuEvent(self, ev) -> None:
        self.menu_requested.emit(ev.globalPos())


def _eye(visible: bool) -> QToolButton:
    eye = QToolButton()
    eye.setObjectName("eye")
    eye.setCheckable(True)
    eye.setChecked(visible)
    eye.setIcon(toggle_icon("eye", "eye_off"))
    eye.setIconSize(QSize(16, 16))
    eye.setToolTip("Show / hide")
    eye.setFocusPolicy(Qt.FocusPolicy.NoFocus)
    return eye


def _dot(colour: str, w: int, h: int) -> QLabel:
    dot = QLabel()
    dot.setFixedSize(w, h)
    dot.setStyleSheet(f"background: {colour}; border-radius: {min(w, h) // 2}px;")
    return dot


class PartRow(_Clickable):
    def __init__(self, part, materials, colour: str, visible: bool, selected: bool):
        super().__init__()
        self.setObjectName("partRow")
        self.part_id = part.id
        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 2, 4, 2)
        layout.setSpacing(6)
        layout.addWidget(_dot(colour, 10, 10))
        self.name = QLabel(f"#{part.id} {part.name or '?'}")
        layout.addWidget(self.name, 1)
        self.material = _MaterialBox()
        self.material.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        items = list(materials) + [NO_MATERIAL]
        if part.material and part.material not in items:
            items.insert(0, part.material)  # a material labels.yaml does not list: still shown
        self.material.addItems(items)
        self.material.setCurrentText(part.material or NO_MATERIAL)
        self.material.setFixedWidth(104)
        layout.addWidget(self.material)
        self.confidence = QLabel(confidence(part.attrs))
        self.confidence.setObjectName("secondary")
        self.confidence.setFixedWidth(34)
        layout.addWidget(self.confidence)
        self.eye = _eye(visible)
        layout.addWidget(self.eye)
        _flag(self, "selected", selected)


class ObjectCard(QFrame):
    def __init__(self, obj, colour: str, visible: bool, expanded: bool, active: bool):
        super().__init__()
        self.setObjectName("objectCard")
        self.object_id = obj.id
        self.rows: list[PartRow] = []
        outer = QVBoxLayout(self)
        outer.setContentsMargins(6, 4, 6, 6)
        outer.setSpacing(2)
        self.header = _Clickable()
        self.header.setObjectName("cardHeader")
        head = QHBoxLayout(self.header)
        head.setContentsMargins(0, 2, 0, 2)
        head.setSpacing(8)
        head.addWidget(_dot(colour, 4, 22))
        self.title = QLabel(f"#{obj.id} {obj.cls or '(no class)'}")
        self.title.setObjectName("cardTitle")
        head.addWidget(self.title, 1)
        self.confidence = QLabel(confidence(obj.attrs))
        self.confidence.setObjectName("secondary")
        head.addWidget(self.confidence)
        self.eye = _eye(visible)
        head.addWidget(self.eye)
        self.chevron = QToolButton()
        self.chevron.setIcon(icon("down" if expanded else "next"))
        self.chevron.setIconSize(QSize(16, 16))
        self.chevron.setToolTip("Show / hide the parts")
        self.chevron.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        head.addWidget(self.chevron)
        outer.addWidget(self.header)
        self.body = QWidget()
        self._rows_layout = QVBoxLayout(self.body)
        self._rows_layout.setContentsMargins(0, 0, 0, 0)
        self._rows_layout.setSpacing(1)
        outer.addWidget(self.body)
        self.expanded = expanded
        self.body.setVisible(expanded)
        _flag(self, "active", active)

    def add_row(self, row: PartRow) -> None:
        self.rows.append(row)
        self._rows_layout.addWidget(row)

    def set_expanded(self, on: bool) -> None:
        self.expanded = on
        self.body.setVisible(on)
        self.chevron.setIcon(icon("down" if on else "next"))


class ObjectCards(QScrollArea):
    part_clicked = pyqtSignal(int, bool)            # part id, add to selection
    visibility_changed = pyqtSignal(str, int, bool)  # "object" | "part", id, visible
    material_set = pyqtSignal(int, str)             # part id, material
    move_requested = pyqtSignal(int, int)           # part id, target object id
    promote_requested = pyqtSignal(int)
    clean_requested = pyqtSignal(int)
    delete_part_requested = pyqtSignal(int)
    delete_object_requested = pyqtSignal(int)

    def __init__(self, labels, parent=None):
        super().__init__(parent)
        self.labels = labels
        self.setObjectName("objectCards")
        self.setWidgetResizable(True)
        self.setFrameShape(QFrame.Shape.NoFrame)
        self.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.cards: list[ObjectCard] = []
        self._opened: dict[int, bool] = {}   # object id -> opened/closed by its chevron
        self._objects: list[tuple[int, str]] = []
        self.setWidget(QWidget())

    def rebuild(self, ann, selected, hidden_parts, hidden_objects, colours=None) -> None:
        """`colours`: part id -> "#rrggbb", as the masks are drawn now."""
        colours = colours or {}
        scroll = self.verticalScrollBar().value()
        container = QWidget()
        container.setObjectName("cardsContainer")
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 4, 0)
        layout.setSpacing(6)
        self.cards, self._objects, focus = [], [], None
        selected = set(selected)
        for o in ann.objects:
            self._objects.append((o.id, f"#{o.id} {o.cls or '(no class)'}"))
            active = any(p.id in selected for p in o.parts)
            card = ObjectCard(o, colours.get(o.parts[0].id, "#9e9e9e") if o.parts else "#9e9e9e",
                              o.id not in hidden_objects, active or self._opened.get(o.id, False), active)
            card.header.clicked.connect(lambda add, o=o: o.parts and self.part_clicked.emit(o.parts[0].id, False))
            card.header.menu_requested.connect(lambda pos, oid=o.id: self._build_menu("object", oid).exec(pos))
            card.eye.toggled.connect(lambda on, oid=o.id: self.visibility_changed.emit("object", oid, on))
            card.chevron.clicked.connect(lambda checked=False, c=card: self._toggle(c))
            for p in o.parts:
                row = PartRow(p, self.labels.materials, colours.get(p.id, "#9e9e9e"),
                              p.id not in hidden_parts, p.id in selected)
                row.clicked.connect(lambda add, pid=p.id: self.part_clicked.emit(pid, add))
                row.menu_requested.connect(lambda pos, pid=p.id: self._build_menu("part", pid).exec(pos))
                row.eye.toggled.connect(lambda on, pid=p.id: self.visibility_changed.emit("part", pid, on))
                row.material.activated.connect(lambda i, pid=p.id, box=row.material: self._material(pid, box))
                card.add_row(row)
                if p.id in selected and focus is None:
                    focus = row
            layout.addWidget(card)
            self.cards.append(card)
        layout.addStretch(1)
        old = self.takeWidget()
        self.setWidget(container)
        if old is not None:
            old.deleteLater()  # not now: a click on one of its rows may be what is rebuilding the list
        self.verticalScrollBar().setValue(scroll)
        if focus is not None:
            self.ensureWidgetVisible(focus, 0, 24)

    def _toggle(self, card: ObjectCard) -> None:
        card.set_expanded(not card.expanded)
        self._opened[card.object_id] = card.expanded

    def _material(self, part_id: int, box: QComboBox) -> None:
        text = box.currentText()
        if text in self.labels.materials:
            self.material_set.emit(part_id, text)

    def _build_menu(self, kind: str, item_id: int) -> QMenu:
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
