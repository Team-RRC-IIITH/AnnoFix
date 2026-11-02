"""The window's look, after CVAT's (Ant Design): light grey bars, white panels, blue accents.
Colours are CVAT's own (cvat-ui/src/base.scss)."""

BLUE = "#1890ff"
BLUE_HOVER = "#40a9ff"
ACTIVE_BG = "#d8ecff"      # CVAT's active label background
BAR = "#f1f1f1"            # CVAT's background-color-2 (controls side bar)
BORDER = "#e4e4e4"
TEXT = "#303030"
TEXT_2 = "rgba(0, 0, 0, 0.45)"

STYLE = f"""
QMainWindow {{ background: #ffffff; }}
QWidget {{ color: {TEXT}; font-size: 13px; }}
QToolTip {{ color: #ffffff; background: rgba(0, 0, 0, 0.85); border: none; border-radius: 4px; padding: 4px 8px; }}

QToolBar#header {{ background: {BAR}; border: none; border-bottom: 1px solid {BORDER}; padding: 4px 10px; spacing: 4px; }}
QToolBar#tools {{ background: {BAR}; border: none; border-right: 1px solid {BORDER}; padding: 8px 5px; spacing: 6px; }}
QToolBar QToolButton {{ background: transparent; border: 1px solid transparent; border-radius: 6px; padding: 5px; }}
QToolBar QToolButton:hover {{ background: rgba(24, 144, 255, 0.08); }}
QToolBar QToolButton:checked {{ background: {ACTIVE_BG}; border-color: #91caff; }}
QToolBar QToolButton:disabled {{ background: transparent; }}
QToolBar#header QToolButton[primary="true"] {{ background: {BLUE}; color: #ffffff; border-color: {BLUE};
    padding: 5px 12px; font-weight: 600; }}
QToolBar#header QToolButton[primary="true"]:hover {{ background: {BLUE_HOVER}; border-color: {BLUE_HOVER}; }}
QToolBar::separator {{ background: {BORDER}; width: 1px; height: 1px; margin: 4px 6px; }}
QLabel#imageLabel {{ font-weight: 600; padding: 0 8px; }}
QLabel#progressLabel {{ color: {TEXT_2}; padding: 0 8px; }}

QFrame#panel {{ background: #ffffff; border: none; }}
QFrame#card {{ background: #ffffff; border: 1px solid #f0f0f0; border-radius: 8px; }}
QLabel#panelTitle {{ font-size: 14px; font-weight: 600; padding: 2px 2px 4px 2px; }}
QLabel#secondary {{ color: {TEXT_2}; }}

QPushButton {{ background: #ffffff; border: 1px solid #d9d9d9; border-radius: 6px; padding: 4px 10px; min-height: 18px; }}
QPushButton:hover {{ color: {BLUE_HOVER}; border-color: {BLUE_HOVER}; }}
QPushButton:pressed {{ color: #096dd9; border-color: #096dd9; }}
QPushButton:disabled {{ color: rgba(0, 0, 0, 0.25); background: #f5f5f5; border-color: #d9d9d9; }}
QPushButton[active="true"] {{ background: {BLUE}; color: #ffffff; border-color: {BLUE}; font-weight: 600; }}

QLineEdit, QComboBox {{ background: #ffffff; border: 1px solid #d9d9d9; border-radius: 6px; padding: 4px 8px; min-height: 18px; }}
QLineEdit:hover, QComboBox:hover {{ border-color: {BLUE_HOVER}; }}
QLineEdit:focus, QComboBox:focus {{ border-color: {BLUE_HOVER}; }}
QLineEdit:disabled, QComboBox:disabled {{ background: #f5f5f5; color: rgba(0, 0, 0, 0.25); }}
QComboBox::drop-down {{ border: none; width: 22px; }}
QComboBox QAbstractItemView {{ background: #ffffff; border: 1px solid #f0f0f0; selection-background-color: {ACTIVE_BG};
    selection-color: {TEXT}; outline: 0; }}

QTreeWidget, QListWidget {{ background: #ffffff; border: none; outline: 0; }}
QListWidget::item {{ min-height: 26px; border-radius: 4px; padding: 0 2px; }}
QListWidget::item:hover {{ background: rgba(24, 144, 255, 0.06); }}
QListWidget::item:selected {{ background: {ACTIVE_BG}; color: {TEXT}; }}
/* the tree sets only its row height (tree_panel.py): hover/selected rules would hide the colour it gives the selected part */
QHeaderView::section {{ background: #ffffff; border: none; border-bottom: 1px solid #f0f0f0; padding: 6px 4px;
    font-weight: 600; }}

QSlider::groove:horizontal {{ height: 4px; background: #e1e1e1; border-radius: 2px; }}
QSlider::sub-page:horizontal {{ background: {BLUE}; border-radius: 2px; }}
QSlider::handle:horizontal {{ width: 12px; height: 12px; margin: -5px 0; border-radius: 7px; background: #ffffff;
    border: 2px solid {BLUE}; }}

QSplitter::handle {{ background: {BORDER}; }}
QSplitter::handle:horizontal {{ width: 1px; }}
QScrollBar:vertical {{ background: transparent; width: 8px; margin: 0; }}
QScrollBar::handle:vertical {{ background: #c1c1c1; border-radius: 4px; min-height: 24px; }}
QScrollBar:horizontal {{ background: transparent; height: 8px; margin: 0; }}
QScrollBar::handle:horizontal {{ background: #c1c1c1; border-radius: 4px; min-width: 24px; }}
QScrollBar::add-line, QScrollBar::sub-line {{ width: 0; height: 0; }}
QScrollBar::add-page, QScrollBar::sub-page {{ background: transparent; }}

QMenu {{ background: #ffffff; border: 1px solid #f0f0f0; border-radius: 6px; padding: 4px; }}
QMenu::item {{ padding: 5px 18px; border-radius: 4px; }}
QMenu::item:selected {{ background: #e6f4ff; color: {TEXT}; }}
QStatusBar {{ background: {BAR}; border-top: 1px solid {BORDER}; }}
QStatusBar QLabel {{ color: rgba(0, 0, 0, 0.65); padding: 2px 6px; }}
"""
