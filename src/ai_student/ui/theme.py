"""深色现代风格主题：调色板 + QSS + 中文字体回退。

只做“外观”相关的事情，不碰业务逻辑。
apply_theme(app) 幂等，可在 QApplication 创建后随时调用。
"""
from __future__ import annotations

from PySide6.QtCore import QRectF, Qt
from PySide6.QtGui import QColor, QFont, QIcon, QLinearGradient, QPainter, QPalette, QPixmap
from PySide6.QtWidgets import QApplication, QPushButton, QWidget

# 中文字体回退链：Windows 优先微软雅黑，其次 Segoe UI，最后交给系统 sans-serif。
FONT_FAMILIES: list[str] = [
    "Microsoft YaHei UI",
    "Microsoft YaHei",
    "Segoe UI",
    "PingFang SC",
    "Noto Sans CJK SC",
    "Source Han Sans SC",
    "WenQuanYi Micro Hei",
    "sans-serif",
]

# 全局配色（QSS 与自绘控件共用同一份，避免两处颜色不一致）。
COLOR: dict[str, str] = {
    "bg": "#0f1115",
    "bg_alt": "#14171d",
    "surface": "#191d24",
    "surface_alt": "#1f242d",
    "elevated": "#252b36",
    "border": "#2b313c",
    "border_strong": "#3b4453",
    "text": "#e8ecf3",
    "text_dim": "#a7b0bf",
    "text_muted": "#78828f",
    "accent": "#4f8cff",
    "accent_dim": "#2f5fb5",
    "accent_soft": "#20304d",
    "success": "#3fb950",
    "warning": "#d9a13b",
    "danger": "#f0605f",
    "idle": "#78828f",
}


def color(key: str, fallback: str = "#ffffff") -> QColor:
    """按名字取 QColor。"""
    return QColor(COLOR.get(key, fallback))


def preferred_font_family() -> str:
    """挑选本机可用的首个中文字体名。"""
    try:
        from PySide6.QtGui import QFontDatabase

        families = set(QFontDatabase.families())
        for name in FONT_FAMILIES:
            if name in families:
                return name
    except Exception:  # pragma: no cover - 字体数据库异常时退回首选项
        pass
    return FONT_FAMILIES[0]


def app_font(point_size: int = 10) -> QFont:
    """带完整回退链的界面字体。"""
    font = QFont()
    font.setFamilies(list(FONT_FAMILIES))
    font.setFamily(preferred_font_family())
    font.setPointSize(point_size)
    font.setStyleStrategy(QFont.StyleStrategy.PreferAntialias)
    return font


def dark_palette() -> QPalette:
    """深色调色板（原生控件在 QSS 未覆盖处也不刺眼）。"""
    pal = QPalette()
    pal.setColor(QPalette.ColorRole.Window, color("bg"))
    pal.setColor(QPalette.ColorRole.WindowText, color("text"))
    pal.setColor(QPalette.ColorRole.Base, color("bg_alt"))
    pal.setColor(QPalette.ColorRole.AlternateBase, color("surface"))
    pal.setColor(QPalette.ColorRole.ToolTipBase, color("elevated"))
    pal.setColor(QPalette.ColorRole.ToolTipText, color("text"))
    pal.setColor(QPalette.ColorRole.Text, color("text"))
    pal.setColor(QPalette.ColorRole.Button, color("surface_alt"))
    pal.setColor(QPalette.ColorRole.ButtonText, color("text"))
    pal.setColor(QPalette.ColorRole.BrightText, color("danger"))
    pal.setColor(QPalette.ColorRole.Link, color("accent"))
    pal.setColor(QPalette.ColorRole.Highlight, color("accent"))
    pal.setColor(QPalette.ColorRole.HighlightedText, QColor("#0b0d11"))
    pal.setColor(QPalette.ColorRole.PlaceholderText, color("text_muted"))
    for role in (QPalette.ColorRole.WindowText, QPalette.ColorRole.Text, QPalette.ColorRole.ButtonText):
        pal.setColor(QPalette.ColorGroup.Disabled, role, color("text_muted"))
    return pal


def app_icon_pixmap(size: int = 128) -> QPixmap:
    """用 QPainter 画应用/托盘图标（不引入任何图片资源文件）。

    造型：圆角深色底 + 蓝色渐变描边 + 白色“学”字形折线（课本 + 对勾的抽象）。
    """
    size = max(16, int(size))
    pm = QPixmap(size, size)
    pm.fill(Qt.GlobalColor.transparent)
    p = QPainter(pm)
    p.setRenderHint(QPainter.RenderHint.Antialiasing, True)

    margin = size * 0.06
    body = QRectF(margin, margin, size - 2 * margin, size - 2 * margin)
    grad = QLinearGradient(body.topLeft(), body.bottomRight())
    grad.setColorAt(0.0, QColor("#22304a"))
    grad.setColorAt(1.0, QColor("#121722"))
    p.setBrush(grad)
    p.setPen(QColor(COLOR["accent"]))
    pen = p.pen()
    pen.setWidthF(max(1.0, size * 0.035))
    p.setPen(pen)
    p.drawRoundedRect(body, size * 0.24, size * 0.24)

    # 书本（两条斜线 + 一条底线）
    p.setPen(QColor("#dfe8ff"))
    pen = p.pen()
    pen.setWidthF(max(1.0, size * 0.055))
    pen.setCapStyle(Qt.PenCapStyle.RoundCap)
    p.setPen(pen)
    left = size * 0.26
    right = size * 0.74
    top = size * 0.40
    bottom = size * 0.62
    p.drawLine(int(left + size * 0.10), int(top), int(left), int(bottom))
    p.drawLine(int(left + size * 0.10), int(top), int(right - size * 0.10), int(top))
    p.drawLine(int(right - size * 0.10), int(top), int(right), int(bottom))
    p.drawLine(int(left), int(bottom), int(right), int(bottom))

    # 对勾
    p.setPen(QColor(COLOR["success"]))
    pen = p.pen()
    pen.setWidthF(max(1.0, size * 0.06))
    p.setPen(pen)
    p.drawLine(int(size * 0.34), int(size * 0.74), int(size * 0.45), int(size * 0.84))
    p.drawLine(int(size * 0.45), int(size * 0.84), int(size * 0.68), int(size * 0.60))
    p.end()
    return pm


def app_icon(size: int = 128) -> QIcon:
    """应用图标（窗口与托盘共用）。"""
    return QIcon(app_icon_pixmap(size))


def apply_button_role(button: QPushButton, role: str = "primary") -> QPushButton:
    """给按钮套用主题角色：primary | danger | ghost。"""
    button.setObjectName({"primary": "primaryButton", "danger": "dangerButton"}.get(role, "ghostButton"))
    return button


def build_stylesheet() -> str:
    """返回全局 QSS 字符串。"""
    c = COLOR
    return f"""
* {{
    outline: 0;
}}
QWidget {{
    color: {c['text']};
    background-color: {c['bg']};
    font-family: "Microsoft YaHei UI", "Microsoft YaHei", "Segoe UI", sans-serif;
}}
QMainWindow, QDialog, QWizard {{
    background-color: {c['bg']};
}}
QToolTip {{
    color: {c['text']};
    background-color: {c['elevated']};
    border: 1px solid {c['border_strong']};
    padding: 4px 8px;
    border-radius: 6px;
}}
QLabel {{
    background: transparent;
}}
QLabel#pageTitle {{
    font-size: 17px;
    font-weight: 600;
    color: {c['text']};
}}
QLabel#sectionTitle {{
    font-size: 13px;
    font-weight: 600;
    color: {c['text_dim']};
}}
QLabel#hint, QLabel#fieldHint {{
    color: {c['text_muted']};
    font-size: 12px;
}}
QLabel#monoValue {{
    font-family: "Consolas", "Cascadia Mono", monospace;
    color: {c['text_dim']};
}}
/* ---------------- 卡片 / 分组 ---------------- */
QFrame#card {{
    background-color: {c['surface']};
    border: 1px solid {c['border']};
    border-radius: 10px;
}}
QFrame#navRail {{
    background-color: {c['bg_alt']};
    border: none;
    border-right: 1px solid {c['border']};
}}
QFrame#banner {{
    background-color: #3a2a12;
    border: 1px solid {c['warning']};
    border-radius: 8px;
}}
QFrame#bannerError {{
    background-color: #3a1a1c;
    border: 1px solid {c['danger']};
    border-radius: 8px;
}}
QLabel#bannerText {{
    color: #ffd88a;
    font-weight: 600;
}}
QLabel#previewLabel {{
    background-color: #0b0d11;
    border: 1px dashed {c['border_strong']};
    border-radius: 8px;
    color: {c['text_muted']};
}}
QGroupBox {{
    border: 1px solid {c['border']};
    border-radius: 8px;
    margin-top: 14px;
    padding: 10px 8px 8px 8px;
    background-color: {c['surface']};
}}
QGroupBox::title {{
    subcontrol-origin: margin;
    left: 10px;
    padding: 0 4px;
    color: {c['text_dim']};
    font-weight: 600;
}}
/* ---------------- 按钮 ---------------- */
QPushButton {{
    background-color: {c['surface_alt']};
    color: {c['text']};
    border: 1px solid {c['border_strong']};
    border-radius: 6px;
    padding: 5px 14px;
    min-height: 18px;
}}
QPushButton:hover {{
    background-color: {c['elevated']};
    border-color: {c['accent_dim']};
}}
QPushButton:pressed {{
    background-color: #1a1f28;
}}
QPushButton:disabled {{
    color: {c['text_muted']};
    background-color: #1a1e26;
    border-color: {c['border']};
}}
QPushButton:checked {{
    background-color: {c['accent_soft']};
    border-color: {c['accent']};
    color: {c['text']};
}}
QPushButton#primaryButton {{
    background-color: {c['accent']};
    border: 1px solid {c['accent']};
    color: #0b0d11;
    font-weight: 600;
}}
QPushButton#primaryButton:hover {{
    background-color: #6ba0ff;
}}
QPushButton#primaryButton:disabled {{
    background-color: {c['accent_dim']};
    border-color: {c['accent_dim']};
    color: #c8d3e6;
}}
QPushButton#dangerButton {{
    background-color: #3a1a1c;
    border: 1px solid {c['danger']};
    color: #ffb3b2;
}}
QPushButton#dangerButton:hover {{
    background-color: #4a2023;
}}
QPushButton#ghostButton {{
    background-color: transparent;
    border: 1px solid {c['border']};
    color: {c['text_dim']};
}}
QPushButton#ghostButton:hover {{
    border-color: {c['accent_dim']};
    color: {c['text']};
}}
QPushButton#tableAction {{
    padding: 2px 8px;
    font-size: 12px;
    min-height: 14px;
    border-radius: 5px;
}}
/* ---------------- 输入控件 ---------------- */
QLineEdit, QPlainTextEdit, QTextEdit, QSpinBox, QDoubleSpinBox, QComboBox, QFontComboBox {{
    background-color: {c['bg_alt']};
    border: 1px solid {c['border_strong']};
    border-radius: 6px;
    padding: 4px 8px;
    selection-background-color: {c['accent']};
    selection-color: #0b0d11;
}}
QLineEdit:focus, QPlainTextEdit:focus, QTextEdit:focus, QSpinBox:focus,
QDoubleSpinBox:focus, QComboBox:focus {{
    border-color: {c['accent']};
}}
QLineEdit:disabled, QPlainTextEdit:disabled, QComboBox:disabled, QSpinBox:disabled {{
    color: {c['text_muted']};
    background-color: #15181e;
}}
QLineEdit[echoMode="2"] {{
    lineedit-password-character: 9679;
}}
QComboBox::drop-down {{
    border: none;
    width: 20px;
}}
QComboBox QAbstractItemView {{
    background-color: {c['elevated']};
    border: 1px solid {c['border_strong']};
    selection-background-color: {c['accent_soft']};
    selection-color: {c['text']};
    outline: 0;
}}
QSpinBox::up-button, QDoubleSpinBox::up-button,
QSpinBox::down-button, QDoubleSpinBox::down-button {{
    background-color: {c['surface_alt']};
    border: none;
    width: 16px;
}}
QCheckBox, QRadioButton {{
    spacing: 7px;
    background: transparent;
}}
QCheckBox::indicator, QRadioButton::indicator {{
    width: 15px;
    height: 15px;
    border: 1px solid {c['border_strong']};
    background-color: {c['bg_alt']};
}}
QCheckBox::indicator {{
    border-radius: 4px;
}}
QRadioButton::indicator {{
    border-radius: 8px;
}}
QCheckBox::indicator:checked, QRadioButton::indicator:checked {{
    background-color: {c['accent']};
    border-color: {c['accent']};
}}
QCheckBox::indicator:hover, QRadioButton::indicator:hover {{
    border-color: {c['accent']};
}}
/* ---------------- 导航 ---------------- */
QListWidget#navList {{
    background-color: {c['bg_alt']};
    border: none;
    outline: 0;
    padding: 8px 6px;
}}
QListWidget#navList::item {{
    color: {c['text_dim']};
    border-radius: 8px;
    padding: 10px 12px;
    margin: 2px 0;
}}
QListWidget#navList::item:hover {{
    background-color: {c['surface']};
    color: {c['text']};
}}
QListWidget#navList::item:selected {{
    background-color: {c['accent_soft']};
    color: {c['text']};
    border-left: 3px solid {c['accent']};
}}
/* ---------------- 列表 / 表格 ---------------- */
QListWidget, QTreeWidget, QTableWidget, QListView, QTreeView, QTableView {{
    background-color: {c['bg_alt']};
    border: 1px solid {c['border']};
    border-radius: 8px;
    alternate-background-color: {c['surface']};
    gridline-color: {c['border']};
    outline: 0;
}}
QListWidget::item, QTreeWidget::item {{
    padding: 5px 6px;
    border-radius: 6px;
}}
QListWidget::item:selected, QTreeWidget::item:selected,
QTableWidget::item:selected, QTableView::item:selected {{
    background-color: {c['accent_soft']};
    color: {c['text']};
}}
QTableWidget::item {{
    padding: 4px 6px;
}}
QHeaderView::section {{
    background-color: {c['surface_alt']};
    color: {c['text_dim']};
    border: none;
    border-right: 1px solid {c['border']};
    border-bottom: 1px solid {c['border']};
    padding: 6px 8px;
    font-weight: 600;
}}
QTableCornerButton::section {{
    background-color: {c['surface_alt']};
    border: none;
}}
QListWidget#historyList::item {{
    padding: 8px 10px;
    border-bottom: 1px solid {c['border']};
    border-radius: 0;
}}
/* ---------------- 标签页 ---------------- */
QTabWidget::pane {{
    border: 1px solid {c['border']};
    border-radius: 8px;
    background-color: {c['surface']};
    top: -1px;
}}
QTabBar::tab {{
    background-color: transparent;
    color: {c['text_muted']};
    padding: 7px 16px;
    margin-right: 2px;
    border: 1px solid transparent;
    border-top-left-radius: 7px;
    border-top-right-radius: 7px;
}}
QTabBar::tab:hover {{
    color: {c['text']};
    background-color: {c['surface']};
}}
QTabBar::tab:selected {{
    color: {c['text']};
    background-color: {c['surface']};
    border-color: {c['border']};
    border-bottom-color: {c['surface']};
}}
/* ---------------- 滚动条 ---------------- */
QScrollBar:vertical {{
    background: transparent;
    width: 10px;
    margin: 2px;
}}
QScrollBar:horizontal {{
    background: transparent;
    height: 10px;
    margin: 2px;
}}
QScrollBar::handle:vertical, QScrollBar::handle:horizontal {{
    background-color: {c['border_strong']};
    border-radius: 5px;
    min-height: 28px;
    min-width: 28px;
}}
QScrollBar::handle:hover {{
    background-color: {c['accent_dim']};
}}
QScrollBar::add-line, QScrollBar::sub-line {{
    height: 0;
    width: 0;
    background: none;
    border: none;
}}
QScrollBar::add-page, QScrollBar::sub-page {{
    background: none;
}}
/* ---------------- 其它 ---------------- */
QProgressBar {{
    background-color: {c['bg_alt']};
    border: 1px solid {c['border']};
    border-radius: 6px;
    text-align: center;
    color: {c['text_dim']};
    min-height: 14px;
}}
QProgressBar::chunk {{
    background-color: {c['accent']};
    border-radius: 5px;
}}
QMenuBar {{
    background-color: {c['bg_alt']};
    border-bottom: 1px solid {c['border']};
}}
QMenuBar::item {{
    background: transparent;
    padding: 6px 12px;
    color: {c['text_dim']};
}}
QMenuBar::item:selected {{
    background-color: {c['surface_alt']};
    color: {c['text']};
    border-radius: 6px;
}}
QMenu {{
    background-color: {c['elevated']};
    border: 1px solid {c['border_strong']};
    border-radius: 8px;
    padding: 4px;
}}
QMenu::item {{
    padding: 6px 22px 6px 14px;
    border-radius: 6px;
    color: {c['text']};
}}
QMenu::item:selected {{
    background-color: {c['accent_soft']};
}}
QMenu::separator {{
    height: 1px;
    background: {c['border']};
    margin: 4px 8px;
}}
QStatusBar {{
    background-color: {c['bg_alt']};
    border-top: 1px solid {c['border']};
    color: {c['text_muted']};
}}
QStatusBar::item {{
    border: none;
}}
QToolBar {{
    background-color: {c['bg_alt']};
    border-bottom: 1px solid {c['border']};
    spacing: 6px;
    padding: 4px;
}}
QSplitter::handle {{
    background-color: {c['border']};
}}
QSplitter::handle:horizontal {{
    width: 3px;
}}
QSplitter::handle:vertical {{
    height: 3px;
}}
QWizard QWizardPage {{
    background-color: {c['bg']};
}}
QPlainTextEdit#notesView, QPlainTextEdit#logView, QTextBrowser {{
    background-color: #0b0d11;
    border: 1px solid {c['border']};
    border-radius: 8px;
}}
QScrollArea {{
    background-color: transparent;
    border: none;
}}
"""


def apply_theme(app: QApplication | QWidget | None = None) -> QApplication | None:
    """把深色主题应用到 QApplication。

    参数可以是 QApplication，也可以省略（取已有实例）。返回 QApplication，
    没有实例时返回 None（不抛异常，方便在无 GUI 环境下调用）。
    """
    if isinstance(app, QWidget):  # 容错：误传窗口时退回当前应用实例
        app = QApplication.instance()
    application = app if isinstance(app, QApplication) else QApplication.instance()
    if application is None:
        return None
    try:
        application.setStyle("Fusion")
    except Exception:  # pragma: no cover - 某些平台无 Fusion 样式
        pass
    application.setPalette(dark_palette())
    application.setFont(app_font())
    application.setStyleSheet(build_stylesheet())
    return application
