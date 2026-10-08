"""可复用自绘控件与异步执行小工具。

约定：
- 所有跨线程调用都用 run_async()（QRunnable + QThreadPool），回调在主线程触发，
  绝不在 UI 线程里做 probe / 网络 / 文件大读写。
- 自绘控件只依赖 theme.COLOR，不引入图片资源。
"""
from __future__ import annotations

import dataclasses
from typing import Any, Callable

from PySide6.QtCore import QObject, QRectF, QRunnable, Qt, QThreadPool, Signal
from PySide6.QtGui import QColor, QLinearGradient, QPainter, QPen
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPlainTextEdit,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from .theme import COLOR, apply_button_role

# ---------------------------------------------------------------- 通用小工具

LEVEL_LABELS = {"DEBUG": "调试", "INFO": "信息", "WARNING": "警告", "ERROR": "错误"}
LEVEL_COLORS = {
    "DEBUG": COLOR["text_muted"],
    "INFO": COLOR["text_dim"],
    "WARNING": COLOR["warning"],
    "ERROR": COLOR["danger"],
}


def hms(seconds: float) -> str:
    """秒 -> HH:MM:SS（与 core.TranscriptSegment.timecode 保持一致）。"""
    try:
        s = max(0, int(float(seconds)))
    except (TypeError, ValueError):
        s = 0
    return f"{s // 3600:02d}:{(s % 3600) // 60:02d}:{s % 60:02d}"


def as_dict(payload: Any) -> dict[str, Any]:
    """把信号载荷统一成 dict：兼容 dict / dataclass / 带 to_dict() 的对象。

    引擎契约规定信号携带 to_dict() 结果，但 UI 必须对 None、异常载荷保持健壮，
    任何时候都不允许因为载荷形状不对而崩溃。
    """
    if payload is None:
        return {}
    if isinstance(payload, dict):
        return dict(payload)
    to_dict = getattr(payload, "to_dict", None)
    if callable(to_dict):
        try:
            data = to_dict()
            if isinstance(data, dict):
                return dict(data)
        except Exception:
            return {}
    if dataclasses.is_dataclass(payload):
        try:
            return dataclasses.asdict(payload)
        except Exception:
            return {}
    return {}


def get_str(data: dict[str, Any], key: str, default: str = "") -> str:
    """安全取字符串字段。"""
    value = data.get(key, default)
    if value is None:
        return default
    return str(value)


def get_float(data: dict[str, Any], key: str, default: float = 0.0) -> float:
    """安全取浮点字段。"""
    value = data.get(key, default)
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def make_button(text: str, role: str = "", tooltip: str = "") -> QPushButton:
    """创建带主题角色的按钮。"""
    btn = QPushButton(text)
    if role:
        apply_button_role(btn, role)
    if tooltip:
        btn.setToolTip(tooltip)
    btn.setCursor(Qt.CursorShape.PointingHandCursor)
    return btn


def clear_layout(layout) -> None:
    """清空布局内所有控件（含嵌套布局）。"""
    if layout is None:
        return
    while layout.count():
        item = layout.takeAt(0)
        widget = item.widget()
        child = item.layout()
        if widget is not None:
            widget.setParent(None)
            widget.deleteLater()
        elif child is not None:
            clear_layout(child)


# ---------------------------------------------------------------- 异步执行


class _WorkerSignals(QObject):
    """worker 的完成/失败信号（在主线程创建，保证回调派发到主线程）。"""

    finished = Signal(object)
    failed = Signal(str)


class AsyncWorker(QRunnable):
    """在 QThreadPool 里执行一个可调用对象。

    autoDelete 关闭，生命周期由模块级 _LIVE_WORKERS 托管，避免 C++ 对象
    先于 Python 引用被销毁导致回调期崩溃。
    """

    def __init__(self, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> None:
        super().__init__()
        self.signals = _WorkerSignals()
        self._fn = fn
        self._args = args
        self._kwargs = kwargs
        self.setAutoDelete(False)

    def run(self) -> None:  # pragma: no cover - 线程内执行
        try:
            result = self._fn(*self._args, **self._kwargs)
        except Exception as exc:  # 任何异常都转成信号，绝不让线程崩掉进程
            try:
                self.signals.failed.emit(f"{type(exc).__name__}: {exc}")
            except Exception:
                pass
        else:
            try:
                self.signals.finished.emit(result)
            except Exception:
                pass


_LIVE_WORKERS: set[AsyncWorker] = set()


def run_async(
    fn: Callable[..., Any],
    on_result: Callable[[Any], None] | None = None,
    on_error: Callable[[str], None] | None = None,
    *args: Any,
    **kwargs: Any,
) -> AsyncWorker:
    """把耗时函数丢到全局线程池执行；结果回到主线程回调。"""
    worker = AsyncWorker(fn, *args, **kwargs)
    _LIVE_WORKERS.add(worker)

    def _release(*_ignored: Any) -> None:
        _LIVE_WORKERS.discard(worker)

    if on_result is not None:
        worker.signals.finished.connect(on_result)
    if on_error is not None:
        worker.signals.failed.connect(on_error)
    worker.signals.finished.connect(_release)
    worker.signals.failed.connect(_release)
    QThreadPool.globalInstance().start(worker)
    return worker


# ---------------------------------------------------------------- 自绘控件


class Card(QFrame):
    """带标题的圆角卡片容器。"""

    def __init__(self, title: str = "", parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("card")
        self._outer = QVBoxLayout(self)
        self._outer.setContentsMargins(12, 10, 12, 12)
        self._outer.setSpacing(8)
        self._title = QLabel(title)
        self._title.setObjectName("sectionTitle")
        self._title.setVisible(bool(title))
        self._outer.addWidget(self._title)
        self._body = QVBoxLayout()
        self._body.setContentsMargins(0, 0, 0, 0)
        self._body.setSpacing(8)
        self._outer.addLayout(self._body)

    def body(self) -> QVBoxLayout:
        """卡片主体布局（往里塞控件）。"""
        return self._body

    def set_title(self, text: str) -> None:
        self._title.setText(text)
        self._title.setVisible(bool(text))

    def add_widget(self, widget: QWidget, stretch: int = 0) -> QWidget:
        self._body.addWidget(widget, stretch)
        return widget


class StatusPill(QWidget):
    """状态胶囊：待机 / 听课中 / 暂停 / 收尾 / 已结束 / 出错。"""

    STATE_TEXT = {
        "idle": "待机",
        "recording": "听课中",
        "paused": "已暂停",
        "finalizing": "收尾中",
        "finished": "已结束",
        "error": "出错",
    }
    STATE_COLOR = {
        "idle": COLOR["idle"],
        "recording": COLOR["success"],
        "paused": COLOR["warning"],
        "finalizing": COLOR["accent"],
        "finished": COLOR["text_dim"],
        "error": COLOR["danger"],
    }

    def __init__(self, parent: QWidget | None = None, state: str = "idle") -> None:
        super().__init__(parent)
        self._state = state if state in self.STATE_TEXT else "idle"
        self._text = self.STATE_TEXT[self._state]
        self.setMinimumHeight(26)
        self.setSizePolicy(QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Fixed)

    def set_status(self, state: str, text: str = "") -> None:
        """state 为 SessionState；text 非空时覆盖默认中文说明。"""
        key = str(state or "idle")
        self._state = key if key in self.STATE_TEXT else "idle"
        self._text = text.strip() or self.STATE_TEXT[self._state]
        self.update()

    def state(self) -> str:
        return self._state

    def text(self) -> str:
        return self._text

    def color(self) -> QColor:
        return QColor(self.STATE_COLOR.get(self._state, COLOR["idle"]))

    def sizeHint(self):  # noqa: N802 - Qt 命名
        from PySide6.QtCore import QSize

        width = self.fontMetrics().horizontalAdvance(self._text) + 46
        return QSize(max(96, width), 26)

    def paintEvent(self, event) -> None:  # noqa: N802 - Qt 命名
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        rect = QRectF(self.rect()).adjusted(0.5, 0.5, -0.5, -0.5)
        hue = self.color()
        fill = QColor(hue)
        fill.setAlpha(38)
        painter.setBrush(fill)
        painter.setPen(QPen(hue, 1))
        painter.drawRoundedRect(rect, rect.height() / 2, rect.height() / 2)

        dot = QRectF(rect.left() + 10, rect.center().y() - 4, 8, 8)
        painter.setBrush(hue)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawEllipse(dot)

        painter.setPen(QPen(hue))
        text_rect = rect.adjusted(24, 0, -10, 0)
        painter.drawText(text_rect, int(Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft), self._text)
        painter.end()


class LevelMeter(QWidget):
    """麦克风电平条（-60 ~ 0 dB），用于听课页实时反馈。"""

    FLOOR_DB = -60.0

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._db = self.FLOOR_DB
        self._peak = self.FLOOR_DB
        self._active = False
        self._caption = "未采集"
        self.setMinimumHeight(30)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

    def set_level(self, db: float, caption: str = "") -> None:
        """设置当前电平；db 为负分贝，<= -60 视为静音。"""
        try:
            value = float(db)
        except (TypeError, ValueError):
            value = self.FLOOR_DB
        self._db = max(self.FLOOR_DB, min(0.0, value))
        self._peak = max(self._peak * 0.92, self._db) if self._active else self.FLOOR_DB
        self._active = True
        self._caption = caption or f"{self._db:5.1f} dB"
        self.update()

    def set_idle(self, caption: str = "未采集") -> None:
        """停止采集状态。"""
        self._db = self.FLOOR_DB
        self._peak = self.FLOOR_DB
        self._active = False
        self._caption = caption
        self.update()

    def is_active(self) -> bool:
        return self._active

    def value_db(self) -> float:
        return self._db

    def paintEvent(self, event) -> None:  # noqa: N802 - Qt 命名
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        rect = QRectF(self.rect()).adjusted(0.5, 0.5, -0.5, -0.5)
        painter.setPen(QPen(QColor(COLOR["border"]), 1))
        painter.setBrush(QColor("#0b0d11"))
        painter.drawRoundedRect(rect, 6, 6)

        inner = rect.adjusted(3, 3, -3, -3)
        if self._active and inner.width() > 4:
            ratio = (self._db - self.FLOOR_DB) / (0.0 - self.FLOOR_DB)
            ratio = max(0.0, min(1.0, ratio))
            bar = QRectF(inner.left(), inner.top(), inner.width() * ratio, inner.height())
            grad = QLinearGradient(inner.topLeft(), inner.topRight())
            grad.setColorAt(0.0, QColor(COLOR["success"]))
            grad.setColorAt(0.72, QColor(COLOR["warning"]))
            grad.setColorAt(1.0, QColor(COLOR["danger"]))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(grad)
            painter.drawRoundedRect(bar, 4, 4)
        elif inner.width() > 4:
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QColor("#161a21"))
            painter.drawRoundedRect(inner, 4, 4)
            painter.setPen(QPen(QColor(COLOR["text_muted"])))
            painter.drawText(rect, int(Qt.AlignmentFlag.AlignCenter), "未开始采集")

        if self._active:
            painter.setPen(QPen(QColor(COLOR["text_dim"])))
            painter.drawText(
                rect.adjusted(10, 0, -10, 0),
                int(Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignRight),
                self._caption,
            )
        painter.end()


class LogView(QPlainTextEdit):
    """只读日志视图：按级别着色，自动截断历史行数。"""

    MAX_BLOCKS = 4000
    LEVEL_ORDER = {"DEBUG": 0, "INFO": 1, "WARNING": 2, "ERROR": 3}

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("logView")
        self.setReadOnly(True)
        self.setUndoRedoEnabled(False)
        self.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)
        self.setMaximumBlockCount(self.MAX_BLOCKS)
        self.setPlaceholderText("引擎日志会显示在这里……")
        self._min_level = "DEBUG"
        self._lines: list[str] = []

    def append_log(self, level: str, message: str) -> None:
        """追加一条日志（level 不在已知集合时按 INFO 处理）。"""
        lvl = str(level or "INFO").upper()
        if lvl not in self.LEVEL_ORDER:
            lvl = "INFO"
        self._lines.append(f"[{LEVEL_LABELS.get(lvl, lvl)}] {message}")
        if len(self._lines) > self.MAX_BLOCKS:
            del self._lines[: len(self._lines) - self.MAX_BLOCKS]
        if self.LEVEL_ORDER[lvl] < self.LEVEL_ORDER.get(self._min_level, 0):
            return
        color = LEVEL_COLORS.get(lvl, COLOR["text_dim"])
        escaped = (
            str(message)
            .replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace(" ", "&nbsp;")
        )
        self.appendHtml(f'<span style="color:{color}">[{LEVEL_LABELS.get(lvl, lvl)}] {escaped}</span>')
        self.verticalScrollBar().setValue(self.verticalScrollBar().maximum())

    def set_min_level(self, level: str) -> None:
        """设置显示级别下限（DEBUG/INFO/WARNING/ERROR）。"""
        self._min_level = level if level in self.LEVEL_ORDER else "DEBUG"
        self._rerender()

    def min_level(self) -> str:
        return self._min_level

    def _rerender(self) -> None:
        self.clear()
        keep = list(self._lines)
        self._lines = []
        for line in keep:
            level = "INFO"
            for key, label in LEVEL_LABELS.items():
                if line.startswith(f"[{label}]"):
                    level = key
                    break
            self.append_log(level, line.split("] ", 1)[-1])

    def export_to(self, path: str) -> str:
        """导出纯文本日志，返回写入路径。"""
        from pathlib import Path

        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("\n".join(self._lines) + ("\n" if self._lines else ""), encoding="utf-8")
        return str(target)

    def text_lines(self) -> list[str]:
        """返回全部日志行（纯文本，忽略级别过滤）。"""
        return list(self._lines)

    def clear_log(self) -> None:
        self._lines.clear()
        self.clear()


class TimecodeList(QListWidget):
    """带时间码的转写列表：支持搜索过滤与自动滚动。"""

    MAX_ITEMS = 5000

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setSelectionMode(QListWidget.SelectionMode.ExtendedSelection)
        self.setWordWrap(True)
        self.setUniformItemSizes(False)
        self.setAlternatingRowColors(True)
        self.setToolTip("课堂实时转写会显示在这里")
        self._auto_scroll = True
        self._filter = ""

    def set_auto_scroll(self, enabled: bool) -> None:
        self._auto_scroll = bool(enabled)

    def add_segment(self, segment: Any) -> None:
        """加入一段转写。segment 可以是 dict 或对象。"""
        data = as_dict(segment)
        text = get_str(data, "text").strip()
        timecode = get_str(data, "timecode") or hms(get_float(data, "t_start"))
        speaker = get_str(data, "speaker")
        prefix = f"{timecode}  {speaker}" if speaker else timecode
        item = QListWidgetItem(f"{prefix}   {text}".rstrip())
        item.setData(Qt.ItemDataRole.UserRole, {"timecode": timecode, "speaker": speaker, "text": text})
        item.setToolTip(text)
        hidden = bool(self._filter) and self._filter not in f"{prefix} {text}".lower()
        item.setHidden(hidden)
        self.addItem(item)
        while self.count() > self.MAX_ITEMS:
            self.takeItem(0)
        if self._auto_scroll and not hidden:
            self.scrollToBottom()

    def apply_filter(self, text: str) -> int:
        """按关键字过滤，返回命中条数。"""
        self._filter = (text or "").strip().lower()
        hits = 0
        for row in range(self.count()):
            item = self.item(row)
            data = item.data(Qt.ItemDataRole.UserRole) or {}
            haystack = f"{data.get('timecode', '')} {data.get('speaker', '')} {data.get('text', '')}".lower()
            hit = (not self._filter) or (self._filter in haystack)
            item.setHidden(not hit)
            hits += 1 if hit else 0
        return hits

    def segments(self) -> list[dict[str, Any]]:
        """返回当前所有条目数据（供导出/断言）。"""
        return [self.item(row).data(Qt.ItemDataRole.UserRole) or {} for row in range(self.count())]

    def clear_all(self) -> None:
        self.clear()


class LabeledRow(QWidget):
    """一行“标签 + 控件”，设置页大量复用。"""

    def __init__(self, label: str, widget: QWidget, hint: str = "", parent: QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)
        self.label = QLabel(label)
        self.label.setMinimumWidth(132)
        layout.addWidget(self.label)
        layout.addWidget(widget, 1)
        self.hint = QLabel(hint)
        self.hint.setObjectName("fieldHint")
        self.hint.setVisible(bool(hint))
        layout.addWidget(self.hint, 0)
