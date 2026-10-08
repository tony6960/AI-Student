"""系统托盘：代码绘制的图标 + 常用操作菜单。

图标完全由 QPainter 生成（见 theme.app_icon_pixmap），不引入任何图片资源文件。
系统托盘不可用时（例如 offscreen 平台）所有方法都要安全降级，不抛异常。
"""
from __future__ import annotations

from PySide6.QtCore import QObject, Signal
from PySide6.QtGui import QAction, QIcon, QPixmap
from PySide6.QtWidgets import QApplication, QMenu, QSystemTrayIcon, QWidget

from .theme import app_icon_pixmap


def tray_icon_pixmap(size: int = 64) -> QPixmap:
    """托盘图标（复用主题里画的图形）。"""
    return app_icon_pixmap(size)


class TrayController(QObject):
    """托盘图标 + 菜单：显示主窗口 / 开始听课 / 暂停 / 停止 / 退出。"""

    quitRequested = Signal()

    def __init__(self, api, window: QWidget | None = None, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._api = api
        self._window = window

        self.icon = QSystemTrayIcon(QIcon(tray_icon_pixmap(64)), self)
        self.icon.setToolTip("AI学生 · 待机")
        self.menu = QMenu()

        self.act_show = QAction("显示主窗口", self)
        self.act_start = QAction("开始听课", self)
        self.act_pause = QAction("暂停 / 恢复", self)
        self.act_stop = QAction("停止并收尾", self)
        self.act_quit = QAction("退出", self)
        self.menu.addAction(self.act_show)
        self.menu.addSeparator()
        self.menu.addAction(self.act_start)
        self.menu.addAction(self.act_pause)
        self.menu.addAction(self.act_stop)
        self.menu.addSeparator()
        self.menu.addAction(self.act_quit)
        self.icon.setContextMenu(self.menu)

        self.act_show.triggered.connect(self.show_window)
        self.act_start.triggered.connect(self.start_session)
        self.act_pause.triggered.connect(self.toggle_pause)
        self.act_stop.triggered.connect(self.stop_session)
        self.act_quit.triggered.connect(self.request_quit)
        self.icon.activated.connect(self._on_activated)

        # 跟随引擎状态刷新提示文字与菜单可用性
        state_signal = getattr(self._api, "stateChanged", None)
        if state_signal is not None:
            try:
                state_signal.connect(self.update_state)
            except Exception:  # pragma: no cover - 非标准引擎
                pass

        self._state = "idle"
        self.update_state("idle")
        # app.py 只负责构造 TrayController（是否创建由系统托盘可用性决定），
        # 这里自动显示图标，避免“创建了却看不见”。
        self.show()

    # ------------------------------------------------------------ 显隐
    def is_available(self) -> bool:
        """系统托盘是否可用（offscreen/无桌面环境时为 False）。"""
        try:
            return bool(QSystemTrayIcon.isSystemTrayAvailable())
        except Exception:  # pragma: no cover
            return False

    def is_visible(self) -> bool:
        try:
            return bool(self.icon.isVisible())
        except Exception:  # pragma: no cover
            return False

    def show(self) -> None:
        """显示托盘图标（不可用时静默跳过）。"""
        try:
            self.icon.show()
        except Exception:  # pragma: no cover - 无托盘环境
            pass

    def hide(self) -> None:
        try:
            self.icon.hide()
        except Exception:  # pragma: no cover
            pass

    def notify(self, title: str, message: str) -> None:
        """弹一条系统通知（不支持时忽略）。"""
        try:
            if self.icon.supportsMessages():
                self.icon.showMessage(title, message, QSystemTrayIcon.MessageIcon.Information, 3000)
        except Exception:  # pragma: no cover
            pass

    # ------------------------------------------------------------ 状态
    def update_state(self, state: str, message: str = "") -> None:
        """根据会话状态刷新提示文字与菜单可用性。"""
        self._state = str(state or "idle")
        text = {
            "idle": "待机",
            "recording": "听课中",
            "paused": "已暂停",
            "finalizing": "收尾中",
            "finished": "已结束",
            "error": "出错",
        }.get(self._state, self._state)
        try:
            self.icon.setToolTip(f"AI学生 · {text}" + (f" · {message}" if message else ""))
        except Exception:  # pragma: no cover
            pass
        active = self._state in ("recording", "paused", "finalizing")
        self.act_start.setEnabled(not active)
        self.act_pause.setEnabled(active)
        self.act_stop.setEnabled(active)
        self.act_pause.setText("恢复听课" if self._state == "paused" else "暂停听课")

    def state(self) -> str:
        return self._state

    # ------------------------------------------------------------ 动作
    def _on_activated(self, reason) -> None:
        try:
            if reason in (
                QSystemTrayIcon.ActivationReason.Trigger,
                QSystemTrayIcon.ActivationReason.DoubleClick,
            ):
                self.show_window()
        except Exception:  # pragma: no cover
            pass

    def show_window(self) -> None:
        """把主窗口带到前台。"""
        window = self._window
        if window is None:
            return
        try:
            window.showNormal()
            window.raise_()
            window.activateWindow()
        except Exception:  # pragma: no cover
            pass

    def start_session(self) -> None:
        """托盘菜单：开始听课（沿用主窗口里填写的科目/标题）。"""
        starter = getattr(self._api, "start_session", None)
        if not callable(starter):
            return
        subject = title = ""
        window = self._window
        panel = getattr(window, "live", None)
        if panel is not None:
            subject = panel.edt_subject.text().strip()
            title = panel.edt_title.text().strip()
        try:
            starter(subject, title)
        except Exception as exc:
            self.notify("无法开始听课", str(exc))
            return
        self.notify("AI学生", "已开始听课")

    def toggle_pause(self) -> None:
        """托盘菜单：暂停/恢复。"""
        setter = getattr(self._api, "set_paused", None)
        if not callable(setter):
            return
        target = self._state != "paused"
        try:
            setter(target)
        except Exception as exc:
            self.notify("操作失败", str(exc))

    def stop_session(self) -> None:
        """托盘菜单：停止并收尾。"""
        stopper = getattr(self._api, "stop_session", None)
        if not callable(stopper):
            return
        try:
            stopper(True)
        except Exception as exc:
            self.notify("停止失败", str(exc))

    def request_quit(self) -> None:
        """托盘菜单：退出。shutdown 由 app.py 监听 quitRequested 完成。"""
        self.quitRequested.emit()
        self.hide()
        application = QApplication.instance()
        if application is not None:
            application.quit()
