"""UI 冒烟测试：offscreen 平台下实例化全部窗口，并把 ClassAgentAPI 的每个信号各 emit 一次。

设计要点：
- QT_QPA_PLATFORM 必须在 import PySide6 之前设置：优先 offscreen，插件缺失时自动退回 minimal。
- 模态框（QMessageBox）在测试里替换为空实现，否则 offscreen 下 exec() 会永久阻塞。
- 不依赖真实设备/网络：probe_devices、ping 之类全部由 MockController 顶替。
"""
from __future__ import annotations

import copy
import os
import re
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))


def _pick_platform() -> str:
    """优先 offscreen；本机缺少该插件时退回 minimal。"""
    forced = (os.environ.get("QT_QPA_PLATFORM") or "").strip()
    if forced:
        return forced
    try:
        import PySide6

        base = Path(PySide6.__file__).parent
        names: set[str] = set()
        for folder in (base / "plugins" / "platforms", base / "Qt" / "plugins" / "platforms"):
            if folder.is_dir():
                names |= {item.name.lower() for item in folder.iterdir()}
        if any("offscreen" in name for name in names):
            return "offscreen"
        if any("minimal" in name for name in names):
            return "minimal"
    except Exception:
        pass
    return "offscreen"


os.environ.setdefault("QT_QPA_PLATFORM", _pick_platform())

import pytest  # noqa: E402
from PySide6.QtWidgets import QApplication, QMessageBox  # noqa: E402

from ai_student.config import AppConfig  # noqa: E402
from ai_student.core.api import ClassAgentAPI  # noqa: E402
from ai_student.core.models import (  # noqa: E402
    Artifact,
    BoardNote,
    ClassTask,
    SessionBundle,
    SessionInfo,
    TranscriptSegment,
)
import ai_student.ui as ui  # noqa: E402
from ai_student.ui import main_window as mw_module  # noqa: E402
from ai_student.ui import settings_dialog as sd_module  # noqa: E402
from ai_student.ui import theme, widgets  # noqa: E402
from ai_student.ui.dialogs import FirstRunWizard, TaskInputDialog, TextPromptDialog  # noqa: E402
from ai_student.ui.main_window import MainWindow  # noqa: E402
from ai_student.ui.settings_dialog import SettingsDialog  # noqa: E402
from ai_student.ui.tray import TrayController  # noqa: E402


# ---------------------------------------------------------------- 测试替身


class _FakeMessageBox(QMessageBox):
    """把模态框替换成直接返回默认值，避免 offscreen 下阻塞。

    继承真 QMessageBox 以保留 StandardButton 枚举（代码里会做位或与比较）。
    """

    @staticmethod
    def warning(*_args, **_kwargs):
        return QMessageBox.StandardButton.Ok

    @staticmethod
    def information(*_args, **_kwargs):
        return QMessageBox.StandardButton.Ok

    @staticmethod
    def question(*_args, **_kwargs):
        return QMessageBox.StandardButton.No


class MockController(ClassAgentAPI):
    """实现 ClassAgentAPI 的全部方法：返回合理空值，并记录调用。"""

    def __init__(self, cfg: AppConfig | None = None) -> None:
        super().__init__()
        self.cfg = copy.deepcopy(cfg) if cfg is not None else AppConfig()
        self.calls: list[tuple[str, tuple]] = []
        self._state = "idle"
        self.preview_bytes: bytes | None = png_bytes()
        self.opened: list[str] = []
        self.exported: list[str] = []
        self.devices: dict = {"audio": [], "windows": [], "monitors": []}
        self.sessions: list[SessionInfo] = []
        self.bundle = SessionBundle(
            info=SessionInfo(id="s-mock", subject="数学", title="一元二次方程", directory=str(_ROOT)),
            transcript=[TranscriptSegment(index=1, t_start=12.0, t_end=30.0, text="今天我们讲一元二次方程")],
            board=[BoardNote(index=1, t=20.0, summary="板书：求根公式", ocr_text="x = (-b ± √(b²-4ac)) / 2a")],
            tasks=[
                ClassTask(
                    id="task_1",
                    title="完成练习册 P32",
                    detail="第 1-5 题",
                    confidence=0.8,
                    artifacts=["C:/tmp/a.md"],
                )
            ],
            notes_markdown="# 课堂笔记\n\n## 知识点\n- 求根公式",
        )

    def _record(self, name: str, *args) -> None:
        self.calls.append((name, args))

    # -- 配置 --------------------------------------------------------
    def get_config(self):
        self._record("get_config")
        return copy.deepcopy(self.cfg)

    def update_config(self, cfg) -> None:
        self._record("update_config", cfg)
        self.cfg = copy.deepcopy(cfg)
        self.configChanged.emit()

    # -- 会话 --------------------------------------------------------
    def start_session(self, subject: str = "", title: str = "") -> str:
        self._record("start_session", subject, title)
        info = SessionInfo(id="s-mock", subject=subject, title=title, state="recording", directory=str(_ROOT))
        self.sessions.append(info)
        self._state = "recording"
        self.sessionStarted.emit(info.to_dict())
        self.stateChanged.emit("recording", "开始采集")
        return info.id

    def stop_session(self, finalize: bool = True) -> None:
        self._record("stop_session", finalize)
        info = self.sessions[-1] if self.sessions else SessionInfo(id="s-mock", state="finished")
        info.state = "finished"
        self._state = "finished"
        self.sessionStopped.emit(info.to_dict())
        self.stateChanged.emit("finished", "听课结束")

    def set_paused(self, paused: bool) -> None:
        self._record("set_paused", paused)
        self._state = "paused" if paused else "recording"
        self.stateChanged.emit("paused" if paused else "recording", "已暂停" if paused else "继续听课")

    def is_recording(self) -> bool:
        return self._state == "recording"

    def audio_level_db(self) -> float:
        self._record("audio_level_db")
        return -23.5 if self._state == "recording" else -96.0

    def preview_png(self) -> bytes | None:
        self._record("preview_png")
        return self.preview_bytes

    def engine_state(self) -> str:
        return self._state

    def analyze_pending(self) -> None:
        self._record("analyze_pending")

    # -- 任务 --------------------------------------------------------
    def solve_task(self, task_id: str) -> None:
        self._record("solve_task", task_id)

    def solve_all_pending(self) -> None:
        self._record("solve_all_pending")

    def regenerate_task(self, task_id: str, feedback: str = "") -> None:
        self._record("regenerate_task", task_id, feedback)

    def delete_task(self, task_id: str) -> None:
        self._record("delete_task", task_id)

    def add_manual_task(self, title: str, detail: str = "") -> str:
        self._record("add_manual_task", title, detail)
        task = ClassTask(id="task_manual", title=title, detail=detail, status="discovered")
        self.taskDiscovered.emit(task.to_dict())
        return task.id

    def update_task(self, task_id: str, **fields) -> None:
        self._record("update_task", task_id, fields)

    # -- 数据 --------------------------------------------------------
    def current_bundle(self):
        self._record("current_bundle")
        return self.bundle

    def list_sessions(self, limit: int = 200) -> list[SessionInfo]:
        self._record("list_sessions", limit)
        return list(self.sessions)

    def load_session(self, session_id: str):
        self._record("load_session", session_id)
        return self.bundle if session_id else None

    def export_session(self, session_id: str) -> list[str]:
        self._record("export_session", session_id)
        return list(self.exported)

    def probe_devices(self) -> dict:
        self._record("probe_devices")
        return dict(self.devices)

    def open_path(self, path: str) -> None:
        self._record("open_path", path)
        self.opened.append(str(path))

    def shutdown(self) -> None:
        self._record("shutdown")


def png_bytes(width: int = 16, height: int = 9) -> bytes:
    """生成一小段真实 PNG 字节（模拟引擎的 preview_png）。"""
    from PySide6.QtCore import QBuffer, QByteArray
    from PySide6.QtGui import QColor, QPixmap

    pixmap = QPixmap(width, height)
    pixmap.fill(QColor("#123456"))
    buffer = QBuffer()
    buffer.open(QBuffer.OpenModeFlag.WriteOnly)
    pixmap.save(buffer, "PNG")
    return bytes(buffer.data())


def configured_cfg() -> AppConfig:
    """一份“配置完整”的配置，用于验证提示条消失。"""
    cfg = AppConfig()
    cfg.model.base_url = "https://example.invalid/v1"
    cfg.model.api_key = "sk-test"
    cfg.model.chat_model = "test-model"
    cfg.person.name = "测试同学"
    cfg.output.output_dir = str(_ROOT / "artifacts" / "test-output")
    return cfg


def emit_all_signals(api: ClassAgentAPI) -> None:
    """把契约里的每一个信号各 emit 一次。"""
    info = SessionInfo(id="s-emit", subject="物理", title="牛顿第二定律", transcript_count=1, board_count=1)
    api.sessionStarted.emit(info.to_dict())
    api.sessionStopped.emit(info.to_dict())
    api.stateChanged.emit("recording", "开始采集")
    api.stateChanged.emit("paused", "已暂停")
    api.stageChanged.emit("transcribe", "正在转写")
    api.transcriptSegment.emit(TranscriptSegment(index=1, t_start=3.0, t_end=20.0, text="大家好").to_dict())
    api.boardNote.emit(BoardNote(index=1, t=8.0, image_path="screen/keyframes/kf_0001.jpg", summary="板书").to_dict())
    api.notesUpdated.emit("# 课堂笔记\n\n## 知识点\n- 牛顿第二定律")
    api.taskDiscovered.emit(ClassTask(id="task_x", title="课后习题 1-3", confidence=0.6).to_dict())
    api.taskUpdated.emit(
        ClassTask(id="task_x", title="课后习题 1-3", status="solved", artifacts=["C:/tmp/a.md"]).to_dict()
    )
    api.artifactWritten.emit(Artifact(task_id="task_x", path="C:/tmp/a.md", fmt="md", title="课后习题", size_bytes=2048).to_dict())
    api.logMessage.emit("INFO", "引擎已启动")
    api.logMessage.emit("WARNING", "网络抖动，正在重试")
    api.errorRaised.emit("模型调用失败", "连接超时")
    api.progress.emit("solve", 42)
    api.configChanged.emit()
    api.captureDevicesChanged.emit({"audio": [{"name": "麦克风"}], "windows": [{"title": "极域电子教室"}], "monitors": []})


# ---------------------------------------------------------------- fixtures


@pytest.fixture(scope="session")
def qt_app():
    """全局唯一 QApplication（主题已应用）。"""
    application = QApplication.instance() or QApplication([])
    theme.apply_theme(application)
    yield application


@pytest.fixture(autouse=True)
def _silence_modals(monkeypatch):
    """禁止任何模态框/输入框阻塞测试（offscreen 下 exec() 会永久挂起）。"""
    monkeypatch.setattr(mw_module, "QMessageBox", _FakeMessageBox)
    monkeypatch.setattr(sd_module, "QMessageBox", _FakeMessageBox)
    monkeypatch.setattr(mw_module.TextPromptDialog, "ask", staticmethod(lambda *a, **k: "反馈内容"))
    monkeypatch.setattr(mw_module.TaskInputDialog, "get_task", staticmethod(lambda *a, **k: ("手动任务", "详情")))
    yield


@pytest.fixture
def api():
    return MockController()


@pytest.fixture
def window(qt_app, api):
    win = MainWindow(api)
    yield win
    win.close()
    win.deleteLater()


def process_events(qt_app, seconds: float = 2.0, until=None) -> bool:
    """跑事件循环直到条件满足或超时。"""
    deadline = time.time() + seconds
    while time.time() < deadline:
        qt_app.processEvents()
        if until is not None and until():
            return True
        time.sleep(0.01)
    qt_app.processEvents()
    return bool(until()) if until is not None else True


# ---------------------------------------------------------------- 契约


def test_required_exports():
    for name in ("MainWindow", "SettingsDialog", "FirstRunWizard", "TrayController", "apply_theme"):
        assert hasattr(ui, name), name
    assert callable(ui.apply_theme)


def test_ui_only_imports_contract_layers():
    """UI 不得 import core.controller / capture / ai 等引擎实现层。"""
    import ai_student.ui as ui_pkg

    forbidden = ("ai_student.capture", "ai_student.ai", "core.controller", "core.capture", "core.engine")
    pattern = re.compile(r"^\s*(?:from|import)\s+([^\n#]+)", re.M)
    ui_dir = Path(ui_pkg.__file__).parent
    files = sorted(ui_dir.glob("*.py"))
    assert files, "没有找到 ui 包源码"
    for path in files:
        text = path.read_text(encoding="utf-8")
        for module in pattern.findall(text):
            for bad in forbidden:
                assert bad not in module, f"{path.name} 不应导入 {bad}"


# ---------------------------------------------------------------- 窗口


def test_main_window_layout(qt_app, window):
    assert window.nav.count() == 6
    assert window.stack.count() == 6
    assert [window.nav.item(i).text() for i in range(6)] == ["听课", "任务", "作业产出", "历史", "日志", "设置"]
    # 任务表：6 列且带操作按钮列
    assert window.tasks.table.columnCount() == 6
    assert window.tasks.table.horizontalHeaderItem(5).text() == "操作"
    # 历史页双列
    assert window.history.list_sessions is not None
    assert window.history.tabs.count() == 4
    assert window.settings.form.tabs.count() == 5


def test_config_banner_visible_when_unconfigured(window):
    assert window.config_problems()
    assert window.banner_visible() is True
    assert "配置尚未完成" in window.lbl_banner.text()


def test_config_banner_hidden_when_configured(qt_app):
    win = MainWindow(MockController(configured_cfg()))
    try:
        assert win.config_problems() == []
        assert win.banner_visible() is False
    finally:
        win.close()
        win.deleteLater()


def test_all_signals_do_not_crash(qt_app, window, api):
    emit_all_signals(api)
    qt_app.processEvents()
    # 转写进入列表、笔记与任务落到界面
    assert window.live.transcript_count() == 1
    assert "牛顿第二定律" in window.live.view_notes.toPlainText()
    assert "课后习题 1-3" in window.tasks.task_ids() or window.tasks.table.rowCount() >= 1
    assert window.artifacts.table.rowCount() == 1
    assert len(window.logs.lines()) >= 4
    assert window.statusBar() is not None


def test_signals_tolerate_odd_payloads(qt_app, window, api):
    for payload in ({}, None, object()):
        api.transcriptSegment.emit(payload)
        api.boardNote.emit(payload)
        api.taskDiscovered.emit(payload)
        api.taskUpdated.emit(payload)
        api.artifactWritten.emit(payload)
        api.sessionStarted.emit(payload)
        api.sessionStopped.emit(payload)
        api.captureDevicesChanged.emit(payload)
    api.stateChanged.emit(None, None)
    api.notesUpdated.emit(None)
    api.logMessage.emit(None, None)
    api.progress.emit(None, None)
    qt_app.processEvents()
    assert window.isEnabled()


def test_window_show_and_grab(qt_app, window):
    window.show()
    qt_app.processEvents()
    pixmap = window.grab()
    assert not pixmap.isNull()
    assert pixmap.width() > 400 and pixmap.height() > 300
    window.hide()


# ---------------------------------------------------------------- 设置 / 向导


def test_settings_dialog_roundtrip(qt_app, monkeypatch):
    monkeypatch.setattr(sd_module, "check_endpoint", lambda *_a, **_k: (True, "本地假探测"))
    api = MockController()
    original = AppConfig()
    dialog = SettingsDialog(original, api=api)
    try:
        assert dialog.form.tabs.count() == 5
        assert [dialog.form.tabs.tabText(i) for i in range(5)] == ["模型 API", "个人信息", "采集", "作业输出", "高级"]
        assert isinstance(dialog.result_config(), AppConfig)
        dialog.form.edt_name.setText("张三")
        dialog.form.edt_output_dir.setText(str(_ROOT / "artifacts" / "out"))
        dialog.form.edt_base_url.setText("https://example.invalid/v1")
        dialog.form.edt_chat.setText("test-chat")
        result = dialog.result_config()
        assert result.person.name == "张三"
        assert result.output.output_dir.endswith("out")
        # 不污染传入的 cfg
        assert original.person.name == ""
        target = AppConfig()
        returned = dialog.apply_to(target)
        assert returned is target
        assert target.person.name == "张三"
        # 测试连接按钮走线程，不阻塞
        dialog.form.btn_test.click()
        process_events(qt_app, 1.0, lambda: dialog.form.btn_test.isEnabled())
        assert isinstance(dialog.form.lbl_test.text(), str)
    finally:
        dialog.close()
        dialog.deleteLater()


def test_settings_dialog_apply_saves_via_api(qt_app):
    api = MockController()
    dialog = SettingsDialog(configured_cfg(), api=api)
    try:
        dialog.form.edt_name.setText("李四")
        dialog.form.edt_vision.setText("vision-x")
        dialog._on_apply()
        names = [name for name, _args in api.calls]
        assert "update_config" in names
        assert api.cfg.person.name == "李四"
        assert dialog.applied_config() is not None
    finally:
        dialog.close()
        dialog.deleteLater()


def test_first_run_wizard_pages(qt_app):
    wizard = FirstRunWizard(AppConfig())
    try:
        assert len(wizard.pageIds()) == 3
        cfg = wizard.result_config()
        assert isinstance(cfg, AppConfig)
        wizard.page_person.edt_name.setText("王五")
        wizard.page_output.edt_dir.setText(str(_ROOT / "artifacts" / "wizard-out"))
        wizard.page_model.edt_base_url.setText("https://example.invalid/v1")
        wizard.page_model.edt_chat.setText("chat-x")
        result = wizard.result_config()
        assert result.person.name == "王五"
        assert result.output.output_dir.endswith("wizard-out")
        assert result.model.base_url == "https://example.invalid/v1"
        # 切到预设会填充地址与模型
        wizard.page_model.cmb_preset.setCurrentIndex(0)
        assert wizard.page_model.edt_base_url.text()
    finally:
        wizard.close()
        wizard.deleteLater()


def test_wizard_accepts_and_returns_config(qt_app):
    wizard = FirstRunWizard(configured_cfg())
    try:
        wizard.accept()
        cfg = wizard.result_config()
        assert cfg.person.name == "测试同学"
    finally:
        wizard.close()
        wizard.deleteLater()


def test_input_dialogs(qt_app):
    prompt = TextPromptDialog("重做任务", "请填写反馈", "默认值")
    try:
        assert prompt.text() == "默认值"
    finally:
        prompt.close()
        prompt.deleteLater()
    task_dialog = TaskInputDialog()
    try:
        task_dialog.edt_title.setText("完成实验报告")
        task_dialog.edt_detail.setPlainText("300 字")
        assert task_dialog.values() == ("完成实验报告", "300 字")
    finally:
        task_dialog.close()
        task_dialog.deleteLater()


# ---------------------------------------------------------------- 面板行为


def test_tasks_panel_rows_and_actions(qt_app, window):
    panel = window.tasks
    panel.upsert_task(ClassTask(id="t1", title="任务一", task_type="作业", status="solved", confidence=0.9,
                                artifacts=["C:/tmp/one.md"]).to_dict())
    panel.upsert_task(ClassTask(id="t2", title="任务二").to_dict())
    panel.upsert_task(ClassTask(id="t1", title="任务一（改名）", status="working").to_dict())
    assert panel.table.rowCount() == 2
    assert panel.table.cellWidget(0, 5) is not None
    assert panel.table.item(0, 0).text() == "任务一（改名）"
    panel.remove_task("t1")
    assert panel.table.rowCount() == 1
    assert panel.task_ids() == ["t2"]
    panel.clear_tasks()
    assert panel.table.rowCount() == 0


def test_artifact_and_history_panels(qt_app, window):
    window.artifacts.add_artifact(Artifact(path="C:/tmp/a.docx", fmt="docx", title="作业", size_bytes=4096).to_dict())
    window.artifacts.add_artifact(Artifact(path="C:/tmp/a.docx", fmt="docx", title="作业", size_bytes=4096).to_dict())
    assert window.artifacts.table.rowCount() == 1
    assert window.artifacts.paths() == ["C:/tmp/a.docx"]

    window.history.set_sessions([window._api.bundle.info])
    assert window.history.list_sessions.count() == 1
    window.history.select_session("s-mock")
    assert window.history.selected_session_id() == "s-mock"
    window.history.show_bundle(window._api.bundle)
    assert "一元二次方程" in window.history.view_transcript.toPlainText()
    assert "求根公式" in window.history.view_board.toPlainText()
    assert "完成练习册 P32" in window.history.view_tasks.toPlainText()
    assert "课堂笔记" in window.history.view_notes.toPlainText()
    window.history.edt_search.setText("不存在")
    assert window.history.list_sessions.item(0).isHidden()


def test_log_panel_filter_and_export(qt_app, window, tmp_path):
    window.logs.append_log("DEBUG", "细节")
    window.logs.append_log("ERROR", "出错了")
    assert len(window.logs.lines()) == 2
    window.logs.cmb_level.setCurrentIndex(3)  # 仅错误
    qt_app.processEvents()
    assert window.logs.view.min_level() == "ERROR"
    target = tmp_path / "ui-log.txt"
    path = window.logs.view.export_to(str(target))
    assert Path(path).is_file()
    assert "出错了" in Path(path).read_text(encoding="utf-8")


def test_async_data_refresh_paths(qt_app, window, api):
    api.sessions.append(api.bundle.info)
    window._refresh_sessions()
    assert process_events(qt_app, 3.0, lambda: window.history.list_sessions.count() > 0)
    window._refresh_current_bundle()
    assert process_events(qt_app, 3.0, lambda: window.tasks.table.rowCount() > 0)
    window._load_session("s-mock")
    assert process_events(qt_app, 5.0, lambda: "一元二次方程" in window.history.view_transcript.toPlainText())
    window._export_session("s-mock")
    process_events(qt_app, 1.0, lambda: "没有需要导出" in window.lbl_status.text())
    window._open_output_dir()
    process_events(qt_app, 1.0, lambda: bool(api.opened))


def test_start_and_stop_session(qt_app, window, api):
    window._start_session("数学", "一元二次方程")
    assert process_events(qt_app, 5.0, lambda: ("start_session", ("数学", "一元二次方程")) in api.calls)
    assert process_events(qt_app, 5.0, lambda: window.live.pill.state() == "recording")
    window._toggle_pause(True)
    qt_app.processEvents()
    assert window.live.pill.state() == "paused"
    window._stop_session()
    assert process_events(qt_app, 5.0, lambda: any(name == "stop_session" for name, _args in api.calls))
    assert process_events(qt_app, 5.0, lambda: window.live.pill.state() == "finished")


def test_live_level_and_preview_polling(qt_app, window, api):
    window._start_session("数学", "一元二次方程")
    assert process_events(qt_app, 5.0, lambda: window.live.pill.state() == "recording")
    window.live.poll_level()
    assert abs(window.live.meter.value_db() + 23.5) < 0.01
    assert window.live.meter.is_active()

    window.live.poll_preview()
    assert process_events(qt_app, 3.0, lambda: not window.live.preview_busy())
    pixmap = window.live.lbl_preview.pixmap()
    assert pixmap is not None and not pixmap.isNull()

    # 引擎暂时给不出画面时退回最近关键帧文件，不崩
    api.preview_bytes = None
    window.live.poll_preview()
    assert process_events(qt_app, 3.0, lambda: not window.live.preview_busy())

    # 停止后电平回到“未采集”
    window._stop_session()
    qt_app.processEvents()
    window.live.poll_level()
    assert not window.live.meter.is_active()


def test_initial_state_comes_from_engine(qt_app):
    api = MockController()
    api._state = "recording"
    win = MainWindow(api)
    try:
        assert win.live.pill.state() == "recording"
        assert win.live._level_timer.isActive()
    finally:
        win.close()
        win.deleteLater()


def test_task_actions_delegate_to_api(qt_app, window, api):
    window.tasks.upsert_task(ClassTask(id="t9", title="任务九").to_dict())
    window._solve_task("t9")
    window._solve_all()
    window._delete_task("t9")      # question 被替换为 No -> 不删除（同步返回，不进线程池）
    window._regenerate_task("t9")  # 反馈输入被替换为固定文本
    window._add_manual_task()      # 手动添加任务对话框同样被替换
    assert process_events(
        qt_app,
        5.0,
        lambda: {"solve_task", "solve_all_pending", "regenerate_task", "add_manual_task"}
        <= {name for name, _args in api.calls},
    )
    calls = dict((name, args) for name, args in api.calls)
    assert "solve_task" in calls
    assert "solve_all_pending" in calls
    assert "delete_task" not in calls
    assert calls.get("regenerate_task") == ("t9", "反馈内容")
    assert "add_manual_task" in calls


# ---------------------------------------------------------------- 控件与异步


def test_widgets_behaviour(qt_app, tmp_path):
    meter = widgets.LevelMeter()
    meter.set_level(-20.5)
    assert abs(meter.value_db() + 20.5) < 0.01
    assert meter.is_active()
    meter.set_idle()
    assert not meter.is_active()

    pill = widgets.StatusPill()
    for state, text in (("recording", "听课中"), ("paused", "已暂停"), ("error", "出错")):
        pill.set_status(state)
        assert pill.text() == text
    pill.set_status("recording", "自定义说明")
    assert pill.text() == "自定义说明"
    assert not pill.grab().isNull()

    view = widgets.LogView()
    view.append_log("INFO", "第一条")
    view.append_log("ERROR", "第二条")
    assert len(view.text_lines()) == 2
    target = tmp_path / "log.txt"
    view.export_to(str(target))
    assert target.read_text(encoding="utf-8").count("\n") == 2

    listing = widgets.TimecodeList()
    listing.add_segment(TranscriptSegment(index=1, t_start=61.0, text="这是一段转写").to_dict())
    assert listing.count() == 1
    assert "00:01:01" in listing.item(0).text()
    assert listing.apply_filter("转写") == 1
    assert listing.apply_filter("找不到") == 0
    assert listing.apply_filter("") == 1
    assert len(listing.segments()) == 1

    assert widgets.as_dict(None) == {}
    assert widgets.as_dict({"a": 1}) == {"a": 1}
    assert widgets.hms(3661) == "01:01:01"


def test_run_async_delivers_result_and_error(qt_app):
    results: dict = {}
    widgets.run_async(lambda: 6 * 7, lambda value: results.setdefault("ok", value))
    assert process_events(qt_app, 3.0, lambda: "ok" in results)
    assert results["ok"] == 42

    errors: dict = {}
    widgets.run_async(lambda: 1 / 0, None, lambda message: errors.setdefault("err", message))
    assert process_events(qt_app, 3.0, lambda: "err" in errors)
    assert "ZeroDivisionError" in errors["err"]


def test_settings_form_device_refresh_is_async(qt_app):
    api = MockController()
    api.devices = {
        "audio": [{"name": "麦克风 (Realtek)", "is_loopback": False}, {"name": "扬声器回环", "is_loopback": True}],
        "windows": [{"title": "极域电子教室 - 教师演示"}],
        "monitors": [{"index": 1, "width": 1920, "height": 1080}],
    }
    dialog = SettingsDialog(configured_cfg(), api=api)
    try:
        dialog.form.refresh_devices()
        assert process_events(qt_app, 3.0, lambda: dialog.form.cmb_mic_device.count() > 1)
        assert dialog.form.cmb_mic_device.count() == 3
        assert dialog.form.cmb_window.count() == 2
        assert "音频设备" in dialog.form.lbl_devices.text()
    finally:
        dialog.close()
        dialog.deleteLater()


def test_theme_and_fonts(qt_app):
    assert theme.apply_theme(qt_app) is qt_app
    assert len(qt_app.styleSheet()) > 500
    assert theme.preferred_font_family() in theme.FONT_FAMILIES
    assert not theme.app_icon(64).isNull()
    assert theme.dark_palette().color(theme.QPalette.ColorRole.Window).isValid()
    assert "Microsoft YaHei UI" in theme.build_stylesheet()


# ---------------------------------------------------------------- 托盘


def test_tray_controller(qt_app, window, api):
    tray = TrayController(api, window)
    try:
        assert tray.menu.actions()
        assert tray.is_available() in (True, False)
        tray.show()
        # 引擎状态变化会自动同步到托盘
        api.stateChanged.emit("recording", "开始采集")
        qt_app.processEvents()
        assert tray.state() == "recording"
        tray.update_state("recording")
        assert tray.state() == "recording"
        assert tray.act_stop.isEnabled()
        tray.update_state("paused")
        assert tray.act_pause.text() == "恢复听课"
        tray.show_window()
        tray.start_session()
        tray.toggle_pause()
        tray.stop_session()
        tray.notify("测试", "通知")
        qt_app.processEvents()
        names = [name for name, _args in api.calls]
        assert "start_session" in names
        assert "set_paused" in names
        assert "stop_session" in names
        fired: list[bool] = []
        tray.quitRequested.connect(lambda: fired.append(True))
        tray.request_quit()
        assert fired == [True]
    finally:
        tray.hide()
        tray.deleteLater()


def test_platform_plugin_is_offscreen_or_minimal():
    assert os.environ["QT_QPA_PLATFORM"] in ("offscreen", "minimal"), os.environ["QT_QPA_PLATFORM"]
    assert QApplication.instance() is not None


def test_message_box_stub_is_active():
    """确认模态框被替换（否则说明测试环境里会有阻塞风险）。"""
    assert mw_module.QMessageBox is _FakeMessageBox
    assert QMessageBox is not _FakeMessageBox
