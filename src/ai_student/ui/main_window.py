"""主窗口：左侧导航 + 右侧堆叠页面，并把 ClassAgentAPI 的全部信号接到界面上。

关键约定：
- 界面只依赖 ClassAgentAPI；closeEvent 只发 closing 信号，shutdown 由 app.py 负责。
- 所有可能耗时的引擎调用（list_sessions / load_session / export_session / current_bundle）
  都通过 run_async 在工作线程执行，UI 线程只负责渲染。
"""
from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QSize, Signal
from PySide6.QtGui import QAction
from PySide6.QtWidgets import (
    QDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QStackedWidget,
    QStatusBar,
    QVBoxLayout,
    QWidget,
)

from ..config import AppConfig
from .dialogs import FirstRunWizard, TaskInputDialog, TextPromptDialog
from .panels import ArtifactsPanel, HistoryPanel, LivePanel, LogPanel, SettingsPanel, TasksPanel
from .settings_dialog import SettingsDialog
from .theme import app_icon
from .widgets import as_dict, get_str, make_button, run_async

NAV_ITEMS = ["听课", "任务", "作业产出", "历史", "日志", "设置"]


class MainWindow(QMainWindow):
    """AI学生主窗口。"""

    closing = Signal()

    def __init__(self, api, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._api = api
        self._cfg: AppConfig = self._safe_get_config()
        self._session_dir = ""
        self._session_id = ""

        self.setWindowTitle("AI学生 · 自动听课与作业助手")
        self.setWindowIcon(app_icon())
        self.resize(1360, 860)
        self.setMinimumSize(QSize(1120, 720))

        central = QWidget()
        root = QVBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        self.banner = self._build_banner()
        root.addWidget(self.banner)

        body = QHBoxLayout()
        body.setContentsMargins(0, 0, 0, 0)
        body.setSpacing(0)
        body.addWidget(self._build_nav())
        self.stack = QStackedWidget()
        body.addWidget(self.stack, 1)
        root.addLayout(body, 1)
        self.setCentralWidget(central)

        self._build_panels()
        self._build_statusbar()
        self._build_menus()
        self._connect_api()

        self._refresh_config_state()
        self.live.set_state(self._engine_state())
        self.nav.setCurrentRow(0)

    # ------------------------------------------------------------ 构建
    def _build_banner(self) -> QFrame:
        banner = QFrame()
        banner.setObjectName("banner")
        banner.setVisible(False)
        layout = QHBoxLayout(banner)
        layout.setContentsMargins(12, 8, 12, 8)
        layout.setSpacing(10)
        self.lbl_banner = QLabel("")
        self.lbl_banner.setObjectName("bannerText")
        self.lbl_banner.setWordWrap(True)
        layout.addWidget(QLabel("⚠"))
        layout.addWidget(self.lbl_banner, 1)
        self.btn_banner_settings = make_button("打开设置", "primary")
        self.btn_banner_wizard = make_button("运行首次向导")
        self.btn_banner_close = make_button("稍后", "ghost")
        layout.addWidget(self.btn_banner_settings)
        layout.addWidget(self.btn_banner_wizard)
        layout.addWidget(self.btn_banner_close)
        self.btn_banner_settings.clicked.connect(self.open_settings)
        self.btn_banner_wizard.clicked.connect(self.run_first_run_wizard)
        self.btn_banner_close.clicked.connect(lambda: banner.setVisible(False))
        return banner

    def _build_nav(self) -> QFrame:
        rail = QFrame()
        rail.setObjectName("navRail")
        rail.setFixedWidth(170)
        layout = QVBoxLayout(rail)
        layout.setContentsMargins(0, 8, 0, 8)
        layout.setSpacing(6)
        brand = QLabel("AI 学生")
        brand.setObjectName("pageTitle")
        brand.setContentsMargins(16, 6, 10, 6)
        layout.addWidget(brand)
        self.nav = QListWidget()
        self.nav.setObjectName("navList")
        for name in NAV_ITEMS:
            item = QListWidgetItem(name)
            item.setSizeHint(QSize(140, 40))
            self.nav.addItem(item)
        layout.addWidget(self.nav, 1)
        self.lbl_nav_hint = QLabel("待机")
        self.lbl_nav_hint.setObjectName("hint")
        self.lbl_nav_hint.setContentsMargins(16, 0, 10, 6)
        self.lbl_nav_hint.setWordWrap(True)
        layout.addWidget(self.lbl_nav_hint)
        self.nav.currentRowChanged.connect(self._on_nav_changed)
        return rail

    def _build_panels(self) -> None:
        self.live = LivePanel(self._api)
        self.tasks = TasksPanel(self._api)
        self.artifacts = ArtifactsPanel(self._api)
        self.history = HistoryPanel(self._api)
        self.logs = LogPanel(self._api)
        self.settings = SettingsPanel(self._cfg, api=self._api)
        self.pages = [self.live, self.tasks, self.artifacts, self.history, self.logs, self.settings]
        for page in self.pages:
            wrapper = QWidget()
            layout = QVBoxLayout(wrapper)
            layout.setContentsMargins(14, 12, 14, 12)
            layout.addWidget(page)
            self.stack.addWidget(wrapper)

        self.live.startRequested.connect(self._start_session)
        self.live.pauseToggled.connect(self._toggle_pause)
        self.live.stopRequested.connect(self._stop_session)
        self.live.analyzeRequested.connect(self._analyze_pending)

        self.tasks.solveRequested.connect(self._solve_task)
        self.tasks.regenerateRequested.connect(self._regenerate_task)
        self.tasks.deleteRequested.connect(self._delete_task)
        self.tasks.openPathRequested.connect(self.open_path)
        self.tasks.addRequested.connect(self._add_manual_task)
        self.tasks.solveAllRequested.connect(self._solve_all)
        self.tasks.refreshRequested.connect(self._refresh_current_bundle)

        self.artifacts.openPathRequested.connect(self.open_path)
        self.artifacts.openFolderRequested.connect(self._open_output_dir)
        self.artifacts.exportRequested.connect(lambda: self._export_session(self._session_id))
        self.artifacts.refreshRequested.connect(self._refresh_current_bundle)

        self.history.refreshRequested.connect(self._refresh_sessions)
        self.history.loadRequested.connect(self._load_session)
        self.history.exportRequested.connect(self._export_session)
        self.history.openFolderRequested.connect(self._open_session_dir)

        self.logs.exportRequested.connect(self._export_logs)

        self.settings.saveRequested.connect(self._save_config)
        self.settings.reloadRequested.connect(self._reload_settings)
        self.settings.fullDialogRequested.connect(self.open_settings)

    def _build_statusbar(self) -> None:
        bar = QStatusBar()
        self.setStatusBar(bar)
        self.lbl_status = QLabel("就绪")
        self.progress = QProgressBar()
        self.progress.setMaximumWidth(240)
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.progress.setVisible(False)
        bar.addWidget(self.lbl_status, 1)
        bar.addPermanentWidget(self.progress)

    def _build_menus(self) -> None:
        menu_session = self.menuBar().addMenu("会话")
        self.act_start = QAction("开始听课", self)
        self.act_pause = QAction("暂停/恢复", self)
        self.act_stop = QAction("停止并收尾", self)
        self.act_analyze = QAction("立即分析", self)
        for action in (self.act_start, self.act_pause, self.act_stop, self.act_analyze):
            menu_session.addAction(action)
        self.act_start.triggered.connect(lambda: self._start_session("", ""))
        self.act_pause.triggered.connect(lambda: self._toggle_pause(self.live.pill.state() != "paused"))
        self.act_stop.triggered.connect(self._stop_session)
        self.act_analyze.triggered.connect(self._analyze_pending)
        menu_session.addSeparator()
        self.act_settings = QAction("设置…", self)
        self.act_settings.triggered.connect(self.open_settings)
        menu_session.addAction(self.act_settings)
        self.act_quit = QAction("退出", self)
        self.act_quit.triggered.connect(self.close)
        menu_session.addAction(self.act_quit)

        menu_help = self.menuBar().addMenu("帮助")
        self.act_wizard = QAction("首次运行向导", self)
        self.act_wizard.triggered.connect(self.run_first_run_wizard)
        menu_help.addAction(self.act_wizard)
        self.act_about = QAction("关于", self)
        self.act_about.triggered.connect(self._show_about)
        menu_help.addAction(self.act_about)

    # ------------------------------------------------------------ API 信号
    def _connect_api(self) -> None:
        api = self._api
        if api is None:
            return
        pairs = [
            ("sessionStarted", self._on_session_started),
            ("sessionStopped", self._on_session_stopped),
            ("stateChanged", self._on_state_changed),
            ("stageChanged", self._on_stage_changed),
            ("transcriptSegment", self._on_transcript),
            ("boardNote", self._on_board_note),
            ("notesUpdated", self._on_notes),
            ("taskDiscovered", self._on_task_discovered),
            ("taskUpdated", self._on_task_updated),
            ("artifactWritten", self._on_artifact_written),
            ("logMessage", self._on_log_message),
            ("errorRaised", self._on_error),
            ("progress", self._on_progress),
            ("configChanged", self._on_config_changed),
            ("captureDevicesChanged", self._on_devices_changed),
        ]
        for name, slot in pairs:
            signal = getattr(api, name, None)
            if signal is None:
                continue
            try:
                signal.connect(slot)
            except Exception:  # pragma: no cover - 引擎信号缺失时跳过
                pass

    # -- 槽 ----------------------------------------------------------
    def _on_session_started(self, payload: object) -> None:
        data = as_dict(payload)
        self._session_id = get_str(data, "id")
        self._session_dir = get_str(data, "directory")
        self.live.set_session_info(data)
        self.live.reset_session()
        self.live.list_transcript.clear_all()
        self.tasks.clear_tasks()
        self.artifacts.clear_all()
        self.artifacts.set_output_dir(get_str(data, "output_dir"))
        self._set_status(f"听课中：{get_str(data, 'title') or get_str(data, 'subject') or self._session_id}")
        self._show_page(0)

    def _on_session_stopped(self, payload: object) -> None:
        data = as_dict(payload)
        self.live.set_state(get_str(data, "state", "finished"))
        self._set_status("本次听课已结束，可到“作业产出”查看文件。")
        self._refresh_current_bundle()
        self._refresh_sessions()

    def _on_state_changed(self, state: str, message: str) -> None:
        self.live.set_state(str(state or "idle"), str(message or ""))
        self.lbl_nav_hint.setText(str(message or "") or "待机")
        if message:
            self._set_status(str(message))

    def _on_stage_changed(self, stage: str, message: str) -> None:
        self.live.set_stage(str(stage or ""), str(message or ""))

    def _on_transcript(self, payload: object) -> None:
        self.live.add_transcript(payload)

    def _on_board_note(self, payload: object) -> None:
        data = as_dict(payload)
        image = get_str(data, "image_path")
        if not image:
            return
        path = Path(image)
        if not path.is_absolute() and self._session_dir:
            path = Path(self._session_dir) / image
        self.live.show_board_image(str(path))

    def _on_notes(self, markdown: str) -> None:
        self.live.set_notes(str(markdown or ""))

    def _on_task_discovered(self, payload: object) -> None:
        self.tasks.upsert_task(payload)
        data = as_dict(payload)
        title = get_str(data, "title")
        if title:
            self.logs.append_log("INFO", f"发现任务：{title}")
            self._set_status(f"发现新任务：{title}")

    def _on_task_updated(self, payload: object) -> None:
        self.tasks.upsert_task(payload)

    def _on_artifact_written(self, payload: object) -> None:
        self.artifacts.add_artifact(payload)
        data = as_dict(payload)
        path = get_str(data, "path")
        if path:
            self.logs.append_log("INFO", f"已产出作业：{path}")
            self._set_status(f"已生成作业文件：{Path(path).name}")

    def _on_log_message(self, level: str, message: str) -> None:
        self.logs.append_log(str(level or "INFO"), str(message or ""))

    def _on_error(self, title: str, detail: str) -> None:
        self.logs.append_log("ERROR", f"{title}：{detail}")
        self._set_status(f"出错了：{title}")
        QMessageBox.warning(self, str(title or "出错了"), str(detail or ""))

    def _on_progress(self, stage: str, value: int) -> None:
        try:
            percent = int(value)
        except (TypeError, ValueError):
            percent = 0
        percent = max(0, min(100, percent))
        self.progress.setVisible(0 < percent < 100)
        self.progress.setValue(percent)
        if stage:
            self.progress.setFormat(f"{stage} %p%")

    def _on_config_changed(self) -> None:
        self._cfg = self._safe_get_config()
        self._refresh_config_state()
        self.settings.load_config(self._cfg)

    def _on_devices_changed(self, payload: object) -> None:
        data = as_dict(payload)
        audio = data.get("audio") or []
        windows = data.get("windows") or []
        self.logs.append_log("INFO", f"采集环境更新：音频设备 {len(audio)} 个 / 窗口 {len(windows)} 个")

    # ------------------------------------------------------------ 界面动作
    def _on_nav_changed(self, row: int) -> None:
        if row < 0:
            return
        self.stack.setCurrentIndex(row)
        if row == 3:
            self._refresh_sessions()
        elif row == 5:
            self._reload_settings()

    def _show_page(self, row: int) -> None:
        if 0 <= row < self.nav.count():
            self.nav.setCurrentRow(row)

    def _set_status(self, text: str) -> None:
        self.lbl_status.setText(text)

    def _engine_state(self) -> str:
        """启动时读取引擎真实状态（例如 --smoke 已经开录）。"""
        getter = getattr(self._api, "engine_state", None)
        if callable(getter):
            try:
                state = str(getter() or "idle")
                if state in ("idle", "recording", "paused", "finalizing", "finished", "error"):
                    return state
            except Exception:
                pass
        return "idle"

    def _safe_get_config(self) -> AppConfig:
        getter = getattr(self._api, "get_config", None)
        if callable(getter):
            try:
                cfg = getter()
                if isinstance(cfg, AppConfig):
                    return cfg
            except Exception:
                pass
        return AppConfig()

    def _refresh_config_state(self) -> None:
        """配置不完整时在顶部显示醒目提示条。"""
        try:
            problems = list(self._cfg.validate())
        except Exception:
            problems = []
        if problems:
            preview = "；".join(problems[:3])
            more = f"（共 {len(problems)} 项）" if len(problems) > 3 else ""
            self.lbl_banner.setText(f"配置尚未完成，暂时无法开始听课：{preview}{more}")
            self.banner.setVisible(True)
        else:
            self.banner.setVisible(False)
            self.lbl_banner.setText("")

    def banner_visible(self) -> bool:
        """配置提示条是否可见。

        用 isHidden() 判断：窗口尚未 show()/在 offscreen 下时 isVisible() 恒为 False，
        isHidden() 才反映“是否被显式隐藏”。
        """
        return not self.banner.isHidden()

    def config_problems(self) -> list[str]:
        try:
            return list(self._cfg.validate())
        except Exception:
            return []

    def _require_api(self, name: str):
        method = getattr(self._api, name, None)
        if not callable(method):
            QMessageBox.information(self, "暂不可用", "当前引擎未实现该功能。")
            return None
        return method

    def _call_api(self, name: str, on_result=None, on_error=None, *args) -> bool:
        """在工作线程里调用引擎方法，回调回到 UI 线程。

        引擎的 start/stop 会开关设备、join 采集线程，直接在 UI 线程调用会卡界面；
        只读写内存标志的调用（set_paused / audio_level_db / engine_state / get_config）
        仍然同步执行，避免按钮反馈变迟钝。
        """
        method = self._require_api(name)
        if method is None:
            return False
        run_async(method, on_result, on_error or self._on_async_error, *args)
        return True

    def _start_session(self, subject: str = "", title: str = "") -> None:
        method = self._require_api("start_session")
        if method is None:
            return
        if not subject:
            subject = self.live.edt_subject.text().strip()
        if not title:
            title = self.live.edt_title.text().strip()
        self.live.btn_start.setEnabled(False)
        self._set_status("正在启动采集（打开音频设备与演示画面）……")
        self._call_api("start_session", self._on_session_start_requested, self._on_start_failed, subject, title)

    def _on_session_start_requested(self, _result: object) -> None:
        self.live.btn_start.setEnabled(True)
        self._set_status("已开始听课，正在采集音频与画面……")
        self._show_page(0)

    def _on_start_failed(self, error: str) -> None:
        self.live.btn_start.setEnabled(True)
        self._refresh_config_state()
        self._set_status(f"无法开始听课：{error}")
        QMessageBox.warning(
            self,
            "无法开始听课",
            f"{error}\n\n请先在“设置”里补全模型 API、个人信息与输出目录。",
        )

    def _toggle_pause(self, paused: bool) -> None:
        method = self._require_api("set_paused")
        if method is None:
            return
        try:
            method(bool(paused))
        except Exception as exc:
            self.logs.append_log("ERROR", f"暂停/恢复失败：{exc}")

    def _stop_session(self) -> None:
        if self._call_api("stop_session", self._on_stop_requested, self._on_stop_failed, True):
            self.live.btn_stop.setEnabled(False)
            self._set_status("正在收尾：补笔记、完成任务、导出作业……")

    def _on_stop_requested(self, _result: object) -> None:
        self._set_status("已停止采集，收尾在后台继续。")

    def _on_stop_failed(self, error: str) -> None:
        self.live.btn_stop.setEnabled(True)
        QMessageBox.warning(self, "停止失败", error)

    def _analyze_pending(self) -> None:
        if self._call_api("analyze_pending", lambda _r: self._set_status("已请求立即分析待处理切片。")):
            self._set_status("已请求立即分析待处理切片……")

    # -- 任务 --------------------------------------------------------
    def _solve_task(self, task_id: str) -> None:
        def _failed(error: str) -> None:
            QMessageBox.warning(self, "作答失败", error)

        if self._call_api("solve_task", lambda _r: None, _failed, str(task_id)):
            self._set_status("已提交作答请求，完成后会出现在“作业产出”。")

    def _regenerate_task(self, task_id: str) -> None:
        method = self._require_api("regenerate_task")
        if method is None:
            return
        feedback = TextPromptDialog.ask(
            self,
            "重做任务",
            "写下你的反馈（例如：过程再详细些、答案不对、换一种解法……），AI 会据此重做。",
            placeholder="可选：不填则按默认风格重做",
        )
        if feedback is None:
            return
        self._call_api(
            "regenerate_task",
            lambda _r: self._set_status("已提交重做请求。"),
            lambda error: QMessageBox.warning(self, "重做失败", error),
            str(task_id),
            feedback,
        )

    def _delete_task(self, task_id: str) -> None:
        method = self._require_api("delete_task")
        if method is None:
            return
        answer = QMessageBox.question(
            self,
            "删除任务",
            "确定要删除这条任务吗？已产出的作业文件不会被删除。",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if answer != QMessageBox.StandardButton.Yes:
            return
        self._call_api(
            "delete_task",
            lambda _r: self.tasks.remove_task(str(task_id)),
            lambda error: QMessageBox.warning(self, "删除失败", error),
            str(task_id),
        )

    def _add_manual_task(self) -> None:
        method = self._require_api("add_manual_task")
        if method is None:
            return
        values = TaskInputDialog.get_task(self)
        if not values:
            return
        title, detail = values
        self._call_api(
            "add_manual_task",
            lambda _r: self._set_status(f"已手动添加任务：{title}"),
            lambda error: QMessageBox.warning(self, "添加失败", error),
            title,
            detail,
        )

    def _solve_all(self) -> None:
        self._call_api(
            "solve_all_pending",
            lambda _r: self._set_status("已请求完成全部待作任务。"),
            lambda error: QMessageBox.warning(self, "批量作答失败", error),
        )

    # -- 数据刷新（全部走工作线程） -------------------------------------------------
    def _refresh_current_bundle(self) -> None:
        getter = getattr(self._api, "current_bundle", None)
        if not callable(getter):
            return
        run_async(getter, self._on_bundle, self._on_async_error)

    def _on_bundle(self, bundle: object) -> None:
        data = as_dict(bundle)
        tasks = data.get("tasks")
        if isinstance(tasks, list):
            self.tasks.set_tasks(tasks)
        if self._api is not None:
            self.artifacts.set_artifacts(
                [artifact for task in (tasks or []) for artifact in self._artifacts_of(task)]
            )

    @staticmethod
    def _artifacts_of(task: object) -> list[dict]:
        data = as_dict(task)
        return [
            {"task_id": get_str(data, "id"), "path": str(path), "title": get_str(data, "title"), "fmt": Path(str(path)).suffix.lstrip(".")}
            for path in (data.get("artifacts") or [])
            if str(path).strip()
        ]

    def _refresh_sessions(self) -> None:
        lister = getattr(self._api, "list_sessions", None)
        if not callable(lister):
            return
        self.history.lbl_detail.setText("正在读取历史会话……")
        run_async(lister, self._on_sessions, self._on_async_error)

    def _on_sessions(self, sessions: object) -> None:
        items = list(sessions) if isinstance(sessions, (list, tuple)) else []
        self.history.set_sessions(items)
        self.history.lbl_detail.setText(f"共 {len(items)} 个历史会话。双击查看详情。")

    def _load_session(self, session_id: str) -> None:
        loader = getattr(self._api, "load_session", None)
        if not callable(loader):
            return
        run_async(loader, self._on_session_loaded, self._on_async_error, str(session_id))

    def _on_session_loaded(self, bundle: object) -> None:
        if not bundle:
            QMessageBox.information(self, "没有找到", "该会话的内容不存在或已被清理。")
            return
        self.history.show_bundle(bundle)

    def _export_session(self, session_id: str) -> None:
        exporter = getattr(self._api, "export_session", None)
        if not callable(exporter):
            return
        target = str(session_id or self._session_id)
        if not target:
            QMessageBox.information(self, "无法导出", "还没有可导出的会话，请先听一次课。")
            return
        self._set_status("正在导出作业……")
        run_async(exporter, self._on_exported, self._on_async_error, target)

    def _on_exported(self, paths: object) -> None:
        items = [str(p) for p in (paths or [])] if isinstance(paths, (list, tuple)) else []
        if not items:
            self._set_status("没有需要导出的作业文件。")
            return
        self._set_status(f"已导出 {len(items)} 个文件。")
        QMessageBox.information(self, "导出完成", "已导出：\n" + "\n".join(items[:10]))

    def _open_session_dir(self, session_id: str) -> None:
        loader = getattr(self._api, "load_session", None)
        if not callable(loader):
            return
        run_async(loader, self._on_open_session_dir, self._on_async_error, str(session_id))

    def _on_open_session_dir(self, bundle: object) -> None:
        info = as_dict(as_dict(bundle).get("info"))
        directory = get_str(info, "directory")
        if directory:
            self.open_path(directory)

    def _open_output_dir(self) -> None:
        directory = ""
        try:
            directory = str(self._cfg.output.output_dir or self._cfg.resolved_output_dir())
        except Exception:
            directory = ""
        if directory:
            self.open_path(directory)

    def _export_logs(self) -> None:
        from ..config.paths import logs_root, timestamp_slug

        try:
            logs_root().mkdir(parents=True, exist_ok=True)
            target = logs_root() / f"ui-log-{timestamp_slug()}.txt"
            path = self.logs.view.export_to(str(target))
        except Exception as exc:
            QMessageBox.warning(self, "导出失败", str(exc))
            return
        self._set_status(f"日志已导出：{path}")

    def _on_async_error(self, error: str) -> None:
        self.logs.append_log("ERROR", f"后台任务失败：{error}")
        self._set_status(f"后台任务失败：{error}")

    # -- 设置 --------------------------------------------------------
    def _save_config(self, cfg: object) -> None:
        updater = getattr(self._api, "update_config", None)
        if not callable(updater) or not isinstance(cfg, AppConfig):
            self.settings.set_status("当前引擎不支持保存配置。")
            return
        try:
            updater(cfg)
        except Exception as exc:
            QMessageBox.warning(self, "保存失败", str(exc))
            return
        self._cfg = cfg
        self._refresh_config_state()
        self.settings.set_status("已保存配置。")
        self.logs.append_log("INFO", "配置已更新。")

    def _reload_settings(self) -> None:
        self._cfg = self._safe_get_config()
        self.settings.load_config(self._cfg)

    def open_settings(self) -> None:
        """打开设置对话框（模态）。"""
        dialog = SettingsDialog(self._cfg, api=self._api, parent=self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            applied = dialog.applied_config()
            self._cfg = applied if isinstance(applied, AppConfig) else self._safe_get_config()
            self._refresh_config_state()
            self.settings.load_config(self._cfg)

    def run_first_run_wizard(self) -> None:
        """运行首次运行向导，完成后保存配置。"""
        wizard = FirstRunWizard(self._cfg, parent=self)
        if wizard.exec() == QDialog.DialogCode.Accepted:
            cfg = wizard.result_config()
            self._save_config(cfg)
            self.settings.load_config(cfg)
            self._set_status("首次运行向导已完成。")

    def _show_about(self) -> None:
        QMessageBox.information(
            self,
            "关于 AI学生",
            "AI学生 1.0\n\n自动听课 · 记录课堂 · 完成老师布置的作业\n"
            "模型/语音识别由你在“设置”中配置，所有数据都保存在本机。",
        )

    def open_path(self, path: str) -> None:
        """调用引擎用资源管理器打开文件或目录。"""
        if not path:
            return
        opener = getattr(self._api, "open_path", None)
        if not callable(opener):
            return
        try:
            opener(str(path))
        except Exception as exc:
            self.logs.append_log("WARNING", f"打开路径失败：{exc}")

    # ------------------------------------------------------------ 关闭
    def closeEvent(self, event) -> None:  # noqa: N802 - Qt 命名
        """只发信号；真正的 shutdown 由 app.py 负责。"""
        self.closing.emit()
        super().closeEvent(event)
