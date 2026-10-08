"""主界面功能面板：听课 / 任务 / 作业产出 / 历史 / 日志 / 设置。

面板只与 ClassAgentAPI 交互，并以信号把“需要弹窗/需要引擎动作”的事情交给 MainWindow，
这样每个面板都能在 offscreen 下单独实例化并测试。
"""
from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QSize, Qt, QTimer, Signal
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMenu,
    QPlainTextEdit,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from .settings_dialog import SettingsForm
from .widgets import (
    Card,
    LevelMeter,
    LogView,
    StatusPill,
    TimecodeList,
    as_dict,
    get_float,
    get_str,
    hms,
    make_button,
    run_async,
)

TASK_STATUS_TEXT = {
    "discovered": "已发现",
    "queued": "排队中",
    "working": "作答中",
    "solved": "已完成",
    "failed": "失败",
    "skipped": "已跳过",
}

TASK_COLUMNS = ["标题", "类型", "状态", "置信度", "产出", "操作"]


def _ro_text(placeholder: str = "") -> QPlainTextEdit:
    """只读文本区。"""
    view = QPlainTextEdit()
    view.setReadOnly(True)
    view.setPlaceholderText(placeholder)
    return view


def _fmt_size(size: int) -> str:
    try:
        value = float(size)
    except (TypeError, ValueError):
        return "-"
    for unit in ("B", "KB", "MB", "GB"):
        if value < 1024 or unit == "GB":
            return f"{value:.0f} {unit}" if unit == "B" else f"{value:.1f} {unit}"
        value /= 1024
    return "-"


# ================================================================ 听课


class LivePanel(QWidget):
    """听课页：状态胶囊 + 会话信息 + 控制按钮 + 电平条 + 画面预览 + 转写 + 笔记。"""

    startRequested = Signal(str, str)      # (科目, 标题)
    pauseToggled = Signal(bool)            # True=暂停
    stopRequested = Signal()
    analyzeRequested = Signal()

    def __init__(self, api=None, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._api = api
        self._state = "idle"
        self._elapsed = 0
        self._last_image: str = ""
        self._last_image_mtime: float = 0.0
        self._pixmap_cache: QPixmap | None = None
        self._preview_busy = False
        self._last_png: bytes = b""

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(10)

        # -- 顶部控制条 --------------------------------------------------
        top = Card()
        header = QHBoxLayout()
        self.pill = StatusPill()
        self.lbl_stage = QLabel("就绪")
        self.lbl_stage.setObjectName("hint")
        self.lbl_timer = QLabel("00:00:00")
        self.lbl_timer.setObjectName("monoValue")
        header.addWidget(self.pill)
        header.addWidget(self.lbl_stage, 1)
        header.addWidget(self.lbl_timer)
        top.body().addLayout(header)

        info = QHBoxLayout()
        self.edt_subject = QLineEdit()
        self.edt_subject.setPlaceholderText("科目（如：数学）")
        self.edt_subject.setMaximumWidth(180)
        self.edt_title = QLineEdit()
        self.edt_title.setPlaceholderText("课堂标题（可留空，默认按时间命名）")
        info.addWidget(QLabel("科目"))
        info.addWidget(self.edt_subject)
        info.addWidget(QLabel("标题"))
        info.addWidget(self.edt_title, 1)
        top.body().addLayout(info)

        controls = QHBoxLayout()
        self.btn_start = make_button("开始听课", "primary")
        self.btn_pause = make_button("暂停")
        self.btn_stop = make_button("停止并收尾", "danger")
        self.btn_analyze = make_button("立即分析", "ghost", "把尚未处理的切片马上送给模型")
        self.btn_pause.setEnabled(False)
        self.btn_stop.setEnabled(False)
        self.btn_analyze.setEnabled(False)
        controls.addWidget(self.btn_start)
        controls.addWidget(self.btn_pause)
        controls.addWidget(self.btn_stop)
        controls.addWidget(self.btn_analyze)
        controls.addStretch(1)
        controls.addWidget(QLabel("麦克风电平"))
        self.meter = LevelMeter()
        self.meter.setMinimumWidth(220)
        controls.addWidget(self.meter, 1)
        top.body().addLayout(controls)
        outer.addWidget(top)

        # -- 主体 --------------------------------------------------------
        splitter = QSplitter(Qt.Orientation.Horizontal)
        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(10)

        preview_card = Card("演示画面")
        self.lbl_preview = QLabel("尚未采集到画面")
        self.lbl_preview.setObjectName("previewLabel")
        self.lbl_preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_preview.setMinimumSize(QSize(320, 200))
        self.lbl_preview.setScaledContents(False)
        preview_card.add_widget(self.lbl_preview, 1)
        left_layout.addWidget(preview_card, 3)

        notes_card = Card("当前课堂笔记")
        self.view_notes = _ro_text("课堂笔记会在听课过程中自动累积……")
        notes_card.add_widget(self.view_notes, 1)
        left_layout.addWidget(notes_card, 2)

        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(10)
        transcript_card = Card("实时转写")
        search_row = QHBoxLayout()
        self.edt_search = QLineEdit()
        self.edt_search.setPlaceholderText("搜索转写内容……")
        self.lbl_count = QLabel("0 段")
        self.lbl_count.setObjectName("hint")
        search_row.addWidget(self.edt_search, 1)
        search_row.addWidget(self.lbl_count)
        transcript_card.body().addLayout(search_row)
        self.list_transcript = TimecodeList()
        transcript_card.add_widget(self.list_transcript, 1)
        right_layout.addWidget(transcript_card, 1)
        splitter.addWidget(left)
        splitter.addWidget(right)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 4)
        splitter.setSizes([560, 620])
        outer.addWidget(splitter, 1)

        self._timer = QTimer(self)
        self._timer.setInterval(1000)
        self._timer.timeout.connect(self._on_tick)
        # 电平表：契约说明 audio_level_db() 可以 100ms 轮询（只是个浮点读取）
        self._level_timer = QTimer(self)
        self._level_timer.setInterval(120)
        self._level_timer.timeout.connect(self.poll_level)
        # 关键帧文件回退路径（引擎不提供 preview_png 时使用）
        self._preview_timer = QTimer(self)
        self._preview_timer.setInterval(1200)
        self._preview_timer.timeout.connect(self.refresh_preview)

        self.btn_start.clicked.connect(
            lambda: self.startRequested.emit(self.edt_subject.text().strip(), self.edt_title.text().strip())
        )
        self.btn_pause.clicked.connect(self._on_pause_clicked)
        self.btn_stop.clicked.connect(self.stopRequested.emit)
        self.btn_analyze.clicked.connect(self.analyzeRequested.emit)
        self.edt_search.textChanged.connect(self._on_search)

    # -- 交互 --------------------------------------------------------
    def _on_pause_clicked(self) -> None:
        pause = self._state != "paused"
        self.pauseToggled.emit(pause)

    def _on_search(self, text: str) -> None:
        hits = self.list_transcript.apply_filter(text)
        self.lbl_count.setText(f"{hits} 段")

    def _on_tick(self) -> None:
        if self._state == "recording":
            self._elapsed += 1
            self.lbl_timer.setText(hms(self._elapsed))
        if self._state in ("recording", "paused"):
            self.poll_preview()

    # -- 引擎信号入口 -------------------------------------------------
    def set_state(self, state: str, message: str = "") -> None:
        """更新状态胶囊与按钮可用性。"""
        self._state = str(state or "idle")
        self.pill.set_status(self._state, message)
        recording = self._state == "recording"
        paused = self._state == "paused"
        active = recording or paused
        self.btn_start.setEnabled(not active)
        self.btn_pause.setEnabled(active)
        self.btn_pause.setText("恢复" if paused else "暂停")
        self.btn_stop.setEnabled(active)
        self.btn_analyze.setEnabled(active)
        if recording:
            self._timer.start()
            self._level_timer.start()
            self._preview_timer.start()
        else:
            self._timer.stop()
            self._level_timer.stop()
            if self._state in ("idle", "finished", "error"):
                self._preview_timer.stop()
        if self._state in ("idle", "finished", "error"):
            self._elapsed = 0
            self.lbl_timer.setText("00:00:00")
        if not active:
            self.meter.set_idle("已停止" if self._state != "idle" else "未采集")

    def set_stage(self, stage: str, message: str = "") -> None:
        """更新阶段说明（转写中 / 分析板书 / 生成笔记……）。"""
        label = message or str(stage or "")
        self.lbl_stage.setText(label)

    def set_session_info(self, payload: object) -> None:
        """会话开始：回填科目与标题。"""
        data = as_dict(payload)
        subject = get_str(data, "subject")
        title = get_str(data, "title")
        if subject:
            self.edt_subject.setText(subject)
        if title:
            self.edt_title.setText(title)

    def reset_session(self) -> None:
        """会话结束后的清理（保留笔记与转写供查看）。"""
        self._elapsed = 0
        self.lbl_timer.setText("00:00:00")
        self.meter.set_idle()

    def add_transcript(self, payload: object) -> None:
        self.list_transcript.add_segment(payload)
        self._on_search(self.edt_search.text())

    def set_notes(self, markdown: str) -> None:
        self.view_notes.setPlainText(markdown or "")
        bar = self.view_notes.verticalScrollBar()
        bar.setValue(bar.maximum())

    def show_board_image(self, image_path: str) -> None:
        """记录最新关键帧路径，交由定时器刷新（避免每次信号都读盘）。"""
        path = str(image_path or "")
        if path:
            self._last_image = path
            self._preview_timer.start()

    def refresh_preview(self) -> None:
        """定时器回调：关键帧文件有更新时重新载入并缩放显示（回退路径）。"""
        if not self._last_image:
            return
        path = Path(self._last_image)
        try:
            stat = path.stat()
        except OSError:
            return
        if self._pixmap_cache is not None and abs(stat.st_mtime - self._last_image_mtime) < 1e-6:
            return
        pixmap = QPixmap(str(path))
        if pixmap.isNull():
            return
        self._last_image_mtime = stat.st_mtime
        self._show_pixmap(pixmap)

    def poll_level(self) -> None:
        """轮询引擎音频电平（audio_level_db 只是读一个浮点，直接在 UI 线程调用）。"""
        getter = getattr(self._api, "audio_level_db", None)
        if not callable(getter):
            return
        try:
            value = float(getter())
        except Exception:
            return
        if value <= -95.0:  # 引擎约定：未采集返回 -96.0
            self.meter.set_idle()
        else:
            self.meter.set_level(value)

    def poll_preview(self) -> None:
        """在工作线程里取最近一帧 PNG，避免编码/读盘卡住 UI 线程。"""
        if self._preview_busy:
            return
        getter = getattr(self._api, "preview_png", None)
        if not callable(getter):
            return
        self._preview_busy = True
        run_async(getter, self._on_preview_bytes, self._on_preview_failed)

    def _on_preview_bytes(self, payload: object) -> None:
        self._preview_busy = False
        if isinstance(payload, (bytes, bytearray)) and payload:
            data = bytes(payload)
            if data != self._last_png:
                pixmap = QPixmap()
                if pixmap.loadFromData(data) and not pixmap.isNull():
                    self._last_png = data
                    self._show_pixmap(pixmap)
                    return
        # 引擎暂时给不出画面：退回最近关键帧文件
        self.refresh_preview()

    def _on_preview_failed(self, _error: str) -> None:
        self._preview_busy = False

    def _show_pixmap(self, pixmap: QPixmap) -> None:
        """按控件大小等比缩放并显示。"""
        self._pixmap_cache = pixmap
        area = self.lbl_preview.size()
        scaled = pixmap.scaled(
            max(64, area.width() - 8),
            max(64, area.height() - 8),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.lbl_preview.setPixmap(scaled)

    def set_level(self, db: float) -> None:
        self.meter.set_level(db)

    def preview_busy(self) -> bool:
        """是否有正在进行的预览取样（供测试等待）。"""
        return self._preview_busy

    def transcript_count(self) -> int:
        return self.list_transcript.count()


# ================================================================ 任务


class TasksPanel(QWidget):
    """任务页：表格 + 每行操作按钮 + 右键菜单。"""

    solveRequested = Signal(str)          # task_id
    regenerateRequested = Signal(str)     # task_id（反馈由 MainWindow 弹框收集）
    deleteRequested = Signal(str)         # task_id
    openPathRequested = Signal(str)       # 产出文件路径
    addRequested = Signal()               # 手动添加任务
    solveAllRequested = Signal()
    refreshRequested = Signal()

    def __init__(self, api=None, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._api = api
        self._tasks: dict[str, dict] = {}
        self._rows: dict[str, int] = {}

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(10)

        bar = QHBoxLayout()
        title = QLabel("课堂任务")
        title.setObjectName("pageTitle")
        bar.addWidget(title)
        bar.addStretch(1)
        self.btn_add = make_button("手动添加任务")
        self.btn_solve_all = make_button("全部立即完成", "primary")
        self.btn_refresh = make_button("刷新", "ghost")
        bar.addWidget(self.btn_add)
        bar.addWidget(self.btn_solve_all)
        bar.addWidget(self.btn_refresh)
        outer.addLayout(bar)

        self.lbl_summary = QLabel("暂无任务。")
        self.lbl_summary.setObjectName("hint")
        outer.addWidget(self.lbl_summary)

        self.table = QTableWidget(0, len(TASK_COLUMNS))
        self.table.setHorizontalHeaderLabels(TASK_COLUMNS)
        self.table.verticalHeader().setVisible(False)
        self.table.setAlternatingRowColors(True)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        for column, width in ((1, 72), (2, 78), (3, 76), (4, 56), (5, 272)):
            header.setSectionResizeMode(column, QHeaderView.ResizeMode.Fixed)
            self.table.setColumnWidth(column, width)
        self.table.verticalHeader().setDefaultSectionSize(38)
        outer.addWidget(self.table, 1)

        self.table.customContextMenuRequested.connect(self._on_context_menu)
        self.btn_add.clicked.connect(self.addRequested.emit)
        self.btn_solve_all.clicked.connect(self.solveAllRequested.emit)
        self.btn_refresh.clicked.connect(self.refreshRequested.emit)

    # -- 数据 --------------------------------------------------------
    def upsert_task(self, payload: object) -> None:
        """新增或刷新一行任务。"""
        data = as_dict(payload)
        task_id = get_str(data, "id")
        if not task_id or not data:
            return
        self._tasks[task_id] = data
        row = self._rows.get(task_id)
        if row is None:
            row = self.table.rowCount()
            self.table.insertRow(row)
            self._rows[task_id] = row
        self._fill_row(row, data)
        self._update_summary()

    def remove_task(self, task_id: str) -> None:
        row = self._rows.pop(str(task_id), None)
        self._tasks.pop(str(task_id), None)
        if row is None:
            self._update_summary()
            return
        self.table.removeRow(row)
        self._rows = {
            key: (value - 1 if value > row else value) for key, value in self._rows.items()
        }
        self._update_summary()

    def set_tasks(self, tasks: list) -> None:
        """整体替换任务列表。"""
        self.clear_tasks()
        for item in tasks or []:
            self.upsert_task(item)

    def clear_tasks(self) -> None:
        self.table.setRowCount(0)
        self._tasks.clear()
        self._rows.clear()
        self._update_summary()

    def task_ids(self) -> list[str]:
        return list(self._tasks.keys())

    def _update_summary(self) -> None:
        total = len(self._tasks)
        if not total:
            self.lbl_summary.setText("暂无任务。老师布置任务后会自动出现在这里。")
            return
        solved = sum(1 for t in self._tasks.values() if get_str(t, "status") == "solved")
        self.lbl_summary.setText(f"共 {total} 条任务 · 已完成 {solved} 条")

    def _fill_row(self, row: int, data: dict) -> None:
        title = get_str(data, "title") or "（未命名任务）"
        detail = get_str(data, "detail")
        values = [
            title,
            get_str(data, "task_type", "作业"),
            TASK_STATUS_TEXT.get(get_str(data, "status"), get_str(data, "status", "-")),
            f"{get_float(data, 'confidence') * 100:.0f}%",
            str(len(data.get("artifacts") or [])),
        ]
        for column, text in enumerate(values):
            item = self.table.item(row, column)
            if item is None:
                item = QTableWidgetItem()
                self.table.setItem(row, column, item)
            item.setText(text)
            if column == 0:
                item.setToolTip(detail or title)
        self.table.setCellWidget(row, 5, self._action_widget(get_str(data, "id"), data))

    def _action_widget(self, task_id: str, data: dict) -> QWidget:
        holder = QWidget()
        layout = QHBoxLayout(holder)
        layout.setContentsMargins(2, 2, 2, 2)
        layout.setSpacing(4)
        artifacts = [str(p) for p in (data.get("artifacts") or []) if str(p).strip()]
        buttons = [
            ("完成", lambda: self.solveRequested.emit(task_id), ""),
            ("重做", lambda: self.regenerateRequested.emit(task_id), ""),
            ("删除", lambda: self.deleteRequested.emit(task_id), "danger"),
        ]
        for text, slot, role in buttons:
            btn = make_button(text, role)
            btn.setObjectName("tableAction" if not role else role + "TableAction")
            btn.clicked.connect(slot)
            layout.addWidget(btn)
        if artifacts:
            open_btn = make_button("打开产出")
            open_btn.setObjectName("tableAction")
            open_btn.clicked.connect(lambda: self.openPathRequested.emit(artifacts[0]))
            layout.addWidget(open_btn)
        layout.addStretch(1)
        return holder

    # -- 右键菜单 ----------------------------------------------------
    def _on_context_menu(self, pos) -> None:
        row = self.table.rowAt(pos.y())
        if row < 0 or row >= self.table.rowCount():
            return
        task_id = self._task_id_at_row(row)
        if not task_id:
            return
        data = self._tasks.get(task_id, {})
        menu = QMenu(self)
        act_solve = menu.addAction("立即完成")
        act_regen = menu.addAction("重做（填写反馈）")
        act_open = menu.addAction("打开产出文件")
        artifacts = [str(p) for p in (data.get("artifacts") or []) if str(p).strip()]
        act_open.setEnabled(bool(artifacts))
        menu.addSeparator()
        act_delete = menu.addAction("删除任务")
        chosen = menu.exec(self.table.viewport().mapToGlobal(pos))
        if chosen is act_solve:
            self.solveRequested.emit(task_id)
        elif chosen is act_regen:
            self.regenerateRequested.emit(task_id)
        elif chosen is act_open and artifacts:
            self.openPathRequested.emit(artifacts[0])
        elif chosen is act_delete:
            self.deleteRequested.emit(task_id)

    def _task_id_at_row(self, row: int) -> str:
        for task_id, index in self._rows.items():
            if index == row:
                return task_id
        return ""


# ================================================================ 作业产出


class ArtifactsPanel(QWidget):
    """作业产出页：产物列表 + 打开输出文件夹 + 一键导出本次会话。"""

    openPathRequested = Signal(str)
    openFolderRequested = Signal()
    exportRequested = Signal()
    refreshRequested = Signal()

    COLUMNS = ["文件", "格式", "大小", "生成时间", "操作"]

    def __init__(self, api=None, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._api = api
        self._items: list[dict] = []

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(10)

        bar = QHBoxLayout()
        title = QLabel("作业产出")
        title.setObjectName("pageTitle")
        bar.addWidget(title)
        bar.addStretch(1)
        self.btn_folder = make_button("打开输出文件夹")
        self.btn_export = make_button("一键导出本次会话", "primary")
        self.btn_refresh = make_button("刷新", "ghost")
        bar.addWidget(self.btn_folder)
        bar.addWidget(self.btn_export)
        bar.addWidget(self.btn_refresh)
        outer.addLayout(bar)

        self.lbl_dir = QLabel("输出目录：未设置")
        self.lbl_dir.setObjectName("hint")
        self.lbl_dir.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        outer.addWidget(self.lbl_dir)

        self.table = QTableWidget(0, len(self.COLUMNS))
        self.table.setHorizontalHeaderLabels(self.COLUMNS)
        self.table.verticalHeader().setVisible(False)
        self.table.setAlternatingRowColors(True)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        for column, width in ((1, 70), (2, 90), (3, 150), (4, 90)):
            header.setSectionResizeMode(column, QHeaderView.ResizeMode.Fixed)
            self.table.setColumnWidth(column, width)
        self.table.verticalHeader().setDefaultSectionSize(34)
        self.table.itemDoubleClicked.connect(self._on_double_click)
        self.table.customContextMenuRequested.connect(self._on_context_menu)
        outer.addWidget(self.table, 1)

        self.btn_folder.clicked.connect(self.openFolderRequested.emit)
        self.btn_export.clicked.connect(self.exportRequested.emit)
        self.btn_refresh.clicked.connect(self.refreshRequested.emit)

    # -- 数据 --------------------------------------------------------
    def add_artifact(self, payload: object) -> None:
        data = as_dict(payload)
        if not data:
            return
        path = get_str(data, "path")
        if any(get_str(item, "path") == path and path for item in self._items):
            return
        self._items.append(data)
        self._append_row(data)

    def set_artifacts(self, items: list) -> None:
        self.clear_all()
        for item in items or []:
            self.add_artifact(item)

    def clear_all(self) -> None:
        self.table.setRowCount(0)
        self._items.clear()

    def paths(self) -> list[str]:
        return [get_str(item, "path") for item in self._items if get_str(item, "path")]

    def set_output_dir(self, path: str) -> None:
        self.lbl_dir.setText(f"输出目录：{path or '未设置'}")

    def _append_row(self, data: dict) -> None:
        row = self.table.rowCount()
        self.table.insertRow(row)
        values = [
            get_str(data, "title") or Path(get_str(data, "path")).name,
            get_str(data, "fmt"),
            _fmt_size(data.get("size_bytes", 0)),
            get_str(data, "created_at").replace("T", " "),
        ]
        for column, text in enumerate(values):
            item = QTableWidgetItem(text)
            item.setToolTip(get_str(data, "path"))
            self.table.setItem(row, column, item)
        holder = QWidget()
        layout = QHBoxLayout(holder)
        layout.setContentsMargins(2, 2, 2, 2)
        open_btn = make_button("打开")
        open_btn.setObjectName("tableAction")
        path = get_str(data, "path")
        open_btn.clicked.connect(lambda: self.openPathRequested.emit(path))
        layout.addWidget(open_btn)
        layout.addStretch(1)
        self.table.setCellWidget(row, 4, holder)

    def _path_at_row(self, row: int) -> str:
        item = self.table.item(row, 0)
        if item is None:
            return ""
        return item.toolTip() or ""

    def _on_double_click(self, item: QTableWidgetItem) -> None:
        path = self._path_at_row(item.row())
        if path:
            self.openPathRequested.emit(path)

    def _on_context_menu(self, pos) -> None:
        row = self.table.rowAt(pos.y())
        if row < 0:
            return
        path = self._path_at_row(row)
        menu = QMenu(self)
        act_open = menu.addAction("打开文件")
        act_open.setEnabled(bool(path))
        act_folder = menu.addAction("打开所在文件夹")
        act_folder.setEnabled(bool(path))
        chosen = menu.exec(self.table.viewport().mapToGlobal(pos))
        if chosen is act_open and path:
            self.openPathRequested.emit(path)
        elif chosen is act_folder and path:
            self.openPathRequested.emit(str(Path(path).parent))


# ================================================================ 历史


class HistoryPanel(QWidget):
    """历史页：左列会话列表 + 右列详情（转写 / 板书 / 任务 / 笔记）。"""

    refreshRequested = Signal()
    loadRequested = Signal(str)
    exportRequested = Signal(str)
    openFolderRequested = Signal(str)

    def __init__(self, api=None, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._api = api

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(10)

        bar = QHBoxLayout()
        title = QLabel("听课历史")
        title.setObjectName("pageTitle")
        bar.addWidget(title)
        bar.addStretch(1)
        self.edt_search = QLineEdit()
        self.edt_search.setPlaceholderText("搜索会话……")
        self.edt_search.setMaximumWidth(220)
        self.btn_refresh = make_button("刷新列表")
        bar.addWidget(self.edt_search)
        bar.addWidget(self.btn_refresh)
        outer.addLayout(bar)

        splitter = QSplitter(Qt.Orientation.Horizontal)

        left_card = Card("会话")
        self.list_sessions = QListWidget()
        self.list_sessions.setObjectName("historyList")
        self.list_sessions.setAlternatingRowColors(False)
        left_card.add_widget(self.list_sessions, 1)
        buttons_row = QHBoxLayout()
        self.btn_load = make_button("载入详情", "primary")
        self.btn_export = make_button("导出作业")
        self.btn_folder = make_button("打开文件夹", "ghost")
        buttons_row.addWidget(self.btn_load)
        buttons_row.addWidget(self.btn_export)
        buttons_row.addWidget(self.btn_folder)
        left_card.body().addLayout(buttons_row)

        right_card = Card("详情")
        self.tabs = QTabWidget()
        self.view_transcript = _ro_text("该会话的转写内容……")
        self.view_board = _ro_text("该会话的板书 / 演示要点……")
        self.view_tasks = _ro_text("该会话识别到的任务……")
        self.view_notes = _ro_text("该会话的课堂笔记……")
        self.tabs.addTab(self.view_transcript, "转写")
        self.tabs.addTab(self.view_board, "板书")
        self.tabs.addTab(self.view_tasks, "任务")
        self.tabs.addTab(self.view_notes, "笔记")
        right_card.add_widget(self.tabs, 1)
        self.lbl_detail = QLabel("双击左侧会话查看详情。")
        self.lbl_detail.setObjectName("hint")
        right_card.body().addWidget(self.lbl_detail)

        splitter.addWidget(left_card)
        splitter.addWidget(right_card)
        splitter.setStretchFactor(0, 2)
        splitter.setStretchFactor(1, 5)
        splitter.setSizes([360, 880])
        outer.addWidget(splitter, 1)

        self.btn_refresh.clicked.connect(self.refreshRequested.emit)
        self.btn_load.clicked.connect(self._emit_load)
        self.btn_export.clicked.connect(self._emit_export)
        self.btn_folder.clicked.connect(self._emit_folder)
        self.list_sessions.itemDoubleClicked.connect(lambda _item: self._emit_load())
        self.edt_search.textChanged.connect(self._apply_filter)

    # -- 数据 --------------------------------------------------------
    def set_sessions(self, sessions: list) -> None:
        """刷新会话列表（保留当前选中项）。"""
        keep = self.selected_session_id()
        self.list_sessions.clear()
        for raw in sessions or []:
            data = as_dict(raw)
            name = get_str(data, "display_name")
            session_id = get_str(data, "id")
            if not session_id:
                continue
            if not name:
                name = f"{get_str(data, 'started_at').replace('T', ' ')[:16]} {get_str(data, 'subject')}".strip()
            item = QListWidgetItem(name or session_id)
            item.setData(Qt.ItemDataRole.UserRole, session_id)
            item.setToolTip(
                f"转写 {data.get('transcript_count', 0)} 段 · 板书 {data.get('board_count', 0)} 帧 · "
                f"任务 {data.get('task_count', 0)} 条 · 产出 {data.get('artifact_count', 0)} 个"
            )
            self.list_sessions.addItem(item)
        if keep:
            self.select_session(keep)
        self._apply_filter(self.edt_search.text())

    def select_session(self, session_id: str) -> None:
        for row in range(self.list_sessions.count()):
            item = self.list_sessions.item(row)
            if item.data(Qt.ItemDataRole.UserRole) == session_id:
                self.list_sessions.setCurrentItem(item)
                return

    def selected_session_id(self) -> str:
        item = self.list_sessions.currentItem()
        return str(item.data(Qt.ItemDataRole.UserRole)) if item else ""

    def _apply_filter(self, text: str) -> None:
        needle = (text or "").strip().lower()
        for row in range(self.list_sessions.count()):
            item = self.list_sessions.item(row)
            item.setHidden(bool(needle) and needle not in item.text().lower())

    def show_bundle(self, payload: object) -> None:
        """展示一个会话的完整内容。"""
        data = as_dict(payload)
        info = as_dict(data.get("info")) if data.get("info") is not None else {}
        header = get_str(info, "display_name") or get_str(info, "id")
        self.lbl_detail.setText(f"当前会话：{header}" if header else "会话内容")
        transcript_lines = []
        for segment in data.get("transcript") or []:
            seg = as_dict(segment)
            timecode = get_str(seg, "timecode") or hms(get_float(seg, "t_start"))
            speaker = get_str(seg, "speaker")
            transcript_lines.append(f"[{timecode}] {speaker} {get_str(seg, 'text')}".strip())
        self.view_transcript.setPlainText("\n".join(transcript_lines) or "（无转写记录）")

        board_lines = []
        for note in data.get("board") or []:
            board = as_dict(note)
            timecode = get_str(board, "timecode") or hms(get_float(board, "t"))
            bits = [f"[{timecode}]"]
            if get_str(board, "summary"):
                bits.append(get_str(board, "summary"))
            if get_str(board, "ocr_text"):
                bits.append(get_str(board, "ocr_text"))
            board_lines.append(" ".join(bits))
        self.view_board.setPlainText("\n".join(board_lines) or "（无板书记录）")

        task_lines = []
        for task in data.get("tasks") or []:
            item = as_dict(task)
            line = f"· [{TASK_STATUS_TEXT.get(get_str(item, 'status'), get_str(item, 'status'))}] {get_str(item, 'title')}"
            if get_str(item, "detail"):
                line += f"\n    {get_str(item, 'detail')}"
            paths = [str(p) for p in (item.get("artifacts") or []) if str(p).strip()]
            for path in paths:
                line += f"\n    产出：{path}"
            task_lines.append(line)
        self.view_tasks.setPlainText("\n".join(task_lines) or "（无任务记录）")
        self.view_notes.setPlainText(get_str(data, "notes_markdown") or "（无课堂笔记）")

    def _emit_load(self) -> None:
        session_id = self.selected_session_id()
        if session_id:
            self.loadRequested.emit(session_id)

    def _emit_export(self) -> None:
        session_id = self.selected_session_id()
        if session_id:
            self.exportRequested.emit(session_id)

    def _emit_folder(self) -> None:
        item = self.list_sessions.currentItem()
        if item is None:
            return
        session_id = str(item.data(Qt.ItemDataRole.UserRole))
        if session_id:
            self.openFolderRequested.emit(session_id)


# ================================================================ 日志


class LogPanel(QWidget):
    """日志页：级别过滤 + 导出。"""

    exportRequested = Signal()
    refreshRequested = Signal()

    def __init__(self, api=None, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._api = api

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(10)

        bar = QHBoxLayout()
        title = QLabel("运行日志")
        title.setObjectName("pageTitle")
        bar.addWidget(title)
        bar.addStretch(1)
        bar.addWidget(QLabel("显示级别"))
        self.cmb_level = QComboBox()
        self.cmb_level.addItem("全部", "DEBUG")
        self.cmb_level.addItem("信息以上", "INFO")
        self.cmb_level.addItem("警告以上", "WARNING")
        self.cmb_level.addItem("仅错误", "ERROR")
        bar.addWidget(self.cmb_level)
        self.btn_clear = make_button("清空")
        self.btn_export = make_button("导出日志")
        bar.addWidget(self.btn_clear)
        bar.addWidget(self.btn_export)
        outer.addLayout(bar)

        self.view = LogView()
        outer.addWidget(self.view, 1)

        self.cmb_level.currentIndexChanged.connect(
            lambda: self.view.set_min_level(str(self.cmb_level.currentData()))
        )
        self.btn_clear.clicked.connect(self.view.clear_log)
        self.btn_export.clicked.connect(self.exportRequested.emit)

    def append_log(self, level: str, message: str) -> None:
        self.view.append_log(level, message)

    def lines(self) -> list[str]:
        return self.view.text_lines()


# ================================================================ 设置


class SettingsPanel(QWidget):
    """设置页：嵌入 SettingsForm，保存时交给引擎落盘。"""

    saveRequested = Signal(object)     # AppConfig
    reloadRequested = Signal()
    fullDialogRequested = Signal()

    def __init__(self, cfg=None, api=None, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._api = api

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(10)

        bar = QHBoxLayout()
        title = QLabel("设置")
        title.setObjectName("pageTitle")
        bar.addWidget(title)
        bar.addStretch(1)
        self.lbl_status = QLabel("")
        self.lbl_status.setObjectName("hint")
        bar.addWidget(self.lbl_status)
        self.btn_reload = make_button("重新载入")
        self.btn_dialog = make_button("独立窗口打开", "ghost")
        self.btn_save = make_button("保存设置", "primary")
        bar.addWidget(self.btn_reload)
        bar.addWidget(self.btn_dialog)
        bar.addWidget(self.btn_save)
        outer.addLayout(bar)

        self.form = SettingsForm(cfg, api=api, parent=self)
        outer.addWidget(self.form, 1)

        self.btn_save.clicked.connect(self._on_save)
        self.btn_reload.clicked.connect(self.reloadRequested.emit)
        self.btn_dialog.clicked.connect(self.fullDialogRequested.emit)

    def load_config(self, cfg) -> None:
        """把引擎最新配置灌进表单。"""
        self.form.set_config(cfg)
        self.lbl_status.setText("已载入当前配置。")

    def set_status(self, text: str) -> None:
        self.lbl_status.setText(text)

    def _on_save(self) -> None:
        cfg = self.form.result_config()
        self.saveRequested.emit(cfg)

