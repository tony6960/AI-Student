"""AppController：把 ClassEngine 包装成 GUI 可直接使用的 ClassAgentAPI。

- 负责配置的加载/保存（密钥加密落盘）
- 把引擎回调翻译成 Qt 信号（引擎回调来自工作线程，Qt 自动走队列投递）
- 提供会话历史、设备探测、文件打开等界面需要的服务
"""
from __future__ import annotations

import copy
import logging
import os
import subprocess
from pathlib import Path
from typing import Any

from PySide6.QtCore import QObject

from ..config.models import AppConfig
from ..config.paths import default_output_dir
from ..config.store import load_config, save_config
from .api import ClassAgentAPI
from .engine import ClassEngine, EngineCallbacks
from .models import SessionBundle, SessionInfo
from .session import SessionStore

log = logging.getLogger(__name__)


class AppController(ClassAgentAPI):
    """GUI 与引擎之间的唯一门面。"""

    def __init__(self, parent: QObject | None = None, config: AppConfig | None = None,
                 store: SessionStore | None = None, engine: ClassEngine | None = None) -> None:
        super().__init__(parent)
        self._cfg = config or load_config()
        if not self._cfg.output.output_dir:
            self._cfg.output.output_dir = str(default_output_dir())
        self._store = store or SessionStore()
        self._engine = engine or ClassEngine(self.get_config, self._callbacks(), store=self._store)

    # ------------------------------------------------------------ 配置
    def get_config(self) -> AppConfig:
        return copy.deepcopy(self._cfg)

    def raw_config(self) -> AppConfig:
        return self._cfg

    def update_config(self, cfg: AppConfig) -> None:
        self._cfg = copy.deepcopy(cfg)
        try:
            save_config(self._cfg)
        except Exception as exc:
            log.exception("保存配置失败")
            self.errorRaised.emit("保存配置失败", str(exc))
        self._engine.get_config = self.get_config
        self.configChanged.emit()

    def needs_setup(self) -> list[str]:
        return self._cfg.validate()

    # ------------------------------------------------------------ 引擎回调 → Qt 信号
    def _callbacks(self) -> EngineCallbacks:
        def on_task(task) -> None:
            if task.status == "discovered":
                self.taskDiscovered.emit(task.to_dict())
            else:
                self.taskUpdated.emit(task.to_dict())

        def on_session(info) -> None:
            # 开始信号由 start_session() 负责发；这里只发结束信号，避免重复
            if info.state != "recording":
                self.sessionStopped.emit(info.to_dict())

        return EngineCallbacks(
            on_state=lambda state, detail: self.stateChanged.emit(state, detail),
            on_stage=lambda stage, detail: self.stageChanged.emit(stage, detail),
            on_progress=lambda stage, pct: self.progress.emit(stage, int(pct)),
            on_transcript=lambda seg: self.transcriptSegment.emit(seg.to_dict()),
            on_board=lambda note: self.boardNote.emit(note.to_dict()),
            on_notes=lambda md: self.notesUpdated.emit(md),
            on_task=on_task,
            on_artifact=lambda art: self.artifactWritten.emit(art.to_dict()),
            on_log=lambda level, msg: self.logMessage.emit(level, msg),
            on_error=lambda title, detail: self.errorRaised.emit(title, detail),
            on_session=on_session,
            on_ai_health=lambda msg: self.logMessage.emit("WARNING", msg),
        )

    # ------------------------------------------------------------ 听课控制
    def start_session(self, subject: str = "", title: str = "") -> str:
        try:
            info = self._engine.start(subject=subject, title=title)
        except ValueError as exc:
            self.errorRaised.emit("无法开始听课", str(exc))
            raise
        except Exception as exc:
            log.exception("开始听课失败")
            self.errorRaised.emit("开始听课失败", str(exc))
            raise
        self.sessionStarted.emit(info.to_dict())
        return info.id

    def stop_session(self, finalize: bool = True) -> None:
        self._engine.stop(finalize=finalize)

    def set_paused(self, paused: bool) -> None:
        self._engine.set_paused(paused)

    def is_recording(self) -> bool:
        return self._engine.is_recording()

    def analyze_pending(self) -> None:
        self._engine.analyze_pending()

    def audio_level_db(self) -> float:
        return self._engine.last_rms_db()

    def preview_png(self) -> bytes | None:
        return self._engine.preview_png()

    def engine_state(self) -> str:
        return self._engine.state

    def session_dir(self) -> str:
        ws = self._engine.workspace
        return str(ws.root) if ws else ""

    # ------------------------------------------------------------ 任务
    def solve_task(self, task_id: str) -> None:
        self._engine.solve_task(task_id)

    def solve_all_pending(self) -> None:
        self._engine.solve_all_pending()

    def regenerate_task(self, task_id: str, feedback: str = "") -> None:
        self._engine.update_task(task_id, status="queued", error="")
        self._engine.solve_task(task_id, feedback=feedback)

    def delete_task(self, task_id: str) -> None:
        self._engine.delete_task(task_id)

    def add_manual_task(self, title: str, detail: str = "") -> str:
        task = self._engine.add_manual_task(title, detail)
        return task.id

    def update_task(self, task_id: str, **fields: Any) -> None:
        self._engine.update_task(task_id, **fields)

    # ------------------------------------------------------------ 会话
    def current_bundle(self) -> SessionBundle | None:
        return self._engine.bundle()

    def list_sessions(self, limit: int = 200) -> list[SessionInfo]:
        return self._store.list(limit=limit)

    def load_session(self, session_id: str) -> SessionBundle | None:
        ws = self._store.open(session_id)
        return ws.bundle() if ws else None

    def export_session(self, session_id: str = "") -> list[str]:
        if not session_id or (self._engine.workspace and self._engine.workspace.info.id == session_id):
            return self._engine.export_session()
        ws = self._store.open(session_id)
        if ws is None:
            return []
        from .artifacts import ArtifactWriter

        cfg = self.get_config()
        writer = ArtifactWriter(cfg.output, cfg.person)
        tasks = ws.load_tasks()
        out: list[str] = []
        try:
            ws.build_timeline()
            timeline = ws.timeline_path.read_text(encoding="utf-8")
        except Exception:
            timeline = ""
        out.extend(writer.write_manual_notes(ws.info, ws.read_notes(), timeline))
        art = writer.write_summary(ws.info, tasks, ws.read_notes())
        if art:
            out.append(art.path)
            self.artifactWritten.emit(art.to_dict())
        return out

    def session_notes(self, session_id: str) -> str:
        ws = self._store.open(session_id)
        return ws.read_notes() if ws else ""

    # ------------------------------------------------------------ 设备
    def probe_devices(self) -> dict:
        try:
            from ..capture.devices import describe_environment

            data = describe_environment()
        except Exception as exc:
            log.exception("枚举设备失败")
            self.errorRaised.emit("枚举设备失败", str(exc))
            data = {"audio": [], "monitors": [], "windows": [], "classroom_windows": [], "ffmpeg": ""}
        self.captureDevicesChanged.emit(data)
        return data

    # ------------------------------------------------------------ 文件
    def open_path(self, path: str) -> None:
        p = Path(path)
        if not p.exists():
            self.errorRaised.emit("路径不存在", str(p))
            return
        try:
            if p.is_dir():
                os.startfile(str(p))  # type: ignore[attr-defined]
            else:
                subprocess.Popen(["explorer", "/select,", str(p)])
        except Exception:
            try:
                os.startfile(str(p.parent))  # type: ignore[attr-defined]
            except Exception as exc:
                self.errorRaised.emit("无法打开", str(p) + " | " + str(exc))

    def open_output_dir(self) -> None:
        self.open_path(self.get_config().output.output_dir or str(default_output_dir()))

    # ------------------------------------------------------------ 生命周期
    def shutdown(self) -> None:
        try:
            self._engine.shutdown()
        except Exception:
            log.exception("关闭引擎失败")
