"""GUI 与核心引擎之间的冻结契约。

UI 只依赖本文件：连接信号、调用方法。核心实现见 core/controller.py。
"""
from __future__ import annotations

from typing import Any

from PySide6.QtCore import QObject, Signal

from .models import SessionBundle, SessionInfo


class ClassAgentAPI(QObject):
    """AI 学生核心引擎的抽象接口（全部信号 + 全部方法）。"""

    # ---------------- 信号 ----------------
    sessionStarted = Signal(object)      # SessionInfo.to_dict()
    sessionStopped = Signal(object)      # SessionInfo.to_dict()
    stateChanged = Signal(str, str)      # (SessionState, 人类可读说明)
    stageChanged = Signal(str, str)      # (PipelineStage, 人类可读说明)

    transcriptSegment = Signal(object)   # TranscriptSegment.to_dict()
    boardNote = Signal(object)           # BoardNote.to_dict()
    notesUpdated = Signal(str)           # 累积的课堂笔记 Markdown

    taskDiscovered = Signal(object)      # ClassTask.to_dict()
    taskUpdated = Signal(object)         # ClassTask.to_dict()
    artifactWritten = Signal(object)     # Artifact.to_dict()

    logMessage = Signal(str, str)        # (level, message)  level in DEBUG/INFO/WARNING/ERROR
    errorRaised = Signal(str, str)       # (title, detail)
    progress = Signal(str, int)          # (stage, 0-100)
    configChanged = Signal()
    captureDevicesChanged = Signal(object)  # {"audio": [...], "windows": [...], "monitors": [...]}

    # ---------------- 方法 ----------------
    def get_config(self):
        """返回 AppConfig 的深拷贝。"""
        raise NotImplementedError

    def update_config(self, cfg) -> None:
        """保存配置（含密钥加密落盘），并发出 configChanged。"""
        raise NotImplementedError

    def start_session(self, subject: str = "", title: str = "") -> str:
        """开始听课。返回 session_id；配置不合法时抛出 ValueError。"""
        raise NotImplementedError

    def stop_session(self, finalize: bool = True) -> None:
        """停止听课；finalize=True 时收尾（补做笔记、完成任务、导出作业）。"""
        raise NotImplementedError

    def set_paused(self, paused: bool) -> None:
        """暂停/恢复采集（AI 理解继续处理已采集内容）。"""
        raise NotImplementedError

    def is_recording(self) -> bool:
        raise NotImplementedError

    def analyze_pending(self) -> None:
        """立即把尚未处理的音频/画面切片送入模型。"""
        raise NotImplementedError

    def solve_task(self, task_id: str) -> None:
        """让 AI 完成指定任务并产出作业文件。"""
        raise NotImplementedError

    def solve_all_pending(self) -> None:
        raise NotImplementedError

    def regenerate_task(self, task_id: str, feedback: str = "") -> None:
        """根据用户反馈重做任务。"""
        raise NotImplementedError

    def delete_task(self, task_id: str) -> None:
        raise NotImplementedError

    def add_manual_task(self, title: str, detail: str = "") -> str:
        """手动补充一条任务（AI 转写漏掉时使用）。"""
        raise NotImplementedError

    def update_task(self, task_id: str, **fields: Any) -> None:
        raise NotImplementedError

    def current_bundle(self) -> SessionBundle | None:
        """当前会话内容快照。"""
        raise NotImplementedError

    def list_sessions(self, limit: int = 200) -> list[SessionInfo]:
        raise NotImplementedError

    def load_session(self, session_id: str) -> SessionBundle | None:
        raise NotImplementedError

    def export_session(self, session_id: str) -> list[str]:
        """把某次会话的全部作业导出到输出目录，返回产出路径列表。"""
        raise NotImplementedError

    def audio_level_db(self) -> float:
        """当前音频电平（dBFS）；未采集时返回 -96.0。UI 可 100ms 轮询。"""
        raise NotImplementedError

    def preview_png(self) -> bytes | None:
        """最近一帧演示画面的 PNG 字节，用于实时预览；不可用时返回 None。"""
        raise NotImplementedError

    def engine_state(self) -> str:
        """引擎当前状态：idle / recording / paused / finalizing / finished / error。"""
        raise NotImplementedError

    def probe_devices(self) -> dict:
        """枚举音频设备 / 可捕获窗口 / 显示器（可能阻塞，建议在工作线程调用）。"""
        raise NotImplementedError

    def open_path(self, path: str) -> None:
        """用资源管理器打开文件或目录。"""
        raise NotImplementedError

    def shutdown(self) -> None:
        """退出前清理（停止采集、落盘）。"""
        raise NotImplementedError
