"""运行时数据模型（课堂记录 / 任务 / 产出 / 会话）。"""
from __future__ import annotations

import dataclasses
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Literal

SessionState = Literal["idle", "recording", "paused", "finalizing", "finished", "error"]

TaskStatus = Literal["discovered", "queued", "working", "solved", "failed", "skipped"]

PipelineStage = Literal[
    "idle", "capture", "transcribe", "vision", "notes", "extract", "solve", "export"
]


def new_id(prefix: str = "") -> str:
    raw = uuid.uuid4().hex[:12]
    return f"{prefix}{raw}" if prefix else raw


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _dc(cls):
    return dataclasses.asdict


@dataclass
class TranscriptSegment:
    """一段语音识别结果（按 chunk_seconds 切分）。"""

    index: int = 0
    t_start: float = 0.0
    t_end: float = 0.0
    text: str = ""
    speaker: str = "老师"
    source: str = "mic"          # mic | loopback
    audio_path: str = ""         # 相对会话目录
    confidence: float = 0.0
    created_at: str = field(default_factory=now_iso)

    @property
    def timecode(self) -> str:
        def fmt(s: float) -> str:
            s = max(0, int(s))
            return f"{s // 3600:02d}:{(s % 3600) // 60:02d}:{s % 60:02d}"
        return fmt(self.t_start)

    def to_dict(self) -> dict[str, Any]:
        d = dataclasses.asdict(self)
        d["timecode"] = self.timecode
        return d


@dataclass
class BoardNote:
    """一帧板书/演示画面及其理解结果。"""

    index: int = 0
    t: float = 0.0
    image_path: str = ""         # 相对会话目录
    ocr_text: str = ""
    summary: str = ""
    changed: bool = True
    created_at: str = field(default_factory=now_iso)

    @property
    def timecode(self) -> str:
        s = max(0, int(self.t))
        return f"{s // 3600:02d}:{(s % 3600) // 60:02d}:{s % 60:02d}"

    def to_dict(self) -> dict[str, Any]:
        d = dataclasses.asdict(self)
        d["timecode"] = self.timecode
        return d


@dataclass
class ClassTask:
    """老师布置的一项任务/作业。"""

    id: str = field(default_factory=lambda: new_id("task_"))
    title: str = ""
    detail: str = ""
    subject: str = ""
    task_type: str = "作业"       # 作业 | 练习 | 背诵 | 实验 | 讨论 | 阅读 | 其他
    requirements: list[str] = field(default_factory=list)
    deadline: str = ""
    confidence: float = 0.0
    status: TaskStatus = "discovered"
    answer: str = ""
    workpaper: str = ""           # 解题过程
    artifacts: list[str] = field(default_factory=list)   # 产出的绝对路径
    source_segments: list[int] = field(default_factory=list)
    source_notes: list[int] = field(default_factory=list)
    error: str = ""
    attempts: int = 0
    created_at: str = field(default_factory=now_iso)
    updated_at: str = field(default_factory=now_iso)

    def touch(self) -> None:
        self.updated_at = now_iso()

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ClassTask":
        known = {f.name for f in dataclasses.fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in known})


@dataclass
class Artifact:
    """作业产出文件。"""

    task_id: str = ""
    path: str = ""
    fmt: str = ""                # md | txt | docx
    title: str = ""
    size_bytes: int = 0
    created_at: str = field(default_factory=now_iso)

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


@dataclass
class SessionInfo:
    """一次听课会话的元数据。"""

    id: str = field(default_factory=lambda: new_id("s"))
    subject: str = ""
    title: str = ""
    teacher: str = ""
    started_at: str = field(default_factory=now_iso)
    ended_at: str = ""
    state: SessionState = "idle"
    directory: str = ""
    audio_path: str = ""
    video_path: str = ""
    transcript_count: int = 0
    board_count: int = 0
    task_count: int = 0
    artifact_count: int = 0
    output_dir: str = ""
    error: str = ""

    @property
    def display_name(self) -> str:
        bits = [self.started_at.replace("T", " ")[:16]]
        if self.subject:
            bits.append(self.subject)
        if self.title:
            bits.append(self.title)
        return " · ".join(bits)

    def to_dict(self) -> dict[str, Any]:
        d = dataclasses.asdict(self)
        d["display_name"] = self.display_name
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "SessionInfo":
        known = {f.name for f in dataclasses.fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in known})


@dataclass
class SessionBundle:
    """会话的完整内容（用于历史浏览）。"""

    info: SessionInfo
    transcript: list[TranscriptSegment] = field(default_factory=list)
    board: list[BoardNote] = field(default_factory=list)
    tasks: list[ClassTask] = field(default_factory=list)
    notes_markdown: str = ""
