"""会话目录布局与持久化。

一次听课 = 一个会话目录：

    <sessions_root>/20261008_1330_数学_s3f2a1bc/
        session.json            元数据
        audio/audio.wav         全程录音
        audio/chunks/chunk_0001.wav ...
        audio/transcript.jsonl  逐段转写
        screen/video.mp4        演示画面全程录像
        screen/keyframes/kf_0001.jpg ...
        screen/board.jsonl      逐帧画面理解
        notes/class_notes.md    累积课堂笔记
        notes/timeline.md       时间轴（转写 + 板书穿插）
        tasks/tasks.json        任务清单
        logs/session.log
"""
from __future__ import annotations

import json
import logging
import os
import shutil
import threading
from pathlib import Path
from typing import Iterable

from ..config.paths import sanitize_filename, sessions_root, timestamp_slug
from .models import BoardNote, ClassTask, SessionBundle, SessionInfo, TranscriptSegment

log = logging.getLogger(__name__)


# 同一个文件可能被多个线程同时写：采集线程写转写、视觉线程写板书、
# 线程池里的作答/督导任务又在写 tasks.json 和 session.json。
# Windows 下两个线程同时打开同一个 .tmp 会直接 PermissionError，
# 所以这里给每个路径一把锁，并且每次用**唯一**的临时文件名。
_WRITE_LOCKS: dict[str, threading.Lock] = {}
_WRITE_LOCKS_GUARD = threading.Lock()


def _lock_for(path: Path) -> threading.Lock:
    key = str(path.resolve()) if path.is_absolute() else str(path)
    with _WRITE_LOCKS_GUARD:
        lock = _WRITE_LOCKS.get(key)
        if lock is None:
            lock = threading.Lock()
            _WRITE_LOCKS[key] = lock
        return lock


def _atomic_write_text(path: Path, text: str) -> None:
    """线程安全 + 进程安全的原子写：先写唯一临时文件，再 os.replace。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name("%s.%d.%d.tmp" % (path.name, os.getpid(), threading.get_ident()))
    with _lock_for(path):
        try:
            tmp.write_text(text, encoding="utf-8")
            os.replace(tmp, path)
        finally:
            try:
                if tmp.exists():
                    tmp.unlink()
            except Exception:
                pass


def _write_json_atomic(path: Path, data: object) -> None:
    _atomic_write_text(path, json.dumps(data, ensure_ascii=False, indent=2))


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows: list[dict] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
            if isinstance(obj, dict):
                rows.append(obj)
        except Exception:
            continue  # 容忍半行/损坏行
    return rows


def _append_jsonl(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(data, ensure_ascii=False) + "\n"
    with _lock_for(path):
        with path.open("a", encoding="utf-8") as fh:
            fh.write(line)
            fh.flush()


class SessionWorkspace:
    """一个会话目录的读写门面。"""

    def __init__(self, root: Path, info: SessionInfo) -> None:
        self.root = Path(root)
        self.info = info
        self.info.directory = str(self.root)
        self.ensure_layout()

    # ------------------------------------------------------------ 布局
    @property
    def audio_dir(self) -> Path:
        return self.root / "audio"

    @property
    def chunks_dir(self) -> Path:
        return self.audio_dir / "chunks"

    @property
    def screen_dir(self) -> Path:
        return self.root / "screen"

    @property
    def keyframes_dir(self) -> Path:
        return self.screen_dir / "keyframes"

    @property
    def frames_dir(self) -> Path:
        return self.screen_dir / "frames"

    @property
    def notes_dir(self) -> Path:
        return self.root / "notes"

    @property
    def tasks_dir(self) -> Path:
        return self.root / "tasks"

    @property
    def logs_dir(self) -> Path:
        return self.root / "logs"

    @property
    def transcript_path(self) -> Path:
        return self.audio_dir / "transcript.jsonl"

    @property
    def board_path(self) -> Path:
        return self.screen_dir / "board.jsonl"

    @property
    def notes_path(self) -> Path:
        return self.notes_dir / "class_notes.md"

    @property
    def timeline_path(self) -> Path:
        return self.notes_dir / "timeline.md"

    @property
    def tasks_path(self) -> Path:
        return self.tasks_dir / "tasks.json"

    @property
    def session_json(self) -> Path:
        return self.root / "session.json"

    def ensure_layout(self) -> None:
        for d in (
            self.root, self.audio_dir, self.chunks_dir, self.screen_dir,
            self.keyframes_dir, self.notes_dir, self.tasks_dir, self.logs_dir,
        ):
            d.mkdir(parents=True, exist_ok=True)

    def rel(self, path: Path | str) -> str:
        """转成相对会话目录的路径（存档用）；不在目录内则原样返回。"""
        try:
            return str(Path(path).resolve().relative_to(self.root.resolve()))
        except Exception:
            return str(path)

    def abs(self, rel_path: str) -> Path:
        p = Path(rel_path)
        return p if p.is_absolute() else (self.root / p)

    # ------------------------------------------------------------ 元数据
    def save_info(self) -> None:
        _write_json_atomic(self.session_json, self.info.to_dict())

    @classmethod
    def load(cls, directory: Path) -> "SessionWorkspace | None":
        directory = Path(directory)
        sj = directory / "session.json"
        if not sj.exists():
            return None
        try:
            info = SessionInfo.from_dict(json.loads(sj.read_text(encoding="utf-8")))
        except Exception as exc:
            log.warning("读取 %s 失败: %s", sj, exc)
            return None
        ws = cls(directory, info)
        ws._sync_counts()
        return ws

    def _sync_counts(self) -> None:
        self.info.transcript_count = len(_read_jsonl(self.transcript_path))
        self.info.board_count = len(_read_jsonl(self.board_path))
        self.info.task_count = len(self.load_tasks())
        out = Path(self.info.output_dir) if self.info.output_dir else None
        self.info.artifact_count = sum(1 for _ in out.rglob("*")) if out and out.exists() else 0

    # ------------------------------------------------------------ 转写 / 板书
    def append_transcript(self, seg: TranscriptSegment) -> None:
        _append_jsonl(self.transcript_path, seg.to_dict())
        self.info.transcript_count += 1

    def append_board(self, note: BoardNote) -> None:
        _append_jsonl(self.board_path, note.to_dict())
        self.info.board_count += 1

    def read_transcript(self) -> list[TranscriptSegment]:
        segs: list[TranscriptSegment] = []
        for i, row in enumerate(_read_jsonl(self.transcript_path)):
            row.setdefault("index", i)
            segs.append(_obj(TranscriptSegment, row))
        return segs

    def read_board(self) -> list[BoardNote]:
        return [_obj(BoardNote, row) for row in _read_jsonl(self.board_path)]

    # ------------------------------------------------------------ 笔记
    def write_notes(self, markdown: str) -> None:
        _atomic_write_text(self.notes_path, markdown or "")

    def read_notes(self) -> str:
        return self.notes_path.read_text(encoding="utf-8", errors="replace") if self.notes_path.exists() else ""

    def build_timeline(self) -> str:
        """把转写与板书按时间穿插成 Markdown 时间轴（供人工/AI 查阅）。"""
        rows: list[tuple[float, str]] = []
        for seg in self.read_transcript():
            text = (seg.text or "").strip()
            if text:
                rows.append((seg.t_start, f"- **[{seg.timecode}] 老师**：{text}"))
        for note in self.read_board():
            bits = [b for b in (note.summary, note.ocr_text) if b and b.strip()]
            if bits:
                joined = " / ".join(x.strip().replace("\n", " ") for x in bits)
                rows.append((note.t, f"- **[{note.timecode}] 板书/演示**：{joined}"))
        rows.sort(key=lambda r: r[0])
        md = ["# 课堂时间轴", "", f"- 会话：{self.info.display_name}", ""]
        md.extend(r[1] for r in rows)
        content = "\n".join(md) + "\n"
        _atomic_write_text(self.timeline_path, content)
        return content

    # ------------------------------------------------------------ 任务
    def save_tasks(self, tasks: Iterable[ClassTask]) -> None:
        task_list = list(tasks)
        _write_json_atomic(self.tasks_path, [t.to_dict() for t in task_list])
        self.info.task_count = len(task_list)

    def load_tasks(self) -> list[ClassTask]:
        if not self.tasks_path.exists():
            return []
        try:
            raw = json.loads(self.tasks_path.read_text(encoding="utf-8"))
        except Exception:
            return []
        return [ClassTask.from_dict(r) for r in raw if isinstance(r, dict)]

    # ------------------------------------------------------------ 汇总
    def bundle(self) -> SessionBundle:
        return SessionBundle(
            info=self.info,
            transcript=self.read_transcript(),
            board=self.read_board(),
            tasks=self.load_tasks(),
            notes_markdown=self.read_notes(),
        )

    def log_file(self) -> Path:
        return self.logs_dir / "session.log"

    def delete(self) -> None:
        shutil.rmtree(self.root, ignore_errors=True)


def _obj(cls: type, row: dict):
    """按 dataclass 字段过滤构造对象。"""
    import dataclasses

    known = {f.name for f in dataclasses.fields(cls)}
    return cls(**{k: v for k, v in row.items() if k in known})


class SessionStore:
    """会话集合管理（扫描 sessions 根目录）。"""

    def __init__(self, root: Path | None = None) -> None:
        self.root = Path(root) if root else sessions_root()
        self.root.mkdir(parents=True, exist_ok=True)

    def create(self, subject: str = "", title: str = "", teacher: str = "") -> SessionWorkspace:
        info = SessionInfo(subject=subject, title=title, teacher=teacher)
        label = sanitize_filename(subject or title or "课堂", "课堂", 20)
        folder = self.root / f"{timestamp_slug()}_{label}_{info.id}"
        ws = SessionWorkspace(folder, info)
        ws.save_info()
        return ws

    def open(self, session_id: str) -> SessionWorkspace | None:
        for d in self.root.iterdir():
            if not d.is_dir():
                continue
            if d.name.endswith(session_id) or d.name == session_id:
                ws = SessionWorkspace.load(d)
                if ws is not None:
                    return ws
        return None

    def open_dir(self, directory: Path | str) -> SessionWorkspace | None:
        return SessionWorkspace.load(Path(directory))

    def list(self, limit: int = 200) -> list[SessionInfo]:
        infos: list[SessionInfo] = []
        if not self.root.exists():
            return infos
        for d in sorted(self.root.iterdir(), reverse=True):
            if not d.is_dir():
                continue
            sj = d / "session.json"
            if not sj.exists():
                continue
            try:
                infos.append(SessionInfo.from_dict(json.loads(sj.read_text(encoding="utf-8"))))
            except Exception:
                continue
            if len(infos) >= limit:
                break
        return infos
