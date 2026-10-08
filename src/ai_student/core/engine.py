"""课堂引擎：把"采集 → 转写 → 画面理解 → 笔记 → 任务 → 作答 → 导出"串起来。

设计要点
- 采集线程只负责入队（非阻塞）；AI 调用全部在网络线程池里跑，绝不阻塞采集或 GUI。
- 所有依赖外部能力的地方（采集器、模型客户端、AI 模块）都可注入替身，便于离线测试。
- 引擎本身不依赖 Qt；事件通过 EngineCallbacks 回调出去，由 controller 转成 Qt 信号。
"""
from __future__ import annotations

import logging
import queue
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from ..config.models import AppConfig
from ..utils import clip
from .artifacts import ArtifactWriter
from .models import Artifact, BoardNote, ClassTask, SessionBundle, SessionInfo, TranscriptSegment
from .session import SessionStore, SessionWorkspace

log = logging.getLogger(__name__)


# ------------------------------------------------------------------ 回调
def _noop(*_args: Any, **_kwargs: Any) -> None:
    return None


@dataclass
class EngineCallbacks:
    on_state: Callable[[str, str], None] = _noop          # (state, detail)
    on_stage: Callable[[str, str], None] = _noop          # (stage, detail)
    on_progress: Callable[[str, int], None] = _noop       # (stage, 0-100)
    on_transcript: Callable[[TranscriptSegment], None] = _noop
    on_board: Callable[[BoardNote], None] = _noop
    on_notes: Callable[[str], None] = _noop
    on_task: Callable[[ClassTask], None] = _noop
    on_artifact: Callable[[Artifact], None] = _noop
    on_log: Callable[[str, str], None] = _noop            # (level, message)
    on_error: Callable[[str, str], None] = _noop          # (title, detail)
    on_session: Callable[[SessionInfo], None] = _noop     # 开始/结束
    on_ai_health: Callable[[str], None] = _noop           # 模型可用性变化描述

    def emit_state(self, state: str, detail: str = "") -> None:
        _safe(self.on_state, state, detail)

    def emit_stage(self, stage: str, detail: str = "") -> None:
        _safe(self.on_stage, stage, detail)

    def emit_progress(self, stage: str, percent: int) -> None:
        _safe(self.on_progress, stage, int(percent))

    def emit_log(self, level: str, message: str) -> None:
        _safe(self.on_log, level, message)

    def emit_error(self, title: str, detail: str) -> None:
        _safe(self.on_error, title, detail)


def _safe(fn: Callable, *args: Any) -> None:
    try:
        fn(*args)
    except Exception:  # pragma: no cover - 回调异常不能影响引擎
        log.exception("引擎回调异常")


class LazyAiModules:
    """按需导入 ai_student.ai.* 子模块（缺失时抛 ImportError 由调用方兜底）。"""

    def __init__(self) -> None:
        self._cache: dict[str, Any] = {}

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            raise AttributeError(name)
        mod = self._cache.get(name)
        if mod is None:
            import importlib

            mod = importlib.import_module(f"ai_student.ai.{name}")
            self._cache[name] = mod
        return mod


def _default_audio_factory(cfg, session_dir, on_chunk, on_error):
    from ..capture.audio import AudioRecorder

    return AudioRecorder(cfg, session_dir, on_chunk=on_chunk, on_error=on_error)


def _default_screen_factory(cfg, session_dir, on_keyframe, on_error):
    from ..capture.screen import ScreenRecorder

    return ScreenRecorder(cfg, session_dir, on_keyframe=on_keyframe, on_error=on_error)


def _default_client_factory(endpoint):
    from ..ai.client import LLMClient

    return LLMClient(endpoint)


def _default_supervisor_factory(engine: Any, cfg: AppConfig) -> Any:
    """默认的主 Agent（课堂督导）构造方式。"""
    from ..ai.agent import TaskSupervisor

    return TaskSupervisor(
        engine._get_client(), cfg.model, cfg.person, engine,
        overrides=cfg.prompt_overrides,
        max_rounds=cfg.behavior.agent_max_tool_rounds,
        min_relevance=cfg.behavior.agent_min_relevance,
        tools_enabled=cfg.behavior.agent_tools_enabled,
        allow_revise=cfg.behavior.auto_revise_solved,
    )


# ------------------------------------------------------------------ 引擎
class ClassEngine:
    """单次听课会话的执行引擎（一个实例可反复 start/stop）。"""

    def __init__(
        self,
        get_config: Callable[[], AppConfig],
        callbacks: EngineCallbacks | None = None,
        *,
        store: SessionStore | None = None,
        ai: Any | None = None,
        audio_factory: Callable | None = None,
        screen_factory: Callable | None = None,
        client_factory: Callable | None = None,
        supervisor_factory: Callable[[Any, AppConfig], Any] | None = None,
    ) -> None:
        self.get_config = get_config
        self.cb = callbacks or EngineCallbacks()
        self.store = store or SessionStore()
        self.ai = ai or LazyAiModules()
        self.audio_factory = audio_factory or _default_audio_factory
        self.screen_factory = screen_factory or _default_screen_factory
        self.client_factory = client_factory or _default_client_factory
        self.supervisor_factory = supervisor_factory or _default_supervisor_factory

        self._lock = threading.RLock()
        self._workspace: SessionWorkspace | None = None
        self._audio: Any = None
        self._screen: Any = None
        self._executor: ThreadPoolExecutor | None = None
        self._audio_q: "queue.Queue[Any]" = queue.Queue()
        self._frame_q: "queue.Queue[Any]" = queue.Queue()
        self._workers: list[threading.Thread] = []
        self._stop_evt = threading.Event()
        self._draining = False
        self._finalizer: threading.Thread | None = None

        self._tasks: dict[str, ClassTask] = {}
        self._notes: str = ""
        self._client: Any = None
        self._client_key: tuple = ()
        self._unmerged: list[str] = []
        self._unmerged_board: list[str] = []
        self._notes_inflight = False
        self._solving: set[str] = set()
        # 主 Agent（课堂督导）：把新听到的信息转成对子 Agent 的任务简报更新
        self._supervisor: Any = None
        self._supervisor_key: tuple = ()
        self._pending_review: list[str] = []
        self._supervise_busy = False
        self._questions: list[str] = []
        self._ai_disabled_reason = ""
        self._chunk_index = 0
        self._last_voice_ts = time.time()
        self._started_at = 0.0
        self._state = "idle"

    # ------------------------------------------------------------ 只读状态
    @property
    def workspace(self) -> SessionWorkspace | None:
        return self._workspace

    @property
    def state(self) -> str:
        return self._state

    def is_recording(self) -> bool:
        return self._state in ("recording", "paused")

    def notes_markdown(self) -> str:
        return self._notes

    def tasks(self) -> list[ClassTask]:
        with self._lock:
            return list(self._tasks.values())

    def bundle(self) -> SessionBundle | None:
        ws = self._workspace
        if ws is None:
            return None
        b = ws.bundle()
        b.tasks = self.tasks() or b.tasks
        b.notes_markdown = self._notes or b.notes_markdown
        return b

    def last_rms_db(self) -> float:
        try:
            return float(self._audio.last_rms_db) if self._audio else -96.0
        except Exception:
            return -96.0

    def preview_png(self) -> bytes | None:
        try:
            return self._screen.preview_png() if self._screen else None
        except Exception:
            return None

    def elapsed(self) -> float:
        return max(0.0, time.time() - self._started_at) if self._started_at else 0.0

    def ai_ready(self) -> tuple[bool, str]:
        """模型是否可用；不可用时给出中文原因。"""
        cfg = self.get_config()
        ep = cfg.model
        if not ep.base_url or not ep.api_key or not ep.chat_model:
            return False, "未配置多模态模型 API（设置 → 模型 API）"
        if self._ai_disabled_reason:
            return False, self._ai_disabled_reason
        return True, ""

    # ------------------------------------------------------------ 启动
    def start(self, subject: str = "", title: str = "") -> SessionInfo:
        with self._lock:
            if self.is_recording():
                raise RuntimeError("已经在听课中")
            cfg = self.get_config()
            problems = cfg.validate()
            if problems:
                raise ValueError("；".join(problems))

            ws = self.store.create(subject=subject or cfg.person.subject, title=title)
            ws.info.output_dir = str(ArtifactWriter(cfg.output, cfg.person).session_dir(ws.info))
            ws.save_info()
            self._workspace = ws
            self._stop_evt.clear()
            self._draining = False
            self._tasks = {t.id: t for t in ws.load_tasks()}
            self._notes = ws.read_notes()
            self._unmerged = []
            self._unmerged_board = []
            self._solving.clear()
            self._pending_review = []
            self._supervise_busy = False
            self._questions = []
            self._chunk_index = ws.info.transcript_count
            self._last_voice_ts = time.time()
            self._started_at = time.time()
            self._state = "recording"
            self._executor = ThreadPoolExecutor(
                max_workers=max(1, min(6, cfg.behavior.max_concurrent_ai)), thread_name_prefix="ai"
            )
            self._client = None
            self._client_key = ()

            self.cb.emit_state("recording", "开始听课")
            self.cb.emit_log("INFO", f"会话已创建：{ws.root}")

            # 先起消费线程，再在后台线程里启动采集：
            # 打开被占用/无响应的音频设备可能要几秒，绝不能卡住 GUI。
            self._start_workers()
            threading.Thread(
                target=self._start_capture, args=(cfg, ws), name="eng-start-capture", daemon=True
            ).start()
            return ws.info

    def _start_capture(self, cfg: AppConfig, ws: SessionWorkspace) -> None:
        if cfg.capture.mic_enabled or cfg.capture.loopback_enabled:
            try:
                self._audio = self.audio_factory(
                    cfg.capture, ws.root, self._on_audio_chunk, self._on_capture_error
                )
                self._audio.start()
                self.cb.emit_log("INFO", "音频采集已启动（麦克风 / 系统回环）")
            except Exception as exc:
                self._audio = None
                self.cb.emit_log("ERROR", f"音频采集启动失败：{exc}")
                self.cb.emit_error("音频采集失败", str(exc))
        if cfg.capture.screen_enabled:
            try:
                self._screen = self.screen_factory(
                    cfg.capture, ws.root, self._on_keyframe, self._on_capture_error
                )
                self._screen.start()
                self.cb.emit_log("INFO", "演示画面采集已启动")
            except Exception as exc:
                self._screen = None
                self.cb.emit_log("ERROR", f"画面采集启动失败：{exc}")
                self.cb.emit_error("画面采集失败", str(exc))

    def _start_workers(self) -> None:
        self._workers = [
            threading.Thread(target=self._audio_worker, name="eng-audio", daemon=True),
            threading.Thread(target=self._vision_worker, name="eng-vision", daemon=True),
            threading.Thread(target=self._watchdog_worker, name="eng-watchdog", daemon=True),
        ]
        for t in self._workers:
            t.start()

    # ------------------------------------------------------------ 采集回调（采集线程内，必须极快）
    def _on_audio_chunk(self, chunk: Any) -> None:
        if not self._stop_evt.is_set() or self._draining:
            try:
                self._audio_q.put_nowait(chunk)
            except queue.Full:
                pass

    def _on_keyframe(self, frame: Any) -> None:
        if not self._stop_evt.is_set() or self._draining:
            try:
                self._frame_q.put_nowait(frame)
            except queue.Full:
                pass

    def _on_capture_error(self, message: str) -> None:
        self.cb.emit_log("WARNING", f"采集异常：{message}")

    # ------------------------------------------------------------ 模型客户端
    def _get_client(self) -> Any:
        cfg = self.get_config()
        ep = cfg.model
        key = (ep.base_url, ep.api_key, ep.chat_model, ep.vision_model, ep.asr_model, ep.asr_kind)
        if self._client is None or key != self._client_key:
            self._client = self.client_factory(ep)
            self._client_key = key
        return self._client

    def _ai_guard(self, action: str) -> bool:
        """模型不可用时跳过 AI 步骤（只记录原始素材）。"""
        ok, reason = self.ai_ready()
        if not ok:
            if self._ai_disabled_reason != reason:
                self._ai_disabled_reason = reason
                self.cb.emit_log("WARNING", f"跳过{action}：{reason}")
                self.cb.emit_ai_health(reason)
            return False
        return True

    # ------------------------------------------------------------ 音频工作线程
    def _audio_worker(self) -> None:
        while True:
            try:
                chunk = self._audio_q.get(timeout=0.3)
            except queue.Empty:
                if self._stop_evt.is_set() and not self._draining:
                    break
                if self._draining and self._audio_q.empty():
                    break
                continue
            try:
                self._process_chunk(chunk)
            except Exception as exc:  # pragma: no cover
                log.exception("处理音频切片失败")
                self.cb.emit_log("ERROR", f"处理音频切片失败：{exc}")
            finally:
                self._audio_q.task_done()

    def _process_chunk(self, chunk: Any) -> None:
        ws = self._workspace
        if ws is None:
            return
        silent = bool(getattr(chunk, "silent", False))
        if not silent:
            self._last_voice_ts = time.time()
        text = ""
        if not silent and self._ai_guard("语音识别"):
            cfg = self.get_config()
            self.cb.emit_stage("transcribe", f"正在识别第 {chunk.index} 段语音")
            try:
                text = (self.ai.asr.transcribe_chunk(self._get_client(), Path(chunk.path), cfg.model) or "").strip()
            except Exception as exc:
                self.cb.emit_log("WARNING", f"第 {chunk.index} 段语音识别失败：{exc}")
                text = ""
        seg = TranscriptSegment(
            index=self._chunk_index,
            t_start=float(getattr(chunk, "t_start", 0.0)),
            t_end=float(getattr(chunk, "t_end", 0.0)),
            text=text,
            speaker="老师",
            source=str(getattr(chunk, "source", "mix")),
            audio_path=ws.rel(getattr(chunk, "path", "")),
            confidence=0.0 if not text else 0.9,
        )
        self._chunk_index += 1
        ws.append_transcript(seg)
        _safe(self.cb.on_transcript, seg)
        if text:
            self._unmerged.append(f"[{seg.timecode}] {text}")
        self._maybe_merge_notes()

    # ------------------------------------------------------------ 画面工作线程
    def _vision_worker(self) -> None:
        while True:
            try:
                frame = self._frame_q.get(timeout=0.3)
            except queue.Empty:
                if self._stop_evt.is_set() and not self._draining:
                    break
                if self._draining and self._frame_q.empty():
                    break
                continue
            try:
                self._process_frame(frame)
            except Exception as exc:  # pragma: no cover
                log.exception("处理画面关键帧失败")
                self.cb.emit_log("ERROR", f"处理画面失败：{exc}")
            finally:
                self._frame_q.task_done()

    def _process_frame(self, frame: Any) -> None:
        ws = self._workspace
        cfg = self.get_config()
        if ws is None:
            return
        ocr = summary = ""
        problems: list[str] = []
        if cfg.behavior.vision_analysis and self._ai_guard("画面理解"):
            self.cb.emit_stage("vision", "正在理解演示画面")
            try:
                analysis = self.ai.vision.analyze_board(
                    self._get_client(), Path(frame.path), cfg.model, context=cfg.person.subject
                )
                ocr = getattr(analysis, "ocr_text", "") or ""
                summary = getattr(analysis, "summary", "") or ""
                problems = list(getattr(analysis, "problems", []) or [])
            except Exception as exc:
                self.cb.emit_log("WARNING", f"画面理解失败：{exc}")
        note = BoardNote(
            index=ws.info.board_count,
            t=float(getattr(frame, "t", 0.0)),
            image_path=ws.rel(getattr(frame, "path", "")),
            ocr_text=ocr,
            summary=summary,
            changed=bool(getattr(frame, "changed", True)),
        )
        ws.append_board(note)
        _safe(self.cb.on_board, note)
        if summary or ocr:
            bits = [x for x in (summary, ocr[:400]) if x]
            self._unmerged_board.append(f"[{note.timecode}] " + " | ".join(bits))
        if problems:
            self._unmerged_board.extend(f"[{note.timecode}] 画面题目：{p}" for p in problems)
        self._maybe_merge_notes()

    # ------------------------------------------------------------ 笔记 / 任务
    def _maybe_merge_notes(self, force: bool = False) -> None:
        cfg = self.get_config()
        with self._lock:
            if self._notes_inflight:
                return
            need = max(1, cfg.behavior.notes_every_n_chunks)
            n_text = len(self._unmerged)
            n_board = len(self._unmerged_board)
            if not force:
                # 触发条件里必须包含"至少听到一句老师原话"。
                # 否则屏幕上刚出现两帧画面就会触发整理+任务识别：
                # 老师还没开口，AI 就已经拿着几乎为空的上下文"抢答"并产出错误作业。
                text_ready = n_text >= need
                board_ready = n_text >= 1 and n_board >= 2
                if not (text_ready or board_ready):
                    return
            if not self._unmerged and not self._unmerged_board and self._notes:
                return
            self._notes_inflight = True
        self._submit(self._merge_notes_job, force)

    def _merge_notes_job(self, force: bool = False) -> None:
        try:
            cfg = self.get_config()
            with self._lock:
                lines = list(self._unmerged)
                board_lines = list(self._unmerged_board)
            if self._ai_guard("课堂笔记整理"):
                self.cb.emit_stage("notes", "正在整理课堂笔记")
                try:
                    notes = self.ai.notes.merge_notes(
                        self._get_client(), self._notes, lines, board_lines,
                        cfg.model, cfg.person, subject=cfg.person.subject,
                    )
                    if notes and notes.strip():
                        self._notes = notes.strip()
                        ws = self._workspace
                        if ws:
                            ws.write_notes(self._notes)
                        _safe(self.cb.on_notes, self._notes)
                        if lines:
                            with self._lock:
                                del self._unmerged[: len(lines)]
                        if board_lines:
                            with self._lock:
                                del self._unmerged_board[: len(board_lines)]
                except Exception as exc:
                    self.cb.emit_log("WARNING", f"课堂笔记整理失败：{exc}")
                    return
            # 本轮真正"新听到/新看到"的内容 -> 交给主 Agent 判断是否影响已知任务
            consumed = chr(10).join(lines + board_lines)
            self._extract_tasks_job(consumed)
        finally:
            with self._lock:
                self._notes_inflight = False

    def _extract_tasks_job(self, consumed_text: str = "") -> None:
        if not self._ai_guard("任务识别"):
            return
        cfg = self.get_config()
        ws = self._workspace
        if ws is None:
            return
        recent = "\n".join(self._unmerged[-20:])
        # 没有任何课堂内容时不要问模型"老师布置了什么"，否则既浪费调用又容易凭空冒任务
        if not (self._notes.strip() or recent.strip()):
            return
        self.cb.emit_stage("extract", "正在识别老师布置的任务")
        try:
            found = self.ai.tasks.extract_tasks(
                self._get_client(), cfg.model, cfg.person, self._notes,
                recent_transcript=recent, known_tasks=self.tasks(), subject=cfg.person.subject,
            )
        except Exception as exc:
            self.cb.emit_log("WARNING", f"任务识别失败：{exc}")
            return
        added = False
        for task in found or []:
            task.id = task.id or ClassTask().id
            with self._lock:
                if task.id in self._tasks:
                    continue
                if task.confidence and task.confidence < cfg.behavior.min_task_confidence:
                    task.status = "skipped"
                self._tasks[task.id] = task
                added = True
            _safe(self.cb.on_task, task)
            self.cb.emit_log("INFO", f"发现任务：{task.title}")
        if added:
            ws.save_tasks(self.tasks())
            ws.save_info()
            if cfg.behavior.auto_solve_tasks:
                self.solve_all_pending()
        # 无论有没有发现新任务，都要让主 Agent 看一眼：老师可能在补充已有任务的要求
        self._supervise(consumed_text)

    # ------------------------------------------------------------ 任务作答
    def _build_context(self, task: ClassTask, limit: int = 6000) -> str:
        ws = self._workspace
        chunks: list[str] = []
        if (task.brief or "").strip():
            chunks.append(
                "## 任务简报（主讲 Agent 维护，优先级最高，必须满足）\n" + task.brief.strip()
            )
        if task.requirements:
            chunks.append("## 老师明确提出的要求\n- " + "\n- ".join(task.requirements))
        if self._notes:
            chunks.append("## 课堂笔记\n" + self._notes)
        if ws is not None:
            keywords = [w for w in _keywords(task.title + " " + task.detail) if len(w) >= 2]
            hits: list[str] = []
            for seg in ws.read_transcript():
                if not seg.text:
                    continue
                if any(k in seg.text for k in keywords) or seg.index in task.source_segments:
                    hits.append(f"[{seg.timecode}] {seg.text}")
                if len(hits) >= 40:
                    break
            if hits:
                chunks.append("## 与本题相关的课堂原话\n" + "\n".join(hits))
            board = ws.read_board()
            if board:
                board_txt = [
                    f"[{n.timecode}] {n.summary} {n.ocr_text[:300]}".strip()
                    for n in board[-12:]
                    if n.summary or n.ocr_text
                ]
                if board_txt:
                    chunks.append("## 板书 / 演示要点\n" + "\n".join(board_txt))
        text = "\n\n".join(chunks)
        return text[:limit]

    def solve_task(self, task_id: str, feedback: str = "") -> None:
        with self._lock:
            task = self._tasks.get(task_id)
            if task is None or task_id in self._solving:
                return
            self._solving.add(task_id)
        self._submit(self._solve_job, task, feedback)

    def solve_all_pending(self) -> None:
        for task in self.tasks():
            if task.status in ("discovered", "queued", "failed"):
                self.solve_task(task.id)

    def _solve_job(self, task: ClassTask, feedback: str = "") -> None:
        cfg = self.get_config()
        ws = self._workspace
        try:
            if not self._ai_guard("作业作答"):
                task.status = "failed"
                task.error = self._ai_disabled_reason or "模型不可用"
                self._emit_task(task)
                return
            task.status = "working"
            task.attempts += 1
            task.revision += 1
            task.touch()
            self._emit_task(task)
            self.cb.emit_stage("solve", f"正在完成：{task.title}")
            context = self._build_context(task)
            if feedback:
                context += f"\n\n## 学生（用户）的修改要求\n{feedback}"
            result = self.ai.solver.solve_task(
                self._get_client(), cfg.model, cfg.person, task, context, cfg.output
            )
            task.answer = getattr(result, "answer", "") or ""
            task.workpaper = getattr(result, "workpaper", "") or ""
            task.status = "solved" if task.answer.strip() else "failed"
            task.error = "" if task.status == "solved" else "模型没有返回内容"
            task.touch()

            if task.status == "solved" and ws is not None:
                writer = ArtifactWriter(cfg.output, cfg.person)
                index = self.tasks().index(task) + 1
                arts = writer.write_task(
                    ws.info, task,
                    answer=task.answer,
                    workpaper=task.workpaper,
                    file_title=getattr(result, "file_title", "") or task.title,
                    index=index,
                )
                task.artifacts = [a.path for a in arts]
                ws.save_tasks(self.tasks())
                for art in arts:
                    _safe(self.cb.on_artifact, art)
                if arts:
                    self.cb.emit_log("INFO", f"作业已产出：{arts[0].path}")
                self._refresh_output_dir()
            self._emit_task(task)
        except Exception as exc:
            log.exception("完成任务失败")
            task.status = "failed"
            task.error = str(exc)
            task.touch()
            self._emit_task(task)
            self.cb.emit_log("ERROR", f"任务「{task.title}」完成失败：{exc}")
        finally:
            with self._lock:
                self._solving.discard(task.id)

    def _emit_task(self, task: ClassTask) -> None:
        ws = self._workspace
        if ws is not None:
            ws.save_tasks(self.tasks())
        _safe(self.cb.on_task, task)

    # ------------------------------------------------------------ 主 Agent（课堂督导）
    def _get_supervisor(self, cfg: AppConfig) -> Any:
        """惰性构造督导 Agent；相关配置变化时重建；构造失败返回 None 并降级。"""
        key = (
            cfg.model.base_url,
            cfg.model.chat_model,
            int(cfg.behavior.agent_max_tool_rounds or 1),
            round(float(cfg.behavior.agent_min_relevance or 0.0), 4),
            bool(cfg.behavior.auto_revise_solved),
        )
        try:
            if self._supervisor is None or key != self._supervisor_key:
                self._supervisor = self.supervisor_factory(self, cfg)
                self._supervisor_key = key
            self._supervisor.client = self._get_client()
        except Exception as exc:  # pragma: no cover - 模块缺失/构造失败
            log.warning("无法构造主 Agent：%s", exc)
            self._supervisor = None
            return None
        return self._supervisor

    def _supervise(self, consumed_text: str) -> None:
        """把新内容丢给督导队列；同一时刻只跑一个督导任务，避免重复烧钱。"""
        if not (consumed_text or "").strip():
            return
        cfg = self.get_config()
        if not cfg.behavior.agent_tools_enabled:
            return
        with self._lock:
            self._pending_review.append(consumed_text.strip())
            if self._supervise_busy:
                return
            self._supervise_busy = True
        self._submit(self._supervise_job)

    def _supervise_job(self) -> None:
        while True:
            with self._lock:
                if not self._pending_review:
                    self._supervise_busy = False
                    return
                text = chr(10).join(self._pending_review)
                self._pending_review = []
            try:
                self._run_review(text)
            except Exception as exc:  # noqa: BLE001 - 督导失败不能影响听课
                log.warning("主 Agent 督导失败：%s", exc)

    def _run_review(self, text: str) -> None:
        cfg = self.get_config()
        if not cfg.behavior.agent_tools_enabled:
            return
        tasks = [t for t in self.tasks() if not t.closed]
        if not tasks:
            return
        if not self._ai_guard("主 Agent 督导"):
            return
        supervisor = self._get_supervisor(cfg)
        if supervisor is None:
            return
        ws = self._workspace
        self.cb.emit_stage("supervise", "正在判断新内容是否影响已知任务")
        review = supervisor.review(text, tasks, session_context=self._session_brief(cfg))
        for act in review.actions:
            level = "INFO" if act.ok else "WARNING"
            self.cb.emit_log(level, f"主 Agent · {act.name}：{act.result}")
        if review.actions:
            self.cb.emit_stage("supervise", "已把最新要求转给作业子 Agent")
            if ws is not None:
                ws.save_tasks(self.tasks())
        elif review.summary:
            self.cb.emit_log("DEBUG", f"主 Agent：{clip(review.summary, 160)}")
        if review.degraded:
            self.cb.emit_log("WARNING", review.degraded_reason)

    @staticmethod
    def _session_brief(cfg: AppConfig) -> str:
        person = cfg.person
        bits: list[str] = []
        if person.subject:
            bits.append("科目：" + person.subject)
        klass = (person.grade or "") + (person.class_name or "")
        if klass:
            bits.append("班级：" + klass)
        if person.teacher_name:
            bits.append("任课教师：" + person.teacher_name)
        return "；".join(bits) if bits else "(无)"

    # -------- 以下方法是主 Agent 的工具实现（由 ai.tools.dispatch_tool 调用） --------
    def agent_update_task_brief(self, task_id: str, requirements: list[str] | None = None,
                                deadline: str = "", detail_append: str = "",
                                subject: str = "") -> str:
        with self._lock:
            task = self._tasks.get(task_id)
            if task is None:
                return f"找不到任务 {task_id}，请从任务清单里原样复制 id"
            added: list[str] = []
            for req in requirements or []:
                req = str(req).strip()
                if req and req not in task.requirements:
                    task.requirements.append(req)
                    added.append(req)
            if deadline and deadline != task.deadline:
                task.deadline = deadline
                added.append("截止时间改为 " + deadline)
            if subject and not task.subject:
                task.subject = subject
            extra = (detail_append or "").strip()
            if extra:
                task.brief = (task.brief + chr(10) + extra).strip() if task.brief else extra
                added.append("补充了 %d 字说明" % len(extra))
            task.touch()
        if not added:
            return "没有新增内容，「" + task.title + "」的简报未变化"
        self._emit_task(task)
        self.cb.emit_log("INFO", "主 Agent 更新任务简报：" + task.title + " → " + "；".join(added))
        return "已更新「" + task.title + "」的任务简报：" + "；".join(added)

    def agent_redispatch_task(self, task_id: str, instruction: str = "", reason: str = "") -> str:
        cfg = self.get_config()
        with self._lock:
            task = self._tasks.get(task_id)
            if task is None:
                return f"找不到任务 {task_id}"
            if task.closed:
                return "「" + task.title + "」已关闭，未重做"
            manual = bool((instruction or "").strip())
            if task.status == "solved" and not manual and not cfg.behavior.auto_revise_solved:
                return "当前设置为不自动重做，已跳过（需要时可在任务页手动点「重做」）"
            limit = max(0, int(cfg.behavior.max_auto_revisions or 0))
            if not manual and task.auto_revised >= limit:
                return ("「" + task.title + "」已自动重做 " + str(task.auto_revised)
                        + " 次，达到上限 " + str(limit) + "，未再重做")
            if manual:
                pass
            else:
                task.auto_revised += 1
            if manual:
                task.brief = (task.brief + chr(10) + instruction.strip()).strip() if task.brief \
                    else instruction.strip()
            task.status = "queued"
            task.error = ""
            task.touch()
        self._emit_task(task)
        self.cb.emit_log("INFO", "主 Agent 派活给子 Agent 重做：" + task.title
                         + "（" + (reason or "要求有更新") + "）")
        self.solve_task(task_id)
        return "已让子 Agent 用最新简报重新作答：「" + task.title + "」"

    def agent_create_task(self, title: str, detail: str = "", task_type: str = "作业",
                          requirements: list[str] | None = None,
                          confidence: float = 0.8) -> str:
        title = (title or "").strip()
        if not title:
            return "标题为空，未创建任务"
        try:
            from ..ai.tasks import title_similarity
        except Exception:  # pragma: no cover
            title_similarity = None  # type: ignore[assignment]
        with self._lock:
            existing = list(self._tasks.values())
        if title_similarity is not None:
            for task in existing:
                try:
                    if title_similarity(task.title, title) > 0.75:
                        return "已存在同类任务「" + task.title + "」，未重复创建"
                except Exception:
                    continue
        task = self.add_manual_task(title, detail, task_type or "作业")
        if requirements:
            with self._lock:
                for req in requirements:
                    req = str(req).strip()
                    if req and req not in task.requirements:
                        task.requirements.append(req)
                task.confidence = float(confidence or 0.8)
                task.touch()
            self._emit_task(task)
        cfg = self.get_config()
        if cfg.behavior.auto_solve_tasks:
            self.solve_task(task.id)
        return "已新建任务「" + task.title + "」" + ("并交给子 Agent 作答" if cfg.behavior.auto_solve_tasks else "")

    def agent_close_task(self, task_id: str, reason: str = "") -> str:
        with self._lock:
            task = self._tasks.get(task_id)
            if task is None:
                return f"找不到任务 {task_id}"
            task.closed = True
            task.status = "skipped"
            if reason:
                task.brief = (task.brief + chr(10) + "【已关闭】" + reason).strip() if task.brief else ("【已关闭】" + reason)
            task.touch()
        self._emit_task(task)
        self.cb.emit_log("INFO", "主 Agent 关闭任务：" + task.title + ("（" + reason + "）" if reason else ""))
        return "已关闭任务「" + task.title + "」"

    def agent_raise_question(self, task_id: str, question: str) -> str:
        text = (question or "").strip()
        if not text:
            return "问题为空，未记录"
        with self._lock:
            self._questions.append(text)
            task = self._tasks.get(task_id) if task_id else None
            if task is not None:
                if text not in task.questions:
                    task.questions.append(text)
                task.touch()
        if task is not None:
            self._emit_task(task)
        self.cb.emit_log("WARNING", "主 Agent 存疑（待人工确认）：" + text)
        return "已记录存疑：" + clip(text, 120)

    # ------------------------------------------------------------ 任务管理
    def add_manual_task(self, title: str, detail: str = "", task_type: str = "作业") -> ClassTask:
        task = ClassTask(title=title.strip() or "未命名任务", detail=detail, task_type=task_type, confidence=1.0)
        cfg = self.get_config()
        task.subject = cfg.person.subject
        with self._lock:
            self._tasks[task.id] = task
        self._emit_task(task)
        self.cb.emit_log("INFO", f"手动添加任务：{task.title}")
        return task

    def update_task(self, task_id: str, **fields: Any) -> None:
        with self._lock:
            task = self._tasks.get(task_id)
            if task is None:
                return
            for key, val in fields.items():
                if hasattr(task, key):
                    setattr(task, key, val)
            task.touch()
        self._emit_task(task)

    def delete_task(self, task_id: str) -> None:
        with self._lock:
            self._tasks.pop(task_id, None)
        ws = self._workspace
        if ws is not None:
            ws.save_tasks(self.tasks())
            ws.save_info()

    # ------------------------------------------------------------ 导出
    def _refresh_output_dir(self) -> None:
        ws = self._workspace
        if ws is None:
            return
        cfg = self.get_config()
        writer = ArtifactWriter(cfg.output, cfg.person)
        ws.info.output_dir = str(writer.session_dir(ws.info))
        ws.save_info()

    def export_session(self, summary: bool = True) -> list[str]:
        """把笔记 / 时间轴 / 汇总写入用户输出目录。返回产出路径。"""
        ws = self._workspace
        if ws is None:
            return []
        cfg = self.get_config()
        writer = ArtifactWriter(cfg.output, cfg.person)
        out: list[str] = []
        try:
            ws.build_timeline()
        except Exception as exc:
            log.warning("生成时间轴失败: %s", exc)
        if summary:
            out.extend(writer.write_manual_notes(ws.info, self._notes or ws.read_notes(),
                                                 ws.timeline_path.read_text(encoding="utf-8")
                                                 if ws.timeline_path.exists() else ""))
            art = writer.write_summary(ws.info, self.tasks(), self._notes or ws.read_notes())
            if art:
                out.append(art.path)
                _safe(self.cb.on_artifact, art)
        self._refresh_output_dir()
        return out

    # ------------------------------------------------------------ 暂停 / 分析 / 停止
    def set_paused(self, paused: bool) -> None:
        for dev in (self._audio, self._screen):
            if dev is None:
                continue
            try:
                dev.pause() if paused else dev.resume()
            except Exception as exc:
                log.warning("暂停/恢复采集失败: %s", exc)
        self._state = "paused" if paused else "recording"
        self.cb.emit_state(self._state, "已暂停" if paused else "已恢复")
        ws = self._workspace
        if ws is not None:
            ws.info.state = self._state  # type: ignore[assignment]
            ws.save_info()
        if not paused:
            self._last_voice_ts = time.time()

    def analyze_pending(self) -> None:
        """立刻把队列里剩余素材送模型（不停止采集）。"""
        self._submit(self._drain_queues_job)
        self._maybe_merge_notes(force=True)

    def _drain_queues_job(self) -> None:
        while not self._audio_q.empty():
            try:
                self._process_chunk(self._audio_q.get_nowait())
            except queue.Empty:
                break
            except Exception as exc:
                log.warning("drain audio: %s", exc)
        while not self._frame_q.empty():
            try:
                self._process_frame(self._frame_q.get_nowait())
            except queue.Empty:
                break
            except Exception as exc:
                log.warning("drain frame: %s", exc)

    def _watchdog_worker(self) -> None:
        """时长上限 + 下课静音自动停止。"""
        while not self._stop_evt.wait(2.0):
            cfg = self.get_config()
            if self._state != "recording":
                continue
            if cfg.capture.max_session_minutes and self.elapsed() > cfg.capture.max_session_minutes * 60:
                self.cb.emit_log("WARNING", "已达单次听课时长上限，自动停止")
                self.stop(finalize=True)
                return
            if cfg.behavior.auto_stop_on_class_end and self._audio is not None:
                try:
                    silence = float(self._audio.silence_seconds())
                except Exception:
                    silence = 0.0
                if silence >= cfg.behavior.end_of_class_silence_s:
                    self.cb.emit_log("INFO", "长时间静音，判定下课，自动收尾")
                    self.stop(finalize=True)
                    return

    def stop(self, finalize: bool = True, wait: bool = False) -> None:
        if self._state in ("idle", "finished", "finalizing"):
            return
        self._state = "finalizing"
        self.cb.emit_state("finalizing", "正在收尾")
        self._stop_evt.set()
        self._draining = True
        self._finalizer = threading.Thread(
            target=self._finalize, args=(finalize,), name="eng-finalize", daemon=True
        )
        self._finalizer.start()
        if wait:
            self._finalizer.join(timeout=900)

    def _finalize(self, finalize: bool) -> None:
        ws = self._workspace
        try:
            self.cb.emit_stage("capture", "正在停止采集")
            for dev in (self._audio, self._screen):
                if dev is None:
                    continue
                try:
                    dev.stop()
                except Exception as exc:
                    log.warning("停止采集失败: %s", exc)
            self._audio = None
            self._screen = None
            self._draining = False  # 让 worker 处理完剩余队列后退出
            for t in self._workers:
                t.join(timeout=120)
            self._workers = []

            if finalize and ws is not None:
                self.cb.emit_stage("notes", "正在整理完整课堂笔记")
                try:
                    self._merge_notes_job(force=True)
                except Exception as exc:
                    log.warning("收尾整理笔记失败: %s", exc)
                if self._questions:
                    pending = [q for q in self._questions if q.strip()]
                    if pending:
                        self._notes = (self._notes.rstrip() + chr(10) + chr(10)
                                       + "## 存疑（AI 提出，待人工确认）" + chr(10)
                                       + chr(10).join("- " + q for q in pending))
                        if ws is not None:
                            ws.write_notes(self._notes)
                cfg = self.get_config()
                if cfg.behavior.solve_on_session_end:
                    pending = [t for t in self.tasks() if t.status in ("discovered", "queued", "failed")]
                    if pending:
                        self.cb.emit_stage("solve", f"正在完成 {len(pending)} 项作业")
                        futures = [self._submit(self._solve_job, t, "") for t in pending]
                        for f in futures:
                            try:
                                f.result(timeout=900)
                            except Exception as exc:
                                log.warning("作业任务异常: %s", exc)
                self.cb.emit_stage("export", "正在导出作业")
                try:
                    self.export_session()
                except Exception as exc:
                    log.exception("导出失败")
                    self.cb.emit_error("导出作业失败", str(exc))
                try:
                    ws.build_timeline()
                except Exception:
                    pass

            if ws is not None:
                ws.info.ended_at = __import__("datetime").datetime.now().isoformat(timespec="seconds")
                ws.info.state = "finished"
                ws.save_info()
                _safe(self.cb.on_session, ws.info)
            self._state = "finished"
            self.cb.emit_progress("export", 100)
            self.cb.emit_state("finished", "本次听课已结束")
            if ws is not None:
                self.cb.emit_log("INFO", f"会话结束，素材目录：{ws.root}")
        except Exception as exc:  # pragma: no cover
            log.exception("收尾失败")
            self._state = "error"
            self.cb.emit_state("error", str(exc))
            self.cb.emit_error("收尾失败", str(exc))
        finally:
            ex = self._executor
            self._executor = None
            if ex is not None:
                ex.shutdown(wait=False)

    # ------------------------------------------------------------ 线程池
    def _submit(self, fn: Callable, *args: Any, **kwargs: Any) -> Future:
        """提交到线程池；池已关闭时退化为"当前线程内联执行"，绝不抛异常。"""
        with self._lock:
            ex = self._executor
        if ex is not None:
            try:
                return ex.submit(fn, *args, **kwargs)
            except RuntimeError:
                log.debug("线程池已关闭，改为内联执行 %s", getattr(fn, "__name__", fn))
        fut: Future = Future()
        try:
            fut.set_result(fn(*args, **kwargs))
        except Exception as exc:  # noqa: BLE001
            fut.set_exception(exc)
        return fut

    def shutdown(self) -> None:
        try:
            if self.is_recording():
                self.stop(finalize=True, wait=True)
        except Exception:
            pass
        ex = self._executor
        if ex is not None:
            ex.shutdown(wait=False)
            self._executor = None


def _keywords(text: str, limit: int = 12) -> list[str]:
    """极简中文关键词抽取：去掉标点后取 2-4 字滑窗的高频片段。"""
    import re

    cleaned = re.sub(r"[^\u4e00-\u9fffA-Za-z0-9]+", " ", text or "")
    tokens: list[str] = []
    for word in cleaned.split():
        if len(word) <= 6:
            tokens.append(word)
        else:
            tokens.extend(word[i : i + 4] for i in range(0, len(word) - 3))
    seen: list[str] = []
    for t in tokens:
        if t and t not in seen:
            seen.append(t)
        if len(seen) >= limit:
            break
    return seen
