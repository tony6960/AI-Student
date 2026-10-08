"""核心引擎 / 配置 / 会话 / 产出 的离线测试。

全部使用注入的假采集器与假模型，不联网、不需要麦克风与教室环境。
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from ai_student.config.models import AppConfig, ModelEndpoint, OutputConfig, PersonalInfo
from ai_student.config.paths import sanitize_filename
from ai_student.config.store import config_from_dict, load_config, save_config
from ai_student.core.artifacts import ArtifactWriter
from ai_student.core.engine import ClassEngine, EngineCallbacks
from ai_student.core.models import BoardNote, ClassTask, SessionInfo, TranscriptSegment
from ai_student.core.session import SessionStore
from ai_student.utils.textutil import clip, hhmmss, parse_json_loose, strip_code_fence

JPEG_HEAD = bytes([255, 216, 255, 224]) + b"fakejpeg"
FENCE = chr(96) * 3


# ------------------------------------------------------------------ 配置
def test_config_roundtrip_encrypts_api_key(tmp_path):
    cfg = AppConfig()
    cfg.model.api_key = "sk-secret-1234567890"
    cfg.person.name = "张三"
    cfg.output.output_dir = str(tmp_path / "out")
    path = tmp_path / "config.json"
    save_config(cfg, path)

    text = path.read_text(encoding="utf-8")
    assert "sk-secret-1234567890" not in text, "API KEY 不能明文落盘"
    assert json.loads(text)["model"]["api_key"]

    back = load_config(path)
    assert back.model.api_key == "sk-secret-1234567890"
    assert back.person.name == "张三"
    assert back.output.output_dir == str(tmp_path / "out")


def test_config_from_dict_tolerates_garbage():
    cfg = config_from_dict({"model": {"temperature": "0.7", "provider": "qwen"},
                            "capture": {"window_title_keywords": "极域"},
                            "behavior": {"max_concurrent_ai": "3"},
                            "unknown_section": 1})
    assert isinstance(cfg.model.temperature, float)
    assert cfg.model.temperature == pytest.approx(0.7)
    assert cfg.capture.window_title_keywords == []
    assert cfg.behavior.max_concurrent_ai == 3


def test_config_validate_reports_missing():
    cfg = AppConfig()
    cfg.output.output_dir = "x"
    problems = cfg.validate()
    assert any("API KEY" in p for p in problems)
    assert any("姓名" in p for p in problems)
    cfg.model.api_key = "k"
    cfg.person.name = "李四"
    assert cfg.validate() == []


def test_model_preset_apply():
    ep = ModelEndpoint()
    ep.apply_preset("openai")
    assert "openai.com" in ep.base_url
    assert ep.chat_model == "gpt-4o"
    assert ep.is_configured() is False


def test_sanitize_filename():
    assert sanitize_filename("a/b:c*d?") == "a_b_c_d_"
    assert sanitize_filename("   ") == "untitled"
    assert sanitize_filename("CON") == "_CON"
    assert len(sanitize_filename("长" * 200)) <= 80


# ------------------------------------------------------------------ 工具函数
def test_parse_json_loose_variants():
    assert parse_json_loose('{"a": 1}') == {"a": 1}
    assert parse_json_loose(FENCE + "json" + chr(10) + "[1, 2]" + chr(10) + FENCE) == [1, 2]
    messy = "好的，结果是：" + chr(10) + '{"a": [1,2,],}' + chr(10) + "希望有帮助"
    assert parse_json_loose(messy) == {"a": [1, 2]}
    with pytest.raises(ValueError):
        parse_json_loose("完全不是 JSON")


def test_strip_code_fence_and_helpers():
    fenced = FENCE + "py" + chr(10) + "print(1)" + chr(10) + FENCE
    assert strip_code_fence(fenced) == "print(1)"
    assert hhmmss(3661) == "01:01:01"
    assert clip("abc", 2).startswith("ab")


# ------------------------------------------------------------------ 会话
def test_session_workspace_roundtrip(tmp_home):
    store = SessionStore()
    ws = store.create(subject="数学", title="一元二次方程")
    assert ws.root.exists() and ws.audio_dir.exists() and ws.keyframes_dir.exists()

    ws.append_transcript(TranscriptSegment(index=0, t_start=0, t_end=5, text="上课"))
    ws.append_board(BoardNote(index=0, t=1.0, image_path="screen/keyframes/kf_0001.jpg", summary="板书"))
    ws.save_tasks([ClassTask(title="课后习题", detail="P45 1-3")])
    ws.write_notes("# 课堂笔记" + chr(10) + "内容")
    ws.info.ended_at = "2026-10-08T14:00:00"
    ws.info.state = "finished"
    ws.save_info()

    loaded = store.open(ws.info.id)
    assert loaded is not None
    bundle = loaded.bundle()
    assert bundle.transcript[0].text == "上课"
    assert bundle.board[0].summary == "板书"
    assert bundle.tasks[0].title == "课后习题"
    assert "课堂笔记" in bundle.notes_markdown
    assert loaded.info.transcript_count == 1
    assert [s.id for s in store.list()] == [ws.info.id]


def test_build_timeline(tmp_home):
    ws = SessionStore().create(subject="语文")
    ws.append_transcript(TranscriptSegment(index=0, t_start=10, t_end=20, text="背诵静夜思"))
    ws.append_board(BoardNote(index=0, t=5, image_path="a.jpg", summary="板书要点"))
    md = ws.build_timeline()
    assert "00:00:05" in md and "00:00:10" in md
    assert md.index("板书要点") < md.index("静夜思")


# ------------------------------------------------------------------ 产出
def test_artifact_writer_all_formats(tmp_path):
    cfg = OutputConfig(output_dir=str(tmp_path), create_session_subfolder=True,
                       write_formats=["md", "txt", "docx"], include_workpaper=True)
    person = PersonalInfo(name="王小明", student_id="20260101", class_name="高一(3)班",
                          school="示例中学", teacher_name="李老师", subject="数学")
    writer = ArtifactWriter(cfg, person)
    info = SessionInfo(subject="数学", title="一元二次方程", started_at="2026-10-08T13:30:00")
    task = ClassTask(title="课本 P45 第 1-3 题", detail="解方程", task_type="作业",
                     requirements=["写出完整过程"])

    answer = "## 解答" + chr(10) + chr(10) + "x=1 或 x=2"
    arts = writer.write_task(info, task, answer=answer, workpaper="配方得 (x-1)^2=0",
                             file_title="P45 1-3题", index=1)
    assert sorted(a.fmt for a in arts) == ["docx", "md", "txt"]
    for a in arts:
        p = Path(a.path)
        assert p.exists() and p.stat().st_size > 0
    md_path = [a.path for a in arts if a.fmt == "md"][0]
    md = Path(md_path).read_text(encoding="utf-8")
    assert "王小明" in md and "解题过程" in md and "课本 P45" in md

    summary = writer.write_summary(info, [task], "# 笔记")
    assert summary is not None and Path(summary.path).exists()
    text = Path(summary.path).read_text(encoding="utf-8")
    assert "作业汇总" in text and "课本 P45" in text


# ------------------------------------------------------------------ 引擎（离线）
class FakeChunk:
    def __init__(self, index, path, t0, t1, silent=False):
        self.index = index
        self.path = path
        self.t_start = t0
        self.t_end = t1
        self.silent = silent
        self.rms_db = -20.0
        self.peak_db = -10.0
        self.source = "mix"


class FakeFrame:
    def __init__(self, index, path, t, changed=True):
        self.index = index
        self.path = path
        self.t = t
        self.changed = changed
        self.hash_distance = 0.5


class FakeAudio:
    def __init__(self, cfg, session_dir, on_chunk=None, on_error=None):
        self.session_dir = Path(session_dir)
        self.on_chunk = on_chunk
        self.is_running = False

    def start(self):
        self.is_running = True
        d = self.session_dir / "audio" / "chunks"
        d.mkdir(parents=True, exist_ok=True)
        for i in range(1, 4):
            p = d / ("chunk_%04d.wav" % i)
            p.write_bytes(b"RIFFfake")
            if self.on_chunk:
                self.on_chunk(FakeChunk(i, p, (i - 1) * 20.0, i * 20.0))

    def stop(self):
        self.is_running = False
        p = self.session_dir / "audio" / "audio.wav"
        p.write_bytes(b"RIFF")
        return p

    def pause(self):
        pass

    def resume(self):
        pass

    @property
    def elapsed(self):
        return 60.0

    @property
    def last_rms_db(self):
        return -33.0

    def silence_seconds(self):
        return 0.0


class FakeScreen:
    def __init__(self, cfg, session_dir, on_keyframe=None, on_error=None):
        self.session_dir = Path(session_dir)
        self.on_keyframe = on_keyframe
        self.is_running = False

    def start(self):
        self.is_running = True
        d = self.session_dir / "screen" / "keyframes"
        d.mkdir(parents=True, exist_ok=True)
        for i in range(1, 3):
            p = d / ("kf_%04d.jpg" % i)
            p.write_bytes(JPEG_HEAD)
            if self.on_keyframe:
                self.on_keyframe(FakeFrame(i, p, i * 8.0))

    def stop(self):
        self.is_running = False
        v = self.session_dir / "screen" / "video.mp4"
        v.write_bytes(b"ftypmp42")
        return v

    def pause(self):
        pass

    def resume(self):
        pass

    @property
    def elapsed(self):
        return 60.0

    def last_frame_path(self):
        return None

    def preview_png(self):
        return None


class FakeAsr:
    @staticmethod
    def transcribe_chunk(client, wav_path, cfg, hint=""):
        return "今天我们学习一元二次方程，课后请完成课本 P45 第 1-3 题。"


class FakeVision:
    @staticmethod
    def analyze_board(client, image, cfg, context=""):
        class A:
            ocr_text = "ax^2+bx+c=0"
            summary = "讲解一元二次方程求根公式"
            key_points = ["判别式"]
            problems = ["课后完成 P45 第 1-3 题"]
            subject_hint = "数学"

        return A()


NOTES_TEMPLATE = """# 课堂笔记

## 知识点
- 一元二次方程

## 老师布置的任务
- 完成课本 P45 第 1-3 题

<!-- {tail} -->"""


class FakeNotes:
    @staticmethod
    def merge_notes(client, previous_markdown, transcript_lines, board_lines, cfg, person, subject=""):
        return NOTES_TEMPLATE.format(tail=chr(10).join(transcript_lines)[:60])


class FakeTasks:
    @staticmethod
    def extract_tasks(client, cfg, person, notes_markdown, recent_transcript="", known_tasks=None, subject=""):
        if known_tasks:
            return []
        return [ClassTask(title="完成课本 P45 第 1-3 题", detail="解一元二次方程",
                          task_type="作业", requirements=["写出完整过程"], confidence=0.95)]


ANSWER = """1. x=1
2. x=2
3. x=3"""


class FakeSolver:
    @staticmethod
    def solve_task(client, cfg, person, task, context, output_cfg):
        class R:
            answer = ANSWER
            workpaper = "因式分解法"
            file_title = "P45第1-3题"
            self_check = "复核无误"
            needs_human = False

        return R()


class FakeAi:
    asr = FakeAsr
    vision = FakeVision
    notes = FakeNotes
    tasks = FakeTasks
    solver = FakeSolver


def _make_cfg(tmp_path: Path) -> AppConfig:
    cfg = AppConfig()
    cfg.model.api_key = "test-key"
    cfg.person.name = "测试学生"
    cfg.person.class_name = "高一(1)班"
    cfg.person.subject = "数学"
    cfg.output.output_dir = str(tmp_path / "作业输出")
    cfg.output.write_formats = ["md", "txt"]
    cfg.behavior.notes_every_n_chunks = 1
    cfg.behavior.auto_solve_tasks = True
    cfg.behavior.solve_on_session_end = True
    return cfg


def _make_engine(cfg, cb=None):
    return ClassEngine(lambda: cfg, cb or EngineCallbacks(), store=SessionStore(), ai=FakeAi,
                       audio_factory=FakeAudio, screen_factory=FakeScreen,
                       client_factory=lambda ep: object())


def test_engine_end_to_end_offline(tmp_home, tmp_path):
    cfg = _make_cfg(tmp_path)
    seen = {"transcript": [], "board": [], "tasks": [], "artifacts": [], "states": [], "notes": []}
    cb = EngineCallbacks(
        on_state=lambda s, d: seen["states"].append(s),
        on_transcript=lambda s: seen["transcript"].append(s),
        on_board=lambda b: seen["board"].append(b),
        on_notes=lambda md: seen["notes"].append(md),
        on_task=lambda t: seen["tasks"].append(t),
        on_artifact=lambda a: seen["artifacts"].append(a),
    )
    engine = _make_engine(cfg, cb)
    info = engine.start(subject="数学", title="一元二次方程")
    assert info.id

    deadline = time.time() + 30
    while time.time() < deadline:
        tasks = engine.tasks()
        if tasks and all(t.status == "solved" for t in tasks):
            break
        time.sleep(0.2)

    engine.stop(finalize=True, wait=True)

    assert "recording" in seen["states"] and "finished" in seen["states"]
    assert len(seen["transcript"]) >= 3
    assert len(seen["board"]) >= 2
    assert seen["notes"], "应当产生课堂笔记"
    assert engine.tasks(), "应当识别出任务"
    task = engine.tasks()[0]
    assert task.status == "solved", task.error
    assert task.artifacts, "任务应当产出文件"
    for p in task.artifacts:
        assert Path(p).exists() and Path(p).stat().st_size > 0

    out_dir = Path(cfg.output.output_dir)
    files = [p.name for p in out_dir.rglob("*") if p.is_file()]
    assert any("作业汇总" in f for f in files), files
    assert any(f.endswith(".md") for f in files)

    ws = engine.workspace
    assert ws is not None
    assert ws.transcript_path.exists() and ws.board_path.exists()
    assert ws.notes_path.exists() and ws.timeline_path.exists()
    assert len(ws.read_transcript()) >= 3


def test_engine_refuses_incomplete_config(tmp_home, tmp_path):
    cfg = AppConfig()
    cfg.output.output_dir = str(tmp_path)
    engine = _make_engine(cfg)
    with pytest.raises(ValueError):
        engine.start()
