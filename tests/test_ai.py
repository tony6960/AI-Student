"""模型层（ai_student.ai）离线单测：不联网、不需要 API KEY（socket 被 autouse fixture 封死）。

覆盖 INTERFACES.md 2.1-2.8：LLMClient 构造参数 / payload 组装 / 重试策略 / 密钥脱敏 / ping、
ASR 三种模式分派与空音频短路、板书 JSON 容错、笔记合并拼接、任务畸形输出免疫与标题去重、
作答三段式解析与署名、提示词覆盖机制。全部用内联假客户端（FakeOpenAI / FakeLLM）。
"""
from __future__ import annotations

import base64
import dataclasses
import json
import socket
import sys
import wave
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

import httpx  # noqa: E402

from ai_student.ai import client as client_mod  # noqa: E402
from ai_student.ai.asr import is_effectively_silent, probe_wav, transcribe_chunk  # noqa: E402
from ai_student.ai.client import LLMClient, as_float, as_str_list, as_text, to_data_url  # noqa: E402
from ai_student.ai.errors import AIClientError, redact_secrets  # noqa: E402
from ai_student.ai.notes import ensure_title, merge_notes  # noqa: E402
from ai_student.ai.prompts import PROMPTS, get_prompt, render  # noqa: E402
from ai_student.ai.solver import SolveResult, parse_solve_output, solve_task, split_sections  # noqa: E402
from ai_student.ai.tasks import DEDUPE_THRESHOLD, extract_tasks, title_similarity  # noqa: E402
from ai_student.ai.vision import BoardAnalysis, analyze_board, parse_board_analysis  # noqa: E402
from ai_student.config.models import ModelEndpoint, OutputConfig, PersonalInfo  # noqa: E402
from ai_student.core.models import ClassTask  # noqa: E402

FENCE = chr(96) * 3          # Markdown 代码围栏，写成 chr(96) 避免转义问题
SLEEPS: list[float] = []     # 记录被替换掉的退避等待


# ====================================================================== 离线保障

@pytest.fixture(autouse=True)
def _no_network(monkeypatch):
    """任何真实网络连接都会直接失败——保证测试完全离线。"""

    def _blocked(*args, **kwargs):
        raise AssertionError("测试禁止联网")

    monkeypatch.setattr(socket, "socket", _blocked)
    monkeypatch.setattr(socket, "create_connection", _blocked)


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch):
    """退避等待不真的睡，只记录下来。"""
    SLEEPS.clear()
    monkeypatch.setattr(client_mod, "_sleep", SLEEPS.append)


# ====================================================================== 假客户端

def _completion(text: str) -> Any:
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=text), finish_reason="stop")])


class FakeOpenAI:
    """最小化的 openai.OpenAI 替身：只实现模型层用到的方法。"""

    def __init__(self, reply: str = "", transcript: str = "转写文本",
                 errors: list[Exception] | None = None, **kwargs: Any) -> None:
        self.kwargs, self.reply, self.transcript = kwargs, reply, transcript
        self.errors: list[Exception] = list(errors or [])
        self.payloads: list[dict[str, Any]] = []
        self.transcripts: list[dict[str, Any]] = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._chat))
        self.audio = SimpleNamespace(transcriptions=SimpleNamespace(create=self._transcribe))

    def _chat(self, **kwargs: Any) -> Any:
        self.payloads.append(kwargs)
        if self.errors:
            raise self.errors.pop(0)
        return _completion(self.reply)

    def _transcribe(self, **kwargs: Any) -> Any:
        self.transcripts.append(kwargs)
        if self.errors:
            raise self.errors.pop(0)
        return SimpleNamespace(text=self.transcript)


class HttpError(Exception):
    """带 status_code 的假 HTTP 异常，模拟 openai.APIStatusError。"""

    def __init__(self, status: int, message: str = "") -> None:
        super().__init__(message or f"HTTP {status}")
        self.status_code = status


def _install(monkeypatch, fake: FakeOpenAI) -> None:
    monkeypatch.setattr(client_mod.openai, "OpenAI", lambda **kwargs: fake)


def _install_factory(monkeypatch, factory) -> None:
    monkeypatch.setattr(client_mod.openai, "OpenAI", factory)


@dataclass
class FakeLLM:
    """内联假 LLM：记录调用参数，按队列返回预设文本（供笔记 / 任务 / 作答 / 视觉用）。"""

    replies: list[str] = field(default_factory=list)
    default: str = "{}"
    error: Exception | None = None
    calls: list[dict[str, Any]] = field(default_factory=list)

    def chat(self, system: str, user: str, images: list | None = None,
             model: str | None = None, json_mode: bool = False,
             temperature: float | None = None, max_tokens: int | None = None) -> str:
        self.calls.append({
            "system": system, "user": user, "images": list(images or []),
            "model": model, "json_mode": json_mode, "temperature": temperature,
            "max_tokens": max_tokens,
        })
        if self.error is not None:
            raise self.error
        return self.replies.pop(0) if self.replies else self.default

    @property
    def last(self) -> dict[str, Any]:
        return self.calls[-1]


# ====================================================================== WAV 辅助

def _pcm(value: int) -> bytes:
    return int(value).to_bytes(2, "little", signed=True)


def make_wav(path: Path, peak: int = 8000, seconds: float = 1.0, rate: int = 16000) -> Path:
    frames = _pcm(peak) * int(rate * seconds)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(frames)
    return path


# ====================================================================== 2.1 client

def test_aiclient_error_fields_and_redaction():
    key = "sk-test-1234567890abcdef"
    err = AIClientError(f"请求失败：Incorrect API key provided: {key}", status=401, retryable=False)
    assert key not in str(err)
    assert key not in err.message
    assert "***" in err.message
    assert (err.status, err.retryable) == (401, False)
    assert str(err) == err.message
    assert "API KEY" in err.user_hint()


def test_redact_secrets_forms():
    assert "sk-abcdef123456" not in redact_secrets("key=sk-abcdef123456", [])
    assert "abcdef123456" not in redact_secrets("Authorization: Bearer abcdef123456xyz", [])
    assert "my-secret-value" not in redact_secrets("key=my-secret-value", ["my-secret-value"])
    assert redact_secrets(None) == ""


def test_client_constructor_passes_sdk_kwargs(monkeypatch):
    captured: dict[str, Any] = {}

    def factory(**kwargs):
        captured.update(kwargs)
        return FakeOpenAI(reply="ok")

    _install_factory(monkeypatch, factory)
    cfg = ModelEndpoint(base_url="https://example.com/v1", api_key="sk-unit-key",
                        chat_model="unit-model", vision_model="unit-vision",
                        timeout_s=33, retries=1, extra_headers={"X-Trace": "abc"})
    client = LLMClient(cfg)
    assert captured["base_url"] == "https://example.com/v1"
    assert captured["api_key"] == "sk-unit-key"
    assert captured["timeout"] == 33.0
    assert captured["max_retries"] == 0      # 重试完全由业务层控制
    assert captured["default_headers"] == {"X-Trace": "abc"}
    assert client.vision_model == "unit-vision"
    # 没有单独配视觉模型时回落到对话模型
    fallback = LLMClient(ModelEndpoint(chat_model="only-chat", vision_model=""))
    assert fallback.vision_model == "only-chat"


def test_chat_payload_roles_and_json_mode(monkeypatch):
    fake = FakeOpenAI(reply="  你好  ")
    _install(monkeypatch, fake)
    cfg = ModelEndpoint(chat_model="m1", temperature=0.3, max_tokens=2048)
    text = LLMClient(cfg).chat("系统提示", "用户问题", json_mode=True,
                               temperature=0.1, max_tokens=64)
    assert text == "你好"
    payload = fake.payloads[0]
    assert payload["model"] == "m1"
    assert payload["messages"][0] == {"role": "system", "content": "系统提示"}
    assert payload["messages"][1] == {"role": "user", "content": "用户问题"}
    assert payload["response_format"] == {"type": "json_object"}
    assert payload["temperature"] == 0.1
    assert payload["max_tokens"] == 64


def test_chat_image_data_url(monkeypatch, tmp_path):
    fake = FakeOpenAI(reply="ok")
    _install(monkeypatch, fake)
    png = b"\x89PNG\r\n\x1a\n" + b"\x00" * 32
    jpg_path = tmp_path / "frame.jpg"
    jpg_path.write_bytes(b"\xff\xd8\xff" + b"\x00" * 32)
    LLMClient(ModelEndpoint(chat_model="m")).chat("s", "看图", images=[png, jpg_path])
    parts = fake.payloads[0]["messages"][1]["content"]
    assert parts[0] == {"type": "text", "text": "看图"}
    assert parts[1]["type"] == "image_url"
    assert parts[1]["image_url"]["url"].startswith("data:image/png;base64,")
    assert parts[2]["image_url"]["url"].startswith("data:image/jpeg;base64,")
    assert base64.b64decode(parts[1]["image_url"]["url"].split(",", 1)[1]) == png
    assert to_data_url(png) == parts[1]["image_url"]["url"]


def test_to_data_url_missing_file(tmp_path):
    with pytest.raises(AIClientError):
        to_data_url(tmp_path / "nope.jpg")


def test_retry_only_on_timeout_rate_limit_and_5xx(monkeypatch):
    cfg = ModelEndpoint(chat_model="m", retries=2)

    fake = FakeOpenAI(reply="ok", errors=[HttpError(429)])
    _install(monkeypatch, fake)
    assert LLMClient(cfg).chat("s", "u") == "ok"
    assert (len(fake.payloads), SLEEPS) == (2, [0.8])

    SLEEPS.clear()
    fake2 = FakeOpenAI(reply="ok", errors=[HttpError(500), HttpError(503)])
    _install(monkeypatch, fake2)
    assert LLMClient(cfg).chat("s", "u") == "ok"
    assert (len(fake2.payloads), SLEEPS) == (3, [0.8, 1.6])

    SLEEPS.clear()
    timeout_error = client_mod.openai.APITimeoutError(
        request=httpx.Request("POST", "https://example.com")
    )
    fake3 = FakeOpenAI(reply="ok", errors=[timeout_error])
    _install(monkeypatch, fake3)
    assert LLMClient(cfg).chat("s", "u") == "ok"
    assert (len(fake3.payloads), SLEEPS) == (2, [0.8])


def test_client_4xx_not_retried(monkeypatch):
    fake = FakeOpenAI(reply="ok", errors=[HttpError(401), HttpError(401)])
    _install(monkeypatch, fake)
    cfg = ModelEndpoint(chat_model="m", retries=3)
    with pytest.raises(AIClientError) as info:
        LLMClient(cfg).chat("s", "u")
    assert len(fake.payloads) == 1
    assert (info.value.status, info.value.retryable) == (401, False)
    assert SLEEPS == []


def test_client_retry_exhausted_is_reported(monkeypatch):
    fake = FakeOpenAI(reply="ok", errors=[HttpError(503), HttpError(503)])
    _install(monkeypatch, fake)
    cfg = ModelEndpoint(chat_model="m", retries=1)
    with pytest.raises(AIClientError) as info:
        LLMClient(cfg).chat("s", "u")
    assert len(fake.payloads) == 2
    assert (info.value.status, info.value.retryable) == (503, True)
    assert SLEEPS == [0.8]


def test_error_message_never_contains_api_key(monkeypatch):
    key = "sk-live-SECRET-abcdef123456"
    fake = FakeOpenAI(errors=[HttpError(401, f"Incorrect API key provided: {key}")])
    _install(monkeypatch, fake)
    cfg = ModelEndpoint(base_url="https://example.com/v1", api_key=key,
                        chat_model="m", retries=0)
    with pytest.raises(AIClientError) as info:
        LLMClient(cfg).chat("s", "u")
    message = str(info.value)
    assert key not in message
    assert "sk-live" not in message
    assert "***" in message


def test_ping_success_and_failure(monkeypatch):
    fake = FakeOpenAI(reply="正常")
    _install(monkeypatch, fake)
    cfg = ModelEndpoint(base_url="https://example.com/v1", api_key="sk-x", chat_model="ping-model")
    ok, message = LLMClient(cfg).ping()
    assert ok is True and "ping-model" in message
    assert fake.payloads[0]["max_tokens"] == 16

    fake2 = FakeOpenAI(errors=[HttpError(401, "bad key")])
    _install(monkeypatch, fake2)
    bad_cfg = ModelEndpoint(base_url="https://example.com/v1", api_key="sk-x",
                            chat_model="m", retries=0)
    ok2, message2 = LLMClient(bad_cfg).ping()
    assert ok2 is False and "401" in message2

    fake3 = FakeOpenAI(reply="x")
    _install(monkeypatch, fake3)
    ok3, _ = LLMClient(ModelEndpoint()).ping()
    assert ok3 is False and fake3.payloads == []


def test_client_transcribe_delegates_to_asr(monkeypatch, tmp_path):
    with pytest.raises(AIClientError):
        LLMClient(ModelEndpoint(asr_kind="none")).transcribe(tmp_path / "a.wav")


def test_output_coercion_helpers():
    assert as_text(None) == "" and as_text("  x ") == "x"
    assert as_text(["a", "", "b"]) == "a\nb"
    assert as_str_list("a\nb；c") == ["a", "b", "c"]
    assert as_str_list(None) == [] and as_str_list(42) == ["42"]
    assert as_float("85%") == 0.85 and as_float("1,5") == 1.5
    assert as_float("没有数字", default=0.4) == 0.4
    assert as_float(9, lo=0, hi=1) == 1.0


# ====================================================================== 2.2 asr

def test_asr_kind_none_raises(tmp_path):
    cfg = ModelEndpoint(asr_kind="none")
    with pytest.raises(AIClientError) as info:
        transcribe_chunk(FakeLLM(), make_wav(tmp_path / "a.wav"), cfg)
    assert "语音识别" in str(info.value)


def test_asr_silent_and_short_audio_skip_api(tmp_path):
    cfg = ModelEndpoint(asr_kind="openai_audio", asr_model="qwen3-asr-flash")
    llm = FakeLLM(default="不应该被调用")
    silent = make_wav(tmp_path / "silent.wav", peak=0)
    assert is_effectively_silent(silent) is True
    assert transcribe_chunk(llm, silent, cfg) == ""
    short = make_wav(tmp_path / "short.wav", peak=9000, seconds=0.1)
    assert is_effectively_silent(short) is True
    assert transcribe_chunk(llm, short, cfg) == ""
    assert llm.calls == []


def test_probe_wav_peak(tmp_path):
    duration, peak = probe_wav(make_wav(tmp_path / "tone.wav", peak=8000, seconds=1.0))
    assert abs(duration - 1.0) < 0.01
    assert peak is not None and abs(peak - 8000 / 32768) < 1e-4
    silence_duration, silence_peak = probe_wav(make_wav(tmp_path / "s.wav", peak=0))
    assert silence_peak == 0.0 and silence_duration > 0.5
    assert probe_wav(tmp_path / "missing.wav") == (0.0, None)
    broken = tmp_path / "broken.wav"
    broken.write_bytes(b"not a wav")
    assert probe_wav(broken)[1] is None


def test_asr_missing_file_raises(tmp_path):
    cfg = ModelEndpoint(asr_kind="openai_audio", asr_model="m")
    with pytest.raises(AIClientError):
        transcribe_chunk(FakeLLM(), tmp_path / "nope.wav", cfg)


def test_asr_openai_audio_payload(monkeypatch, tmp_path):
    fake = FakeOpenAI(reply="  老师讲勾股定理  ")
    _install(monkeypatch, fake)
    cfg = ModelEndpoint(asr_kind="openai_audio", asr_model="qwen3-asr-flash")
    wav = make_wav(tmp_path / "chunk.wav", peak=6000)
    text = transcribe_chunk(LLMClient(cfg), wav, cfg, hint="勾股定理")
    assert text == "老师讲勾股定理"
    payload = fake.payloads[0]
    assert payload["model"] == "qwen3-asr-flash"
    assert payload["temperature"] == 0.0
    system = payload["messages"][0]
    assert system["role"] == "system" and "只输出课堂原话" in system["content"]
    assert "勾股定理" in system["content"]
    part = payload["messages"][1]["content"][0]
    assert part["type"] == "input_audio"
    assert part["input_audio"]["format"] == "wav"
    assert base64.b64decode(part["input_audio"]["data"]) == wav.read_bytes()


def test_asr_openai_transcription_payload(monkeypatch, tmp_path):
    fake = FakeOpenAI(transcript=" 转写结果 ")
    _install(monkeypatch, fake)
    cfg = ModelEndpoint(asr_kind="openai_transcription", asr_model="whisper-1")
    wav = make_wav(tmp_path / "c.wav")
    text = transcribe_chunk(LLMClient(cfg), wav, cfg, hint="本节课术语：函数")
    assert text == "转写结果"
    call = fake.transcripts[0]
    assert call["model"] == "whisper-1"
    assert call["file"][0] == "audio.wav" and call["file"][2] == "audio/wav"
    assert call["prompt"] == "本节课术语：函数"
    assert fake.payloads == []


def test_asr_openai_transcription_retries_without_prompt(monkeypatch, tmp_path):
    """转写接口不接受 prompt 时，自动去掉术语提示重试一次。"""
    fake = FakeOpenAI(transcript="转写结果", errors=[HttpError(400, "unknown parameter prompt")])
    _install(monkeypatch, fake)
    cfg = ModelEndpoint(asr_kind="openai_transcription", asr_model="whisper-1", retries=0)
    text = transcribe_chunk(LLMClient(cfg), make_wav(tmp_path / "c.wav"), cfg, hint="术语")
    assert text == "转写结果"
    assert len(fake.transcripts) == 2
    assert fake.transcripts[0]["prompt"] == "术语"
    assert "prompt" not in fake.transcripts[1]


def test_asr_openai_audio_retries_without_temperature(monkeypatch, tmp_path):
    """部分 ASR 模型不接受 temperature：报 400 时自动去掉该参数重试一次。"""
    fake = FakeOpenAI(reply="老师好", errors=[HttpError(400, "unsupported parameter temperature")])
    _install(monkeypatch, fake)
    cfg = ModelEndpoint(asr_kind="openai_audio", asr_model="qwen3-asr-flash", retries=0)
    wav = make_wav(tmp_path / "c.wav")
    assert transcribe_chunk(LLMClient(cfg), wav, cfg) == "老师好"
    assert len(fake.payloads) == 2
    assert "temperature" in fake.payloads[0]
    assert "temperature" not in fake.payloads[1]


def test_asr_endpoint_override_builds_second_client(monkeypatch, tmp_path):
    built: list[dict[str, Any]] = []

    def factory(**kwargs):
        built.append(kwargs)
        return FakeOpenAI(transcript="ok")

    _install_factory(monkeypatch, factory)
    cfg = ModelEndpoint(base_url="https://chat.example.com/v1", api_key="sk-chat",
                        chat_model="m", asr_kind="openai_transcription",
                        asr_model="whisper-1", asr_base_url="https://asr.example.com/v1",
                        asr_api_key="sk-asr")
    client = LLMClient(cfg)
    assert len(built) == 1
    transcribe_chunk(client, make_wav(tmp_path / "c.wav"), cfg)
    assert len(built) == 2
    assert built[1]["base_url"] == "https://asr.example.com/v1"
    assert built[1]["api_key"] == "sk-asr"
    transcribe_chunk(client, make_wav(tmp_path / "d.wav"), cfg)
    assert len(built) == 2      # ASR 客户端被复用


# ====================================================================== 2.3 vision

def test_parse_board_analysis_json_with_fence():
    payload = {"ocr_text": "勾股定理\na^2+b^2=c^2", "summary": "讲勾股定理",
               "key_points": ["直角边平方和等于斜边平方"], "problems": ["例3 求斜边"],
               "subject_hint": "数学"}
    text = f"{FENCE}json\n{json.dumps(payload, ensure_ascii=False)}\n{FENCE}"
    result = parse_board_analysis(text)
    assert result.ocr_text.startswith("勾股定理")
    assert result.summary == "讲勾股定理"
    assert result.key_points == ["直角边平方和等于斜边平方"]
    assert result.problems == ["例3 求斜边"]
    assert result.subject_hint == "数学"


def test_parse_board_analysis_tolerates_bad_shapes():
    assert parse_board_analysis("") == BoardAnalysis()
    assert parse_board_analysis("null") == BoardAnalysis()
    prose = parse_board_analysis("这一屏在讲勾股定理")
    assert prose.summary == "这一屏在讲勾股定理" and prose.ocr_text == ""
    coerced = parse_board_analysis(json.dumps(
        {"key_points": "要点一；要点二", "problems": 42, "ocr_text": None, "subject_hint": ["数学"]},
        ensure_ascii=False))
    assert coerced.key_points == ["要点一", "要点二"]
    assert coerced.problems == ["42"]
    assert coerced.ocr_text == ""
    assert coerced.subject_hint == "数学"


def test_board_analysis_fields_match_contract():
    assert [f.name for f in dataclasses.fields(BoardAnalysis)] == [
        "ocr_text", "summary", "key_points", "problems", "subject_hint"]


def test_analyze_board_uses_vision_model():
    reply = json.dumps({"summary": "讲定理", "ocr_text": "板书"}, ensure_ascii=False)
    llm = FakeLLM(replies=[reply])
    cfg = ModelEndpoint(chat_model="text-model", vision_model="vision-model")
    result = analyze_board(llm, b"\xff\xd8\xff", cfg, context="数学")
    assert result.summary == "讲定理"
    call = llm.last
    assert call["model"] == "vision-model"
    assert call["json_mode"] is True
    assert call["images"] == [b"\xff\xd8\xff"]
    assert "数学" in call["user"]
    assert "ocr_text" in call["system"]


# ====================================================================== 2.4 notes

def test_merge_notes_assembles_inputs():
    llm = FakeLLM(replies=["# 课堂笔记\n## 知识点\n- 勾股定理"])
    cfg = ModelEndpoint(chat_model="m")
    person = PersonalInfo(name="张三", grade="初二", class_name="3班")
    merged = merge_notes(llm, "## 知识点\n- 旧内容", ["[00:00:10] 老师讲勾股定理"],
                         ["板书：a^2+b^2=c^2"], cfg, person, subject="数学")
    assert merged.startswith("# 课堂笔记")
    call = llm.last
    assert call["model"] == "m"
    user = call["user"]
    for token in ("旧内容", "勾股定理", "a^2+b^2=c^2", "[00:00:10]", "张三", "数学"):
        assert token in user, token
    system = call["system"]
    for heading in ("# 课堂笔记", "## 课程信息", "## 知识点", "## 板书与演示要点",
                    "## 老师布置的任务", "## 存疑"):
        assert heading in system, heading


def test_merge_notes_empty_previous_and_empty_reply():
    llm = FakeLLM(replies=["# 课堂笔记\n## 知识点\n- x"])
    merge_notes(llm, "", ["新内容"], [], ModelEndpoint(chat_model="m"), PersonalInfo(name="李四"))
    assert "从零开始" in llm.last["user"]

    blank = FakeLLM(replies=[""])
    kept = merge_notes(blank, "旧笔记内容", ["新内容"], [], ModelEndpoint(), PersonalInfo())
    assert kept == "旧笔记内容"


def test_ensure_title():
    assert ensure_title("## 知识点\n- x").startswith("# 课堂笔记")
    assert ensure_title("# 课堂笔记\n## 知识点").startswith("# 课堂笔记")
    assert ensure_title("") == "# 课堂笔记"


# ====================================================================== 2.5 tasks

def test_title_similarity_pure_function():
    assert title_similarity("完成练习册第12页第3题", "完成练习册第 12 页第 3 题！") == 1.0
    assert title_similarity("数学作业", "英语作文") == 0.0
    assert title_similarity("", "任何东西") == 0.0
    assert 0.0 < title_similarity("完成课本P12第3题", "完成课本P12第4题") < 1.0
    assert DEDUPE_THRESHOLD == 0.75


@pytest.mark.parametrize("reply", [
    "这不是 JSON", "", "   ", "{}", "null", "[1,2,3]", "[]",
    FENCE + "json\n{坏}\n" + FENCE,
])
def test_extract_tasks_malformed_output_returns_empty(reply):
    notes = "## 老师布置的任务\n- 完成练习册 P12"
    llm = FakeLLM(replies=[reply])
    assert extract_tasks(llm, ModelEndpoint(), PersonalInfo(), notes) == []
    assert len(llm.calls) == 1


def test_extract_tasks_without_material_skips_api():
    llm = FakeLLM(default="[]")
    assert extract_tasks(llm, ModelEndpoint(), PersonalInfo(), "", "") == []
    assert llm.calls == []


def test_extract_tasks_parses_and_normalizes():
    payload = [
        {"title": "完成练习册第12页第3题", "detail": "P12 第3题", "task_type": "随堂练习",
         "requirements": "写清过程\n不要用铅笔", "deadline": "周五", "confidence": "85%",
         "evidence": "把练习册第12页第3题做完", "subject": ""},
        {"detail": "没有标题的任务", "confidence": 0.2},
    ]
    llm = FakeLLM(replies=[json.dumps(payload, ensure_ascii=False)])
    found = extract_tasks(llm, ModelEndpoint(chat_model="m"), PersonalInfo(name="张三"),
                          "笔记内容", "老师：把第3题做完", subject="数学")
    assert len(found) == 1
    task = found[0]
    assert task.title == "完成练习册第12页第3题"
    assert task.task_type == "作业"          # 未知类型归一
    assert task.requirements == ["写清过程", "不要用铅笔"]
    assert task.deadline == "周五"
    assert task.confidence == 0.85           # 百分数折算
    assert task.subject == "数学"            # 回落到入参
    assert "老师原话" in task.detail and "把练习册第12页第3题做完" in task.detail
    assert task.status == "discovered"
    assert llm.last["json_mode"] is True


def test_extract_tasks_dict_wrapper_and_confidence_clamp():
    wrapped = {"tasks": [{"title": "背《静夜思》", "confidence": 85},
                         {"title": "预习第五课", "confidence": 1.5}]}
    llm = FakeLLM(replies=[json.dumps(wrapped, ensure_ascii=False)])
    found = extract_tasks(llm, ModelEndpoint(), PersonalInfo(), "笔记", subject="语文")
    assert [t.title for t in found] == ["背《静夜思》", "预习第五课"]
    assert found[0].confidence == 0.85
    assert found[1].confidence == 1.0
    assert all(t.subject == "语文" for t in found)


def test_extract_tasks_dedupes_against_known_and_batch():
    known = [ClassTask(title="完成练习册第12页第3题")]
    reply = json.dumps([{"title": "完成练习册第 12 页第 3 题！"}], ensure_ascii=False)
    llm = FakeLLM(replies=[reply])
    assert extract_tasks(llm, ModelEndpoint(), PersonalInfo(), "笔记", known_tasks=known) == []
    assert "完成练习册第12页第3题" in llm.last["user"]

    batch = [{"title": "背诵《静夜思》全文"}, {"title": "背诵静夜思全文！"}]
    llm2 = FakeLLM(replies=[json.dumps(batch, ensure_ascii=False)])
    found = extract_tasks(llm2, ModelEndpoint(), PersonalInfo(), "笔记")
    assert len(found) == 1


def test_extract_tasks_api_error_propagates():
    llm = FakeLLM(error=AIClientError("模型调用失败"))
    with pytest.raises(AIClientError):
        extract_tasks(llm, ModelEndpoint(), PersonalInfo(), "笔记内容")


# ====================================================================== 2.6 solver

SOLVE_REPLY = """### 题目理解
已知直角边 a=3、b=4，求斜边 c。题目没有给单位，这里按厘米处理。
### 解答
由勾股定理得：
c = sqrt(a^2 + b^2) = sqrt(9 + 16) = 5（厘米）
### 自检
计算正确，步骤完整。
结论：通过"""


def test_solve_task_parses_three_sections():
    llm = FakeLLM(replies=[SOLVE_REPLY])
    cfg = ModelEndpoint(chat_model="m", temperature=0.25, max_tokens=1024)
    person = PersonalInfo(name="张三", grade="初二", class_name="3班", student_id="2024001",
                          teacher_name="李老师", answer_style="详细讲解", sign_artifacts=True)
    task = ClassTask(title="第一章/作业：求斜边 3<5 的值？", task_type="作业", subject="数学",
                     detail="已知 a=3,b=4，求 c", requirements=["写出过程"], deadline="周五")
    result = solve_task(llm, cfg, person, task, "课堂资料：勾股定理", OutputConfig())
    assert result.answer.startswith("由勾股定理得")
    assert "###" not in result.answer
    assert "已知直角边" in result.workpaper
    assert result.self_check.startswith("计算正确")
    assert result.needs_human is False
    assert result.file_title
    assert not any(ch in result.file_title for ch in '/\\:*?"<>|')
    call = llm.last
    assert call["model"] == "m"
    assert call["temperature"] == 0.25 and call["max_tokens"] == 1024
    assert "详细讲解" in call["user"]
    assert "勾股定理" in call["user"]
    assert "写出过程" in call["user"]
    assert "张三" in call["system"] and "署名" in call["system"]


def test_solve_task_without_signature():
    llm = FakeLLM(replies=[SOLVE_REPLY])
    person = PersonalInfo(name="张三", sign_artifacts=False)
    solve_task(llm, ModelEndpoint(), person, ClassTask(title="作业"), "ctx", OutputConfig())
    assert "张三" not in llm.last["system"]


def test_solve_result_fields_match_contract():
    assert [f.name for f in dataclasses.fields(SolveResult)] == [
        "answer", "workpaper", "file_title", "self_check", "needs_human"]


def test_parse_solve_output_plain_text_and_needs_human():
    plain = parse_solve_output("直接写的一段答案", "标题", "数学")
    assert plain.answer == "直接写的一段答案" and plain.workpaper == ""
    assert plain.needs_human is False

    human = parse_solve_output(
        "### 题目理解\n信息不足。\n### 解答\n按假设求解得 5。\n### 自检\n结论：需要人工复核",
        "作业", "数学")
    assert human.needs_human is True

    empty = parse_solve_output("", "作业", "数学")
    assert empty.answer == "" and empty.needs_human is True and empty.file_title == "作业"


def test_split_sections_tolerates_bold_markers():
    text = "**题目理解**\n题意\n**解答：**\n答案\n**自检**\n结论：通过"
    sections = split_sections(text)
    assert sections["题目理解"] == "题意"
    assert sections["解答"] == "答案"
    assert sections["自检"] == "结论：通过"


def test_file_title_sanitized():
    result = parse_solve_output("### 解答\n答案", "第一章/作业：求 3<5 的值？", "数学")
    assert result.file_title
    assert not any(ch in result.file_title for ch in "/\\:*?\"<>|")
    assert parse_solve_output("### 解答\n答案", "", "") .file_title == "作业"


# ====================================================================== 2.7 prompts

def test_get_prompt_and_overrides():
    for key in ("asr", "vision_board", "notes_merge", "task_extract", "solve",
                "self_check", "summary"):
        assert get_prompt(key).strip()
    assert get_prompt("solve", {"solve": "自定义解题提示词"}) == "自定义解题提示词"
    assert get_prompt("solve", {"solve": "   "}) == PROMPTS["solve"]
    assert get_prompt("solve", {"别的键": "x"}) == PROMPTS["solve"]
    assert get_prompt(" solve ") == PROMPTS["solve"]
    with pytest.raises(KeyError):
        get_prompt("不存在的键")


def test_prompts_contain_output_contracts():
    assert '"ocr_text"' in PROMPTS["vision_board"]
    assert '"confidence"' in PROMPTS["task_extract"]
    assert "### 题目理解" in PROMPTS["solve"] and "### 自检" in PROMPTS["solve"]
    assert "结论：通过" in PROMPTS["self_check"]
    assert "## 存疑" in PROMPTS["notes_merge"]
    assert "只输出课堂原话" in PROMPTS["asr"]


def test_render_is_safe():
    assert render("a{x}b", x=1) == "a1b"
    assert render("a{y}b") == "a{y}b"
    assert render("a{x}b", x=None) == "ab"


# ====================================================================== 引擎调用方式

def test_engine_style_flow_with_fake_client(tmp_path):
    """按 core.engine 的真实调用方式（位置参数）把模型层串一遍。"""
    llm = FakeLLM(replies=[
        "# 课堂笔记\n## 知识点\n- 勾股定理",
        json.dumps([{"title": "完成练习册第12页第3题", "confidence": 0.9}], ensure_ascii=False),
        SOLVE_REPLY,
        json.dumps({"summary": "讲勾股定理", "ocr_text": "a^2+b^2=c^2"}, ensure_ascii=False),
    ])
    cfg = ModelEndpoint(chat_model="c", vision_model="v")
    person = PersonalInfo(name="张三", subject="数学")

    notes = merge_notes(llm, "", ["老师讲勾股定理"], ["板书：a^2+b^2=c^2"], cfg, person, subject="数学")
    assert notes.startswith("# 课堂笔记")

    tasks = extract_tasks(llm, cfg, person, notes, recent_transcript="把第3题做完",
                          known_tasks=[], subject="数学")
    assert len(tasks) == 1 and tasks[0].title == "完成练习册第12页第3题"

    result = solve_task(llm, cfg, person, tasks[0], "课堂资料", OutputConfig())
    assert result.answer and result.needs_human is False

    analysis = analyze_board(llm, b"\xff\xd8\xff", cfg, context="数学")
    assert analysis.summary == "讲勾股定理" and analysis.ocr_text == "a^2+b^2=c^2"

    # 未配置语音识别时明确报错（引擎会记一条 WARNING 并继续）
    with pytest.raises(AIClientError):
        transcribe_chunk(llm, make_wav(tmp_path / "chunk.wav", peak=5000),
                         ModelEndpoint(asr_kind="none"))


# ====================================================================== 2.8 包导出

def test_ai_package_exports():
    import ai_student.ai as ai
    for name in ("AIClientError", "LLMClient", "transcribe_chunk", "BoardAnalysis",
                 "analyze_board", "merge_notes", "extract_tasks", "title_similarity",
                 "SolveResult", "solve_task", "get_prompt", "PROMPTS"):
        assert hasattr(ai, name), name
