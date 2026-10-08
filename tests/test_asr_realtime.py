"""实时语音识别（DashScope WebSocket）与"文本/视觉分属不同服务商"的离线测试。

不联网：WebSocket 与 OpenAI SDK 都被替身接管，只验证协议拼装与路由逻辑。
"""
from __future__ import annotations

import json
import wave
from pathlib import Path

import numpy as np
import pytest

from ai_student.ai.asr_realtime import (
    DEFAULT_WS_URL, read_pcm16, resolve_ws_url, transcribe_chunk_realtime,
)
from ai_student.config.models import AppConfig, ModelEndpoint
from ai_student.config.store import load_config, save_config


def _write_wav(path: Path, seconds: float = 1.0, rate: int = 16000,
               channels: int = 1, amp: int = 12000, width: int = 2) -> Path:
    n = int(seconds * rate)
    t = np.linspace(0, seconds, n, endpoint=False)
    wave_data = (amp * np.sin(2 * np.pi * 440 * t)).astype("<i2")
    if channels > 1:
        wave_data = np.repeat(wave_data.reshape(-1, 1), channels, axis=1).reshape(-1)
    with wave.open(str(path), "wb") as fh:
        fh.setnchannels(channels)
        fh.setsampwidth(width)
        fh.setframerate(rate)
        fh.writeframes(wave_data.tobytes())
    return path


# ------------------------------------------------------------------ 地址解析
def test_resolve_ws_url_variants():
    assert resolve_ws_url(ModelEndpoint()) == DEFAULT_WS_URL
    ep = ModelEndpoint(asr_base_url="wss://my.host/ws/asr")
    assert resolve_ws_url(ep) == "wss://my.host/ws/asr"
    ep2 = ModelEndpoint(asr_base_url="https://dashscope.aliyuncs.com/compatible-mode/v1")
    assert resolve_ws_url(ep2) == "wss://dashscope.aliyuncs.com/api-ws/v1/inference"
    ep3 = ModelEndpoint(asr_base_url="http://127.0.0.1:9000/v1")
    assert resolve_ws_url(ep3) == "wss://127.0.0.1:9000/api-ws/v1/inference"


# ------------------------------------------------------------------ 音频预处理
def test_read_pcm16_passthrough(tmp_path):
    wav = _write_wav(tmp_path / "a.wav", 1.0, 16000, 1)
    pcm, rate = read_pcm16(wav)
    assert rate == 16000
    assert len(pcm) == 32000, "1 秒 16k 单声道 16bit = 32000 字节"


def test_read_pcm16_downmixes_stereo_and_resamples(tmp_path):
    wav = _write_wav(tmp_path / "b.wav", 1.0, 44100, 2)
    pcm, rate = read_pcm16(wav)
    assert rate == 16000, "统一重采样到 16k"
    assert abs(len(pcm) - 32000) <= 4, "重采样后长度应接近 1 秒"


def test_read_pcm16_rejects_non_16bit(tmp_path):
    wav = _write_wav(tmp_path / "c.wav", 0.2, 16000, 1, width=1)
    with pytest.raises(Exception) as exc:
        read_pcm16(wav)
    assert "16 位" in str(exc.value)


# ------------------------------------------------------------------ 双服务商路由
def test_vision_and_asr_keys_are_encrypted_roundtrip(tmp_path):
    cfg = AppConfig()
    cfg.model.api_key = "sk-deepseek-plain"
    cfg.model.vision_api_key = "sk-dashscope-vision"
    cfg.model.asr_api_key = "sk-dashscope-asr"
    cfg.model.vision_base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    path = tmp_path / "config.json"
    save_config(cfg, path)

    raw = path.read_text(encoding="utf-8")
    for secret in ("sk-deepseek-plain", "sk-dashscope-vision", "sk-dashscope-asr"):
        assert secret not in raw, "密钥不能明文落盘：" + secret

    back = load_config(path)
    assert back.model.api_key == "sk-deepseek-plain"
    assert back.model.vision_api_key == "sk-dashscope-vision"
    assert back.model.asr_api_key == "sk-dashscope-asr"


def test_llmclient_routes_vision_to_separate_endpoint(monkeypatch):
    import ai_student.ai.client as client_mod

    built: list[dict] = []

    class _FakeOpenAI:
        def __init__(self, **kwargs):
            built.append(kwargs)
            self.chat = type("C", (), {"completions": type("D", (), {"create": staticmethod(lambda **k: None)})()})()

    monkeypatch.setattr(client_mod.openai, "OpenAI", _FakeOpenAI)

    ep = ModelEndpoint(
        base_url="https://api.deepseek.com/v1", api_key="ds",
        chat_model="deepseek-chat",
        vision_base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        vision_api_key="qw", vision_model="qwen3-vl-plus",
    )
    client = client_mod.LLMClient(ep)
    assert client.vision_client() is not client._client, "视觉必须走独立客户端"
    urls = {b.get("base_url") for b in built}
    assert "https://api.deepseek.com/v1" in urls
    assert "https://dashscope.aliyuncs.com/compatible-mode/v1" in urls


def test_effective_vision_endpoint_falls_back():
    ep = ModelEndpoint(base_url="https://x/v1", api_key="k", vision_model="vm")
    assert ep.effective_vision_base_url == "https://x/v1"
    assert ep.effective_vision_api_key == "k"
    assert ep.effective_vision_model == "vm"


# ------------------------------------------------------------------ 分发与参数
def test_transcribe_chunk_dispatches_to_realtime(monkeypatch, tmp_path):
    import ai_student.ai.asr as asr_mod
    import ai_student.ai.asr_realtime as rt_mod

    calls: list[dict] = []

    def fake_realtime(client, wav_path, cfg, hint=""):
        calls.append({"path": Path(wav_path), "model": cfg.asr_model, "hint": hint})
        return "这是实时识别的结果"

    monkeypatch.setattr(rt_mod, "transcribe_chunk_realtime", fake_realtime)
    wav = _write_wav(tmp_path / "chunk.wav", 0.5)
    cfg = ModelEndpoint(asr_kind="dashscope_realtime", asr_model="fun-asr-realtime-2026-02-28")
    text = asr_mod.transcribe_chunk(object(), wav, cfg)
    assert text == "这是实时识别的结果"
    assert calls and calls[0]["model"] == "fun-asr-realtime-2026-02-28"


def test_realtime_requires_model_and_key(tmp_path):
    from ai_student.ai.errors import AIClientError

    wav = _write_wav(tmp_path / "chunk.wav", 0.5)
    with pytest.raises(AIClientError) as exc:
        transcribe_chunk_realtime(
            object(), wav, ModelEndpoint(asr_kind="dashscope_realtime", asr_model="")
        )
    assert "模型名称" in str(exc.value)

    cfg = ModelEndpoint(asr_kind="dashscope_realtime", asr_model="m", api_key="", asr_api_key="")
    with pytest.raises(AIClientError) as exc2:
        transcribe_chunk_realtime(object(), wav, cfg)
    assert "API KEY" in str(exc2.value)


def test_ws_protocol_frames(monkeypatch, tmp_path):
    """用假 WebSocket 走一遍 run-task 帧 -> 音频 -> finish-task 帧的完整拼装。"""
    import ai_student.ai.asr_realtime as rt_mod

    sent: list = []

    class _FakeWS:
        def __init__(self) -> None:
            self.replies: list[str] = [
                json.dumps({"header": {"event": "task-started"}}),
                json.dumps({"header": {"event": "result-generated"},
                            "payload": {"output": {"sentence": {"sentence_id": 1, "text": "同学们"}}}}),
                json.dumps({"header": {"event": "task-completed"}}),
                json.dumps({"header": {"event": "task-finished"}}),
            ]
            self.sock = None

        def settimeout(self, _t):
            pass

        def send(self, data):
            sent.append(json.loads(data))

        def send_binary(self, data):
            sent.append(("binary", len(data)))

        def recv(self):
            if self.replies:
                return self.replies.pop(0)
            raise TimeoutError("没有更多事件")

        def close(self):
            pass

    monkeypatch.setattr(rt_mod, "_open_socket", lambda url, key, timeout: _FakeWS())
    wav = _write_wav(tmp_path / "chunk.wav", 0.5)
    text = transcribe_chunk_realtime(
        object(), wav, ModelEndpoint(asr_kind="dashscope_realtime", asr_model="fun-asr-realtime-2026-02-28",
                                     api_key="k")
    )
    assert text == "同学们", "应汇总 result-generated 里的句子"
    run_task = sent[0]
    assert run_task["header"]["action"] == "run-task"
    assert run_task["header"]["streaming"] == "duplex"
    payload = run_task["payload"]
    assert payload["task"] == "asr" and payload["function"] == "recognition"
    assert payload["model"] == "fun-asr-realtime-2026-02-28"
    assert payload["parameters"]["format"] == "pcm"
    assert payload["parameters"]["sample_rate"] == 16000
    assert any(isinstance(item, tuple) and item[0] == "binary" for item in sent), "必须发送二进制音频帧"
    assert sent[-1]["header"]["action"] == "finish-task"
