"""阿里云百炼「实时语音识别」(fun-asr-realtime) 客户端。

realtime 系列**只能走 WebSocket**，和 OpenAI 兼容接口不是一回事：

    1. 连 wss://dashscope.aliyuncs.com/api-ws/v1/inference（Authorization: bearer <key>）
    2. 发 run-task 文本帧：task=asr / function=recognition / model=... / parameters.format=pcm
    3. 收到 task-started 后，把 16k 单声道 16bit PCM 以二进制帧持续发过去
    4. 发 finish-task，等 task-finished，汇总所有 result-generated 里的句子

我们的流水线是按"切片文件"工作的，所以这里对每个 wav 切片建立一次会话，把整片音频流过去
再收回完整文本——既能用上 realtime 模型的中文识别质量，又不用改动采集与引擎的线程模型。
"""
from __future__ import annotations

import json
import logging
import select
import ssl
import time
import uuid
import wave
from pathlib import Path
from typing import Any

from ..config.models import ModelEndpoint
from .errors import AIClientError

logger = logging.getLogger(__name__)

DEFAULT_WS_URL = "wss://dashscope.aliyuncs.com/api-ws/v1/inference"

# 100ms 一帧（16k / 16bit / 单声道 = 3200 字节）：既能及时被识别，又不会把帧切得太碎
_FRAME_BYTES = 3200
_TARGET_RATE = 16000
_EVENT_FAILED = "task-failed"
_EVENT_DONE = "task-finished"


# ---------------------------------------------------------------------- 地址与音频
def resolve_ws_url(cfg: ModelEndpoint) -> str:
    """从 asr_base_url 推导 WebSocket 地址；没配或不是 ws 地址就用百炼官方地址。"""
    raw = (cfg.asr_base_url or "").strip()
    if raw.startswith(("ws://", "wss://")):
        return raw
    if raw:
        host = raw.split("://", 1)[-1].split("/", 1)[0].strip()
        if host:
            return "wss://" + host + "/api-ws/v1/inference"
    return DEFAULT_WS_URL


def read_pcm16(path: Path) -> tuple[bytes, int]:
    """把 wav 读成 16k 单声道 16bit PCM；多声道取平均，采样率不符则线性插值。"""
    import numpy as np

    with wave.open(str(path), "rb") as fh:
        channels = fh.getnchannels()
        width = fh.getsampwidth()
        rate = fh.getframerate()
        raw = fh.readframes(fh.getnframes())
    if width != 2:
        raise AIClientError("实时语音识别只支持 16 位 PCM（当前 %d 位）" % (width * 8))
    arr = np.frombuffer(raw, dtype="<i2")
    if arr.size == 0:
        return b"", _TARGET_RATE
    if channels > 1:
        usable = (arr.size // channels) * channels
        arr = arr[:usable].reshape(-1, channels).astype(np.float32).mean(axis=1)
    else:
        arr = arr.astype(np.float32)
    if rate != _TARGET_RATE and arr.size > 1:
        count = max(1, int(round(arr.size * float(_TARGET_RATE) / rate)))
        grid = np.linspace(0.0, arr.size - 1.0, count)
        arr = np.interp(grid, np.arange(arr.size, dtype=np.float32), arr)
    pcm = np.clip(arr, -32768.0, 32767.0).astype("<i2")
    return pcm.tobytes(), _TARGET_RATE


# ---------------------------------------------------------------------- 连接与收发
def _open_socket(url: str, api_key: str, timeout: float):
    try:
        import websocket  # websocket-client
    except ImportError as exc:  # pragma: no cover
        raise AIClientError("缺少 websocket-client，无法使用实时语音识别") from exc
    headers = [
        "Authorization: bearer " + (api_key or ""),
        "X-DashScope-DataInspection: enable",
        "User-Agent: ai-student/1.0",
    ]
    try:
        return websocket.create_connection(url, header=headers, timeout=timeout)
    except ssl.SSLError as exc:
        logger.warning("实时语音识别 TLS 校验失败（%s），本次改为不校验证书重连", exc)
        return websocket.create_connection(
            url, header=headers, timeout=timeout, sslopt={"cert_reqs": ssl.CERT_NONE}
        )


def _recv_json(ws: Any, deadline: float) -> dict:
    """收一条 JSON 事件；期间收到二进制帧或非 JSON 就跳过。"""
    import websocket

    while time.monotonic() < deadline:
        try:
            raw = ws.recv()
        except websocket.WebSocketTimeoutException:
            continue
        if isinstance(raw, (bytes, bytearray)) or not raw:
            continue
        try:
            data = json.loads(raw)
        except Exception:
            continue
        if isinstance(data, dict):
            return data
    raise AIClientError("实时语音识别超时（等待服务端事件）")


def _drain(ws: Any, ordered: list, texts: dict, budget: float = 0.0) -> None:
    """非阻塞地把已经到达的识别结果收下来，避免服务端发送缓冲被写满。"""
    import websocket

    end = time.monotonic() + budget
    while True:
        sock = getattr(ws, "sock", None)
        if sock is None:
            return
        try:
            ready, _, _ = select.select([sock], [], [], max(0.0, end - time.monotonic()))
        except (OSError, ValueError):
            return
        if not ready:
            return
        try:
            raw = ws.recv()
        except websocket.WebSocketTimeoutException:
            return
        except Exception:
            return
        if isinstance(raw, (bytes, bytearray)) or not raw:
            continue
        try:
            data = json.loads(raw)
        except Exception:
            continue
        _collect(data, ordered, texts)


def _collect(message: dict, ordered: list, texts: dict) -> None:
    """同一个句子会多次下发（中间结果），这里按 sentence_id 保留最新文本。"""
    header = message.get("header") or {}
    event = header.get("event")
    if event == _EVENT_FAILED:
        return
    output = (message.get("payload") or {}).get("output") or {}
    sentence = output.get("sentence") or {}
    text = sentence.get("text")
    if text is None:
        text = output.get("text")
    if not isinstance(text, str) or not text.strip():
        return
    sid = sentence.get("sentence_id")
    if sid is None:
        sid = output.get("sentence_id")
    if sid is None:
        sid = len(ordered)
    if sid not in texts:
        ordered.append(sid)
    texts[sid] = text.strip()


def _fail_reason(message: dict) -> str:
    header = message.get("header") or {}
    return str(header.get("error_message") or header.get("error_code") or "未知错误")


# ---------------------------------------------------------------------- 主入口
def transcribe_wav(path: Path, *, api_key: str, model: str,
                   url: str = DEFAULT_WS_URL, timeout: float = 120.0,
                   language: str = "zh") -> str:
    """把一个 wav 切片流式识别成文字。"""
    path = Path(path)
    pcm, rate = read_pcm16(path)
    if not pcm:
        return ""
    task_id = uuid.uuid4().hex
    deadline = time.monotonic() + max(10.0, float(timeout))
    ws = _open_socket(url, api_key, timeout=min(30.0, max(10.0, float(timeout))))
    ordered: list = []
    texts: dict = {}
    try:
        ws.settimeout(1.0)
        ws.send(json.dumps({
            "header": {"action": "run-task", "task_id": task_id, "streaming": "duplex"},
            "payload": {
                "task_group": "audio",
                "task": "asr",
                "function": "recognition",
                "model": model,
                "parameters": {
                    "format": "pcm",
                    "sample_rate": rate,
                    "language_hints": [language or "zh"],
                },
                "input": {},
            },
        }, ensure_ascii=False))

        while True:
            message = _recv_json(ws, deadline)
            event = (message.get("header") or {}).get("event")
            if event == "task-started":
                break
            if event == _EVENT_FAILED:
                raise AIClientError("实时语音识别启动失败：" + _fail_reason(message))
            _collect(message, ordered, texts)

        for offset in range(0, len(pcm), _FRAME_BYTES):
            ws.send_binary(pcm[offset:offset + _FRAME_BYTES])
            _drain(ws, ordered, texts)

        ws.send(json.dumps({
            "header": {"action": "finish-task", "task_id": task_id, "streaming": "duplex"},
            "payload": {"input": {}},
        }, ensure_ascii=False))

        while True:
            message = _recv_json(ws, deadline)
            event = (message.get("header") or {}).get("event")
            if event == _EVENT_DONE:
                break
            if event == _EVENT_FAILED:
                raise AIClientError("实时语音识别失败：" + _fail_reason(message))
            if event == "result-generated":
                _collect(message, ordered, texts)
    finally:
        try:
            ws.close()
        except Exception:
            pass
    return "".join(texts[sid] for sid in ordered).strip()


def transcribe_chunk_realtime(client: Any, wav_path: Path, cfg: ModelEndpoint,
                              hint: str = "") -> str:
    """与 ai.asr.transcribe_chunk 同签名的实时识别入口。"""
    model = (cfg.asr_model or "").strip()
    if not model:
        raise AIClientError("未配置语音识别模型名称（设置 → 模型 API）")
    api_key = cfg.effective_asr_api_key or cfg.api_key
    if not api_key:
        raise AIClientError("未配置语音识别 API KEY（设置 → 模型 API）")
    language = "zh"
    if hint and hint.strip() in ("en", "zh", "ja", "ko", "yue"):
        language = hint.strip()
    return transcribe_wav(
        Path(wav_path),
        api_key=api_key,
        model=model,
        url=resolve_ws_url(cfg),
        timeout=float(cfg.timeout_s or 120),
        language=language,
    )
