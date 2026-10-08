"""语音识别：按 ModelEndpoint.asr_kind 分派三种模式。

契约见 INTERFACES.md 2.2：
- openai_transcription：client.audio.transcriptions.create(model=cfg.asr_model, file=...)
  （Whisper / SenseVoice 这类走这条，hint 作为转写提示词传入）
- openai_audio：chat.completions，content 里放 input_audio（Qwen qwen3-asr-flash / GLM 走这条）
- none：抛 AIClientError("未配置语音识别模型")
空音频（全静音或者过短）直接返回空字符串，不调用 API。
"""
from __future__ import annotations

import base64
import wave
from array import array
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ..config.models import ModelEndpoint
from ..utils import get_logger
from .client import response_text
from .errors import AIClientError
from .prompts import PROMPT_ASR

if TYPE_CHECKING:
    from .client import LLMClient

logger = get_logger(__name__)

SILENCE_PEAK = 1e-4        # 归一化峰值阈值：低于它视为全静音
MIN_AUDIO_SECONDS = 0.30   # 短于它的片段视为空音频
HINT_LIMIT = 800


# ---------------------------------------------------------------------- 空音频判定

def probe_wav(path: Path) -> tuple[float, float | None]:
    """返回 (时长秒, 归一化峰值)；无法解析时峰值为 None（当作未知，交给 API 判断）。"""
    try:
        with wave.open(str(path), "rb") as handle:
            width = handle.getsampwidth()
            rate = handle.getframerate() or 1
            frames = handle.readframes(handle.getnframes())
            duration = handle.getnframes() / float(rate)
    except Exception as exc:   # noqa: BLE001 - 损坏文件不应该让采集流程崩掉
        logger.debug("无法解析 WAV 文件 %s：%s", path, exc)
        return 0.0, None
    if not frames:
        return duration, 0.0
    if width != 2:
        # 非 16bit：只能粗略判断是否全 0
        return duration, 0.0 if not any(frames) else None
    if len(frames) % 2:
        frames = frames[:-1]
    samples = array("h")
    samples.frombytes(frames)
    peak = max((abs(int(s)) for s in samples), default=0)
    return duration, peak / 32768.0


def is_effectively_silent(path: Path) -> bool:
    """全静音 / 过短的音频返回 True（用于跳过 API 调用）。"""
    duration, peak = probe_wav(path)
    if duration < MIN_AUDIO_SECONDS:
        return True
    if peak is None:
        return False
    return peak <= SILENCE_PEAK


# ---------------------------------------------------------------------- 主入口

def transcribe_chunk(client: LLMClient, wav_path: Path, cfg: ModelEndpoint,
                     hint: str = "") -> str:
    """识别一个音频切片，返回去空白后的文本；全静音直接返回空字符串。"""
    kind = (cfg.asr_kind or "none").strip()
    if kind == "none":
        raise AIClientError("未配置语音识别模型（请在 设置 → 模型 API 选择支持语音识别的服务）")
    if kind not in ("openai_transcription", "openai_audio"):
        raise AIClientError(f"不支持的语音识别类型：{kind}")

    path = Path(wav_path)
    if not path.exists():
        raise AIClientError(f"音频文件不存在：{path}")
    if is_effectively_silent(path):
        logger.info("音频 %s 判定为静音或者过短，跳过语音识别", path.name)
        return ""
    try:
        data = path.read_bytes()
    except OSError as exc:
        raise AIClientError(f"读取音频失败：{path}（{exc}）") from exc
    if not data:
        return ""

    if kind == "openai_transcription":
        text = _transcribe_via_transcriptions(client, cfg, data, hint)
    else:
        text = _transcribe_via_chat(client, cfg, data, hint)
    return (text or "").strip()


# ---------------------------------------------------------------------- 两种实现

def _require_asr_model(cfg: ModelEndpoint) -> str:
    model = (cfg.asr_model or "").strip()
    if not model:
        raise AIClientError("未配置语音识别模型名称（设置 → 模型 API）")
    return model


def _transcribe_via_transcriptions(client: LLMClient, cfg: ModelEndpoint,
                                   data: bytes, hint: str) -> str:
    cli = client.asr_client()
    payload: dict[str, Any] = {
        "model": _require_asr_model(cfg),
        "file": ("audio.wav", data, "audio/wav"),
    }
    text_hint = (hint or "").strip()[:HINT_LIMIT]
    if text_hint:
        payload["prompt"] = text_hint
    try:
        resp = client.request(lambda: cli.audio.transcriptions.create(**payload), "语音识别")
    except AIClientError as exc:
        # 部分兼容服务不接受 prompt 参数，去掉提示词再试一次（最多多花一次调用）。
        if "prompt" not in payload or exc.status not in (400, 404, 422):
            raise
        logger.warning("该服务的转写接口不接受 prompt 参数，改为不带术语提示重试")
        payload.pop("prompt", None)
        resp = client.request(lambda: cli.audio.transcriptions.create(**payload), "语音识别")
    return _text_of(resp)


def _transcribe_via_chat(client: LLMClient, cfg: ModelEndpoint,
                         data: bytes, hint: str) -> str:
    cli = client.asr_client()
    system = PROMPT_ASR
    text_hint = (hint or "").strip()[:HINT_LIMIT]
    if text_hint:
        system = f"{system}\n\n【本课术语提示（只用于帮助识别，不要解释）】\n{text_hint}"
    encoded = base64.b64encode(data).decode("ascii")
    payload: dict[str, Any] = {
        "model": _require_asr_model(cfg),
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": [
                {"type": "input_audio", "input_audio": {"data": encoded, "format": "wav"}},
            ]},
        ],
        "temperature": 0.0,
    }
    try:
        resp = client.request(lambda: cli.chat.completions.create(**payload), "语音识别")
    except AIClientError as exc:
        # 少数 ASR 模型不接受 temperature 参数，去掉后再试一次。
        if "temperature" not in payload or exc.status not in (400, 404, 422):
            raise
        logger.warning("该 ASR 模型不接受 temperature 参数，改为不带 temperature 重试")
        payload.pop("temperature", None)
        resp = client.request(lambda: cli.chat.completions.create(**payload), "语音识别")
    return _text_of(resp)


def _text_of(resp: Any) -> str:
    """转写响应取文本：优先 text 字段，其次按 chat completion 解析。"""
    text = resp.get("text") if isinstance(resp, dict) else getattr(resp, "text", None)
    if text is not None:
        return str(text).strip()
    return response_text(resp)
