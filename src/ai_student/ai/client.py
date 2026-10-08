"""OpenAI 兼容接口客户端（对话 / 视觉 / 语音识别入口）。

契约见 INTERFACES.md 2.1：
- 同步阻塞，不做 UI 操作，失败抛 AIClientError；
- 图片以 data URL（data:image/jpeg;base64,...）放进 image_url content part；
- 自动重试 cfg.retries 次，指数退避 0.8s / 1.6s / 3.2s，只对超时 / 429 / 5xx 重试；
- 错误信息里绝不包含 API KEY。
本模块还提供模型输出取值小工具（as_text / as_str_list / as_float），供其它子模块复用。
"""
from __future__ import annotations

import base64
import json
import re
import time
from pathlib import Path
from typing import Any, Callable, Sequence

import openai

from ..config.models import ModelEndpoint
from ..utils import get_logger
from .errors import AIClientError, redact_secrets

logger = get_logger(__name__)

RETRY_BASE_DELAY = 0.8      # 指数退避基数（秒）：0.8 / 1.6 / 3.2 ...
RETRY_MAX_DELAY = 30.0
_PLACEHOLDER_KEY = "not-configured"   # openai SDK 要求 api_key 非空，未配置时占位


def _sleep(seconds: float) -> None:
    """单独抽出来，便于离线单测替换。"""
    time.sleep(seconds)


# ---------------------------------------------------------------------- 图片处理

def guess_mime(data: bytes) -> str:
    """按文件头猜图片类型，默认按 JPEG 处理。"""
    if data.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if data[:6] in (b"GIF87a", b"GIF89a"):
        return "image/gif"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    if data.startswith(b"BM"):
        return "image/bmp"
    return "image/jpeg"


def _read_image(image: bytes | Path | str) -> tuple[bytes, str]:
    if isinstance(image, (bytes, bytearray, memoryview)):
        raw = bytes(image)
    else:
        path = Path(image)
        if not path.exists():
            raise AIClientError(f"图片文件不存在：{path}")
        try:
            raw = path.read_bytes()
        except OSError as exc:
            raise AIClientError(f"读取图片失败：{path}（{exc}）") from exc
    if not raw:
        raise AIClientError("图片内容为空")
    return raw, guess_mime(raw)


def to_data_url(image: bytes | Path | str) -> str:
    """把图片转成 data URL（契约要求的 content part 形式）。"""
    raw, mime = _read_image(image)
    return f"data:{mime};base64,{base64.b64encode(raw).decode('ascii')}"


def _user_content(user: str, images: Sequence[bytes | Path] | None) -> Any:
    text = user or ""
    if not images:
        return text
    parts: list[dict[str, Any]] = []
    if text.strip():
        parts.append({"type": "text", "text": text})
    for image in images:
        parts.append({"type": "image_url", "image_url": {"url": to_data_url(image)}})
    return parts


# ---------------------------------------------------------------------- 响应取值

def _get(obj: Any, key: str) -> Any:
    if isinstance(obj, dict):
        return obj.get(key)
    return getattr(obj, key, None)


def _part_text(part: Any) -> str:
    if isinstance(part, str):
        return part
    text = _get(part, "text")
    if text is None:
        text = _get(_get(part, "delta"), "content")
    return str(text or "")


def truncated_by_reasoning(resp: Any) -> bool:
    """推理模型（如 DeepSeek 的 flash 系列）把 max_tokens 全花在思考过程上的情况。

    这时 finish_reason 是 length、content 是空、而 reasoning_content 有内容。
    如果不识别出来，调用方只会拿到空字符串，最后表现为莫名其妙的"模型没有返回内容"。
    """
    choices = _get(resp, "choices") or []
    if not choices:
        return False
    choice = choices[0]
    message = _get(choice, "message")
    content = _get(message, "content")
    reasoning = _get(message, "reasoning_content") or _get(message, "reasoning")
    return (
        not str(content or "").strip()
        and bool(str(reasoning or "").strip())
        and str(_get(choice, "finish_reason") or "") == "length"
    )


def response_tool_calls(resp: Any) -> list[dict[str, Any]]:
    """从响应里取出 tool_calls，统一成 [{"id","name","arguments"(dict),"raw"}]。"""
    choices = _get(resp, "choices") or []
    if not choices:
        return []
    message = _get(choices[0], "message")
    calls = _get(message, "tool_calls") or []
    out: list[dict[str, Any]] = []
    for call in calls:
        fn = _get(call, "function")
        raw = _get(fn, "arguments")
        if isinstance(raw, str):
            try:
                args = json.loads(raw) if raw.strip() else {}
            except Exception:
                args = {}
        elif isinstance(raw, dict):
            args = dict(raw)
        else:
            args = {}
        out.append({
            "id": str(_get(call, "id") or ""),
            "name": str(_get(fn, "name") or ""),
            "arguments": args if isinstance(args, dict) else {},
            "raw": raw if isinstance(raw, str) else "",
        })
    return out


def assistant_message(resp: Any) -> dict[str, Any]:
    """把响应里的 assistant 消息还原成可回灌的消息字典（保留 tool_calls）。"""
    choices = _get(resp, "choices") or []
    message = _get(choices[0], "message") if choices else None
    content = _get(message, "content") or ""
    if not isinstance(content, str):
        content = str(content)
    out: dict[str, Any] = {"role": "assistant", "content": content}
    calls = _get(message, "tool_calls") or []
    if calls:
        out["tool_calls"] = [
            {
                "id": str(_get(c, "id") or ""),
                "type": "function",
                "function": {
                    "name": str(_get(_get(c, "function"), "name") or ""),
                    "arguments": _get(_get(c, "function"), "arguments") or "{}",
                },
            }
            for c in calls
        ]
    return out


def response_text(resp: Any) -> str:
    """从 chat completion 响应里取正文，兼容 content 为字符串 / 分片数组 / 字典。"""
    choices = _get(resp, "choices") or []
    if not choices:
        raise AIClientError("模型返回内容为空（choices 为空）")
    first = choices[0]
    if str(_get(first, "finish_reason") or "") == "content_filter":
        raise AIClientError("内容被模型的安全策略拦截，请调整输入后重试")
    message = _get(first, "message")
    content = _get(message, "content") if message is not None else None
    if content is None:
        content = _get(first, "text")
    if isinstance(content, list):
        content = "".join(_part_text(part) for part in content)
    return str(content or "").strip()


# ---------------------------------------------------------------------- 输出强转

def as_text(value: Any) -> str:
    """把模型返回的任意值转成去空白的文本。"""
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, bool):
        return "是" if value else "否"
    if isinstance(value, (list, tuple, set)):
        return "\n".join(part for part in (as_text(x) for x in value) if part)
    if isinstance(value, dict):
        pairs = [f"{k}：{as_text(v)}" for k, v in value.items()]
        return "；".join(p for p in pairs if not p.endswith("："))
    return str(value).strip()


def as_str_list(value: Any) -> list[str]:
    """把模型返回的任意值转成字符串列表（字符串会按行 / 分号拆开）。"""
    if value is None:
        return []
    if isinstance(value, str):
        parts = re.split(r"[\n;；]+", value)
        return [p.strip(" -·\t") for p in parts if p.strip(" -·\t")]
    if isinstance(value, dict):
        return [f"{k}：{as_text(v)}" for k, v in value.items() if as_text(v)]
    if isinstance(value, (list, tuple, set)):
        return [text for text in (as_text(item) for item in value) if text]
    text = as_text(value)
    return [text] if text else []


def as_float(value: Any, default: float = 0.0, lo: float | None = None,
             hi: float | None = None) -> float:
    """把模型返回的任意值转成浮点数（支持 0.85、85%、1,5 这类写法）。"""
    if isinstance(value, bool):
        num = float(value)
    elif isinstance(value, (int, float)):
        num = float(value)
    else:
        text = as_text(value).replace(",", ".").replace("％", "%")
        if not text:
            return float(default)
        found = re.findall(r"-?\d+(?:\.\d+)?", text)
        if not found:
            return float(default)
        num = float(found[0])
        if text.rstrip().endswith("%"):
            num /= 100.0
    if lo is not None:
        num = max(lo, num)
    if hi is not None:
        num = min(hi, num)
    return float(num)


# ---------------------------------------------------------------------- 客户端

class LLMClient:
    """一个 ModelEndpoint 对应一个实例；openai SDK 自身线程安全。"""

    def __init__(self, cfg: ModelEndpoint) -> None:
        self.cfg = cfg
        self._client = self._build_client(cfg.base_url, cfg.api_key)
        self._asr_client: Any | None = None
        self._vision_client: Any | None = None

    # -------------------------------------------------- 构造
    def _build_client(self, base_url: str, api_key: str) -> Any:
        cfg = self.cfg
        headers = {str(k): str(v) for k, v in (cfg.extra_headers or {}).items()}
        timeout = float(cfg.timeout_s) if cfg.timeout_s and cfg.timeout_s > 0 else None
        # max_retries=0：重试次数完全由 request() 控制，避免 SDK 与业务双重退避。
        return openai.OpenAI(
            base_url=base_url or None,
            api_key=api_key or _PLACEHOLDER_KEY,
            timeout=timeout,
            max_retries=0,
            default_headers=headers or None,
        )

    def asr_client(self) -> Any:
        """语音识别客户端：asr_base_url / asr_api_key 未单独配置时复用主客户端。"""
        cfg = self.cfg
        same_endpoint = (cfg.effective_asr_base_url == cfg.base_url
                         and cfg.effective_asr_api_key == cfg.api_key)
        if same_endpoint:
            return self._client
        if self._asr_client is None:
            self._asr_client = self._build_client(
                cfg.effective_asr_base_url, cfg.effective_asr_api_key
            )
        return self._asr_client

    def vision_client(self) -> Any:
        """视觉客户端：常见组合是"文本用 DeepSeek、看图用通义 VL"。"""
        cfg = self.cfg
        same_endpoint = (cfg.effective_vision_base_url == cfg.base_url
                         and cfg.effective_vision_api_key == cfg.api_key)
        if same_endpoint:
            return self._client
        if self._vision_client is None:
            self._vision_client = self._build_client(
                cfg.effective_vision_base_url, cfg.effective_vision_api_key
            )
        return self._vision_client

    # -------------------------------------------------- 请求 + 重试
    def _secrets(self) -> list[str]:
        return [self.cfg.api_key, self.cfg.asr_api_key, self.cfg.vision_api_key]

    def request(self, call: Callable[[], Any], what: str = "模型请求") -> Any:
        """执行一次 SDK 调用：带指数退避重试，并把异常统一包装成 AIClientError。"""
        attempts = max(0, int(self.cfg.retries or 0)) + 1
        last: AIClientError | None = None
        for attempt in range(attempts):
            try:
                return call()
            except Exception as exc:   # noqa: BLE001 - 统一包装所有 SDK 异常
                err = self._to_error(exc, what)
                last = err
                if not err.retryable or attempt >= attempts - 1:
                    raise err from exc
                delay = min(RETRY_BASE_DELAY * (2 ** attempt), RETRY_MAX_DELAY)
                logger.warning("%s失败（%s），%.1f 秒后重试（第 %d 次）",
                               what, err.message, delay, attempt + 1)
                _sleep(delay)
        raise last if last is not None else AIClientError(f"{what}失败")

    def _to_error(self, exc: BaseException, what: str) -> AIClientError:
        if isinstance(exc, AIClientError):
            return exc
        status = _status_of(exc)
        raw = f"{what}失败：{type(exc).__name__}: {exc}"
        message = redact_secrets(raw, self._secrets()).strip()
        return AIClientError(message, status=status, retryable=_is_retryable(exc, status))

    # -------------------------------------------------- 对话
    def chat(self, system: str, user: str, images: Sequence[bytes | Path] | None = None,
             model: str | None = None, json_mode: bool = False,
             temperature: float | None = None, max_tokens: int | None = None,
             endpoint: str = "chat") -> str:
        messages: list[dict[str, Any]] = []
        if (system or "").strip():
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": _user_content(user, images)})
        return self.chat_messages(messages, model=model, json_mode=json_mode,
                                  temperature=temperature, max_tokens=max_tokens,
                                  endpoint=endpoint)

    def chat_messages(self, messages: list[dict], model: str | None = None,
                      json_mode: bool = False, temperature: float | None = None,
                      max_tokens: int | None = None, endpoint: str = "chat") -> str:
        payload: dict[str, Any] = {
            "model": model or self.cfg.chat_model,
            "messages": messages,
        }
        temp = self.cfg.temperature if temperature is None else temperature
        if temp is not None:
            payload["temperature"] = float(temp)
        limit = int(max_tokens if max_tokens is not None else (self.cfg.max_tokens or 0))
        if limit > 0:
            payload["max_tokens"] = limit
        if json_mode:
            payload["response_format"] = {"type": "json_object"}
        cli = self.vision_client() if endpoint == "vision" else self._client
        what = "画面理解请求" if endpoint == "vision" else "对话请求"
        try:
            resp = self.request(lambda: cli.chat.completions.create(**payload), what)
        except AIClientError as exc:
            # 少数兼容服务不支持 response_format，退回「提示词约束 JSON」再试一次。
            if not json_mode or exc.status not in (400, 404, 422):
                raise
            logger.warning("该服务不接受 response_format=json_object，改用提示词约束重试")
            payload.pop("response_format", None)
            resp = self.request(lambda: cli.chat.completions.create(**payload), what)
        if truncated_by_reasoning(resp):
            raise AIClientError(
                "模型把 max_tokens 全部用在思考过程上，没有输出正式内容；"
                "请在 设置 → 模型 API 调大「单次最大 tokens」后重试"
            )
        return response_text(resp)

    # -------------------------------------------------- 工具调用（function calling）
    def chat_with_tools(self, messages: list[dict], tools: list[dict],
                        model: str | None = None, tool_choice: str = "auto",
                        temperature: float | None = None,
                        max_tokens: int | None = None) -> Any:
        """带工具的一次对话，返回 SDK 原始响应对象（需要读 tool_calls）。

        少数兼容服务不支持 tools 参数，会返回 400/404/422；此时抛出的 AIClientError
        会被上层 ai.agent 捕获并降级为启发式策略，而不是让整条听课链路失败。
        """
        payload: dict[str, Any] = {
            "model": model or self.cfg.chat_model,
            "messages": messages,
            "tools": tools,
            "tool_choice": tool_choice,
        }
        temp = self.cfg.temperature if temperature is None else temperature
        if temp is not None:
            payload["temperature"] = float(temp)
        limit = int(max_tokens if max_tokens is not None else (self.cfg.max_tokens or 0))
        if limit > 0:
            payload["max_tokens"] = limit
        resp = self.request(
            lambda: self._client.chat.completions.create(**payload), "工具调用请求"
        )
        if truncated_by_reasoning(resp):
            raise AIClientError(
                "模型把 max_tokens 全部用在思考过程上，没有输出工具调用；"
                "请在 设置 → 模型 API 调大「单次最大 tokens」后重试"
            )
        return resp

    # -------------------------------------------------- 语音 / 视觉入口
    def transcribe(self, wav_path: Path) -> str:
        from .asr import transcribe_chunk   # 局部导入，避免与 asr 形成循环依赖
        return transcribe_chunk(self, Path(wav_path), self.cfg)

    @property
    def vision_model(self) -> str:
        return self.cfg.effective_vision_model

    # -------------------------------------------------- 连通性
    def ping(self) -> tuple[bool, str]:
        """发一条极短消息验证连通性，供 GUI「测试连接」使用。"""
        cfg = self.cfg
        if not (cfg.base_url or "").strip():
            return False, "未配置模型服务地址（设置 → 模型 API）"
        if not (cfg.api_key or "").strip():
            return False, "未配置 API KEY（设置 → 模型 API）"
        if not (cfg.chat_model or "").strip():
            return False, "未配置对话模型名称（设置 → 模型 API）"
        started = time.perf_counter()
        try:
            text = self.chat(system="你是连通性测试助手。", user="请只回复两个字：正常",
                             temperature=0.0, max_tokens=16)
        except AIClientError as exc:
            return False, exc.user_hint()
        except Exception as exc:   # pragma: no cover - 兜底
            return False, redact_secrets(str(exc), self._secrets())
        elapsed = int((time.perf_counter() - started) * 1000)
        if not text:
            return False, f"模型 {cfg.chat_model} 有响应但内容为空，请检查模型名称"
        return True, f"连接正常：{cfg.chat_model}（{elapsed} ms）"


# ---------------------------------------------------------------------- 异常判定

def _status_of(exc: BaseException) -> int | None:
    for attr in ("status_code", "status", "http_status"):
        value = getattr(exc, attr, None)
        if isinstance(value, bool):
            continue
        if isinstance(value, int):
            return value
        if isinstance(value, str) and value.isdigit():
            return int(value)
    response = getattr(exc, "response", None)
    code = getattr(response, "status_code", None)
    return int(code) if isinstance(code, int) else None


def _is_retryable(exc: BaseException, status: int | None) -> bool:
    """只对 超时 / 连接中断 / 429 / 5xx 重试；4xx（密钥、参数、权限）不重试。"""
    if isinstance(exc, (openai.APITimeoutError, openai.APIConnectionError, TimeoutError)):
        return True
    if status is None:
        return False
    return status == 429 or 500 <= status <= 599
