"""模型层异常：GUI / 引擎只需要认这一种异常类型。

契约（INTERFACES.md 第 2 节）：模型层失败一律抛 AIClientError，含 message / status / retryable。
安全要求：错误信息里**绝不能包含 API KEY**，本模块统一做脱敏（显式替换 + 常见密钥形态兜底）。
"""
from __future__ import annotations

import re
from typing import Iterable

# 兜底脱敏：即使调用方忘了把自己的 key 传进来，也能识别常见密钥形态。
_SECRET_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\bsk-[A-Za-z0-9\-_]{6,}"),
    re.compile(r"(?i)\b(bearer\s+)[A-Za-z0-9\-_.=]{6,}"),
    re.compile(r"(?i)\b((?:api[-_]?key|access[-_]?token|secret|authorization)\s*[:=]\s*)(\S+)"),
)

# 常见状态码的中文解释，GUI 直接展示 user_hint()。
_STATUS_HINTS: dict[int, str] = {
    400: "请求被拒绝（400）：可能是模型名不支持该功能，或参数不被接受",
    401: "API KEY 无效或已过期（401），请在设置中重新填写",
    403: "没有访问权限（403）：请确认该模型已经开通",
    404: "接口或模型不存在（404）：请检查服务地址与模型名称",
    413: "请求内容过大（413）：请缩短输入，或换用上下文更大的模型",
    422: "参数不被接受（422）：请检查模型名称与调用参数",
    429: "调用过于频繁或额度不足（429），稍后会自动重试",
    500: "模型服务内部错误（500），稍后会自动重试",
    502: "模型服务网关错误（502），稍后会自动重试",
    503: "模型服务暂不可用（503），稍后会自动重试",
    504: "模型服务超时（504），稍后会自动重试",
}


def redact_secrets(text: object, secrets: Iterable[str] = ()) -> str:
    """把文本里的 API KEY / Bearer token 替换成 ***。"""
    out = "" if text is None else str(text)
    for secret in secrets:
        token = (secret or "").strip()
        if len(token) >= 4:
            out = out.replace(token, "***")
    for pattern in _SECRET_PATTERNS:
        if pattern.groups:
            out = pattern.sub(lambda m: m.group(1) + "***", out)
        else:
            out = pattern.sub("***", out)
    return out


class AIClientError(Exception):
    """模型调用失败。message 已经脱敏，可以直接展示给用户。"""

    def __init__(self, message: str, status: int | None = None,
                 retryable: bool = False, *, detail: str = "") -> None:
        clean = redact_secrets(message).strip() or "模型调用失败"
        super().__init__(clean)
        self.message = clean
        self.status = int(status) if status is not None else None
        self.retryable = bool(retryable)
        self.detail = redact_secrets(detail)

    def __str__(self) -> str:
        return self.message

    def user_hint(self) -> str:
        """在原始信息后补一句中文原因，供 GUI「测试连接」等场景展示。"""
        hint = _STATUS_HINTS.get(self.status) if self.status is not None else None
        return f"{self.message}（{hint}）" if hint else self.message
