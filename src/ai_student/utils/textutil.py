"""文本与 JSON 处理小工具。"""
from __future__ import annotations

import json
import re

_FENCE_RE = re.compile(r"^```[a-zA-Z0-9_+-]*\s*\n(.*?)\n?```\s*$", re.S)


def strip_code_fence(text: str) -> str:
    t = (text or "").strip()
    m = _FENCE_RE.match(t)
    return m.group(1).strip() if m else t


def parse_json_loose(text: str) -> object:
    """尽量从模型输出中提取 JSON（容忍代码块、前后废话、尾逗号）。"""
    if not text:
        raise ValueError("empty")
    t = strip_code_fence(text)
    try:
        return json.loads(t)
    except Exception:
        pass
    # 找到第一个 { 或 [ 与其匹配的结尾
    for opener, closer in (("{", "}"), ("[", "]")):
        start = t.find(opener)
        if start < 0:
            continue
        depth = 0
        in_str = False
        esc = False
        for i in range(start, len(t)):
            ch = t[i]
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
                continue
            if ch == '"':
                in_str = True
            elif ch == opener:
                depth += 1
            elif ch == closer:
                depth -= 1
                if depth == 0:
                    candidate = t[start:i + 1]
                    candidate = re.sub(r",\s*([}\]])", r"\1", candidate)
                    try:
                        return json.loads(candidate)
                    except Exception:
                        break
    raise ValueError(f"无法解析 JSON: {text[:200]}")


def normalize_ws(text: str) -> str:
    return re.sub(r"[ \t]+", " ", (text or "").replace("\r\n", "\n")).strip()


def compact_join(parts: list[str], sep: str = "\n") -> str:
    return sep.join(p for p in (x.strip() for x in parts) if p)


def clip(text: str, limit: int = 4000, tail: str = "…（已截断）") -> str:
    t = text or ""
    return t if len(t) <= limit else t[:limit] + tail


def hhmmss(seconds: float) -> str:
    s = max(0, int(seconds))
    return f"{s // 3600:02d}:{(s % 3600) // 60:02d}:{s % 60:02d}"
