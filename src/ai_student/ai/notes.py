"""课堂笔记：把新增的转写 / 板书合并进已有笔记。"""
from __future__ import annotations

import re
from typing import TYPE_CHECKING, Iterable

from ..config.models import ModelEndpoint, PersonalInfo
from ..utils import clip, compact_join, get_logger, normalize_ws, strip_code_fence
from .prompts import PROMPT_NOTES_USER, get_prompt, render

if TYPE_CHECKING:
    from .client import LLMClient

logger = get_logger(__name__)

NOTES_TITLE = "# 课堂笔记"
PREVIOUS_LIMIT = 7000
TRANSCRIPT_LIMIT = 6000
BOARD_LIMIT = 4000


def merge_notes(client: LLMClient, previous_markdown: str, transcript_lines: list[str],
                board_lines: list[str], cfg: ModelEndpoint, person: PersonalInfo,
                subject: str = "") -> str:
    """返回**完整**的新版课堂笔记 Markdown（previous_markdown 为空时从零生成）。"""
    previous = (previous_markdown or "").strip()
    user = render(
        PROMPT_NOTES_USER,
        subject=(subject or person.subject or "（未指定）"),
        student=describe_person(person),
        previous=clip(previous, PREVIOUS_LIMIT) if previous else "（无，本次课从零开始记录）",
        transcript=_as_bullets(transcript_lines, TRANSCRIPT_LIMIT) or "（本轮没有新的语音内容）",
        board=_as_bullets(board_lines, BOARD_LIMIT) or "（本轮没有新的板书内容）",
    )
    text = client.chat(system=get_prompt("notes_merge"), user=user, model=cfg.chat_model,
                       temperature=0.2, max_tokens=cfg.max_tokens)
    merged = strip_code_fence(text or "").strip()
    if not merged:
        logger.warning("笔记合并返回空内容，保留上一版笔记")
        return previous
    return ensure_title(merged)


def _as_bullets(lines: Iterable[str], limit: int) -> str:
    items = [normalize_ws(line) for line in (lines or []) if line and str(line).strip()]
    return clip(compact_join([f"- {item}" for item in items if item]), limit)


def describe_person(person: PersonalInfo) -> str:
    """给提示词用的一行学生信息。"""
    if person is None:
        return "（未填写）"
    bits = [person.name, person.grade, person.class_name, person.school]
    return " ".join(bit for bit in bits if bit) or "（未填写）"


def ensure_title(markdown: str) -> str:
    """保证笔记有一级标题（模型偶尔会漏掉标题）。"""
    text = (markdown or "").strip()
    if not text:
        return NOTES_TITLE
    if re.search(r"^[ \t]*#\s+\S", text, re.M):
        return text
    return f"{NOTES_TITLE}\n\n{text}"
