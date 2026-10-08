"""板书 / 演示画面理解（视觉模型）。"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from ..config.models import ModelEndpoint
from ..utils import clip, get_logger, parse_json_loose, strip_code_fence
from .client import as_str_list, as_text
from .prompts import PROMPT_VISION_USER, get_prompt, render

if TYPE_CHECKING:
    from .client import LLMClient

logger = get_logger(__name__)

FALLBACK_LIMIT = 500   # 模型没按 JSON 输出时，把原文当作摘要保留的长度上限


@dataclass
class BoardAnalysis:
    """一屏演示画面的理解结果。"""

    ocr_text: str = ""      # 画面上的文字（板书 / PPT / 题目），保留原样
    summary: str = ""       # 这一屏在讲什么（1-3 句）
    key_points: list[str] = field(default_factory=list)
    problems: list[str] = field(default_factory=list)   # 画面上出现的题目 / 任务原文
    subject_hint: str = ""


def analyze_board(client: LLMClient, image: bytes | Path, cfg: ModelEndpoint,
                  context: str = "") -> BoardAnalysis:
    """调用视觉模型理解一屏画面；解析失败也不会抛错（退化成纯文本摘要）。"""
    user = render(PROMPT_VISION_USER, context=(context or "").strip() or "（无）")
    text = client.chat(
        system=get_prompt("vision_board"),
        user=user,
        images=[image],
        model=cfg.effective_vision_model,
        json_mode=True,
        temperature=0.1,
        endpoint="vision",
    )
    return parse_board_analysis(text)


def parse_board_analysis(text: str) -> BoardAnalysis:
    """把模型输出解析成 BoardAnalysis；不是合法 JSON 时把原文当作摘要。"""
    raw = strip_code_fence(text or "").strip()
    if not raw:
        return BoardAnalysis()
    try:
        data = parse_json_loose(raw)
    except Exception:   # noqa: BLE001 - 模型输出不可控，不能让它中断听课流程
        logger.warning("板书理解结果不是合法 JSON，退化为纯文本摘要")
        return BoardAnalysis(summary=clip(raw, FALLBACK_LIMIT))
    if data is None:
        return BoardAnalysis()
    if isinstance(data, list):
        data = data[0] if data and isinstance(data[0], dict) else None
    if not isinstance(data, dict):
        logger.warning("板书理解结果不是 JSON 对象，退化为纯文本摘要")
        return BoardAnalysis(summary=clip(raw, FALLBACK_LIMIT))
    return BoardAnalysis(
        ocr_text=as_text(data.get("ocr_text")),
        summary=as_text(data.get("summary")),
        key_points=as_str_list(data.get("key_points")),
        problems=as_str_list(data.get("problems")),
        subject_hint=as_text(data.get("subject_hint")),
    )
