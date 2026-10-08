"""老师布置的任务识别 + 标题去重。

契约见 INTERFACES.md 2.5：
- 输出 JSON 数组，字段 title/detail/task_type/requirements/deadline/confidence/evidence/subject；
- 用 utils.parse_json_loose 解析，解析失败返回空列表（不抛错，只记日志）；
- 与 known_tasks 标题相似度 > 0.75 视为同一条，不重复产出；
- title_similarity(a, b) 用字符 2-gram 的 Jaccard 相似度（纯函数）。
说明：ClassTask 没有 evidence 字段，老师原话会附加进 detail（不影响其它字段）。
"""
from __future__ import annotations

import re
from typing import TYPE_CHECKING

from ..config.models import ModelEndpoint, PersonalInfo
from ..core.models import ClassTask
from ..utils import clip, compact_join, get_logger, normalize_ws, parse_json_loose, strip_code_fence
from .client import as_float, as_str_list, as_text
from .prompts import PROMPT_TASK_USER, get_prompt, render

if TYPE_CHECKING:
    from .client import LLMClient

logger = get_logger(__name__)

DEDUPE_THRESHOLD = 0.75           # 标题相似度超过它视为同一条任务
DEFAULT_CONFIDENCE = 0.5          # 模型没给 confidence 时的默认值
TASK_TYPES: tuple[str, ...] = ("作业", "练习", "背诵", "实验", "讨论", "阅读", "其他")
NOTES_LIMIT = 6000
TRANSCRIPT_LIMIT = 4000
TITLE_LIMIT = 120

_KEEP_RE = re.compile(r"[^0-9a-z\u4e00-\u9fff]+")


# ---------------------------------------------------------------------- 纯函数

def _normalize_title(text: str) -> str:
    """去掉空白、标点、大小写差异，只留下中英文与数字。"""
    return _KEEP_RE.sub("", (text or "").lower())


def _bigrams(text: str) -> set[str]:
    normalized = _normalize_title(text)
    if not normalized:
        return set()
    if len(normalized) == 1:
        return {normalized}
    return {normalized[i:i + 2] for i in range(len(normalized) - 1)}


def title_similarity(a: str, b: str) -> float:
    """字符 2-gram 的 Jaccard 相似度，范围 0-1；任一为空返回 0。"""
    grams_a = _bigrams(a)
    grams_b = _bigrams(b)
    if not grams_a or not grams_b:
        return 0.0
    union = grams_a | grams_b
    if not union:
        return 0.0
    return len(grams_a & grams_b) / len(union)


# ---------------------------------------------------------------------- 主入口

def extract_tasks(client: LLMClient, cfg: ModelEndpoint, person: PersonalInfo,
                  notes_markdown: str, recent_transcript: str = "",
                  known_tasks: list[ClassTask] | None = None,
                  subject: str = "") -> list[ClassTask]:
    """从笔记 / 转写里提取老师布置的任务；畸形输出一律当成「没有任务」。"""
    notes = (notes_markdown or "").strip()
    recent = (recent_transcript or "").strip()
    if not notes and not recent:
        return []   # 没有素材就不调用模型，避免凭空捏造任务

    known = [t for t in (known_tasks or []) if as_text(getattr(t, "title", ""))]
    default_subject = as_text(subject) or as_text(getattr(person, "subject", ""))
    user = render(
        PROMPT_TASK_USER,
        subject=default_subject or "（未指定）",
        student=_person_brief(person),
        notes=clip(notes, NOTES_LIMIT) or "（暂无）",
        transcript=clip(recent, TRANSCRIPT_LIMIT) or "（暂无）",
        known="\n".join(f"- {t.title}" for t in known) or "（暂无）",
    )
    raw = client.chat(system=get_prompt("task_extract"), user=user, model=cfg.chat_model,
                      json_mode=True, temperature=0.1)

    seen_titles: list[str] = [t.title for t in known]
    found: list[ClassTask] = []
    for item in _coerce_items(raw):
        task = _to_task(item, default_subject)
        if task is None:
            continue
        duplicate = next((title for title in seen_titles
                          if title_similarity(task.title, title) > DEDUPE_THRESHOLD), "")
        if duplicate:
            logger.info("任务与已有任务重复，已忽略：%s ≈ %s", task.title, duplicate)
            continue
        seen_titles.append(task.title)
        found.append(task)
    logger.info("任务识别完成：新增 %d 条", len(found))
    return found


# ---------------------------------------------------------------------- 解析与转换

def _coerce_items(raw: str) -> list[dict]:
    """把模型输出变成 dict 列表；任何畸形输入都返回空列表，绝不抛错。"""
    text = strip_code_fence(raw or "").strip()
    if not text:
        logger.warning("任务识别返回空内容，按「没有任务」处理")
        return []
    try:
        data = parse_json_loose(text)
    except Exception as exc:   # noqa: BLE001 - 模型输出不可控
        logger.warning("任务识别结果不是合法 JSON，按「没有任务」处理：%s", exc)
        return []
    if isinstance(data, dict):
        for key in ("tasks", "items", "data", "result", "list"):
            value = data.get(key)
            if isinstance(value, list):
                data = value
                break
        else:
            data = [data] if as_text(data.get("title")) else []
    if not isinstance(data, list):
        logger.warning("任务识别结果不是 JSON 数组，按「没有任务」处理")
        return []
    return [item for item in data if isinstance(item, dict)]


def _to_task(item: dict, default_subject: str) -> ClassTask | None:
    title = clip(normalize_ws(as_text(item.get("title"))), TITLE_LIMIT)
    if not title:
        logger.warning("任务缺少标题，已忽略：%s", item)
        return None
    detail = as_text(item.get("detail"))
    evidence = as_text(item.get("evidence"))
    if evidence and evidence not in detail:
        detail = compact_join([detail, f"老师原话：{evidence}"], sep="\n\n")
    task_type = as_text(item.get("task_type"))
    if task_type not in TASK_TYPES:
        task_type = "作业"
    return ClassTask(
        title=title,
        detail=detail,
        subject=as_text(item.get("subject")) or default_subject,
        task_type=task_type,
        requirements=as_str_list(item.get("requirements")),
        deadline=clip(as_text(item.get("deadline")), 60),
        confidence=_coerce_confidence(item.get("confidence")),
    )


def _coerce_confidence(value: object, default: float = DEFAULT_CONFIDENCE) -> float:
    """置信度统一到 0-1；模型偶尔会给 85 这种百分数，自动折算。"""
    num = as_float(value, default=default)
    if 2.0 <= num <= 100.0:      # 模型偶尔给 85 这种百分数
        num /= 100.0
    return min(1.0, max(0.0, num))


def _person_brief(person: PersonalInfo) -> str:
    if person is None:
        return "（未填写）"
    bits = [person.name, person.grade, person.class_name]
    return " ".join(bit for bit in bits if bit) or "（未填写）"
