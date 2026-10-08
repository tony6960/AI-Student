"""任务作答：先理解题意，再给最终正文，最后自检。

契约见 INTERFACES.md 2.6：
- 内部用 ### 题目理解 / ### 解答 / ### 自检 三段解析，answer 只要最终正文（不含分段标记）；
- 提示词要求先理解题意、不确定写假设、按 person.answer_style 决定详略、按 person.sign_artifacts 署名；
- 数学 / 理科给完整过程，作文满足字数，编程题给可运行代码。
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, TYPE_CHECKING

from ..config.models import ModelEndpoint, OutputConfig, PersonalInfo
from ..config.paths import sanitize_filename
from ..core.models import ClassTask
from ..utils import clip, get_logger, normalize_ws, strip_code_fence
from .prompts import PROMPT_SOLVE_USER, SIGN_ARTIFACTS_RULE, get_prompt, render

if TYPE_CHECKING:
    from .client import LLMClient

logger = get_logger(__name__)

CONTEXT_LIMIT = 8000
DEFAULT_TITLE = "作业"
TITLE_MAX_LEN = 60

# 段落标题：容忍 ###、**加粗**、全角冒号等写法
SECTION_MARKER_RE = re.compile(
    r"^[ \t]*(?:#{1,6}[ \t]*)?(?:\*\*|__)?[ \t]*(题目理解|解答|自检)[ \t]*(?:\*\*|__)?[ \t]*[:：]?[ \t]*(?:\*\*|__)?[ \t]*$",
    re.M,
)
NEEDS_HUMAN_RE = re.compile(r"结论\s*[:：]\s*(?:需要人工|需人工|需要复核|需人工复核|需要老师|需老师)")
PASS_RE = re.compile(r"结论\s*[:：]\s*(?:通过|合格|正确|无误)")
NEEDS_HUMAN_WORDS = (
    "需要人工复核", "需要人工", "需要老师确认", "题目信息不足", "题目不完整",
    "缺少必要信息", "无法确定答案", "无法作答",
)


@dataclass
class SolveResult:
    """一次作业作答的结果。"""

    answer: str = ""            # 最终作业正文（Markdown，可直接给学生看）
    workpaper: str = ""         # 解题过程 / 思路（Markdown），可为空
    file_title: str = ""        # 建议的文件名（不含扩展名，不带路径分隔符）
    self_check: str = ""        # 自检结论
    needs_human: bool = False   # 是否需要人工复核


def solve_task(client: LLMClient, cfg: ModelEndpoint, person: PersonalInfo,
               task: ClassTask, context: str, output_cfg: OutputConfig) -> SolveResult:
    """完成一项作业：context 是引擎提供的课堂资料（笔记 + 相关转写 + 板书要点）。"""
    system = get_prompt("solve")
    if bool(getattr(person, "sign_artifacts", True)):
        signature = _signature(person)
        if signature:
            system = f"{system}\n\n{render(SIGN_ARTIFACTS_RULE, signature=signature)}"
    values = _person_values(person)
    values.update(
        subject=_value(task, "subject") or values["subject"] or "（未指定）",
        task_title=_value(task, "title") or "（未命名任务）",
        task_type=_value(task, "task_type") or "作业",
        task_detail=_value(task, "detail") or _value(task, "title") or "（题目原文缺失，请按标题理解题意）",
        requirements="；".join(_value_list(task, "requirements")) or "（题目没有额外要求）",
        deadline=_value(task, "deadline") or "（未提及）",
        context=clip((context or "").strip(), CONTEXT_LIMIT) or "（暂无课堂资料）",
        output_preference=_output_preference(output_cfg),
    )
    text = client.chat(system=system, user=render(PROMPT_SOLVE_USER, **values),
                       model=cfg.chat_model, json_mode=False,
                       temperature=cfg.temperature, max_tokens=cfg.max_tokens)
    result = parse_solve_output(text, _value(task, "title"), values["subject"])
    logger.info("作答完成：%s（正文 %d 字，需人工复核=%s）",
                result.file_title, len(result.answer), result.needs_human)
    return result


# ---------------------------------------------------------------------- 解析

def split_sections(text: str) -> dict[str, str]:
    """按 ### 题目理解 / ### 解答 / ### 自检 切分；找不到段落时返回空字典。"""
    raw = strip_code_fence(text or "").strip()
    marks = list(SECTION_MARKER_RE.finditer(raw))
    if not marks:
        return {}
    sections: dict[str, str] = {}
    for index, mark in enumerate(marks):
        end = marks[index + 1].start() if index + 1 < len(marks) else len(raw)
        name = mark.group(1)
        body = raw[mark.end():end].strip()
        sections[name] = f"{sections[name]}\n{body}".strip() if name in sections else body
    return sections


def parse_solve_output(text: str, task_title: str = "", subject: str = "") -> SolveResult:
    """解析三段式输出；answer 只保留最终正文，不含任何分段标记。"""
    raw = strip_code_fence(text or "").strip()
    sections = split_sections(raw)
    if sections:
        answer = sections.get("解答", "").strip()
        workpaper = sections.get("题目理解", "").strip()
        self_check = sections.get("自检", "").strip()
    else:
        logger.warning("作答输出没有三段式标题，整段作为正文")
        answer, workpaper, self_check = raw, "", ""
    return SolveResult(
        answer=answer,
        workpaper=workpaper,
        file_title=_file_title(task_title, subject),
        self_check=self_check,
        needs_human=(not answer) or _needs_human(self_check, answer),
    )


def _needs_human(self_check: str, answer: str) -> bool:
    check = (self_check or "").strip()
    if NEEDS_HUMAN_RE.search(check):
        return True
    if PASS_RE.search(check):
        return False
    blob = check or (answer or "")[:800]
    return any(word in blob for word in NEEDS_HUMAN_WORDS)


def _file_title(task_title: str, subject: str) -> str:
    title = normalize_ws(task_title or "").strip() or normalize_ws(subject or "").strip()
    return sanitize_filename(title or DEFAULT_TITLE, fallback=DEFAULT_TITLE,
                             max_len=TITLE_MAX_LEN)


# ---------------------------------------------------------------------- 取值小工具

def _value(obj: Any, name: str, default: str = "") -> str:
    value = getattr(obj, name, default)
    return str(value).strip() if value is not None else default


def _value_list(obj: Any, name: str) -> list[str]:
    value = getattr(obj, name, None) or []
    if isinstance(value, str):
        return [value.strip()] if value.strip() else []
    return [str(item).strip() for item in value if str(item).strip()]


def _person_values(person: PersonalInfo) -> dict[str, str]:
    if person is None:
        person = PersonalInfo()
    school_class = " ".join(
        bit for bit in (person.school, person.grade, person.class_name) if bit
    )
    return {
        "name": person.name or "（未填写）",
        "school_class": school_class or "（未填写）",
        "student_id": person.student_id or "（未填写）",
        "teacher": person.teacher_name or "（未指定）",
        "answer_style": person.answer_style or "标准作答",
        "language": person.language or "zh-CN",
        "extra_notes": person.extra_notes or "（无）",
        "subject": person.subject or "",
    }


def _signature(person: PersonalInfo) -> str:
    if person is None:
        return ""
    line = ""
    try:
        line = person.header_line()
    except Exception:   # noqa: BLE001 - 个人信息不完整也不能影响作答
        line = ""
    return (line or person.name or "").strip()


def _output_preference(output_cfg: OutputConfig) -> str:
    bits: list[str] = []
    if output_cfg is not None:
        bits.append("需要保留解题过程（正文之外单独保存）"
                    if bool(getattr(output_cfg, "include_workpaper", True))
                    else "只需要最终正文，解题过程从简")
        formats = getattr(output_cfg, "write_formats", None) or []
        if formats:
            bits.append("导出格式：" + " / ".join(str(fmt) for fmt in formats))
    return "；".join(bits) or "按标准作业格式输出"
