"""主 Agent（课堂督导）可以调用的工具：把课堂新听到的信息补给「作业子 Agent」。

设计要点
- 工具用 OpenAI function calling 的 JSON Schema 描述。
- 工具本身**只做参数校验与归一化**，真正改数据由注入的 executor 完成（引擎实现），
  这样 ai 层不依赖 core，也便于离线测试。
- 模型经常把数组写成字符串、把数字写成字符串，这里统一容错。
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

log = logging.getLogger(__name__)

TOOL_UPDATE_BRIEF = "update_task_brief"
TOOL_REDISPATCH = "redispatch_task"
TOOL_CREATE_TASK = "create_task"
TOOL_CLOSE_TASK = "close_task"
TOOL_RAISE_QUESTION = "raise_question"

TOOL_NAMES = [
    TOOL_UPDATE_BRIEF,
    TOOL_REDISPATCH,
    TOOL_CREATE_TASK,
    TOOL_CLOSE_TASK,
    TOOL_RAISE_QUESTION,
]

# 工具名 -> 引擎实现的方法名
_METHOD_OF = {
    TOOL_UPDATE_BRIEF: "agent_update_task_brief",
    TOOL_REDISPATCH: "agent_redispatch_task",
    TOOL_CREATE_TASK: "agent_create_task",
    TOOL_CLOSE_TASK: "agent_close_task",
    TOOL_RAISE_QUESTION: "agent_raise_question",
}


@runtime_checkable
class TaskToolExecutor(Protocol):
    """引擎需要实现的方法集合（全部返回一句给模型看的中文结果）。"""

    def agent_update_task_brief(self, task_id: str, requirements: list[str] | None = None,
                                deadline: str = "", detail_append: str = "",
                                subject: str = "") -> str: ...

    def agent_redispatch_task(self, task_id: str, instruction: str = "", reason: str = "") -> str: ...

    def agent_create_task(self, title: str, detail: str = "", task_type: str = "作业",
                          requirements: list[str] | None = None,
                          confidence: float = 0.8) -> str: ...

    def agent_close_task(self, task_id: str, reason: str = "") -> str: ...

    def agent_raise_question(self, task_id: str, question: str) -> str: ...


TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": TOOL_UPDATE_BRIEF,
            "description": (
                "把老师刚刚补充的要求 / 截止时间 / 细节**增补**进某个任务的简报。"
                "只增补，不要删除或改写已有内容。新信息与任务无关时不要调用。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "task_id": {"type": "string", "description": "任务 ID，必须从任务清单里原样复制"},
                    "requirements": {
                        "type": "array", "items": {"type": "string"},
                        "description": "新增或修正的作答要求，每条一句话；没有就传空数组",
                    },
                    "deadline": {"type": "string", "description": "新的截止时间；老师没提到就留空"},
                    "detail_append": {"type": "string", "description": "需要补充的范围 / 格式 / 背景说明"},
                    "subject": {"type": "string", "description": "学科；判断不出留空"},
                },
                "required": ["task_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": TOOL_REDISPATCH,
            "description": (
                "让作业子 Agent 用**更新后的简报**重新作答（原答案会被覆盖）。"
                "仅当简报确实被补充、且该任务已经作答过时调用；同一任务不要反复重做。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "task_id": {"type": "string", "description": "任务 ID"},
                    "instruction": {"type": "string", "description": "给子 Agent 的额外交代；没有就留空"},
                    "reason": {"type": "string", "description": "为什么要重做，一句话，会写进日志"},
                },
                "required": ["task_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": TOOL_CREATE_TASK,
            "description": "仅当老师明确布置了一个此前没有记录过的任务时，新建一条任务。",
            "parameters": {
                "type": "object",
                "properties": {
                    "title": {"type": "string", "description": "任务标题，简洁明确"},
                    "detail": {"type": "string", "description": "任务详情 / 老师原话"},
                    "task_type": {
                        "type": "string",
                        "description": "作业 / 练习 / 背诵 / 实验 / 讨论 / 阅读 / 其他",
                    },
                    "requirements": {
                        "type": "array", "items": {"type": "string"},
                        "description": "作答要求",
                    },
                    "confidence": {"type": "number", "description": "0~1，你有多确定这是老师布置的任务"},
                },
                "required": ["title"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": TOOL_CLOSE_TASK,
            "description": "老师明确取消了该任务，或该任务已经完成并交代完毕时调用。",
            "parameters": {
                "type": "object",
                "properties": {
                    "task_id": {"type": "string", "description": "任务 ID"},
                    "reason": {"type": "string", "description": "关闭原因，一句话"},
                },
                "required": ["task_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": TOOL_RAISE_QUESTION,
            "description": "老师的要求听不清、拿不准时记录存疑，交给人工确认。不要靠猜。",
            "parameters": {
                "type": "object",
                "properties": {
                    "task_id": {"type": "string", "description": "相关任务 ID；与具体任务无关时留空"},
                    "question": {"type": "string", "description": "需要人工确认的问题"},
                },
                "required": ["question"],
            },
        },
    },
]


@dataclass
class ToolOutcome:
    """一次工具调用的结果（给界面 / 日志 / 模型看）。"""

    name: str
    arguments: dict[str, Any] = field(default_factory=dict)
    ok: bool = True
    result: str = ""


# ------------------------------------------------------------------ 参数归一化
def as_text(value: Any, limit: int = 2000) -> str:
    if value is None:
        return ""
    if isinstance(value, (list, tuple)):
        value = "；".join(str(v) for v in value if str(v).strip())
    text = str(value).strip()
    return text[:limit]


def as_text_list(value: Any) -> list[str]:
    """模型可能给数组、字符串、或者带编号的一段话，统一成字符串列表。"""
    if value is None:
        return []
    items: list[Any]
    if isinstance(value, str):
        raw = value.replace(chr(13), chr(10))
        parts = [p for chunk in raw.split(chr(10)) for p in chunk.split("；")]
        items = parts if len(parts) > 1 else [raw]
    elif isinstance(value, (list, tuple, set)):
        items = list(value)
    else:
        items = [value]
    out: list[str] = []
    for item in items:
        text = str(item).strip().lstrip("-•*0123456789.、) ").strip()
        if text and text not in out:
            out.append(text[:300])
    return out


def as_confidence(value: Any, default: float = 0.8) -> float:
    try:
        num = float(value)
    except (TypeError, ValueError):
        return default
    if num > 1.0:  # 有的模型给百分比
        num = num / 100.0
    return max(0.0, min(1.0, num))


# ------------------------------------------------------------------ 分发
def dispatch_tool(name: str, arguments: dict[str, Any] | None, executor: Any) -> ToolOutcome:
    """执行一个工具调用；任何异常都被吞掉并转成给模型看的错误说明。"""
    args = dict(arguments or {})
    tool = (name or "").strip()
    method_name = _METHOD_OF.get(tool)
    if method_name is None:
        return ToolOutcome(tool, args, False, f"未知工具：{tool}")
    handler = getattr(executor, method_name, None)
    if handler is None:
        return ToolOutcome(tool, args, False, f"当前引擎不支持工具 {tool}")

    try:
        result = _call(handler, tool, args)
    except Exception as exc:  # noqa: BLE001 - 工具失败不能让督导流程崩掉
        log.warning("工具 %s 执行失败：%s", tool, exc)
        return ToolOutcome(tool, args, False, f"{tool} 执行失败：{exc}")
    return ToolOutcome(tool, args, True, str(result or "已完成"))


def _call(handler: Any, tool: str, args: dict[str, Any]) -> str:
    if tool == TOOL_UPDATE_BRIEF:
        return handler(
            as_text(args.get("task_id")),
            as_text_list(args.get("requirements")),
            as_text(args.get("deadline"), 200),
            as_text(args.get("detail_append"), 1500),
            as_text(args.get("subject"), 50),
        )
    if tool == TOOL_REDISPATCH:
        return handler(
            as_text(args.get("task_id")),
            as_text(args.get("instruction"), 800),
            as_text(args.get("reason"), 200),
        )
    if tool == TOOL_CREATE_TASK:
        return handler(
            as_text(args.get("title"), 200),
            as_text(args.get("detail"), 2000),
            as_text(args.get("task_type"), 20) or "作业",
            as_text_list(args.get("requirements")),
            as_confidence(args.get("confidence")),
        )
    if tool == TOOL_CLOSE_TASK:
        return handler(as_text(args.get("task_id")), as_text(args.get("reason"), 200))
    if tool == TOOL_RAISE_QUESTION:
        return handler(as_text(args.get("task_id")), as_text(args.get("question"), 500))
    raise ValueError(f"未实现的工具：{tool}")


def describe_tools() -> str:
    """给日志/界面用的一行说明。"""
    return " / ".join(TOOL_NAMES)
