"""主 Agent：课堂督导。

职责：盯着"刚刚听到 / 看到的新内容"，判断它是否影响已知任务；如果影响，就调用工具
把最新信息补充进「作业子 Agent」的任务简报，必要时让子 Agent 重做。

两条路径：
1. 首选：让模型做 function calling（工具定义见 ai/tools.py），由它自己决定调哪个工具；
2. 降级：服务端不支持 tools 时，用纯本地的「相关度 + 新颖度」启发式兜底，
   保证"老师后补要求 → 子 Agent 重做"这个能力在任何服务商上都能工作。
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any

from ..config.models import ModelEndpoint, PersonalInfo
from ..core.models import ClassTask
from ..utils import clip
from .client import assistant_message, response_text, response_tool_calls
from .errors import AIClientError
from .prompts import USER_AGENT_REVIEW, get_prompt, render
from .tools import TOOL_SCHEMAS, ToolOutcome, dispatch_tool

log = logging.getLogger(__name__)

_CJK = "".join(chr(c) for c in range(0x4E00, 0xA000))
_SPLIT_RE = re.compile("[^0-9A-Za-z" + _CJK + "]+")

# 出现这些词说明"这段话可能在布置或补充要求"，值得让模型判断一次
_STRONG_CUES = (
    "作业", "任务", "练习", "提交", "截止", "要求", "格式", "字数",
    "背诵", "预习", "复习", "实验", "讨论", "报告", "完成",
)

# 这些措辞说明"老师在修改/追加要求"，即使没有重复任务里的名词也值得让模型判断一次。
# 督导每 N 个切片才跑一次（默认约 1 分钟一次），多花一次调用的代价可以接受；
# 而漏掉一句"必须用公式法"会让整份作业作废。
_MODIFIER_CUES = (
    "必须", "不能", "不要", "改成", "改用", "换成", "统一", "注意", "补充", "另外",
    "格式", "字数", "截止", "之前交", "明天交", "下节课交", "记得", "别忘",
)

_STATUS_CN = {
    "discovered": "已发现（还没作答）",
    "queued": "排队中",
    "working": "作答中",
    "solved": "已完成（已作答过）",
    "failed": "上次作答失败",
    "skipped": "已跳过",
}


# ------------------------------------------------------------------ 纯函数
def tokens(text: str) -> set[str]:
    """中文按 2-gram、英文按单词切分，用于本地相关度计算。"""
    cleaned = _SPLIT_RE.sub(" ", text or "")
    out: set[str] = set()
    for chunk in cleaned.split():
        if chunk.isascii():
            if len(chunk) >= 2:
                out.add(chunk.lower())
        else:
            if len(chunk) >= 2:
                out.update(chunk[i:i + 2] for i in range(len(chunk) - 1))
            out.add(chunk)
    return out


def task_text(task: ClassTask) -> str:
    parts = [task.title or "", task.detail or "", task.brief or "",
             " ".join(task.requirements or [])]
    return " ".join(p for p in parts if p)


def relevance(new_text: str, task: ClassTask) -> float:
    """新内容与某个任务的相关度（Jaccard，0~1）。"""
    a = tokens(new_text)
    b = tokens(task_text(task))
    if not a or not b:
        return 0.0
    return len(a & b) / float(len(a | b))


def novelty(new_text: str, task: ClassTask) -> float:
    """新内容里有多少比例是任务简报**还没覆盖**的（0~1）。"""
    a = tokens(new_text)
    b = tokens(task_text(task))
    if not a:
        return 0.0
    return len(a - b) / float(len(a))


def shared_tokens(new_text: str, task: ClassTask) -> set[str]:
    """新内容与任务共有的词（判断"这段话是不是在说这件事"最稳的指标）。"""
    return tokens(new_text) & tokens(task_text(task))


def looks_like_assignment(text: str) -> bool:
    return any(cue in (text or "") for cue in _STRONG_CUES)


# ------------------------------------------------------------------ 结果对象
@dataclass
class AgentReview:
    """一次督导的结果。"""

    actions: list[ToolOutcome] = field(default_factory=list)
    summary: str = ""
    degraded: bool = False
    degraded_reason: str = ""

    @property
    def changed(self) -> bool:
        return any(a.ok for a in self.actions)

    def describe(self) -> str:
        if not self.actions:
            return self.summary or "无需处理"
        parts = []
        for act in self.actions:
            flag = "" if act.ok else "[失败] "
            parts.append(flag + act.name + "：" + clip(act.result, 120, "…"))
        return "；".join(parts)


# ------------------------------------------------------------------ 督导 Agent
class TaskSupervisor:
    """把课堂新信息转成对子 Agent 的任务简报更新。"""

    def __init__(
        self,
        client: Any,
        cfg: ModelEndpoint,
        person: PersonalInfo,
        executor: Any,
        *,
        overrides: dict[str, str] | None = None,
        max_rounds: int = 3,
        min_relevance: float = 0.2,
        tools_enabled: bool = True,
        allow_revise: bool = True,
    ) -> None:
        self.client = client
        self.cfg = cfg
        self.person = person
        self.executor = executor
        self.overrides = overrides or {}
        self.max_rounds = max(1, int(max_rounds or 1))
        self.min_relevance = max(0.0, min(1.0, float(min_relevance)))
        self.tools_enabled = bool(tools_enabled)
        self.allow_revise = bool(allow_revise)
        self._tools_supported = bool(tools_enabled)

    # -------------------------------------------------------------- 主入口
    def review(self, new_content: str, tasks: list[ClassTask],
               session_context: str = "") -> AgentReview:
        text = (new_content or "").strip()
        active = [t for t in tasks if not getattr(t, "closed", False)]
        if not text or not active:
            return AgentReview(summary="没有需要处理的内容")
        if not self._worth_reviewing(text, active):
            return AgentReview(summary="与已知任务无关，跳过")

        if self._tools_supported:
            try:
                return self._llm_review(text, active, session_context)
            except AIClientError as exc:
                reason = self._degrade_reason(exc)
                log.warning("主 Agent 工具调用不可用：%s", reason)
                self._tools_supported = False
                review = self._heuristic_review(text, active)
                review.degraded = True
                review.degraded_reason = reason
                return review
            except Exception as exc:  # noqa: BLE001
                reason = f"工具调用异常（{type(exc).__name__}: {exc}），已改用启发式策略"
                log.warning("%s", reason)
                review = self._heuristic_review(text, active)
                review.degraded = True
                review.degraded_reason = reason
                return review

        return self._heuristic_review(text, active)

    # -------------------------------------------------------------- 预筛
    def _worth_reviewing(self, text: str, tasks: list[ClassTask]) -> bool:
        """是否值得让主 Agent 看一眼（纯本地判断，用来省调用）。

        宁可宽松一点：督导本身约每分钟才跑一次，而漏掉一句补充要求会让整份作业作废。
        """
        if not text:
            return False
        if looks_like_assignment(text):
            return True
        if any(shared_tokens(text, t) for t in tasks):
            return True
        return any(cue in text for cue in _MODIFIER_CUES)

    def _degrade_reason(self, exc: AIClientError) -> str:
        status = getattr(exc, "status", None)
        message = getattr(exc, "message", "") or str(exc)
        if status in (400, 404, 422) or "tool" in message.lower():
            return f"该模型服务不支持工具调用（status={status}），已改用启发式策略"
        return f"工具调用失败（{message}），已改用启发式策略"

    # -------------------------------------------------------------- 路径一：模型 function calling
    def _llm_review(self, text: str, tasks: list[ClassTask],
                    session_context: str) -> AgentReview:
        system = get_prompt("agent_supervisor", self.overrides)
        user = render(
            USER_AGENT_REVIEW,
            tasks=self._tasks_block(tasks),
            new_content=clip(text, 3000, "…（已截断）"),
            context=session_context or "(无)",
        )
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        actions: list[ToolOutcome] = []
        summary = ""
        for round_no in range(self.max_rounds):
            resp = self.client.chat_with_tools(messages, TOOL_SCHEMAS)
            calls = response_tool_calls(resp)
            text_out = (response_text(resp) or "").strip()
            if text_out:
                summary = text_out
            if not calls:
                break
            messages.append(assistant_message(resp))
            for call in calls:
                outcome = dispatch_tool(call.get("name", ""), call.get("arguments"), self.executor)
                actions.append(outcome)
                log.info("主 Agent 调用工具 %s -> %s", outcome.name, outcome.result)
                messages.append({
                    "role": "tool",
                    "tool_call_id": call.get("id") or outcome.name,
                    "content": clip(outcome.result, 1500, "…"),
                })
            if round_no == self.max_rounds - 1:
                log.info("主 Agent 达到最大工具轮数 %d，停止", self.max_rounds)
        return AgentReview(actions=actions, summary=summary)

    # -------------------------------------------------------------- 路径二：本地启发式
    def _heuristic_review(self, text: str, tasks: list[ClassTask]) -> AgentReview:
        """服务端不支持工具调用时的兜底：相关且新颖的内容就补进简报并重做。"""
        actions: list[ToolOutcome] = []
        for task in tasks:
            # 短标题下 Jaccard 天然偏低，用"共有词 >= 2"作为主判据更稳；
            # 预筛已经确认这段话确实像在布置/补充要求，这里不必再卡很高的相似度。
            shared = shared_tokens(text, task)
            rel = relevance(text, task)
            if len(shared) < 2 and rel < self.min_relevance:
                continue
            if not shared:
                continue
            nov = novelty(text, task)
            if nov < 0.34:
                continue  # 这段话的内容简报里基本都有了，没必要重做
            snippet = clip(text, 400, "…")
            try:
                result = self.executor.agent_update_task_brief(
                    task.id, [], "", snippet, ""
                )
                actions.append(ToolOutcome("update_task_brief",
                                           {"task_id": task.id, "relevance": round(rel, 3)},
                                           True, str(result)))
            except Exception as exc:  # noqa: BLE001
                actions.append(ToolOutcome("update_task_brief", {"task_id": task.id}, False, str(exc)))
                continue
            if self.allow_revise and task.status == "solved":
                try:
                    result = self.executor.agent_redispatch_task(
                        task.id, "", "听到与本题相关的新内容（启发式策略）"
                    )
                    actions.append(ToolOutcome("redispatch_task", {"task_id": task.id}, True, str(result)))
                except Exception as exc:  # noqa: BLE001
                    actions.append(ToolOutcome("redispatch_task", {"task_id": task.id}, False, str(exc)))
        return AgentReview(actions=actions, summary="启发式处理：%d 个动作" % len(actions))

    # -------------------------------------------------------------- 提示词素材
    @staticmethod
    def _tasks_block(tasks: list[ClassTask]) -> str:
        lines: list[str] = []
        for task in tasks:
            bits = [
                "id=" + task.id,
                "标题：" + (task.title or "(无)"),
                "类型：" + (task.task_type or "作业"),
                "状态：" + _STATUS_CN.get(task.status, task.status),
            ]
            if task.requirements:
                bits.append("要求：" + "；".join(task.requirements))
            if task.deadline:
                bits.append("截止：" + task.deadline)
            if task.brief:
                bits.append("已有简报：" + clip(task.brief, 200, "…"))
            if task.auto_revised:
                bits.append("已自动重做 %d 次" % task.auto_revised)
            if task.answer:
                bits.append("已作答摘要：" + clip(task.answer.replace(chr(10), " "), 120, "…"))
            lines.append("- " + " | ".join(bits))
        return chr(10).join(lines) if lines else "（当前没有任务）"
