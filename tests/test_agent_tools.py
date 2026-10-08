"""主 Agent（工具调用）与"听到新要求 → 子 Agent 自动重做"的离线测试。

全部离线：FakeToolClient 直接返回伪造的 tool_calls，不联网、不需要 API KEY。
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from ai_student.ai.agent import (
    AgentReview, TaskSupervisor, looks_like_assignment, novelty, relevance, tokens,
)
from ai_student.ai.errors import AIClientError
from ai_student.ai.tools import (
    TOOL_CLOSE_TASK, TOOL_CREATE_TASK, TOOL_NAMES, TOOL_RAISE_QUESTION,
    TOOL_REDISPATCH, TOOL_SCHEMAS, TOOL_UPDATE_BRIEF, ToolOutcome, as_confidence,
    as_text, as_text_list, dispatch_tool,
)
from ai_student.config.models import AppConfig, ModelEndpoint, PersonalInfo
from ai_student.core.engine import ClassEngine, EngineCallbacks
from ai_student.core.models import ClassTask
from ai_student.core.session import SessionStore

from test_core import FakeAudio, FakeScreen, FakeAi, _make_cfg


# ------------------------------------------------------------------ 工具 schema
def test_tool_schemas_are_wellformed():
    names = []
    for schema in TOOL_SCHEMAS:
        assert schema.get("type") == "function"
        fn = schema["function"]
        assert fn["name"], "工具必须有名字"
        assert len(fn.get("description", "")) > 10, fn["name"] + " 缺少说明（模型靠它选工具）"
        params = fn.get("parameters")
        assert params and params.get("type") == "object"
        assert "properties" in params
        names.append(fn["name"])
    assert names == TOOL_NAMES, "schema 与 TOOL_NAMES 必须一致"


def test_tool_descriptions_mention_when_not_to_use():
    text = json.dumps(TOOL_SCHEMAS, ensure_ascii=False)
    assert "不要调用" in text or "无关时不要" in text, "必须明确告诉模型什么时候不要调用工具"


# ------------------------------------------------------------------ 参数归一化
def test_as_text_list_handles_model_quirks():
    assert as_text_list(["a", "b"]) == ["a", "b"]
    assert as_text_list("用公式法；写出过程") == ["用公式法", "写出过程"]
    assert as_text_list("1. 用公式法\n2. 写出过程") == ["用公式法", "写出过程"]
    assert as_text_list(None) == []
    assert as_text_list(["a", "a", " b "]) == ["a", "b"]


def test_as_text_and_confidence():
    assert as_text(["甲", "乙"]) == "甲；乙"
    assert as_text(None) == ""
    assert as_confidence("0.9") == pytest.approx(0.9)
    assert as_confidence(90) == pytest.approx(0.9)
    assert as_confidence("不确定") == pytest.approx(0.8)
    assert as_confidence(-1) == 0.0


# ------------------------------------------------------------------ 分发
class RecordingExecutor:
    def __init__(self):
        self.calls = []

    def agent_update_task_brief(self, task_id, requirements=None, deadline="", detail_append="", subject=""):
        self.calls.append(("update", task_id, requirements, deadline, detail_append, subject))
        return "已更新简报"

    def agent_redispatch_task(self, task_id, instruction="", reason=""):
        self.calls.append(("redispatch", task_id, instruction, reason))
        return "已重做"

    def agent_create_task(self, title, detail="", task_type="作业", requirements=None, confidence=0.8):
        self.calls.append(("create", title, detail, task_type, requirements, confidence))
        return "已新建"

    def agent_close_task(self, task_id, reason=""):
        self.calls.append(("close", task_id, reason))
        return "已关闭"

    def agent_raise_question(self, task_id, question):
        self.calls.append(("question", task_id, question))
        return "已记录"


def test_dispatch_unknown_tool():
    out = dispatch_tool("delete_everything", {}, RecordingExecutor())
    assert out.ok is False and "未知工具" in out.result


def test_dispatch_without_handler():
    class Bare:
        pass

    out = dispatch_tool(TOOL_UPDATE_BRIEF, {"task_id": "t1"}, Bare())
    assert out.ok is False and "不支持" in out.result


def test_dispatch_coerces_arguments():
    ex = RecordingExecutor()
    out = dispatch_tool(TOOL_UPDATE_BRIEF, {
        "task_id": "task_1",
        "requirements": "用公式法；保留过程",
        "deadline": "明天上课前",
        "detail_append": "老师强调不能用因式分解",
    }, ex)
    assert out.ok is True
    kind, task_id, reqs, deadline, detail, _subject = ex.calls[0]
    assert kind == "update" and task_id == "task_1"
    assert reqs == ["用公式法", "保留过程"]
    assert deadline == "明天上课前"
    assert "因式分解" in detail


def test_dispatch_swallows_handler_exception():
    class Boom(RecordingExecutor):
        def agent_redispatch_task(self, task_id, instruction="", reason=""):
            raise RuntimeError("引擎炸了")

    out = dispatch_tool(TOOL_REDISPATCH, {"task_id": "t"}, Boom())
    assert out.ok is False and "执行失败" in out.result


# ------------------------------------------------------------------ 相关度纯函数
def test_tokens_relevance_novelty():
    task = ClassTask(title="完成练习册 P32 第 1-5 题", requirements=["用公式法"])
    assert tokens("用公式法解方程") >= {"公式", "式法"}
    assert relevance("现在完成练习册第 32 页第 1 到第 5 题", task) > 0.1
    assert relevance("今天天气不错", task) == 0.0
    assert novelty("补充要求：必须用配方法", task) > 0.5
    assert novelty("完成练习册 P32 第 1-5 题", task) < 0.2
    assert looks_like_assignment("下节课交作业") is True
    assert looks_like_assignment("今天天气不错") is False


# ------------------------------------------------------------------ 假客户端
def make_response(calls, text=""):
    return {
        "choices": [{
            "message": {
                "content": text,
                "tool_calls": [
                    {
                        "id": c.get("id", "call_%d" % i),
                        "type": "function",
                        "function": {
                            "name": c["name"],
                            "arguments": json.dumps(c.get("args", {}), ensure_ascii=False),
                        },
                    }
                    for i, c in enumerate(calls)
                ],
            }
        }]
    }


class FakeToolClient:
    def __init__(self, responses, fail_with=None):
        self.responses = list(responses)
        self.fail_with = fail_with
        self.seen = []

    def chat_with_tools(self, messages, tools, **kwargs):
        self.seen.append({"messages": list(messages), "tools": tools})
        if self.fail_with is not None:
            raise self.fail_with
        if not self.responses:
            return make_response([], "无需处理")
        return self.responses.pop(0)


CFG = ModelEndpoint(base_url="http://x/v1", api_key="k", chat_model="m")
PERSON = PersonalInfo(name="测试学生", subject="数学")


def _supervisor(client, executor, **kw):
    return TaskSupervisor(client, CFG, PERSON, executor, **kw)


# ------------------------------------------------------------------ 督导：模型路径
def test_supervisor_executes_tool_calls():
    ex = RecordingExecutor()
    client = FakeToolClient([
        make_response([
            {"name": TOOL_UPDATE_BRIEF, "args": {"task_id": "t1", "requirements": ["必须用公式法"]}},
            {"name": TOOL_REDISPATCH, "args": {"task_id": "t1", "reason": "老师补充了要求"}},
        ], "已处理"),
    ])
    task = ClassTask(id="t1", title="练习册 P32 第 1-5 题", status="solved")
    review = _supervisor(client, ex).review("老师补充：必须用公式法", [task])
    assert review.changed is True
    assert [c[0] for c in ex.calls] == ["update", "redispatch"]
    assert review.degraded is False
    # 工具结果必须回灌给模型，否则它无法进行下一轮判断
    assert any(m.get("role") == "tool" for m in client.seen[-1]["messages"])


def test_supervisor_no_tool_call_means_no_action():
    ex = RecordingExecutor()
    client = FakeToolClient([make_response([], "无需处理")])
    review = _supervisor(client, ex).review("同学们把书翻到第 30 页", [ClassTask(id="t1", title="作业")])
    assert review.actions == [] and ex.calls == []


def test_supervisor_respects_max_rounds():
    ex = RecordingExecutor()
    looping = [make_response([{"name": TOOL_RAISE_QUESTION, "args": {"question": "第几题？"}}])
               for _ in range(10)]
    client = FakeToolClient(looping)
    review = _supervisor(client, ex, max_rounds=3).review("老师说要做作业", [ClassTask(id="t1", title="作业")])
    assert len(review.actions) == 3, "必须受最大轮数限制，避免无限调用工具"


def test_supervisor_skips_when_not_worth_reviewing():
    ex = RecordingExecutor()
    client = FakeToolClient([])
    review = _supervisor(client, ex).review("嗯，这个嘛", [ClassTask(id="t1", title="练习册 P45 第 1-3 题")])
    assert review.actions == []
    assert client.seen == [], "不值得判断时不应该调用模型（省钱）"


# ------------------------------------------------------------------ 督导：降级路径
def test_supervisor_degrades_when_tools_unsupported():
    ex = RecordingExecutor()
    exc = AIClientError("工具调用请求失败：400 unsupported parameter tools", status=400)
    client = FakeToolClient([], fail_with=exc)
    task = ClassTask(id="t1", title="完成练习册 P32 第 1-5 题", status="solved")
    review = _supervisor(client, ex).review("老师补充：练习册第 32 页的题必须用公式法",
                                            [task])
    assert review.degraded is True
    assert "不支持工具调用" in review.degraded_reason
    kinds = [c[0] for c in ex.calls]
    assert "update" in kinds, "降级后仍然要把新信息补进简报"
    assert "redispatch" in kinds, "降级后仍然要触发重做"


def test_supervisor_heuristic_skips_already_known_content():
    ex = RecordingExecutor()
    client = FakeToolClient([], fail_with=AIClientError("no tools", status=400))
    task = ClassTask(id="t1", title="完成练习册 P32 第 1-5 题",
                     requirements=["用公式法"], brief="完成练习册 P32 第 1-5 题，用公式法",
                     status="solved")
    review = _supervisor(client, ex).review("完成练习册 P32 第 1-5 题", [task])
    assert ex.calls == [], "简报里已经有的内容不该触发重做"
    assert review.actions == []


def test_supervisor_remembers_tools_unsupported():
    ex = RecordingExecutor()
    client = FakeToolClient([], fail_with=AIClientError("no tools", status=400))
    sup = _supervisor(client, ex)
    sup.review("老师补充：练习册第 32 页的题必须用公式法", [ClassTask(id="t1", title="练习册 P32 第 1-5 题", status="solved")])
    calls_after_first = len(client.seen)
    sup.review("老师又说：练习册第 32 页的题要用配方法", [ClassTask(id="t1", title="练习册 P32 第 1-5 题", status="solved")])
    assert len(client.seen) == calls_after_first, "已知不支持工具后不该每次都再试一遍"


# ------------------------------------------------------------------ 引擎：补要求 → 自动重做
class RecordingSolver:
    contexts: list[str] = []

    @staticmethod
    def solve_task(client, cfg, person, task, context, output_cfg):
        RecordingSolver.contexts.append(context)

        class R:
            answer = "第 1-5 题解答（按最新要求）"
            workpaper = "用公式法逐步求解"
            file_title = "P32第1-5题"
            self_check = "已复核"
            needs_human = False

        return R()


class EngineAi:
    asr = FakeAi.asr
    vision = FakeAi.vision
    notes = FakeAi.notes
    tasks = FakeAi.tasks
    solver = RecordingSolver


class ScriptedSupervisor:
    """模拟"模型发现老师补充了要求"的主 Agent。"""

    def __init__(self, engine):
        self.engine = engine
        self.client = None
        self.review_count = 0

    def review(self, new_content, tasks, session_context=""):
        self.review_count += 1
        if "公式法" not in (new_content or ""):
            return AgentReview(summary="无需处理")
        actions = []
        for task in tasks:
            if task.status != "solved":
                continue
            text = self.engine.agent_update_task_brief(
                task.id, ["必须用公式法"], "", "老师补充：解方程必须用公式法，不能用因式分解。", "",
            )
            actions.append(ToolOutcome("update_task_brief", {"task_id": task.id}, True, text))
            text2 = self.engine.agent_redispatch_task(task.id, "", "老师补充了公式法要求")
            actions.append(ToolOutcome("redispatch_task", {"task_id": task.id}, True, text2))
        return AgentReview(actions=actions, summary="已把最新要求转给子 Agent")


def _wait(predicate, timeout=40.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return True
        time.sleep(0.2)
    return False


def _build_engine(cfg):
    return ClassEngine(
        lambda: cfg, EngineCallbacks(), store=SessionStore(), ai=EngineAi,
        audio_factory=FakeAudio, screen_factory=FakeScreen,
        client_factory=lambda ep: object(),
        supervisor_factory=lambda eng, c: ScriptedSupervisor(eng),
    )


def test_engine_auto_revision_after_teacher_adds_requirement(tmp_home, tmp_path):
    RecordingSolver.contexts = []
    cfg = _make_cfg(tmp_path)
    engine = _build_engine(cfg)
    engine.start(subject="数学", title="补充要求")

    assert _wait(lambda: engine.tasks() and engine.tasks()[0].status == "solved")
    task = engine.tasks()[0]
    first_revision = task.revision
    first_context = RecordingSolver.contexts[-1]
    assert "任务简报" not in first_context, "第一次作答时还没有简报"

    # 老师在课上补充要求 -> 主 Agent 应该把新信息补给子 Agent 并让它重做
    engine._supervise("老师补充：解方程必须用公式法，不能用因式分解。")
    assert _wait(lambda: task.auto_revised >= 1 and task.revision > first_revision
                 and task.status == "solved")

    assert "必须用公式法" in task.requirements
    assert "公式法" in task.brief
    assert task.auto_revised == 1

    latest = RecordingSolver.contexts[-1]
    assert "任务简报" in latest, "子 Agent 必须拿到主 Agent 维护的简报"
    assert "必须用公式法" in latest or "公式法" in latest

    engine.stop(finalize=True, wait=True)

    # 重做后的产物同样落到输出目录
    out = Path(cfg.output.output_dir)
    files = [p.name for p in out.rglob("*") if p.is_file()]
    assert any(f.endswith(".md") for f in files), files


def test_engine_respects_auto_revision_cap(tmp_home, tmp_path):
    RecordingSolver.contexts = []
    cfg = _make_cfg(tmp_path)
    cfg.behavior.max_auto_revisions = 1
    engine = _build_engine(cfg)
    engine.start(subject="数学", title="重做上限")

    assert _wait(lambda: engine.tasks() and engine.tasks()[0].status == "solved")
    task = engine.tasks()[0]

    engine._supervise("老师补充：必须用公式法")
    assert _wait(lambda: task.auto_revised == 1 and task.status == "solved")

    engine._supervise("老师又补充：必须用公式法，而且要写检验过程")
    time.sleep(3.0)
    assert task.auto_revised == 1, "达到上限后不得继续自动重做（控制 API 开销）"

    engine.stop(finalize=True, wait=True)


def test_engine_supervisor_disabled_by_config(tmp_home, tmp_path):
    RecordingSolver.contexts = []
    cfg = _make_cfg(tmp_path)
    cfg.behavior.agent_tools_enabled = False
    engine = _build_engine(cfg)
    engine.start(subject="数学", title="关闭督导")
    assert _wait(lambda: engine.tasks() and engine.tasks()[0].status == "solved")
    before = len(RecordingSolver.contexts)
    engine._supervise("老师补充：必须用公式法")
    time.sleep(2.0)
    assert len(RecordingSolver.contexts) == before, "关闭开关后不应有任何督导动作"
    engine.stop(finalize=True, wait=True)


# ------------------------------------------------------------------ 人工确认
def test_raise_question_is_recorded(tmp_home, tmp_path):
    cfg = _make_cfg(tmp_path)
    engine = _build_engine(cfg)
    engine.start(subject="数学", title="存疑")
    assert _wait(lambda: engine.tasks())
    task = engine.tasks()[0]
    msg = engine.agent_raise_question(task.id, "老师说的第 6 题是指课后题还是课本例题？")
    assert "已记录" in msg
    assert task.questions and "第 6 题" in task.questions[0]
    engine.stop(finalize=True, wait=True)


def test_close_task_stops_processing(tmp_home, tmp_path):
    cfg = _make_cfg(tmp_path)
    engine = _build_engine(cfg)
    engine.start(subject="数学", title="关闭任务")
    assert _wait(lambda: engine.tasks())
    task = engine.tasks()[0]
    engine.agent_close_task(task.id, "老师当堂取消了这项作业")
    assert task.closed is True and task.status == "skipped"
    engine.stop(finalize=True, wait=True)
