r"""真机端到端：老师分两次说完要求 -> 主 Agent 调工具 -> 子 Agent 自动重做。

与 e2e_real.py 的区别：这里跑的是**真实的 ai.agent.TaskSupervisor**（提示词、工具 schema、
响应解析、参数归一化、dispatch、引擎工具实现），只把"模型网络调用"换成一个会按脚本
返回 tool_calls 的假客户端。真实采集（麦克风 + 屏幕）保持不变。

用法（项目根）：
    set PYTHONPATH=%CD%\\src
    .venv\\Scripts\\python.exe tools\\e2e_revision.py --seconds 30
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from ai_student.ai.agent import TaskSupervisor                          # noqa: E402
from ai_student.ai.tools import TOOL_REDISPATCH, TOOL_UPDATE_BRIEF      # noqa: E402
from ai_student.config.models import AppConfig                          # noqa: E402
from ai_student.core.engine import ClassEngine, EngineCallbacks         # noqa: E402
from ai_student.core.models import ClassTask                            # noqa: E402
from ai_student.core.session import SessionStore                        # noqa: E402

T0 = time.time()


def stamp() -> str:
    return "[+%5.1fs] " % (time.time() - T0)


# 老师分三次说：先布置任务 -> 闲聊 -> 补充关键要求
SCRIPT = [
    "同学们，现在大家完成这个任务：练习册第 32 页的题目。",
    "做完的同学先自己检查一遍。",
    "补充一下：解方程必须用公式法，而且要写出检验过程。",
]


class ScriptedAsr:
    _i = 0

    @staticmethod
    def transcribe_chunk(client, wav_path, cfg, hint=""):
        i = ScriptedAsr._i
        ScriptedAsr._i = i + 1
        return SCRIPT[i] if i < len(SCRIPT) else ""


class ScriptedVision:
    @staticmethod
    def analyze_board(client, image, cfg, context=""):
        class A:
            ocr_text = "练习册 P32"
            summary = "屏幕上是练习册第 32 页"
            key_points = []
            problems = []
            subject_hint = "数学"

        return A()


class ScriptedNotes:
    @staticmethod
    def merge_notes(client, prev, transcript_lines, board_lines, cfg, person, subject=""):
        return "# 课堂笔记" + chr(10) + chr(10) + "## 老师原话" + chr(10) + chr(10).join(transcript_lines)


class ScriptedTasks:
    @staticmethod
    def extract_tasks(client, cfg, person, notes, recent_transcript="", known_tasks=None, subject=""):
        if known_tasks:
            return []
        if "练习册" not in (notes or "") + (recent_transcript or ""):
            return []
        return [ClassTask(title="完成练习册第 32 页的题目", detail="练习册 P32",
                          task_type="作业", confidence=0.95, subject="数学")]


class ScriptedSolver:
    """第一次给 v1，重做后给 v2——用来说明"子 Agent 拿到了主 Agent 补充的简报"。"""

    calls: list[dict] = []

    @staticmethod
    def solve_task(client, cfg, person, task, context, output_cfg):
        has_brief = "任务简报" in context
        ScriptedSolver.calls.append({"brief": task.brief, "has_brief_in_context": has_brief})

        class R:
            pass

        if has_brief:
            R.answer = ("第 1-5 题（公式法）：" + chr(10)
                        + "1. x=1  2. x=2  3. x=3  4. x=4  5. x=5" + chr(10) + chr(10)
                        + "检验：把每个根代回原方程，左右两边相等。")
            R.workpaper = "统一用求根公式 x=(-b±√(b²-4ac))/(2a)，每题最后代入验算。"
            R.self_check = "已逐题代回检验"
        else:
            R.answer = "第 1-5 题：1. x=1  2. x=2  3. x=3  4. x=4  5. x=5"
            R.workpaper = "按常规方法求解。"
            R.self_check = "已完成"
        R.file_title = "P32第1-5题"
        R.needs_human = False
        return R()


class ScriptedAi:
    asr = ScriptedAsr
    vision = ScriptedVision
    notes = ScriptedNotes
    tasks = ScriptedTasks
    solver = ScriptedSolver


# ------------------------------------------------------------------ 模拟"支持工具调用的模型"
def _response(calls, text=""):
    return {"choices": [{"message": {
        "content": text,
        "tool_calls": [
            {"id": "call_%d" % i, "type": "function",
             "function": {"name": c["name"],
                          "arguments": json.dumps(c["args"], ensure_ascii=False)}}
            for i, c in enumerate(calls)
        ],
    }}]}


class DemoAgentClient:
    """只在"听到公式法"时决定调用工具，其余情况回答"无需处理"。"""

    def __init__(self):
        self.seen_user_messages: list[str] = []
        self.tool_rounds = 0
        self.done = False

    def chat_with_tools(self, messages, tools, **kwargs):
        user = ""
        for msg in messages:
            if msg.get("role") == "user":
                user = str(msg.get("content") or "")
        self.seen_user_messages.append(user)
        if self.done or "公式法" not in user:
            return _response([], "无需处理")
        match = re.search(r"id=(task_[0-9a-zA-Z]+)", user)
        if not match:
            return _response([], "无需处理")
        self.done = True
        self.tool_rounds += 1
        task_id = match.group(1)
        return _response([
            {"name": TOOL_UPDATE_BRIEF,
             "args": {"task_id": task_id,
                      "requirements": ["必须用公式法", "写出检验过程"],
                      "detail_append": "老师补充：解方程必须用公式法，而且要写出检验过程。"}},
            {"name": TOOL_REDISPATCH,
             "args": {"task_id": task_id, "reason": "老师补充了公式法与检验要求"}},
        ], "已把老师补充的要求转给子 Agent")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=30.0)
    ap.add_argument("--chunk", type=float, default=6.0)
    ap.add_argument("--out", default="")
    ap.add_argument("--no-loopback", action="store_true")
    args = ap.parse_args(argv)

    cfg = AppConfig()
    cfg.model.api_key = "e2e-revision"
    cfg.person.name = "演示学生"
    cfg.person.class_name = "高一(1)班"
    cfg.person.subject = "数学"
    cfg.output.output_dir = args.out or str(ROOT / ".runs" / "revision_out")
    cfg.output.write_formats = ["md"]
    cfg.capture.chunk_seconds = args.chunk
    cfg.capture.keyframe_interval_s = 10
    cfg.capture.loopback_enabled = not args.no_loopback
    cfg.behavior.notes_every_n_chunks = 1
    cfg.behavior.auto_solve_tasks = True
    cfg.behavior.agent_tools_enabled = True
    cfg.behavior.auto_revise_solved = True
    cfg.behavior.max_auto_revisions = 2

    agent_client = DemoAgentClient()
    events = {"revisions": 0, "artifacts": [], "states": []}

    cb = EngineCallbacks(
        on_state=lambda s, d: (events["states"].append(s), print(stamp() + "[状态] " + s + " " + d, flush=True)),
        on_transcript=lambda seg: print(stamp() + "老师：" + seg.text, flush=True),
        on_task=lambda t: print(stamp() + "[任务] " + t.title + " (" + t.status
                                + ", 自动重做 " + str(t.auto_revised) + " 次)", flush=True),
        on_artifact=lambda a: (events["artifacts"].append(a.path), print(stamp() + "[产出] " + Path(a.path).name, flush=True)),
        on_stage=lambda s, d: print(stamp() + "  · " + s + "：" + d, flush=True),
        on_log=lambda lvl, msg: print(stamp() + "  " + lvl + " " + msg, flush=True),
        on_error=lambda t, d: print(stamp() + "[错误] " + t + " " + d, flush=True),
    )

    engine = ClassEngine(
        lambda: cfg, cb, store=SessionStore(), ai=ScriptedAi,
        client_factory=lambda ep: agent_client,
        supervisor_factory=lambda eng, c: TaskSupervisor(
            agent_client, c.model, c.person, eng,
            max_rounds=c.behavior.agent_max_tool_rounds,
            min_relevance=c.behavior.agent_min_relevance,
            tools_enabled=True, allow_revise=True,
        ),
    )

    print("=" * 74)
    print("真机端到端：老师分两次说完要求 -> 主 Agent 调工具 -> 子 Agent 自动重做")
    print("=" * 74)
    engine.start(subject="数学", title="补充要求演示")

    deadline = time.time() + args.seconds
    while time.time() < deadline:
        time.sleep(0.5)
    engine.stop(finalize=True, wait=True)

    task = engine.tasks()[0] if engine.tasks() else None
    print("")
    print("=" * 74)
    ok = True

    def line(label, value, expect=True):
        nonlocal ok
        good = bool(value) if expect else True
        ok = ok and good
        print(("  [OK]  " if good else "  [!!]  ") + label + "：" + str(value))

    line("识别到任务", task.title if task else "(无)")
    line("子 Agent 作答次数", task.revision if task else 0)
    line("自动重做次数", task.auto_revised if task else 0)
    line("新增要求", task.requirements if task else [])
    line("任务简报已补充", "公式法" in (task.brief if task else ""))
    line("主 Agent 收到过任务清单", bool(agent_client.seen_user_messages
                                        and "id=task_" in agent_client.seen_user_messages[-1]))
    line("子 Agent 第二次拿到简报", bool(ScriptedSolver.calls and ScriptedSolver.calls[-1]["has_brief_in_context"]))
    line("状态流转", " -> ".join(events["states"]))
    files = [Path(p) for p in events["artifacts"]]
    line("产出文件数", len(files))
    task_file = ROOT / ".runs" / "revision_out"
    md_files = sorted(task_file.rglob("01_*.md"))
    if md_files:
        text = md_files[0].read_text(encoding="utf-8")
        line("最终作业已含公式法要求", "公式法" in text or "求根公式" in text)
        print("  ---- 最终作业内容 ----")
        for ln in text.splitlines()[:16]:
            print("    " + ln)
    print("=" * 74)
    print("结果：" + ("全部通过" if ok else "存在失败项"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
