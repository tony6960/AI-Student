r"""真机端到端冒烟：真实麦克风/回环 + 真实屏幕采集 + 脚本化 AI，
跑通"开始听课 -> 实时采集/转写 -> 关键帧 -> 笔记 -> 任务 -> 作业产出 -> 收尾"。

用途：在没有 API KEY 的情况下，验证整条产品链路（除真实大模型调用外）在本机可用。

用法（项目根）：
    set PYTHONPATH=%CD%\src
    .venv\Scripts\python.exe tools\e2e_real.py --seconds 20
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from ai_student.config.models import AppConfig                      # noqa: E402
from ai_student.core.engine import ClassEngine, EngineCallbacks     # noqa: E402
from ai_student.core.models import ClassTask                        # noqa: E402
from ai_student.core.session import SessionStore                    # noqa: E402

SCRIPT = [
    "同学们，今天这节课我们复习一元二次方程。",
    "现在大家按照刚才讲的，完成这个任务：练习册第 32 页第 1 到第 5 题，下节课交。",
]

T0 = time.time()


def stamp() -> str:
    """事件发生时的墙钟偏移，用来量化"老师说完 -> AI 产出"的延迟。"""
    return "[+%5.1fs] " % (time.time() - T0)


class ScriptedAsr:
    _i = 0

    @staticmethod
    def transcribe_chunk(client, wav_path, cfg, hint=""):
        i = ScriptedAsr._i
        ScriptedAsr._i = i + 1
        return SCRIPT[i % len(SCRIPT)]


class ScriptedVision:
    @staticmethod
    def analyze_board(client, image, cfg, context=""):
        class A:
            ocr_text = "练习册 P32 第 1-5 题"
            summary = "屏幕上是练习册第 32 页的题目"
            key_points = ["一元二次方程"]
            problems = ["练习册 P32 第 1-5 题"]
            subject_hint = "数学"

        return A()


def _has_assignment(text: str) -> bool:
    """只有当"老师确实布置了任务"那句话出现过，才算有任务——真模型也是这个行为。"""
    return "32 页" in (text or "")


class ScriptedNotes:
    @staticmethod
    def merge_notes(client, prev, transcript_lines, board_lines, cfg, person, subject=""):
        body = chr(10).join(transcript_lines)
        notes = "# 课堂笔记" + chr(10) + chr(10) + "## 老师原话" + chr(10) + body
        if _has_assignment(body):
            notes += (chr(10) + chr(10) + "## 老师布置的任务"
                      + chr(10) + "- 练习册第 32 页第 1 到第 5 题，下节课交")
        return notes


class ScriptedTasks:
    @staticmethod
    def extract_tasks(client, cfg, person, notes, recent_transcript="", known_tasks=None, subject=""):
        if known_tasks:
            return []
        if not _has_assignment((notes or "") + chr(10) + (recent_transcript or "")):
            return []
        return [ClassTask(title="练习册 P32 第 1-5 题", detail="一元二次方程练习，下节课交",
                          task_type="作业", confidence=0.95, subject="数学")]


class ScriptedSolver:
    @staticmethod
    def solve_task(client, cfg, person, task, context, output_cfg):
        class R:
            answer = "1. x=1" + chr(10) + "2. x=2" + chr(10) + "3. x=3" + chr(10) + "4. x=4" + chr(10) + "5. x=5"
            workpaper = "用因式分解法逐题求解，并代回验算。"
            file_title = "P32第1-5题"
            self_check = "已验算"
            needs_human = False

        return R()


class ScriptedAi:
    asr = ScriptedAsr
    vision = ScriptedVision
    notes = ScriptedNotes
    tasks = ScriptedTasks
    solver = ScriptedSolver


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=20.0)
    ap.add_argument("--out", default="")
    ap.add_argument("--no-loopback", action="store_true")
    ap.add_argument("--chunk", type=float, default=20.0, help="音频切片秒数（默认 20）")
    ap.add_argument("--notes-every", type=int, default=3, help="每 N 个切片汇总一次笔记（默认 3）")
    args = ap.parse_args(argv)

    cfg = AppConfig()
    cfg.model.api_key = "e2e-local"
    cfg.person.name = "冒烟学生"
    cfg.person.class_name = "高一(1)班"
    cfg.person.subject = "数学"
    cfg.output.output_dir = args.out or str(ROOT / ".runs" / "e2e_out")
    cfg.output.write_formats = ["md", "txt"]
    cfg.capture.chunk_seconds = args.chunk
    cfg.capture.keyframe_interval_s = 8
    cfg.capture.video_fps = 2
    cfg.capture.loopback_enabled = not args.no_loopback
    cfg.behavior.notes_every_n_chunks = args.notes_every

    events = {"transcript": 0, "board": 0, "tasks": 0, "artifacts": [], "states": []}
    cb = EngineCallbacks(
        on_state=lambda s, d: (events["states"].append(s), print(stamp() + "[状态] " + s + " " + d, flush=True)),
        on_transcript=lambda seg: (events.__setitem__("transcript", events["transcript"] + 1),
                                   print(stamp() + "<" + seg.timecode + "> 老师：" + seg.text, flush=True)),
        on_board=lambda n: (events.__setitem__("board", events["board"] + 1),
                            print(stamp() + "<" + n.timecode + "> 画面：" + (n.summary or n.ocr_text)[:50], flush=True)),
        on_task=lambda t: (events.__setitem__("tasks", events["tasks"] + 1),
                           print(stamp() + "[任务] " + t.title + " (" + t.status + ")", flush=True)),
        on_artifact=lambda a: (events["artifacts"].append(a.path),
                               print(stamp() + "[产出] " + a.path, flush=True)),
        on_log=lambda lvl, msg: print(stamp() + "  " + lvl + " " + msg, flush=True),
        on_error=lambda t, d: print(stamp() + "[错误] " + t + " " + d, flush=True),
    )

    engine = ClassEngine(lambda: cfg, cb, store=SessionStore(), ai=ScriptedAi,
                         client_factory=lambda ep: object())

    print("=" * 74)
    print("真机端到端冒烟：真实采集 + 脚本化 AI，录制 %.0f 秒" % args.seconds)
    print("=" * 74)
    t_start = time.time()
    info = engine.start(subject="数学", title="真机冒烟")
    print("[会话] 用时 %.2fs 启动 -> %s" % (time.time() - t_start, engine.workspace.root), flush=True)

    deadline = time.time() + args.seconds
    while time.time() < deadline:
        time.sleep(0.5)
    engine.stop(finalize=True, wait=True)

    ws = engine.workspace
    print("")
    print("=" * 74)
    ok = True

    def line(label, value, expect=True):
        nonlocal ok
        good = bool(value) if expect else True
        ok = ok and good
        print(("  [OK]  " if good else "  [!!]  ") + label + "：" + str(value))

    audio = ws.root / "audio" / "audio.wav"
    video = ws.root / "screen" / "video.mp4"
    line("会话目录", ws.root)
    line("全程录音存在", audio.exists() and audio.stat().st_size > 1000)
    line("录音字节", audio.stat().st_size if audio.exists() else 0)
    line("演示录像存在", video.exists() and video.stat().st_size > 1000)
    line("关键帧数", len(list((ws.root / "screen" / "keyframes").glob("*.jpg"))))
    line("转写段数", events["transcript"])
    line("画面理解数", events["board"])
    line("任务事件数（发现/作答中/完成）", events["tasks"])
    line("产出文件数", len(events["artifacts"]))
    for p in events["artifacts"]:
        line("  -> " + Path(p).name, Path(p).exists())
    line("会话时间轴", ws.timeline_path.exists())
    line("课堂笔记", ws.notes_path.exists())
    out = Path(cfg.output.output_dir)
    if out.exists():
        print("  输出目录：")
        for p in sorted(out.rglob("*")):
            if p.is_file():
                print("    " + str(p.relative_to(out)) + "  (" + str(p.stat().st_size) + "B)")
    print("=" * 74)
    print("结果：" + ("全部通过" if ok else "存在失败项"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
