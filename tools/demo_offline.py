r"""离线全流程演示：不需要麦克风、不需要 API KEY，直接跑通
"采集 -> 转写 -> 画面理解 -> 课堂笔记 -> 任务识别 -> 作业作答 -> 产出文件"。

用法（项目根目录）：
    set PYTHONPATH=%CD%\src
    .venv\Scripts\python.exe tools\demo_offline.py
    .venv\Scripts\python.exe tools\demo_offline.py --real-ai --subject 数学
    .venv\Scripts\python.exe tools\demo_offline.py --out D:\我的作业

--real-ai 会真正调用你在 %APPDATA%\AIStudent\config.json 里配置的多模态模型
（先用软件设置界面配置好 KEY），用合成的课堂音频/画面走一遍真实链路。
"""
from __future__ import annotations

import argparse
import math
import sys
import time
import wave
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ai_student.config.models import AppConfig                      # noqa: E402
from ai_student.config.store import load_config                     # noqa: E402
from ai_student.core.engine import ClassEngine, EngineCallbacks     # noqa: E402
from ai_student.core.models import ClassTask                        # noqa: E402
from ai_student.core.session import SessionStore                    # noqa: E402

# 脚本化"课堂"：老师说的话（离线演示用）
SCRIPT = [
    "同学们好，这节课我们学习一元二次方程的解法。",
    "一般形式是 ax 平方加 bx 加 c 等于零，其中 a 不等于零。",
    "求根公式是 x 等于二 a 分之负 b 加减根号下 b 平方减四 ac。",
    "判别式 b 平方减四 ac 大于零时有两个不相等的实数根。",
    "下面布置今天的作业：完成课本第四十五页第一题到第三题，写出完整解题过程，明天上课前交。",
]
NOTES_MD = """# 课堂笔记

## 课程信息
- 科目：数学
- 内容：一元二次方程的解法

## 知识点
- 一元二次方程一般形式 ax^2+bx+c=0 (a!=0)
- 求根公式 x = (-b +- sqrt(b^2-4ac)) / (2a)
- 判别式 b^2-4ac 与根的个数关系

## 板书与演示要点
- 推导求根公式（配方法）
- 例题：解 x^2-5x+6=0

## 老师布置的任务
- 完成课本第 45 页第 1-3 题，写出完整解题过程，明天上课前交

## 存疑
- 无
"""
ANSWER_MD = """### 第 1 题
解方程：x^2 - 5x + 6 = 0

因式分解得 (x-2)(x-3)=0，所以 x1=2，x2=3。

### 第 2 题
解方程：x^2 - 4x + 4 = 0

(x-2)^2 = 0，所以 x1 = x2 = 2（两个相等的实数根）。

### 第 3 题
解方程：2x^2 + 3x - 2 = 0

用求根公式：a=2, b=3, c=-2，判别式 = 9 + 16 = 25 > 0，
x = (-3 +- 5) / 4，所以 x1 = 1/2，x2 = -2。
"""
WORKPAPER = """1. 先判断能否因式分解；能分解的直接写根。
2. 不能分解时用求根公式，先算判别式确定根的个数。
3. 每题最后代回原方程验算。
"""


# ------------------------------------------------------------------ 合成素材
def write_wav(path: Path, seconds: float, sr: int = 16000, seed: int = 0) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = int(seconds * sr)
    t = np.linspace(0, seconds, n, endpoint=False)
    rng = np.random.default_rng(seed)
    sig = 0.35 * np.sin(2 * math.pi * 180 * t) * (1 + 0.5 * np.sin(2 * math.pi * 3 * t))
    sig += 0.05 * rng.standard_normal(n)
    pcm = np.clip(sig, -1, 1)
    pcm16 = (pcm * 32767).astype("<i2")
    with wave.open(str(path), "wb") as fh:
        fh.setnchannels(1)
        fh.setsampwidth(2)
        fh.setframerate(sr)
        fh.writeframes(pcm16.tobytes())


def write_jpeg(path: Path, seed: int, text: str) -> None:
    from PIL import Image, ImageDraw

    w, h = 960, 540
    img = Image.new("RGB", (w, h), (250, 250, 245))
    d = ImageDraw.Draw(img)
    d.rectangle([0, 0, w, 70], fill=(30, 60, 120))
    d.text((20, 25), "TOP DOMAIN 教师演示 / 数学", fill=(255, 255, 255))
    for i in range(6):
        d.line([(60, 140 + i * 55), (w - 60, 140 + i * 55)], fill=(210, 215, 225), width=1)
    d.text((70, 120), "ax^2 + bx + c = 0   (a != 0)", fill=(20, 20, 20))
    d.text((70, 175), "x = (-b +- sqrt(b^2 - 4ac)) / 2a", fill=(20, 20, 20))
    d.text((70, 230), text, fill=(160, 40, 40))
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path, format="JPEG", quality=80)


class DemoChunk:
    def __init__(self, index, path, t0, t1):
        self.index = index
        self.path = path
        self.t_start = t0
        self.t_end = t1
        self.silent = False
        self.rms_db = -22.0
        self.peak_db = -9.0
        self.source = "mix"


class DemoFrame:
    def __init__(self, index, path, t):
        self.index = index
        self.path = path
        self.t = t
        self.changed = True
        self.hash_distance = 1.0


class DemoAudio:
    def __init__(self, cfg, session_dir, on_chunk=None, on_error=None):
        self.dir = Path(session_dir)
        self.cb = on_chunk
        self.is_running = False
        self._t0 = time.time()

    def start(self):
        self.is_running = True
        total = max(2, int(self._seconds()))
        for i in range(1, total + 1):
            p = self.dir / "audio" / "chunks" / ("chunk_%04d.wav" % i)
            write_wav(p, 3.0, seed=i)
            if self.cb:
                self.cb(DemoChunk(i, p, (i - 1) * 20.0, i * 20.0))
        full = self.dir / "audio" / "audio.wav"
        write_wav(full, 30.0, seed=99)

    def _seconds(self):
        return 5

    def stop(self):
        self.is_running = False
        return self.dir / "audio" / "audio.wav"

    def pause(self):
        pass

    def resume(self):
        pass

    @property
    def elapsed(self):
        return time.time() - self._t0

    @property
    def last_rms_db(self):
        return -22.0

    def silence_seconds(self):
        return 0.0


class DemoFrames:
    def __init__(self, cfg, session_dir, on_keyframe=None, on_error=None):
        self.dir = Path(session_dir)
        self.cb = on_keyframe
        self.is_running = False
        self._t0 = time.time()

    def start(self):
        self.is_running = True
        texts = ["板书：配方法推导", "板书：求根公式", "例题：x^2-5x+6=0",
                 "作业：课本 P45 第 1-3 题"]
        for i, t in enumerate(texts, 1):
            p = self.dir / "screen" / "keyframes" / ("kf_%04d.jpg" % i)
            write_jpeg(p, i, t)
            if self.cb:
                self.cb(DemoFrame(i, p, i * 8.0))
        (self.dir / "screen").mkdir(parents=True, exist_ok=True)
        (self.dir / "screen" / "video.mp4").write_bytes(b"demo")

    def stop(self):
        self.is_running = False
        return self.dir / "screen" / "video.mp4"

    def pause(self):
        pass

    def resume(self):
        pass

    @property
    def elapsed(self):
        return time.time() - self._t0

    def last_frame_path(self):
        return None

    def preview_png(self):
        return None


# ------------------------------------------------------------------ 离线假模型
class ScriptedAsr:
    _i = 0

    @staticmethod
    def transcribe_chunk(client, wav_path, cfg, hint=""):
        idx = ScriptedAsr._i
        ScriptedAsr._i = idx + 1
        return SCRIPT[idx % len(SCRIPT)]


class ScriptedVision:
    @staticmethod
    def analyze_board(client, image, cfg, context=""):
        class A:
            ocr_text = "ax^2+bx+c=0 ; x = (-b +- sqrt(b^2-4ac))/2a"
            summary = "老师正在讲解一元二次方程求根公式并布置作业"
            key_points = ["求根公式", "判别式"]
            problems = ["课本 P45 第 1-3 题"]
            subject_hint = "数学"

        return A()


class ScriptedNotes:
    @staticmethod
    def merge_notes(client, previous_markdown, transcript_lines, board_lines, cfg, person, subject=""):
        return NOTES_MD


class ScriptedTasks:
    _done = False

    @staticmethod
    def extract_tasks(client, cfg, person, notes_markdown, recent_transcript="", known_tasks=None, subject=""):
        if known_tasks:
            return []
        return [ClassTask(title="完成课本 P45 第 1-3 题", detail="解一元二次方程，写出完整过程，明天上课前交",
                          task_type="作业", requirements=["写出完整解题过程", "明天上课前交"],
                          deadline="明天上课前", confidence=0.96, subject="数学")]


class ScriptedSolver:
    @staticmethod
    def solve_task(client, cfg, person, task, context, output_cfg):
        class R:
            answer = ANSWER_MD
            workpaper = WORKPAPER
            file_title = "P45第1-3题"
            self_check = "三题均已代回验算"
            needs_human = False

        return R()


class ScriptedAi:
    asr = ScriptedAsr
    vision = ScriptedVision
    notes = ScriptedNotes
    tasks = ScriptedTasks
    solver = ScriptedSolver


# ------------------------------------------------------------------ 主流程
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="AI学生 离线全流程演示")
    ap.add_argument("--real-ai", action="store_true", help="使用配置里的真实多模态模型")
    ap.add_argument("--subject", default="数学")
    ap.add_argument("--title", default="一元二次方程（离线演示）")
    ap.add_argument("--out", default="", help="作业输出目录（默认用配置里的目录）")
    ap.add_argument("--keep", action="store_true", help="保留会话素材目录（默认也保留）")
    args = ap.parse_args(argv)

    if args.real_ai:
        cfg = load_config()
        problems = cfg.validate()
        if problems:
            print("[!] 真实模型模式需要先完成配置：")
            for p in problems:
                print("    - " + p)
            return 2
        ai = None            # 用真实的 ai_student.ai.*
        client_factory = None
        print("[i] 使用真实多模态模型：" + cfg.model.display())
    else:
        cfg = AppConfig()
        cfg.model.api_key = "offline-demo"
        cfg.person.name = "演示学生"
        cfg.person.class_name = "高一(1)班"
        cfg.person.school = "示例中学"
        cfg.person.teacher_name = "王老师"
        cfg.person.subject = args.subject
        cfg.behavior.notes_every_n_chunks = 1
        ai = ScriptedAi
        client_factory = lambda ep: object()   # noqa: E731
        print("[i] 离线演示模式（脚本化课堂，不联网）")

    cfg.person.subject = args.subject
    if args.out:
        cfg.output.output_dir = args.out
    cfg.output.write_formats = ["md", "txt", "docx"]
    cfg.output.include_workpaper = True
    cfg.output.create_session_subfolder = True

    events: list[str] = []

    def log(level, msg):
        events.append("[" + level + "] " + msg)
        print("    " + level + "  " + msg, flush=True)

    cb = EngineCallbacks(
        on_state=lambda s, d: print("[状态] " + s + " " + d, flush=True),
        on_stage=lambda s, d: print("[阶段] " + s + " - " + d, flush=True),
        on_transcript=lambda seg: print("  <" + seg.timecode + "> " + seg.text, flush=True),
        on_board=lambda n: print("  <" + n.timecode + "> 画面：" + (n.summary or n.ocr_text)[:60], flush=True),
        on_notes=lambda md: print("  [笔记] 已更新 " + str(len(md)) + " 字", flush=True),
        on_task=lambda t: print("  [任务] " + t.title + "  (" + t.status + ")", flush=True),
        on_artifact=lambda a: print("  [产出] " + a.path, flush=True),
        on_log=log,
        on_error=lambda t, d: print("[错误] " + t + ": " + d, flush=True),
    )

    kwargs = dict(store=SessionStore(), ai=ai,
                  audio_factory=DemoAudio, screen_factory=DemoFrames)
    if client_factory is not None:
        kwargs["client_factory"] = client_factory
    engine = ClassEngine(lambda: cfg, cb, **kwargs)

    print("=" * 78)
    print("AI学生 全流程演示  |  科目：" + args.subject)
    print("=" * 78)
    info = engine.start(subject=args.subject, title=args.title)
    print("[会话] " + info.id + "  ->  " + str(engine.workspace.root))

    deadline = time.time() + (600 if args.real_ai else 30)
    while time.time() < deadline:
        tasks = engine.tasks()
        if tasks and all(t.status in ("solved", "failed") for t in tasks):
            break
        time.sleep(0.3)

    engine.stop(finalize=True, wait=True)

    ws = engine.workspace
    print("")
    print("=" * 78)
    print("结果")
    print("=" * 78)
    tasks = engine.tasks()
    print("识别到任务：" + str(len(tasks)))
    for t in tasks:
        print("  - " + t.title + "  [" + t.status + "]")
        for a in t.artifacts:
            print("      " + a)
    print("")
    print("素材留档目录：" + str(ws.root))
    print("作业输出目录：" + str(cfg.output.output_dir))
    out = Path(cfg.output.output_dir)
    if out.exists():
        print("产出文件：")
        for p in sorted(out.rglob("*")):
            if p.is_file():
                print("  " + str(p) + "  (" + str(p.stat().st_size) + " 字节)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
