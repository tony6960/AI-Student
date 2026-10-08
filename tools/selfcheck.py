r"""一键自检：验证安装、依赖、采集环境、真实应用入口与引擎全链路。

用法（项目根）：
    set PYTHONPATH=%CD%\src
    .venv\Scripts\python.exe tools\selfcheck.py

退出码 0 表示全部通过。所有涉及 GUI / 引擎的检查都在**独立子进程**里跑，
避免 Qt 与线程池状态互相污染。
"""
from __future__ import annotations

import os
import subprocess
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
RUNS = ROOT / ".runs" / "selfcheck"
for _p in (str(SRC), str(ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

os.environ.setdefault("AI_STUDENT_HOME", str(RUNS / "home"))

CHECKS: list[tuple[str, object]] = []


def check(name: str):
    def deco(fn):
        CHECKS.append((name, fn))
        return fn

    return deco


def _child_env(extra: dict | None = None) -> dict:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(SRC)
    env["PYTHONIOENCODING"] = "utf-8"
    env["QT_QPA_PLATFORM"] = "offscreen"
    # offscreen 平台自带字体目录缺失时中文会渲染成方块，指向系统字体目录即可
    env.setdefault("QT_QPA_FONTDIR", r"C:\Windows\Fonts")
    env["AI_STUDENT_HOME"] = str(RUNS / "child_home")
    if extra:
        env.update(extra)
    return env


def _run_child(args: list[str], timeout: float = 180.0) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, *args],
        cwd=str(ROOT),
        env=_child_env(),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
    )


# ------------------------------------------------------------------ 环境
@check("Python 版本 >= 3.10")
def _py():
    assert sys.version_info >= (3, 10), sys.version
    return sys.version.split()[0]


@check("依赖导入（PySide6 / numpy / PIL / mss / pywin32 / openai / soundfile / docx）")
def _deps():
    import docx, mss, numpy, openai, PIL, PySide6, soundfile, win32api  # noqa: F401

    return "ok"


@check("音频后端（PyAudioWPatch / SoundCard）")
def _audio_backend():
    found = []
    for name, mod in (("pyaudiowpatch", "pyaudiowpatch"), ("soundcard", "soundcard")):
        try:
            __import__(mod)
            found.append(name)
        except Exception:
            pass
    assert found, "没有可用的 WASAPI 采集后端"
    return "+".join(found)


@check("ffmpeg 可用（imageio-ffmpeg 自带）")
def _ffmpeg():
    from ai_student.capture.ffmpeg import ffmpeg_available, ffmpeg_path

    path = ffmpeg_path()
    assert ffmpeg_available(), "ffmpeg 无法执行: " + path
    return path


@check("枚举音频设备 / 显示器 / 窗口 / 极域候选")
def _devices():
    from ai_student.capture.devices import describe_environment

    env = describe_environment()
    return "音频 %d / 显示器 %d / 窗口 %d / 极域候选 %d" % (
        len(env.get("audio") or []),
        len(env.get("monitors") or []),
        len(env.get("windows") or []),
        len(env.get("classroom_windows") or []),
    )


# ------------------------------------------------------------------ 真实入口
@check("真实应用入口可启动（python -m ai_student --smoke）")
def _app_smoke():
    proc = _run_child(["-m", "ai_student", "--smoke", "--no-tray"], timeout=180)
    assert proc.returncode == 0, "exit=%s stderr=%s" % (proc.returncode, (proc.stderr or "")[-500:])
    return "exit 0"


@check("真实应用入口可渲染界面（--preview 生成 PNG）")
def _app_preview():
    png = RUNS / "app_preview.png"
    png.parent.mkdir(parents=True, exist_ok=True)
    if png.exists():
        png.unlink()
    proc = _run_child(["-m", "ai_student", "--preview", str(png), "--no-tray"], timeout=180)
    assert proc.returncode == 0, "exit=%s stderr=%s" % (proc.returncode, (proc.stderr or "")[-500:])
    assert png.exists() and png.stat().st_size > 5000, "截图未生成或过小"
    return "%s（%d 字节）" % (png, png.stat().st_size)


# ------------------------------------------------------------------ 引擎全链路
@check("引擎离线全流程（合成课堂 -> 转写 -> 笔记 -> 任务 -> 作业产出）")
def _engine():
    out = RUNS / "demo_out"
    proc = _run_child(["tools/demo_offline.py", "--out", str(out)], timeout=300)
    assert proc.returncode == 0, "exit=%s tail=%s" % (
        proc.returncode,
        (proc.stdout or "")[-600:],
    )
    files = [p for p in out.rglob("*") if p.is_file()] if out.exists() else []
    assert any(p.suffix == ".docx" for p in files), "没有生成 docx 作业"
    assert any("作业汇总" in p.name for p in files), "没有生成作业汇总"
    assert any(p.suffix == ".md" for p in files), "没有生成 md 作业"
    return "产出 %d 个文件 -> %s" % (len(files), out)


@check("采集层真机自检（枚举 + 6 秒试录，允许无麦克风）")
def _capture():
    proc = _run_child(
        ["tools/probe_capture.py", "--record", "6", "--session-dir", str(RUNS / "probe")],
        timeout=240,
    )
    tail = (proc.stdout or "").strip().splitlines()
    assert proc.returncode == 0, "exit=%s tail=%s" % (proc.returncode, chr(10).join(tail[-8:]))
    media = [ln.strip() for ln in tail if "audio.wav" in ln or "video.mp4" in ln]
    return " | ".join(media[:2]) or "exit 0"


# ------------------------------------------------------------------ 主流程
def main() -> int:
    RUNS.mkdir(parents=True, exist_ok=True)
    print("=" * 78)
    print("AI学生 自检")
    print("=" * 78)
    passed = 0
    for name, fn in CHECKS:
        try:
            detail = str(fn() or "")
            ok = True
        except Exception as exc:
            ok = False
            detail = "%s | %s" % (exc, traceback.format_exc(limit=2).splitlines()[-1])
        print(("[PASS] " if ok else "[FAIL] ") + name)
        if detail:
            for line in detail.splitlines():
                print("       " + line)
        passed += 1 if ok else 0
    total = len(CHECKS)
    print("-" * 78)
    print("通过 %d/%d" % (passed, total))
    return 0 if passed == total else 1


if __name__ == "__main__":
    raise SystemExit(main())
