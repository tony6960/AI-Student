r"""端到端：用本地视频当"极域演示"，跑完整的 AI 学生链路（真采集 + 真模型）。

流程与真实上课完全一致，只是演示源换成了视频文件：
    tools/play_video.py 播放视频（窗口标题带"极域"）
    -> 本脚本启动 AppController（真实引擎）
    -> 窗口抓屏 + 系统回环录音 -> 实时语音识别 -> 画面理解 -> 笔记 -> 任务 -> 作业

用法：
    set PYTHONPATH=%CD%\src
    .venv\Scripts\python.exe tools\demo_video_class.py --seconds 300
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from PySide6.QtCore import QTimer                                  # noqa: E402
from PySide6.QtWidgets import QApplication                         # noqa: E402

from ai_student.core.controller import AppController               # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", default=str(ROOT / ".runs" / "video" / "lecture.mp4"))
    ap.add_argument("--start", type=float, default=60.0)
    ap.add_argument("--seconds", type=float, default=300.0)
    ap.add_argument("--out", default="")
    ap.add_argument("--no-player", action="store_true", help="不自己起播放器（外部已经在放）")
    args = ap.parse_args(argv)

    if not args.no_player:
        player = subprocess.Popen(
            [sys.executable, str(ROOT / "tools" / "play_video.py"),
             "--file", args.file, "--start", str(args.start), "--duration", str(args.seconds + 20)],
            cwd=str(ROOT),
        )
        print("[demo] 播放器已启动 pid=%d，等待窗口就绪…" % player.pid, flush=True)
        time.sleep(8)
    else:
        player = None

    app = QApplication(sys.argv[:1])
    controller = AppController()

    if args.out:
        cfg = controller.get_config()
        cfg.output.output_dir = args.out
        controller.update_config(cfg)

    started = time.time()
    stats = {"transcript": 0, "board": 0, "tasks": 0, "artifacts": []}

    def stamp() -> str:
        return "[+%5.1fs] " % (time.time() - started)

    controller.stateChanged.connect(lambda s, d: print(stamp() + "[状态] " + s + " " + d, flush=True))
    controller.stageChanged.connect(lambda s, d: print(stamp() + "  · " + s + "：" + d, flush=True))
    controller.logMessage.connect(lambda lvl, m: print(stamp() + "  " + lvl + " " + str(m)[:240], flush=True))
    controller.errorRaised.connect(lambda t, d: print(stamp() + "[错误] " + t + " " + str(d)[:200], flush=True))

    def on_transcript(seg):
        stats["transcript"] += 1
        text = (seg.get("text") or "").strip()
        print(stamp() + "<" + seg.get("timecode", "") + "> " + (text or "（静音/无语音）")[:160], flush=True)

    def on_board(note):
        stats["board"] += 1
        print(stamp() + "[画面] " + ((note.get("summary") or note.get("ocr_text") or "")[:120]), flush=True)

    def on_task(task):
        stats["tasks"] += 1
        print(stamp() + "[任务] " + task.get("title", "") + " (" + task.get("status", "") + ")", flush=True)

    def on_artifact(art):
        stats["artifacts"].append(art.get("path", ""))
        print(stamp() + "[产出] " + Path(art.get("path", "")).name, flush=True)

    controller.transcriptSegment.connect(on_transcript)
    controller.boardNote.connect(on_board)
    controller.taskDiscovered.connect(on_task)
    controller.taskUpdated.connect(on_task)
    controller.artifactWritten.connect(on_artifact)

    print("=" * 76)
    print("AI学生 · 真实链路测试（演示源：%s，起始 %.0fs，共 %.0fs）"
          % (Path(args.file).name, args.start, args.seconds))
    print("=" * 76)

    try:
        session_id = controller.start_session(subject="Python 编程", title="B站公开课实测")
        print(stamp() + "会话已开始：" + session_id, flush=True)
    except Exception as exc:
        print("[错误] 无法开始听课：" + str(exc), flush=True)
        if player:
            player.terminate()
        return 2

    QTimer.singleShot(int(args.seconds * 1000), app.quit)
    app.exec()

    print(stamp() + "播放结束，开始收尾（整理笔记 / 完成任务 / 导出作业）…", flush=True)
    controller.stop_session(finalize=True)
    deadline = time.time() + 600
    while controller.engine_state() not in ("finished", "error") and time.time() < deadline:
        app.processEvents()
        time.sleep(0.3)
    app.processEvents()

    if player:
        player.terminate()

    print("")
    print("=" * 76)
    print("结果统计")
    print("=" * 76)
    print("  转写段数：%d" % stats["transcript"])
    print("  画面理解：%d" % stats["board"])
    print("  任务事件：%d" % stats["tasks"])
    print("  产出文件：%d" % len(stats["artifacts"]))
    bundle = controller.current_bundle()
    if bundle is not None:
        print("  素材目录：%s" % bundle.info.directory)
        print("  输出目录：%s" % bundle.info.output_dir)
        print("")
        print("  识别到的任务：")
        for t in bundle.tasks:
            print("    - [%s] %s（自动重做 %d 次）" % (t.status, t.title, t.auto_revised))
            for a in t.artifacts:
                print("        " + a)
    out_dir = Path(controller.get_config().output.output_dir)
    if out_dir.exists():
        print("")
        print("  输出目录内容：")
        for p in sorted(out_dir.rglob("*")):
            if p.is_file():
                print("    %s  (%d 字节)" % (p.relative_to(out_dir), p.stat().st_size))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
