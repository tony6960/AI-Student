"""采集层自检工具（命令行）。

用法（项目根目录，务必用项目自带解释器）：

    $env:PYTHONPATH='D:\Code\MixProject\AI学生\src'
    & 'D:\Code\MixProject\AI学生\.venv\Scripts\python.exe' tools\probe_capture.py
    & 'D:\Code\MixProject\AI学生\.venv\Scripts\python.exe' tools\probe_capture.py --record 6

不带参数：打印音频设备 / 显示器 / 顶层窗口 / 极域候选窗口 / ffmpeg 路径。
带 --record N：在临时目录（或 --session-dir 指定目录）真实录制 N 秒音频 + 画面，
结束后打印产物路径、时长、字节数、关键帧与切片数量。

退出码：0 = 至少产出一个媒体文件（或纯探测成功）；1 = 录制什么都没产出。
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from ai_student.capture import (  # noqa: E402
    AudioRecorder,
    ScreenRecorder,
    describe_environment,
    ffmpeg_available,
    ffmpeg_path,
    ffmpeg_version,
    probe_video,
    wav_duration,
)
from ai_student.config.models import CaptureConfig  # noqa: E402


def out(text: str = "") -> None:
    """控制台是 GBK：遇到无法编码的字符时降级替换，绝不因打印而崩。"""
    try:
        print(text, flush=True)
    except UnicodeEncodeError:
        encoding = getattr(sys.stdout, "encoding", None) or "utf-8"
        safe = text.encode(encoding, errors="replace").decode(encoding, errors="replace")
        print(safe, flush=True)


def human_size(num: int) -> str:
    size = float(max(0, int(num)))
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024 or unit == "GB":
            return f"{size:.0f} {unit}" if unit == "B" else f"{size:.1f} {unit}"
        size /= 1024.0
    return f"{size:.1f} GB"


def print_environment(as_json: bool = False) -> dict:
    env = describe_environment()
    if as_json:
        out(json.dumps(env, ensure_ascii=False, indent=2))
        return env
    out("=" * 68)
    out("AI学生 采集层自检")
    out("=" * 68)
    out(f"ffmpeg 可用：{'是' if ffmpeg_available() else '否'}")
    out(f"ffmpeg 路径：{ffmpeg_path()}")
    out(f"ffmpeg 版本：{ffmpeg_version() or '（无法获取）'}")

    out("")
    out(f"-- 音频设备（{len(env['audio'])} 个可录音端点） --")
    if not env["audio"]:
        out("  （无：可能是没有麦克风或 PyAudioWPatch 不可用）")
    for dev in env["audio"]:
        flags = []
        if dev["is_loopback"]:
            flags.append("回环")
        if dev["is_default_input"]:
            flags.append("默认输入")
        if dev["is_default_output"]:
            flags.append("默认输出")
        tail = f"  [{'/'.join(flags)}]" if flags else ""
        out(
            f"  [{dev['index']:>2}] {dev['name']}  "
            f"{dev['channels']}ch {dev['sample_rate']}Hz {dev['host_api']}{tail}"
        )

    out("")
    out(f"-- 显示器（{len(env['monitors'])} 个） --")
    for mon in env["monitors"]:
        mark = " 主屏" if mon["primary"] else ""
        out(
            f"  [{mon['index']}] {mon['width']}x{mon['height']} "
            f"@ ({mon['left']},{mon['top']}){mark}"
        )

    out("")
    out(f"-- 顶层窗口（{len(env['windows'])} 个，仅列前 15） --")
    for win in env["windows"][:15]:
        state = " 最小化" if win["minimized"] else ""
        out(f"  {win['width']}x{win['height']:<5} {win['process']:<20} {win['title'][:40]}{state}")

    out("")
    out(f"-- 极域电子教室候选窗口（{len(env['classroom_windows'])} 个） --")
    if not env["classroom_windows"]:
        out("  （未找到，画面采集会自动回退整屏）")
    for win in env["classroom_windows"]:
        out(f"  #{win['rank']} {win['title']}  ({win['width']}x{win['height']}) {win['process']}")
    out("")
    return env


def record(args: argparse.Namespace) -> int:
    seconds = float(args.record)
    session_dir = (
        Path(args.session_dir).expanduser().resolve()
        if args.session_dir
        else Path(tempfile.mkdtemp(prefix="ai_student_probe_"))
    )
    session_dir.mkdir(parents=True, exist_ok=True)

    cfg = CaptureConfig()
    cfg.sample_rate = int(args.sample_rate)
    cfg.channels = int(args.channels)
    cfg.chunk_seconds = max(1, int(args.chunk))
    cfg.keyframe_interval_s = max(1, int(args.keyframe_interval))
    cfg.video_fps = max(1, int(args.fps))
    cfg.record_video = not args.no_video
    cfg.screen_enabled = not args.no_screen
    cfg.mic_enabled = not args.no_mic
    cfg.loopback_enabled = not args.no_loopback
    if args.source:
        cfg.screen_source = args.source
    if args.mic_device:
        cfg.mic_device = args.mic_device
    if args.loopback_device:
        cfg.loopback_device = args.loopback_device

    out("=" * 68)
    out(f"开始录制 {seconds:g} 秒")
    out(f"会话目录：{session_dir}")
    out(
        f"配置：{cfg.sample_rate}Hz/{cfg.channels}ch 切片{cfg.chunk_seconds}s "
        f"画面={cfg.screen_source} 视频={'开' if cfg.record_video else '关'} "
        f"{cfg.video_fps}fps 关键帧间隔{cfg.keyframe_interval_s}s"
    )
    out("-" * 68)

    chunks: list = []
    keyframes: list = []
    problems: list[str] = []

    def on_error(message: str) -> None:
        problems.append(message)
        out(f"  [提示] {message}")

    def on_chunk(chunk) -> None:
        chunks.append(chunk)
        out(
            f"  [切片 {chunk.index}] {chunk.t_start:.1f}-{chunk.t_end:.1f}s "
            f"RMS {chunk.rms_db:.1f}dB 峰值 {chunk.peak_db:.1f}dB "
            f"{'静音' if chunk.silent else '有声'} -> {chunk.path.name}"
        )

    def on_keyframe(frame) -> None:
        keyframes.append(frame)
        out(f"  [关键帧 {frame.index}] t={frame.t:.1f}s 变化度 {frame.hash_distance:.3f} -> {frame.path.name}")

    audio = AudioRecorder(cfg, session_dir, on_chunk=on_chunk, on_error=on_error)
    screen = ScreenRecorder(cfg, session_dir, on_keyframe=on_keyframe, on_error=on_error)

    audio_started = False
    if not args.no_audio:
        try:
            audio.start()
            audio_started = True
            out(f"音频已开始（来源：{audio.source_kind}）")
        except RuntimeError as exc:
            problems.append(str(exc))
            out(f"  [提示] 音频不可用，已跳过：{exc}")

    screen_started = False
    if not args.no_screen:
        try:
            screen.start()
            screen_started = True
            source = screen.source
            out(f"画面已开始（{source.get('kind')}）：{source.get('reason')}")
        except Exception as exc:  # 画面失败也不该中断自检
            problems.append(f"画面采集启动失败：{exc}")
            out(f"  [提示] 画面采集启动失败：{exc}")

    if not audio_started and not screen_started:
        out("音频与画面都未能启动，自检失败。")
        return 1

    started = time.monotonic()
    try:
        while time.monotonic() - started < seconds:
            time.sleep(0.1)
    except KeyboardInterrupt:
        out("收到中断，提前停止。")

    audio_path = audio.stop() if audio_started else None
    video_path = screen.stop() if screen_started else None

    out("-" * 68)
    out("录制结束，产物：")
    result: dict = {
        "session_dir": str(session_dir),
        "audio": None,
        "video": None,
        "keyframes": len(keyframes),
        "chunks": len(chunks),
        "problems": problems,
    }

    if audio_path is not None and Path(audio_path).exists():
        duration = wav_duration(audio_path)
        size = Path(audio_path).stat().st_size
        silent = sum(1 for c in chunks if c.silent)
        out(f"  音频：{audio_path}")
        out(
            f"        时长 {duration:.2f}s  大小 {size} 字节（{human_size(size)}）  "
            f"切片 {len(chunks)} 个（静音 {silent} 个）"
        )
        result["audio"] = {
            "path": str(audio_path),
            "duration": round(duration, 3),
            "bytes": size,
            "chunks": len(chunks),
            "silent_chunks": silent,
            "source": audio.source_kind,
        }
        for index in range(min(3, len(chunks))):
            chunk = chunks[index]
            chunk_size = Path(chunk.path).stat().st_size if Path(chunk.path).exists() else 0
            out(
                f"        切片{chunk.index}：{chunk.path.name} "
                f"{wav_duration(chunk.path):.2f}s {chunk_size} 字节 "
                f"RMS {chunk.rms_db:.1f}dB"
            )
    else:
        out("  音频：未产出（设备不可用或全部静音）")

    if video_path is not None and Path(video_path).exists():
        info = probe_video(video_path)
        size = Path(video_path).stat().st_size
        out(f"  视频：{video_path}")
        out(
            f"        时长 {info['duration']:.2f}s  {info['width']}x{info['height']} "
            f"@{info['fps']:.2f}fps  大小 {size} 字节（{human_size(size)}）"
        )
        out(f"        关键帧 {len(keyframes)} 张  录像帧 {screen.frame_count} 帧")
        result["video"] = {
            "path": str(video_path),
            "duration": round(float(info["duration"]), 3),
            "fps": round(float(info["fps"]), 3),
            "width": int(info["width"]),
            "height": int(info["height"]),
            "bytes": size,
            "keyframes": len(keyframes),
        }
        keyframe_dir = Path(session_dir) / "screen" / "keyframes"
        if keyframe_dir.is_dir():
            for kf in sorted(keyframe_dir.glob("kf_*.jpg"))[:3]:
                out(f"        关键帧文件：{kf}  {kf.stat().st_size} 字节")
    else:
        out("  视频：未产出（未开录制或没有抓到帧）")

    if args.json:
        out(json.dumps(result, ensure_ascii=False, indent=2))
    out("=" * 68)
    if not result["audio"] and not result["video"]:
        return 1
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="probe_capture.py",
        description="AI学生 采集层自检（音频设备/显示器/窗口/ffmpeg + 真实录制验证）",
    )
    parser.add_argument("--record", type=float, default=0.0, help="录制 N 秒音频+画面（默认只探测）")
    parser.add_argument("--session-dir", default="", help="录制产物目录（默认系统临时目录）")
    parser.add_argument("--json", action="store_true", help="额外输出 JSON 摘要")
    parser.add_argument("--source", choices=["window", "region", "fullscreen"], help="画面来源")
    parser.add_argument("--no-audio", action="store_true", help="不录音频")
    parser.add_argument("--no-screen", action="store_true", help="不录画面")
    parser.add_argument("--no-video", action="store_true", help="不合成视频（只留关键帧）")
    parser.add_argument("--no-mic", action="store_true", help="关闭麦克风")
    parser.add_argument("--no-loopback", action="store_true", help="关闭系统回环")
    parser.add_argument("--mic-device", default="", help="麦克风设备名子串")
    parser.add_argument("--loopback-device", default="", help="回环设备名子串")
    parser.add_argument("--sample-rate", type=int, default=16000, help="目标采样率（默认 16000）")
    parser.add_argument("--channels", type=int, default=1, help="目标声道数（默认 1）")
    parser.add_argument("--chunk", type=int, default=5, help="音频切片秒数（默认 5）")
    parser.add_argument("--fps", type=int, default=2, help="录像帧率（默认 2）")
    parser.add_argument("--keyframe-interval", type=int, default=8, help="关键帧间隔秒（默认 8）")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    print_environment(as_json=args.json and args.record <= 0)
    if args.record > 0:
        return record(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
