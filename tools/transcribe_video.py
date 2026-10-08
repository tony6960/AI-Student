r"""把一个视频文件的音轨整段转写成带时间码的文字稿。

用途：
1) 在没有教室的情况下，快速了解视频/录音里老师讲了什么，便于挑选测试片段；
2) 验证语音识别在长音频上的稳定性与准确度。

用法：
    set PYTHONPATH=%CD%\src
    .venv\Scripts\python.exe tools\transcribe_video.py --file .runs\video\lecture.mp4 --out .runs\video\transcript.md
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
import wave
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from ai_student.ai.asr import transcribe_chunk                     # noqa: E402
from ai_student.ai.client import LLMClient                         # noqa: E402
from ai_student.capture.ffmpeg import ffmpeg_path                  # noqa: E402
from ai_student.config.store import load_config                    # noqa: E402


def extract_wav(video: Path, out: Path, start: float, duration: float) -> None:
    ff = ffmpeg_path()
    cmd = [ff, "-y", "-hide_banner", "-loglevel", "error"]
    if start > 0:
        cmd += ["-ss", str(start)]
    if duration > 0:
        cmd += ["-t", str(duration)]
    cmd += ["-i", str(video), "-vn", "-ac", "1", "-ar", "16000", "-acodec", "pcm_s16le", str(out)]
    subprocess.run(cmd, check=True, capture_output=True)


def split_wav(wav: Path, out_dir: Path, seconds: float) -> list[tuple[float, Path]]:
    out_dir.mkdir(parents=True, exist_ok=True)
    with wave.open(str(wav), "rb") as fh:
        rate = fh.getframerate()
        channels = fh.getnchannels()
        width = fh.getsampwidth()
        per = int(rate * seconds)
        parts: list[tuple[float, Path]] = []
        index = 0
        while True:
            data = fh.readframes(per)
            if not data:
                break
            path = out_dir / ("part_%04d.wav" % index)
            with wave.open(str(path), "wb") as out:
                out.setnchannels(channels)
                out.setsampwidth(width)
                out.setframerate(rate)
                out.writeframes(data)
            parts.append((index * seconds, path))
            index += 1
    return parts


def fmt(seconds: float) -> str:
    s = int(seconds)
    return "%02d:%02d:%02d" % (s // 3600, (s % 3600) // 60, s % 60)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", required=True)
    ap.add_argument("--out", default="")
    ap.add_argument("--start", type=float, default=0.0)
    ap.add_argument("--duration", type=float, default=0.0, help="0 表示整段")
    ap.add_argument("--chunk", type=float, default=15.0)
    args = ap.parse_args(argv)

    video = Path(args.file).resolve()
    if not video.exists():
        print("找不到文件：%s" % video)
        return 2
    work = ROOT / ".runs" / "asr_batch"
    work.mkdir(parents=True, exist_ok=True)
    wav = work / "full.wav"
    print("[1/3] 抽取音轨 …", flush=True)
    extract_wav(video, wav, args.start, args.duration)
    parts = split_wav(wav, work / "parts", args.chunk)
    print("[2/3] 共 %d 个切片，开始识别 …" % len(parts), flush=True)

    cfg = load_config()
    client = LLMClient(cfg.model)
    lines: list[str] = []
    t0 = time.time()
    for i, (offset, path) in enumerate(parts, 1):
        stamp = fmt(args.start + offset)
        try:
            text = transcribe_chunk(client, path, cfg.model)
        except Exception as exc:
            text = "（识别失败：%s）" % str(exc)[:80]
        lines.append("- **[%s]** %s" % (stamp, (text or "（无语音）").strip()))
        if i % 5 == 0 or i == len(parts):
            print("    %d/%d  用时 %.0fs" % (i, len(parts), time.time() - t0), flush=True)

    header = "# 视频文字稿\n\n- 文件：%s\n- 起点：%s\n- 切片：%.0f 秒\n\n" % (
        video.name, fmt(args.start), args.chunk)
    out = Path(args.out) if args.out else (work / "transcript.md")
    out.write_text(header + "\n".join(lines) + "\n", encoding="utf-8")
    print("[3/3] 已写入 %s（%d 行）" % (out, len(lines)), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
