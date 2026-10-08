"""ffmpeg 封装。

优先使用 imageio-ffmpeg 自带的静态 ffmpeg.exe（免安装、免配置 PATH），
取不到时回退到系统 PATH 中的 "ffmpeg"。

约定：本模块的函数**不抛异常**（除非显式说明），失败一律返回 None / 空值并记日志，
避免采集层把 GUI 进程带崩。
"""
from __future__ import annotations

import logging
import os
import re
import subprocess
import tempfile
import wave
from pathlib import Path

log = logging.getLogger(__name__)

_CREATE_NO_WINDOW = 0x08000000 if os.name == "nt" else 0

_DURATION_RE = re.compile(r"Duration:\s*(\d+):(\d{2}):(\d{2}(?:\.\d+)?)")
_SIZE_RE = re.compile(r"(\d{2,5})x(\d{2,5})")
_FPS_RE = re.compile(r"([\d.]+)\s*fps")


# ------------------------------------------------------------------ 可执行文件
def ffmpeg_path() -> str:
    """返回可用的 ffmpeg 可执行文件路径（可能只是 "ffmpeg"，交给 PATH 解析）。"""
    try:
        import imageio_ffmpeg

        exe = imageio_ffmpeg.get_ffmpeg_exe()
        if exe:
            return str(exe)
    except Exception as exc:  # pragma: no cover - 依赖缺失时才走到
        log.debug("imageio_ffmpeg 不可用：%s", exc)
    return "ffmpeg"


def run_ffmpeg(args: list[str], timeout: float | None = 300.0) -> subprocess.CompletedProcess:
    """执行 ffmpeg 命令（args 不需要包含可执行文件本身）。

    已自动加上 -nostdin -hide_banner；超时/找不到可执行文件时**不抛异常**，
    而是返回 returncode 为负数的 CompletedProcess，方便调用方统一判断。
    """
    cmd: list[str] = [ffmpeg_path()]
    text_args = [str(a) for a in args]
    if "-nostdin" not in text_args:
        cmd.append("-nostdin")
    if "-hide_banner" not in text_args:
        cmd.append("-hide_banner")
    cmd.extend(text_args)
    try:
        return subprocess.run(
            cmd,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            creationflags=_CREATE_NO_WINDOW,
        )
    except subprocess.TimeoutExpired:
        log.warning("ffmpeg 执行超时（%.0fs）：%s", timeout or 0, " ".join(text_args))
        return subprocess.CompletedProcess(cmd, -9, "", "ffmpeg 执行超时")
    except Exception as exc:
        log.warning("无法执行 ffmpeg：%s", exc)
        return subprocess.CompletedProcess(cmd, -2, "", f"无法执行 ffmpeg：{exc}")


def ffmpeg_available() -> bool:
    """ffmpeg 是否可执行。"""
    proc = run_ffmpeg(["-version"], timeout=20.0)
    return proc.returncode == 0


def ffmpeg_version() -> str:
    """ffmpeg 版本首行；不可用时返回空串。"""
    proc = run_ffmpeg(["-version"], timeout=20.0)
    if proc.returncode != 0:
        return ""
    lines = [ln.strip() for ln in (proc.stdout or "").splitlines() if ln.strip()]
    return lines[0] if lines else ""


# ------------------------------------------------------------------ 视频合成
def encode_frames_to_mp4(
    frame_dir: Path,
    fps: int,
    out_path: Path,
    crf: int = 28,
    pattern: str = "f_%06d.jpg",
) -> Path | None:
    """把图片序列合成为 H.264 mp4；失败返回 None（不抛异常）。"""
    frame_dir = Path(frame_dir)
    out_path = Path(out_path)
    if not frame_dir.is_dir():
        log.warning("帧目录不存在：%s", frame_dir)
        return None
    try:
        if not any(frame_dir.glob("*.jpg")):
            log.warning("帧目录里没有可用的 jpg 帧：%s", frame_dir)
            return None
    except Exception as exc:
        log.warning("扫描帧目录失败：%s", exc)
        return None

    rate = max(1, int(fps or 1))
    try:
        out_path.parent.mkdir(parents=True, exist_ok=True)
    except Exception as exc:
        log.warning("创建视频输出目录失败：%s", exc)
        return None

    args = [
        "-y",
        "-framerate", str(rate),
        "-start_number", "1",
        "-i", str(frame_dir / pattern),
        "-c:v", "libx264",
        "-preset", "veryfast",
        "-crf", str(int(crf)),
        "-pix_fmt", "yuv420p",
        # H.264 要求宽高为偶数，用 pad 补齐（不缩放，避免画面变形）
        "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",
        "-r", str(rate),
        "-movflags", "+faststart",
        str(out_path),
    ]
    proc = run_ffmpeg(args, timeout=3600.0)
    if proc.returncode != 0 or not out_path.exists() or out_path.stat().st_size <= 0:
        log.warning("合成视频失败（%s）：%s", proc.returncode, (proc.stderr or "")[-400:])
        return None
    return out_path


def mux_audio_video(video: Path, audio: Path, out_path: Path) -> Path | None:
    """把音轨混进视频（视频流直接 copy）；失败返回 None。"""
    video = Path(video)
    audio = Path(audio)
    out_path = Path(out_path)
    if not video.exists() or not audio.exists():
        log.warning("合流输入缺失：video=%s audio=%s", video.exists(), audio.exists())
        return None
    try:
        out_path.parent.mkdir(parents=True, exist_ok=True)
    except Exception as exc:
        log.warning("创建合流输出目录失败：%s", exc)
        return None
    args = [
        "-y",
        "-i", str(video),
        "-i", str(audio),
        "-c:v", "copy",
        "-c:a", "aac",
        "-b:a", "128k",
        "-shortest",
        str(out_path),
    ]
    proc = run_ffmpeg(args, timeout=3600.0)
    if proc.returncode != 0 or not out_path.exists() or out_path.stat().st_size <= 0:
        log.warning("音视频合流失败（%s）：%s", proc.returncode, (proc.stderr or "")[-400:])
        return None
    return out_path


# ------------------------------------------------------------------ 音频工具
def wav_duration(path: Path) -> float:
    """WAV 时长（秒）；读不出来返回 0.0。"""
    path = Path(path)
    try:
        with wave.open(str(path), "rb") as handle:
            rate = handle.getframerate() or 0
            if rate > 0:
                return handle.getnframes() / float(rate)
    except Exception:
        pass
    try:  # 非 PCM wav 时用 soundfile
        import soundfile as sf

        info = sf.info(str(path))
        if info.samplerate:
            return float(info.frames) / float(info.samplerate)
    except Exception:
        pass
    try:  # 最后回退：解析 ffmpeg 输出
        proc = run_ffmpeg(["-i", str(path)], timeout=60.0)
        match = _DURATION_RE.search((proc.stderr or "") + (proc.stdout or ""))
        if match:
            return int(match.group(1)) * 3600 + int(match.group(2)) * 60 + float(match.group(3))
    except Exception:
        pass
    return 0.0


def concat_wavs(paths: list[Path], out_path: Path) -> Path | None:
    """按顺序拼接多个 WAV（参数一致时用标准库直拼，否则退回 ffmpeg）；失败返回 None。"""
    items = [Path(p) for p in paths if Path(p).exists()]
    out_path = Path(out_path)
    if not items:
        return None
    try:
        out_path.parent.mkdir(parents=True, exist_ok=True)
    except Exception as exc:
        log.warning("创建拼接输出目录失败：%s", exc)
        return None
    if len(items) == 1:
        try:
            out_path.write_bytes(items[0].read_bytes())
            return out_path
        except Exception as exc:
            log.warning("复制单个 WAV 失败：%s", exc)
            return None
    try:
        with wave.open(str(items[0]), "rb") as first:
            params = first.getparams()
        with wave.open(str(out_path), "wb") as dest:
            dest.setparams(params)
            for item in items:
                with wave.open(str(item), "rb") as src:
                    if (src.getnchannels(), src.getsampwidth(), src.getframerate()) != (
                        params.nchannels,
                        params.sampwidth,
                        params.framerate,
                    ):
                        raise ValueError("WAV 参数不一致")
                    dest.writeframes(src.readframes(src.getnframes()))
        if out_path.stat().st_size > 44:
            return out_path
    except Exception as exc:
        log.debug("标准库拼接 WAV 失败（改用 ffmpeg）：%s", exc)

    list_file = Path(tempfile.mkdtemp(prefix="ai_student_wav_")) / "list.txt"
    try:
        lines = []
        for item in items:
            safe = str(item).replace("'", "'\\''")
            lines.append(f"file '{safe}'")
        list_file.write_text("\n".join(lines) + "\n", encoding="utf-8")
    except Exception as exc:
        log.warning("写 WAV 拼接清单失败：%s", exc)
        return None
    args = [
        "-y",
        "-f", "concat",
        "-safe", "0",
        "-i", str(list_file),
        "-c", "copy",
        str(out_path),
    ]
    proc = run_ffmpeg(args, timeout=1800.0)
    if proc.returncode != 0 or not out_path.exists() or out_path.stat().st_size <= 0:
        log.warning("拼接 WAV 失败（%s）：%s", proc.returncode, (proc.stderr or "")[-400:])
        return None
    return out_path


# ------------------------------------------------------------------ 探测
def probe_video(path: Path) -> dict:
    """读取视频信息 -> {"duration": float, "fps": float, "width": int, "height": int}。

    读不出来时返回全 0 的字典（不抛异常）。
    """
    info = {"duration": 0.0, "fps": 0.0, "width": 0, "height": 0}
    path = Path(path)
    if not path.exists():
        return info
    proc = run_ffmpeg(["-i", str(path)], timeout=120.0)
    text = (proc.stderr or "") + "\n" + (proc.stdout or "")
    match = _DURATION_RE.search(text)
    if match:
        info["duration"] = (
            int(match.group(1)) * 3600 + int(match.group(2)) * 60 + float(match.group(3))
        )
    for line in text.splitlines():
        if "Video:" not in line:
            continue
        sizes = _SIZE_RE.findall(line)
        if sizes:
            info["width"], info["height"] = int(sizes[-1][0]), int(sizes[-1][1])
        fps_match = _FPS_RE.search(line)
        if fps_match:
            try:
                info["fps"] = float(fps_match.group(1))
            except ValueError:
                pass
        break
    return info


def probe_wav(path: Path) -> dict:
    """WAV 简要信息 -> {"duration": float, "sample_rate": int, "channels": int}。"""
    info = {"duration": wav_duration(path), "sample_rate": 0, "channels": 0}
    try:
        with wave.open(str(Path(path)), "rb") as handle:
            info["sample_rate"] = int(handle.getframerate())
            info["channels"] = int(handle.getnchannels())
    except Exception:
        pass
    return info
