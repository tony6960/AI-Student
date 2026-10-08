"""演示画面采集：极域电子教室窗口定位 + 区域抓屏 + dHash 关键帧去重 + 帧序列合成视频。

要点：
- **mss 实例必须在采集线程内创建**（跨线程用会崩）；窗口定位（pywin32）在调用线程
  （通常是 GUI 主线程）解析好，只把 rect 传进采集线程。
- 关键帧：每 cfg.keyframe_interval_s 秒抓一帧，与上一张关键帧比 dHash 汉明距离比例，
  超过 cfg.keyframe_change_threshold 才落盘并回调；会话第一帧永远算关键帧。
- 录像帧写 <session>/screen/frames/f_%06d.jpg，stop() 时用 ffmpeg 合成
  <session>/screen/video.mp4（H.264 + crf），成功后删除 frames 目录。
- 窗口被最小化/关闭时退回整屏抓取，只警告一次，不中断录制。
- stop() 幂等；后台线程异常一律吞掉并回调 on_error，绝不让进程崩。
"""
from __future__ import annotations

import ctypes
import io
import logging
import os
import shutil
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
from PIL import Image

from ..config.models import CaptureConfig
from .devices import (
    ensure_dpi_aware,
    find_classroom_windows,
    list_monitors,
    new_mss,
    window_rect,
)
from .ffmpeg import encode_frames_to_mp4

log = logging.getLogger(__name__)

ensure_dpi_aware()  # 模块导入即声明 DPI 感知，保证抓屏拿到物理像素

_JOIN_TIMEOUT = 20.0
_READY_TIMEOUT = 5.0
_PREVIEW_WIDTH = 640
_MIN_GRAB_SIDE = 16


# ------------------------------------------------------------------ dHash
def _to_pil(image) -> Image.Image:
    """把 numpy HxWx3/HxWx4 uint8 或 PIL.Image 统一成 PIL.Image（RGB）。"""
    if isinstance(image, Image.Image):
        return image.convert("RGB") if image.mode != "RGB" else image
    arr = np.asarray(image)
    if arr.ndim == 2:
        return Image.fromarray(arr.astype(np.uint8), mode="L").convert("RGB")
    if arr.ndim == 3 and arr.shape[2] >= 3:
        return Image.fromarray(arr[:, :, :3].astype(np.uint8), mode="RGB")
    raise ValueError("不支持的图像类型")


def dhash(image, hash_size: int = 16) -> int:
    """差异哈希：横向相邻像素比较，返回 hash_size*hash_size 位的整数。"""
    size = max(1, int(hash_size))
    gray = _to_pil(image).convert("L").resize((size + 1, size), Image.LANCZOS)
    arr = np.asarray(gray, dtype=np.int16)
    diff = arr[:, 1:] > arr[:, :-1]           # (size, size) bool
    packed = np.packbits(diff.reshape(-1))
    return int.from_bytes(packed.tobytes(), "big")


def hamming(a: int, b: int) -> int:
    """两个整数哈希的汉明距离（不同位数）。"""
    return int(a ^ b).bit_count()


def dhash_distance(a: int, b: int, hash_size: int = 16) -> float:
    """归一化汉明距离比例（0~1）。"""
    bits = max(1, int(hash_size) * int(hash_size))
    return hamming(a, b) / float(bits)


def should_keep_keyframe(
    prev_hash: int | None,
    cur_hash: int,
    threshold: float = 0.05,
    hash_size: int = 16,
) -> tuple[bool, float]:
    """关键帧去重判定（纯函数）：返回 (是否落盘, 距离比例)。

    prev_hash 为 None（会话第一帧）时永远保留，距离记为 1.0。
    """
    if prev_hash is None:
        return True, 1.0
    distance = dhash_distance(prev_hash, cur_hash, hash_size)
    return bool(distance > float(threshold)), distance


# ------------------------------------------------------------------ 数据类
@dataclass
class FrameCapture:
    """一张已落盘的关键帧。"""

    index: int
    t: float
    path: Path
    changed: bool
    hash_distance: float

    def to_dict(self) -> dict:
        return {
            "index": self.index,
            "t": round(self.t, 3),
            "path": str(self.path),
            "changed": self.changed,
            "hash_distance": round(self.hash_distance, 4),
        }


# ------------------------------------------------------------------ 来源解析
def _monitor_by_index(monitors: list[dict], index: int) -> dict:
    for mon in monitors:
        if int(mon.get("index", -1)) == int(index):
            return mon
    for mon in monitors:
        if mon.get("primary"):
            return mon
    return monitors[0] if monitors else {"index": 1, "left": 0, "top": 0, "width": 1920, "height": 1080}


def _monitor_for_rect(monitors: list[dict], rect: list[int]) -> int:
    """rect=[l,t,w,h] 落在哪个显示器上（按中心点，其次按最大重叠面积）。"""
    if not monitors:
        return 1
    left, top, width, height = (int(v) for v in rect[:4])
    cx, cy = left + width / 2.0, top + height / 2.0
    for mon in monitors:
        if (
            mon["left"] <= cx < mon["left"] + mon["width"]
            and mon["top"] <= cy < mon["top"] + mon["height"]
        ):
            return int(mon["index"])
    best, best_area = monitors[0], -1.0
    for mon in monitors:
        ox = max(0, min(left + width, mon["left"] + mon["width"]) - max(left, mon["left"]))
        oy = max(0, min(top + height, mon["top"] + mon["height"]) - max(top, mon["top"]))
        area = float(ox) * float(oy)
        if area > best_area:
            best, best_area = mon, area
    return int(best["index"])


class ScreenSourceResolver:
    """把 cfg.screen_source 解析成一个具体抓取矩形（window / region / monitor）。"""

    def __init__(self, cfg: CaptureConfig) -> None:
        self.cfg = cfg
        self._cache: dict | None = None

    # -------------------------------------------------- 内部
    def _monitor_rect(self, monitors: list[dict], index: int) -> tuple[list[int], int]:
        mon = _monitor_by_index(monitors, index)
        return [mon["left"], mon["top"], mon["width"], mon["height"]], int(mon["index"])

    # -------------------------------------------------- 公开
    def resolve(self) -> dict:
        """解析当前抓取来源；找不到极域窗口时自动退回整屏。"""
        if self._cache is not None:
            return dict(self._cache)
        cfg = self.cfg
        monitors = list_monitors()
        source = str(getattr(cfg, "screen_source", "window") or "window")
        result: dict

        if source == "window":
            wins = []
            try:
                wins = find_classroom_windows(list(getattr(cfg, "window_title_keywords", []) or []))
            except Exception as exc:
                log.warning("查找演示窗口失败：%s", exc)
            if wins:
                win = wins[0]
                rect = [int(win["rect"][0]), int(win["rect"][1]), int(win["width"]), int(win["height"])]
                monitor = _monitor_for_rect(monitors, rect)
                note = "（窗口已最小化，将暂时整屏抓取）" if win.get("minimized") else ""
                result = {
                    "kind": "window",
                    "rect": rect,
                    "hwnd": int(win.get("hwnd") or 0),
                    "title": str(win.get("title") or ""),
                    "monitor": monitor,
                    "reason": f"已定位演示窗口：{win.get('title')}{note}",
                }
            else:
                rect, monitor = self._monitor_rect(monitors, int(getattr(cfg, "monitor_index", 1) or 1))
                result = {
                    "kind": "monitor",
                    "rect": rect,
                    "hwnd": 0,
                    "title": "",
                    "monitor": monitor,
                    "reason": "未找到极域电子教室窗口，已回退整屏",
                }
        elif source == "region":
            raw = list(getattr(cfg, "region", []) or [])
            values = (raw + [0, 0, 0, 0])[:4]
            left, top, width, height = (int(v) for v in values)
            if width <= 0 or height <= 0:
                rect, monitor = self._monitor_rect(
                    monitors, int(getattr(cfg, "monitor_index", 1) or 1)
                )
                result = {
                    "kind": "region",
                    "rect": rect,
                    "hwnd": 0,
                    "title": "",
                    "monitor": monitor,
                    "reason": "未设置有效抓取区域，已改用显示器全屏",
                }
            else:
                rect = [left, top, width, height]
                result = {
                    "kind": "region",
                    "rect": rect,
                    "hwnd": 0,
                    "title": "",
                    "monitor": _monitor_for_rect(monitors, rect),
                    "reason": f"使用自定义区域 {left},{top} {width}x{height}",
                }
        else:  # fullscreen / monitor
            rect, monitor = self._monitor_rect(monitors, int(getattr(cfg, "monitor_index", 1) or 1))
            result = {
                "kind": "monitor",
                "rect": rect,
                "hwnd": 0,
                "title": "",
                "monitor": monitor,
                "reason": f"使用显示器 {monitor} 整屏",
            }
        self._cache = result
        return dict(result)

    def refresh(self) -> dict:
        """窗口移动/缩放后重新解析。"""
        self._cache = None
        return self.resolve()


# ------------------------------------------------------------------ 辅助
def _window_alive(hwnd: int) -> bool:
    if hwnd <= 0:
        return False
    try:
        return bool(ctypes.windll.user32.IsWindow(int(hwnd)))
    except Exception:
        return False


def _window_minimized(hwnd: int) -> bool:
    if hwnd <= 0:
        return False
    try:
        return bool(ctypes.windll.user32.IsIconic(int(hwnd)))
    except Exception:
        return False


def _save_jpeg(image: Image.Image, path: Path, max_width: int = 0, quality: int = 85) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    out = image
    if max_width and out.width > int(max_width):
        ratio = int(max_width) / float(out.width)
        out = out.resize((int(max_width), max(1, int(round(out.height * ratio)))), Image.LANCZOS)
    out.save(str(path), format="JPEG", quality=int(quality))
    return path


# ------------------------------------------------------------------ 录制器
class ScreenRecorder:
    """演示画面录制器（关键帧 + 帧序列视频）。线程安全，stop() 幂等。"""

    def __init__(
        self,
        cfg: CaptureConfig,
        session_dir: Path,
        on_keyframe: Callable[[FrameCapture], None] | None = None,
        on_error: Callable[[str], None] | None = None,
    ) -> None:
        self.cfg = cfg
        self.session_dir = Path(session_dir)
        self.on_keyframe = on_keyframe
        self.on_error = on_error

        self.screen_dir = self.session_dir / "screen"
        self.frames_dir = self.screen_dir / "frames"
        self.keyframes_dir = self.screen_dir / "keyframes"
        self.video_path = self.screen_dir / "video.mp4"

        self._fps = max(1, int(getattr(cfg, "video_fps", 2) or 2))
        self._interval = 1.0 / float(self._fps)
        self._keyframe_interval = max(0.5, float(getattr(cfg, "keyframe_interval_s", 8) or 8))
        self._threshold = float(getattr(cfg, "keyframe_change_threshold", 0.05))
        self._max_keyframe_width = int(getattr(cfg, "max_keyframe_width", 1280) or 1280)
        self._crf = int(getattr(cfg, "video_crf", 28) or 28)

        self._resolver = ScreenSourceResolver(cfg)
        self._source: dict | None = None

        self._lock = threading.RLock()
        self._stop_lock = threading.Lock()
        self._stop_evt = threading.Event()
        self._pause_evt = threading.Event()
        self._ready_evt = threading.Event()

        self._thread: threading.Thread | None = None
        self._t = 0.0
        self._next_keyframe_t = self._keyframe_interval
        self._last_hash: int | None = None
        self._keyframe_count = 0
        self._video_frames = 0
        self._last_frame: Path | None = None
        self._last_keyframe: Path | None = None
        self._preview: bytes | None = None
        self._fallback_warned = False
        self._start_error = ""
        self._stopped = False
        self._result: Path | None = None

    # ---------------------------------------------------------- 属性
    @property
    def is_running(self) -> bool:
        thread = self._thread
        return bool(thread is not None and thread.is_alive() and not self._stop_evt.is_set())

    @property
    def elapsed(self) -> float:
        return float(self._t)

    @property
    def frame_count(self) -> int:
        return int(self._video_frames)

    @property
    def keyframe_count(self) -> int:
        return int(self._keyframe_count)

    @property
    def source(self) -> dict:
        return dict(self._source or {})

    @property
    def last_error(self) -> str:
        return self._start_error

    def last_frame_path(self) -> Path | None:
        return self._last_frame or self._last_keyframe

    def preview_png(self) -> bytes | None:
        """最近一帧的 PNG 字节（缩略图），供 GUI 预览；失败返回 None。"""
        data = self._preview
        if data:
            return data
        image = self._last_frame or self._last_keyframe
        if image is None or not Path(image).exists():
            return None
        try:
            with Image.open(str(image)) as img:
                return self._encode_preview(img)
        except Exception:
            return None

    # ---------------------------------------------------------- 生命周期
    def start(self) -> None:
        """解析来源并起采集线程。"""
        with self._lock:
            if self.is_running:
                return
            if not getattr(self.cfg, "screen_enabled", True):
                log.info("未开启画面采集，跳过")
                return
            try:
                self._source = self._resolver.resolve()
            except Exception as exc:
                log.warning("解析抓取来源失败，改用整屏：%s", exc)
                self._source = {
                    "kind": "monitor",
                    "rect": [0, 0, 1920, 1080],
                    "hwnd": 0,
                    "title": "",
                    "monitor": 1,
                    "reason": f"解析失败回退整屏：{exc}",
                }
            self.frames_dir.mkdir(parents=True, exist_ok=True)
            self.keyframes_dir.mkdir(parents=True, exist_ok=True)
            self._clear_old_frames()

            self._t = 0.0
            self._loop_t0 = None
            self._next_keyframe_t = self._keyframe_interval
            self._last_hash = None
            self._keyframe_count = 0
            self._video_frames = 0
            self._last_frame = None
            self._last_keyframe = None
            self._preview = None
            self._fallback_warned = False
            self._start_error = ""
            self._stopped = False
            self._result = None
            self._stop_evt.clear()
            self._pause_evt.clear()
            self._ready_evt.clear()
            self._thread = threading.Thread(target=self._run, name="ai-student-screen", daemon=True)
            self._thread.start()

        self._ready_evt.wait(_READY_TIMEOUT)
        if self._start_error:
            log.warning("画面采集启动异常：%s", self._start_error)

    def stop(self) -> Path | None:
        """停线程 + 合成视频；返回 <session>/screen/video.mp4，未录视频返回 None。幂等。"""
        with self._stop_lock:
            with self._lock:
                if self._stopped:
                    return self._result
                self._stop_evt.set()
                thread = self._thread
            if thread is not None and thread.is_alive():
                thread.join(timeout=_JOIN_TIMEOUT)
                if thread.is_alive():
                    self._report("画面采集线程未能在超时内退出，已放弃等待")
            with self._lock:
                self._thread = None
                self._stopped = True
                video: Path | None = None
                if getattr(self.cfg, "record_video", True) and self._video_frames > 0:
                    # 用"帧数 / 实际时长"作为编码帧率，保证录像时长 == 真实录制时长，
                    # 这样录像里的位置和转写时间码才能对上（丢帧时自动降帧率，不压缩时间轴）。
                    real = max(0.5, float(self._t or 0.0))
                    eff_fps = self._video_frames / real
                    eff_fps = max(0.2, min(30.0, eff_fps))
                    if abs(eff_fps - self._fps) > 0.05:
                        log.info(
                            "画面丢帧：目标 %.2ffps，实际 %.2ffps（%d 帧 / %.1fs），按实际帧率编码",
                            self._fps, eff_fps, self._video_frames, real,
                        )
                    video = encode_frames_to_mp4(
                        self.frames_dir, eff_fps, self.video_path, crf=self._crf
                    )
                    if video is not None:
                        self._drop_frames_dir()
                self._result = video
                return video

    def pause(self) -> None:
        self._pause_evt.set()

    def resume(self) -> None:
        self._pause_evt.clear()

    @property
    def is_paused(self) -> bool:
        return self._pause_evt.is_set()

    def refresh_source(self) -> dict:
        """窗口移动/缩放后重新解析（线程安全）。"""
        try:
            self._source = self._resolver.refresh()
        except Exception as exc:
            log.warning("重新解析抓取来源失败：%s", exc)
        return self.source

    # ---------------------------------------------------------- 内部
    def _report(self, message: str) -> None:
        log.warning("%s", message)
        if self.on_error is None:
            return
        try:
            self.on_error(message)
        except Exception as exc:
            log.debug("on_error 回调异常：%s", exc)

    def _clear_old_frames(self) -> None:
        try:
            for old in self.frames_dir.glob("f_*.jpg"):
                old.unlink(missing_ok=True)
        except Exception:
            pass

    def _drop_frames_dir(self) -> None:
        try:
            shutil.rmtree(self.frames_dir, ignore_errors=True)
        except Exception as exc:
            log.debug("清理帧目录失败：%s", exc)

    def _encode_preview(self, image: Image.Image) -> bytes | None:
        try:
            out = image
            if out.width > _PREVIEW_WIDTH:
                ratio = _PREVIEW_WIDTH / float(out.width)
                out = out.resize(
                    (_PREVIEW_WIDTH, max(1, int(round(out.height * ratio)))), Image.BILINEAR
                )
            buf = io.BytesIO()
            out.save(buf, format="PNG")
            return buf.getvalue()
        except Exception:
            return None

    def _run(self) -> None:
        sct = None
        try:
            sct = new_mss()  # 必须在采集线程内创建
            # 关键：立刻放行 start()。此前只在 finally 里 set，导致每次开始听课都白等 5 秒，
            # 而且采集线程已经在跑，会让"录制时间轴"和"会话时间轴"错位。
            self._ready_evt.set()
            self._loop_t0 = time.monotonic()
            next_tick = time.monotonic()
            while not self._stop_evt.is_set():
                if self._pause_evt.is_set():
                    self._stop_evt.wait(0.05)
                    next_tick = time.monotonic()
                    continue
                self._tick(sct)
                next_tick += self._interval
                delay = next_tick - time.monotonic()
                if delay > 0:
                    self._stop_evt.wait(delay)
                else:
                    next_tick = time.monotonic()
        except Exception as exc:
            self._start_error = f"画面采集线程异常：{exc}"
            log.exception("画面采集线程异常")
            self._report(self._start_error)
        finally:
            if sct is not None:
                try:
                    sct.close()
                except Exception:
                    pass
            self._ready_evt.set()

    def _tick(self, sct) -> None:
        image = self._grab(sct)
        if image is None:
            return
        # 时间轴一律以**真实经过时间**为准：抓帧/编码偶尔超过间隔时，
        # 按"帧数 × 间隔"累计会让录像时间轴比真实课堂时间越来越短，
        # 导致录像位置与转写时间码对不上。
        t0 = getattr(self, "_loop_t0", None)
        self._t = (time.monotonic() - t0) if t0 else (self._t + self._interval)
        t = self._t

        if getattr(self.cfg, "record_video", True):
            self._video_frames += 1
            path = self.frames_dir / f"f_{self._video_frames:06d}.jpg"
            try:
                _save_jpeg(image, path, quality=80)
                self._last_frame = path
            except Exception as exc:
                self._video_frames -= 1
                self._report(f"写入录像帧失败：{exc}")

        try:
            self._preview = self._encode_preview(image)
        except Exception:
            self._preview = None

        if self._last_hash is not None and t + 1e-6 < self._next_keyframe_t:
            return
        self._next_keyframe_t = t + self._keyframe_interval
        self._handle_keyframe(image, t)

    def _handle_keyframe(self, image: Image.Image, t: float) -> None:
        try:
            current = dhash(image, 16)
        except Exception as exc:
            self._report(f"计算画面哈希失败：{exc}")
            return
        keep, distance = should_keep_keyframe(self._last_hash, current, self._threshold, 16)
        if not keep:
            return
        self._last_hash = current
        index = self._keyframe_count
        path = self.keyframes_dir / f"kf_{index:04d}.jpg"
        try:
            _save_jpeg(image, path, max_width=self._max_keyframe_width, quality=85)
        except Exception as exc:
            self._report(f"写入关键帧失败：{exc}")
            return
        self._keyframe_count = index + 1
        self._last_keyframe = path
        frame = FrameCapture(index=index, t=t, path=path, changed=True, hash_distance=distance)
        if self.on_keyframe is None:
            return
        started = time.monotonic()
        try:
            self.on_keyframe(frame)
        except Exception as exc:
            log.warning("关键帧回调异常：%s", exc)
        if time.monotonic() - started > 0.05:
            log.debug("关键帧回调耗时过长（约定不超过 50ms）")

    def _monitor_rect(self, sct) -> list[int]:
        """当前显示器的矩形（窗口不可用时的整屏回退目标）。"""
        cached = self._source or {}
        try:
            monitors = sct.monitors
            index = int(cached.get("monitor") or 1)
            if 0 < index < len(monitors):
                mon = monitors[index]
            else:
                mon = monitors[1] if len(monitors) > 1 else monitors[0]
            return [int(mon["left"]), int(mon["top"]), int(mon["width"]), int(mon["height"])]
        except Exception:
            rect = list(cached.get("rect") or [0, 0, 1920, 1080])
            return [int(rect[0]), int(rect[1]), int(rect[2]), int(rect[3])]

    def _grab(self, sct) -> Image.Image | None:
        source = self._source or {}
        kind = str(source.get("kind") or "monitor")
        rect = list(source.get("rect") or [0, 0, 0, 0])
        hwnd = int(source.get("hwnd") or 0)

        if kind == "window" and hwnd:
            if not _window_alive(hwnd) or _window_minimized(hwnd):
                if not self._fallback_warned:
                    self._report("演示窗口已最小化或关闭，暂时改用整屏抓取")
                    self._fallback_warned = True
                rect = self._monitor_rect(sct)
            else:
                left, top, right, bottom = window_rect(hwnd)
                width, height = right - left, bottom - top
                if width >= _MIN_GRAB_SIDE and height >= _MIN_GRAB_SIDE:
                    rect = [left, top, width, height]
                    self._fallback_warned = False
                else:
                    if not self._fallback_warned:
                        self._report("演示窗口尺寸异常，暂时改用整屏抓取")
                        self._fallback_warned = True
                    rect = self._monitor_rect(sct)

        if len(rect) < 4 or int(rect[2]) < _MIN_GRAB_SIDE or int(rect[3]) < _MIN_GRAB_SIDE:
            rect = self._monitor_rect(sct)

        bbox = {
            "left": int(rect[0]),
            "top": int(rect[1]),
            "width": max(_MIN_GRAB_SIDE, int(rect[2])),
            "height": max(_MIN_GRAB_SIDE, int(rect[3])),
        }
        try:
            shot = sct.grab(bbox)
        except Exception as exc:
            if not self._fallback_warned:
                self._report(f"抓屏失败（{exc}），稍后自动重试")
                self._fallback_warned = True
            self._stop_evt.wait(0.2)
            return None
        try:
            return Image.frombytes("RGB", shot.size, shot.bgra, "raw", "BGRX")
        except Exception as exc:
            self._report(f"转换画面失败：{exc}")
            return None


__all__ = [
    "FrameCapture",
    "ScreenRecorder",
    "ScreenSourceResolver",
    "dhash",
    "dhash_distance",
    "hamming",
    "should_keep_keyframe",
]
