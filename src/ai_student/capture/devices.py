"""设备与窗口枚举：音频设备（PyAudioWPatch/WASAPI）、显示器（mss）、顶层窗口（pywin32）。

设计原则：GUI 会直接调用这些函数做"设备探测"，所以**任何异常都必须吞掉**，
返回已经收集到的部分结果，绝不让界面崩溃。
"""
from __future__ import annotations

import ctypes
import logging
import os
from ctypes import wintypes
from dataclasses import asdict, dataclass
from pathlib import Path

log = logging.getLogger(__name__)

_dpi_ready = False


def _ensure_dpi_aware() -> None:
    """进程级 DPI 感知（每显示器 v2），保证抓屏坐标是物理像素。"""
    global _dpi_ready
    if _dpi_ready:
        return
    _dpi_ready = True
    if os.name != "nt":
        return
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(2)
    except Exception:
        try:
            ctypes.windll.user32.SetProcessDPIAware()
        except Exception:
            pass


def ensure_dpi_aware() -> None:
    """公开入口：其它采集模块在导入时调用（幂等）。"""
    _ensure_dpi_aware()


_ensure_dpi_aware()


def new_mss():
    """创建 mss 抓屏实例（**必须在采集线程内调用**，跨线程使用会崩）。"""
    import mss

    try:
        return mss.MSS()
    except AttributeError:  # 兼容 mss < 10.2
        return mss.mss()


# ------------------------------------------------------------------ 音频设备
@dataclass
class AudioDeviceInfo:
    """一个音频端点（输入设备或 WASAPI 回环设备）。"""

    index: int
    name: str
    channels: int
    sample_rate: int
    host_api: str
    is_loopback: bool
    is_default_input: bool
    is_default_output: bool

    @property
    def is_input(self) -> bool:
        return bool(self.channels) and not self.is_loopback

    def display(self) -> str:
        tags = []
        if self.is_loopback:
            tags.append("系统回环")
        if self.is_default_input:
            tags.append("默认麦克风")
        if self.is_default_output and self.is_loopback:
            tags.append("默认扬声器")
        suffix = f"（{'/'.join(tags)}）" if tags else ""
        return f"[{self.index}] {self.name}{suffix}"

    def to_dict(self) -> dict:
        data = asdict(self)
        data["display"] = self.display()
        return data


def _is_loopback_name(name: str) -> bool:
    low = (name or "").lower()
    return "loopback" in low or "回环" in low


def list_audio_devices(include_loopback: bool = True) -> list[AudioDeviceInfo]:
    """枚举音频输入设备（含 WASAPI 回环）。

    只列出"可输入"的设备（maxInputChannels > 0，回环设备也算）；
    include_loopback=False 时过滤掉回环设备。任何异常都吞掉并返回已收集结果。
    """
    found: list[AudioDeviceInfo] = []
    pa = None
    try:
        import pyaudiowpatch as pyaudio

        pa = pyaudio.PyAudio()
        default_in = -1
        default_out = -1
        default_loop = -1
        try:
            default_in = int(pa.get_default_input_device_info()["index"])
        except Exception:
            pass
        try:
            default_out = int(pa.get_default_output_device_info()["index"])
        except Exception:
            pass
        try:
            default_loop = int(pa.get_default_wasapi_loopback()["index"])
        except Exception:
            pass

        host_names: dict[int, str] = {}
        for i in range(int(pa.get_device_count())):
            try:
                info = pa.get_device_info_by_index(i)
            except Exception:
                continue
            try:
                in_ch = int(info.get("maxInputChannels") or 0)
                out_ch = int(info.get("maxOutputChannels") or 0)
            except Exception:
                in_ch, out_ch = 0, 0
            if in_ch <= 0:
                continue  # 只保留可录音的端点
            name = str(info.get("name") or "")
            loop = bool(info.get("isLoopbackDevice")) or _is_loopback_name(name)
            if loop and not include_loopback:
                continue
            try:
                host_index = int(info.get("hostApi"))
            except Exception:
                host_index = -1
            if host_index not in host_names:
                try:
                    host_names[host_index] = str(pa.get_host_api_info_by_index(host_index)["name"])
                except Exception:
                    host_names[host_index] = ""
            try:
                rate = int(round(float(info.get("defaultSampleRate") or 0)))
            except Exception:
                rate = 0
            found.append(
                AudioDeviceInfo(
                    index=i,
                    name=name,
                    channels=in_ch or out_ch,
                    sample_rate=rate,
                    host_api=host_names.get(host_index, ""),
                    is_loopback=loop,
                    is_default_input=(i == default_in),
                    is_default_output=bool(loop and (i == default_loop or i == default_out)),
                )
            )
    except Exception as exc:
        log.warning("枚举音频设备失败：%s", exc)
    finally:
        if pa is not None:
            try:
                pa.terminate()
            except Exception:
                pass
    return found


# ------------------------------------------------------------------ 显示器
def list_monitors() -> list[dict]:
    """[{"index":1,"left":0,"top":0,"width":1920,"height":1080,"primary":True}, ...]"""
    monitors: list[dict] = []
    try:
        with new_mss() as sct:
            raw = list(sct.monitors[1:])  # [0] 是"所有显示器"的虚拟屏
        for i, mon in enumerate(raw, start=1):
            left = int(mon.get("left", 0))
            top = int(mon.get("top", 0))
            width = int(mon.get("width", 0))
            height = int(mon.get("height", 0))
            if width <= 0 or height <= 0:
                continue
            primary = bool(mon.get("is_primary")) or (left == 0 and top == 0)
            monitors.append(
                {
                    "index": i,
                    "left": left,
                    "top": top,
                    "width": width,
                    "height": height,
                    "primary": primary,
                }
            )
    except Exception as exc:
        log.warning("枚举显示器失败：%s", exc)
    if monitors and not any(m["primary"] for m in monitors):
        monitors[0]["primary"] = True
    return monitors


# ------------------------------------------------------------------ 窗口
def window_rect(hwnd: int) -> tuple[int, int, int, int]:
    """窗口在屏幕上的物理像素矩形 (left, top, right, bottom)。

    优先用 DWM 扩展边框（去掉 Win10/11 不可见的阴影边框），失败回退 GetWindowRect。
    """
    rect: tuple[int, int, int, int] | None = None
    if os.name == "nt":
        try:
            r = wintypes.RECT()
            dwm = ctypes.windll.dwmapi
            res = dwm.DwmGetWindowAttribute(
                wintypes.HWND(int(hwnd)), ctypes.c_uint(9), ctypes.byref(r), ctypes.sizeof(r)
            )
            if res == 0:
                rect = (int(r.left), int(r.top), int(r.right), int(r.bottom))
        except Exception:
            rect = None
    if rect is None or rect[2] <= rect[0] or rect[3] <= rect[1]:
        try:
            import win32gui

            raw = win32gui.GetWindowRect(int(hwnd))
            rect = (int(raw[0]), int(raw[1]), int(raw[2]), int(raw[3]))
        except Exception:
            rect = rect or (0, 0, 0, 0)
    return rect


def _process_name(pid: int) -> str:
    """进程可执行文件名；失败返回空串（权限不足是常态，不要报错）。"""
    if pid <= 0 or os.name != "nt":
        return ""
    handle = None
    try:
        kernel32 = ctypes.windll.kernel32
        handle = kernel32.OpenProcess(0x1000, False, int(pid))  # PROCESS_QUERY_LIMITED_INFORMATION
        if not handle:
            return ""
        size = ctypes.c_uint(1024)
        buf = ctypes.create_unicode_buffer(1024)
        if kernel32.QueryFullProcessImageNameW(handle, 0, buf, ctypes.byref(size)):
            return Path(buf.value).name
    except Exception:
        return ""
    finally:
        if handle:
            try:
                ctypes.windll.kernel32.CloseHandle(handle)
            except Exception:
                pass
    return ""


def list_windows(visible_only: bool = True) -> list[dict]:
    """枚举顶层窗口；跳过空标题窗口。任何异常都吞掉。"""
    windows: list[dict] = []
    try:
        import win32gui
        import win32process
    except Exception as exc:
        log.warning("pywin32 不可用，无法枚举窗口：%s", exc)
        return windows

    def _collect(hwnd: int, _param) -> bool:
        try:
            if visible_only and not win32gui.IsWindowVisible(hwnd):
                return True
            title = (win32gui.GetWindowText(hwnd) or "").strip()
            if not title:
                return True
            left, top, right, bottom = window_rect(hwnd)
            width = max(0, right - left)
            height = max(0, bottom - top)
            if width <= 0 or height <= 0:
                return True
            try:
                _thread, pid = win32process.GetWindowThreadProcessId(hwnd)
            except Exception:
                pid = 0
            windows.append(
                {
                    "hwnd": int(hwnd),
                    "title": title,
                    "rect": [left, top, right, bottom],
                    "width": width,
                    "height": height,
                    "pid": int(pid or 0),
                    "process": _process_name(int(pid or 0)),
                    "minimized": bool(win32gui.IsIconic(hwnd)),
                }
            )
        except Exception:
            pass
        return True

    try:
        win32gui.EnumWindows(_collect, None)
    except Exception as exc:
        log.warning("EnumWindows 失败：%s", exc)
    return windows


def find_classroom_windows(keywords: list[str] | None = None) -> list[dict]:
    """按标题关键字找"极域电子教室"类窗口，按最可能正在演示排序（最大化/面积大优先）。"""
    if keywords is None:
        from ..config.models import CaptureConfig

        keywords = CaptureConfig().window_title_keywords
    wanted = [str(k).strip().lower() for k in (keywords or []) if str(k).strip()]
    if not wanted:
        return []
    hits: list[dict] = []
    for win in list_windows(visible_only=False):
        haystack = f"{win.get('title', '')} {win.get('process', '')}".lower()
        if any(k in haystack for k in wanted):
            hits.append(win)

    def _score(win: dict) -> float:
        area = float(win.get("width", 0)) * float(win.get("height", 0))
        if win.get("minimized"):
            area *= 0.02
        if win.get("width", 0) >= 640 and win.get("height", 0) >= 480:
            area *= 1.5
        return area

    hits.sort(key=_score, reverse=True)
    for rank, win in enumerate(hits):
        win["score"] = _score(win)
        win["rank"] = rank
    return hits


# ------------------------------------------------------------------ 汇总
def describe_environment() -> dict:
    """一次性给 GUI / 自检工具用的环境快照。"""
    from .ffmpeg import ffmpeg_path

    audio = list_audio_devices(include_loopback=True)
    return {
        "audio": [d.to_dict() for d in audio],
        "monitors": list_monitors(),
        "windows": list_windows(),
        "classroom_windows": find_classroom_windows(),
        "ffmpeg": ffmpeg_path(),
    }
