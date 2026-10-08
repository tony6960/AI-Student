"""音频采集：麦克风（可选）+ 系统回环（可选）混音录制。

要点：
- 用 PyAudioWPatch（WASAPI）打开设备；PyAudio 实例与流**都在采集线程内创建/打开**，
  避免跨线程使用 PortAudio 句柄。
- 两路设备原生采样率/声道数不同也没关系：统一线性插值重采样到
  cfg.sample_rate / cfg.channels（int16）后再混音。
- 整段写入 <session>/audio/audio.wav（边录边写，进程崩溃也能留下已录部分）；
  每 cfg.chunk_seconds 秒写一片 <session>/audio/chunks/chunk_%04d.wav 并回调 on_chunk。
- 设备中途失效（拔麦/独占）不崩：记错误、继续重试，连续失败到上限才优雅停止。
- 读取绝不阻塞在设备上：每路设备由独立读取线程用 get_read_available() 轮询取数，
  WASAPI 回环在没有声音播放时不会把线程永久卡住（实测阻塞式 read() 会），
  stop() 时线程一定能退出、流一定能关闭。
- start() 在**主线程**完成设备解析，找不到设备直接抛 RuntimeError；
  stop() 幂等；所有线程可干净退出。
"""
from __future__ import annotations

import logging
import math
import queue
import threading
import time
import wave
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np

from ..config.models import CaptureConfig
from .devices import AudioDeviceInfo, list_audio_devices

log = logging.getLogger(__name__)

_BLOCK_SECONDS = 0.10      # 每个读取块的长度（100ms，兼顾延迟与开销）
_MIN_RMS_DB = -120.0       # 电平下限
_READY_TIMEOUT = 6.0       # start() 等待设备打开的上限
_JOIN_TIMEOUT = 8.0        # stop() 等待线程退出的上限
_MAX_CONSECUTIVE_ERRORS = 8
_OPEN_TIMEOUT = 4.0        # 单个设备打开超时：某些 WASAPI 回环驱动会永久卡住 pa.open()
_TERMINATE_TIMEOUT = 2.0   # pa.terminate() 也可能卡住，同样要限时


# ------------------------------------------------------------------ 纯函数工具
def _as2d(data) -> np.ndarray:
    arr = np.asarray(data)
    if arr.ndim == 0:
        return arr.reshape(1, 1)
    if arr.ndim == 1:
        return arr.reshape(-1, 1)
    return arr


def rms_db(samples) -> float:
    """整体 RMS 电平（dBFS，int16 满量程 32768）；静音返回 -120。"""
    arr = _as2d(samples).astype(np.float32)
    if arr.size == 0:
        return _MIN_RMS_DB
    rms = float(np.sqrt(np.mean(np.square(arr))))
    if rms <= 0.0:
        return _MIN_RMS_DB
    return max(_MIN_RMS_DB, 20.0 * math.log10(rms / 32768.0))


def peak_db(samples) -> float:
    """峰值电平（dBFS）；静音返回 -120。"""
    arr = _as2d(samples).astype(np.float32)
    if arr.size == 0:
        return _MIN_RMS_DB
    peak = float(np.max(np.abs(arr)))
    if peak <= 0.0:
        return _MIN_RMS_DB
    return max(_MIN_RMS_DB, 20.0 * math.log10(peak / 32768.0))


def to_channels(data: np.ndarray, channels: int) -> np.ndarray:
    """把 (n, c) 数据变成 (n, channels)：多→少取均值/截断，少→多用复制补齐。"""
    arr = _as2d(data)
    want = max(1, int(channels))
    if arr.shape[1] == want:
        return arr
    if arr.shape[1] > want:
        if want == 1:
            return arr.mean(axis=1, keepdims=True)
        return arr[:, :want]
    reps = int(math.ceil(want / max(1, arr.shape[1])))
    return np.tile(arr, (1, reps))[:, :want]


def resample(data: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
    """线性插值重采样，返回 float32 的 (n', c) 数组。"""
    arr = _as2d(data).astype(np.float32)
    src = int(src_rate or 0)
    dst = int(dst_rate or 0)
    n = arr.shape[0]
    if n == 0 or src <= 0 or dst <= 0 or src == dst:
        return arr
    out_n = max(1, int(round(n * dst / float(src))))
    if n == 1:
        return np.repeat(arr, out_n, axis=0)
    pos = np.clip(np.arange(out_n, dtype=np.float64) * (src / float(dst)), 0.0, n - 1.0)
    x0 = np.floor(pos).astype(np.int64)
    x1 = np.minimum(x0 + 1, n - 1)
    frac = (pos - x0).astype(np.float32).reshape(-1, 1)
    return arr[x0] * (1.0 - frac) + arr[x1] * frac


def to_int16(data: np.ndarray) -> np.ndarray:
    """float 数据转 int16（带饱和裁剪）。"""
    arr = _as2d(data).astype(np.float32)
    return np.clip(np.rint(arr), -32768.0, 32767.0).astype(np.int16)


def mix_int16(a: np.ndarray, b: np.ndarray, gain: float = 1.0) -> np.ndarray:
    """两路 int16 相加混音（饱和裁剪），长度取较短的一路。"""
    x = to_int16(a)
    y = to_int16(b)
    channels = max(x.shape[1], y.shape[1])
    x = to_channels(x, channels)
    y = to_channels(y, channels)
    n = min(x.shape[0], y.shape[0])
    if n <= 0:
        return np.zeros((0, channels), dtype=np.int16)
    mixed = x[:n].astype(np.float32) * float(gain) + y[:n].astype(np.float32) * float(gain)
    return np.clip(mixed, -32768.0, 32767.0).astype(np.int16)


def write_wav_int16(path: Path, samples: np.ndarray, rate: int) -> Path:
    """把 (n, c) / (n,) 的 int16 数据写成标准 WAV。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = to_int16(samples)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(int(data.shape[1]))
        handle.setsampwidth(2)
        handle.setframerate(int(rate))
        handle.writeframes(data.tobytes())
    return path


def read_wav_int16(path: Path) -> tuple[np.ndarray, int]:
    """读 WAV -> ((n, c) int16, sample_rate)；失败返回 (空数组, 0)。"""
    try:
        with wave.open(str(Path(path)), "rb") as handle:
            channels = max(1, int(handle.getnchannels()))
            rate = int(handle.getframerate())
            width = int(handle.getsampwidth())
            raw = handle.readframes(handle.getnframes())
    except Exception as exc:
        log.debug("读取 WAV 失败：%s", exc)
        return np.zeros((0, 1), dtype=np.int16), 0
    if width != 2:
        return np.zeros((0, channels), dtype=np.int16), rate
    data = np.frombuffer(raw, dtype=np.int16)
    usable = (data.size // channels) * channels
    return data[:usable].reshape(-1, channels), rate


# ------------------------------------------------------------------ 设备解析
@dataclass
class AudioPlan:
    """本次录音实际使用的设备组合。"""

    mic: AudioDeviceInfo | None = None
    loopback: AudioDeviceInfo | None = None
    warnings: list[str] = field(default_factory=list)

    @property
    def sources(self) -> list[AudioDeviceInfo]:
        return [d for d in (self.mic, self.loopback) if d is not None]

    @property
    def kind(self) -> str:
        count = len(self.sources)
        if count >= 2:
            return "mix"
        if count == 1:
            return "loopback" if self.loopback is not None else "mic"
        return "mic"


def select_device(
    devices: list[AudioDeviceInfo],
    name_substring: str = "",
    *,
    loopback: bool | None = None,
    prefer_default: bool = False,
) -> AudioDeviceInfo | None:
    """在候选设备里挑一个：优先精确名、其次子串；都没给就挑默认设备。"""
    pool = [d for d in devices if loopback is None or bool(d.is_loopback) == bool(loopback)]
    if not pool:
        return None
    want = (name_substring or "").strip().lower()
    if want:
        for dev in pool:
            if dev.name.strip().lower() == want:
                return dev
        for dev in pool:
            if want in dev.name.lower():
                return dev
    if prefer_default:
        for dev in pool:
            if loopback and dev.is_default_output:
                return dev
            if not loopback and dev.is_default_input:
                return dev
    return pool[0]


def resolve_audio_devices(cfg: CaptureConfig) -> AudioPlan:
    """按配置解析麦克风 / 回环设备；异常吞掉，返回尽量可用的组合。"""
    plan = AudioPlan()
    try:
        devices = list_audio_devices(include_loopback=True)
    except Exception as exc:
        plan.warnings.append(f"枚举音频设备失败：{exc}")
        return plan
    if cfg.mic_enabled:
        mic = select_device(devices, cfg.mic_device, loopback=False, prefer_default=True)
        if mic is None:
            plan.warnings.append("未找到可用的麦克风设备")
        elif cfg.mic_device and cfg.mic_device.strip().lower() not in mic.name.lower():
            plan.warnings.append(f"未找到名称包含「{cfg.mic_device}」的麦克风，已改用：{mic.name}")
        plan.mic = mic
    if cfg.loopback_enabled:
        loop = select_device(devices, cfg.loopback_device, loopback=True, prefer_default=True)
        if loop is None:
            plan.warnings.append("未找到可用的系统回环设备")
        elif cfg.loopback_device and cfg.loopback_device.strip().lower() not in loop.name.lower():
            plan.warnings.append(
                f"未找到名称包含「{cfg.loopback_device}」的回环设备，已改用：{loop.name}"
            )
        plan.loopback = loop
    if (cfg.mic_enabled or cfg.loopback_enabled) and not plan.sources:
        plan.warnings.append("没有任何可用的音频输入设备")
    return plan


def _load_pyaudio():
    """延迟导入 PyAudioWPatch（模块级导入会让无音频后端的机器直接崩）。"""
    import pyaudiowpatch as pyaudio

    return pyaudio


# ------------------------------------------------------------------ 数据类
@dataclass
class AudioChunk:
    """一个已落盘的音频切片。"""

    index: int
    t_start: float
    t_end: float
    path: Path
    rms_db: float
    peak_db: float
    silent: bool
    source: str  # "mix" | "mic" | "loopback"

    @property
    def duration(self) -> float:
        return max(0.0, float(self.t_end) - float(self.t_start))

    def to_dict(self) -> dict:
        return {
            "index": self.index,
            "t_start": round(self.t_start, 3),
            "t_end": round(self.t_end, 3),
            "path": str(self.path),
            "rms_db": round(self.rms_db, 1),
            "peak_db": round(self.peak_db, 1),
            "silent": self.silent,
            "source": self.source,
        }


# ------------------------------------------------------------------ 卡死句柄回收站
# 某些驱动（实测 Realtek WASAPI 回环）会让 read()/open() 永久阻塞。为了不触发 PortAudio
# 访问冲突，我们**不强行关闭**这些句柄，而是登记到这里；等对应线程自己退出后再补关。
_ABANDONED: list = []
_ABANDONED_LOCK = threading.Lock()


def _register_abandoned(obj) -> None:
    with _ABANDONED_LOCK:
        _ABANDONED.append(obj)


def _reap_abandoned() -> int:
    """尝试回收之前卡住的句柄；返回本次成功释放的数量。"""
    with _ABANDONED_LOCK:
        pending = list(_ABANDONED)
        _ABANDONED.clear()
    freed = 0
    for item in pending:
        try:
            if item.try_release():
                freed += 1
            else:
                _register_abandoned(item)
        except Exception:
            pass
    return freed


class _PendingOpen:
    """被判定超时、但线程可能仍在驱动里打开的句柄；恢复后由回收站补关。"""

    def __init__(self, thread: threading.Thread, box: dict) -> None:
        self.thread = thread
        self.box = box

    def try_release(self) -> bool:
        if self.thread.is_alive():
            return False
        stream = self.box.get("stream")
        if stream is not None:
            _quiet(stream.stop_stream)
            _quiet(stream.close)
        return True


# ------------------------------------------------------------------ 设备读取泵
def _open_stream_with_timeout(pa, pyaudio, plan: dict, timeout: float = _OPEN_TIMEOUT):
    """带超时地打开一个输入流。

    某些 WASAPI 回环驱动（实测 Realtek 扬声器回环）会让 pa.open() 永久阻塞，
    因此必须在独立线程里打开；超时就放弃这个设备，用剩下的设备继续录。
    返回 (stream, error_message, abandoned)；abandoned=True 表示打开线程仍卡在驱动里，
    此时**绝对不能再调用 pa.terminate()**（会造成 PortAudio 访问冲突崩溃）。
    """
    box: dict = {}

    def worker() -> None:
        try:
            box["stream"] = pa.open(
                format=pyaudio.paInt16,
                channels=int(plan["channels"]),
                rate=int(plan["rate"]),
                input=True,
                input_device_index=int(plan["index"]),
                frames_per_buffer=int(plan["block_frames"]),
            )
        except Exception as exc:  # noqa: BLE001
            box["error"] = str(exc)

    t = threading.Thread(target=worker, name="ai-student-audio-open", daemon=True)
    t.start()
    t.join(timeout)
    if t.is_alive():
        _register_abandoned(_PendingOpen(t, box))
        return None, f"打开超时（{timeout:g}s，驱动无响应）", True
    if "error" in box:
        return None, str(box["error"]), False
    return box.get("stream"), "", False


def _safe_terminate(pa, timeout: float = _TERMINATE_TIMEOUT) -> None:
    """限时调用 pa.terminate()，避免被卡住的 PortAudio 拖死采集线程。"""
    if pa is None:
        return
    t = threading.Thread(target=lambda: _quiet(pa.terminate), name="ai-student-pa-term", daemon=True)
    t.start()
    t.join(timeout)


def _quiet(fn) -> None:
    try:
        fn()
    except Exception:
        pass


class _StreamPump:
    """把单个设备流的 read() 放到独立线程，主循环永远不会阻塞在设备上。

    设备中途失效（拔麦 / 驱动无响应）时线程自行退出并记下 error，主循环据此优雅降级。
    """

    def __init__(self, plan: dict, stream: object) -> None:
        self.plan = plan
        self.stream = stream
        self.rate = int(plan["rate"])
        self.frames = int(plan["block_frames"])
        self.channels = max(1, int(plan["channels"]))
        self.queue: "queue.Queue[np.ndarray]" = queue.Queue(maxsize=32)
        self.error: Exception | None = None
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="ai-student-audio-read", daemon=True)

    def start(self) -> None:
        self._thread.start()

    def alive(self) -> bool:
        return self._thread.is_alive()

    def _run(self) -> None:
        while not self._stop.is_set():
            # 实测（Realtek WASAPI 回环）：当前没有任何声音在播放时，PortAudio 的阻塞式
            # read() 会**永久**不返回，线程再也退不出去、句柄也关不掉。所以先问"有多少帧
            # 可读"，只读真正就绪的数据；一帧都没有就短暂让出 CPU 后再问。
            frames = self.frames
            try:
                available = int(self.stream.get_read_available())  # type: ignore[attr-defined]
            except Exception:
                available = -1  # 后端不支持查询：退回阻塞式读取（由回收站兜底）
            if available == 0:
                self._stop.wait(0.005)
                continue
            if available > 0:
                frames = max(1, min(available, self.frames))
            try:
                raw = self.stream.read(frames, exception_on_overflow=False)  # type: ignore[attr-defined]
            except Exception as exc:  # noqa: BLE001
                if not self._stop.is_set():
                    self.error = exc
                return
            if not raw:
                continue
            arr = np.frombuffer(raw, dtype=np.int16)
            if self.channels > 1:
                usable = (arr.size // self.channels) * self.channels
                if usable <= 0:
                    continue
                arr = arr[:usable].reshape(-1, self.channels)
            else:
                arr = arr.reshape(-1, 1)
            try:
                self.queue.put_nowait(arr)
            except queue.Full:  # 丢弃最旧的一块，保持低延迟
                try:
                    self.queue.get_nowait()
                except queue.Empty:
                    pass
                _quiet(lambda a=arr: self.queue.put_nowait(a))

    def drain(self) -> list[np.ndarray]:
        """取出当前所有已就绪的数据块。

        录音必须连续：早先"只保留最新一块"的写法会在主循环稍慢时丢掉中间音频，
        造成录音出现空洞、时长与实际内容对不上。
        """
        items: list[np.ndarray] = []
        while True:
            try:
                items.append(self.queue.get_nowait())
            except queue.Empty:
                return items

    def try_release(self) -> bool:
        """回收站调用：只在读取线程确实退出后才关流。"""
        if self._thread.is_alive():
            return False
        _quiet(self.stream.stop_stream)  # type: ignore[attr-defined]
        _quiet(self.stream.close)        # type: ignore[attr-defined]
        return True

    def close(self, join_timeout: float = 1.0) -> bool:
        """停止读取线程并关闭流。

        **必须等读取线程真正退出后才能关流**：PortAudio 的 read() 与 stop_stream()/close()
        并发调用会直接触发访问冲突（0xC0000005）把整个进程干掉。
        返回 True 表示流已安全关闭；False 表示线程仍卡在 read()，此时宁可泄漏也不碰流。
        """
        self._stop.set()
        if self._thread.is_alive():
            self._thread.join(join_timeout)
        if self._thread.is_alive():
            return False
        _quiet(self.stream.stop_stream)  # type: ignore[attr-defined]
        _quiet(self.stream.close)        # type: ignore[attr-defined]
        return True


# ------------------------------------------------------------------ 录制器
class AudioRecorder:
    """麦克风 + 系统回环混音录制器（线程安全，stop() 幂等）。"""

    def __init__(
        self,
        cfg: CaptureConfig,
        session_dir: Path,
        on_chunk: Callable[[AudioChunk], None] | None = None,
        on_error: Callable[[str], None] | None = None,
    ) -> None:
        self.cfg = cfg
        self.session_dir = Path(session_dir)
        self.on_chunk = on_chunk
        self.on_error = on_error

        self.audio_dir = self.session_dir / "audio"
        self.chunks_dir = self.audio_dir / "chunks"
        self.full_path = self.audio_dir / "audio.wav"

        self._rate = max(8000, int(getattr(cfg, "sample_rate", 16000) or 16000))
        self._channels = max(1, min(2, int(getattr(cfg, "channels", 1) or 1)))
        chunk_seconds = float(getattr(cfg, "chunk_seconds", 20) or 20)
        # 单个切片最长 10 分钟，防止配置异常时把内存吃光
        self._chunk_samples = max(1, int(round(min(chunk_seconds, 600.0) * self._rate)))
        self._vad_db = float(getattr(cfg, "vad_silence_db", -45.0))

        self._state_lock = threading.RLock()
        self._stop_lock = threading.Lock()
        self._stop_evt = threading.Event()
        self._pause_evt = threading.Event()
        self._ready_evt = threading.Event()

        self._thread: threading.Thread | None = None
        self._plans: list[dict] = []
        self._start_error = ""
        self._source_kind = "mic"
        self._chunk_count = 0
        self._written_samples = 0
        self._last_rms = _MIN_RMS_DB
        self._last_peak = _MIN_RMS_DB
        self._last_loud = 0.0
        self._failure_streak = 0
        self._stopped = False
        self._abandoned_open = False
        self._result: Path | None = None

    # ---------------------------------------------------------- 属性
    @property
    def is_running(self) -> bool:
        thread = self._thread
        return bool(thread is not None and thread.is_alive() and not self._stop_evt.is_set())

    @property
    def elapsed(self) -> float:
        """已录制（非暂停）时长，秒。"""
        return self._written_samples / float(self._rate)

    @property
    def last_rms_db(self) -> float:
        return float(self._last_rms)

    @property
    def last_peak_db(self) -> float:
        return float(self._last_peak)

    @property
    def full_audio_path(self) -> Path:
        return self.full_path

    @property
    def source_kind(self) -> str:
        return self._source_kind

    @property
    def last_error(self) -> str:
        return self._start_error

    def silence_seconds(self) -> float:
        """距最后一次"有声"的秒数（用于判定下课）。"""
        if self._last_loud <= 0:
            return self.elapsed
        return max(0.0, time.monotonic() - self._last_loud)

    # ---------------------------------------------------------- 生命周期
    def start(self) -> None:
        """起采集线程；无可用设备时抛 RuntimeError。"""
        with self._state_lock:
            if self.is_running:
                return
            if not self.cfg.mic_enabled and not self.cfg.loopback_enabled:
                raise RuntimeError("麦克风与系统回环都未启用，无法录音")
            plan = resolve_audio_devices(self.cfg)
            for msg in plan.warnings:
                log.warning("%s", msg)
            sources = plan.sources
            if not sources:
                raise RuntimeError("找不到可用的音频输入设备（麦克风 / 系统回环）")
            missing = (self.cfg.mic_enabled and plan.mic is None) or (
                self.cfg.loopback_enabled and plan.loopback is None
            )
            if missing:  # 用户要了但没找到：明确告诉引擎，别让它以为两路都在录
                self._report("；".join(plan.warnings) or "部分音频设备不可用，已用剩余设备继续录制")

            self._plans = [
                {
                    "index": int(dev.index),
                    "name": dev.name,
                    "channels": max(1, int(dev.channels or 1)),
                    "rate": int(dev.sample_rate or 44100),
                    "loopback": bool(dev.is_loopback),
                    "block_frames": max(128, int((dev.sample_rate or 44100) * _BLOCK_SECONDS)),
                }
                for dev in sources
            ]
            self._source_kind = plan.kind
            self._start_error = ""
            self._chunk_count = 0
            self._written_samples = 0
            self._last_rms = _MIN_RMS_DB
            self._last_peak = _MIN_RMS_DB
            self._failure_streak = 0
            self._stopped = False
            self._abandoned_open = False
            self._result = None
            self._stop_evt.clear()
            self._pause_evt.clear()
            self._ready_evt.clear()
            _reap_abandoned()
            self.audio_dir.mkdir(parents=True, exist_ok=True)
            self.chunks_dir.mkdir(parents=True, exist_ok=True)
            self._thread = threading.Thread(target=self._run, name="ai-student-audio", daemon=True)
            self._thread.start()

        # 等线程把设备真正打开；失败就抛错，避免"看起来在录、其实没录"
        self._ready_evt.wait(_READY_TIMEOUT)
        if self._start_error:
            with self._state_lock:
                self._stop_evt.set()
                self._thread = None
            raise RuntimeError(self._start_error)

    def stop(self) -> Path | None:
        """停线程 + 收尾写盘；返回整段 wav（<session>/audio/audio.wav）。幂等。"""
        with self._stop_lock:
            with self._state_lock:
                if self._stopped:
                    return self._result
                self._stop_evt.set()
                thread = self._thread
            if thread is not None and thread.is_alive():
                thread.join(timeout=_JOIN_TIMEOUT)
                if thread.is_alive():
                    self._report("音频采集线程未能在超时内退出，已放弃等待")
            with self._state_lock:
                self._thread = None
                self._stopped = True
                path: Path | None = self.full_path if self.full_path.exists() else None
                if path is not None and path.stat().st_size <= 44:
                    path = None  # 只有 WAV 头，没有实际数据
                self._result = path
                return path

    def pause(self) -> None:
        self._pause_evt.set()

    def resume(self) -> None:
        self._pause_evt.clear()

    @property
    def is_paused(self) -> bool:
        return self._pause_evt.is_set()

    # ---------------------------------------------------------- 线程内部
    def _report(self, message: str) -> None:
        log.warning("%s", message)
        if self.on_error is None:
            return
        try:
            self.on_error(message)
        except Exception as exc:
            log.debug("on_error 回调异常：%s", exc)

    def _run(self) -> None:
        pyaudio = None
        pa = None
        pumps: list[_StreamPump] = []
        full = None
        try:
            pyaudio = _load_pyaudio()
            pa = pyaudio.PyAudio()
            for plan in self._plans:
                stream, err, abandoned = _open_stream_with_timeout(pa, pyaudio, plan)
                if abandoned:
                    self._abandoned_open = True
                if stream is None:
                    self._report(f"打开音频设备失败：{plan['name']}（{err}）")
                    continue
                pump = _StreamPump(plan, stream)
                pump.start()
                pumps.append(pump)
            if not pumps:
                self._start_error = "无法打开任何音频输入设备"
                return
            self._source_kind = (
                "mix" if len(pumps) > 1 else ("loopback" if pumps[0].plan["loopback"] else "mic")
            )
            if len(pumps) < len(self._plans):
                self._report("部分音频设备不可用，已用剩余设备继续录制")
            full = self._open_full()
            self._last_loud = time.monotonic()
            self._ready_evt.set()
            self._loop(pumps, full)
        except Exception as exc:
            self._start_error = self._start_error or f"音频采集线程异常：{exc}"
            log.exception("音频采集线程异常")
            self._report(self._start_error)
        finally:
            clean = True
            for pump in pumps:
                if not pump.close():
                    clean = False
                    _register_abandoned(pump)  # 交给回收站，等线程退出后再关
                    log.warning("音频读取线程未能在超时内退出，已登记待回收（规避 PortAudio 崩溃）")
            _reap_abandoned()
            # 只有在"没有卡住的打开线程 + 所有流都干净关闭"时才释放 PortAudio
            if clean and not self._abandoned_open:
                _safe_terminate(pa)
            else:
                log.warning("跳过 PortAudio 释放（存在卡住的驱动句柄），本次会话结束后不再复用该实例")
            if full is not None:
                _quiet(full.close)
            self._ready_evt.set()

    def _open_full(self):
        self.audio_dir.mkdir(parents=True, exist_ok=True)
        handle = wave.open(str(self.full_path), "wb")
        handle.setnchannels(self._channels)
        handle.setsampwidth(2)
        handle.setframerate(self._rate)
        return handle

    def _loop(self, pumps: list[_StreamPump], full) -> None:
        """主循环：从各设备的读取泵取最新数据 -> 重采样 -> 混音 -> 落盘。

        因为读取在独立线程里，这里永远不会被某个无响应的设备卡住。
        """
        parts: list[np.ndarray] = []
        pending = 0
        live: set[str] = set()
        while not self._stop_evt.is_set():
            blocks: list[np.ndarray] = []
            read_error: Exception | None = None
            for pump in pumps:
                if pump.error is not None:
                    read_error = pump.error
                chunks = pump.drain()
                if not chunks:
                    continue
                arr = chunks[0] if len(chunks) == 1 else np.concatenate(chunks, axis=0)
                if arr.shape[0] <= 0:
                    continue
                live.add("loopback" if pump.plan["loopback"] else "mic")
                arr = to_int16(to_channels(resample(arr, pump.rate, self._rate), self._channels))
                if arr.shape[0] > 0:
                    blocks.append(arr)
            if live and len(live) == 1:
                self._source_kind = next(iter(live))
            elif live:
                self._source_kind = "mix"

            if read_error is not None:
                self._failure_streak += 1
                if self._failure_streak <= 3:
                    self._report(f"读取音频数据失败：{read_error}")
                if self._failure_streak >= _MAX_CONSECUTIVE_ERRORS:
                    self._report("音频设备连续读取失败，已停止录音")
                    break
            else:
                self._failure_streak = 0

            if not blocks:
                self._stop_evt.wait(0.01)
                continue

            count = min(block.shape[0] for block in blocks)
            if count <= 0:
                continue
            mixed = blocks[0][:count] if len(blocks) == 1 else mix_int16(blocks[0][:count], blocks[1][:count])

            level = rms_db(mixed)
            self._last_rms = level
            self._last_peak = peak_db(mixed)
            if level >= self._vad_db:
                self._last_loud = time.monotonic()

            if self._pause_evt.is_set():
                continue  # 暂停时照读（防止缓冲溢出）但不落盘

            self._write_full(full, mixed)
            parts.append(mixed)
            pending += mixed.shape[0]
            if pending >= self._chunk_samples:
                self._flush_chunk(parts, pending)
                parts = []
                pending = 0

        if pending > 0:
            self._flush_chunk(parts, pending)

    def _write_full(self, full, data: np.ndarray) -> None:
        if full is None:
            return
        try:
            full.writeframes(data.tobytes())
            self._written_samples += int(data.shape[0])
        except Exception as exc:
            self._report(f"写入整段音频失败：{exc}")

    def _flush_chunk(self, parts: list[np.ndarray], count: int) -> None:
        if not parts or count <= 0:
            return
        data = np.concatenate(parts, axis=0) if len(parts) > 1 else parts[0]
        index = self._chunk_count
        self._chunk_count = index + 1
        t_end = self._written_samples / float(self._rate)
        t_start = max(0.0, t_end - data.shape[0] / float(self._rate))
        path = self.chunks_dir / f"chunk_{index:04d}.wav"
        try:
            write_wav_int16(path, data, self._rate)
        except Exception as exc:
            self._report(f"写入音频切片失败：{exc}")
            return
        level = rms_db(data)
        chunk = AudioChunk(
            index=index,
            t_start=t_start,
            t_end=t_end,
            path=path,
            rms_db=level,
            peak_db=peak_db(data),
            silent=bool(level < self._vad_db),
            source=self._source_kind,
        )
        if self.on_chunk is None:
            return
        started = time.monotonic()
        try:
            self.on_chunk(chunk)
        except Exception as exc:
            log.warning("音频切片回调异常：%s", exc)
        cost = time.monotonic() - started
        if cost > 0.05:
            log.debug("音频切片回调耗时 %.0fms（约定不超过 50ms）", cost * 1000)
