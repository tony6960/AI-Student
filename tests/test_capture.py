"""采集层测试（INTERFACES.md 1.6）。

设计目标：在**普通笔记本（无教室环境、可能没有麦克风）**上也能全绿。
因此只测纯函数、枚举不崩、异常兜底，以及用 monkeypatch 伪造设备/抓屏后端的
"端到端"流程；需要真实硬件的部分一律不依赖。
"""
from __future__ import annotations

import shutil
import sys
import threading
import time
import wave
from pathlib import Path

import numpy as np
import pytest

_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from ai_student.capture import audio as audio_mod  # noqa: E402
from ai_student.capture import devices as devices_mod  # noqa: E402
from ai_student.capture import screen as screen_mod  # noqa: E402
from ai_student.capture.audio import (  # noqa: E402
    AudioChunk,
    AudioPlan,
    AudioRecorder,
    mix_int16,
    peak_db,
    read_wav_int16,
    resample,
    rms_db,
    select_device,
    to_channels,
    write_wav_int16,
)
from ai_student.capture.devices import (  # noqa: E402
    AudioDeviceInfo,
    describe_environment,
    find_classroom_windows,
    list_audio_devices,
    list_monitors,
    list_windows,
    window_rect,
)
from ai_student.capture.ffmpeg import (  # noqa: E402
    concat_wavs,
    encode_frames_to_mp4,
    ffmpeg_available,
    ffmpeg_path,
    mux_audio_video,
    probe_video,
    run_ffmpeg,
    wav_duration,
)
from ai_student.capture.screen import (  # noqa: E402
    FrameCapture,
    ScreenRecorder,
    ScreenSourceResolver,
    dhash,
    dhash_distance,
    hamming,
    should_keep_keyframe,
)
from ai_student.config.models import CaptureConfig  # noqa: E402

FFMPEG_OK = ffmpeg_available()
needs_ffmpeg = pytest.mark.skipif(not FFMPEG_OK, reason="本机没有可用的 ffmpeg")


# ------------------------------------------------------------------ 测试素材
def make_image(seed: int = 0, size: int = 64) -> np.ndarray:
    """结构化合成图（渐变 + 色块），dHash 才有意义。"""
    rng = np.random.default_rng(seed)
    xs = np.linspace(0, 255, size, dtype=np.float32)[None, :]
    ys = np.linspace(0, 255, size, dtype=np.float32)[:, None]
    img = ((xs * 0.6 + ys * 0.4) % 256).astype(np.uint8)
    img = np.repeat(img[:, :, None], 3, axis=2)
    for _ in range(6):
        y0, x0 = int(rng.integers(0, size - 12)), int(rng.integers(0, size - 12))
        img[y0:y0 + 10, x0:x0 + 10] = int(rng.integers(0, 256))
    return img


def make_wav(path: Path, seconds: float, rate: int = 8000, freq: float = 440.0) -> Path:
    t = np.arange(int(seconds * rate), dtype=np.float32) / rate
    data = (np.sin(2 * np.pi * freq * t) * 8000).astype(np.int16).reshape(-1, 1)
    return write_wav_int16(path, data, rate)


# ------------------------------------------------------------------ ffmpeg
def test_ffmpeg_path_is_usable():
    exe = ffmpeg_path()
    assert isinstance(exe, str) and exe
    assert Path(exe).exists() or shutil.which(exe), f"ffmpeg 路径不可用：{exe}"


def test_ffmpeg_available_returns_bool():
    assert isinstance(ffmpeg_available(), bool)


@needs_ffmpeg
def test_run_ffmpeg_version():
    proc = run_ffmpeg(["-version"], timeout=30.0)
    assert proc.returncode == 0
    assert "ffmpeg" in (proc.stdout or "").lower()


def test_run_ffmpeg_never_raises_on_bad_args():
    proc = run_ffmpeg(["-i", "根本不存在的文件.mp4", "-f", "null", "-"])
    assert proc.returncode != 0
    assert isinstance(proc.stderr, str)


def test_wav_duration_and_roundtrip(tmp_path):
    path = make_wav(tmp_path / "tone.wav", 0.7)
    assert path.exists() and path.stat().st_size > 44
    assert abs(wav_duration(path) - 0.7) < 0.05
    data, rate = read_wav_int16(path)
    assert rate == 8000 and data.shape[0] == 5600
    assert wav_duration(tmp_path / "没有这个文件.wav") == 0.0


@needs_ffmpeg
def test_concat_wavs(tmp_path):
    first = make_wav(tmp_path / "a.wav", 0.4, freq=440.0)
    second = make_wav(tmp_path / "b.wav", 0.6, freq=880.0)
    out = concat_wavs([first, second], tmp_path / "joined.wav")
    assert out is not None and out.exists()
    assert abs(wav_duration(out) - 1.0) < 0.06
    assert concat_wavs([], tmp_path / "empty.wav") is None


@needs_ffmpeg
def test_encode_frames_to_mp4_and_probe(tmp_path):
    from PIL import Image

    frame_dir = tmp_path / "frames"
    frame_dir.mkdir()
    # 故意用奇数宽高，验证 ffmpeg 的 pad 补齐（H.264 要求偶数）
    for i in range(1, 7):
        arr = make_image(seed=i, size=64)
        Image.fromarray(arr).resize((161, 121)).save(frame_dir / f"f_{i:06d}.jpg", quality=85)

    out = encode_frames_to_mp4(frame_dir, fps=2, out_path=tmp_path / "video.mp4", crf=30)
    assert out is not None and out.exists() and out.stat().st_size > 0
    info = probe_video(out)
    assert info["fps"] == pytest.approx(2.0, abs=0.01)
    assert info["width"] % 2 == 0 and info["height"] % 2 == 0
    assert info["width"] in (160, 161, 162)
    assert abs(info["duration"] - 3.0) < 0.4  # 6 帧 @2fps

    assert encode_frames_to_mp4(tmp_path / "不存在", 2, tmp_path / "x.mp4") is None
    empty = tmp_path / "empty_frames"
    empty.mkdir()
    assert encode_frames_to_mp4(empty, 2, tmp_path / "y.mp4") is None
    assert probe_video(tmp_path / "没有这个视频.mp4") == {
        "duration": 0.0, "fps": 0.0, "width": 0, "height": 0
    }


@needs_ffmpeg
def test_mux_audio_video_and_duration(tmp_path):
    from PIL import Image

    frame_dir = tmp_path / "frames"
    frame_dir.mkdir()
    for i in range(1, 5):
        Image.fromarray(make_image(seed=i, size=64)).save(frame_dir / f"f_{i:06d}.jpg")
    video = encode_frames_to_mp4(frame_dir, fps=2, out_path=tmp_path / "v.mp4")
    wav = make_wav(tmp_path / "a.wav", 2.0)
    assert video is not None
    merged = mux_audio_video(video, wav, tmp_path / "merged.mp4")
    assert merged is not None and merged.exists()
    proc = run_ffmpeg(["-i", str(merged)])
    assert "Audio:" in (proc.stderr or "")
    assert mux_audio_video(tmp_path / "缺.mp4", wav, tmp_path / "no.mp4") is None


# ------------------------------------------------------------------ 设备枚举
def test_list_monitors_fields():
    monitors = list_monitors()
    assert monitors, "至少应该枚举到一个显示器"
    for mon in monitors:
        assert set(mon) >= {"index", "left", "top", "width", "height", "primary"}
        assert isinstance(mon["index"], int) and mon["index"] >= 1
        assert mon["width"] > 0 and mon["height"] > 0
        assert isinstance(mon["primary"], bool)
    assert sum(1 for m in monitors if m["primary"]) == 1


def test_list_windows_never_raises_and_fields_complete():
    windows = list_windows()
    assert isinstance(windows, list)
    for win in windows:
        assert set(win) >= {"hwnd", "title", "rect", "width", "height", "pid", "process", "minimized"}
        assert isinstance(win["hwnd"], int) and win["hwnd"] > 0
        assert win["title"].strip()
        assert len(win["rect"]) == 4
        assert win["width"] == win["rect"][2] - win["rect"][0]
        assert win["height"] == win["rect"][3] - win["rect"][1]
        assert isinstance(win["minimized"], bool)
    # visible_only=False 也不应抛异常
    assert isinstance(list_windows(visible_only=False), list)


def test_window_rect_bad_handle():
    rect = window_rect(0)
    assert isinstance(rect, tuple) and len(rect) == 4


def test_find_classroom_windows_matches_and_sorts(monkeypatch):
    fake = [
        {"hwnd": 1, "title": "极域电子教室 学生端", "rect": [0, 0, 800, 600], "width": 800,
         "height": 600, "pid": 10, "process": "StudentMain.exe", "minimized": False},
        {"hwnd": 2, "title": "TopDomain 屏幕广播", "rect": [0, 0, 1920, 1080], "width": 1920,
         "height": 1080, "pid": 11, "process": "StudentMain.exe", "minimized": False},
        {"hwnd": 3, "title": "极域电子教室（最小化）", "rect": [0, 0, 1920, 1080], "width": 1920,
         "height": 1080, "pid": 12, "process": "StudentMain.exe", "minimized": True},
        {"hwnd": 4, "title": "记事本", "rect": [0, 0, 400, 300], "width": 400, "height": 300,
         "pid": 13, "process": "notepad.exe", "minimized": False},
    ]
    monkeypatch.setattr(devices_mod, "list_windows", lambda visible_only=True: list(fake))
    hits = find_classroom_windows(["极域", "TopDomain", "电子教室"])
    assert [h["hwnd"] for h in hits] == [2, 1, 3], "应按面积排序且最小化窗口靠后"
    assert hits[0]["rank"] == 0 and hits[0]["score"] >= hits[1]["score"]
    assert find_classroom_windows(["根本不存在的关键字"]) == []
    assert find_classroom_windows([]) == []


def test_list_audio_devices_never_raises():
    devices = list_audio_devices()
    assert isinstance(devices, list)
    for dev in devices:
        assert isinstance(dev, AudioDeviceInfo)
        assert dev.index >= 0 and dev.name
        assert dev.channels >= 1
        assert isinstance(dev.is_loopback, bool)
    # 关闭回环时不应出现回环设备
    assert all(not d.is_loopback for d in list_audio_devices(include_loopback=False))


def test_describe_environment_keys():
    env = describe_environment()
    assert set(env) >= {"audio", "monitors", "windows", "classroom_windows", "ffmpeg"}
    assert isinstance(env["audio"], list) and isinstance(env["ffmpeg"], str)


# ------------------------------------------------------------------ dHash / 关键帧
def test_dhash_same_and_inverted():
    img = make_image(seed=1, size=96)
    same = dhash(img)
    assert hamming(same, dhash(img.copy())) == 0
    inverted = 255 - img
    distance = hamming(same, dhash(inverted))
    assert distance > 128, f"黑白反转过后的距离应该很大，实际 {distance}"
    assert dhash_distance(same, dhash(inverted)) > 0.5
    assert hamming(0, 0) == 0 and hamming(0b1011, 0b0110) == 3  # 1011 ^ 0110 = 1101


def test_dhash_accepts_pil_and_hash_sizes():
    from PIL import Image

    img = make_image(seed=3, size=64)
    assert dhash(Image.fromarray(img)) == dhash(img)
    small = dhash(img, hash_size=8)
    assert 0 <= small < (1 << 64)
    assert dhash(img, hash_size=4).bit_length() <= 16


def test_dhash_uniform_images_are_flat():
    """dHash 是梯度哈希：纯色图（无论黑白）比较位全为 0 —— 这是算法固有性质。"""
    black = np.zeros((32, 32, 3), dtype=np.uint8)
    white = np.full((32, 32, 3), 255, dtype=np.uint8)
    assert dhash(black) == 0 and dhash(white) == 0


def test_should_keep_keyframe_threshold_pure_function():
    base = make_image(seed=5, size=96)
    changed = base.copy()
    changed[:48, :, :] = 255 - changed[:48, :, :]  # 上半屏大幅变化

    keep_first, distance_first = should_keep_keyframe(None, dhash(base), 0.05)
    assert keep_first is True and distance_first == 1.0

    keep_same, distance_same = should_keep_keyframe(dhash(base), dhash(base.copy()), 0.05)
    assert keep_same is False and distance_same == 0.0

    keep_changed, distance_changed = should_keep_keyframe(dhash(base), dhash(changed), 0.05)
    assert keep_changed is True and distance_changed > 0.05

    # 阈值极高时即使有变化也不保留
    keep_strict, _ = should_keep_keyframe(dhash(base), dhash(changed), 0.99)
    assert keep_strict is False


def test_frame_capture_dataclass(tmp_path):
    frame = FrameCapture(index=0, t=1.5, path=tmp_path / "kf_0000.jpg", changed=True,
                         hash_distance=0.25)
    data = frame.to_dict()
    assert data["index"] == 0 and data["changed"] is True
    assert data["t"] == 1.5 and data["hash_distance"] == 0.25


# ------------------------------------------------------------------ 来源解析
def test_resolver_falls_back_to_monitor(monkeypatch):
    monkeypatch.setattr(screen_mod, "find_classroom_windows", lambda keywords=None: [])
    cfg = CaptureConfig()
    cfg.screen_source = "window"
    resolver = ScreenSourceResolver(cfg)
    result = resolver.resolve()
    assert result["kind"] == "monitor"
    assert "回退" in result["reason"]
    assert result["rect"][2] > 0 and result["rect"][3] > 0
    assert result["hwnd"] == 0
    assert isinstance(result["monitor"], int) and result["monitor"] >= 1


def test_resolver_uses_classroom_window(monkeypatch):
    fake = [{"hwnd": 4242, "title": "极域电子教室 学生端", "rect": [10, 20, 810, 620],
             "width": 800, "height": 600, "pid": 5, "process": "StudentMain.exe",
             "minimized": False, "rank": 0, "score": 1.0}]
    monkeypatch.setattr(screen_mod, "find_classroom_windows", lambda keywords=None: list(fake))
    cfg = CaptureConfig()
    cfg.screen_source = "window"
    result = ScreenSourceResolver(cfg).resolve()
    assert result["kind"] == "window"
    assert result["rect"] == [10, 20, 800, 600]
    assert result["hwnd"] == 4242
    assert "极域" in result["title"]


def test_resolver_refresh_requeries(monkeypatch):
    calls = {"n": 0}

    def fake_find(keywords=None):
        calls["n"] += 1
        return []

    monkeypatch.setattr(screen_mod, "find_classroom_windows", fake_find)
    cfg = CaptureConfig()
    cfg.screen_source = "window"
    resolver = ScreenSourceResolver(cfg)
    resolver.resolve()
    resolver.resolve()  # 命中缓存
    assert calls["n"] == 1
    resolver.refresh()
    assert calls["n"] == 2


def test_resolver_region_and_fullscreen():
    cfg = CaptureConfig()
    cfg.screen_source = "region"
    cfg.region = [0, 0, 0, 0]  # w/h<=0 → 自动用显示器
    auto = ScreenSourceResolver(cfg).resolve()
    assert auto["kind"] == "region" and auto["rect"][2] > 0

    cfg.region = [100, 100, 640, 480]
    fixed = ScreenSourceResolver(cfg).refresh()
    assert fixed["rect"] == [100, 100, 640, 480]

    cfg.screen_source = "fullscreen"
    full = ScreenSourceResolver(cfg).refresh()
    assert full["kind"] == "monitor" and full["rect"][2] > 0


# ------------------------------------------------------------------ 音频纯函数
def test_resample_and_channels():
    src = (np.sin(np.linspace(0, 20, 4410)) * 8000).astype(np.int16).reshape(-1, 1)
    out = resample(src, 44100, 16000)
    assert out.shape[0] == 1600
    assert abs(out.shape[0] / 16000 - src.shape[0] / 44100) < 0.01
    assert resample(src, 16000, 16000).shape[0] == src.shape[0]
    assert resample(np.zeros((0, 1), dtype=np.int16), 44100, 16000).shape[0] == 0

    mono = np.arange(10, dtype=np.int16).reshape(-1, 1)
    stereo = to_channels(mono, 2)
    assert stereo.shape == (10, 2)
    assert np.array_equal(stereo[:, 0], stereo[:, 1])
    back = to_channels(stereo, 1)
    assert back.shape == (10, 1)
    assert np.allclose(back[:, 0], mono[:, 0])


def test_mix_and_levels():
    a = np.full((8, 1), 20000, dtype=np.int16)
    b = np.full((8, 1), 20000, dtype=np.int16)
    mixed = mix_int16(a, b)
    assert mixed.shape == (8, 1) and mixed.dtype == np.int16
    assert int(mixed.max()) == 32767, "相加应饱和裁剪，不能溢出回绕"

    quiet = mix_int16(np.zeros((4, 1), np.int16), np.zeros((4, 1), np.int16))
    assert int(np.abs(quiet).max()) == 0

    assert rms_db(np.zeros((100, 1), dtype=np.int16)) == -120.0
    assert peak_db(np.zeros((100, 1), dtype=np.int16)) == -120.0
    loud = np.full((100, 1), 32767, dtype=np.int16)
    assert -1.0 < peak_db(loud) < 0.0
    assert -1.0 < rms_db(loud) < 0.0

    short_a = np.zeros((3, 1), np.int16)
    short_b = np.zeros((9, 1), np.int16)
    assert mix_int16(short_a, short_b).shape[0] == 3


def test_audio_chunk_dataclass(tmp_path):
    chunk = AudioChunk(index=2, t_start=40.0, t_end=60.0, path=tmp_path / "chunk_0002.wav",
                       rms_db=-30.5, peak_db=-12.0, silent=False, source="mix")
    assert chunk.duration == pytest.approx(20.0)
    data = chunk.to_dict()
    assert data["source"] == "mix" and data["silent"] is False
    assert data["path"].endswith("chunk_0002.wav")


def test_select_device_pure():
    devices = [
        AudioDeviceInfo(0, "扬声器 (Realtek) [Loopback]", 2, 48000, "Windows WASAPI", True, False, True),
        AudioDeviceInfo(1, "麦克风阵列 (Realtek)", 2, 48000, "Windows WASAPI", False, True, False),
        AudioDeviceInfo(2, "USB 麦克风", 1, 44100, "MME", False, False, False),
    ]
    assert select_device(devices, "", loopback=False, prefer_default=True).index == 1
    assert select_device(devices, "", loopback=True, prefer_default=True).index == 0
    assert select_device(devices, "USB", loopback=False).index == 2
    assert select_device(devices, "usb 麦克风", loopback=False).index == 2  # 大小写不敏感
    # 名字没匹配上时回退到默认设备（resolve_audio_devices 会同时记一条 warning）
    fallback = select_device(devices, "找不到的设备", loopback=False, prefer_default=True)
    assert fallback is not None and fallback.index == 1  # 回退到默认麦克风
    assert select_device([], "", loopback=False) is None
    assert select_device(devices, "", loopback=None).index == 0


def test_resolve_audio_devices_uses_config(monkeypatch):
    devices = [
        AudioDeviceInfo(0, "扬声器 (Realtek) [Loopback]", 2, 48000, "WASAPI", True, False, True),
        AudioDeviceInfo(1, "麦克风阵列 (Realtek)", 2, 48000, "WASAPI", False, True, False),
    ]
    monkeypatch.setattr(audio_mod, "list_audio_devices", lambda include_loopback=True: list(devices))
    cfg = CaptureConfig()
    plan = resolve_plan(cfg)
    assert plan.kind == "mix" and plan.mic.index == 1 and plan.loopback.index == 0
    assert plan.warnings == []

    cfg.loopback_enabled = False
    plan = resolve_plan(cfg)
    assert plan.kind == "mic" and plan.loopback is None

    cfg.mic_enabled = False
    cfg.loopback_enabled = True
    plan = resolve_plan(cfg)
    assert plan.kind == "loopback"

    cfg.loopback_device = "不存在的回环"
    plan = resolve_plan(cfg)
    assert plan.loopback is not None and plan.loopback.index == 0
    assert any("不存在的回环" in w for w in plan.warnings)

    monkeypatch.setattr(
        audio_mod, "list_audio_devices",
        lambda include_loopback=True: (_ for _ in ()).throw(RuntimeError("设备枚举炸了")),
    )
    plan = resolve_plan(CaptureConfig())
    assert plan.sources == [] and plan.warnings


def resolve_plan(cfg: CaptureConfig) -> AudioPlan:
    return audio_mod.resolve_audio_devices(cfg)


# ------------------------------------------------------------------ 音频录制器
def test_audio_recorder_raises_without_devices(monkeypatch, tmp_path):
    monkeypatch.setattr(audio_mod, "resolve_audio_devices", lambda cfg: AudioPlan())
    recorder = AudioRecorder(CaptureConfig(), tmp_path)
    with pytest.raises(RuntimeError):
        recorder.start()
    assert recorder.is_running is False
    assert recorder.stop() is None  # 没启动过也不能抛
    assert recorder.stop() is None


def test_audio_recorder_raises_when_all_sources_disabled(tmp_path):
    cfg = CaptureConfig()
    cfg.mic_enabled = False
    cfg.loopback_enabled = False
    with pytest.raises(RuntimeError):
        AudioRecorder(cfg, tmp_path).start()


class _FakeStream:
    """伪造的 PortAudio 输入流：按真实速率产生 440Hz 正弦（int16）。"""

    def __init__(self, rate: int, channels: int, silent: bool = False) -> None:
        self.rate = int(rate)
        self.channels = max(1, int(channels))
        self.silent = silent
        self.position = 0
        self.stopped = False
        self.closed = False
        self.opened_at = time.monotonic()

    def get_read_available(self) -> int:
        """真实 PortAudio 的行为：返回缓冲里可读的帧数（0 = 现在读会阻塞）。"""
        available = int((time.monotonic() - self.opened_at) * self.rate)
        return max(0, available - self.position)

    def read(self, frames: int, exception_on_overflow: bool = False) -> bytes:
        frames = int(frames)
        # 真实设备在后台回调里按墙钟并行缓冲：只有"要的比已产出的多"时才等待
        available = int((time.monotonic() - self.opened_at) * self.rate)
        deficit = (self.position + frames) - available
        if deficit > 0:
            time.sleep(deficit / float(self.rate))
        t = (np.arange(self.position, self.position + frames, dtype=np.float64)) / self.rate
        self.position += frames
        if self.silent:
            block = np.zeros(frames, dtype=np.int16)
        else:
            block = (np.sin(2 * np.pi * 440.0 * t) * 8000).astype(np.int16)
        if self.channels > 1:
            block = np.repeat(block.reshape(-1, 1), self.channels, axis=1).reshape(-1)
        return block.tobytes()

    def stop_stream(self) -> None:
        self.stopped = True

    def close(self) -> None:
        self.closed = True


class _IdleLoopbackStream(_FakeStream):
    """模拟"当前没有任何声音在播放"的 WASAPI 回环设备。

    真机上这种设备的阻塞式 read() 会**永久**不返回（实测 Realtek 回环），
    因此可读帧数恒为 0，并且这里把每次 read() 调用记下来当"绊线"。
    """

    def __init__(self, rate: int, channels: int) -> None:
        super().__init__(rate, channels, silent=True)
        self.read_calls = 0

    def get_read_available(self) -> int:
        return 0

    def read(self, frames: int, exception_on_overflow: bool = False) -> bytes:
        self.read_calls += 1
        time.sleep(0.5)  # 真机上是"永久阻塞"
        return b""


class _FakePyAudioModule:
    """伪造 pyaudiowpatch 模块。"""

    paInt16 = 8

    def __init__(self, silent: bool = False, idle_indexes: tuple[int, ...] = ()) -> None:
        self.silent = silent
        self.idle_indexes = idle_indexes
        self.streams: list[_FakeStream] = []
        self.terminated = False

    def PyAudio(self):  # noqa: N802 - 模拟第三方 API 命名
        return self

    def open(self, **kwargs) -> _FakeStream:
        index = int(kwargs.get("input_device_index", -1))
        if index in self.idle_indexes:
            stream: _FakeStream = _IdleLoopbackStream(int(kwargs["rate"]), int(kwargs["channels"]))
        else:
            stream = _FakeStream(int(kwargs["rate"]), int(kwargs["channels"]), self.silent)
        self.streams.append(stream)
        return stream

    def terminate(self) -> None:
        self.terminated = True


def _fake_plan(*, mic: bool = True, loopback: bool = False) -> AudioPlan:
    plan = AudioPlan()
    if mic:
        plan.mic = AudioDeviceInfo(1, "伪造麦克风", 2, 44100, "MME", False, True, False)
    if loopback:
        plan.loopback = AudioDeviceInfo(2, "伪造扬声器 [Loopback]", 2, 48000, "WASAPI", True, False, True)
    return plan


def test_audio_recorder_end_to_end_with_fake_devices(monkeypatch, tmp_path):
    fake = _FakePyAudioModule()
    monkeypatch.setattr(audio_mod, "_load_pyaudio", lambda: fake)
    monkeypatch.setattr(audio_mod, "resolve_audio_devices", lambda cfg: _fake_plan())

    cfg = CaptureConfig()
    cfg.sample_rate = 8000
    cfg.channels = 1
    cfg.chunk_seconds = 1
    chunks: list[AudioChunk] = []
    errors: list[str] = []
    recorder = AudioRecorder(cfg, tmp_path, on_chunk=chunks.append, on_error=errors.append)

    recorder.start()
    assert recorder.is_running is True
    assert recorder.source_kind == "mic"
    time.sleep(1.35)
    assert recorder.last_rms_db > -45.0, "有正弦波输入时电平应明显高于静音阈值"
    assert recorder.silence_seconds() < 0.5
    assert 0.9 < recorder.elapsed < 2.0

    first = recorder.stop()
    assert first is not None and first == tmp_path / "audio" / "audio.wav"
    assert recorder.is_running is False
    assert recorder.stop() == first, "stop() 必须幂等"
    assert fake.terminated is True
    assert all(stream.closed for stream in fake.streams)
    assert not [t for t in threading.enumerate() if t.name.startswith("ai-student-audio")], (
        "采集相关的线程必须全部退出"
    )

    duration = wav_duration(first)
    assert 0.9 < duration < 2.0, f"整段录音时长异常：{duration}"
    data, rate = read_wav_int16(first)
    assert rate == 8000 and data.shape[1] == 1

    assert len(chunks) >= 1
    assert chunks[0].path.name == "chunk_0000.wav"
    assert chunks[0].path.exists()
    assert chunks[0].source == "mic"
    assert chunks[0].silent is False
    assert chunks[0].t_start == pytest.approx(0.0, abs=0.01)
    assert chunks[0].t_end == pytest.approx(1.0, abs=0.05)
    assert wav_duration(chunks[0].path) == pytest.approx(1.0, abs=0.05)


def test_audio_recorder_pause_resume_and_silence(monkeypatch, tmp_path):
    fake = _FakePyAudioModule(silent=True)
    monkeypatch.setattr(audio_mod, "_load_pyaudio", lambda: fake)
    monkeypatch.setattr(audio_mod, "resolve_audio_devices", lambda cfg: _fake_plan(mic=True, loopback=True))

    cfg = CaptureConfig()
    cfg.sample_rate = 8000
    cfg.channels = 1
    cfg.chunk_seconds = 5
    recorder = AudioRecorder(cfg, tmp_path)
    recorder.start()
    assert recorder.source_kind == "mix", "两路设备时应为混音"
    time.sleep(0.4)
    recorder.pause()
    paused_at = recorder.elapsed
    time.sleep(0.4)
    assert recorder.elapsed == pytest.approx(paused_at, abs=0.15), "暂停期间不应推进录制时间"
    recorder.resume()
    time.sleep(0.4)
    assert recorder.elapsed > paused_at + 0.2
    assert recorder.silence_seconds() > 0.5, "静音输入应让静音计时增长"
    path = recorder.stop()
    assert path is not None
    assert 0.6 < wav_duration(path) < 1.4


def test_audio_recorder_idle_loopback_does_not_hang(monkeypatch, tmp_path):
    """系统当前没有任何声音在播放时，回环设备一帧数据都没有。

    真机上这种设备的阻塞式 read() 会永久不返回（实测），必须靠"先问可读帧数"绕开：
    线程要能干净退出、麦克风要照常录、stop() 不能卡住。
    """
    fake = _FakePyAudioModule(idle_indexes=(2,))
    monkeypatch.setattr(audio_mod, "_load_pyaudio", lambda: fake)
    monkeypatch.setattr(audio_mod, "resolve_audio_devices", lambda cfg: _fake_plan(mic=True, loopback=True))

    cfg = CaptureConfig()
    cfg.sample_rate = 8000
    cfg.channels = 1
    cfg.chunk_seconds = 10
    recorder = AudioRecorder(cfg, tmp_path)
    recorder.start()
    time.sleep(0.8)
    started = time.monotonic()
    path = recorder.stop()
    assert time.monotonic() - started < 5.0, "stop() 不应被空闲回环设备卡住"
    assert path is not None
    assert 0.4 < wav_duration(path) < 1.6, "麦克风数据应照常录下来"
    idle = [s for s in fake.streams if isinstance(s, _IdleLoopbackStream)]
    assert idle and all(s.read_calls == 0 for s in idle), "空闲设备上不能调用阻塞式 read()"
    assert not [t for t in threading.enumerate() if t.name.startswith("ai-student-audio")]


def test_stream_pump_drain_keeps_all_blocks():
    """录音必须连续：读取泵里的积压块要全部取出，不能只留最新一块。"""
    plan = {"index": 1, "name": "伪造", "channels": 1, "rate": 8000, "loopback": False,
            "block_frames": 800}
    pump = audio_mod._StreamPump(plan, object())
    assert pump.drain() == []
    for value in (1, 2, 3, 4):
        pump.queue.put_nowait(np.full((10, 1), value, dtype=np.int16))
    items = pump.drain()
    assert len(items) == 4
    assert [int(item[0, 0]) for item in items] == [1, 2, 3, 4]
    assert pump.drain() == []


def test_audio_recorder_open_failure_raises_runtime_error(monkeypatch, tmp_path):
    class _Broken(_FakePyAudioModule):
        def open(self, **kwargs):
            raise OSError("设备被占用")

    monkeypatch.setattr(audio_mod, "_load_pyaudio", lambda: _Broken())
    monkeypatch.setattr(audio_mod, "resolve_audio_devices", lambda cfg: _fake_plan())
    with pytest.raises(RuntimeError):
        AudioRecorder(CaptureConfig(), tmp_path).start()


# ------------------------------------------------------------------ 画面录制器
class _FakeShot:
    def __init__(self, bgra: bytes, size: tuple[int, int]) -> None:
        self.bgra = bgra
        self.size = size


class _FakeMSS:
    """伪造 mss：每次抓屏换一张不同的图，方便产生关键帧。"""

    def __init__(self, width: int = 160, height: int = 120) -> None:
        self.width = width
        self.height = height
        self.grabs = 0
        self.closed = False
        self.monitors = [
            {"left": 0, "top": 0, "width": width, "height": height},
            {"left": 0, "top": 0, "width": width, "height": height, "is_primary": True},
        ]

    def grab(self, bbox) -> _FakeShot:
        self.grabs += 1
        xs = np.linspace(0, 255, self.width, dtype=np.float32)[None, :]
        ys = np.linspace(0, 255, self.height, dtype=np.float32)[:, None]
        grid = ((xs + ys + self.grabs * 20) % 256).astype(np.uint8)
        arr = np.zeros((self.height, self.width, 4), dtype=np.uint8)
        arr[:, :, 0] = grid
        arr[:, :, 1] = 255 - grid
        arr[:, :, 2] = 128
        arr[:, :, 3] = 255
        return _FakeShot(arr.tobytes(), (self.width, self.height))

    def close(self) -> None:
        self.closed = True


def test_screen_recorder_stop_before_start_is_none(tmp_path):
    recorder = ScreenRecorder(CaptureConfig(), tmp_path)
    assert recorder.stop() is None
    assert recorder.stop() is None
    assert recorder.is_running is False
    assert recorder.last_frame_path() is None
    assert recorder.preview_png() is None
    assert recorder.frame_count == 0 and recorder.keyframe_count == 0


def test_screen_recorder_disabled_is_noop(tmp_path):
    cfg = CaptureConfig()
    cfg.screen_enabled = False
    recorder = ScreenRecorder(cfg, tmp_path)
    recorder.start()
    assert recorder.is_running is False
    assert recorder.stop() is None


@needs_ffmpeg
def test_screen_recorder_end_to_end_with_fake_mss(monkeypatch, tmp_path):
    import shutil as _shutil

    fake = _FakeMSS()
    monkeypatch.setattr(screen_mod, "new_mss", lambda: fake)

    cfg = CaptureConfig()
    cfg.screen_source = "region"
    cfg.region = [0, 0, 160, 120]
    cfg.video_fps = 4
    cfg.keyframe_interval_s = 1
    cfg.record_video = True
    cfg.video_crf = 32
    keyframes: list[FrameCapture] = []
    errors: list[str] = []
    recorder = ScreenRecorder(cfg, tmp_path, on_keyframe=keyframes.append, on_error=errors.append)

    recorder.start()
    assert recorder.is_running is True
    assert recorder.source["kind"] == "region"
    time.sleep(1.4)
    assert recorder.elapsed > 0.8
    assert recorder.frame_count >= 3
    assert 0 < len(recorder.preview_png() or b"")
    assert recorder.last_frame_path() is not None

    video = recorder.stop()
    assert video is not None and video == tmp_path / "screen" / "video.mp4"
    assert video.exists() and video.stat().st_size > 0
    assert recorder.stop() == video, "stop() 必须幂等"
    assert fake.closed is True
    assert not (tmp_path / "screen" / "frames").exists(), "合成后应删除帧目录"
    assert len(keyframes) >= 1 and keyframes[0].index == 0
    assert all(frame.path.exists() for frame in keyframes)

    info = probe_video(video)
    assert info["width"] == 160 and info["height"] == 120
    assert abs(info["duration"] - recorder.elapsed) < 1.0
    assert errors == [] or all(isinstance(e, str) for e in errors)
    assert _shutil is not None


@needs_ffmpeg
def test_screen_recorder_keyframe_dedup_keeps_first_and_skips_static(monkeypatch, tmp_path):
    class _StaticMSS(_FakeMSS):
        def __init__(self) -> None:
            super().__init__()
            self._cache: bytes | None = None

        def grab(self, bbox) -> _FakeShot:
            if self._cache is None:
                self._cache = super().grab(bbox).bgra
            self.grabs += 1
            return _FakeShot(self._cache, (self.width, self.height))

    monkeypatch.setattr(screen_mod, "new_mss", lambda: _StaticMSS())
    cfg = CaptureConfig()
    cfg.screen_source = "region"
    cfg.region = [0, 0, 160, 120]
    cfg.video_fps = 4
    cfg.keyframe_interval_s = 1
    cfg.record_video = False
    keyframes: list[FrameCapture] = []
    recorder = ScreenRecorder(cfg, tmp_path, on_keyframe=keyframes.append)

    recorder.start()
    time.sleep(1.3)
    video = recorder.stop()
    assert video is None, "record_video=False 时不应产出视频"
    assert len(keyframes) == 1, "画面不变时只应保留第一帧"
    assert keyframes[0].path.name == "kf_0000.jpg"
    assert keyframes[0].changed is True and keyframes[0].hash_distance == 1.0
    assert recorder.frame_count == 0


def test_screen_recorder_window_fallback_warns_once(monkeypatch, tmp_path):
    """窗口已最小化时退回整屏抓取，只警告一次且不中断。"""
    fake = _FakeMSS()
    monkeypatch.setattr(screen_mod, "new_mss", lambda: fake)
    monkeypatch.setattr(screen_mod, "_window_alive", lambda hwnd: True)
    monkeypatch.setattr(screen_mod, "_window_minimized", lambda hwnd: True)
    monkeypatch.setattr(screen_mod, "_WINDOW_GRACE_SECONDS", 0.4)
    monkeypatch.setattr(
        screen_mod,
        "find_classroom_windows",
        lambda keywords=None: [
            {"hwnd": 123, "title": "极域电子教室 学生端", "rect": [0, 0, 800, 600],
             "width": 800, "height": 600, "pid": 9, "process": "StudentMain.exe",
             "minimized": True, "rank": 0, "score": 1.0}
        ],
    )

    cfg = CaptureConfig()
    cfg.screen_source = "window"
    cfg.record_video = False
    cfg.video_fps = 5
    cfg.keyframe_interval_s = 1
    errors: list[str] = []
    recorder = ScreenRecorder(cfg, tmp_path, on_error=errors.append)
    recorder.start()
    assert recorder.source["kind"] == "window" and recorder.source["hwnd"] == 123

    # 宽限期内：一帧都不抓。整屏里是老师屏幕之外的桌面内容，录进去会污染课堂记录。
    time.sleep(0.2)
    assert fake.grabs == 0, "窗口刚不可用时应先跳过帧，不能立刻抓整屏"
    assert not [e for e in errors if "整屏" in e], "宽限期内不该告警"

    # 超过宽限期：回退整屏继续录，并且只告警一次
    time.sleep(1.2)
    recorder.stop()
    assert fake.grabs >= 2, "持续不可用后应回退整屏继续抓屏"
    assert sum(1 for e in errors if "整屏" in e) == 1, "只应告警一次"


def test_screen_recorder_survives_broken_grab(monkeypatch, tmp_path):
    class _BrokenMSS(_FakeMSS):
        def grab(self, bbox):
            raise RuntimeError("抓屏炸了")

    monkeypatch.setattr(screen_mod, "new_mss", lambda: _BrokenMSS())
    cfg = CaptureConfig()
    cfg.screen_source = "region"
    cfg.region = [0, 0, 160, 120]
    cfg.video_fps = 5
    cfg.record_video = False
    errors: list[str] = []
    recorder = ScreenRecorder(cfg, tmp_path, on_error=errors.append)
    recorder.start()
    time.sleep(0.6)
    assert recorder.is_running is True, "单次抓屏失败不应终止采集线程"
    assert recorder.stop() is None
    assert any("抓屏失败" in e for e in errors)


def test_import_surface_matches_contract():
    """契约里的名字都必须存在（防止重构时误删）。"""
    import ai_student.capture as capture

    for name in (
        "ffmpeg_path", "ffmpeg_available", "run_ffmpeg", "encode_frames_to_mp4",
        "mux_audio_video", "wav_duration", "concat_wavs", "probe_video",
        "AudioDeviceInfo", "list_audio_devices", "list_monitors", "list_windows",
        "find_classroom_windows", "describe_environment",
        "AudioChunk", "AudioRecorder",
        "FrameCapture", "dhash", "hamming", "ScreenSourceResolver", "ScreenRecorder",
    ):
        assert hasattr(capture, name), f"缺少契约成员：{name}"
