"""采集层：音频（麦克风 + 系统回环混音）与演示画面（极域电子教室窗口）采集。

对外入口（详见 docs/INTERFACES.md 第 1 节）：

- 音频：AudioRecorder / AudioChunk
- 画面：ScreenRecorder / ScreenSourceResolver / FrameCapture / dhash / hamming
- 设备：list_audio_devices / list_monitors / list_windows / find_classroom_windows /
  describe_environment
- 转码：ffmpeg_path / ffmpeg_available / run_ffmpeg / encode_frames_to_mp4 /
  mux_audio_video / wav_duration / concat_wavs / probe_video

本包不做任何 UI 操作；所有函数/方法都不应让调用方进程崩溃。
"""
from __future__ import annotations

from .audio import (
    AudioChunk,
    AudioPlan,
    AudioRecorder,
    mix_int16,
    peak_db,
    read_wav_int16,
    resample,
    resolve_audio_devices,
    rms_db,
    select_device,
    to_channels,
    write_wav_int16,
)
from .devices import (
    AudioDeviceInfo,
    describe_environment,
    ensure_dpi_aware,
    find_classroom_windows,
    list_audio_devices,
    list_monitors,
    list_windows,
    new_mss,
    window_rect,
)
from .ffmpeg import (
    concat_wavs,
    encode_frames_to_mp4,
    ffmpeg_available,
    ffmpeg_path,
    ffmpeg_version,
    mux_audio_video,
    probe_video,
    probe_wav,
    run_ffmpeg,
    wav_duration,
)
from .screen import (
    FrameCapture,
    ScreenRecorder,
    ScreenSourceResolver,
    dhash,
    dhash_distance,
    hamming,
    should_keep_keyframe,
)

__all__ = [
    # audio
    "AudioChunk", "AudioPlan", "AudioRecorder", "mix_int16", "peak_db", "read_wav_int16",
    "resample", "resolve_audio_devices", "rms_db", "select_device", "to_channels",
    "write_wav_int16",
    # devices
    "AudioDeviceInfo", "describe_environment", "ensure_dpi_aware", "find_classroom_windows",
    "list_audio_devices", "list_monitors", "list_windows", "new_mss", "window_rect",
    # ffmpeg
    "concat_wavs", "encode_frames_to_mp4", "ffmpeg_available", "ffmpeg_path",
    "ffmpeg_version", "mux_audio_video", "probe_video", "probe_wav", "run_ffmpeg",
    "wav_duration",
    # screen
    "FrameCapture", "ScreenRecorder", "ScreenSourceResolver", "dhash", "dhash_distance",
    "hamming", "should_keep_keyframe",
]
