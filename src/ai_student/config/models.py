"""应用配置数据模型。

全部配置持久化在 %APPDATA%\\AIStudent\\config.json。
API KEY 字段在落盘时使用 Windows DPAPI 加密（见 store.py）。
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Any, Literal

CONFIG_VERSION = 1

# ---------------------------------------------------------------- 模型服务预设
# 均为 OpenAI 兼容接口，方便一处配置多模态对话 + 语音识别。
MODEL_PRESETS: dict[str, dict[str, Any]] = {
    "qwen": {
        "label": "阿里云百炼 / 通义千问（推荐，中文课堂最佳，含 ASR）",
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "chat_model": "qwen-vl-max-latest",
        "vision_model": "qwen-vl-max-latest",
        "asr_model": "qwen3-asr-flash",
        "asr_kind": "openai_audio",
        "docs": "https://bailian.console.aliyun.com/",
    },
    "openai": {
        "label": "OpenAI",
        "base_url": "https://api.openai.com/v1",
        "chat_model": "gpt-4o",
        "vision_model": "gpt-4o",
        "asr_model": "whisper-1",
        "asr_kind": "openai_transcription",
        "docs": "https://platform.openai.com/api-keys",
    },
    "zhipu": {
        "label": "智谱 AI / GLM（glm-4v 视觉，中文好）",
        "base_url": "https://open.bigmodel.cn/api/paas/v4",
        "chat_model": "glm-4v-plus",
        "vision_model": "glm-4v-plus",
        "asr_model": "glm-asr",
        "asr_kind": "openai_audio",
        "docs": "https://open.bigmodel.cn/",
    },
    "siliconflow": {
        "label": "硅基流动 SiliconFlow（聚合多家模型）",
        "base_url": "https://api.siliconflow.cn/v1",
        "chat_model": "Qwen/Qwen2.5-VL-72B-Instruct",
        "vision_model": "Qwen/Qwen2.5-VL-72B-Instruct",
        "asr_model": "FunAudioLLM/SenseVoiceSmall",
        "asr_kind": "openai_transcription",
        "docs": "https://cloud.siliconflow.cn/",
    },
    "moonshot": {
        "label": "月之暗面 Kimi",
        "base_url": "https://api.moonshot.cn/v1",
        "chat_model": "moonshot-v1-8k-vision-preview",
        "vision_model": "moonshot-v1-8k-vision-preview",
        "asr_model": "",
        "asr_kind": "openai_transcription",
        "docs": "https://platform.moonshot.cn/",
    },
    "gemini": {
        "label": "Google Gemini（OpenAI 兼容端点）",
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
        "chat_model": "gemini-2.0-flash",
        "vision_model": "gemini-2.0-flash",
        "asr_model": "whisper-1",
        "asr_kind": "openai_transcription",
        "docs": "https://aistudio.google.com/apikey",
    },
    "ollama": {
        "label": "本地 Ollama（离线，需自备视觉模型）",
        "base_url": "http://127.0.0.1:11434/v1",
        "chat_model": "qwen2.5vl:7b",
        "vision_model": "qwen2.5vl:7b",
        "asr_model": "",
        "asr_kind": "none",
        "docs": "https://ollama.com/",
    },
    "custom": {
        "label": "自定义 OpenAI 兼容服务",
        "base_url": "",
        "chat_model": "",
        "vision_model": "",
        "asr_model": "",
        "asr_kind": "openai_transcription",
        "docs": "",
    },
}

AsrKind = Literal["openai_transcription", "openai_audio", "none"]


def get_preset(key: str) -> dict[str, Any]:
    return MODEL_PRESETS.get(key) or MODEL_PRESETS["custom"]


def preset_labels() -> list[tuple[str, str]]:
    return [(k, v["label"]) for k, v in MODEL_PRESETS.items()]


# ---------------------------------------------------------------- 模型接入
@dataclass
class ModelEndpoint:
    """多模态模型接入配置。"""

    provider: str = "qwen"
    base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    api_key: str = ""
    chat_model: str = "qwen-vl-max-latest"
    vision_model: str = "qwen-vl-max-latest"
    asr_model: str = "qwen3-asr-flash"
    asr_kind: AsrKind = "openai_audio"
    asr_base_url: str = ""
    asr_api_key: str = ""
    temperature: float = 0.3
    max_tokens: int = 4096
    timeout_s: int = 120
    retries: int = 2
    extra_headers: dict[str, str] = field(default_factory=dict)

    # -- 派生字段 ----------------------------------------------------------
    @property
    def effective_asr_base_url(self) -> str:
        return self.asr_base_url or self.base_url

    @property
    def effective_asr_api_key(self) -> str:
        return self.asr_api_key or self.api_key

    @property
    def effective_vision_model(self) -> str:
        return self.vision_model or self.chat_model

    def is_configured(self) -> bool:
        return bool(self.base_url and self.api_key and self.chat_model)

    def apply_preset(self, key: str) -> None:
        preset = get_preset(key)
        self.provider = key
        for attr in ("base_url", "chat_model", "vision_model", "asr_model", "asr_kind"):
            val = preset.get(attr)
            if val:
                setattr(self, attr, val)

    def display(self) -> str:
        return f"{self.provider} / {self.chat_model}"


# ---------------------------------------------------------------- 个人信息
@dataclass
class PersonalInfo:
    """AI 扮演的学生身份。作业文件中会使用这些信息署名。"""

    name: str = ""
    student_id: str = ""
    class_name: str = ""
    grade: str = ""
    school: str = ""
    teacher_name: str = ""
    subject: str = ""
    answer_style: str = "标准作答"  # 标准作答 | 简洁 | 详细讲解 | 按考试要求
    language: str = "zh-CN"
    sign_artifacts: bool = True
    extra_notes: str = ""

    def header_line(self) -> str:
        parts = [p for p in (self.school, self.grade, self.class_name) if p]
        line = " ".join(parts)
        if self.name:
            line = f"{line}  姓名：{self.name}".strip()
        if self.student_id:
            line = f"{line}  学号：{self.student_id}"
        if self.teacher_name:
            line = f"{line}  任课教师：{self.teacher_name}"
        return line.strip()

    def is_configured(self) -> bool:
        return bool(self.name.strip())


# ---------------------------------------------------------------- 采集
ScreenSource = Literal["window", "region", "fullscreen"]


@dataclass
class CaptureConfig:
    """麦克风 + 极域电子教室远程演示画面的采集配置。"""

    # 音频
    mic_enabled: bool = True
    mic_device: str = ""            # 设备名（可为子串）；空 = 系统默认麦克风
    loopback_enabled: bool = True   # 同时录制系统声音（老师端共享音频）
    loopback_device: str = ""       # 空 = 默认扬声器回环
    sample_rate: int = 16000
    channels: int = 1
    chunk_seconds: int = 20         # 送入语音识别的切片长度
    vad_silence_db: float = -45.0   # 静音判定阈值
    keep_full_audio: bool = True    # 全程录音留档

    # 屏幕 / 演示画面
    screen_enabled: bool = True
    screen_source: ScreenSource = "window"
    window_title_keywords: list[str] = field(
        default_factory=lambda: ["极域", "TopDomain", "电子教室", "教师演示", "学生演示", "屏幕广播"]
    )
    monitor_index: int = 1
    region: list[int] = field(default_factory=lambda: [0, 0, 0, 0])  # x, y, w, h；w/h<=0 表示自动
    keyframe_interval_s: int = 8
    keyframe_change_threshold: float = 0.05  # dHash 汉明距离比例阈值
    max_keyframe_width: int = 1280
    record_video: bool = True
    video_fps: int = 2
    video_crf: int = 28

    # 录制总控
    max_session_minutes: int = 240


# ---------------------------------------------------------------- 输出
@dataclass
class OutputConfig:
    """作业产出位置与格式。"""

    output_dir: str = ""
    create_session_subfolder: bool = True
    write_formats: list[str] = field(default_factory=lambda: ["md", "docx"])
    naming_template: str = "{date}_{subject}_{session}"
    include_workpaper: bool = True       # 附解题过程
    include_transcript: bool = True      # 附课堂记录
    auto_open_folder: bool = True
    write_summary: bool = True


# ---------------------------------------------------------------- 行为
@dataclass
class BehaviorConfig:
    """AI 听课与完成作业的行为策略。"""

    auto_start_on_launch: bool = False
    auto_understand: bool = True         # 边听课边实时转写/理解
    auto_solve_tasks: bool = True        # 识别到任务自动作答
    solve_on_session_end: bool = True    # 下课时统一收尾
    min_task_confidence: float = 0.35    # 低于该置信度只记录不自动作答
    end_of_class_silence_s: int = 300    # 连续静音多久判定下课
    auto_stop_on_class_end: bool = False
    max_concurrent_ai: int = 2
    vision_analysis: bool = True         # 是否用视觉模型解析板书/PPT
    notes_every_n_chunks: int = 3        # 每 N 个音频切片汇总一次课堂笔记


# ---------------------------------------------------------------- 顶层配置
@dataclass
class AppConfig:
    version: int = CONFIG_VERSION
    model: ModelEndpoint = field(default_factory=ModelEndpoint)
    person: PersonalInfo = field(default_factory=PersonalInfo)
    capture: CaptureConfig = field(default_factory=CaptureConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    behavior: BehaviorConfig = field(default_factory=BehaviorConfig)
    prompt_overrides: dict[str, str] = field(default_factory=dict)
    last_session_id: str = ""

    # -- 便捷方法 ----------------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    def validate(self) -> list[str]:
        """返回人类可读的配置问题列表；空列表代表可以开始听课。"""
        problems: list[str] = []
        if not self.model.base_url:
            problems.append("未配置模型服务地址（设置 → 模型 API）")
        if not self.model.api_key:
            problems.append("未配置多模态模型 API KEY（设置 → 模型 API）")
        if not self.model.chat_model:
            problems.append("未配置多模态对话模型名称（设置 → 模型 API）")
        if not self.person.name.strip():
            problems.append("未填写学生姓名（设置 → 个人信息）")
        if not self.output.output_dir:
            problems.append("未指定作业输出文件夹（设置 → 作业输出）")
        return problems

    def resolved_output_dir(self):
        from pathlib import Path

        from .paths import default_output_dir

        return Path(self.output.output_dir) if self.output.output_dir else default_output_dir()
