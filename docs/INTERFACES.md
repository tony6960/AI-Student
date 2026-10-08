# AI学生 —— 冻结接口契约 (v1)

> 本文件是并行开发的唯一协调依据。**不要修改本文件**；如发现契约有缺陷，
> 在代码中用最小改动兼容，并在说明里写清楚。

## 0. 环境与约定

- 项目根：`D:\Code\MixProject\AI学生`
- 解释器（务必用这个）：`D:\Code\MixProject\AI学生\.venv\Scripts\python.exe`
  （基于 CPython 3.14 + `--system-site-packages`，已装 PySide6 6.11 / opencv / numpy /
  Pillow / mss 10.2 / PyAudioWPatch 0.2.12.9 / soundcard / soundfile / imageio-ffmpeg 0.6 /
  openai 2.29 / dashscope / python-docx / pywin32 / pytest / pytest-qt）
- 源码根：`src/`，包名 `ai_student`
- 运行测试（PowerShell，项目根）：

      $env:PYTHONPATH='D:\Code\MixProject\AI学生\src'
      & 'D:\Code\MixProject\AI学生\.venv\Scripts\python.exe' -m pytest tests -q

- Windows 控制台是 GBK：**不要用中文 print 做断言**；写文件一律 encoding="utf-8"。
- 全部面向用户的文案用简体中文；代码注释用中文。
- 不要引入 requirements.txt 之外的新依赖。
- 单文件尽量 < 700 行；类型标注齐全（from __future__ import annotations）。

已经存在、**不要改动**的基础设施（可以直接 import）：

- ai_student.config：AppConfig / ModelEndpoint / PersonalInfo / CaptureConfig /
  OutputConfig / BehaviorConfig；load_config() / save_config() / protect_secret() /
  unprotect_secret()；app_root() / config_path() / sessions_root() / logs_root() /
  cache_root() / ensure_dirs() / default_output_dir() / local_now() / timestamp_slug() /
  sanitize_filename()
- ai_student.core：TranscriptSegment / BoardNote / ClassTask / Artifact / SessionInfo /
  SessionBundle / new_id() / now_iso()
- ai_student.core.api.ClassAgentAPI：GUI ↔ 引擎契约（信号 + 方法，见该文件）
- ai_student.utils：setup_logging / get_logger / compact_join / clip / normalize_ws /
  parse_json_loose / strip_code_fence / hhmmss

---

## 1. ai_student.capture —— 采集层（音频 + 极域演示画面）

### 1.1 capture/ffmpeg.py

    def ffmpeg_path() -> str            # imageio_ffmpeg.get_ffmpeg_exe()，失败回退 "ffmpeg"
    def ffmpeg_available() -> bool      # 能执行 -version
    def run_ffmpeg(args: list[str], timeout: float | None = 300.0) -> subprocess.CompletedProcess
    def encode_frames_to_mp4(frame_dir: Path, fps: int, out_path: Path, crf: int = 28,
                             pattern: str = "f_%06d.jpg") -> Path | None   # 失败返回 None，不抛
    def mux_audio_video(video: Path, audio: Path, out_path: Path) -> Path | None
    def wav_duration(path: Path) -> float
    def concat_wavs(paths: list[Path], out_path: Path) -> Path | None
    def probe_video(path: Path) -> dict     # {"duration": float, "fps": float, "width": int, "height": int}

### 1.2 capture/devices.py

    @dataclass
    class AudioDeviceInfo:
        index: int; name: str; channels: int; sample_rate: int
        host_api: str; is_loopback: bool; is_default_input: bool; is_default_output: bool

    def list_audio_devices(include_loopback: bool = True) -> list[AudioDeviceInfo]
        # 用 PyAudioWPatch；任何异常都要吞掉并返回已收集到的列表（不要让 GUI 崩）

    def list_monitors() -> list[dict]
        # [{"index":1,"left":0,"top":0,"width":1920,"height":1080,"primary":True}, ...]  用 mss

    def list_windows(visible_only: bool = True) -> list[dict]
        # [{"hwnd":int,"title":str,"rect":[l,t,r,b],"width":int,"height":int,
        #   "pid":int,"process":str,"minimized":bool}, ...]  用 pywin32；跳过空标题

    def find_classroom_windows(keywords: list[str] | None = None) -> list[dict]
        # 默认 keywords = CaptureConfig().window_title_keywords；标题包含任一关键字
        # （大小写不敏感）的窗口，按"最可能正在演示"排序（最大化/面积大优先）

    def describe_environment() -> dict
        # {"audio": [...], "monitors": [...], "windows": [...],
        #  "classroom_windows": [...], "ffmpeg": str}

### 1.3 capture/audio.py

    @dataclass
    class AudioChunk:
        index: int; t_start: float; t_end: float; path: Path
        rms_db: float; peak_db: float; silent: bool; source: str   # "mix" | "mic" | "loopback"

    class AudioRecorder:
        def __init__(self, cfg: CaptureConfig, session_dir: Path,
                     on_chunk: Callable[[AudioChunk], None] | None = None,
                     on_error: Callable[[str], None] | None = None) -> None
        def start(self) -> None                  # 起线程；无可用设备时抛 RuntimeError
        def stop(self) -> Path | None            # 停线程 + 写盘；返回整段 wav（audio/audio.wav）
        def pause(self) -> None
        def resume(self) -> None
        @property
        def is_running(self) -> bool
        @property
        def elapsed(self) -> float
        @property
        def last_rms_db(self) -> float           # 供 GUI 电平表
        def silence_seconds(self) -> float       # 距最后一次"有声"的秒数（用于判定下课）

行为要求：
- 麦克风（cfg.mic_enabled，cfg.mic_device 为名称子串，空=默认输入）与系统回环
  （cfg.loopback_enabled，cfg.loopback_device 同理，空=默认扬声器回环，用
  pyaudiowpatch.get_default_wasapi_loopback()）可独立开关，两者都开时混音。
- 统一重采样到 cfg.sample_rate / cfg.channels（int16）。设备原生采样率不同也要能工作
  （线性插值即可）。
- 整段写入 <session>/audio/audio.wav；每 cfg.chunk_seconds 秒把该片写入
  <session>/audio/chunks/chunk_%04d.wav 并回调 on_chunk（回调在采集线程里，实现方保证
  不阻塞 >50ms；引擎侧会自己丢队列）。
- silent：该片 RMS 低于 cfg.vad_silence_db。
- 设备中途失效（拔麦）不要崩：记录错误、继续或优雅停止，调 on_error。
- 线程安全退出；stop() 幂等。

### 1.4 capture/screen.py

    @dataclass
    class FrameCapture:
        index: int; t: float; path: Path; changed: bool; hash_distance: float

    def dhash(image, hash_size: int = 16) -> int          # numpy HxWx3 uint8 / PIL.Image
    def hamming(a: int, b: int) -> int

    class ScreenSourceResolver:
        def __init__(self, cfg: CaptureConfig) -> None
        def resolve(self) -> dict
            # {"kind": "window"|"region"|"monitor", "rect": [l,t,w,h], "hwnd": int,
            #  "title": str, "monitor": int, "reason": str}
        def refresh(self) -> dict   # 窗口移动/缩放后重新解析；找不到窗口时退回 monitor

    class ScreenRecorder:
        def __init__(self, cfg: CaptureConfig, session_dir: Path,
                     on_keyframe: Callable[[FrameCapture], None] | None = None,
                     on_error: Callable[[str], None] | None = None) -> None
        def start(self) -> None
        def stop(self) -> Path | None          # 返回 <session>/screen/video.mp4；未开录制返回 None
        def pause(self) -> None
        def resume(self) -> None
        @property
        def is_running(self) -> bool
        @property
        def elapsed(self) -> float
        def last_frame_path(self) -> Path | None
        def preview_png(self) -> bytes | None   # 最近一帧 PNG 字节，供 GUI 预览（可失败返回 None）

行为要求：
- 关键帧：每 cfg.keyframe_interval_s 秒抓一帧（并按 cfg.video_fps 抓录像帧）；与上一张关键帧
  算 dHash 汉明距离比例，> cfg.keyframe_change_threshold 才落盘为
  <session>/screen/keyframes/kf_%04d.jpg（JPEG，宽度压到 cfg.max_keyframe_width）并回调。
  会话第一帧永远算关键帧。t = 距 start() 的秒数。
- 录像帧写 <session>/screen/frames/f_%06d.jpg，stop() 时用 ffmpeg 合成
  <session>/screen/video.mp4（H.264, cfg.video_crf），合成后删除 frames 目录。
- **mss 实例必须在采集线程内创建**（跨线程会崩）；pywin32 窗口 API 在主线程解析后把 rect 传进去。
- 窗口最小化/被遮挡时：记录一次警告，退回整屏抓取，不要中断。
- DPI：模块导入时尝试 ctypes.windll.shcore.SetProcessDpiAwareness(2)（异常忽略）。
- 线程安全退出；stop() 幂等。

### 1.5 tools/probe_capture.py（也归你写）

命令行自检工具：python tools/probe_capture.py [--record 10]
打印音频设备 / 显示器 / 窗口 / 极域候选窗口 / ffmpeg 路径；带 --record N 时在临时目录录
N 秒音频 + 画面，打印产物路径、时长、大小。

### 1.6 tests/test_capture.py

必须能在**普通笔记本（无教室环境）**上通过：只测纯函数、枚举不崩、异常兜底。
至少覆盖：ffmpeg_available()、ffmpeg_path() 指向存在的文件、list_monitors() 非空且字段完整、
list_windows() 不抛且字段完整、dhash()/hamming() 语义（相同图距离 0，黑白反转距离大）、
ScreenSourceResolver 在找不到窗口时回退 monitor、AudioRecorder 在无设备时抛 RuntimeError
或优雅失败（用 monkeypatch 伪造）。另加一个纯函数级测试：用合成 numpy 图验证关键帧
去重阈值判定辅助函数。

---

## 2. ai_student.ai —— 模型层（转写 / 视觉 / 笔记 / 任务 / 作答）

全部函数**同步阻塞**（引擎放在线程池里调用），**不做 UI 操作**，失败抛
AIClientError（定义在 ai/errors.py，含 message / status / retryable）。

### 2.1 ai/client.py

    class LLMClient:
        def __init__(self, cfg: ModelEndpoint) -> None
        def chat(self, system: str, user: str, images: list[bytes | Path] | None = None,
                 model: str | None = None, json_mode: bool = False,
                 temperature: float | None = None, max_tokens: int | None = None) -> str
        def chat_messages(self, messages: list[dict], model: str | None = None,
                          json_mode: bool = False) -> str
        def transcribe(self, wav_path: Path) -> str     # 见 2.2
        @property
        def vision_model(self) -> str

- 用 openai.OpenAI(base_url=..., api_key=..., timeout=...)；extra_headers 透传。
  图片以 data URL (data:image/jpeg;base64,...) 放进 {"type":"image_url","image_url":{"url":...}}
  content part。
- 自动重试：cfg.retries 次，指数退避（0.8s/1.6s/3.2s），仅对超时/429/5xx 重试。
- 失败抛 AIClientError；错误信息里**不能包含 API KEY**。
- 加一个 ping() -> tuple[bool, str]：发一条极短消息验证连通性，GUI「测试连接」用。

### 2.2 ai/asr.py

    def transcribe_chunk(client: LLMClient, wav_path: Path, cfg: ModelEndpoint,
                         hint: str = "") -> str

按 cfg.asr_kind 分派：
- openai_transcription：client.audio.transcriptions.create(model=cfg.asr_model, file=...)
- openai_audio：chat.completions，content 里放 {"type":"input_audio","input_audio":{"data":<b64>,"format":"wav"}}
  （Qwen qwen3-asr-flash / GLM 走这条）；系统提示词要求"只输出课堂原话文字"。
- none：抛 AIClientError("未配置语音识别模型")
返回去空白后的文本；空音频（全静音）直接返回 "" （不调 API）。

### 2.3 ai/vision.py

    @dataclass
    class BoardAnalysis:
        ocr_text: str = ""      # 画面上的文字（板书/PPT/题目），保留原样
        summary: str = ""       # 这一屏在讲什么（1-3 句）
        key_points: list[str] = []
        problems: list[str] = []   # 画面上出现的题目/任务原文
        subject_hint: str = ""

    def analyze_board(client: LLMClient, image: bytes | Path, cfg: ModelEndpoint,
                      context: str = "") -> BoardAnalysis

### 2.4 ai/notes.py

    def merge_notes(client: LLMClient, previous_markdown: str,
                    transcript_lines: list[str], board_lines: list[str],
                    cfg: ModelEndpoint, person: PersonalInfo,
                    subject: str = "") -> str      # 返回**完整**的新版课堂笔记 Markdown

笔记结构固定为：# 课堂笔记 / ## 课程信息 / ## 知识点 / ## 板书与演示要点 /
## 老师布置的任务 / ## 存疑。previous_markdown 为空时从零生成。

### 2.5 ai/tasks.py

    def extract_tasks(client: LLMClient, cfg: ModelEndpoint, person: PersonalInfo,
                      notes_markdown: str, recent_transcript: str = "",
                      known_tasks: list[ClassTask] | None = None,
                      subject: str = "") -> list[ClassTask]

- 输出 JSON 数组，字段：title(必填)、detail、task_type、requirements(list[str])、deadline、
  confidence(0-1)、evidence(老师原话片段)、subject。
- 用 utils.parse_json_loose；解析失败返回 []（**不要抛**，只记日志）。
- **去重**：与 known_tasks 标题相似度 > 0.75 视为同一条，不重复产出
  （写一个 title_similarity(a, b) -> float 纯函数，用字符 2-gram 的 Jaccard）。
- 非"老师布置的任务"（如老师闲聊）不能产出。置信度低也要返回，由引擎按阈值处理。

### 2.6 ai/solver.py

    @dataclass
    class SolveResult:
        answer: str            # 最终作业正文（Markdown，可直接给学生看）
        workpaper: str         # 解题过程 / 思路（Markdown），可为空
        file_title: str        # 建议的文件名（不含扩展名，不要带路径分隔符）
        self_check: str = ""   # 自检结论
        needs_human: bool = False

    def solve_task(client: LLMClient, cfg: ModelEndpoint, person: PersonalInfo,
                   task: ClassTask, context: str, output_cfg: OutputConfig) -> SolveResult

- context 是引擎提供的相关资料（课堂笔记 + 相关转写片段 + 板书要点）。
- 提示词要求：**先理解题意**，再作答；不确定就写清楚假设；按 person.answer_style 决定详略；
  署名按 person.sign_artifacts。
- 数学/理科题给完整过程；语文英语作文按字数要求；编程题给可运行代码块。
- 内部解析用 ### 题目理解 / ### 解答 / ### 自检 三段，但 answer 只要最终正文（不含分段标记）。

### 2.7 ai/prompts.py

所有提示词常量集中在这里（简体中文），文件头写清楚每个提示词的用途与输出契约。
提供 get_prompt(key: str, overrides: dict[str,str] | None = None) -> str，
key 至少包括：asr、vision_board、notes_merge、task_extract、solve、self_check、summary。

### 2.8 tests/test_ai.py

用**假 client**（内联 FakeLLM 类，不做网络调用）覆盖：title_similarity 去重、
extract_tasks 对畸形 JSON/空返回的健壮性、merge_notes 拼接、SolveResult 解析、
get_prompt 覆盖机制、AIClientError 不含密钥。**测试必须完全离线可跑。**

---

## 3. ai_student.ui —— PySide6 界面

只能依赖 ai_student.core.api.ClassAgentAPI（**不要 import core.controller / capture / ai**）
以及 ai_student.config 的数据类。

### 3.1 入口

    class MainWindow(QMainWindow):
        def __init__(self, api: ClassAgentAPI, parent: QWidget | None = None) -> None
        def closeEvent(self, event) -> None   # 只发信号；shutdown 由 app.py 负责

### 3.2 主界面布局（左侧导航 + 右侧堆叠页面）

1. **听课（Live）**：状态胶囊（待机/听课中/暂停/收尾）、会话信息（科目、标题可编辑）、
   开始/暂停/停止按钮、麦克风电平条、演示画面实时预览（QLabel，定时器刷新）、
   实时转写列表（时间码 + 文本，自动滚动，可搜索）、当前课堂笔记预览。
2. **任务**：表格（标题/类型/状态/置信度/产出数），右键或按钮：立即完成 / 重做（弹输入框收反馈）/
   打开产出文件 / 删除 / 手动添加任务。
3. **作业产出**：产物列表 + 打开输出文件夹按钮 + 一键导出本次会话。
4. **历史**：会话列表（api.list_sessions()），双击载入并展示转写/板书/任务。
5. **日志**：滚动日志 + 导出日志。
6. **设置**：SettingsDialog 五个页签 —— 模型 API（预设下拉、base_url、api_key(密文框)、
   对话模型、视觉模型、语音识别模型与类型、温度、超时、**测试连接**按钮）、个人信息、
   采集（音频设备下拉刷新、回环开关、演示画面来源:窗口/区域/整屏、窗口关键字、刷新窗口列表、
   关键帧间隔、录制视频开关、时长上限）、作业输出（目录选择、格式勾选 md/txt/docx、
   子文件夹、命名模板、自动打开）、高级（行为策略开关、并发数、下课静音判定、提示词覆盖编辑框）。

### 3.3 其它

- theme.py：apply_theme(app)，深色现代风格 QSS + 中文字体回退
  ("Microsoft YaHei UI", "Microsoft YaHei", "Segoe UI")。
- tray.py：class TrayController(QObject)，托盘图标 + 菜单（显示主窗口/开始听课/停止/退出）。
  图标用代码绘制（QPixmap + QPainter），**不要引入图片资源文件**。
- widgets.py：LevelMeter(QWidget)（电平）、StatusPill(QWidget)、LogView(QPlainTextEdit)、
  TimecodeList(QListWidget)。
- dialogs.py：首次运行向导 FirstRunWizard(QWizard)（3 步：模型 API → 个人信息 → 输出目录），
  以及 TextPromptDialog（重做反馈用）。
- 所有耗时操作（probe_devices、ping）**不得阻塞 UI 线程**。

### 3.35 【Lead 补充 v1.1】新增三个只读轮询方法

为支持实时电平条与画面预览，ClassAgentAPI 追加了三个**只读**方法（实现已就绪，直接用）：

    def audio_level_db(self) -> float      # 当前音频电平 dBFS，未采集返回 -96.0，可 100ms 轮询
    def preview_png(self) -> bytes | None  # 最近一帧演示画面 PNG 字节，不可用返回 None
    def engine_state(self) -> str          # idle/recording/paused/finalizing/finished/error

### 3.4 tests/test_ui_smoke.py

QT_QPA_PLATFORM=offscreen 下用内联 MockController(ClassAgentAPI)（实现全部方法，返回合理空值）
实例化 MainWindow、SettingsDialog、FirstRunWizard，把 api 的全部信号各 emit 一次，断言不抛异常。
测试文件里自己设置 os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")（在 import PySide6 之前）。
