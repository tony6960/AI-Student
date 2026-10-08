"""设置界面：五个页签的 SettingsForm + 对话框外壳 SettingsDialog。

- SettingsForm 可独立嵌入主界面“设置”页，也可被 SettingsDialog 包一层按钮使用。
- 设备枚举（probe_devices）与“测试连接”都在工作线程执行，不阻塞 UI 线程。
- 保存动作只是把配置交给 api.update_config（毫秒级：json + DPAPI 写盘），
  因此直接调用；出错只弹提示，不抛异常。
"""
from __future__ import annotations

import copy
import json

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QTabWidget,
    QPlainTextEdit,
    QVBoxLayout,
    QWidget,
)

from ..config import AppConfig, MODEL_PRESETS
from ..config.paths import default_output_dir
from .dialogs import ASR_KINDS, ANSWER_STYLES, WRITE_FORMATS, check_endpoint
from .widgets import as_dict, make_button, run_async

SCREEN_SOURCES: list[tuple[str, str]] = [
    ("window", "极域演示窗口（推荐，自动跟随）"),
    ("region", "自定义区域"),
    ("fullscreen", "整屏"),
]

SAMPLE_RATES: list[int] = [8000, 16000, 22050, 44100, 48000]

PROMPT_KEYS = ["asr", "vision_board", "notes_merge", "task_extract", "solve", "self_check", "summary"]


def _hint(text: str) -> QLabel:
    label = QLabel(text)
    label.setObjectName("fieldHint")
    label.setWordWrap(True)
    return label


def _scrollable(inner: QWidget) -> QScrollArea:
    """把长表单放进滚动区，小窗口下也能操作。"""
    area = QScrollArea()
    area.setWidgetResizable(True)
    area.setFrameShape(QScrollArea.Shape.NoFrame)
    area.setWidget(inner)
    return area


def _spin(minimum: int, maximum: int, value: int, suffix: str = "", step: int = 1) -> QSpinBox:
    box = QSpinBox()
    box.setRange(minimum, maximum)
    box.setValue(value)
    box.setSingleStep(step)
    if suffix:
        box.setSuffix(suffix)
    return box


def _dspin(minimum: float, maximum: float, value: float, step: float = 0.1, decimals: int = 2) -> QDoubleSpinBox:
    box = QDoubleSpinBox()
    box.setRange(minimum, maximum)
    box.setValue(value)
    box.setSingleStep(step)
    box.setDecimals(decimals)
    return box


class SettingsForm(QWidget):
    """五页签设置表单（不含底部按钮）。"""

    def __init__(self, cfg: AppConfig | None = None, api=None, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._api = api
        self._cfg: AppConfig = copy.deepcopy(cfg) if cfg is not None else AppConfig()
        self._warnings: list[str] = []
        self._devices: dict[str, list] = {"audio": [], "windows": [], "monitors": []}

        self.tabs = QTabWidget()
        self.tabs.addTab(_scrollable(self._build_model_page()), "模型 API")
        self.tabs.addTab(_scrollable(self._build_person_page()), "个人信息")
        self.tabs.addTab(_scrollable(self._build_capture_page()), "采集")
        self.tabs.addTab(_scrollable(self._build_output_page()), "作业输出")
        self.tabs.addTab(_scrollable(self._build_advanced_page()), "高级")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.tabs)
        self.set_config(self._cfg)

    # ------------------------------------------------------------ 页签 1：模型 API
    def _build_model_page(self) -> QWidget:
        page = QWidget()
        form = QFormLayout(page)
        form.setLabelAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        form.setSpacing(9)

        self.cmb_preset = QComboBox()
        for key, meta in MODEL_PRESETS.items():
            self.cmb_preset.addItem(str(meta.get("label", key)), key)
        self.cmb_preset.addItem("自定义（手动填写）", "custom_manual")

        self.edt_base_url = QLineEdit()
        self.edt_api_key = QLineEdit()
        self.edt_api_key.setEchoMode(QLineEdit.EchoMode.Password)
        self.chk_show_key = QCheckBox("显示")
        key_row = QWidget()
        key_layout = QHBoxLayout(key_row)
        key_layout.setContentsMargins(0, 0, 0, 0)
        key_layout.addWidget(self.edt_api_key, 1)
        key_layout.addWidget(self.chk_show_key)

        self.edt_chat = QLineEdit()
        self.edt_vision = QLineEdit()
        self.edt_asr = QLineEdit()
        self.cmb_asr_kind = QComboBox()
        for key, label in ASR_KINDS:
            self.cmb_asr_kind.addItem(label, key)
        self.edt_asr_base_url = QLineEdit()
        self.edt_asr_base_url.setPlaceholderText("留空则沿用上面的服务地址")
        self.spn_temperature = _dspin(0.0, 2.0, 0.3, 0.1, 2)
        self.spn_max_tokens = _spin(256, 32768, 4096, "", 256)
        self.spn_timeout = _spin(10, 1800, 120, " 秒", 10)
        self.spn_retries = _spin(0, 5, 2, " 次")

        self.btn_test = make_button("测试连接", "ghost", "请求服务地址的 /models 接口")
        self.lbl_test = _hint("未测试。")
        test_row = QWidget()
        test_layout = QHBoxLayout(test_row)
        test_layout.setContentsMargins(0, 0, 0, 0)
        test_layout.addWidget(self.btn_test)
        test_layout.addWidget(self.lbl_test, 1)

        form.addRow("服务预设", self.cmb_preset)
        form.addRow("服务地址", self.edt_base_url)
        form.addRow("API KEY", key_row)
        form.addRow("对话模型", self.edt_chat)
        form.addRow("视觉模型", self.edt_vision)
        form.addRow("语音识别模型", self.edt_asr)
        form.addRow("识别方式", self.cmb_asr_kind)
        form.addRow("识别接口地址", self.edt_asr_base_url)
        form.addRow("温度", self.spn_temperature)
        form.addRow("单次最大 tokens", self.spn_max_tokens)
        form.addRow("请求超时", self.spn_timeout)
        form.addRow("失败重试", self.spn_retries)
        form.addRow("", test_row)

        self.cmb_preset.currentIndexChanged.connect(self._on_preset_changed)
        self.chk_show_key.toggled.connect(
            lambda on: self.edt_api_key.setEchoMode(
                QLineEdit.EchoMode.Normal if on else QLineEdit.EchoMode.Password
            )
        )
        self.btn_test.clicked.connect(self._on_test_clicked)
        return page

    def _on_preset_changed(self, index: int) -> None:
        key = self.cmb_preset.itemData(index)
        if key in (None, "custom_manual"):
            return
        meta = MODEL_PRESETS.get(str(key), {})
        self.edt_base_url.setText(str(meta.get("base_url", "")))
        self.edt_chat.setText(str(meta.get("chat_model", "")))
        self.edt_vision.setText(str(meta.get("vision_model", "")))
        self.edt_asr.setText(str(meta.get("asr_model", "")))
        pos = self.cmb_asr_kind.findData(str(meta.get("asr_kind", "openai_transcription")))
        if pos >= 0:
            self.cmb_asr_kind.setCurrentIndex(pos)

    def _on_test_clicked(self) -> None:
        self.btn_test.setEnabled(False)
        self.lbl_test.setText("正在测试连接……")
        run_async(
            check_endpoint,
            self._on_test_done,
            self._on_test_failed,
            self.edt_base_url.text().strip(),
            self.edt_api_key.text().strip(),
        )

    def _on_test_done(self, result: object) -> None:
        ok, message = result if isinstance(result, tuple) and len(result) == 2 else (False, "未知结果")
        self.lbl_test.setText(("✓ " if ok else "✗ ") + str(message))
        self.btn_test.setEnabled(True)

    def _on_test_failed(self, error: str) -> None:
        self.lbl_test.setText(f"✗ 测试失败：{error}")
        self.btn_test.setEnabled(True)

    # ------------------------------------------------------------ 页签 2：个人信息
    def _build_person_page(self) -> QWidget:
        page = QWidget()
        form = QFormLayout(page)
        form.setLabelAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        form.setSpacing(9)

        self.edt_name = QLineEdit()
        self.edt_student_id = QLineEdit()
        self.edt_class = QLineEdit()
        self.edt_grade = QLineEdit()
        self.edt_school = QLineEdit()
        self.edt_teacher = QLineEdit()
        self.edt_subject = QLineEdit()
        self.cmb_style = QComboBox()
        self.cmb_style.addItems(ANSWER_STYLES)
        self.edt_language = QLineEdit()
        self.chk_sign = QCheckBox("在作业中署名")
        self.edt_extra = QPlainTextEdit()
        self.edt_extra.setFixedHeight(80)

        form.addRow("姓名 *", self.edt_name)
        form.addRow("学号", self.edt_student_id)
        form.addRow("班级", self.edt_class)
        form.addRow("年级", self.edt_grade)
        form.addRow("学校", self.edt_school)
        form.addRow("任课教师", self.edt_teacher)
        form.addRow("默认科目", self.edt_subject)
        form.addRow("作答风格", self.cmb_style)
        form.addRow("语言", self.edt_language)
        form.addRow("", self.chk_sign)
        form.addRow("补充说明", self.edt_extra)
        return page

    # ------------------------------------------------------------ 页签 3：采集
    def _build_capture_page(self) -> QWidget:
        page = QWidget()
        outer = QVBoxLayout(page)
        outer.setSpacing(10)

        audio_form = QFormLayout()
        audio_form.setLabelAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        audio_form.setSpacing(9)
        self.chk_mic = QCheckBox("录制麦克风（老师现场讲课）")
        self.cmb_mic_device = QComboBox()
        self.cmb_mic_device.setEditable(True)
        self.chk_loopback = QCheckBox("录制系统声音（极域共享音频）")
        self.cmb_loopback_device = QComboBox()
        self.cmb_loopback_device.setEditable(True)
        self.btn_refresh_devices = make_button("刷新设备", "ghost")
        self.spn_sample_rate = QComboBox()
        self.spn_sample_rate.setEditable(True)
        for rate in SAMPLE_RATES:
            self.spn_sample_rate.addItem(str(rate), rate)
        self.spn_channels = _spin(1, 2, 1, " 声道")
        self.spn_chunk = _spin(5, 120, 20, " 秒", 5)
        self.spn_vad = _dspin(-90.0, 0.0, -45.0, 1.0, 1)
        self.chk_keep_audio = QCheckBox("保留整段录音文件")

        audio_form.addRow("麦克风", self.chk_mic)
        audio_form.addRow("麦克风设备", self.cmb_mic_device)
        audio_form.addRow("系统回环", self.chk_loopback)
        audio_form.addRow("回环设备", self.cmb_loopback_device)
        device_row = QWidget()
        device_layout = QHBoxLayout(device_row)
        device_layout.setContentsMargins(0, 0, 0, 0)
        device_layout.addWidget(self.btn_refresh_devices)
        self.lbl_devices = _hint("未枚举设备。")
        device_layout.addWidget(self.lbl_devices, 1)
        audio_form.addRow("", device_row)
        audio_form.addRow("采样率", self.spn_sample_rate)
        audio_form.addRow("声道数", self.spn_channels)
        audio_form.addRow("切片长度", self.spn_chunk)
        audio_form.addRow("静音阈值", self.spn_vad)
        audio_form.addRow("", self.chk_keep_audio)
        outer.addLayout(audio_form)
        outer.addWidget(_hint(
            "切片长度决定语音识别的粒度，也是「老师说完 → AI 动手」延迟的主要来源："
            "切片越短越灵敏，但调用更频繁、更费钱。\n"
            "想让 AI 边听边跟着做：5~10 秒；只在下课后统一整理：20 秒以上。"
        ))

        screen_form = QFormLayout()
        screen_form.setLabelAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        screen_form.setSpacing(9)
        self.chk_screen = QCheckBox("采集演示画面（板书 / PPT）")
        self.cmb_screen_source = QComboBox()
        for key, label in SCREEN_SOURCES:
            self.cmb_screen_source.addItem(label, key)
        self.edt_keywords = QLineEdit()
        self.edt_keywords.setPlaceholderText("极域、TopDomain、电子教室……（用、或逗号分隔）")
        self.cmb_window = QComboBox()
        self.cmb_window.setEditable(True)
        self.btn_refresh_windows = make_button("刷新窗口列表", "ghost")
        window_row = QWidget()
        window_layout = QHBoxLayout(window_row)
        window_layout.setContentsMargins(0, 0, 0, 0)
        window_layout.addWidget(self.cmb_window, 1)
        window_layout.addWidget(self.btn_refresh_windows)
        self.spn_monitor = _spin(1, 16, 1)
        self.spn_region = [_spin(0, 100000, 0) for _ in range(4)]
        region_row = QWidget()
        region_layout = QHBoxLayout(region_row)
        region_layout.setContentsMargins(0, 0, 0, 0)
        region_layout.setSpacing(4)
        for label, box in zip(("X", "Y", "宽", "高"), self.spn_region):
            region_layout.addWidget(QLabel(label))
            region_layout.addWidget(box)
        self.spn_keyframe = _spin(2, 120, 8, " 秒")
        self.spn_change = _dspin(0.0, 1.0, 0.05, 0.01, 2)
        self.spn_max_width = _spin(320, 3840, 1280, " px", 160)
        self.chk_record_video = QCheckBox("同时录制演示视频（mp4）")
        self.spn_fps = _spin(1, 30, 2, " fps")
        self.spn_crf = _spin(0, 51, 28)
        self.spn_max_minutes = _spin(5, 1440, 240, " 分钟", 5)

        screen_form.addRow("演示画面", self.chk_screen)
        screen_form.addRow("来源方式", self.cmb_screen_source)
        screen_form.addRow("窗口关键字", self.edt_keywords)
        screen_form.addRow("指定窗口", window_row)
        screen_form.addRow("显示器序号", self.spn_monitor)
        screen_form.addRow("区域", region_row)
        screen_form.addRow("关键帧间隔", self.spn_keyframe)
        screen_form.addRow("变化阈值", self.spn_change)
        screen_form.addRow("关键帧最大宽度", self.spn_max_width)
        screen_form.addRow("", self.chk_record_video)
        screen_form.addRow("录像帧率", self.spn_fps)
        screen_form.addRow("录像质量(CRF)", self.spn_crf)
        screen_form.addRow("单次最长时长", self.spn_max_minutes)
        outer.addLayout(screen_form)
        outer.addWidget(_hint("提示：极域电子教室的演示窗口标题通常含“极域/电子教室/TopDomain”，关键字命中后自动跟随窗口。"))
        outer.addStretch(1)

        self.btn_refresh_devices.clicked.connect(self.refresh_devices)
        self.btn_refresh_windows.clicked.connect(self.refresh_devices)
        return page

    def refresh_devices(self) -> None:
        """在工作线程里枚举音频设备/窗口/显示器。"""
        probe = getattr(self._api, "probe_devices", None)
        if not callable(probe):
            self.lbl_devices.setText("当前引擎未提供设备枚举接口。")
            return
        self.lbl_devices.setText("正在枚举设备……")
        self.btn_refresh_devices.setEnabled(False)
        run_async(probe, self._on_devices, self._on_devices_failed)

    def _on_devices(self, payload: object) -> None:
        data = as_dict(payload)
        self._devices = {
            "audio": list(data.get("audio") or []),
            "windows": list(data.get("windows") or []),
            "monitors": list(data.get("monitors") or []),
        }
        audio = [as_dict(item) for item in self._devices["audio"]]
        windows = [as_dict(item) for item in self._devices["windows"]]
        current_mic = self.cmb_mic_device.currentText().strip()
        current_loop = self.cmb_loopback_device.currentText().strip()
        for combo, keep in ((self.cmb_mic_device, current_mic), (self.cmb_loopback_device, current_loop)):
            combo.clear()
            combo.addItem("（默认设备）", "")
            for item in audio:
                name = str(item.get("name", "")).strip()
                if not name:
                    continue
                tag = "回环" if item.get("is_loopback") else "输入"
                combo.addItem(f"{name}  [{tag}]", name)
            if keep:
                combo.setCurrentText(keep)
            else:
                combo.setCurrentIndex(0)
        self.cmb_window.clear()
        self.cmb_window.addItem("（按关键字自动选择）", "")
        for item in windows:
            title = str(item.get("title", "")).strip()
            if title:
                self.cmb_window.addItem(title, title)
        self.lbl_devices.setText(
            f"音频设备 {len(audio)} 个 · 窗口 {len(windows)} 个 · 显示器 {len(self._devices['monitors'])} 个"
        )
        self.btn_refresh_devices.setEnabled(True)

    def _on_devices_failed(self, error: str) -> None:
        self.lbl_devices.setText(f"枚举失败：{error}")
        self.btn_refresh_devices.setEnabled(True)

    # ------------------------------------------------------------ 页签 4：作业输出
    def _build_output_page(self) -> QWidget:
        page = QWidget()
        form = QFormLayout(page)
        form.setLabelAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        form.setSpacing(9)

        self.edt_output_dir = QLineEdit()
        self.btn_browse_dir = make_button("浏览…", "ghost")
        self.btn_open_dir = make_button("打开", "ghost")
        dir_row = QWidget()
        dir_layout = QHBoxLayout(dir_row)
        dir_layout.setContentsMargins(0, 0, 0, 0)
        dir_layout.addWidget(self.edt_output_dir, 1)
        dir_layout.addWidget(self.btn_browse_dir)
        dir_layout.addWidget(self.btn_open_dir)

        self.format_boxes: dict[str, QCheckBox] = {}
        fmt_row = QWidget()
        fmt_layout = QHBoxLayout(fmt_row)
        fmt_layout.setContentsMargins(0, 0, 0, 0)
        fmt_layout.setSpacing(14)
        for key, label in WRITE_FORMATS:
            box = QCheckBox(label)
            self.format_boxes[key] = box
            fmt_layout.addWidget(box)
        fmt_layout.addStretch(1)

        self.chk_subfolder = QCheckBox("每次听课建立独立子文件夹")
        self.edt_naming = QLineEdit()
        self.edt_naming.setPlaceholderText("{date}_{subject}_{session}")
        self.chk_workpaper = QCheckBox("附上解题过程")
        self.chk_transcript = QCheckBox("附上课堂记录（转写摘要）")
        self.chk_summary = QCheckBox("生成课堂小结")
        self.chk_auto_open = QCheckBox("导出完成后自动打开文件夹")

        form.addRow("输出目录", dir_row)
        form.addRow("写入格式", fmt_row)
        form.addRow("", self.chk_subfolder)
        form.addRow("命名模板", self.edt_naming)
        form.addRow("", self.chk_workpaper)
        form.addRow("", self.chk_transcript)
        form.addRow("", self.chk_summary)
        form.addRow("", self.chk_auto_open)
        form.addRow("", _hint("可用占位符：{date} 日期、{subject} 科目、{session} 会话标题、{title} 任务标题。"))
        return page

    def _on_browse_dir(self) -> None:
        current = self.edt_output_dir.text().strip() or str(default_output_dir())
        chosen = QFileDialog.getExistingDirectory(self, "选择作业输出目录", current)
        if chosen:
            self.edt_output_dir.setText(chosen)

    def _on_open_dir(self) -> None:
        path = self.edt_output_dir.text().strip()
        opener = getattr(self._api, "open_path", None)
        if path and callable(opener):
            try:
                opener(path)
            except Exception:
                pass

    # ------------------------------------------------------------ 页签 5：高级
    def _build_advanced_page(self) -> QWidget:
        page = QWidget()
        outer = QVBoxLayout(page)
        outer.setSpacing(10)

        form = QFormLayout()
        form.setLabelAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        form.setSpacing(9)
        self.chk_auto_start = QCheckBox("启动软件后自动开始听课")
        self.chk_understand = QCheckBox("边听课边实时转写与理解")
        self.chk_auto_solve = QCheckBox("识别到任务后自动作答")
        self.chk_solve_end = QCheckBox("下课时统一收尾（补笔记 + 完成任务）")
        self.chk_auto_stop = QCheckBox("判定下课后自动停止录制")
        self.chk_vision = QCheckBox("用视觉模型解析板书 / PPT")
        self.spn_confidence = _dspin(0.0, 1.0, 0.35, 0.05, 2)
        self.spn_silence = _spin(30, 3600, 300, " 秒", 30)
        self.spn_concurrent = _spin(1, 8, 2, " 路")
        self.spn_notes_every = _spin(1, 20, 3, " 片")
        self.chk_agent = QCheckBox("启用主 Agent：听到新信息后自动转给作业子 Agent")
        self.chk_revise = QCheckBox("老师补充要求后，自动让 AI 重做该作业")
        self.spn_max_revise = _spin(0, 10, 2, " 次")
        self.spn_tool_rounds = _spin(1, 8, 3, " 轮")

        form.addRow("", self.chk_auto_start)
        form.addRow("", self.chk_understand)
        form.addRow("", self.chk_auto_solve)
        form.addRow("", self.chk_solve_end)
        form.addRow("", self.chk_auto_stop)
        form.addRow("", self.chk_vision)
        form.addRow("自动作答置信度下限", self.spn_confidence)
        form.addRow("下课静音判定", self.spn_silence)
        form.addRow("并发 AI 请求数", self.spn_concurrent)
        form.addRow("每 N 片汇总笔记", self.spn_notes_every)
        form.addRow("", self.chk_agent)
        form.addRow("", self.chk_revise)
        form.addRow("每任务最多自动重做", self.spn_max_revise)
        form.addRow("单次督导最多工具轮数", self.spn_tool_rounds)
        outer.addLayout(form)
        outer.addWidget(_hint(
            "主 Agent 会盯着老师新讲的内容：一旦发现「这是在补充某个任务的要求」，"
            "就调用工具把新要求写进该任务的简报，并让作业子 Agent 用最新简报重做。\n"
            "「每任务最多自动重做」用来兜住 API 开销；设为 0 就只补充简报、不自动重做。"
        ))
        outer.addWidget(_hint(
            "「每 N 片汇总笔记」决定多久回头整理一次课堂内容并识别新任务：越小越灵敏。\n"
            "配合上面的切片长度使用——例如 切片 5 秒 + 每 1 片汇总，老师刚布置完作业就会开始作答；"
            "切片 20 秒 + 每 3 片汇总，最坏要等约 1 分钟。"
        ))

        outer.addWidget(QLabel("提示词覆盖（JSON，键：" + " / ".join(PROMPT_KEYS) + "）"))
        self.edt_overrides = QPlainTextEdit()
        self.edt_overrides.setPlaceholderText('{"solve": "自定义的解题提示词……"}')
        self.edt_overrides.setFixedHeight(140)
        outer.addWidget(self.edt_overrides)
        self.lbl_overrides = _hint("留空表示使用内置提示词。")
        outer.addWidget(self.lbl_overrides)
        outer.addStretch(1)
        return page

    def _parse_overrides(self) -> dict[str, str] | None:
        raw = self.edt_overrides.toPlainText().strip()
        if not raw:
            return {}
        try:
            data = json.loads(raw)
        except Exception as exc:
            self.lbl_overrides.setText(f"✗ JSON 解析失败：{exc}（已保留原设置）")
            return None
        if not isinstance(data, dict):
            self.lbl_overrides.setText("✗ 提示词覆盖必须是 JSON 对象（已保留原设置）")
            return None
        clean = {str(k): str(v) for k, v in data.items() if str(v).strip()}
        self.lbl_overrides.setText(f"✓ 已解析 {len(clean)} 条覆盖。")
        return clean

    # ------------------------------------------------------------ 存取
    def set_config(self, cfg: AppConfig) -> None:
        """把 cfg 的值灌进界面。"""
        self._cfg = copy.deepcopy(cfg) if cfg is not None else AppConfig()
        model = self._cfg.model
        pos = self.cmb_preset.findData(model.provider)
        self.cmb_preset.setCurrentIndex(pos if pos >= 0 else self.cmb_preset.count() - 1)
        self.edt_base_url.setText(model.base_url)
        self.edt_api_key.setText(model.api_key)
        self.edt_chat.setText(model.chat_model)
        self.edt_vision.setText(model.vision_model)
        self.edt_asr.setText(model.asr_model)
        kind_pos = self.cmb_asr_kind.findData(model.asr_kind)
        self.cmb_asr_kind.setCurrentIndex(kind_pos if kind_pos >= 0 else 0)
        self.edt_asr_base_url.setText(model.asr_base_url)
        self.spn_temperature.setValue(float(model.temperature))
        self.spn_max_tokens.setValue(int(model.max_tokens))
        self.spn_timeout.setValue(int(model.timeout_s))
        self.spn_retries.setValue(int(model.retries))

        person = self._cfg.person
        self.edt_name.setText(person.name)
        self.edt_student_id.setText(person.student_id)
        self.edt_class.setText(person.class_name)
        self.edt_grade.setText(person.grade)
        self.edt_school.setText(person.school)
        self.edt_teacher.setText(person.teacher_name)
        self.edt_subject.setText(person.subject)
        idx = self.cmb_style.findText(person.answer_style)
        self.cmb_style.setCurrentIndex(idx if idx >= 0 else 0)
        self.edt_language.setText(person.language)
        self.chk_sign.setChecked(bool(person.sign_artifacts))
        self.edt_extra.setPlainText(person.extra_notes)

        cap = self._cfg.capture
        self.chk_mic.setChecked(bool(cap.mic_enabled))
        self.cmb_mic_device.setCurrentText(cap.mic_device or "（默认设备）")
        self.chk_loopback.setChecked(bool(cap.loopback_enabled))
        self.cmb_loopback_device.setCurrentText(cap.loopback_device or "（默认设备）")
        rate_pos = self.spn_sample_rate.findData(int(cap.sample_rate))
        if rate_pos < 0:
            self.spn_sample_rate.addItem(str(cap.sample_rate), int(cap.sample_rate))
            rate_pos = self.spn_sample_rate.findData(int(cap.sample_rate))
        self.spn_sample_rate.setCurrentIndex(max(0, rate_pos))
        self.spn_channels.setValue(int(cap.channels))
        self.spn_chunk.setValue(int(cap.chunk_seconds))
        self.spn_vad.setValue(float(cap.vad_silence_db))
        self.chk_keep_audio.setChecked(bool(cap.keep_full_audio))
        self.chk_screen.setChecked(bool(cap.screen_enabled))
        src_pos = self.cmb_screen_source.findData(cap.screen_source)
        self.cmb_screen_source.setCurrentIndex(src_pos if src_pos >= 0 else 0)
        self.edt_keywords.setText("、".join(cap.window_title_keywords or []))
        if self.cmb_window.count() == 0:
            self.cmb_window.addItem("（按关键字自动选择）", "")
        self.spn_monitor.setValue(int(cap.monitor_index))
        region = list(cap.region or [0, 0, 0, 0])[:4]
        region += [0] * (4 - len(region))
        for box, value in zip(self.spn_region, region):
            box.setValue(int(value))
        self.spn_keyframe.setValue(int(cap.keyframe_interval_s))
        self.spn_change.setValue(float(cap.keyframe_change_threshold))
        self.spn_max_width.setValue(int(cap.max_keyframe_width))
        self.chk_record_video.setChecked(bool(cap.record_video))
        self.spn_fps.setValue(int(cap.video_fps))
        self.spn_crf.setValue(int(cap.video_crf))
        self.spn_max_minutes.setValue(int(cap.max_session_minutes))

        out = self._cfg.output
        self.edt_output_dir.setText(out.output_dir or str(default_output_dir()))
        formats = set(out.write_formats or [])
        for key, box in self.format_boxes.items():
            box.setChecked(key in formats)
        self.chk_subfolder.setChecked(bool(out.create_session_subfolder))
        self.edt_naming.setText(out.naming_template)
        self.chk_workpaper.setChecked(bool(out.include_workpaper))
        self.chk_transcript.setChecked(bool(out.include_transcript))
        self.chk_summary.setChecked(bool(out.write_summary))
        self.chk_auto_open.setChecked(bool(out.auto_open_folder))

        beh = self._cfg.behavior
        self.chk_auto_start.setChecked(bool(beh.auto_start_on_launch))
        self.chk_understand.setChecked(bool(beh.auto_understand))
        self.chk_auto_solve.setChecked(bool(beh.auto_solve_tasks))
        self.chk_solve_end.setChecked(bool(beh.solve_on_session_end))
        self.chk_auto_stop.setChecked(bool(beh.auto_stop_on_class_end))
        self.chk_vision.setChecked(bool(beh.vision_analysis))
        self.spn_confidence.setValue(float(beh.min_task_confidence))
        self.spn_silence.setValue(int(beh.end_of_class_silence_s))
        self.spn_concurrent.setValue(int(beh.max_concurrent_ai))
        self.spn_notes_every.setValue(int(beh.notes_every_n_chunks))
        self.chk_agent.setChecked(bool(getattr(beh, "agent_tools_enabled", True)))
        self.chk_revise.setChecked(bool(getattr(beh, "auto_revise_solved", True)))
        self.spn_max_revise.setValue(int(getattr(beh, "max_auto_revisions", 2)))
        self.spn_tool_rounds.setValue(int(getattr(beh, "agent_max_tool_rounds", 3)))

        overrides = self._cfg.prompt_overrides or {}
        self.edt_overrides.setPlainText(json.dumps(overrides, ensure_ascii=False, indent=2) if overrides else "")
        self.lbl_overrides.setText("留空表示使用内置提示词。")
        self._warnings = []

    def apply_to(self, cfg: AppConfig) -> AppConfig:
        """把界面内容写进传入的 cfg（原地修改）并返回它。"""
        self._warnings = []
        model = cfg.model
        provider = self.cmb_preset.currentData()
        if provider and provider != "custom_manual":
            model.provider = str(provider)
        model.base_url = self.edt_base_url.text().strip()
        model.api_key = self.edt_api_key.text().strip()
        model.chat_model = self.edt_chat.text().strip()
        model.vision_model = self.edt_vision.text().strip()
        model.asr_model = self.edt_asr.text().strip()
        kind = self.cmb_asr_kind.currentData()
        if kind:
            model.asr_kind = str(kind)
        model.asr_base_url = self.edt_asr_base_url.text().strip()
        model.temperature = float(self.spn_temperature.value())
        model.max_tokens = int(self.spn_max_tokens.value())
        model.timeout_s = int(self.spn_timeout.value())
        model.retries = int(self.spn_retries.value())

        person = cfg.person
        person.name = self.edt_name.text().strip()
        person.student_id = self.edt_student_id.text().strip()
        person.class_name = self.edt_class.text().strip()
        person.grade = self.edt_grade.text().strip()
        person.school = self.edt_school.text().strip()
        person.teacher_name = self.edt_teacher.text().strip()
        person.subject = self.edt_subject.text().strip()
        person.answer_style = self.cmb_style.currentText()
        person.language = self.edt_language.text().strip() or "zh-CN"
        person.sign_artifacts = self.chk_sign.isChecked()
        person.extra_notes = self.edt_extra.toPlainText().strip()

        cap = cfg.capture
        cap.mic_enabled = self.chk_mic.isChecked()
        cap.mic_device = self._device_value(self.cmb_mic_device)
        cap.loopback_enabled = self.chk_loopback.isChecked()
        cap.loopback_device = self._device_value(self.cmb_loopback_device)
        rate = self.spn_sample_rate.currentData()
        try:
            cap.sample_rate = int(rate if rate is not None else self.spn_sample_rate.currentText())
        except (TypeError, ValueError):
            self._warnings.append("采样率无法识别，已保留原值。")
        cap.channels = int(self.spn_channels.value())
        cap.chunk_seconds = int(self.spn_chunk.value())
        cap.vad_silence_db = float(self.spn_vad.value())
        cap.keep_full_audio = self.chk_keep_audio.isChecked()
        cap.screen_enabled = self.chk_screen.isChecked()
        source = self.cmb_screen_source.currentData()
        if source:
            cap.screen_source = str(source)
        keywords = [part.strip() for part in self.edt_keywords.text().replace(",", "、").split("、")]
        cap.window_title_keywords = [k for k in keywords if k]
        window = self._device_value(self.cmb_window)
        if window:
            existing = [k for k in cap.window_title_keywords]
            if window not in existing:
                cap.window_title_keywords = [window] + existing
        cap.monitor_index = int(self.spn_monitor.value())
        cap.region = [int(box.value()) for box in self.spn_region]
        cap.keyframe_interval_s = int(self.spn_keyframe.value())
        cap.keyframe_change_threshold = float(self.spn_change.value())
        cap.max_keyframe_width = int(self.spn_max_width.value())
        cap.record_video = self.chk_record_video.isChecked()
        cap.video_fps = int(self.spn_fps.value())
        cap.video_crf = int(self.spn_crf.value())
        cap.max_session_minutes = int(self.spn_max_minutes.value())

        out = cfg.output
        out.output_dir = self.edt_output_dir.text().strip()
        picked = [key for key, box in self.format_boxes.items() if box.isChecked()]
        out.write_formats = picked or ["md"]
        out.create_session_subfolder = self.chk_subfolder.isChecked()
        out.naming_template = self.edt_naming.text().strip() or "{date}_{subject}_{session}"
        out.include_workpaper = self.chk_workpaper.isChecked()
        out.include_transcript = self.chk_transcript.isChecked()
        out.write_summary = self.chk_summary.isChecked()
        out.auto_open_folder = self.chk_auto_open.isChecked()

        beh = cfg.behavior
        beh.auto_start_on_launch = self.chk_auto_start.isChecked()
        beh.auto_understand = self.chk_understand.isChecked()
        beh.auto_solve_tasks = self.chk_auto_solve.isChecked()
        beh.solve_on_session_end = self.chk_solve_end.isChecked()
        beh.auto_stop_on_class_end = self.chk_auto_stop.isChecked()
        beh.vision_analysis = self.chk_vision.isChecked()
        beh.min_task_confidence = float(self.spn_confidence.value())
        beh.end_of_class_silence_s = int(self.spn_silence.value())
        beh.max_concurrent_ai = int(self.spn_concurrent.value())
        beh.notes_every_n_chunks = int(self.spn_notes_every.value())
        beh.agent_tools_enabled = self.chk_agent.isChecked()
        beh.auto_revise_solved = self.chk_revise.isChecked()
        beh.max_auto_revisions = int(self.spn_max_revise.value())
        beh.agent_max_tool_rounds = int(self.spn_tool_rounds.value())

        parsed = self._parse_overrides()
        if parsed is not None:
            cfg.prompt_overrides = parsed
        elif not cfg.prompt_overrides:
            cfg.prompt_overrides = {}
        return cfg

    @staticmethod
    def _device_value(combo: QComboBox) -> str:
        """设备下拉框取值：编辑框内容优先（用户可手填设备名）。"""
        text = combo.currentText().strip()
        if not text or text.startswith("（"):
            return ""
        data = combo.currentData()
        return str(data) if data else text

    def result_config(self) -> AppConfig:
        """基于最近一次 load 的配置生成新配置（不污染调用方对象）。"""
        return self.apply_to(copy.deepcopy(self._cfg))

    def warnings(self) -> list[str]:
        """最近一次 apply_to 产生的非致命问题。"""
        return list(self._warnings)

    def config_problems(self) -> list[str]:
        """当前界面内容对应的配置问题列表（用于保存前提示）。"""
        return self.result_config().validate()


class SettingsDialog(QDialog):
    """设置对话框（五个页签 + 保存/应用/取消）。"""

    def __init__(self, cfg: AppConfig | None = None, api=None, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._api = api
        self._origin: AppConfig = copy.deepcopy(cfg) if cfg is not None else AppConfig()
        self.setWindowTitle("AI学生 · 设置")
        self.setMinimumSize(880, 660)

        self.form = SettingsForm(self._origin, api=api, parent=self)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 14, 14, 12)
        layout.setSpacing(10)
        header = QLabel("设置")
        header.setObjectName("pageTitle")
        layout.addWidget(header)
        layout.addWidget(self.form, 1)

        self.btn_defaults = make_button("恢复默认值", "ghost")
        self.btn_defaults.clicked.connect(self._on_defaults)
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok
            | QDialogButtonBox.StandardButton.Cancel
            | QDialogButtonBox.StandardButton.Apply
        )
        buttons.button(QDialogButtonBox.StandardButton.Ok).setText("保存")
        buttons.button(QDialogButtonBox.StandardButton.Cancel).setText("取消")
        buttons.button(QDialogButtonBox.StandardButton.Apply).setText("应用")
        buttons.accepted.connect(self._on_accept)
        buttons.rejected.connect(self.reject)
        buttons.button(QDialogButtonBox.StandardButton.Apply).clicked.connect(self._on_apply)
        row = QWidget()
        row_layout = QHBoxLayout(row)
        row_layout.setContentsMargins(0, 0, 0, 0)
        row_layout.addWidget(self.btn_defaults)
        row_layout.addStretch(1)
        row_layout.addWidget(buttons)
        layout.addWidget(row)
        self._buttons = buttons
        self._applied: AppConfig | None = None

    # -- 数据 --------------------------------------------------------
    def result_config(self) -> AppConfig:
        """返回界面上的配置（深拷贝，不修改调用方对象）。"""
        return self.form.result_config()

    def apply_to(self, cfg: AppConfig) -> AppConfig:
        """把界面内容写进指定 cfg 并返回（原地修改）。"""
        return self.form.apply_to(cfg)

    def applied_config(self) -> AppConfig | None:
        """最近一次“应用/保存”提交成功的配置。"""
        return self._applied

    # -- 槽 ----------------------------------------------------------
    def _on_defaults(self) -> None:
        self.form.set_config(AppConfig())

    def _commit(self) -> bool:
        cfg = self.form.result_config()
        problems = cfg.validate()
        if problems:
            detail = "\n".join(f"· {item}" for item in problems)
            answer = QMessageBox.question(
                self,
                "配置还不完整",
                f"以下项目尚未配置完整，仍要保存吗？\n\n{detail}",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if answer != QMessageBox.StandardButton.Yes:
                return False
        updater = getattr(self._api, "update_config", None)
        if callable(updater):
            try:
                updater(cfg)
            except Exception as exc:
                QMessageBox.warning(self, "保存失败", f"配置未能保存：{exc}")
                return False
        self._applied = cfg
        self.form.set_config(cfg)
        return True

    def _on_apply(self) -> None:
        self._commit()

    def _on_accept(self) -> None:
        if self._commit():
            self.accept()
