"""对话框：首次运行向导 / 文本输入 / 手动任务输入。

说明：
- 向导三步：模型 API -> 个人信息 -> 输出目录，产出 result_config()。
- “测试连接”用标准库 urllib 直接探测 OpenAI 兼容端点，放在工作线程执行，
  不阻塞 UI 线程，也不引入新的第三方依赖。
"""
from __future__ import annotations

import copy
import json
import urllib.error
import urllib.request

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPlainTextEdit,
    QVBoxLayout,
    QWidget,
    QWizard,
    QWizardPage,
)

from ..config import AppConfig, MODEL_PRESETS
from ..config.paths import default_output_dir
from .widgets import make_button, run_async

ASR_KINDS: list[tuple[str, str]] = [
    ("openai_transcription", "OpenAI 转写接口（/audio/transcriptions）"),
    ("openai_audio", "对话接口内嵌音频（通义千问 / GLM）"),
    ("none", "不启用语音识别（不转写）"),
]

ANSWER_STYLES: list[str] = ["标准作答", "简洁", "详细讲解", "按考试要求"]

WRITE_FORMATS: list[tuple[str, str]] = [
    ("md", "Markdown（.md）"),
    ("txt", "纯文本（.txt）"),
    ("docx", "Word 文档（.docx）"),
]


def preset_items() -> list[tuple[str, str]]:
    """模型服务预设下拉项 [(key, label), ...]（与 config.MODEL_PRESETS 同序）。"""
    return [(key, str(meta.get("label", key))) for key, meta in MODEL_PRESETS.items()]


def check_endpoint(base_url: str, api_key: str = "", timeout: float = 10.0) -> tuple[bool, str]:
    """探测 OpenAI 兼容端点可用性（GET {base_url}/models）。

    返回 (是否可用, 人类可读说明)。任何异常都转成说明文本，不向外抛。
    """
    url = (base_url or "").strip().rstrip("/")
    if not url:
        return False, "未填写服务地址"
    request = urllib.request.Request(f"{url}/models", method="GET")
    if api_key:
        request.add_header("Authorization", f"Bearer {api_key}")
    request.add_header("User-Agent", "AIStudent/1.0")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            status = int(getattr(response, "status", 200) or 200)
            body = response.read(8192).decode("utf-8", "ignore")
        extra = ""
        try:
            data = json.loads(body)
            if isinstance(data, dict) and isinstance(data.get("data"), list):
                extra = f"，可用模型 {len(data['data'])} 个"
        except Exception:
            extra = ""
        return True, f"连接成功（HTTP {status}{extra}）"
    except urllib.error.HTTPError as exc:
        if exc.code in (401, 403):
            return False, f"地址可达但鉴权失败（HTTP {exc.code}），请检查 API KEY"
        if exc.code == 404:
            return True, "地址可达（该服务未提供 /models 列表，HTTP 404）"
        return False, f"服务返回 HTTP {exc.code}"
    except Exception as exc:  # 网络/证书/DNS 等
        return False, f"无法连接：{type(exc).__name__}: {exc}"


def _hint(text: str) -> QLabel:
    label = QLabel(text)
    label.setObjectName("fieldHint")
    label.setWordWrap(True)
    return label


# ---------------------------------------------------------------- 向导页


class _ModelPage(QWizardPage):
    """第 1 步：模型 API。"""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setTitle("模型 API")
        self.setSubTitle("AI学生需要一个 OpenAI 兼容的多模态服务（对话 + 视觉 + 语音识别）。")

        self.cmb_preset = QComboBox()
        for key, label in preset_items():
            self.cmb_preset.addItem(label, key)
        self.cmb_preset.addItem("自定义（手动填写下面各项）", "custom_manual")

        self.edt_base_url = QLineEdit()
        self.edt_base_url.setPlaceholderText("https://dashscope.aliyuncs.com/compatible-mode/v1")
        self.edt_api_key = QLineEdit()
        self.edt_api_key.setEchoMode(QLineEdit.EchoMode.Password)
        self.edt_api_key.setPlaceholderText("sk-……（本机加密保存，不会明文落盘）")
        self.chk_show_key = QCheckBox("显示密钥")
        key_row = QWidget()
        key_layout = QHBoxLayout(key_row)
        key_layout.setContentsMargins(0, 0, 0, 0)
        key_layout.setSpacing(8)
        key_layout.addWidget(self.edt_api_key, 1)
        key_layout.addWidget(self.chk_show_key)

        self.edt_chat = QLineEdit()
        self.edt_vision = QLineEdit()
        self.edt_vision.setPlaceholderText("留空则与对话模型相同")
        self.edt_asr = QLineEdit()
        self.cmb_asr_kind = QComboBox()
        for key, label in ASR_KINDS:
            self.cmb_asr_kind.addItem(label, key)

        form = QFormLayout()
        form.setLabelAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        form.setSpacing(8)
        form.addRow("服务预设", self.cmb_preset)
        form.addRow("服务地址", self.edt_base_url)
        form.addRow("API KEY", key_row)
        form.addRow("对话模型", self.edt_chat)
        form.addRow("视觉模型", self.edt_vision)
        form.addRow("语音识别模型", self.edt_asr)
        form.addRow("识别方式", self.cmb_asr_kind)

        self.btn_test = make_button("测试连接", "ghost", "在工作线程里请求服务地址的 /models 接口")
        self.lbl_test = _hint("尚未测试。")
        test_row = QWidget()
        test_layout = QHBoxLayout(test_row)
        test_layout.setContentsMargins(0, 0, 0, 0)
        test_layout.setSpacing(8)
        test_layout.addWidget(self.btn_test)
        test_layout.addWidget(self.lbl_test, 1)
        form.addRow("", test_row)

        layout = QVBoxLayout(self)
        layout.addLayout(form)
        layout.addWidget(
            _hint("提示：阿里云百炼（qwen）同时提供视觉理解与 qwen3-asr-flash 语音识别，中文课堂效果最好。")
        )
        layout.addStretch(1)

        self.cmb_preset.currentIndexChanged.connect(self._on_preset_changed)
        self.chk_show_key.toggled.connect(self._on_toggle_key)
        self.btn_test.clicked.connect(self._on_test_clicked)

    # -- 交互 --------------------------------------------------------
    def _on_toggle_key(self, checked: bool) -> None:
        self.edt_api_key.setEchoMode(QLineEdit.EchoMode.Normal if checked else QLineEdit.EchoMode.Password)

    def _on_preset_changed(self, index: int) -> None:
        key = self.cmb_preset.itemData(index)
        if key in (None, "custom_manual"):
            return
        meta = MODEL_PRESETS.get(str(key), {})
        self.edt_base_url.setText(str(meta.get("base_url", "")))
        self.edt_chat.setText(str(meta.get("chat_model", "")))
        self.edt_vision.setText(str(meta.get("vision_model", "")))
        self.edt_asr.setText(str(meta.get("asr_model", "")))
        kind = str(meta.get("asr_kind", "openai_transcription"))
        pos = self.cmb_asr_kind.findData(kind)
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

    # -- 数据 --------------------------------------------------------
    def load(self, cfg: AppConfig) -> None:
        endpoint = cfg.model
        pos = self.cmb_preset.findData(endpoint.provider)
        self.cmb_preset.setCurrentIndex(pos if pos >= 0 else self.cmb_preset.count() - 1)
        self.edt_base_url.setText(endpoint.base_url)
        self.edt_api_key.setText(endpoint.api_key)
        self.edt_chat.setText(endpoint.chat_model)
        self.edt_vision.setText(endpoint.vision_model)
        self.edt_asr.setText(endpoint.asr_model)
        kind_pos = self.cmb_asr_kind.findData(endpoint.asr_kind)
        self.cmb_asr_kind.setCurrentIndex(kind_pos if kind_pos >= 0 else 0)

    def collect(self, cfg: AppConfig) -> None:
        endpoint = cfg.model
        provider = self.cmb_preset.currentData()
        if provider and provider != "custom_manual":
            endpoint.provider = str(provider)
        endpoint.base_url = self.edt_base_url.text().strip()
        endpoint.api_key = self.edt_api_key.text().strip()
        endpoint.chat_model = self.edt_chat.text().strip()
        endpoint.vision_model = self.edt_vision.text().strip()
        endpoint.asr_model = self.edt_asr.text().strip()
        asr_kind = self.cmb_asr_kind.currentData()
        if asr_kind:
            endpoint.asr_kind = str(asr_kind)


class _PersonPage(QWizardPage):
    """第 2 步：个人信息。"""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setTitle("个人信息")
        self.setSubTitle("作业文件会用这些信息署名，只有姓名是必填项。")

        self.edt_name = QLineEdit()
        self.edt_student_id = QLineEdit()
        self.edt_class = QLineEdit()
        self.edt_grade = QLineEdit()
        self.edt_school = QLineEdit()
        self.edt_teacher = QLineEdit()
        self.edt_subject = QLineEdit()
        self.edt_subject.setPlaceholderText("例如：数学（留空则按课堂内容自动判断）")
        self.cmb_style = QComboBox()
        self.cmb_style.addItems(ANSWER_STYLES)
        self.chk_sign = QCheckBox("在作业中署名")
        self.chk_sign.setChecked(True)
        self.edt_extra = QPlainTextEdit()
        self.edt_extra.setPlaceholderText("补充说明，例如：字迹要求、老师偏好……（可留空）")
        self.edt_extra.setFixedHeight(72)

        form = QFormLayout()
        form.setLabelAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        form.setSpacing(8)
        form.addRow("姓名 *", self.edt_name)
        form.addRow("学号", self.edt_student_id)
        form.addRow("班级", self.edt_class)
        form.addRow("年级", self.edt_grade)
        form.addRow("学校", self.edt_school)
        form.addRow("任课教师", self.edt_teacher)
        form.addRow("默认科目", self.edt_subject)
        form.addRow("作答风格", self.cmb_style)
        form.addRow("", self.chk_sign)
        form.addRow("补充说明", self.edt_extra)

        layout = QVBoxLayout(self)
        layout.addLayout(form)
        layout.addStretch(1)

    def load(self, cfg: AppConfig) -> None:
        person = cfg.person
        self.edt_name.setText(person.name)
        self.edt_student_id.setText(person.student_id)
        self.edt_class.setText(person.class_name)
        self.edt_grade.setText(person.grade)
        self.edt_school.setText(person.school)
        self.edt_teacher.setText(person.teacher_name)
        self.edt_subject.setText(person.subject)
        idx = self.cmb_style.findText(person.answer_style)
        self.cmb_style.setCurrentIndex(idx if idx >= 0 else 0)
        self.chk_sign.setChecked(bool(person.sign_artifacts))
        self.edt_extra.setPlainText(person.extra_notes)

    def collect(self, cfg: AppConfig) -> None:
        person = cfg.person
        person.name = self.edt_name.text().strip()
        person.student_id = self.edt_student_id.text().strip()
        person.class_name = self.edt_class.text().strip()
        person.grade = self.edt_grade.text().strip()
        person.school = self.edt_school.text().strip()
        person.teacher_name = self.edt_teacher.text().strip()
        person.subject = self.edt_subject.text().strip()
        person.answer_style = self.cmb_style.currentText()
        person.sign_artifacts = self.chk_sign.isChecked()
        person.extra_notes = self.edt_extra.toPlainText().strip()


class _OutputPage(QWizardPage):
    """第 3 步：输出目录与格式。"""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setTitle("作业输出")
        self.setSubTitle("AI 完成的作业会写到这里，可随时在设置里修改。")

        self.edt_dir = QLineEdit()
        self.btn_browse = make_button("浏览…", "ghost")
        dir_row = QWidget()
        dir_layout = QHBoxLayout(dir_row)
        dir_layout.setContentsMargins(0, 0, 0, 0)
        dir_layout.setSpacing(8)
        dir_layout.addWidget(self.edt_dir, 1)
        dir_layout.addWidget(self.btn_browse)

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
        self.chk_subfolder.setChecked(True)
        self.chk_auto_open = QCheckBox("导出后自动打开文件夹")
        self.chk_auto_open.setChecked(True)

        form = QFormLayout()
        form.setLabelAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        form.setSpacing(8)
        form.addRow("输出目录", dir_row)
        form.addRow("写入格式", fmt_row)
        form.addRow("", self.chk_subfolder)
        form.addRow("", self.chk_auto_open)

        layout = QVBoxLayout(self)
        layout.addLayout(form)
        layout.addWidget(_hint("建议勾选 Markdown + Word：Markdown 便于阅读，Word 便于直接交作业。"))
        layout.addStretch(1)

        self.btn_browse.clicked.connect(self._on_browse)

    def _on_browse(self) -> None:
        current = self.edt_dir.text().strip() or str(default_output_dir())
        chosen = QFileDialog.getExistingDirectory(self, "选择作业输出目录", current)
        if chosen:
            self.edt_dir.setText(chosen)

    def load(self, cfg: AppConfig) -> None:
        self.edt_dir.setText(cfg.output.output_dir or str(default_output_dir()))
        formats = set(cfg.output.write_formats or [])
        for key, box in self.format_boxes.items():
            box.setChecked(key in formats)
        self.chk_subfolder.setChecked(bool(cfg.output.create_session_subfolder))
        self.chk_auto_open.setChecked(bool(cfg.output.auto_open_folder))

    def collect(self, cfg: AppConfig) -> None:
        cfg.output.output_dir = self.edt_dir.text().strip()
        picked = [key for key, box in self.format_boxes.items() if box.isChecked()]
        cfg.output.write_formats = picked or ["md"]
        cfg.output.create_session_subfolder = self.chk_subfolder.isChecked()
        cfg.output.auto_open_folder = self.chk_auto_open.isChecked()


class FirstRunWizard(QWizard):
    """首次运行向导：3 步收集最小可用配置。"""

    def __init__(self, cfg: AppConfig | None = None, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._cfg: AppConfig = copy.deepcopy(cfg) if cfg is not None else AppConfig()
        self.setWindowTitle("AI学生 · 首次运行向导")
        self.setWizardStyle(QWizard.WizardStyle.ModernStyle)
        self.setOption(QWizard.WizardOption.NoBackButtonOnStartPage, True)
        self.setOption(QWizard.WizardOption.HaveHelpButton, False)
        self.setMinimumSize(660, 500)

        self.page_model = _ModelPage()
        self.page_person = _PersonPage()
        self.page_output = _OutputPage()
        self.addPage(self.page_model)
        self.addPage(self.page_person)
        self.addPage(self.page_output)

        self._load()
        self._apply_button_text()

    # -- 生命周期 ----------------------------------------------------
    def showEvent(self, event) -> None:  # noqa: N802 - Qt 命名
        self._apply_button_text()
        super().showEvent(event)

    def _apply_button_text(self) -> None:
        try:
            self.setButtonText(QWizard.WizardButton.BackButton, "上一步")
            self.setButtonText(QWizard.WizardButton.NextButton, "下一步")
            self.setButtonText(QWizard.WizardButton.FinishButton, "完成")
            self.setButtonText(QWizard.WizardButton.CancelButton, "取消")
        except Exception:  # pragma: no cover - 某些平台样式按钮尚未创建
            pass

    # -- 数据 --------------------------------------------------------
    def _load(self) -> None:
        self.page_model.load(self._cfg)
        self.page_person.load(self._cfg)
        self.page_output.load(self._cfg)

    def collect(self) -> AppConfig:
        """把界面上的内容写回内部配置并返回。"""
        self.page_model.collect(self._cfg)
        self.page_person.collect(self._cfg)
        self.page_output.collect(self._cfg)
        return self._cfg

    def result_config(self) -> AppConfig:
        """返回向导收集到的配置（AppConfig 深拷贝）。"""
        return copy.deepcopy(self.collect())

    def accept(self) -> None:  # noqa: D102 - Qt 命名
        self.collect()
        super().accept()


# ---------------------------------------------------------------- 普通输入框


class TextPromptDialog(QDialog):
    """多行文本输入框（任务重做时的反馈输入）。"""

    def __init__(
        self,
        title: str = "请输入",
        label: str = "",
        default: str = "",
        placeholder: str = "",
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setMinimumWidth(460)

        layout = QVBoxLayout(self)
        layout.setSpacing(10)
        if label:
            text = QLabel(label)
            text.setWordWrap(True)
            layout.addWidget(text)
        self.editor = QPlainTextEdit()
        self.editor.setPlainText(default)
        self.editor.setPlaceholderText(placeholder)
        self.editor.setFixedHeight(120)
        layout.addWidget(self.editor)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.button(QDialogButtonBox.StandardButton.Ok).setText("确定")
        buttons.button(QDialogButtonBox.StandardButton.Cancel).setText("取消")
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def text(self) -> str:
        return self.editor.toPlainText().strip()

    @staticmethod
    def ask(
        parent: QWidget | None,
        title: str,
        label: str,
        default: str = "",
        placeholder: str = "",
    ) -> str | None:
        """弹出对话框；确定返回文本（可能为空串），取消返回 None。"""
        dialog = TextPromptDialog(title, label, default, placeholder, parent)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            return dialog.text()
        return None


class TaskInputDialog(QDialog):
    """手动添加任务：标题 + 详细要求。"""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("手动添加任务")
        self.setMinimumWidth(480)

        self.edt_title = QLineEdit()
        self.edt_title.setPlaceholderText("任务标题，例如：完成练习册 P32 第 1-5 题")
        self.edt_detail = QPlainTextEdit()
        self.edt_detail.setPlaceholderText("补充说明（可选）：题目原文、要求、截止时间……")
        self.edt_detail.setFixedHeight(110)

        form = QFormLayout()
        form.setLabelAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        form.addRow("任务标题 *", self.edt_title)
        form.addRow("详细要求", self.edt_detail)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.button(QDialogButtonBox.StandardButton.Ok).setText("添加")
        buttons.button(QDialogButtonBox.StandardButton.Cancel).setText("取消")
        buttons.accepted.connect(self._on_accept)
        buttons.rejected.connect(self.reject)

        layout = QVBoxLayout(self)
        layout.addLayout(form)
        layout.addWidget(buttons)

    def _on_accept(self) -> None:
        if not self.edt_title.text().strip():
            self.edt_title.setPlaceholderText("请先填写任务标题")
            self.edt_title.setFocus()
            return
        self.accept()

    def values(self) -> tuple[str, str]:
        """返回 (标题, 详细要求)。"""
        return self.edt_title.text().strip(), self.edt_detail.toPlainText().strip()

    @staticmethod
    def get_task(parent: QWidget | None) -> tuple[str, str] | None:
        """弹出对话框；确定返回 (标题, 详情)，取消返回 None。"""
        dialog = TaskInputDialog(parent)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            return dialog.values()
        return None
