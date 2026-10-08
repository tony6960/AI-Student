"""AI学生 PySide6 图形界面包。

对外只暴露四件套：主窗口、设置对话框、首次运行向导、主题。
界面层只依赖 ai_student.core.api.ClassAgentAPI 与 ai_student.config 的数据类，
不 import core.controller / capture / ai，保证可以脱离引擎单独实例化（测试用 Mock）。
"""
from __future__ import annotations

from .dialogs import FirstRunWizard, TaskInputDialog, TextPromptDialog
from .main_window import MainWindow
from .settings_dialog import SettingsDialog
from .theme import apply_theme
from .tray import TrayController

__all__ = [
    "MainWindow",
    "SettingsDialog",
    "FirstRunWizard",
    "TextPromptDialog",
    "TaskInputDialog",
    "TrayController",
    "apply_theme",
]
