"""应用数据目录与路径解析。

默认根目录: %APPDATA%\\AIStudent
可用环境变量 AI_STUDENT_HOME 覆盖（测试与便携模式使用）。
"""
from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path

APP_FOLDER = "AIStudent"


def app_root() -> Path:
    """应用数据根目录。"""
    override = os.environ.get("AI_STUDENT_HOME")
    if override:
        return Path(override).expanduser().resolve()
    base = os.environ.get("APPDATA") or os.environ.get("LOCALAPPDATA")
    if not base:
        base = str(Path.home() / "AppData" / "Roaming")
    return Path(base) / APP_FOLDER


def config_path() -> Path:
    return app_root() / "config.json"


def sessions_root() -> Path:
    return app_root() / "sessions"


def logs_root() -> Path:
    return app_root() / "logs"


def cache_root() -> Path:
    return app_root() / "cache"


def default_output_dir() -> Path:
    """默认作业输出目录（用户可在设置中修改）。"""
    docs = Path.home() / "Documents"
    base = docs if docs.exists() else Path.home()
    return base / "AI学生作业"


def ensure_dirs() -> None:
    for p in (app_root(), sessions_root(), logs_root(), cache_root()):
        p.mkdir(parents=True, exist_ok=True)


def local_now() -> datetime:
    return datetime.now()


def timestamp_slug(dt: datetime | None = None) -> str:
    return (dt or local_now()).strftime("%Y%m%d_%H%M%S")


def sanitize_filename(name: str, fallback: str = "untitled", max_len: int = 80) -> str:
    """把任意文本变成 Windows 合法文件名。"""
    bad = '<>:"/\\|?*'
    cleaned = "".join(("_" if ch in bad else ch) for ch in str(name))
    cleaned = "".join(ch for ch in cleaned if ord(ch) >= 32)
    cleaned = cleaned.strip().strip(".")
    cleaned = " ".join(cleaned.split())
    if not cleaned:
        cleaned = fallback
    if len(cleaned) > max_len:
        cleaned = cleaned[:max_len].rstrip()
    reserved = {
        "CON", "PRN", "AUX", "NUL",
        *(f"COM{i}" for i in range(1, 10)),
        *(f"LPT{i}" for i in range(1, 10)),
    }
    if cleaned.upper() in reserved:
        cleaned = f"_{cleaned}"
    return cleaned
