"""日志：控制台 + 滚动文件。"""
from __future__ import annotations

import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

from ..config.paths import logs_root

_configured = False


class QtLogBridge(logging.Handler):
    """把日志转发到 GUI（core.controller 会注册回调）。"""

    def __init__(self) -> None:
        super().__init__()
        self.callback = None

    def emit(self, record: logging.LogRecord) -> None:  # pragma: no cover - UI 相关
        cb = self.callback
        if cb is None:
            return
        try:
            cb(record.levelname, self.format(record))
        except Exception:
            pass


qt_bridge = QtLogBridge()


def setup_logging(level: str = "INFO", session_log: Path | None = None) -> None:
    global _configured
    root = logging.getLogger()
    if _configured:
        if session_log is not None:
            _attach_session_file(session_log)
        return
    root.setLevel(getattr(logging, level.upper(), logging.INFO))
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s", "%H:%M:%S")

    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(fmt)
    root.addHandler(stream)

    logs_root().mkdir(parents=True, exist_ok=True)
    fileh = RotatingFileHandler(
        logs_root() / "ai_student.log", maxBytes=4 * 1024 * 1024, backupCount=3, encoding="utf-8"
    )
    fileh.setFormatter(fmt)
    root.addHandler(fileh)

    qt_bridge.setFormatter(logging.Formatter("%(message)s"))
    root.addHandler(qt_bridge)

    if session_log is not None:
        _attach_session_file(session_log)
    _configured = True


def _attach_session_file(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    for h in logging.getLogger().handlers:
        if isinstance(h, logging.FileHandler) and Path(h.baseFilename) == path:
            return
    fh = logging.FileHandler(path, encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
    logging.getLogger().addHandler(fh)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
