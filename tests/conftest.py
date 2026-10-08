"""pytest 全局配置：把 src 加入 sys.path，隔离应用数据目录，GUI 用 offscreen。"""
from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

TEST_HOME = ROOT / ".runs" / "test_home"
os.environ["AI_STUDENT_HOME"] = str(TEST_HOME)

import pytest  # noqa: E402


def pytest_sessionstart(session) -> None:  # noqa: ARG001
    shutil.rmtree(TEST_HOME, ignore_errors=True)
    TEST_HOME.mkdir(parents=True, exist_ok=True)


@pytest.fixture()
def tmp_home(tmp_path, monkeypatch):
    """把应用数据目录指向临时目录。"""
    monkeypatch.setenv("AI_STUDENT_HOME", str(tmp_path))
    return tmp_path
