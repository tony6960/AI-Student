"""打包入口脚本（PyInstaller 用）。"""
from __future__ import annotations

import multiprocessing
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def main() -> None:
    multiprocessing.freeze_support()
    from ai_student.app import run

    sys.exit(run())


if __name__ == "__main__":
    main()
