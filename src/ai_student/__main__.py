"""python -m ai_student 入口。"""
from __future__ import annotations

import sys


def main(argv: list[str] | None = None) -> int:
    from .app import run

    return run(argv)


if __name__ == "__main__":
    sys.exit(main())
