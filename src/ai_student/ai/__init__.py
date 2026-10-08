"""ai_student.ai —— 模型层（语音转写 / 视觉理解 / 课堂笔记 / 任务 / 作答）。

全部函数同步阻塞（引擎放在线程池里调用），不做任何 UI 操作，失败抛 AIClientError。
"""
from __future__ import annotations

from .asr import transcribe_chunk
from .client import LLMClient
from .errors import AIClientError
from .notes import merge_notes
from .prompts import PROMPTS, get_prompt
from .solver import SolveResult, solve_task
from .tasks import extract_tasks, title_similarity
from .vision import BoardAnalysis, analyze_board

__all__ = [
    "AIClientError", "LLMClient",
    "transcribe_chunk",
    "BoardAnalysis", "analyze_board",
    "merge_notes",
    "extract_tasks", "title_similarity",
    "SolveResult", "solve_task",
    "PROMPTS", "get_prompt",
]
