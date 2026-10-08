"""入口层与契约完整性测试（不需要网络/设备）。"""
from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest  # noqa: E402

from ai_student.config.models import AppConfig, ModelEndpoint  # noqa: E402
from ai_student.core.api import ClassAgentAPI  # noqa: E402


API_METHODS = [
    "get_config", "update_config", "start_session", "stop_session", "set_paused",
    "is_recording", "analyze_pending", "solve_task", "solve_all_pending",
    "regenerate_task", "delete_task", "add_manual_task", "update_task",
    "current_bundle", "list_sessions", "load_session", "export_session",
    "probe_devices", "open_path", "shutdown",
    "audio_level_db", "preview_png", "engine_state",
]

API_SIGNALS = [
    "sessionStarted", "sessionStopped", "stateChanged", "stageChanged",
    "transcriptSegment", "boardNote", "notesUpdated", "taskDiscovered",
    "taskUpdated", "artifactWritten", "logMessage", "errorRaised",
    "progress", "configChanged", "captureDevicesChanged",
]


def test_api_declares_expected_surface():
    for name in API_METHODS:
        assert hasattr(ClassAgentAPI, name), "ClassAgentAPI 缺少方法 " + name
    for name in API_SIGNALS:
        assert hasattr(ClassAgentAPI, name), "ClassAgentAPI 缺少信号 " + name


def test_controller_overrides_every_api_method(tmp_home):
    from ai_student.core.controller import AppController

    missing = []
    for name in API_METHODS:
        fn = getattr(AppController, name, None)
        base = getattr(ClassAgentAPI, name, None)
        if fn is None or (base is not None and getattr(fn, "__func__", fn) is getattr(base, "__func__", base)):
            missing.append(name)
    assert not missing, "AppController 未实现：" + ", ".join(missing)


def test_controller_constructs_and_roundtrips_config(tmp_home):
    from ai_student.core.controller import AppController

    ctrl = AppController()
    cfg = ctrl.get_config()
    assert cfg.output.output_dir, "默认输出目录应当被填上"
    assert isinstance(ctrl.needs_setup(), list)
    assert ctrl.engine_state() == "idle"
    assert ctrl.audio_level_db() <= 0.0
    assert ctrl.preview_png() is None
    assert ctrl.list_sessions() == []
    assert ctrl.current_bundle() is None

    cfg.person.name = "集成测试"
    cfg.model.api_key = "sk-integration-test"
    ctrl.update_config(cfg)
    assert ctrl.get_config().person.name == "集成测试"

    # 停止/关停不应抛异常
    ctrl.stop_session()
    ctrl.shutdown()


def test_controller_start_rejects_incomplete_config(tmp_home):
    from ai_student.core.controller import AppController

    ctrl = AppController()
    errors = []
    ctrl.errorRaised.connect(lambda t, d: errors.append((t, d)))
    with pytest.raises(ValueError):
        ctrl.start_session()
    assert errors and errors[0][0] == "无法开始听课"
    ctrl.shutdown()


def test_app_parser_and_entrypoint_modules():
    from ai_student.app import build_parser

    args = build_parser().parse_args(["--smoke", "--no-tray", "--output-dir", "X:/out"])
    assert args.smoke and args.no_tray
    assert args.output_dir == "X:/out"

    import ai_student.__main__ as m

    assert callable(m.main)


def test_default_factories_resolve_real_implementations(tmp_home, tmp_path):
    from ai_student.capture.audio import AudioRecorder
    from ai_student.capture.screen import ScreenRecorder
    from ai_student.core.engine import (
        _default_audio_factory, _default_client_factory, _default_screen_factory,
    )

    cfg = AppConfig().capture
    a = _default_audio_factory(cfg, tmp_path, None, None)
    s = _default_screen_factory(cfg, tmp_path, None, None)
    assert isinstance(a, AudioRecorder)
    assert isinstance(s, ScreenRecorder)
    assert not a.is_running and not s.is_running

    ep = ModelEndpoint(base_url="http://127.0.0.1:9/v1", api_key="x", chat_model="m")
    client = _default_client_factory(ep)
    assert client is not None


def test_engine_uses_real_ai_modules_when_not_injected():
    """确认引擎默认拿到的是真实 ai 子模块（而不是被测试替身污染）。"""
    from ai_student.core.engine import ClassEngine, EngineCallbacks

    engine = ClassEngine(lambda: AppConfig(), EngineCallbacks())
    for name in ("asr", "vision", "notes", "tasks", "solver"):
        mod = getattr(engine.ai, name)
        assert mod is not None, name
        assert hasattr(mod, "transcribe_chunk") or hasattr(mod, "analyze_board") \
            or hasattr(mod, "merge_notes") or hasattr(mod, "extract_tasks") \
            or hasattr(mod, "solve_task")
