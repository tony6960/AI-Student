"""应用入口：装配配置、引擎、界面与托盘。"""
from __future__ import annotations

import argparse
import logging
import sys

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication, QMessageBox, QSystemTrayIcon

from . import APP_DISPLAY_NAME, __version__
from .config.paths import ensure_dirs
from .utils.logging import setup_logging

log = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="ai-student", description=APP_DISPLAY_NAME + " —— 自动听课的 AI 学生")
    p.add_argument("--version", action="version", version=APP_DISPLAY_NAME + " " + __version__)
    p.add_argument("--log-level", default="INFO", help="DEBUG/INFO/WARNING/ERROR")
    p.add_argument("--config", default="", help="指定配置文件路径")
    p.add_argument("--output-dir", default="", help="覆盖作业输出目录")
    p.add_argument("--smoke", action="store_true", help="只构建界面后立即退出（自检用）")
    p.add_argument("--no-tray", action="store_true", help="不创建托盘图标")
    p.add_argument("--auto-start", action="store_true", help="启动后立即开始听课")
    p.add_argument("--preview", default="", help="把主窗口渲染成 PNG 保存后退出（自检用）")
    return p


def run(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv if argv is not None else sys.argv[1:])
    setup_logging(args.log_level)
    ensure_dirs()

    QApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
    )
    app = QApplication(sys.argv[:1])
    app.setApplicationName(APP_DISPLAY_NAME)
    app.setApplicationDisplayName(APP_DISPLAY_NAME)
    app.setOrganizationName("AIStudent")
    app.setQuitOnLastWindowClosed(False)

    from .ui.theme import apply_theme

    apply_theme(app)

    from .core.controller import AppController

    if args.config:
        from .config.store import load_config

        controller = AppController(config=load_config(args.config))
    else:
        controller = AppController()
    if args.output_dir:
        cfg = controller.get_config()
        cfg.output.output_dir = args.output_dir
        controller.update_config(cfg)

    from .ui.main_window import MainWindow

    window = MainWindow(controller)

    if not args.no_tray and QSystemTrayIcon.isSystemTrayAvailable():
        try:
            from .ui.tray import TrayController

            TrayController(controller, window, app)
        except Exception as exc:  # pragma: no cover
            log.warning("托盘不可用: %s", exc)
    else:
        app.setQuitOnLastWindowClosed(True)

    window.show()

    problems = controller.needs_setup()
    if problems and not args.smoke and not args.preview:
        try:
            from .ui.dialogs import FirstRunWizard

            wizard = FirstRunWizard(controller.get_config(), window)
            if wizard.exec():
                controller.update_config(wizard.result_config())
        except Exception as exc:  # pragma: no cover
            log.warning("首次运行向导失败: %s", exc)
            QMessageBox.information(
                window,
                "需要先完成设置",
                "请先在「设置」中填写多模态模型 API KEY、个人信息与作业输出目录。" + chr(10) + chr(10) + "缺少：" + chr(10) + "- "
                + (chr(10) + "- ").join(problems),
            )

    app.aboutToQuit.connect(controller.shutdown)

    from PySide6.QtCore import QTimer

    if args.preview:
        def _grab() -> None:
            try:
                window.grab().save(args.preview)
                log.info("界面截图已保存: %s", args.preview)
            except Exception as exc:
                log.error("截图失败: %s", exc)
            app.quit()

        QTimer.singleShot(700, _grab)

    if args.auto_start:
        QTimer.singleShot(900, lambda: _safe_start(controller, window))

    if args.smoke:
        QTimer.singleShot(500, app.quit)

    return app.exec()


def _safe_start(controller, window) -> None:
    try:
        controller.start_session()
    except Exception as exc:
        QMessageBox.warning(window, "无法开始听课", str(exc))


def main() -> None:
    raise SystemExit(run())


if __name__ == "__main__":
    main()
