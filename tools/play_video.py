r"""把本地视频当作"极域电子教室演示窗口"播放，用来在没有真实教室时跑真实链路。

窗口标题里带「极域」「教师演示」，所以软件原有的窗口定位逻辑会像找极域一样找到它。
播放时声音走系统扬声器，软件的"系统回环"采集就能听到——整条链路（窗口抓屏 + 回环录音）
与真实上课完全一致，只是演示源换成了视频文件。

用法：
    python tools/play_video.py --file .runs\video\lecture.mp4 --start 60 --duration 300
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from PySide6.QtCore import Qt, QTimer, QUrl
from PySide6.QtWidgets import QApplication, QLabel, QMainWindow, QVBoxLayout, QWidget

WINDOW_TITLE = "极域电子教室 - 教师演示（视频测试源）"


def build_window(video: Path, start: float, size: tuple[int, int]):
    from PySide6.QtMultimedia import QAudioOutput, QMediaPlayer
    from PySide6.QtMultimediaWidgets import QVideoWidget

    class PlayerWindow(QMainWindow):
        def __init__(self) -> None:
            super().__init__()
            self.setWindowTitle(WINDOW_TITLE)
            # 极域广播窗口本身就是置顶的；这里也置顶，避免被其它窗口盖住导致抓屏抓到别人
            self.setWindowFlag(Qt.WindowType.WindowStaysOnTopHint, True)
            self.resize(*size)
            central = QWidget()
            layout = QVBoxLayout(central)
            layout.setContentsMargins(0, 0, 0, 0)
            self.video = QVideoWidget()
            layout.addWidget(self.video)
            self.setCentralWidget(central)

            self.player = QMediaPlayer(self)
            self.audio = QAudioOutput(self)
            self.audio.setVolume(0.85)
            self.player.setAudioOutput(self.audio)
            self.player.setVideoOutput(self.video)
            self.player.mediaStatusChanged.connect(self._on_status)
            self._start = max(0.0, start)
            self._seeked = False
            self.player.setSource(QUrl.fromLocalFile(str(video)))

        def _on_status(self, status) -> None:
            if status in (QMediaPlayer.MediaStatus.LoadedMedia,
                          QMediaPlayer.MediaStatus.BufferedMedia) and not self._seeked:
                self._seeked = True
                if self._start > 0:
                    self.player.setPosition(int(self._start * 1000))
                self.player.play()
                self.raise_()
                self.activateWindow()
                print("[player] 开始播放：%s  位置 %.1fs" % (video.name, self.player.position() / 1000.0), flush=True)
            if status == QMediaPlayer.MediaStatus.EndOfMedia:
                print("[player] 播放结束", flush=True)

    return PlayerWindow()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", required=True, help="视频文件路径")
    ap.add_argument("--start", type=float, default=0.0, help="从第几秒开始播")
    ap.add_argument("--duration", type=float, default=0.0, help="播放多少秒后自动退出（0=不自动退出）")
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    args = ap.parse_args(argv)

    video = Path(args.file).resolve()
    if not video.exists():
        print("找不到视频文件：%s" % video)
        return 2

    app = QApplication(sys.argv[:1])
    window = build_window(video, args.start, (args.width, args.height))
    window.show()
    print("[player] 窗口标题：%s" % WINDOW_TITLE, flush=True)

    if args.duration > 0:
        QTimer.singleShot(int(args.duration * 1000), app.quit)
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
