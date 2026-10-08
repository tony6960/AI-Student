# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller 打包配置：AI学生 Windows 桌面端。"""
from PyInstaller.utils.hooks import collect_all

datas, binaries, hiddenimports = [], [], []

for pkg in ("imageio_ffmpeg", "soundfile", "docx", "mss", "pyaudiowpatch", "soundcard"):
    try:
        d, b, h = collect_all(pkg)
        datas += d
        binaries += b
        hiddenimports += h
    except Exception as exc:  # 允许缺失可选依赖
        print("[spec] skip", pkg, exc)

hiddenimports += [
    "ai_student",
    "ai_student.app",
    "ai_student.capture.audio",
    "ai_student.capture.screen",
    "ai_student.capture.devices",
    "ai_student.capture.ffmpeg",
    "ai_student.ai.client",
    "ai_student.ai.asr",
    "ai_student.ai.vision",
    "ai_student.ai.notes",
    "ai_student.ai.tasks",
    "ai_student.ai.solver",
    "ai_student.ai.prompts",
    "ai_student.ui.main_window",
    "ai_student.ui.settings_dialog",
    "ai_student.ui.dialogs",
    "ai_student.ui.panels",
    "ai_student.ui.tray",
    "numpy",
    "PIL",
    "openai",
    "httpx",
]

a = Analysis(
    ["..\\run_app.py"],
    pathex=["..\\src"],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    runtime_hooks=[],
    excludes=["tkinter", "matplotlib", "pytest", "IPython"],
    noarchive=False,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="AI学生",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="AI学生",
)
