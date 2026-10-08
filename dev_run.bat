@echo off
chcp 65001 >nul
cd /d "%~dp0"
set PYTHONPATH=%~dp0src
"%~dp0.venv\Scripts\python.exe" -m ai_student --log-level DEBUG %*