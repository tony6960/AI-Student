@echo off
chcp 65001 >nul
setlocal
cd /d "%~dp0"
set PYTHONPATH=%~dp0src
if exist "%~dp0.venv\Scripts\pythonw.exe" (
  start "" "%~dp0.venv\Scripts\pythonw.exe" -m ai_student %*
  goto :eof
)
where python >nul 2>nul || (echo 未找到 Python，请先安装 Python 3.10+ 并执行 pip install -r requirements.txt & pause & exit /b 1)
python -m ai_student %*
endlocal