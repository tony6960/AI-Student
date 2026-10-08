@echo off
chcp 65001 >nul
setlocal
cd /d "%~dp0"
echo ============================================
echo   AI学生 - 打包为 Windows 可执行程序
echo ============================================
if not exist ".venv\Scripts\python.exe" (
  echo [!] 未找到 .venv，请先执行: python -m venv .venv ^&^& .venv\Scripts\pip install -r requirements.txt
  pause & exit /b 1
)
".venv\Scripts\python.exe" -m pip show pyinstaller >nul 2>nul || ".venv\Scripts\python.exe" -m pip install pyinstaller
echo [1/2] 正在打包（首次可能需要几分钟）...
".venv\Scripts\python.exe" -m PyInstaller --noconfirm --clean --distpath dist --workpath build "packaging\ai_student.spec"
if errorlevel 1 ( echo [!] 打包失败 & pause & exit /b 1 )
echo [2/2] 完成！产物： dist\AI学生\AI学生.exe
echo 提示：可把整个 dist\AI学生 文件夹拷贝到任意 Windows 机器直接运行。
pause
endlocal