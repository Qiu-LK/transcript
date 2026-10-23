@echo off
chcp 65001 > nul
cd /d "%~dp0"
python -m venv .venv
if errorlevel 1 goto fail
".venv\Scripts\python.exe" -m pip install --upgrade pip
if errorlevel 1 goto fail
".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 goto fail
echo.
echo 安装完成，请双击 start.bat。
pause
exit /b 0
:fail
echo 安装失败，请检查网络和 Python 安装。
pause
exit /b 1
