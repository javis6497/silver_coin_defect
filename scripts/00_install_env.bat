@echo off
chcp 65001 >nul
setlocal
if not exist .venv (
  py -3 -m venv .venv
)
call .venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
echo [OK] 环境安装完成
pause
