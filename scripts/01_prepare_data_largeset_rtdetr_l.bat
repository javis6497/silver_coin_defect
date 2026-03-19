@echo off
chcp 65001 >nul
setlocal
call .venv\Scripts\activate
python prepare_data.py --config configs/largeset_rtdetr_l.yaml
pause
