@echo off
chcp 65001 >nul
setlocal
call .venv\Scripts\activate
set SOURCE_DIR=D:\silver_coin_dataset\images\test
python infer.py --config configs/largeset_yolov8m.yaml --source %SOURCE_DIR%
pause
