@echo off
chcp 65001 >nul
setlocal
call .venv\Scripts\activate
set SOURCE_DIR=D:\silver_coin_dataset\images\test
python infer.py --config configs/fewshot_yolov8n.yaml --source %SOURCE_DIR%
pause
