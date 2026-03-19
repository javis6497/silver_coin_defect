@echo off
chcp 65001 >nul
setlocal
call .venv\Scripts\activate
python evaluate.py --config configs/fewshot_yolov8n.yaml
pause
