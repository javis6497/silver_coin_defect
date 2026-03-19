@echo off
chcp 65001 >nul
setlocal
call .venv\Scripts\activate
python train.py --config configs/fewshot_yolov8n.yaml
python visualize_logs.py --config configs/fewshot_yolov8n.yaml
pause
