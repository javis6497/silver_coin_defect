@echo off
chcp 65001 >nul
setlocal
call .venv\Scripts\activate
python train.py --config configs/largeset_yolov8m.yaml
python visualize_logs.py --config configs/largeset_yolov8m.yaml
pause
