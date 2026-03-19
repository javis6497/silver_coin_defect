@echo off
chcp 65001 >nul
setlocal
call .venv\Scripts\activate
python infer.py --config configs/fewshot_fasterrcnn.yaml --source test
pause
