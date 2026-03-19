from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

from src.common.utils import load_yaml


def _print_banner():
    print('=' * 72)
    print('银币表面瑕疵检测 - Python 统一启动器')
    print('=' * 72)


def _print_step(title: str):
    print(f'\n[阶段] {title}')
    print('-' * 72)


def _load_runtime(config_path: str) -> dict | None:
    cfg = load_yaml(config_path)
    runtime_path = Path(cfg['project']['output_dir']) / cfg['project']['name'] / 'runtime.json'
    if runtime_path.exists():
        return json.loads(runtime_path.read_text(encoding='utf-8'))
    return None


def _resolve_source(config_path: str, source: str | None) -> str | None:
    if source is None:
        return None
    if source not in {'train', 'val', 'test'}:
        return source
    runtime = _load_runtime(config_path)
    if runtime is None:
        raise FileNotFoundError('未找到 runtime.json，请先执行预处理（prepare）。')
    prepared_dir = Path(runtime['prepared_dataset_dir']) / 'images' / source
    return str(prepared_dir)


def run_install(index_url: str | None = None):
    _print_step('安装依赖')
    python_exe = sys.executable
    cmds = [
        [python_exe, '-m', 'pip', 'install', '--upgrade', 'pip'],
        [python_exe, '-m', 'pip', 'install', '-r', 'requirements.txt'],
    ]
    if index_url:
        cmds[1].extend(['-i', index_url])
    for cmd in cmds:
        print('[执行]', ' '.join(cmd))
        subprocess.run(cmd, check=True)
    print('[OK] 依赖安装完成')


def run_prepare(config_path: str):
    _print_step('数据预处理 / 圆币定位裁切 / 标签派生')
    from prepare_data import main as prepare_main
    prepare_main(config_path)


def run_train(config_path: str):
    _print_step('模型训练')
    from train import main as train_main
    train_main(config_path)


def run_evaluate(config_path: str):
    _print_step('模型评估')
    from evaluate import main as eval_main
    eval_main(config_path)


def run_infer(config_path: str, source: str):
    _print_step('模型推理')
    from infer import main as infer_main
    infer_main(config_path, source)


def run_visualize(config_path: str):
    _print_step('日志曲线可视化')
    from visualize_logs import main as vis_main
    vis_main(config_path)


def print_summary(config_path: str):
    cfg = load_yaml(config_path)
    out_root = Path(cfg['project']['output_dir']) / cfg['project']['name']
    print('\n[完成] 关键输出目录：')
    print(f'  训练输出: {out_root / "train"}')
    print(f'  评估输出: {out_root / "eval"}')
    print(f'  曲线输出: {out_root / "curves"}')
    print(f'  推理输出: {out_root / "infer"}')
    if cfg['model']['framework'] == 'torchvision':
        print(f'  最佳模型: {out_root / "train" / "best_fasterrcnn.pth"}')
    else:
        print(f'  最佳模型: {out_root / "train" / "weights" / "best.pt"}')


def interactive_menu():
    _print_banner()
    print('1. 安装依赖')
    print('2. 预处理数据')
    print('3. 训练模型')
    print('4. 评估模型')
    print('5. 推理图片')
    print('6. 生成曲线')
    print('7. 一键全流程（prepare -> train -> evaluate -> visualize）')
    print('0. 退出')
    config_path = input('\n请输入配置文件路径（例如 configs/fewshot_yolov8n.yaml）: ').strip()
    while True:
        choice = input('\n请输入菜单编号: ').strip()
        try:
            if choice == '1':
                run_install(None)
            elif choice == '2':
                run_prepare(config_path)
            elif choice == '3':
                run_train(config_path)
            elif choice == '4':
                run_evaluate(config_path)
            elif choice == '5':
                source = input('请输入推理来源（train/val/test 或 图片目录绝对路径）: ').strip()
                run_infer(config_path, _resolve_source(config_path, source))
            elif choice == '6':
                run_visualize(config_path)
            elif choice == '7':
                run_prepare(config_path)
                run_train(config_path)
                run_evaluate(config_path)
                run_visualize(config_path)
                print_summary(config_path)
            elif choice == '0':
                break
            else:
                print('无效编号，请重试。')
        except Exception as exc:
            print(f'[ERROR] {exc}')


def main():
    parser = argparse.ArgumentParser(description='银币表面瑕疵检测 Python 统一启动器')
    parser.add_argument('--config', help='配置文件路径，例如 configs/fewshot_yolov8n.yaml')
    parser.add_argument('--stage', default='menu', choices=['menu', 'install', 'prepare', 'train', 'evaluate', 'infer', 'visualize', 'all'])
    parser.add_argument('--source', default=None, help='推理来源；可写 train/val/test 或绝对路径')
    parser.add_argument('--index-url', default=None, help='pip 镜像地址，可选')
    args = parser.parse_args()

    start = time.time()
    if args.stage == 'menu' and not args.config:
        interactive_menu()
        return

    if args.stage == 'install':
        run_install(args.index_url)
        return

    if not args.config:
        raise ValueError('除 install/menu 外，其余 stage 都需要提供 --config')

    _print_banner()
    print(f'配置文件: {args.config}')
    print(f'执行阶段: {args.stage}')

    if args.stage == 'prepare':
        run_prepare(args.config)
    elif args.stage == 'train':
        run_train(args.config)
    elif args.stage == 'evaluate':
        run_evaluate(args.config)
    elif args.stage == 'infer':
        resolved_source = _resolve_source(args.config, args.source)
        if not resolved_source:
            raise ValueError('infer 阶段需要提供 --source')
        run_infer(args.config, resolved_source)
    elif args.stage == 'visualize':
        run_visualize(args.config)
    elif args.stage == 'all':
        run_prepare(args.config)
        run_train(args.config)
        run_evaluate(args.config)
        run_visualize(args.config)
    elif args.stage == 'menu':
        interactive_menu()
        return

    print_summary(args.config)
    print(f'\n总耗时: {(time.time() - start):.1f} 秒')


if __name__ == '__main__':
    main()
