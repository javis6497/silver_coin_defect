from __future__ import annotations

import json
from pathlib import Path

from src.common.utils import load_yaml, set_seed
from src.models.ultralytics_runner import train_ultralytics
from src.models.torchvision_trainer import TorchvisionTrainer


def inject_runtime(cfg: dict) -> dict:
    runtime_path = Path(cfg['project']['output_dir']) / cfg['project']['name'] / 'runtime.json'
    if not runtime_path.exists():
        raise FileNotFoundError(f'请先执行 prepare_data.py，未找到: {runtime_path}')
    cfg['runtime'] = json.loads(runtime_path.read_text(encoding='utf-8'))
    return cfg


def main(config_path: str):
    cfg = load_yaml(config_path)
    set_seed(cfg.get('seed', 42))
    cfg = inject_runtime(cfg)

    if cfg['model']['framework'] == 'ultralytics':
        train_ultralytics(cfg)
    elif cfg['model']['framework'] == 'torchvision':
        TorchvisionTrainer(cfg).run()
    else:
        raise ValueError(f"未知 framework: {cfg['model']['framework']}")
    print('[OK] 训练完成')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    args = parser.parse_args()
    main(args.config)
