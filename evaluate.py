from __future__ import annotations

import json
from pathlib import Path

from src.common.utils import ensure_dir, load_yaml, save_json
from src.eval.coco_eval import evaluate_torchvision
from src.models.ultralytics_runner import validate_ultralytics


def inject_runtime(cfg: dict) -> dict:
    runtime_path = Path(cfg['project']['output_dir']) / cfg['project']['name'] / 'runtime.json'
    if not runtime_path.exists():
        raise FileNotFoundError(f'请先执行 prepare_data.py，未找到: {runtime_path}')
    cfg['runtime'] = json.loads(runtime_path.read_text(encoding='utf-8'))
    return cfg


def main(config_path: str):
    cfg = inject_runtime(load_yaml(config_path))
    eval_dir = ensure_dir(Path(cfg['project']['output_dir']) / cfg['project']['name'] / 'eval')
    if cfg['model']['framework'] == 'ultralytics':
        res = validate_ultralytics(cfg)
        metrics = {
            'map': float(getattr(res.box, 'map', 0.0)),
            'map50': float(getattr(res.box, 'map50', 0.0)),
            'map75': float(getattr(res.box, 'map75', 0.0)),
            'mr': float(getattr(res.box, 'mr', 0.0)),
            'mp': float(getattr(res.box, 'mp', 0.0)),
        }
        save_json(metrics, eval_dir / 'coco_metrics.json')
    else:
        evaluate_torchvision(cfg)
    print(f'[OK] 评估结果输出到: {eval_dir}')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    args = parser.parse_args()
    main(args.config)
