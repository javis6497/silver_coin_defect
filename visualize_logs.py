from __future__ import annotations

from pathlib import Path

from src.common.utils import load_yaml
from src.visualize.plot_curves import plot_ultralytics_results_csv


def main(config_path: str):
    cfg = load_yaml(config_path)
    out_root = Path(cfg['project']['output_dir']) / cfg['project']['name']
    csv_path = out_root / 'train' / 'results.csv'
    if csv_path.exists():
        plot_ultralytics_results_csv(csv_path, out_root / 'curves')
        print(f'[OK] 曲线输出到: {out_root / "curves"}')
    else:
        print(f'[WARN] 未找到 Ultralytics results.csv: {csv_path}')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    args = parser.parse_args()
    main(args.config)
