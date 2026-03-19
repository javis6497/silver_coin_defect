from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def plot_ultralytics_results_csv(csv_path: str, out_dir: str):
    csv_path = Path(csv_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(csv_path)
    columns = list(df.columns)
    metric_candidates = [c for c in columns if any(k in c.lower() for k in ['loss', 'map', 'precision', 'recall'])]
    for col in metric_candidates:
        if col == 'epoch':
            continue
        plt.figure(figsize=(8, 5))
        x = df['epoch'] if 'epoch' in df.columns else range(len(df))
        plt.plot(x, df[col])
        plt.xlabel('epoch')
        plt.ylabel(col)
        plt.title(col)
        plt.tight_layout()
        safe_name = col.replace('/', '_').replace('(', '').replace(')', '').replace(' ', '_')
        plt.savefig(out_dir / f'{safe_name}.png', dpi=150)
        plt.close()
