import json
import os
import random
from pathlib import Path
from typing import Any, Dict

import numpy as np
import yaml


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def load_yaml(path: str | Path) -> Dict[str, Any]:
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def save_json(data: Dict[str, Any], path: str | Path) -> None:
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
    except Exception:
        pass


def abs_path(base: str | Path, maybe_rel: str | Path) -> str:
    p = Path(maybe_rel)
    return str(p if p.is_absolute() else Path(base) / p)


def coco_categories_to_names(coco: Dict[str, Any]) -> list[str]:
    cats = sorted(coco['categories'], key=lambda x: x['id'])
    return [c['name'] for c in cats]


def count_annotations_per_class(coco: Dict[str, Any]) -> Dict[int, int]:
    d = {}
    for ann in coco.get('annotations', []):
        cid = ann['category_id']
        d[cid] = d.get(cid, 0) + 1
    return d
