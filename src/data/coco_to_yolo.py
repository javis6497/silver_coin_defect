from __future__ import annotations
import json
from collections import defaultdict
from pathlib import Path
from src.common.utils import ensure_dir


def coco_to_yolo_detection(images_dir: str, ann_path: str, labels_dir: str):
    """将 COCO 格式标注转换为 YOLO 格式文本标注"""
    ensure_dir(labels_dir)
    with open(ann_path, 'r', encoding='utf-8') as f:
        coco = json.load(f)

    img_map = {img['id']: img for img in coco['images']}
    cat_ids = sorted([c['id'] for c in coco['categories']])
    cat_id2idx = {cid: i for i, cid in enumerate(cat_ids)}
    anns_by_img = defaultdict(list)
    for ann in coco['annotations']:
        anns_by_img[ann['image_id']].append(ann)

    for img_id, img_item in img_map.items():
        w, h = img_item['width'], img_item['height']
        txt_path = Path(labels_dir) / (Path(img_item['file_name']).stem + '.txt')
        lines = []
        for ann in anns_by_img.get(img_id, []):
            x, y, bw, bh = ann['bbox']
            xc = (x + bw / 2.0) / w
            yc = (y + bh / 2.0) / h
            nw = bw / w
            nh = bh / h
            cls = cat_id2idx[ann['category_id']]
            lines.append(f'{cls} {xc:.6f} {yc:.6f} {nw:.6f} {nh:.6f}')
        txt_path.write_text('\n'.join(lines), encoding='utf-8')

    return cat_id2idx


def build_ultralytics_yaml(out_path: str, dataset_root: str, names: list, has_test=True):
    """
    生成 Ultralytics YOLOv8 所需 YAML 配置文件
    has_test: 是否写 test 字段
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    dataset_root_path = Path(dataset_root)

    lines = []
    lines.append("path: " + dataset_root_path.as_posix())
    lines.append("train: images/train")
    lines.append("val: images/val")
    if has_test:
        lines.append("test: images/test")
    lines.append("names:")

    for i, n in enumerate(names):
        lines.append(f"  {i}: {n}")

    out_path.write_text('\n'.join(lines), encoding='utf-8')

# 测试输出，确保 Python 使用的是新版模块
print("[INFO] 使用了新版 coco_to_yolo.py")