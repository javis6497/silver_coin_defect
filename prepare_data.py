from __future__ import annotations

from pathlib import Path

from src.common.utils import ensure_dir, load_yaml, save_json, coco_categories_to_names
from src.preprocess.build_preprocessed_dataset import COCOPreprocessor
from src.data.coco_to_yolo import coco_to_yolo_detection, build_ultralytics_yaml


def main(config_path: str):
    cfg = load_yaml(config_path)
    dataset_root = Path(cfg['dataset']['root_dir'])
    out_root = Path(cfg['project']['output_dir']) / cfg['project']['name'] / 'prepared'
    ensure_dir(out_root)
    ensure_dir(out_root / 'images' / 'train')
    ensure_dir(out_root / 'images' / 'val')
    ensure_dir(out_root / 'annotations')
    ensure_dir(out_root / 'labels' / 'train')
    ensure_dir(out_root / 'labels' / 'val')

    pp = COCOPreprocessor(
        image_size=cfg['dataset'].get('image_size', 640),
        enable_enhance=cfg['preprocess'].get('enable_enhance', True),
        pad_ratio=cfg['preprocess'].get('pad_ratio', 0.08),
    )

    split_to_json = {
        'train': dataset_root / 'annotations' / 'instances_train.json',
        'val': dataset_root / 'annotations' / 'instances_val.json',
        'test': dataset_root / 'annotations' / 'instances_test.json',
    }
    available_splits = []
    for split, ann_path in split_to_json.items():
        images_dir = dataset_root / 'images' / split
        if not ann_path.exists() or not images_dir.exists():
            print(f'[INFO] 跳过 {split}: 未找到 {ann_path} 或 {images_dir}')
            continue
        ensure_dir(out_root / 'images' / split)
        ensure_dir(out_root / 'labels' / split)
        pp.process_split(
            images_dir=str(images_dir),
            ann_path=str(ann_path),
            out_images_dir=str(out_root / 'images' / split),
            out_ann_path=str(out_root / 'annotations' / f'instances_{split}.json'),
        )
        available_splits.append(split)

    if 'train' not in available_splits or 'val' not in available_splits:
        raise FileNotFoundError('至少需要 train 和 val 两个 split。请检查 images/ 与 annotations/ 目录。')

    import json
    with open(out_root / 'annotations' / 'instances_train.json', 'r', encoding='utf-8') as f:
        coco = json.load(f)
    names = coco_categories_to_names(coco)

    for split in available_splits:
        coco_to_yolo_detection(
            images_dir=str(out_root / 'images' / split),
            ann_path=str(out_root / 'annotations' / f'instances_{split}.json'),
            labels_dir=str(out_root / 'labels' / split),
        )

    yolo_yaml = out_root / 'silver_coin_yolo.yaml'
    build_ultralytics_yaml(str(yolo_yaml), str(out_root), names, has_test=('test' in available_splits))

    runtime = {
        'prepared_dataset_dir': str(out_root),
        'ultralytics_data_yaml': str(yolo_yaml),
        'class_names': names,
        'available_splits': available_splits,
    }
    save_json(runtime, Path(cfg['project']['output_dir']) / cfg['project']['name'] / 'runtime.json')
    print(f'[OK] prepared dataset -> {out_root}')
    print(f'[OK] yolo yaml -> {yolo_yaml}')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    args = parser.parse_args()
    main(args.config)
