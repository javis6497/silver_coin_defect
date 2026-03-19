from __future__ import annotations

from pathlib import Path

from ultralytics import RTDETR, YOLO


def _make_model(model_name: str, weights: str):
    if 'rtdetr' in model_name.lower():
        return RTDETR(weights)
    return YOLO(weights)


def train_ultralytics(cfg: dict):
    model = _make_model(cfg['model']['name'], cfg['model']['weights'])
    train_cfg = cfg['train']
    output_dir = Path(cfg['project']['output_dir']) / cfg['project']['name'] / 'train'
    output_dir.mkdir(parents=True, exist_ok=True)
    model.train(
        data=cfg['runtime']['ultralytics_data_yaml'],
        imgsz=train_cfg.get('imgsz', 640),
        epochs=train_cfg.get('epochs', 100),
        batch=train_cfg.get('batch_size', 8),
        device=train_cfg.get('device', 0),
        workers=train_cfg.get('workers', 4),
        project=str(output_dir.parent),
        name='train',
        pretrained=True,
        patience=train_cfg.get('patience', 30),
        optimizer=train_cfg.get('optimizer', 'auto'),
        lr0=train_cfg.get('lr0', 0.001),
        hsv_h=train_cfg.get('hsv_h', 0.015),
        hsv_s=train_cfg.get('hsv_s', 0.7),
        hsv_v=train_cfg.get('hsv_v', 0.4),
        degrees=train_cfg.get('degrees', 3.0),
        translate=train_cfg.get('translate', 0.03),
        scale=train_cfg.get('scale', 0.15),
        fliplr=train_cfg.get('fliplr', 0.5),
        mosaic=train_cfg.get('mosaic', 0.2),
        mixup=train_cfg.get('mixup', 0.0),
        save=True,
        exist_ok=True,
        verbose=True,
    )


def validate_ultralytics(cfg: dict):
    output_root = Path(cfg['project']['output_dir']) / cfg['project']['name']
    weights = output_root / 'train' / 'weights' / 'best.pt'
    model = _make_model(cfg['model']['name'], str(weights))
    res = model.val(
        data=cfg['runtime']['ultralytics_data_yaml'],
        split='test' if ('test' in cfg['runtime'].get('available_splits', []) and cfg['dataset'].get('use_test_for_eval', True)) else 'val',
        imgsz=cfg['train'].get('imgsz', 640),
        batch=cfg['train'].get('batch_size', 8),
        device=cfg['train'].get('device', 0),
        save_json=True,
        project=str(output_root),
        name='eval_ultralytics',
        exist_ok=True,
    )
    return res


def infer_ultralytics(cfg: dict, source: str):
    output_root = Path(cfg['project']['output_dir']) / cfg['project']['name']
    weights = output_root / 'train' / 'weights' / 'best.pt'
    model = _make_model(cfg['model']['name'], str(weights))
    model.predict(
        source=source,
        imgsz=cfg['train'].get('imgsz', 640),
        conf=cfg['infer'].get('conf', 0.25),
        iou=cfg['infer'].get('iou', 0.5),
        device=cfg['train'].get('device', 0),
        save=True,
        save_txt=True,
        save_conf=True,
        project=str(output_root),
        name='infer',
        exist_ok=True,
    )
