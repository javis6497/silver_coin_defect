from __future__ import annotations

import json
from pathlib import Path

import cv2
import torch
from tqdm import tqdm

from src.common.utils import load_yaml
from src.data.coco_dataset import COCODetectionDataset
from src.models.torchvision_trainer import TorchvisionTrainer
from src.models.ultralytics_runner import infer_ultralytics


def inject_runtime(cfg: dict) -> dict:
    runtime_path = Path(cfg['project']['output_dir']) / cfg['project']['name'] / 'runtime.json'
    if not runtime_path.exists():
        raise FileNotFoundError(f'请先执行 prepare_data.py，未找到: {runtime_path}')
    cfg['runtime'] = json.loads(runtime_path.read_text(encoding='utf-8'))
    return cfg


def infer_torchvision(cfg: dict, source: str):
    out_root = Path(cfg['project']['output_dir']) / cfg['project']['name']
    infer_dir = out_root / 'infer'
    infer_dir.mkdir(parents=True, exist_ok=True)

    split = Path(source).name if Path(source).name in ['train', 'val', 'test'] else None
    if split is not None:
        images_dir = Path(cfg['runtime']['prepared_dataset_dir']) / 'images' / split
        ann_path = Path(cfg['runtime']['prepared_dataset_dir']) / 'annotations' / f'instances_{split}.json'
    else:
        raise ValueError('当前 torchvision 推理入口为保证一次跑通，要求 source 使用 prepared 数据集下的 train/val/test 目录名。')

    ds = COCODetectionDataset(images_dir, ann_path)
    trainer = TorchvisionTrainer(cfg)
    model = trainer.build_model(len(ds.cats) + 1)
    ckpt = out_root / 'train' / 'best_fasterrcnn.pth'
    model.load_state_dict(torch.load(ckpt, map_location=trainer.device))
    model.to(trainer.device)
    model.eval()

    label2name = {i + 1: c['name'] for i, c in enumerate(ds.cats)}
    for image, target in tqdm(ds, desc='Infer FasterRCNN'):
        image_id = int(target['image_id'].item())
        image_np = (image.permute(1, 2, 0).numpy() * 255).astype('uint8')[:, :, ::-1].copy()
        with torch.no_grad():
            output = model([image.to(trainer.device)])[0]
        for box, score, label in zip(output['boxes'], output['scores'], output['labels']):
            if float(score) < cfg['infer'].get('conf', 0.25):
                continue
            x1, y1, x2, y2 = [int(v) for v in box.detach().cpu().tolist()]
            cls_name = label2name[int(label.item())]
            cv2.rectangle(image_np, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(image_np, f'{cls_name}:{float(score):.2f}', (x1, max(20, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.imwrite(str(infer_dir / f'{image_id}.jpg'), image_np)


def main(config_path: str, source: str):
    cfg = inject_runtime(load_yaml(config_path))
    if cfg['model']['framework'] == 'ultralytics':
        infer_ultralytics(cfg, source)
    else:
        infer_torchvision(cfg, source)
    print('[OK] 推理完成')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--source', required=True)
    args = parser.parse_args()
    main(args.config, args.source)
