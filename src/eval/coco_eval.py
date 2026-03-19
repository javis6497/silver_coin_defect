from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, List

import torch
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from tqdm import tqdm

from src.data.coco_dataset import COCODetectionDataset
from src.models.torchvision_trainer import TorchvisionTrainer


def evaluate_torchvision(cfg: dict):
    out_root = Path(cfg['project']['output_dir']) / cfg['project']['name']
    eval_dir = out_root / 'eval'
    eval_dir.mkdir(parents=True, exist_ok=True)

    ds_root = Path(cfg['runtime']['prepared_dataset_dir'])
    split = 'test' if cfg['dataset'].get('use_test_for_eval', True) else 'val'
    ann_path = ds_root / 'annotations' / f'instances_{split}.json'
    images_dir = ds_root / 'images' / split
    ds = COCODetectionDataset(images_dir, ann_path)

    trainer = TorchvisionTrainer(cfg)
    model = trainer.build_model(len(ds.cats) + 1)
    ckpt = out_root / 'train' / 'best_fasterrcnn.pth'
    model.load_state_dict(torch.load(ckpt, map_location=trainer.device))
    model.to(trainer.device)
    model.eval()

    predictions = []
    for image, target in tqdm(ds, desc='Eval FasterRCNN'):
        image = image.to(trainer.device)
        with torch.no_grad():
            output = model([image])[0]
        image_id = int(target['image_id'].item())
        for box, score, label in zip(output['boxes'], output['scores'], output['labels']):
            score = float(score.cpu())
            if score < cfg['infer'].get('conf', 0.25):
                continue
            x1, y1, x2, y2 = box.detach().cpu().tolist()
            predictions.append({
                'image_id': image_id,
                'category_id': ds.cats[int(label.item()) - 1]['id'],
                'bbox': [x1, y1, x2 - x1, y2 - y1],
                'score': score,
            })

    pred_json = eval_dir / 'predictions.json'
    with open(pred_json, 'w', encoding='utf-8') as f:
        json.dump(predictions, f, indent=2)

    coco_gt = COCO(str(ann_path))
    coco_dt = coco_gt.loadRes(str(pred_json)) if predictions else coco_gt.loadRes([])
    coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    metrics = {
        'mAP@[0.50:0.95]': float(coco_eval.stats[0]),
        'AP50': float(coco_eval.stats[1]),
        'AP75': float(coco_eval.stats[2]),
        'AR@1': float(coco_eval.stats[6]),
        'AR@10': float(coco_eval.stats[7]),
    }
    with open(eval_dir / 'coco_metrics.json', 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    _save_per_class_ap50(coco_eval, coco_gt, eval_dir / 'per_class_ap50.csv')
    _save_confusion_stub(eval_dir / 'confusion_stub.csv', coco_gt)


def _save_per_class_ap50(coco_eval: COCOeval, coco_gt: COCO, out_csv: Path):
    precisions = coco_eval.eval['precision']
    cat_ids = coco_gt.getCatIds()
    cats = coco_gt.loadCats(cat_ids)
    rows = []
    for idx, cat in enumerate(cats):
        precision = precisions[0, :, idx, 0, 2]
        precision = precision[precision > -1]
        ap50 = float(precision.mean()) if precision.size else 0.0
        rows.append((cat['name'], ap50))
    with open(out_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['class_name', 'ap50'])
        writer.writerows(rows)


def _save_confusion_stub(out_csv: Path, coco_gt: COCO):
    cats = coco_gt.loadCats(coco_gt.getCatIds())
    with open(out_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['class_name', 'tp_placeholder', 'fp_placeholder', 'fn_placeholder'])
        for c in cats:
            writer.writerow([c['name'], '', '', ''])
