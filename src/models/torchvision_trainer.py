from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torchvision
from torch.utils.data import DataLoader
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from tqdm import tqdm

from src.data.coco_dataset import COCODetectionDataset, collate_fn


class TorchvisionTrainer:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.device = torch.device('cuda' if torch.cuda.is_available() and str(cfg['train'].get('device', '0')) != 'cpu' else 'cpu')
        out_root = Path(cfg['project']['output_dir']) / cfg['project']['name']
        self.train_dir = out_root / 'train'
        self.curve_dir = out_root / 'curves'
        self.train_dir.mkdir(parents=True, exist_ok=True)
        self.curve_dir.mkdir(parents=True, exist_ok=True)

    def build_model(self, num_classes: int):
        model = fasterrcnn_resnet50_fpn(weights='DEFAULT')
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
        return model

    def run(self):
        ds_root = Path(self.cfg['runtime']['prepared_dataset_dir'])
        train_ds = COCODetectionDataset(ds_root / 'images' / 'train', ds_root / 'annotations' / 'instances_train.json')
        val_ds = COCODetectionDataset(ds_root / 'images' / 'val', ds_root / 'annotations' / 'instances_val.json')
        num_classes = len(train_ds.cats) + 1

        train_loader = DataLoader(
            train_ds,
            batch_size=self.cfg['train'].get('batch_size', 2),
            shuffle=True,
            num_workers=self.cfg['train'].get('workers', 2),
            collate_fn=collate_fn,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=1,
            shuffle=False,
            num_workers=self.cfg['train'].get('workers', 2),
            collate_fn=collate_fn,
        )

        model = self.build_model(num_classes).to(self.device)
        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(params, lr=self.cfg['train'].get('lr0', 1e-4), weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=max(1, self.cfg['train'].get('epochs', 20) // 3), gamma=0.1)

        best_loss = 1e18
        log_path = self.train_dir / 'train_log.csv'
        with open(log_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['epoch', 'train_loss', 'val_score_proxy', 'lr'])
            for epoch in range(1, self.cfg['train'].get('epochs', 20) + 1):
                model.train()
                total_loss = 0.0
                pbar = tqdm(train_loader, desc=f'Epoch {epoch}')
                for images, targets in pbar:
                    images = [img.to(self.device) for img in images]
                    targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]
                    loss_dict = model(images, targets)
                    losses = sum(loss for loss in loss_dict.values())
                    optimizer.zero_grad()
                    losses.backward()
                    optimizer.step()
                    total_loss += losses.item()
                    pbar.set_postfix(loss=f'{losses.item():.4f}')

                train_loss = total_loss / max(1, len(train_loader))
                val_score_proxy = self._validate_proxy(model, val_loader)
                lr = optimizer.param_groups[0]['lr']
                writer.writerow([epoch, round(train_loss, 6), round(val_score_proxy, 6), lr])
                f.flush()
                if train_loss < best_loss:
                    best_loss = train_loss
                    torch.save(model.state_dict(), self.train_dir / 'best_fasterrcnn.pth')
                scheduler.step()

        self._plot_curves(log_path)

    @torch.no_grad()
    def _validate_proxy(self, model, data_loader):
        model.eval()
        scores = []
        for images, _ in data_loader:
            images = [img.to(self.device) for img in images]
            outputs = model(images)
            for out in outputs:
                if len(out['scores']) > 0:
                    scores.append(float(out['scores'][0].detach().cpu()))
                else:
                    scores.append(0.0)
        return sum(scores) / max(1, len(scores))

    def _plot_curves(self, csv_path: Path):
        import pandas as pd
        df = pd.read_csv(csv_path)
        plt.figure(figsize=(8, 5))
        plt.plot(df['epoch'], df['train_loss'])
        plt.xlabel('epoch')
        plt.ylabel('train_loss')
        plt.title('FasterRCNN Train Loss')
        plt.tight_layout()
        plt.savefig(self.curve_dir / 'fasterrcnn_train_loss.png', dpi=150)
        plt.close()

        plt.figure(figsize=(8, 5))
        plt.plot(df['epoch'], df['val_score_proxy'])
        plt.xlabel('epoch')
        plt.ylabel('val_score_proxy')
        plt.title('FasterRCNN Validation Score Proxy')
        plt.tight_layout()
        plt.savefig(self.curve_dir / 'fasterrcnn_val_proxy.png', dpi=150)
        plt.close()
