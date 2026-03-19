from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from PIL import Image
from torch.utils.data import Dataset


class COCODetectionDataset(Dataset):
    def __init__(self, images_dir: str, ann_path: str, transforms=None):
        self.images_dir = Path(images_dir)
        self.transforms = transforms
        with open(ann_path, 'r', encoding='utf-8') as f:
            self.coco = json.load(f)
        self.images = self.coco['images']
        self.cats = sorted(self.coco['categories'], key=lambda x: x['id'])
        self.cat_id2label = {c['id']: i + 1 for i, c in enumerate(self.cats)}
        self.label2name = {i + 1: c['name'] for i, c in enumerate(self.cats)}
        self.anns_by_img = defaultdict(list)
        for ann in self.coco.get('annotations', []):
            self.anns_by_img[ann['image_id']].append(ann)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        info = self.images[idx]
        img_path = self.images_dir / info['file_name']
        image = Image.open(img_path).convert('RGB')
        anns = self.anns_by_img.get(info['id'], [])

        boxes, labels, areas, iscrowd = [], [], [], []
        for ann in anns:
            x, y, w, h = ann['bbox']
            if w <= 1 or h <= 1:
                continue
            boxes.append([x, y, x + w, y + h])
            labels.append(self.cat_id2label[ann['category_id']])
            areas.append(w * h)
            iscrowd.append(ann.get('iscrowd', 0))

        if len(boxes) == 0:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
            areas = torch.zeros((0,), dtype=torch.float32)
            iscrowd = torch.zeros((0,), dtype=torch.int64)
        else:
            boxes = torch.tensor(boxes, dtype=torch.float32)
            labels = torch.tensor(labels, dtype=torch.int64)
            areas = torch.tensor(areas, dtype=torch.float32)
            iscrowd = torch.tensor(iscrowd, dtype=torch.int64)

        target = {
            'boxes': boxes,
            'labels': labels,
            'image_id': torch.tensor([info['id']]),
            'area': areas,
            'iscrowd': iscrowd,
        }

        image = torch.from_numpy(__import__('numpy').array(image)).permute(2, 0, 1).float() / 255.0
        if self.transforms is not None:
            image, target = self.transforms(image, target)
        return image, target


def collate_fn(batch):
    return tuple(zip(*batch))
