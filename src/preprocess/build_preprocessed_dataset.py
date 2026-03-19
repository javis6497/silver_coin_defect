from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, Tuple

import cv2
from tqdm import tqdm

from src.common.utils import ensure_dir
from src.preprocess.coin_cropper import CoinAutoCropper, enhance_image, remap_bbox_to_crop


class COCOPreprocessor:
    def __init__(self, image_size: int = 640, enable_enhance: bool = True, pad_ratio: float = 0.08):
        self.image_size = image_size
        self.enable_enhance = enable_enhance
        self.cropper = CoinAutoCropper(pad_ratio=pad_ratio)

    def process_split(self, images_dir: str, ann_path: str, out_images_dir: str, out_ann_path: str):
        ensure_dir(out_images_dir)
        ensure_dir(Path(out_ann_path).parent)

        with open(ann_path, 'r', encoding='utf-8') as f:
            coco = json.load(f)

        anns_by_img = defaultdict(list)
        for ann in coco.get('annotations', []):
            anns_by_img[ann['image_id']].append(ann)

        new_images = []
        new_anns = []
        ann_id = 1

        for img_item in tqdm(coco['images'], desc=f'Preprocess {Path(ann_path).stem}'):
            file_name = img_item['file_name']
            image_path = Path(images_dir) / file_name
            image = cv2.imread(str(image_path))
            if image is None:
                print(f'[WARN] 读取失败，跳过: {image_path}')
                continue

            cropped, crop_result = self.cropper.crop_image(image)
            if self.enable_enhance:
                cropped = enhance_image(cropped)
            resized = cv2.resize(cropped, (self.image_size, self.image_size), interpolation=cv2.INTER_LINEAR)

            out_name = Path(file_name).stem + '.jpg'
            out_path = Path(out_images_dir) / out_name
            cv2.imwrite(str(out_path), resized)

            new_img = {
                'id': img_item['id'],
                'file_name': out_name,
                'width': self.image_size,
                'height': self.image_size,
            }
            new_images.append(new_img)

            crop_xyxy = crop_result.crop_xyxy
            for ann in anns_by_img.get(img_item['id'], []):
                remapped = remap_bbox_to_crop(
                    ann['bbox'], crop_xyxy, image.shape[:2], dst_shape=(self.image_size, self.image_size)
                )
                if remapped is None:
                    continue
                x, y, w, h = remapped
                if w < 2 or h < 2:
                    continue
                new_ann = dict(ann)
                new_ann['id'] = ann_id
                new_ann['bbox'] = [round(v, 4) for v in remapped]
                new_ann['area'] = round(w * h, 4)
                new_ann['iscrowd'] = int(ann.get('iscrowd', 0))
                new_anns.append(new_ann)
                ann_id += 1

        new_coco = {
            'info': coco.get('info', {}),
            'licenses': coco.get('licenses', []),
            'images': new_images,
            'annotations': new_anns,
            'categories': coco['categories'],
        }
        with open(out_ann_path, 'w', encoding='utf-8') as f:
            json.dump(new_coco, f, indent=2, ensure_ascii=False)
        print(f'[OK] 输出: {out_ann_path}')
