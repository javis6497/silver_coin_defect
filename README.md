# 银币表面瑕疵检测项目（COCO 标注）

本项目面向**圆形银币表面瑕疵检测**，支持：

- 缺陷位置与类别检测
- 圆币自动定位、裁切与增强
- COCO 数据集预处理
- 小样本 / 大样本 两类训练方案
- 缺陷评测指标输出（mAP、AP50、Recall、每类 AP50）
- 训练日志和可视化曲线
- Python 统一启动器（不依赖 `.bat` 作为主入口）

## 提供的 4 种算法

### 小样本方案
1. **fewshot_yolov8n**：YOLOv8n 迁移学习，适合样本少、希望快速起模
2. **fewshot_fasterrcnn**：Faster R-CNN + FPN，适合小样本下稳健训练

### 大样本方案
3. **largeset_yolov8m**：YOLOv8m，精度 / 速度平衡，推荐作为主力基线
4. **largeset_rtdetr_l**：RT-DETR-L，适合更高精度追求

## 推荐使用策略

- **小样本（每类 < 100~300 个缺陷实例）**：先跑 `fewshot_yolov8n` 和 `fewshot_fasterrcnn`
- **大样本（每类 > 500 个缺陷实例）**：优先跑 `largeset_yolov8m`，再用 `largeset_rtdetr_l` 冲精度

## 目录结构

```text
silver_coin_defect_project/
├─ configs/
├─ scripts/
├─ src/
├─ outputs/
├─ install_env.py
├─ run_pipeline.py
├─ prepare_data.py
├─ train.py
├─ infer.py
├─ evaluate.py
├─ visualize_logs.py
├─ requirements.txt
└─ README.md
```

## 数据集组织（原始 COCO）

```text
your_dataset/
├─ images/
│  ├─ train/
│  ├─ val/
│  └─ test/
└─ annotations/
   ├─ instances_train.json
   ├─ instances_val.json
   └─ instances_test.json
```

## 一键运行顺序

### 1）编辑配置
修改 `configs/*.yaml` 中的 `dataset.root_dir` 为你的数据集路径。

### 2）数据预处理（自动圆币定位 + 裁切 + 增强 + COCO/YOLO 派生）
```bash
python prepare_data.py --config configs/largeset_yolov8m.yaml
```

### 3）训练
```bash
python train.py --config configs/largeset_yolov8m.yaml
```

### 4）验证/测试评估
```bash
python evaluate.py --config configs/largeset_yolov8m.yaml
```

### 5）推理
```bash
python infer.py --config configs/largeset_yolov8m.yaml --source your_test_images
```

## 推荐运行方式（纯 Python）

### 1）安装环境
```bash
python install_env.py
```

### 2）只做预处理
```bash
python run_pipeline.py --config configs/fewshot_yolov8n.yaml --stage prepare
```

### 3）只训练
```bash
python run_pipeline.py --config configs/fewshot_yolov8n.yaml --stage train
```

### 4）训练后评估
```bash
python run_pipeline.py --config configs/fewshot_yolov8n.yaml --stage evaluate
```

### 5）推理
```bash
python run_pipeline.py --config configs/fewshot_yolov8n.yaml --stage infer --source val
```

### 6）一键全流程
```bash
python run_pipeline.py --config configs/fewshot_yolov8n.yaml --stage all
```

### 7）交互菜单模式
```bash
python run_pipeline.py
```

## 自动圆币定位与裁切增强

处理流程：

1. 读取原图
2. 自动检测圆币区域（Hough Circle + 轮廓兜底）
3. 根据圆币外接正方形自动裁切
4. 重新映射 COCO bbox
5. 可选增强：CLAHE、锐化、边缘保留

## 指标输出

输出目录示例：

```text
outputs/largeset_yolov8m/
├─ train/
├─ eval/
│  ├─ coco_metrics.json
│  ├─ per_class_ap50.csv
│  └─ confusion_stub.csv
├─ infer/
└─ curves/
```

## 训练日志与曲线

- Ultralytics 模型会自动保存 `results.csv`
- 本项目会统一再生成曲线图 PNG
- Faster R-CNN 会输出 `train_log.csv` 并生成曲线

## 说明

Ultralytics 文档显示其训练流程支持自定义数据训练，并保存训练相关结果；其文档也提供了 RT-DETR 的训练 / 验证 / 推理入口。PyTorch 官方教程与 TorchVision 文档也提供了基于预训练检测模型的微调实践，这正适合本项目中的 Faster R-CNN 小样本方案。 citeturn967843search1turn967843search0turn967843search2turn967843search8

## 你最先应该跑哪个

默认建议你第一轮先跑这两个：

1. `fewshot_yolov8n`：最快出结果
2. `largeset_yolov8m`：最稳的主力方案

如果你要我继续，我下一步可以基于你的**缺陷类别名**、**图片分辨率**、**显卡型号**，把这套配置再替你改成更贴近产线的默认参数。
