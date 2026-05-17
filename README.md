# HW2 深度学习与空间智能

本项目按课程作业要求实现 3 个任务：

1. **任务1：102 Flowers 分类微调**（ImageNet 预训练 + 微调 + 注意力机制）
2. **任务2：道路场景目标检测与多目标跟踪**（YOLOv8 + Tracking + 越线计数）
3. **任务3：从零实现 U-Net 语义分割**（CE/Dice/CE+Dice 对比）

## 目录结构

```text
HW2/
├── task1_classification/
│   ├── models/attention.py
│   ├── train.py
│   ├── evaluate.py
│   └── sweep.py
├── task2_detection_tracking/
│   ├── train_yolov8.py
│   ├── track_and_count.py
│   └── occlusion_analysis.py
├── task3_segmentation/
│   ├── datasets/stanford_bg.py
│   ├── models/unet.py
│   ├── losses.py
│   └── train.py
├── scripts/
│   ├── run_task1.sh
│   ├── run_task2.sh
│   └── run_task3.sh
├── requirements.txt
└── HW2_作业报告.md
```

## 环境配置

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r /workspace/HW2/requirements.txt
```

## 数据准备

### 任务1：Flowers102
`train.py` 会自动下载 `torchvision.datasets.Flowers102`。

### 任务2：RoadVehicleImagesDataset
请将数据按 YOLO 格式整理，并提供 `data.yaml`，示例：

```yaml
path: /workspace/HW2/data/road_vehicle
train: images/train
val: images/val
test: images/test
names:
  0: car
  1: bus
  2: truck
  3: bike
```

### 任务3：StanfordBackgroundDataset
按如下结构放置：

```text
/workspace/HW2/data/stanford_bg/
├── images/
│   ├── xxx.jpg
│   └── ...
└── masks/
    ├── xxx.png
    └── ...
```

要求图像与掩码同名（不同后缀可）。

## 运行方式

### 任务1：分类微调

```bash
/workspace/HW2/scripts/run_task1.sh
```

手动对照实验示例：

```bash
# 预训练微调 baseline
python /workspace/HW2/task1_classification/train.py --model resnet18 --pretrained --finetune_backbone

# 随机初始化消融
python /workspace/HW2/task1_classification/train.py --model resnet18

# 注意力模块（SE / CBAM）
python /workspace/HW2/task1_classification/train.py --model resnet18_se --pretrained --finetune_backbone
python /workspace/HW2/task1_classification/train.py --model resnet18_cbam --pretrained --finetune_backbone

# 轻量级ViT
python /workspace/HW2/task1_classification/train.py --model vit_tiny --pretrained --finetune_backbone
```

### 任务2：检测 + 跟踪 + 越线计数

```bash
/workspace/HW2/scripts/run_task2.sh
```

遮挡片段导出（3~4 帧）：

```bash
python /workspace/HW2/task2_detection_tracking/occlusion_analysis.py \
  --model /workspace/HW2/outputs/task2/yolo_road/weights/best.pt \
  --video /workspace/HW2/data/road_vehicle/test_video.mp4 \
  --start_frame 300 --num_frames 4 \
  --output_dir /workspace/HW2/outputs/task2/occlusion_frames
```

### 任务3：U-Net 分割训练

```bash
/workspace/HW2/scripts/run_task3.sh
```

损失函数对比实验：

```bash
python /workspace/HW2/task3_segmentation/train.py --data_root /workspace/HW2/data/stanford_bg --loss_type ce
python /workspace/HW2/task3_segmentation/train.py --data_root /workspace/HW2/data/stanford_bg --loss_type dice
python /workspace/HW2/task3_segmentation/train.py --data_root /workspace/HW2/data/stanford_bg --loss_type ce_dice
```

## 输出结果说明

- 任务1：`metrics.csv`、`best.pth`
- 任务2：`tracking.mp4`、`tracking_log.csv`、`occlusion_frames/`
- 任务3：`metrics.csv`、`best.pth`

建议使用 wandb 或 swanlab 记录训练曲线，并将截图粘贴到 `HW2_作业报告.md`。
