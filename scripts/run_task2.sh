#!/usr/bin/env bash
set -e

# 1) 训练检测模型（需提前准备 yolov8 data yaml）
python /workspace/HW2/task2_detection_tracking/train_yolov8.py \
  --data /workspace/HW2/data/road_vehicle/data.yaml \
  --model yolov8n.pt \
  --epochs 100 \
  --imgsz 640 \
  --batch 16 \
  --project /workspace/HW2/outputs/task2 \
  --name yolo_road

# 2) 视频跟踪与越线计数
python /workspace/HW2/task2_detection_tracking/track_and_count.py \
  --model /workspace/HW2/outputs/task2/yolo_road/weights/best.pt \
  --video /workspace/HW2/data/road_vehicle/test_video.mp4 \
  --output /workspace/HW2/outputs/task2/tracking.mp4 \
  --log_csv /workspace/HW2/outputs/task2/tracking_log.csv \
  --line 100 320 1100 320
