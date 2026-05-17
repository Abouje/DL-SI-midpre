#!/usr/bin/env bash
set -e

python /workspace/HW2/task3_segmentation/train.py \
  --data_root /workspace/HW2/data/stanford_bg \
  --save_dir /workspace/HW2/outputs/task3/ce_dice \
  --num_classes 8 \
  --epochs 60 \
  --batch_size 8 \
  --lr 1e-3 \
  --loss_type ce_dice \
  --ce_weight 1.0 \
  --dice_weight 1.0
