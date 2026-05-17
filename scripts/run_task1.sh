#!/usr/bin/env bash
set -e
export CUDA_VISIBLE_DEVICES=7
python /workspace/zhouyangjie/HW2/task1_classification/train.py \
  --data_root /workspace/zhouyangjie/HW2/data/flowers102 \
  --save_dir /workspace/zhouyangjie/HW2/outputs/task1/baseline_pretrained_se \
  --model resnet18_se \
  --pretrained \
  --finetune_backbone \
  --epochs 30 \
  --batch_size 32 \
  --base_lr 1e-4 \
  --head_lr 1e-3
