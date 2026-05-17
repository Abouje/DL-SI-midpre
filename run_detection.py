#!/usr/bin/env python
import argparse
import csv
import os
from pathlib import Path

import cv2
from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Task2: 在验证集图片上运行检测")
    parser.add_argument("--model", type=str, required=True, help="训练后的检测权重路径")
    parser.add_argument("--images_dir", type=str, required=True, help="验证集图片目录")
    parser.add_argument("--output_dir", type=str, default="outputs/task2/detection_results")
    parser.add_argument("--conf", type=float, default=0.25, help="置信度阈值")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    model = YOLO(args.model)

    # 获取所有图片
    images_dir = Path(args.images_dir)
    image_files = sorted(images_dir.glob("*.jpg")) + sorted(images_dir.glob("*.png"))

    print(f"找到 {len(image_files)} 张图片")

    # CSV 输出
    csv_path = os.path.join(args.output_dir, "detection_results.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        csv_writer = csv.writer(f)
        csv_writer.writerow(["image", "count", "boxes"])

        for img_path in image_files:
            img = cv2.imread(str(img_path))
            if img is None:
                continue

            # 运行检测
            results = model(img, conf=args.conf, verbose=False)[0]

            boxes = []
            if results.boxes is not None:
                for box in results.boxes:
                    cls_id = int(box.cls[0])
                    conf = float(box.conf[0])
                    boxes.append(f"{cls_id}:{conf:.3f}")

            print(f"{img_path.name}: {len(boxes)} 个检测")
            csv_writer.writerow([img_path.name, len(boxes), ";".join(boxes)])

    print(f"检测结果已保存到: {csv_path}")


if __name__ == "__main__":
    main()