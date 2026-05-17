#!/usr/bin/env python
import cv2
import glob
from pathlib import Path

def create_video_from_images(image_dir, output_video, fps=10):
    """从图片目录创建视频"""
    image_dir = Path(image_dir)

    # 获取所有图片并排序
    image_files = sorted(image_dir.glob("*.jpg")) + sorted(image_dir.glob("*.png"))

    if not image_files:
        print(f"未找到图片文件在 {image_dir}")
        return

    # 读取第一张图片获取尺寸
    first_img = cv2.imread(str(image_files[0]))
    if first_img is None:
        print(f"无法读取图片: {image_files[0]}")
        return

    height, width = first_img.shape[:2]

    # 创建视频写入器
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

    # 写入所有图片
    for img_path in image_files:
        img = cv2.imread(str(img_path))
        if img is not None:
            # 如果需要调整大小，可以在这里调整
            # img = cv2.resize(img, (width, height))
            writer.write(img)

    writer.release()
    print(f"视频已创建: {output_video}")
    print(f"帧数: {len(image_files)}, 尺寸: {width}x{height}")

if __name__ == "__main__":
    image_dir = "/workspace/HW2/data/trafic_data/valid/images"
    output_video = "/workspace/HW2/data/trafic_data/test_video.mp4"
    create_video_from_images(image_dir, output_video)