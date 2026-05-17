import os
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


class StanfordBackgroundDataset(Dataset):
    def __init__(
        self,
        root: str,
        split: str,
        image_size: Tuple[int, int] = (256, 256),
        val_ratio: float = 0.2,
    ) -> None:
        self.root = Path(root)
        self.split = split
        self.image_size = image_size

        img_dir = self.root / "images"
        mask_dir = self.root / "masks"
        if not img_dir.exists() or not mask_dir.exists():
            raise FileNotFoundError(
                f"请将 StanfordBackgroundDataset 组织为 {img_dir} 与 {mask_dir} 目录"
            )

        names = sorted([p.stem for p in img_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}])
        if len(names) == 0:
            raise RuntimeError("未发现可用图像，请检查数据路径")

        split_idx = int(len(names) * (1 - val_ratio))
        if split == "train":
            self.names = names[:split_idx]
        elif split == "val":
            self.names = names[split_idx:]
        else:
            raise ValueError("split 必须是 train 或 val")

        self.img_dir = img_dir
        self.mask_dir = mask_dir

    def __len__(self) -> int:
        return len(self.names)

    def _read_image(self, name: str) -> np.ndarray:
        for ext in [".jpg", ".jpeg", ".png"]:
            path = self.img_dir / f"{name}{ext}"
            if path.exists():
                image = cv2.imread(str(path), cv2.IMREAD_COLOR)
                if image is None:
                    raise RuntimeError(f"图像读取失败: {path}")
                return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        raise FileNotFoundError(f"未找到图像文件: {name}")

    def _read_mask(self, name: str) -> np.ndarray:
        for ext in [".png", ".jpg"]:
            path = self.mask_dir / f"{name}{ext}"
            if path.exists():
                mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
                if mask is None:
                    raise RuntimeError(f"标签读取失败: {path}")
                return mask
        raise FileNotFoundError(f"未找到掩码文件: {name}")

    def __getitem__(self, idx: int):
        name = self.names[idx]
        image = self._read_image(name)
        mask = self._read_mask(name)

        image = cv2.resize(image, self.image_size, interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, self.image_size, interpolation=cv2.INTER_NEAREST)

        image = image.astype(np.float32) / 255.0
        image = (image - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
        image = np.transpose(image, (2, 0, 1))

        image_tensor = torch.from_numpy(image).float()
        mask_tensor = torch.from_numpy(mask).long()

        return image_tensor, mask_tensor
