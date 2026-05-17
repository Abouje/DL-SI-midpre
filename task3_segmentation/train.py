import argparse
import csv
import os
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from datasets.stanford_bg import StanfordBackgroundDataset
from losses import DiceLoss
from models.unet import UNet


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Task3: U-Net 从零训练")
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--save_dir", type=str, default="outputs")
    parser.add_argument("--num_classes", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--loss_type", type=str, default="ce", choices=["ce", "dice", "ce_dice"])
    parser.add_argument("--ce_weight", type=float, default=1.0)
    parser.add_argument("--dice_weight", type=float, default=1.0)
    parser.add_argument("--image_size", type=int, nargs=2, default=[256, 256])
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def fast_hist(pred: np.ndarray, target: np.ndarray, num_classes: int) -> np.ndarray:
    mask = (target >= 0) & (target < num_classes)
    hist = np.bincount(
        num_classes * target[mask].astype(int) + pred[mask], minlength=num_classes**2
    ).reshape(num_classes, num_classes)
    return hist


def compute_miou(hist: np.ndarray) -> float:
    iou = np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist) + 1e-9)
    return float(np.nanmean(iou))


def train_or_eval(
    model: nn.Module,
    loader: DataLoader,
    ce_loss: nn.Module,
    dice_loss: nn.Module,
    loss_type: str,
    device: torch.device,
    optimizer: torch.optim.Optimizer = None,
    num_classes: int = 8,
    ce_weight: float = 1.0,
    dice_weight: float = 1.0,
) -> Dict[str, float]:
    is_train = optimizer is not None
    model.train(is_train)

    total_loss = 0.0
    hist = np.zeros((num_classes, num_classes), dtype=np.float64)
    total = 0

    for images, masks in loader:
        images, masks = images.to(device), masks.to(device)

        logits = model(images)

        ce = ce_loss(logits, masks)
        dice = dice_loss(logits, masks)
        if loss_type == "ce":
            loss = ce
        elif loss_type == "dice":
            loss = dice
        else:
            loss = ce_weight * ce + dice_weight * dice

        if is_train:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        preds = torch.argmax(logits, dim=1)
        total_loss += loss.item() * masks.size(0)
        total += masks.size(0)

        hist += fast_hist(preds.detach().cpu().numpy(), masks.detach().cpu().numpy(), num_classes)

    return {
        "loss": total_loss / total,
        "miou": compute_miou(hist),
    }


def main() -> None:
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_set = StanfordBackgroundDataset(
        root=args.data_root,
        split="train",
        image_size=tuple(args.image_size),
    )
    val_set = StanfordBackgroundDataset(
        root=args.data_root,
        split="val",
        image_size=tuple(args.image_size),
    )

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    model = UNet(in_channels=3, num_classes=args.num_classes).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    ce_loss = nn.CrossEntropyLoss()
    dice_loss = DiceLoss()

    best_miou = -1.0
    history = []

    for epoch in range(1, args.epochs + 1):
        train_metrics = train_or_eval(
            model,
            train_loader,
            ce_loss,
            dice_loss,
            args.loss_type,
            device,
            optimizer=optimizer,
            num_classes=args.num_classes,
            ce_weight=args.ce_weight,
            dice_weight=args.dice_weight,
        )

        with torch.no_grad():
            val_metrics = train_or_eval(
                model,
                val_loader,
                ce_loss,
                dice_loss,
                args.loss_type,
                device,
                optimizer=None,
                num_classes=args.num_classes,
                ce_weight=args.ce_weight,
                dice_weight=args.dice_weight,
            )

        scheduler.step()

        row = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_miou": train_metrics["miou"],
            "val_loss": val_metrics["loss"],
            "val_miou": val_metrics["miou"],
            "lr": scheduler.get_last_lr()[0],
        }
        history.append(row)

        print(
            f"[Epoch {epoch:03d}] train_mIoU={train_metrics['miou']:.4f} "
            f"val_mIoU={val_metrics['miou']:.4f}"
        )

        if val_metrics["miou"] > best_miou:
            best_miou = val_metrics["miou"]
            torch.save(
                {
                    "model": model.state_dict(),
                    "args": vars(args),
                    "best_val_miou": best_miou,
                    "epoch": epoch,
                },
                os.path.join(args.save_dir, "best.pth"),
            )

    with open(os.path.join(args.save_dir, "metrics.csv"), "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)

    print(f"训练完成，最佳验证 mIoU={best_miou:.4f}")


if __name__ == "__main__":
    main()
