import argparse
import csv
import os
from dataclasses import dataclass
from typing import Dict, Iterable, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms

from models.attention import CBAM, SEBlock


@dataclass
class EpochMetrics:
    loss: float
    acc: float


class ResNetWithAttention(nn.Module):
    def __init__(self, backbone: nn.Module, attention: nn.Module) -> None:
        super().__init__()
        self.conv1 = backbone.conv1
        self.bn1 = backbone.bn1
        self.relu = backbone.relu
        self.maxpool = backbone.maxpool
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4
        self.attention = attention
        self.avgpool = backbone.avgpool
        self.fc = backbone.fc

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.attention(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


def build_dataloaders(data_root: str, batch_size: int, num_workers: int) -> Dict[str, DataLoader]:
    train_tf = transforms.Compose(
        [
            transforms.Resize((256, 256)),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    eval_tf = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    train_set = datasets.Flowers102(root=data_root, split="train", transform=train_tf, download=True)
    val_set = datasets.Flowers102(root=data_root, split="val", transform=eval_tf, download=True)
    test_set = datasets.Flowers102(root=data_root, split="test", transform=eval_tf, download=True)

    return {
        "train": DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True),
        "val": DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True),
        "test": DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True),
    }


def build_model(model_name: str, num_classes: int, pretrained: bool, **kwargs) -> nn.Module:
    name = model_name.lower()

    if name in {"resnet18", "resnet18_se", "resnet18_cbam"}:
        weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = models.resnet18(weights=weights)
    elif name in {"resnet34", "resnet34_se", "resnet34_cbam"}:
        weights = models.ResNet34_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = models.resnet34(weights=weights)
    elif name in {"vit_tiny", "swin_tiny"}:
        try:
            import timm
        except ImportError as exc:
            raise ImportError("使用 ViT/Swin 需要先安装 timm: pip install timm") from exc

        timm_name = "vit_tiny_patch16_224" if name == "vit_tiny" else "swin_tiny_patch4_window7_224"
        model = timm.create_model(timm_name, pretrained=False, num_classes=num_classes)
        local_path = kwargs.get("local_weights", None)
        if local_path and os.path.isfile(local_path):
            import torch as _torch
            sd = _torch.load(local_path, map_location="cpu", weights_only=False)
            # remove head weights (ImageNet 1000-class → task 102-class)
            sd = {k: v for k, v in sd.items() if not k.startswith("head.")}
            missing, unexpected = model.load_state_dict(sd, strict=False)
            print(f"[ViT] loaded local weights: missing={len(missing)} unexpected={len(unexpected)}")
        elif pretrained:
            model = timm.create_model(timm_name, pretrained=True, num_classes=num_classes)
        return model
    else:
        raise ValueError(f"不支持的模型: {model_name}")

    in_features = backbone.fc.in_features
    backbone.fc = nn.Linear(in_features, num_classes)

    if name.endswith("_se"):
        attention = SEBlock(channels=512 if "18" in name else 512)
        return ResNetWithAttention(backbone, attention)
    if name.endswith("_cbam"):
        attention = CBAM(channels=512)
        return ResNetWithAttention(backbone, attention)
    return backbone


def make_optimizer(
    model: nn.Module,
    base_lr: float,
    head_lr: float,
    weight_decay: float,
    finetune_backbone: bool,
) -> torch.optim.Optimizer:
    if not finetune_backbone:
        return torch.optim.AdamW(model.parameters(), lr=head_lr, weight_decay=weight_decay)

    backbone_params = []
    head_params = []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if any(key in name for key in ["fc", "head", "classifier"]):
            head_params.append(p)
        else:
            backbone_params.append(p)

    return torch.optim.AdamW(
        [
            {"params": backbone_params, "lr": base_lr},
            {"params": head_params, "lr": head_lr},
        ],
        weight_decay=weight_decay,
    )


def run_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer = None,
) -> EpochMetrics:
    train = optimizer is not None
    model.train(train)

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        logits = model(images)
        loss = criterion(logits, labels)

        if train:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        preds = logits.argmax(dim=1)
        total_correct += (preds == labels).sum().item()
        total_loss += loss.item() * labels.size(0)
        total_samples += labels.size(0)

    return EpochMetrics(loss=total_loss / total_samples, acc=total_correct / total_samples)


def evaluate(
    model: nn.Module,
    loaders: Dict[str, DataLoader],
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[EpochMetrics, EpochMetrics]:
    with torch.no_grad():
        val_metrics = run_one_epoch(model, loaders["val"], criterion, device)
        test_metrics = run_one_epoch(model, loaders["test"], criterion, device)
    return val_metrics, test_metrics


def save_metrics_csv(records: Iterable[dict], out_path: str) -> None:
    if not records:
        return
    keys = list(records[0].keys())
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(records)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Task1: 102 Flowers 分类微调")
    parser.add_argument("--data_root", type=str, default="./data")
    parser.add_argument("--save_dir", type=str, default="./outputs")
    parser.add_argument("--model", type=str, default="resnet18", help="resnet18/resnet34/resnet18_se/resnet18_cbam/vit_tiny/swin_tiny")
    parser.add_argument("--num_classes", type=int, default=102)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--base_lr", type=float, default=1e-4, help="预训练骨干学习率")
    parser.add_argument("--head_lr", type=float, default=1e-3, help="新分类头学习率")
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--pretrained", action="store_true", help="启用 ImageNet 预训练")
    parser.add_argument("--local_weights", type=str, default=None, help="本地权重文件路径（用于 ViT/Swin 离线加载）")
    parser.add_argument("--finetune_backbone", action="store_true", help="对骨干网络进行小学习率微调")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loaders = build_dataloaders(args.data_root, args.batch_size, args.num_workers)

    model = build_model(args.model, args.num_classes, pretrained=args.pretrained,
                        local_weights=args.local_weights).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = make_optimizer(
        model,
        base_lr=args.base_lr,
        head_lr=args.head_lr,
        weight_decay=args.weight_decay,
        finetune_backbone=args.finetune_backbone,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_val = 0.0
    records = []

    for epoch in range(1, args.epochs + 1):
        train_metrics = run_one_epoch(model, loaders["train"], criterion, device, optimizer)
        val_metrics, test_metrics = evaluate(model, loaders, criterion, device)
        scheduler.step()

        row = {
            "epoch": epoch,
            "train_loss": train_metrics.loss,
            "train_acc": train_metrics.acc,
            "val_loss": val_metrics.loss,
            "val_acc": val_metrics.acc,
            "test_loss": test_metrics.loss,
            "test_acc": test_metrics.acc,
            "lr": scheduler.get_last_lr()[0],
        }
        records.append(row)

        print(
            f"[Epoch {epoch:03d}] "
            f"train_acc={train_metrics.acc:.4f} val_acc={val_metrics.acc:.4f} test_acc={test_metrics.acc:.4f}"
        )

        if val_metrics.acc > best_val:
            best_val = val_metrics.acc
            ckpt_path = os.path.join(args.save_dir, "best.pth")
            torch.save(
                {
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "args": vars(args),
                    "best_val_acc": best_val,
                },
                ckpt_path,
            )

    save_metrics_csv(records, os.path.join(args.save_dir, "metrics.csv"))
    print(f"训练完成，最佳验证集准确率: {best_val:.4f}")


if __name__ == "__main__":
    main()
