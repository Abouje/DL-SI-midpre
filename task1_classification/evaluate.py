import argparse

import torch
import torch.nn as nn

from train import build_dataloaders, build_model, run_one_epoch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Task1 模型评估")
    parser.add_argument("--data_root", type=str, default="./data/flowers102")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    train_args = ckpt["args"]
    model = build_model(train_args["model"], train_args["num_classes"], pretrained=False)
    model.load_state_dict(ckpt["model"])
    model.to(device)

    loaders = build_dataloaders(args.data_root, args.batch_size, args.num_workers)
    criterion = nn.CrossEntropyLoss()

    model.eval()
    with torch.no_grad():
        metrics = run_one_epoch(model, loaders[args.split], criterion, device)

    print(f"{args.split} loss={metrics.loss:.4f}, acc={metrics.acc:.4f}")


if __name__ == "__main__":
    main()
