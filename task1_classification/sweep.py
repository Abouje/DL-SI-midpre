import argparse
import itertools
import os
import subprocess


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Task1 超参数网格实验")
    parser.add_argument("--python", type=str, default="python")
    parser.add_argument("--train_script", type=str, default="train.py")
    parser.add_argument("--save_root", type=str, default="./sweeps")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(args.save_root, exist_ok=True)

    epoch_space = [20, 30]
    base_lr_space = [5e-5, 1e-4]
    head_lr_space = [5e-4, 1e-3]

    for i, (epochs, base_lr, head_lr) in enumerate(
        itertools.product(epoch_space, base_lr_space, head_lr_space), start=1
    ):
        save_dir = os.path.join(args.save_root, f"exp_{i:02d}_e{epochs}_bl{base_lr}_hl{head_lr}")
        cmd = [
            args.python,
            args.train_script,
            "--model",
            "resnet18",
            "--pretrained",
            "--finetune_backbone",
            "--epochs",
            str(epochs),
            "--base_lr",
            str(base_lr),
            "--head_lr",
            str(head_lr),
            "--save_dir",
            save_dir,
        ]
        print("Running:", " ".join(cmd))
        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
