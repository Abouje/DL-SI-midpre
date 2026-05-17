import os, csv, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

BASE = "/workspace/HW2"
OUT1 = os.path.join(BASE, "outputs/task1")
OUT2 = os.path.join(BASE, "outputs/task2")
OUT3 = os.path.join(BASE, "outputs/task3")

def read_csv(path, *keys):
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        import csv as _csv
        for row in _csv.DictReader(f):
            rows.append({k: float(row[k]) for k in keys if k in row})
    return {k: [r[k] for r in rows if k in r] for k in keys}

def plot_task1():
    exp_cfgs = [
        ("Exp-1 (20ep 5e-5/5e-4)","baseline_pretrained_20_5e5"),
        ("Exp-2 (20ep 1e-4/1e-3)","exp2"),
        ("Exp-3 (30ep 5e-5/5e-4)","exp3"),
        ("Exp-4 (30ep 1e-4/1e-3)","baseline_pretrained_30"),
        ("Random Init","random_init"),
        ("ResNet-18+SE","baseline_pretrained_se"),
        ("ResNet-18+CBAM","cbam_pretrained"),
        ("ViT-Tiny (pretrained)","vit_tiny_pretrained"),
    ]
    fig, ax = plt.subplots(figsize=(11,6))
    for label, folder in exp_cfgs:
        p = os.path.join(OUT1, folder, "metrics.csv")
        if not os.path.exists(p): continue
        d = read_csv(p, "epoch", "val_acc")
        ax.plot(d["epoch"], [v*100 for v in d["val_acc"]], marker=".", label=label)
    ax.set_xlabel("Epoch"); ax.set_ylabel("Val Accuracy (%)")
    ax.set_title("Task1: Validation Accuracy"); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(OUT1, "val_acc_curves.png"), dpi=150); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for label, folder in exp_cfgs[:5]:
        p = os.path.join(OUT1, folder, "metrics.csv")
        if not os.path.exists(p): continue
        d = read_csv(p, "epoch", "train_loss", "val_loss")
        axes[0].plot(d["epoch"], d["train_loss"], marker=".", label=label)
        axes[1].plot(d["epoch"], d["val_loss"], marker=".", label=label)
    for ax, t in zip(axes, ["Train Loss", "Val Loss"]):
        ax.set_xlabel("Epoch"); ax.set_ylabel("Loss"); ax.set_title(f"Task1: {t}")
        ax.legend(fontsize=7); ax.grid(True, alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(OUT1, "loss_curves.png"), dpi=150); plt.close(fig)
    print("Task1 plots saved")

def plot_task2():
    p = os.path.join(OUT2, "yolo_road/results.csv")
    if not os.path.exists(p): print("No task2 csv"); return
    d = read_csv(p, "epoch", "metrics/mAP50(B)", "metrics/mAP50-95(B)", "train/box_loss", "train/cls_loss", "val/box_loss", "val/cls_loss")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(d["epoch"], [v*100 for v in d["metrics/mAP50(B)"]], marker=".", label="mAP@50")
    axes[0].plot(d["epoch"], [v*100 for v in d["metrics/mAP50-95(B)"]], marker=".", label="mAP@50-95")
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("mAP (%)"); axes[0].set_title("Task2: YOLOv8 mAP Curves")
    axes[0].legend(); axes[0].grid(True, alpha=0.3)
    axes[1].plot(d["epoch"], d["train/box_loss"], label="train box")
    axes[1].plot(d["epoch"], d["train/cls_loss"], label="train cls")
    axes[1].plot(d["epoch"], d["val/box_loss"], linestyle="--", label="val box")
    axes[1].plot(d["epoch"], d["val/cls_loss"], linestyle="--", label="val cls")
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Loss"); axes[1].set_title("Task2: Loss Curves")
    axes[1].legend(fontsize=8); axes[1].grid(True, alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(OUT2, "training_curves.png"), dpi=150); plt.close(fig)
    print("Task2 plots saved")

def plot_task3():
    exps = [("CE Loss","ce"),("Dice Loss","dice"),("CE+Dice","ce_dice")]
    colors = ["#4C72B0","#DD8452","#55A868"]
    fig = plt.figure(figsize=(18,5))
    gs = gridspec.GridSpec(1,3,figure=fig)
    axes = [fig.add_subplot(gs[i]) for i in range(3)]
    for (label,folder),c in zip(exps,colors):
        p = os.path.join(OUT3, folder, "metrics.csv")
        if not os.path.exists(p): continue
        d = read_csv(p, "epoch", "train_loss", "val_loss", "val_miou")
        axes[0].plot(d["epoch"],d["train_loss"],color=c,marker=".",label=label)
        axes[1].plot(d["epoch"],d["val_loss"],color=c,marker=".",label=label)
        vm = [v*100 for v in d["val_miou"]]
        axes[2].plot(d["epoch"],vm,color=c,marker=".",label=label)
        bv=max(vm); be=d["epoch"][vm.index(bv)]
        axes[2].annotate(f"{bv:.2f}%",(be,bv),xytext=(3,3),textcoords="offset points",fontsize=7,color=c)
    for ax,t,y in zip(axes,["Train Loss","Val Loss","Val mIoU (%)"],["Loss","Loss","mIoU (%)"]):
        ax.set_xlabel("Epoch"); ax.set_ylabel(y); ax.set_title(f"Task3: {t}"); ax.legend(fontsize=8); ax.grid(True,alpha=0.3)
    fig.suptitle("Task3: U-Net Loss Ablation",fontsize=13,fontweight="bold")
    fig.tight_layout(); fig.savefig(os.path.join(OUT3, "training_curves.png"), dpi=150); plt.close(fig)
    print("Task3 plots saved")

plot_task1(); plot_task2(); plot_task3()
print("All done!")
