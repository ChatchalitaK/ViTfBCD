"""
explore_pata.py -  Visualize and inspect the BreakHis dataset
Run: python explore_data.py  --data_dir ./data/BreakHis_v1/Histology_slides/breast
"""

import argparse
import random
from pathlib import Path
from collections import Counter

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from PIL import Image

# Import from dataset.py
from src.dataset import (
    BreakHisDataset,
    build_dataloaders,
    build_stain_normalizer,
    build_bach_loader,
    get_transforms,
    IDX_TO_BINARY,
    IDX_TO_SUBTYPE,
    BENIGN_SUBTYPES,
)


# basic dataset statistics
def print_stats(root_dir: str, magnification: str = "all"):
    print("\n" + "=" *60)
    print("DATASET STATISTICS")
    print("=" * 60)

    for mode in ["binary", "multiclass"]:
        print(f"\n- Mode: {mode.upper()} | Magnification: {magnification} ---")
        for splits in ["train", "val", "test"]:
            ds = BreakHisDataset(
                root_dir, magnification=magnification,
                mode=mode, split=splits,
                transform=get_transforms("test") 
            )
            counts = ds.class_counts
            print(f"  [{splits.upper():5s}] total={len(ds):5d} | ", end="")
            for idx, name in (IDX_TO_BINARY if mode == "binary" else IDX_TO_SUBTYPE).items():
                print(f"{name}: {counts.get(idx, 0):4d}", end=" ")
            print()

# ─────────────────────────────────────────────────────────────────────────────
# 2.  PLOT: class distribution bar chart
# ─────────────────────────────────────────────────────────────────────────────
def plot_class_distribution(root_dir: str, magnification: str = "all",
                            save_path: str = "outputs/class_distribution.png"):
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    fig.suptitle(f"BreakHis Class Distribution  (mag={magnification})", fontsize=14, fontweight="bold")

    for ax, mode, idx_map in zip(
        axes,
        ["binary", "multiclass"],
        [IDX_TO_BINARY, IDX_TO_SUBTYPE],
    ):
        splits_counts = {}
        colors_map = {"train": "#4C72B0", "val": "#DD8452", "test": "#55A868"}

        for splits in ["train", "val", "test"]:
            ds = BreakHisDataset(root_dir, magnification, mode, splits,
                                 transform=get_transforms("test"))
            splits_counts[splits] = ds.class_counts

        class_names = list(idx_map.values())
        x = np.arange(len(class_names))
        width = 0.25

        for i, (splits, color) in enumerate(colors_map.items()):
            vals = [splits_counts[splits].get(idx, 0) for idx in idx_map]
            bars = ax.bar(x + i * width, vals, width, label=splits.capitalize(), color=color, alpha=0.85)
            for bar, val in zip(bars, vals):
                if val > 0:
                    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5,
                            str(val), ha="center", va="bottom", fontsize=7)

        # Color-code benign vs malignant in multiclass
        if mode == "multiclass":
            for tick, name in zip(ax.get_xticklabels(), class_names):
                tick.set_color("steelblue" if name in BENIGN_SUBTYPES else "tomato")

        ax.set_xticks(x + width)
        ax.set_xticklabels([n.replace("_", "\n") for n in class_names], fontsize=8)
        ax.set_title(f"{mode.capitalize()} Classification", fontsize=12)
        ax.set_ylabel("Number of Images")
        ax.legend()
        ax.grid(axis="y", alpha=0.3)

        if mode == "multiclass":
            b_patch = mpatches.Patch(color="steelblue", label="Benign (label color)")
            m_patch = mpatches.Patch(color="tomato",    label="Malignant (label color)")
            ax.legend(handles=ax.get_legend().legend_handles + [b_patch, m_patch])

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"\n[Saved] {save_path}")
    plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# 3.  PLOT: sample images grid  (raw, before augmentation)
# ─────────────────────────────────────────────────────────────────────────────
def plot_sample_images(root_dir: str, magnification: str = "all",
                       n_per_class: int = 3,
                       save_path: str = "outputs/sample_images.png"):
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)

    ds = BreakHisDataset(
        root_dir, magnification, mode="multiclass", 
        split="train",   # 🌟 FIX THIS: Change 'splits' to 'split'
        transform=None,  # raw PIL images, no transform
    )      

    # Gather n_per_class images per subtype
    buckets: dict = {i: [] for i in range(8)}
    random.seed(0)
    indices = list(range(len(ds)))
    random.shuffle(indices)

    for idx in indices:
        path, label = ds.samples[idx]
        if len(buckets[label]) < n_per_class:
            buckets[label].append(path)
        if all(len(v) == n_per_class for v in buckets.values()):
            break

    class_names = list(IDX_TO_SUBTYPE.values())
    n_classes = 8
    fig, axes = plt.subplots(n_classes, n_per_class, figsize=(n_per_class * 3, n_classes * 3))
    fig.suptitle(f"BreakHis Sample Images  (mag={magnification})", fontsize=14, fontweight="bold")

    for row, (class_idx, paths) in enumerate(buckets.items()):
        name = class_names[class_idx]
        is_benign = name in BENIGN_SUBTYPES
        for col, path in enumerate(paths):
            ax = axes[row][col]
            ax.imshow(Image.open(path).convert("RGB"))
            ax.axis("off")
            if col == 0:
                color = "steelblue" if is_benign else "tomato"
                tag   = "BENIGN" if is_benign else "MALIGNANT"
                ax.set_ylabel(f"{name.replace('_',' ').title()}\n({tag})",
                            color=color, fontsize=8, fontweight="bold",
                            rotation=0, labelpad=90, va="center")

    plt.tight_layout()
    fig.savefig(save_path, dpi=120, bbox_inches="tight")
    print(f"[Saved] {save_path}")
    plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# 4.  PLOT: augmentation effect  (same image, multiple augmented versions)
# ─────────────────────────────────────────────────────────────────────────────
def plot_augmentation(root_dir: str, magnification: str = "all",
                    n_versions: int = 6,
                    save_path: str = "outputs/augmentation_effect.png"):
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)

    ds_raw = BreakHisDataset(root_dir, magnification, "multiclass", "train", transform=None)
    ds_aug = BreakHisDataset(root_dir, magnification, "multiclass", "train",
                            transform=get_transforms("train"))

    # Pick one sample
    sample_idx = 0
    path, label = ds_raw.samples[sample_idx]
    class_name  = IDX_TO_SUBTYPE[label].replace("_", " ").title()

    orig = Image.open(path).convert("RGB").resize((384, 384))

    # Denormalize helper
    mean = np.array([0.485, 0.456, 0.406])
    std  = np.array([0.229, 0.224, 0.225])

    def denorm(tensor):
        img = tensor.permute(1, 2, 0).numpy()
        img = img * std + mean
        return np.clip(img, 0, 1)

    fig, axes = plt.subplots(1, n_versions + 1, figsize=((n_versions + 1) * 3, 3.5))
    fig.suptitle(f"Augmentation Effect  —  {class_name}", fontsize=13, fontweight="bold")

    axes[0].imshow(orig)
    axes[0].set_title("Original", fontsize=10, fontweight="bold")
    axes[0].axis("off")

    for i in range(n_versions):
        tensor, _ = ds_aug[sample_idx]    # re-samples augmentation randomly
        axes[i + 1].imshow(denorm(tensor))
        axes[i + 1].set_title(f"Aug {i+1}", fontsize=10)
        axes[i + 1].axis("off")

    plt.tight_layout()
    fig.savefig(save_path, dpi=120, bbox_inches="tight")
    print(f"[Saved] {save_path}")
    plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# 5.  PRINT: one batch from DataLoader  (shape check)
# ─────────────────────────────────────────────────────────────────────────────
def check_dataloader(root_dir: str, magnification: str = "all"):
    print("\n── DataLoader Batch Shape Check ──")
    config = {
        "magnification": magnification,
        "mode": "binary",
        "batch_size": 8,
        "num_workers": 0,
        "image_size": 384,
        "seed": 42,
    }
    train_loader, val_loader, test_loader = build_dataloaders(root_dir, config)
    images, labels = next(iter(train_loader))
    print(f"  images shape : {images.shape}")   # [8, 3, 384, 384]
    print(f"  labels shape : {labels.shape}")   # [8]
    print(f"  label values : {labels.tolist()}")
    print(f"  pixel range  : [{images.min():.2f}, {images.max():.2f}]")
    print(f"  train batches: {len(train_loader)}")
    print(f"  val   batches: {len(val_loader)}")
    print(f"  test  batches: {len(test_loader)}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", type=str,
                default="./data/BreaKHis_v1/histology_slides/breast")
    p.add_argument("--magnification", type=str, default="all",
                choices=["40X", "100X", "200X", "400X", "all"])
    p.add_argument("--output_dir", type=str, default="./outputs")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    print_stats(args.data_dir, args.magnification)

    plot_class_distribution(
        args.data_dir, args.magnification,
        save_path=f"{args.output_dir}/class_distribution.png",
    )
    plot_sample_images(
        args.data_dir, args.magnification,
        save_path=f"{args.output_dir}/sample_images.png",
    )
    plot_augmentation(
        args.data_dir, args.magnification,
        save_path=f"{args.output_dir}/augmentation_effect.png",
    )
    check_dataloader(args.data_dir, args.magnification)

    print("\nDone! Check outputs/ folder for all plots.")