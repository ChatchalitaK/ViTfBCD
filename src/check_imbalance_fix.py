import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt

from dataset import BreakHisDataset, make_weighted_sampler, IDX_TO_SUBTYPE, IDX_TO_BINARY
from model import build_model

try:
    from sklearn.metrics import confusion_matrix, f1_score, classification_report
    _SKLEARN_AVAILABLE = True
except ImportError:
    _SKLEARN_AVAILABLE = False
    print("[Warning] scikit-learn not installed — confusion matrix step will be skipped.")
    print("Install with: pip install scikit-learn")


# ── CONFIG — edit these for your setup ──────────────────────────────────────
CONFIG = {
    "data_dir": "/home/user/Proj-Ploy/vit_breast_cancer/data/BreaKHis_v1/histology_slides/breast/",
    "checkpoint": "/home/user/Proj-Ploy/vit_breast_cancer/outputs/best_model.pt",
    "history": "/home/user/Proj-Ploy/vit_breast_cancer/outputs/history.json",
    "output_dir": "/home/user/Proj-Ploy/vit_breast_cancer/outputs/diagnostics",
    "mode": "binary",          
    "magnification": "all",
    "image_size": 384,
    "model_size": "base",
    "batch_size": 32,
    "sampler_beta": 0.999,
    "eval_split": "test",         
}


def class_names_for(mode: str):
    return list(IDX_TO_BINARY.values()) if mode == "binary" else list(IDX_TO_SUBTYPE.values())


# ── 1. Effective sampling distribution ──────────────────────────────────────
def plot_sampling_distribution(cfg: dict, output_dir: Path):
    train_ds = BreakHisDataset(
        cfg["data_dir"], magnification=cfg["magnification"], mode=cfg["mode"],
        split="train", image_size=cfg["image_size"],
    )
    sampler = make_weighted_sampler(train_ds, beta=cfg["sampler_beta"])

    raw_counts = train_ds.class_counts
    sampled_labels = [train_ds.samples[i][1] for i in sampler]
    sampled_counts = Counter(sampled_labels)

    names = class_names_for(cfg["mode"])
    idx_order = sorted(raw_counts.keys())
    labels = [names[i] for i in idx_order]
    raw_vals = [raw_counts.get(i, 0) for i in idx_order]
    sampled_vals = [sampled_counts.get(i, 0) for i in idx_order]

    x = np.arange(len(labels))
    width = 0.38

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(x - width / 2, raw_vals, width, label="Raw on-disk count", color="#4c72b0")
    ax.bar(x + width / 2, sampled_vals, width, label="Sampler output (1 epoch)", color="#dd8452")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel("Samples")
    ax.set_title("Raw Class Counts vs. What the Sampler Actually Feeds the Model")
    ax.legend()
    plt.tight_layout()

    out_path = output_dir / "sampling_distribution.png"
    plt.savefig(out_path)
    plt.close()
    print(f"[OK] Saved {out_path}")

    # ── Imbalance Ratio (IR) — checkable immediately, no training needed ────
    # IR = count of the biggest class / count of the smallest class.
    #   IR < 3   -> mild, usually fine as-is
    #   IR 3-10  -> moderate (BreakHis raw data is in this range)
    #   IR > 10  -> severe, needs correction
    def imbalance_ratio(vals):
        nonzero = [v for v in vals if v > 0]
        return max(nonzero) / min(nonzero) if nonzero else float("nan")

    raw_ir = imbalance_ratio(raw_vals)
    sampled_ir = imbalance_ratio(sampled_vals)

    def verdict(ir):
        if ir < 3:
            return "mild"
        elif ir < 10:
            return "moderate"
        return "severe"

    print(f"\n     Imbalance Ratio — raw on-disk data:    {raw_ir:.2f}x ({verdict(raw_ir)})")
    print(f"     Imbalance Ratio — sampler output:      {sampled_ir:.2f}x ({verdict(sampled_ir)})")
    if sampled_ir < raw_ir * 0.5:
        print("     -> Sampler is clearly flattening the imbalance. Good sign before you train.")
    elif sampled_ir < raw_ir:
        print("     -> Sampler is helping some, but not by much. Consider lowering "
              "sampler_beta (closer to 0) for a stronger correction.")
    else:
        print("     -> Sampler isn't reducing imbalance at all — check make_weighted_sampler() "
              "is actually being passed to the train DataLoader.")


# ── 2. Confusion matrix on a trained checkpoint ─────────────────────────────
def plot_confusion_matrix(cfg: dict, output_dir: Path):
    if not _SKLEARN_AVAILABLE:
        return
    ckpt_path = Path(cfg["checkpoint"])
    if not ckpt_path.exists():
        print(f"[SKIP] No checkpoint found at {ckpt_path} — train a model first.")
        return

    names = class_names_for(cfg["mode"])
    num_classes = len(names)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = build_model({
        "model_size": cfg["model_size"], "num_classes": num_classes, "pretrained": False,
    })
    if cfg["image_size"] != 224:
        model.resize_position_embeddings()
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    model.to(device).eval()

    eval_ds = BreakHisDataset(
        cfg["data_dir"], magnification=cfg["magnification"], mode=cfg["mode"],
        split=cfg["eval_split"], image_size=cfg["image_size"],
    )
    loader = torch.utils.data.DataLoader(eval_ds, batch_size=cfg["batch_size"], shuffle=False)

    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            logits = model(images)
            preds = logits.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

    cm = confusion_matrix(all_labels, all_preds, labels=list(range(num_classes)))
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1)
    macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(num_classes))
    ax.set_yticks(range(num_classes))
    ax.set_xticklabels(names, rotation=45, ha="right")
    ax.set_yticklabels(names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"Confusion Matrix ({cfg['eval_split']} set, row-normalized)\nMacro-F1: {macro_f1:.4f}")
    for i in range(num_classes):
        for j in range(num_classes):
            ax.text(j, i, f"{cm[i, j]}", ha="center", va="center",
                     color="white" if cm_norm[i, j] > 0.5 else "black", fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()

    out_path = output_dir / "confusion_matrix.png"
    plt.savefig(out_path)
    plt.close()
    print(f"[OK] Saved {out_path}")
    print(f"     Macro-F1 on {cfg['eval_split']} set: {macro_f1:.4f}")
    print("     Look at the ductal_carcinoma column — if minority classes are being")
    print("     misclassified as ductal_carcinoma, that column will be dark outside its own row.")

    # ── Per-class breakdown + recall-gap verdict ────────────────────────────
    # This is the number that actually answers "is imbalance still hurting
    # training?" -- not raw accuracy, which the majority class dominates.
    report = classification_report(
        all_labels, all_preds, target_names=names, labels=list(range(num_classes)),
        output_dict=True, zero_division=0,
    )
    support = Counter(all_labels)
    print(f"\n     {'Class':<22s} {'Recall':>8s} {'F1':>8s} {'Support':>8s}")
    recalls_with_support = []
    for i, name in enumerate(names):
        r = report[name]["recall"]
        f1c = report[name]["f1-score"]
        n = support.get(i, 0)
        print(f"     {name:<22s} {r:8.3f} {f1c:8.3f} {n:8d}")
        if n > 0:
            recalls_with_support.append((name, r))

    if recalls_with_support:
        best_cls, best_r = max(recalls_with_support, key=lambda t: t[1])
        worst_cls, worst_r = min(recalls_with_support, key=lambda t: t[1])
        gap = best_r - worst_r
        print(f"\n     Recall gap: {gap:.3f}  (best: {best_cls}={best_r:.3f}, "
              f"worst: {worst_cls}={worst_r:.3f})")
        if gap < 0.15:
            print("     -> Imbalance looks resolved: recall is fairly even across classes.")
        elif gap < 0.35:
            print("     -> Imbalance is improved but not gone: keep an eye on "
                  f"'{worst_cls}' — it's still lagging noticeably.")
        else:
            print(f"     -> Imbalance still a real problem: '{worst_cls}' recall is far "
                  f"behind '{best_cls}'. Consider raising mixup_alpha, trying "
                  "focal_alpha_mode='sqrt_inv_freq', or gathering more data/patients "
                  f"for '{worst_cls}'.")


# ── 3. Training curves from history.json (single-run or K-Fold) ────────────
def _find_fold_histories(history_path: Path):
    """
    main.py's K-Fold mode writes history to outputs/fold_1/history.json,
    outputs/fold_2/history.json, etc. -- it never touches outputs/history.json.
    So CONFIG["history"] pointing at a single top-level file will silently load
    a stale file from an old non-K-Fold run instead of the current K-Fold run.
    This looks for fold_*/history.json siblings next to the configured path
    and, if found, uses those instead.
    """
    parent = history_path.parent
    fold_dirs = sorted(parent.glob("fold_*"), key=lambda p: p.name)
    fold_histories = []
    for fd in fold_dirs:
        hp = fd / "history.json"
        if hp.exists():
            with open(hp) as f:
                fold_histories.append((fd.name, json.load(f)))
    return fold_histories


def plot_training_curves(cfg: dict, output_dir: Path):
    history_path = Path(cfg["history"])
    fold_histories = _find_fold_histories(history_path)

    if fold_histories:
        print(f"[Info] Found {len(fold_histories)} per-fold history files "
              f"({', '.join(name for name, _ in fold_histories)}) next to "
              f"{history_path} -- plotting those instead of the (likely stale) "
              "single history.json, since main.py's K-Fold mode never writes there.")
        _plot_kfold_curves(fold_histories, output_dir)
        return

    if not history_path.exists():
        print(f"[SKIP] No history.json (or fold_*/history.json) found near {history_path} "
              "-- train a model first.")
        return

    with open(history_path) as f:
        history = json.load(f)
    _plot_single_run_curves(history, output_dir)


def _plot_kfold_curves(fold_histories, output_dir: Path):
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    cmap = plt.get_cmap("tab10")

    for i, (name, history) in enumerate(fold_histories):
        color = cmap(i % 10)
        epochs = range(1, len(history["train_loss"]) + 1)
        axes[0].plot(epochs, history["val_loss"], label=name, color=color)
        axes[1].plot(epochs, history["val_acc"], label=name, color=color)
        f1_vals = [v if v is not None else np.nan for v in history.get("val_macro_f1", [])]
        if f1_vals:
            axes[2].plot(epochs, f1_vals, label=name, color=color)

    axes[0].set_title("Val Loss per Fold")
    axes[1].set_title("Val Accuracy per Fold")
    axes[2].set_title("Val Macro-F1 per Fold")
    for ax in axes:
        ax.set_xlabel("Epoch")
        ax.legend(fontsize=8)

    plt.suptitle("K-Fold Training Curves (val set per fold)")
    plt.tight_layout()

    out_path = output_dir / "training_curves.png"
    plt.savefig(out_path)
    plt.close()
    print(f"[OK] Saved {out_path}")

    best_f1s = [max(v for v in h.get("val_macro_f1", [0]) if v is not None) for _, h in fold_histories]
    print(f"     Best macro-F1 per fold: {[f'{v:.3f}' for v in best_f1s]}")
    print(f"     Mean: {np.mean(best_f1s):.3f}  Std: {np.std(best_f1s):.3f}")


def _plot_single_run_curves(history: dict, output_dir: Path):

    epochs = range(1, len(history["train_loss"]) + 1)
    has_f1 = "val_macro_f1" in history and any(v is not None for v in history["val_macro_f1"])

    fig, axes = plt.subplots(1, 3 if has_f1 else 2, figsize=(16 if has_f1 else 11, 4.5))

    axes[0].plot(epochs, history["train_loss"], label="Train Loss")
    axes[0].plot(epochs, history["val_loss"], label="Val Loss")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    axes[1].plot(epochs, history["train_acc"], label="Train Acc")
    axes[1].plot(epochs, history["val_acc"], label="Val Acc")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()

    if has_f1:
        f1_vals = [v if v is not None else np.nan for v in history["val_macro_f1"]]
        axes[2].plot(epochs, f1_vals, label="Val Macro-F1", color="#c44e52")
        axes[2].set_title("Macro-F1 (Imbalance-Aware Metric)")
        axes[2].set_xlabel("Epoch")
        axes[2].legend()

    plt.suptitle("Training Curves")
    plt.tight_layout()

    out_path = output_dir / "training_curves.png"
    plt.savefig(out_path)
    plt.close()
    print(f"[OK] Saved {out_path}")
    if has_f1:
        best_f1 = max(v for v in history["val_macro_f1"] if v is not None)
        print(f"     Best val macro-F1 across training: {best_f1:.4f}")


if __name__ == "__main__":
    output_dir = Path(CONFIG["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("ViTfBCD Imbalance-Fix Diagnostics")
    print("=" * 60)

    print("\n[1/3] Sampling distribution...")
    plot_sampling_distribution(CONFIG, output_dir)

    print("\n[2/3] Confusion matrix...")
    plot_confusion_matrix(CONFIG, output_dir)

    print("\n[3/3] Training curves...")
    plot_training_curves(CONFIG, output_dir)

    print(f"\nDone. Check {output_dir} for the plots.")