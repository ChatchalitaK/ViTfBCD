"""
finetune_on_bach.py
Light domain-adaptation fine-tune on a BACH subset, to test whether the
"weak but above chance" AUC-ROC on BACH (found via cross_validation.py /
threshold_recalibration_check.py) improves with a small amount of direct
BACH supervision -- rather than continuing to chase preprocessing-level
fixes (stain normalization, threshold tuning) that have already been shown
to top out around a couple of AUC points.

Design choices, and why:
- FROZEN, REUSABLE split: BACH is split once into finetune/held-out subsets
  and saved to bach_finetune_split.json (same "freeze the split before you
  touch the model" philosophy as benchmark_fold.py) -- so repeated runs are
  comparable and the held-out portion is never seen during fine-tuning.
- LIGHT fine-tuning, not full retraining: only the last few ViT blocks + the
  classification head are unfrozen (via model.py's existing
  freeze_backbone()), with a much lower LR on whatever backbone layers ARE
  unfrozen than on the head (via model.py's existing
  get_layered_parameters()) -- this is the same layer-wise-LR machinery
  already used for the primary model, reused here rather than reinvented.
  Full fine-tuning risks catastrophic forgetting of BreakHis features on a
  fine-tune set this small (~280 images).
- Zero-shot baseline is re-computed on the SAME held-out subset (not the
  full 400-image BACH set from earlier runs) so the before/after comparison
  is apples-to-apples.
- The fine-tuned checkpoint is saved to a SEPARATE path
  (outputs/bach_finetuned_model.pt) and never overwrites outputs/best_model.pt
  -- the primary BreakHis-only model stays canonical for every other script
  in this project (evaluate.py, run_distill.py, etc.).

Usage:
    python finetune_on_bach.py                      # uses defaults below
    python finetune_on_bach.py --finetune_frac 0.7 --epochs 15
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset

from cross_validation import CONFIG as CV_CONFIG, load_model
from src.dataset import get_transforms, build_stain_normalizer
from src.external_datasets import BACHDataset
from src.trainer import EarlyStopping

try:
    from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
    _SKLEARN_AVAILABLE = True
except ImportError:
    _SKLEARN_AVAILABLE = False


SPLIT_PATH = Path("outputs/cross_dataset/bach_finetune_split.json")
FINETUNED_CKPT_PATH = Path("outputs/bach_finetuned_model.pt")


class _SampleListDataset(Dataset):
    def __init__(self, samples, transform):
        self.samples = samples  # list of (path, label)
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        from PIL import Image
        path, label = self.samples[idx]
        image = Image.open(path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label


def build_or_load_split(bach_root: str, finetune_frac: float, seed: int, split_path: Path):
    """
    Splits BACH once (stratified by binary label) and freezes it to disk --
    every subsequent run of this script reuses the exact same split rather
    than re-randomizing, so results are comparable across fine-tuning runs
    with different hyperparameters.

    NOTE: BACH (ICIAR2018) does not ship patient IDs the way BreakHis does --
    each of the 400 images is treated as an independent unit here. If you
    know some images share a source patient/slide, group by that instead of
    doing a plain per-image split.
    """
    if split_path.exists():
        with open(split_path) as f:
            split = json.load(f)
        print(f"[OK] Loaded existing frozen split from {split_path} "
              f"({len(split['finetune'])} finetune / {len(split['held_out'])} held-out).")
        return split

    base_ds = BACHDataset(bach_root, transform=None)
    by_label = {0: [], 1: []}
    for path, label in base_ds.samples:
        by_label[label].append(path)

    rng = np.random.RandomState(seed)
    finetune_paths, held_out_paths = [], []
    for label, paths in by_label.items():
        paths = sorted(paths)  # deterministic order before shuffling
        rng.shuffle(paths)
        n_finetune = int(round(len(paths) * finetune_frac))
        finetune_paths.extend((p, label) for p in paths[:n_finetune])
        held_out_paths.extend((p, label) for p in paths[n_finetune:])

    split = {
        "finetune_frac": finetune_frac,
        "seed": seed,
        "finetune": finetune_paths,
        "held_out": held_out_paths,
    }
    split_path.parent.mkdir(parents=True, exist_ok=True)
    with open(split_path, "w") as f:
        json.dump(split, f, indent=2)
    print(f"[OK] Froze new BACH finetune split -> {split_path} "
          f"({len(finetune_paths)} finetune / {len(held_out_paths)} held-out)")
    return split


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    all_true, all_pred, all_prob = [], [], []
    for images, labels in loader:
        images = images.to(device)
        probs = F.softmax(model(images), dim=1).cpu().numpy()
        all_true.extend(labels.tolist() if torch.is_tensor(labels) else list(labels))
        all_pred.extend(probs.argmax(axis=1).tolist())
        all_prob.extend(probs[:, 1].tolist())

    acc = accuracy_score(all_true, all_pred)
    f1 = f1_score(all_true, all_pred, average="macro", zero_division=0)
    try:
        auc = roc_auc_score(all_true, all_prob)
    except Exception:
        auc = float("nan")
    return {"accuracy": float(acc), "macro_f1": float(f1), "auc_roc": float(auc)}


def finetune(model, train_loader, val_loader, device, epochs, head_lr, backbone_lr,
             weight_decay, patience):
    layered = model.get_layered_parameters()
    param_groups = []
    if layered["backbone"]:
        param_groups.append({"params": layered["backbone"], "lr": backbone_lr})
    if layered["head"]:
        param_groups.append({"params": layered["head"], "lr": head_lr})
    optimizer = AdamW(param_groups, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    early_stopping = EarlyStopping(patience=patience, mode="max")

    best_state = None
    best_val_auc = -1.0
    history = []

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss, n_seen = 0.0, 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)
            n_seen += images.size(0)

        val_metrics = evaluate(model, val_loader, device)
        train_loss = running_loss / max(n_seen, 1)
        print(f"Epoch [{epoch}/{epochs}] train_loss={train_loss:.4f}  "
              f"val_acc={val_metrics['accuracy']:.4f}  val_auc={val_metrics['auc_roc']:.4f}  "
              f"val_macro_f1={val_metrics['macro_f1']:.4f}")
        history.append({"epoch": epoch, "train_loss": train_loss, **val_metrics})

        if val_metrics["auc_roc"] > best_val_auc:
            best_val_auc = val_metrics["auc_roc"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            print(f"  \u2713 New best val AUC-ROC: {best_val_auc:.4f} -- checkpoint saved (in memory)")

        if early_stopping(val_metrics["auc_roc"]):
            print(f"  Early stopping triggered at epoch {epoch}.")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, history, best_val_auc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--finetune_frac", type=float, default=0.7)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--num_blocks_to_freeze", type=int, default=10,
                         help="ViT-Base has 12 blocks; default freezes the first 10, "
                              "leaving the last 2 + head trainable.")
    parser.add_argument("--head_lr", type=float, default=1e-4)
    parser.add_argument("--backbone_lr", type=float, default=1e-6)
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--stain_method", default="macenko", choices=["macenko", "none"])
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    stain_method = None if args.stain_method == "none" else args.stain_method

    split = build_or_load_split(CV_CONFIG["bach_root"], args.finetune_frac, args.seed, SPLIT_PATH)

    stain_norm = build_stain_normalizer(stain_method) if stain_method else None
    train_tf = get_transforms("train", image_size=CV_CONFIG["image_size"], stain_normalizer=stain_norm)
    eval_tf = get_transforms("val", image_size=CV_CONFIG["image_size"], stain_normalizer=stain_norm)

    train_ds = _SampleListDataset(split["finetune"], train_tf)
    val_ds = _SampleListDataset(split["held_out"], eval_tf)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=4)

    print(f"\nLoading zero-shot primary model from {CV_CONFIG['checkpoint']}...")
    model = load_model(CV_CONFIG, device)

    print("\nEvaluating ZERO-SHOT baseline on the held-out BACH split "
          f"({len(val_ds)} images -- same split fine-tuning will later be judged against)...")
    zero_shot_metrics = evaluate(model, val_loader, device)
    print(f"Zero-shot: acc={zero_shot_metrics['accuracy']:.4f}  "
          f"AUC-ROC={zero_shot_metrics['auc_roc']:.4f}  macro-F1={zero_shot_metrics['macro_f1']:.4f}")

    print(f"\nFreezing the first {args.num_blocks_to_freeze} backbone blocks "
          f"(layer-wise LR: backbone={args.backbone_lr:g}, head={args.head_lr:g})...")
    model.freeze_backbone(num_blocks_to_freeze=args.num_blocks_to_freeze)

    print(f"\nFine-tuning on {len(train_ds)} BACH images "
          f"(held out {len(val_ds)} for evaluation, never trained on)...")
    model, history, best_val_auc = finetune(
        model, train_loader, val_loader, device,
        epochs=args.epochs, head_lr=args.head_lr, backbone_lr=args.backbone_lr,
        weight_decay=args.weight_decay, patience=args.patience,
    )

    finetuned_metrics = evaluate(model, val_loader, device)

    print(f"\n{'='*66}\nFINE-TUNING RESULT (held-out BACH subset, n={len(val_ds)})\n{'='*66}")
    print(f"{'Metric':<15s}{'Zero-shot':>15s}{'Fine-tuned':>15s}{'Delta':>12s}")
    for key, label in (("accuracy", "Accuracy"), ("auc_roc", "AUC-ROC"), ("macro_f1", "Macro-F1")):
        b, a = zero_shot_metrics[key], finetuned_metrics[key]
        print(f"{label:<15s}{b:>15.4f}{a:>15.4f}{a-b:>+12.4f}")

    auc_gain = finetuned_metrics["auc_roc"] - zero_shot_metrics["auc_roc"]
    if auc_gain >= 0.05:
        print(f"\n-> Fine-tuning meaningfully improved BACH AUC-ROC (+{auc_gain:.3f}). "
              "Direct domain supervision helps more than preprocessing-level fixes did.")
    elif auc_gain > 0:
        print(f"\n-> Fine-tuning helped a little (+{auc_gain:.3f}) but not dramatically -- "
              "consider unfreezing more blocks, more epochs, or more fine-tune data if available.")
    else:
        print(f"\n-> Fine-tuning did NOT improve BACH AUC-ROC ({auc_gain:+.3f}). With only "
              f"{len(train_ds)} images, this could be overfitting to the fine-tune subset despite "
              "early stopping on held-out AUC -- try fewer trainable blocks, a lower head_lr, or "
              "stronger weight_decay before concluding fine-tuning doesn't help here.")

    torch.save({"model_state_dict": model.state_dict(),
                "finetune_args": vars(args),
                "zero_shot_metrics": zero_shot_metrics,
                "finetuned_metrics": finetuned_metrics}, FINETUNED_CKPT_PATH)
    print(f"\n[OK] Saved fine-tuned checkpoint -> {FINETUNED_CKPT_PATH} "
          "(separate from outputs/best_model.pt -- the primary model is untouched)")

    results_path = Path(CV_CONFIG["output_dir"]) / "bach_finetune_results.json"
    with open(results_path, "w") as f:
        json.dump({"zero_shot": zero_shot_metrics, "finetuned": finetuned_metrics,
                    "history": history, "args": vars(args)}, f, indent=2)
    print(f"[OK] Saved -> {results_path}")


if __name__ == "__main__":
    main()

    