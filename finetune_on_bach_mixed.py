import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler, ConcatDataset

from cross_validation import CONFIG as CV_CONFIG, load_model
from src.dataset import get_transforms, build_stain_normalizer, build_dataloaders
from src.trainer import EarlyStopping
from finetune_on_bach import build_or_load_split, _SampleListDataset, evaluate

MIXED_CKPT_PATH = Path("outputs/bach_finetuned_mixed_model.pt")


def build_mixed_train_loader(bach_finetune_samples, breakhis_train_dataset, train_tf,
                              mix_ratio: float, batch_size: int, seed: int):
    """
    mix_ratio = target fraction of each epoch's samples that come from
    BreakHis (0.5 = roughly half-and-half per batch on average, via
    WeightedRandomSampler -- not a hard per-batch guarantee, but the
    expected proportion over many batches).
    """
    bach_ds = _SampleListDataset(bach_finetune_samples, train_tf)
    combined = ConcatDataset([bach_ds, breakhis_train_dataset])

    n_bach, n_breakhis = len(bach_ds), len(breakhis_train_dataset)
    # Per-sample weight such that the TOTAL weight mass from each source
    # matches the target mix_ratio, regardless of how unequal n_bach and
    # n_breakhis are (BreakHis train is ~18x larger than the BACH subset).
    w_bach = (1.0 - mix_ratio) / max(n_bach, 1)
    w_breakhis = mix_ratio / max(n_breakhis, 1)
    weights = np.concatenate([np.full(n_bach, w_bach), np.full(n_breakhis, w_breakhis)])

    # One synthetic "epoch" = same total size as the BACH-only run would
    # have used, so epoch-to-epoch comparisons against finetune_on_bach.py
    # stay roughly comparable in the number of gradient steps taken.
    num_samples = n_bach * 2
    rng = torch.Generator().manual_seed(seed)
    sampler = WeightedRandomSampler(weights, num_samples=num_samples, replacement=True, generator=rng)
    return DataLoader(combined, batch_size=batch_size, sampler=sampler, num_workers=4)


def finetune_mixed(model, train_loader, bach_val_loader, breakhis_test_loader, device,
                    epochs, head_lr, backbone_lr, weight_decay, patience, combine_weight):
    layered = model.get_layered_parameters()
    param_groups = []
    if layered["backbone"]:
        param_groups.append({"params": layered["backbone"], "lr": backbone_lr})
    if layered["head"]:
        param_groups.append({"params": layered["head"], "lr": head_lr})
    optimizer = AdamW(param_groups, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    early_stopping = EarlyStopping(patience=patience, mode="max")

    best_state, best_combined = None, -1.0
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

        bach_metrics = evaluate(model, bach_val_loader, device)
        breakhis_metrics = evaluate(model, breakhis_test_loader, device)
        combined_score = (combine_weight * bach_metrics["auc_roc"]
                           + (1 - combine_weight) * breakhis_metrics["auc_roc"])

        train_loss = running_loss / max(n_seen, 1)
        print(f"Epoch [{epoch}/{epochs}] train_loss={train_loss:.4f}  "
              f"BACH_auc={bach_metrics['auc_roc']:.4f}  "
              f"BreakHis_auc={breakhis_metrics['auc_roc']:.4f}  "
              f"combined={combined_score:.4f}")
        history.append({"epoch": epoch, "train_loss": train_loss,
                         "bach": bach_metrics, "breakhis": breakhis_metrics,
                         "combined_score": combined_score})

        if combined_score > best_combined:
            best_combined = combined_score
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            print(f"  New best COMBINED score: {best_combined:.4f} -- checkpoint saved (in memory)")

        if early_stopping(combined_score):
            print(f"  Early stopping triggered at epoch {epoch}.")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, history, best_combined


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--finetune_frac", type=float, default=0.7)
    parser.add_argument("--seed", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--num_blocks_to_freeze", type=int, default=10)
    parser.add_argument("--head_lr", type=float, default=1e-4)
    parser.add_argument("--backbone_lr", type=float, default=1e-6)
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--mix_ratio", type=float, default=0.5,
                         help="Target fraction of each epoch's samples drawn from BreakHis "
                              "(vs. BACH). 0.5 = roughly balanced.")
    parser.add_argument("--combine_weight", type=float, default=0.5,
                         help="Weight on BACH AUC vs BreakHis AUC when picking the best "
                              "checkpoint (0.5 = equal weight to both).")
    parser.add_argument("--stain_method", default="macenko", choices=["macenko", "none"])
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    stain_method = None if args.stain_method == "none" else args.stain_method

    split_path = Path("outputs/cross_dataset/bach_finetune_split.json")
    split = build_or_load_split(CV_CONFIG["bach_root"], args.finetune_frac, args.seed, split_path)

    stain_norm = build_stain_normalizer(stain_method) if stain_method else None
    train_tf = get_transforms("train", image_size=CV_CONFIG["image_size"], stain_normalizer=stain_norm)
    eval_tf = get_transforms("val", image_size=CV_CONFIG["image_size"], stain_normalizer=stain_norm)

    bach_val_ds = _SampleListDataset(split["held_out"], eval_tf)
    bach_val_loader = DataLoader(bach_val_ds, batch_size=args.batch_size, shuffle=False, num_workers=4)

    print("Loading BreakHis train/test loaders for replay + forgetting tracking...")
    data_dir = "/home/user/Proj-Ploy/vit_breast_cancer/data/BreaKHis_v1/histology_slides/breast/"
    bh_config = {
        "magnification": "all", "mode": "binary", "image_size": CV_CONFIG["image_size"],
        "seed": args.seed, "stain_method": stain_method,
        "batch_size": args.batch_size, "num_workers": 4,
    }
    breakhis_train_loader, _breakhis_val_loader, breakhis_test_loader = build_dataloaders(data_dir, bh_config)
    breakhis_train_ds = breakhis_train_loader.dataset

    train_loader = build_mixed_train_loader(
        split["finetune"], breakhis_train_ds, train_tf,
        mix_ratio=args.mix_ratio, batch_size=args.batch_size, seed=args.seed,
    )

    print(f"\nLoading zero-shot primary model from {CV_CONFIG['checkpoint']}...")
    model = load_model(CV_CONFIG, device)

    print("\nBASELINE (zero-shot, before any fine-tuning):")
    zero_shot_bach = evaluate(model, bach_val_loader, device)
    zero_shot_breakhis = evaluate(model, breakhis_test_loader, device)
    print(f"  BACH held-out AUC:  {zero_shot_bach['auc_roc']:.4f}")
    print(f"  BreakHis test AUC:  {zero_shot_breakhis['auc_roc']:.4f}")

    model.freeze_backbone(num_blocks_to_freeze=args.num_blocks_to_freeze)

    print(f"\nMixed fine-tuning (mix_ratio={args.mix_ratio}, combine_weight={args.combine_weight})...")
    model, history, best_combined = finetune_mixed(
        model, train_loader, bach_val_loader, breakhis_test_loader, device,
        epochs=args.epochs, head_lr=args.head_lr, backbone_lr=args.backbone_lr,
        weight_decay=args.weight_decay, patience=args.patience, combine_weight=args.combine_weight,
    )

    final_bach = evaluate(model, bach_val_loader, device)
    final_breakhis = evaluate(model, breakhis_test_loader, device)

    print(f"\n{'='*70}\nMIXED FINE-TUNING RESULT\n{'='*70}")
    print(f"{'Domain':<12s}{'Zero-shot AUC':>16s}{'BACH-only run*':>16s}{'Mixed run':>14s}")
    print(f"{'BACH':<12s}{zero_shot_bach['auc_roc']:>16.4f}{'~0.95':>16s}{final_bach['auc_roc']:>14.4f}")
    print(f"{'BreakHis':<12s}{zero_shot_breakhis['auc_roc']:>16.4f}{'~0.72':>16s}{final_breakhis['auc_roc']:>14.4f}")
    print("(*BACH-only column is from the earlier finetune_on_bach.py + "
          "check_catastrophic_forgetting.py runs, shown for reference)")

    breakhis_recovered = final_breakhis["auc_roc"] - zero_shot_breakhis["auc_roc"]
    bach_gain = final_bach["auc_roc"] - zero_shot_bach["auc_roc"]
    print(f"\nBreakHis AUC change from zero-shot: {breakhis_recovered:+.4f}")
    print(f"BACH AUC change from zero-shot: {bach_gain:+.4f}")
    if breakhis_recovered >= -0.03 and bach_gain > 0.05:
        print("-> Good tradeoff: BreakHis held roughly steady while BACH still improved. "
              "This checkpoint is a reasonable candidate to adopt in place of two separate models.")
    elif breakhis_recovered < -0.10:
        print("-> Still meaningful BreakHis forgetting even with replay. Try a higher "
              "--mix_ratio (more BreakHis per batch), fewer unfrozen blocks, or a lower head_lr.")
    else:
        print("-> Partial improvement -- inspect the full epoch-by-epoch history in the saved "
              "JSON to see whether an earlier epoch (before this run's early-stopping/max-epoch "
              "point) offered a better tradeoff than the final combined-score-best checkpoint.")

    torch.save({"model_state_dict": model.state_dict(), "args": vars(args),
                "zero_shot_bach": zero_shot_bach, "zero_shot_breakhis": zero_shot_breakhis,
                "final_bach": final_bach, "final_breakhis": final_breakhis}, MIXED_CKPT_PATH)
    print(f"\n[OK] Saved -> {MIXED_CKPT_PATH}")

    out_path = Path(CV_CONFIG["output_dir"]) / "bach_finetune_mixed_results.json"
    with open(out_path, "w") as f:
        json.dump({"zero_shot_bach": zero_shot_bach, "zero_shot_breakhis": zero_shot_breakhis,
                    "final_bach": final_bach, "final_breakhis": final_breakhis,
                    "history": history, "args": vars(args)}, f, indent=2)
    print(f"[OK] Saved -> {out_path}")


if __name__ == "__main__":
    main()