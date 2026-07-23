import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from cross_validation import CONFIG as CV_CONFIG, load_model
from src.dataset import get_transforms, build_stain_normalizer
from finetune_on_bach import (
    build_or_load_split, _SampleListDataset, evaluate, finetune,
)

def run_one_seed(seed: int, args, device):
    split_path = Path(f"outputs/cross_dataset/bach_finetune_split_seed{seed}.json")
    split = build_or_load_split(CV_CONFIG["bach_root"], args.finetune_frac, seed, split_path)

    stain_method = None if args.stain_method == "none" else args.stain_method
    stain_norm = build_stain_normalizer(stain_method) if stain_method else None
    train_tf = get_transforms("train", image_size=CV_CONFIG["image_size"], stain_normalizer=stain_norm)
    eval_tf = get_transforms("val", image_size=CV_CONFIG["image_size"], stain_normalizer=stain_norm)

    train_ds = _SampleListDataset(split["finetune"], train_tf)
    val_ds = _SampleListDataset(split["held_out"], eval_tf)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=4)

    model = load_model(CV_CONFIG, device)
    zero_shot_metrics = evaluate(model, val_loader, device)

    model.freeze_backbone(num_blocks_to_freeze=args.num_blocks_to_freeze)
    model, history, best_val_auc = finetune(
        model, train_loader, val_loader, device,
        epochs=args.epochs, head_lr=args.head_lr, backbone_lr=args.backbone_lr,
        weight_decay=args.weight_decay, patience=args.patience,
    )
    finetuned_metrics = evaluate(model, val_loader, device)

    ran_to_max_epochs = len(history) == args.epochs
    return {
        "seed": seed, "zero_shot": zero_shot_metrics, "finetuned": finetuned_metrics,
        "n_epochs_run": len(history), "ran_to_max_epochs": ran_to_max_epochs,
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_seeds", type=int, default=5)
    parser.add_argument("--base_seed", type=int, default=100,
                         help="Seeds used are base_seed, base_seed+1, ... -- deliberately "
                              "different from finetune_on_bach.py's default seed=42 split, "
                              "which is kept as its own frozen reference.")
    parser.add_argument("--finetune_frac", type=float, default=0.7)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--num_blocks_to_freeze", type=int, default=10)
    parser.add_argument("--head_lr", type=float, default=1e-4)
    parser.add_argument("--backbone_lr", type=float, default=1e-6)
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--stain_method", default="macenko", choices=["macenko", "none"])
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seeds = [args.base_seed + i for i in range(args.n_seeds)]

    all_runs = []
    for seed in seeds:
        print(f"\n{'#'*66}\n# Seed {seed} ({seeds.index(seed)+1}/{len(seeds)})\n{'#'*66}")
        result = run_one_seed(seed, args, device)
        all_runs.append(result)
        print(f"Seed {seed}: zero-shot AUC={result['zero_shot']['auc_roc']:.4f}  "
              f"fine-tuned AUC={result['finetuned']['auc_roc']:.4f}  "
              f"(ran {result['n_epochs_run']}/{args.epochs} epochs"
              f"{', DID NOT PLATEAU' if result['ran_to_max_epochs'] else ', converged/early-stopped'})")

    zero_shot_aucs = [r["zero_shot"]["auc_roc"] for r in all_runs]
    finetuned_aucs = [r["finetuned"]["auc_roc"] for r in all_runs]
    zero_shot_accs = [r["zero_shot"]["accuracy"] for r in all_runs]
    finetuned_accs = [r["finetuned"]["accuracy"] for r in all_runs]
    n_no_plateau = sum(r["ran_to_max_epochs"] for r in all_runs)

    print(f"\n{'='*66}\nCROSS-SPLIT SUMMARY ({len(seeds)} independent splits, seeds={seeds})\n{'='*66}")
    print(f"{'Metric':<20s}{'Zero-shot (mean±std)':>28s}{'Fine-tuned (mean±std)':>28s}")
    print(f"{'AUC-ROC':<20s}{np.mean(zero_shot_aucs):>18.4f} ± {np.std(zero_shot_aucs):<6.4f}"
          f"{np.mean(finetuned_aucs):>18.4f} ± {np.std(finetuned_aucs):<6.4f}")
    print(f"{'Accuracy':<20s}{np.mean(zero_shot_accs):>18.4f} ± {np.std(zero_shot_accs):<6.4f}"
          f"{np.mean(finetuned_accs):>18.4f} ± {np.std(finetuned_accs):<6.4f}")

    mean_gain = np.mean(finetuned_aucs) - np.mean(zero_shot_aucs)
    std_finetuned = np.std(finetuned_aucs)

    print(f"\nMean AUC-ROC gain across {len(seeds)} splits: {mean_gain:+.4f}")
    print(f"Fine-tuned AUC-ROC std across splits: {std_finetuned:.4f}")
    if n_no_plateau > 0:
        print(f"\n[Note] {n_no_plateau}/{len(seeds)} seeds hit max_epochs without early-stopping "
              "-- consider re-running with --epochs higher (e.g. 25-30) and --patience higher "
              "to find the true plateau before reporting a final number.")

    if std_finetuned < 0.03 and mean_gain > 0.15:
        print("\n-> STABLE, LARGE gain across independent splits. This is real signal, not a "
              "single lucky split -- safe to report fine-tuning as the primary fix.")
    elif std_finetuned >= 0.05:
        print(f"\n-> HIGH VARIANCE across splits (std={std_finetuned:.3f}) despite a positive "
              "mean gain. The single-split result may have been optimistic -- report the mean "
              "and std together, not the single best split's number, and investigate what "
              "differs between the best- and worst-performing splits.")
    else:
        print("\n-> Positive but check magnitude/variance above against your own threshold for "
              "'meaningful and reliable' before reporting.")

    out_path = Path(CV_CONFIG["output_dir"]) / "bach_finetune_cv_results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({
            "seeds": seeds,
            "runs": all_runs,
            "summary": {
                "zero_shot_auc_mean": float(np.mean(zero_shot_aucs)),
                "zero_shot_auc_std": float(np.std(zero_shot_aucs)),
                "finetuned_auc_mean": float(np.mean(finetuned_aucs)),
                "finetuned_auc_std": float(np.std(finetuned_aucs)),
                "mean_gain": float(mean_gain),
            },
        }, f, indent=2)
    print(f"\n[OK] Saved -> {out_path}")


if __name__ == "__main__":
    main()