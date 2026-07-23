"""
analyze_bach_error_breakdown.py
The binary (benign/malignant) BACH report hides which of BACH's 4 ORIGINAL
classes (Normal, Benign, InSitu, Invasive) the errors are actually coming
from. A model that's fine on 3 of 4 classes but fails hard on one specific
class points to something fixable and specific (e.g. that class's staining
looks unusually different, or its tissue morphology differs most from
BreakHis); a model that's uniformly ~55% across all 4 classes points to a
more general representation problem that needs retraining/fine-tuning, not
a targeted fix.

Reuses cross_validation.py's CONFIG / load_model / build_stain_normalizer /
get_transforms so this analyzes the EXACT same checkpoint + preprocessing
that produced the numbers already reported -- it does not re-derive its own
pipeline.

Usage:
    python analyze_bach_error_breakdown.py
    python analyze_bach_error_breakdown.py --stain no_stain_norm
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from cross_validation import CONFIG, load_model
from src.dataset import get_transforms, build_stain_normalizer
from src.external_datasets import BACHDataset, BACH_CLASS_TO_BINARY

try:
    from sklearn.metrics import accuracy_score
    _SKLEARN_AVAILABLE = True
except ImportError:
    _SKLEARN_AVAILABLE = False


BACH_CLASS_NAMES = list(BACH_CLASS_TO_BINARY.keys())  # ["Normal", "Benign", "InSitu", "Invasive"]


def _original_class_from_path(path: str) -> str:
    for name in BACH_CLASS_NAMES:
        if f"/{name}/" in path.replace("\\", "/"):
            return name
    return "UNKNOWN"


class _PathTrackingWrapper(Dataset):
    """Wraps BACHDataset to also yield the source path, so predictions can
    be traced back to which of BACH's 4 original classes they came from."""
    def __init__(self, base_ds: BACHDataset):
        self.base_ds = base_ds

    def __len__(self):
        return len(self.base_ds)

    def __getitem__(self, idx):
        image, label = self.base_ds[idx]
        path = self.base_ds.samples[idx][0]
        return image, label, path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stain", default="macenko", choices=["macenko", "no_stain_norm"])
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading checkpoint from {CONFIG['checkpoint']}...")
    model = load_model(CONFIG, device)

    stain_method = None if args.stain == "no_stain_norm" else args.stain
    stain_norm = build_stain_normalizer(stain_method) if stain_method else None
    eval_tf = get_transforms("val", image_size=CONFIG["image_size"], stain_normalizer=stain_norm)

    bach_path = Path(CONFIG["bach_root"])
    base_ds = BACHDataset(str(bach_path), transform=eval_tf)
    wrapped_ds = _PathTrackingWrapper(base_ds)
    loader = DataLoader(wrapped_ds, batch_size=CONFIG["batch_size"], shuffle=False,
                         num_workers=0)  # 0 so results stay in this process, no ordering surprises

    per_class = defaultdict(lambda: {"y_true": [], "y_pred": [], "y_prob": []})

    model.eval()
    with torch.no_grad():
        for images, labels, paths in loader:
            images = images.to(device)
            logits = model(images)
            probs = F.softmax(logits, dim=1).cpu().numpy()
            preds = probs.argmax(axis=1)
            malignant_prob = probs[:, 1]

            for i, path in enumerate(paths):
                cls = _original_class_from_path(path)
                per_class[cls]["y_true"].append(int(labels[i]))
                per_class[cls]["y_pred"].append(int(preds[i]))
                per_class[cls]["y_prob"].append(float(malignant_prob[i]))

    print(f"\n{'='*72}")
    print(f"BACH ERROR BREAKDOWN BY ORIGINAL CLASS (stain={args.stain})")
    print(f"{'='*72}")
    print(f"{'Original class':<14s}{'n':>6s}{'Binary label':>14s}{'Accuracy':>12s}"
          f"{'Mean P(malig)':>16s}")

    results = {}
    for cls in BACH_CLASS_NAMES:
        d = per_class.get(cls)
        if not d or not d["y_true"]:
            print(f"{cls:<14s}  (no samples found)")
            continue
        y_true, y_pred, y_prob = d["y_true"], d["y_pred"], d["y_prob"]
        acc = accuracy_score(y_true, y_pred) if _SKLEARN_AVAILABLE else float(np.mean(np.array(y_true) == np.array(y_pred)))
        binary_label = BACH_CLASS_TO_BINARY[cls]
        mean_prob = float(np.mean(y_prob))
        print(f"{cls:<14s}{len(y_true):>6d}{binary_label:>14d}{acc:>11.1%} {mean_prob:>15.3f}")
        results[cls] = {
            "n": len(y_true), "binary_label": binary_label, "accuracy": float(acc),
            "mean_predicted_malignant_prob": mean_prob,
        }

    print(f"{'='*72}")
    # Highlight the worst class explicitly -- this is the actionable finding.
    if results:
        worst_cls = min(results, key=lambda c: results[c]["accuracy"])
        best_cls = max(results, key=lambda c: results[c]["accuracy"])
        spread = results[best_cls]["accuracy"] - results[worst_cls]["accuracy"]
        print(f"\nWorst: {worst_cls} ({results[worst_cls]['accuracy']:.1%})  "
              f"Best: {best_cls} ({results[best_cls]['accuracy']:.1%})  "
              f"Spread: {spread:.1%}")
        if spread >= 0.20:
            print(f"-> CONCENTRATED error pattern: '{worst_cls}' is driving most of the gap. "
                  "Worth a closer look at that class specifically -- open a few of its tiles "
                  f"(root/ICIAR2018_BACH_Challenge/Photos/{worst_cls}/) and eyeball whether "
                  "staining, tissue density, or image quality looks unusually different from "
                  "the other 3 classes or from BreakHis.")
        else:
            print("-> DIFFUSE error pattern: performance is fairly uniform across all 4 "
                  "original classes -- this points to a general representation/generalization "
                  "gap rather than one specific problem class. Targeted fixes for a single "
                  "class won't help much; broader retraining/fine-tuning is the more promising "
                  "direction.")

        # Normal is a particularly informative single case: if the model is
        # confusing histologically-normal tissue for malignant, that's a
        # distinct (and clinically worse) failure mode from confusing two
        # abnormal-but-different categories with each other.
        if "Normal" in results and results["Normal"]["accuracy"] < 0.5:
            print(f"\n[Note] 'Normal' tissue accuracy is only {results['Normal']['accuracy']:.1%} "
                  "-- the model is calling truly normal tissue malignant more often than not. "
                  "This is a distinct failure mode from subtype confusion and worth flagging "
                  "separately in the report (false-positive-heavy on clearly benign tissue).")

    out_path = Path(CONFIG["output_dir"]) / f"bach_error_breakdown_{args.stain}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[OK] Saved -> {out_path}")


if __name__ == "__main__":
    main()