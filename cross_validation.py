"""
cross_dataset_validation.py
Evaluates a trained ViTfBCD checkpoint (trained on BreaKHis) against BACH and
PCam -- external histopathology datasets from completely different scanners/
labs/staining batches -- to check whether it generalizes, or just memorized
BreaKHis-specific artifacts.

This is a BINARY (benign vs malignant) comparison only. BACH's 4 classes and
PCam's tumor/no-tumor label don't correspond to BreaKHis's 8 subtypes, so
subtype-level cross-dataset comparison isn't meaningful -- a multiclass
checkpoint's predicted subtype is collapsed to benign/malignant using
dataset.py's BENIGN_SUBTYPES/MALIGNANT_SUBTYPES before comparing.
"""
from pathlib import Path

import json
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from src.dataset import get_transforms, build_stain_normalizer, BENIGN_SUBTYPES, IDX_TO_SUBTYPE, IDX_TO_BINARY
from src.model import build_model
from src.external_datasets import BACHDataset, PCamBinaryDataset

try:
    from sklearn.metrics import classification_report, confusion_matrix, f1_score, roc_auc_score
    _SKLEARN_AVAILABLE = True
except ImportError:
    _SKLEARN_AVAILABLE = False
    print("[Warning] scikit-learn not installed -- reports will be skipped. "
          "Install with: pip install scikit-learn")


# ── CONFIG  ──────────────────────────────────────
CONFIG = {
    "checkpoint": "/home/user/Proj-Ploy/vit_breast_cancer/outputs/best_model.pt",
    "output_dir": "/home/user/Proj-Ploy/vit_breast_cancer/outputs/cross_dataset",
    "model_size": "base",
    "num_classes": 2,         
    "image_size": 384,
    "batch_size": 32,
    "num_workers": 4,
    "stain_methods_to_compare": ["macenko", None],
    "bach_root": "/home/user/Proj-Ploy/vit_breast_cancer/data/external/BACH",
    "pcam_root": "/home/user/Proj-Ploy/vit_breast_cancer/data/external/PCam",
    "pcam_split": "test",
    "pcam_max_samples": 2000,   
    "seed": 42,
}

# Maps each of BreaKHis's 8 subtype indices to binary benign(0)/malignant(1),
# used only when CONFIG["num_classes"] == 8 (a multiclass checkpoint).
SUBTYPE_TO_BINARY_IDX = {
    i: (0 if IDX_TO_SUBTYPE[i] in BENIGN_SUBTYPES else 1) for i in range(8)
}


def load_model(cfg: dict, device: torch.device):
    model = build_model({
        "model_size": cfg["model_size"], "num_classes": cfg["num_classes"], "pretrained": False,
    })
    ckpt = torch.load(cfg["checkpoint"], map_location=device)
    state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt

    keys_to_delete = [
        "vit.heads.head.1.running_mean", 
        "vit.heads.head.1.running_var", 
        "vit.heads.head.1.num_batches_tracked"
    ]
    for key in keys_to_delete:
        if key in state_dict:
            del state_dict[key]
    # ====================================================================================================

    # Resize position embeddings first if the checkpoint's shape doesn't match
    # a freshly built model (see the same fix in trainer.py's load_best()).
    pos_key = "vit.encoder.pos_embedding"
    if (pos_key in state_dict and hasattr(model, "vit")
            and state_dict[pos_key].shape != model.vit.encoder.pos_embedding.shape):
        model.resize_position_embeddings()

    result = model.load_state_dict(state_dict, strict=False)
    if result.missing_keys or result.unexpected_keys:
        print("[Warning] state_dict did not match model.py exactly:")
        if result.missing_keys:
            print(f"  Missing (in model.py's architecture but not in the checkpoint): {result.missing_keys}")
        if result.unexpected_keys:
            print(f"  Unexpected (in the checkpoint but not in model.py's current architecture): "
                  f"{result.unexpected_keys}")

    model.to(device).eval()
    return model


def predict_binary(model, loader, device, num_classes: int):
    """Runs the model and collapses its output to binary (0=benign, 1=malignant)."""
    all_preds, all_labels, all_probs = [], [], []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            logits = model(images)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            preds = probs.argmax(axis=1)
            if num_classes == 8:
                # AUC-ROC isn't well-defined after an 8->2 argmax remap (the
                # softmax was over 8 subtype classes, not benign/malignant),
                # so malignant-probability is left as None for multiclass checkpoints.
                malignant_prob = np.full(len(preds), np.nan)
                preds = np.array([SUBTYPE_TO_BINARY_IDX[p] for p in preds])
            else:
                malignant_prob = probs[:, 1]
            all_preds.extend(preds.tolist())
            all_probs.extend(malignant_prob.tolist())
            labels_list = labels.tolist() if torch.is_tensor(labels) else list(labels)
            all_labels.extend(labels_list)
    return all_labels, all_preds, all_probs


def report(y_true: list, y_pred: list, y_prob: list, dataset_name: str, output_dir: Path, suffix: str = ""):
    if not _SKLEARN_AVAILABLE or not y_true:
        return None
    import matplotlib.pyplot as plt
    from sklearn.metrics import accuracy_score, precision_score, recall_score

    names = ["benign", "malignant"]
    f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    title = f"{dataset_name}{suffix}"
    print(f"\n{'='*60}\n{title} — Cross-Dataset Generalization Report (binary)\n{'='*60}")
    print(f"Samples: {len(y_true)} | Macro-F1: {f1:.4f}\n")
    print(classification_report(y_true, y_pred, target_names=names, labels=[0, 1], zero_division=0))

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1)
    tn, fp, fn, tp = cm.ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else float("nan")

    auc = float("nan")
    if y_prob and not any(p != p for p in y_prob):  # NaN check without importing math
        try:
            auc = roc_auc_score(y_true, y_prob)
        except Exception:
            auc = float("nan")

    metrics = {
        "n_samples": len(y_true),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "sensitivity": float(recall_score(y_true, y_pred, zero_division=0)),
        "specificity": float(specificity),
        "f1_macro": float(f1),
        "auc_roc": float(auc),
    }

    fig, ax = plt.subplots(figsize=(4.5, 4))
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(names)
    ax.set_yticklabels(names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"{title}\nMacro-F1: {f1:.4f}")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]}", ha="center", va="center",
                     color="white" if cm_norm[i, j] > 0.5 else "black")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()

    fname_suffix = suffix.strip().lower().replace(" ", "_").replace("(", "").replace(")", "")
    fname_suffix = f"_{fname_suffix}" if fname_suffix else ""
    out_path = output_dir / f"{dataset_name.lower()}{fname_suffix}_confusion_matrix.png"
    plt.savefig(out_path)
    plt.close()
    print(f"[OK] Saved {out_path}")
    return metrics


def _load_dataset_for_stain(stain_method, cfg: dict):
    """Builds fresh BACH/PCam datasets+loaders under a given stain_method's transform."""
    stain_norm = build_stain_normalizer(stain_method) if stain_method else None
    eval_tf = get_transforms("val", image_size=cfg["image_size"], stain_normalizer=stain_norm)

    loaders = {}

    bach_path = Path(cfg["bach_root"])
    if bach_path.exists():
        bach_ds = BACHDataset(str(bach_path), transform=eval_tf)
        loaders["BACH"] = DataLoader(bach_ds, batch_size=cfg["batch_size"], shuffle=False,
                                      num_workers=cfg["num_workers"])
    else:
        print(f"[SKIP] BACH not found at {bach_path} -- run download_external_datasets.py first.")

    pcam_path = Path(cfg["pcam_root"])
    if pcam_path.exists():
        pcam_ds = PCamBinaryDataset(str(pcam_path), split=cfg["pcam_split"], transform=eval_tf, download=False)
        n = len(pcam_ds)
        max_n = cfg.get("pcam_max_samples")
        if max_n and n > max_n:
            rng = np.random.RandomState(cfg["seed"])
            idx = rng.choice(n, size=max_n, replace=False)
            pcam_ds = Subset(pcam_ds, idx.tolist())
        loaders["PCam"] = DataLoader(pcam_ds, batch_size=cfg["batch_size"], shuffle=False,
                                      num_workers=cfg["num_workers"])
    else:
        print(f"[SKIP] PCam not found at {pcam_path} -- run download_external_datasets.py first.")

    return loaders, stain_norm


def main():
    output_dir = Path(CONFIG["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Loading checkpoint from {CONFIG['checkpoint']}...")
    model = load_model(CONFIG, device)

    stain_variants = CONFIG.get("stain_methods_to_compare", [CONFIG.get("stain_method", "macenko")])
    results: dict = {}

    for stain_method in stain_variants:
        stain_label = stain_method if stain_method else "no_stain_norm"
        suffix = f" (stain={stain_label})" if len(stain_variants) > 1 else ""
        print(f"\n{'#'*60}\n# Evaluating with stain_method={stain_label!r}\n{'#'*60}")

        loaders, stain_norm = _load_dataset_for_stain(stain_method, CONFIG)
        for dataset_name, loader in loaders.items():
            n_total = len(loader.dataset)
            print(f"\n{dataset_name}: evaluating {n_total} images (stain={stain_label})")
            y_true, y_pred, y_prob = predict_binary(model, loader, device, CONFIG["num_classes"])
            metrics = report(y_true, y_pred, y_prob, dataset_name, output_dir, suffix=suffix)
            results.setdefault(dataset_name, {})[stain_label] = metrics

            if stain_norm is not None and hasattr(stain_norm, "fallback_stats"):
                fb = stain_norm.fallback_stats
                if CONFIG.get("num_workers", 0) > 0:
                    print(f"[Note] num_workers={CONFIG['num_workers']} > 0 -- the Macenko "
                          "fallback counter only sees this main process's share and is NOT "
                          "a reliable total; set num_workers=0 to get an accurate count.")
                elif fb["total"] > 0:
                    rate = fb["count"] / fb["total"] * 100
                    print(f"[Stain Normalization] Macenko succeeded on {fb['total']-fb['count']}/"
                          f"{fb['total']} images ({100-rate:.1f}%); fell back to raw (unnormalized) "
                          f"on {fb['count']} ({rate:.1f}%) for {dataset_name}.")

    out_path = output_dir / "cross_dataset_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[OK] Saved full metrics -> {out_path}  (use this to fill cross_validation_section.md)")

    if len(stain_variants) > 1 and results:
        print(f"\n{'='*68}\nSTAIN NORMALIZATION COMPARISON SUMMARY\n{'='*68}")
        print("PRIMARY metric: AUC-ROC (threshold-independent -- compares the model's\n"
              "underlying ranking quality, not accuracy/F1 at a fixed 0.5 cutoff, which\n"
              "can flip which condition 'wins' depending on where each condition's\n"
              "probabilities happen to sit relative to 0.5).\n")

        header = f"{'Dataset':<10s}" + "".join(f"{(s if s else 'no_stain_norm'):>18s}" for s in stain_variants)
        print("AUC-ROC:")
        print(header)
        for dataset_name, per_stain in results.items():
            row = f"{dataset_name:<10s}"
            for s in stain_variants:
                label = s if s else "no_stain_norm"
                m = per_stain.get(label)
                val = m["auc_roc"] if m else None
                row += f"{val:>18.4f}" if val is not None and val == val else f"{'n/a':>18s}"
            print(row)

        print("\nMacro-F1 @ threshold=0.5 (secondary, threshold-dependent -- do not use "
              "this alone to decide which condition generalizes better):")
        print(header)
        for dataset_name, per_stain in results.items():
            row = f"{dataset_name:<10s}"
            for s in stain_variants:
                label = s if s else "no_stain_norm"
                m = per_stain.get(label)
                val = m["f1_macro"] if m else None
                row += f"{val:>18.4f}" if val is not None else f"{'n/a':>18s}"
            print(row)

        print("\nRead the AUC-ROC row as the headline comparison. If 'no_stain_norm'")
        print("scores notably higher AUC-ROC than 'macenko' here, the stain normalizer's")
        print("reference (fit on BreaKHis only) is likely distorting colors on this")
        print("external dataset rather than helping -- consider evaluating (and possibly")
        print("training) without it for true cross-dataset generalization checks, or")
        print("fitting a separate stain reference per target dataset. Prefer this over")
        print("comparing macro-F1/accuracy at a fixed threshold, which can disagree with")
        print("AUC-ROC even when nothing about the underlying model changed.")


if __name__ == "__main__":
    main()
