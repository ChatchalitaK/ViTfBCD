import json
from pathlib import Path

import numpy as np
import torch

from cross_validation import CONFIG, load_model, predict_binary, _load_dataset_for_stain

try:
    from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
    _SKLEARN_AVAILABLE = True
except ImportError:
    _SKLEARN_AVAILABLE = False


THRESHOLDS = np.arange(0.05, 0.96, 0.05)


def sweep_thresholds(y_true, y_prob):
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob, dtype=float)

    rows = []
    for t in THRESHOLDS:
        y_pred = (y_prob >= t).astype(int)
        acc = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
        pos_rate = y_pred.mean()
        rows.append({"threshold": round(float(t), 2), "accuracy": acc, "macro_f1": f1,
                     "predicted_positive_rate": float(pos_rate)})
    best = max(rows, key=lambda r: r["macro_f1"])
    return rows, best


def diagnose(dataset_name: str, stain_label: str, y_true, y_prob):
    if not _SKLEARN_AVAILABLE:
        print("[SKIP] scikit-learn not installed.")
        return None

    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob, dtype=float)

    if np.isnan(y_prob).any():
        print(f"[SKIP] {dataset_name} ({stain_label}): probabilities are NaN "
              "(checkpoint is an 8-class multiclass model -- threshold analysis "
              "only applies to a genuine binary head).")
        return None

    try:
        auc = roc_auc_score(y_true, y_prob)
    except Exception:
        auc = float("nan")

    y_pred_default = (y_prob >= 0.5).astype(int)
    default_acc = accuracy_score(y_true, y_pred_default)
    default_f1 = f1_score(y_true, y_pred_default, average="macro", zero_division=0)
    default_pos_rate = float(y_pred_default.mean())
    true_pos_rate = float(y_true.mean())

    rows, best = sweep_thresholds(y_true, y_prob)

    print(f"\n{'='*66}\n{dataset_name} (stain={stain_label})\n{'='*66}")
    print(f"AUC-ROC (threshold-independent): {auc:.4f}")
    print(f"True positive (malignant) rate in data: {true_pos_rate:.3f}")
    print(f"At threshold=0.50 -> predicted positive rate: {default_pos_rate:.3f}  "
          f"accuracy: {default_acc:.4f}  macro-F1: {default_f1:.4f}")
    print(f"Best threshold={best['threshold']:.2f} -> accuracy: {best['accuracy']:.4f}  "
          f"macro-F1: {best['macro_f1']:.4f}  predicted positive rate: "
          f"{best['predicted_positive_rate']:.3f}")

    collapsed = abs(default_pos_rate - 1.0) < 0.05 or abs(default_pos_rate - 0.0) < 0.05
    if collapsed:
        print(f"-> COLLAPSED PREDICTION at threshold=0.5: predicted-positive rate "
              f"{default_pos_rate:.3f} means the model is essentially predicting one "
              "class for almost every image at this threshold.")

    if auc >= 0.65:
        gain = best["macro_f1"] - default_f1
        print(f"-> AUC-ROC is reasonable ({auc:.3f}): the model DOES separate the "
              f"classes on {dataset_name}. Recalibrating the threshold to "
              f"{best['threshold']:.2f} recovers {gain:+.4f} macro-F1 with NO "
              "retraining -- use a per-dataset threshold rather than a fixed 0.5.")
    elif auc >= 0.55:
        print(f"-> AUC-ROC is weak but above chance ({auc:.3f}): threshold tuning helps "
              "a little, but the deeper issue is limited discrimination -- retraining "
              "with stronger augmentation / fine-tuning will help more than threshold "
              "tuning alone.")
    else:
        print(f"-> AUC-ROC is near chance ({auc:.3f}): the model is NOT meaningfully "
              f"separating benign/malignant on {dataset_name} at all. No threshold fixes "
              "this -- the representation itself needs to change (retraining with "
              "stronger augmentation, fine-tuning on labeled external data, or checking "
              "for a resolution/preprocessing mismatch).")

    return {
        "dataset": dataset_name, "stain": stain_label,
        "auc_roc": float(auc), "true_positive_rate": true_pos_rate,
        "default_threshold_0.5": {"accuracy": default_acc, "macro_f1": default_f1,
                                   "predicted_positive_rate": default_pos_rate},
        "best_threshold": best,
        "sweep": rows,
    }


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading checkpoint from {CONFIG['checkpoint']}...")
    model = load_model(CONFIG, device)

    stain_variants = CONFIG.get("stain_methods_to_compare", [CONFIG.get("stain_method", "macenko")])
    all_results = []

    for stain_method in stain_variants:
        stain_label = stain_method if stain_method else "no_stain_norm"
        loaders, _stain_norm = _load_dataset_for_stain(stain_method, CONFIG)
        for dataset_name, loader in loaders.items():
            y_true, _y_pred, y_prob = predict_binary(model, loader, device, CONFIG["num_classes"])
            result = diagnose(dataset_name, stain_label, y_true, y_prob)
            if result:
                all_results.append(result)

    out_path = Path(CONFIG["output_dir"]) / "threshold_recalibration_results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n[OK] Saved -> {out_path}")


if __name__ == "__main__":
    main()