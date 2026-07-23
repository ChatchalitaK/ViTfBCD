"""
check_catastrophic_forgetting.py
finetune_on_bach.py showed a large, stable AUC-ROC gain on BACH after
lightly fine-tuning 2 backbone blocks + the head. Before treating that as a
free win, check the obvious risk: did adapting toward BACH cost the model
some of its original BreakHis competence?

Evaluates BOTH the zero-shot primary checkpoint (outputs/best_model.pt) and
the BACH-finetuned checkpoint (outputs/bach_finetuned_model.pt, saved by
finetune_on_bach.py) on the SAME held-out BreakHis test set, so the
before/after comparison is apples-to-apples.

Usage:
    python check_catastrophic_forgetting.py
"""
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from cross_validation import CONFIG as CV_CONFIG
from src.dataset import build_dataloaders, build_stain_normalizer, get_transforms
from src.model import build_model

try:
    from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
    _SKLEARN_AVAILABLE = True
except ImportError:
    _SKLEARN_AVAILABLE = False

FINETUNED_CKPT_PATH = Path("outputs/bach_finetuned_model.pt")
PRIMARY_CKPT_PATH = Path(CV_CONFIG["checkpoint"])
FORGETTING_ALERT_PTS = 5.0  # AUC-ROC points -- flag anything worse than this as real forgetting


@torch.no_grad()
def evaluate_on_loader(model, loader, device):
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


def load_checkpoint_into_fresh_model(ckpt_path: Path, num_classes: int, image_size: int, device):
    model = build_model({"model_size": "base", "num_classes": num_classes, "pretrained": False})
    ckpt = torch.load(ckpt_path, map_location=device)
    state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt

    pos_key = "vit.encoder.pos_embedding"
    if (pos_key in state_dict and hasattr(model, "vit")
            and state_dict[pos_key].shape != model.vit.encoder.pos_embedding.shape):
        model.resize_position_embeddings()

    result = model.load_state_dict(state_dict, strict=False)
    if result.missing_keys or result.unexpected_keys:
        print(f"[Warning] {ckpt_path}: state_dict did not match exactly -- "
              f"missing={result.missing_keys}, unexpected={result.unexpected_keys}")
    return model.to(device).eval()


def main():
    if not FINETUNED_CKPT_PATH.exists():
        raise FileNotFoundError(
            f"{FINETUNED_CKPT_PATH} not found -- run finetune_on_bach.py first "
            "(this checks that specific checkpoint, not the cross-validated seeds, "
            "which don't save their models to keep that script lightweight)."
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Build the BreakHis test loader with the SAME stain normalization the
    # primary model was trained/evaluated with, so this isn't itself a
    # confound.
    stain_norm = build_stain_normalizer(CV_CONFIG.get("stain_method", "macenko"))
    _train_tf = get_transforms("train", image_size=CV_CONFIG["image_size"], stain_normalizer=stain_norm)
    # build_dataloaders needs a data_dir + config; reuse the same BreakHis
    # root used throughout this project's other scripts.
    data_dir = "/home/user/Proj-Ploy/vit_breast_cancer/data/BreaKHis_v1/histology_slides/breast/"
    config = {
        "magnification": "all", "mode": "binary", "image_size": CV_CONFIG["image_size"],
        "seed": 42, "stain_method": CV_CONFIG.get("stain_method", "macenko"),
        "batch_size": CV_CONFIG.get("batch_size", 32), "num_workers": 4,
    }
    _train_loader, _val_loader, test_loader = build_dataloaders(data_dir, config)
    print(f"BreakHis held-out test set: {len(test_loader.dataset)} images")

    print(f"\nLoading PRIMARY (zero-shot on BACH) checkpoint: {PRIMARY_CKPT_PATH}")
    primary_model = load_checkpoint_into_fresh_model(
        PRIMARY_CKPT_PATH, CV_CONFIG["num_classes"], CV_CONFIG["image_size"], device)
    primary_metrics = evaluate_on_loader(primary_model, test_loader, device)

    print(f"Loading BACH-FINETUNED checkpoint: {FINETUNED_CKPT_PATH}")
    finetuned_model = load_checkpoint_into_fresh_model(
        FINETUNED_CKPT_PATH, CV_CONFIG["num_classes"], CV_CONFIG["image_size"], device)
    finetuned_metrics = evaluate_on_loader(finetuned_model, test_loader, device)

    print(f"\n{'='*66}\nBreakHis TEST SET: primary vs. BACH-finetuned\n{'='*66}")
    print(f"{'Metric':<15s}{'Primary (BreakHis)':>22s}{'BACH-finetuned':>18s}{'Delta':>12s}")
    deltas = {}
    for key, label in (("accuracy", "Accuracy"), ("auc_roc", "AUC-ROC"), ("macro_f1", "Macro-F1")):
        b, a = primary_metrics[key], finetuned_metrics[key]
        deltas[key] = a - b
        print(f"{label:<15s}{b:>22.4f}{a:>18.4f}{a-b:>+12.4f}")

    auc_drop_pts = -deltas["auc_roc"] * 100
    print()
    if auc_drop_pts >= FORGETTING_ALERT_PTS:
        print(f"-> CATASTROPHIC FORGETTING DETECTED: BreakHis AUC-ROC dropped "
              f"{auc_drop_pts:.1f} points after BACH fine-tuning. The BACH-finetuned "
              "checkpoint should NOT replace the primary model for BreakHis-facing use -- "
              "keep them as two separate, purpose-specific checkpoints (or fine-tune with "
              "fewer unfrozen blocks / more weight decay / a BreakHis-BACH mixed batch to "
              "reduce forgetting).")
    elif auc_drop_pts > 0:
        print(f"-> Minor BreakHis AUC-ROC drop ({auc_drop_pts:.1f} points) -- likely "
              "acceptable, but worth noting as a small tradeoff in the report rather than "
              "presenting BACH fine-tuning as a pure, free win.")
    else:
        print(f"-> No forgetting: BreakHis performance held steady or even improved "
              f"({-auc_drop_pts:+.1f} points). BACH fine-tuning looks safe to adopt without "
              "a BreakHis-side tradeoff.")

    out_path = Path(CV_CONFIG["output_dir"]) / "catastrophic_forgetting_check.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"primary_on_breakhis_test": primary_metrics,
                    "bach_finetuned_on_breakhis_test": finetuned_metrics,
                    "deltas": deltas}, f, indent=2)
    print(f"\n[OK] Saved -> {out_path}")


if __name__ == "__main__":
    main()