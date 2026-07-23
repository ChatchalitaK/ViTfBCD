import argparse
import json
from pathlib import Path

import torch

from src.dataset import build_dataloaders
from src.model import ViTfBCD
from src.distillation import build_student, FeatureDistillationTrainer

DEFAULT_HISTORY_PATH = "outputs/distill_results/distill_history.json"
TEACHER_CKPT_PATH = "outputs/best_model.pt"
DATA_DIR = "/home/user/Proj-Ploy/vit_breast_cancer/data/BreaKHis_v1/histology_slides/breast/"
FALLBACK_OUTPUT_DIR = "outputs/distill_results_feature_fallback"


def check_collapse(history: dict, min_loss_decrease: float, collapse_acc_threshold: float) -> dict:
    reasons = []

    train_loss = history.get("train_loss", [])
    val_loss = history.get("val_loss", [])
    val_acc = history.get("val_acc", [])
    val_patient_acc = history.get("val_patient_acc", [])

    def has_nan_inf(values):
        return any(v != v or v in (float("inf"), float("-inf")) for v in values)

    if has_nan_inf(train_loss) or has_nan_inf(val_loss):
        reasons.append("NaN/Inf detected in train_loss or val_loss -- numerical collapse "
                       "(check learning rate, gradient clipping, or mixed-precision scaling).")

    if len(train_loss) >= 2:
        start, end = train_loss[0], train_loss[-1]
        rel_decrease = (start - end) / start if start > 0 else 0.0
        if rel_decrease < min_loss_decrease:
            reasons.append(
                f"train_loss barely moved: {start:.4f} -> {end:.4f} "
                f"({rel_decrease*100:.1f}% relative decrease, threshold is "
                f"{min_loss_decrease*100:.0f}%) -- the optimizer doesn't appear to be "
                "fitting anything."
            )

    if val_acc and max(val_acc) < collapse_acc_threshold:
        reasons.append(
            f"val_acc (image-level) never exceeded {max(val_acc):.4f} at any point in "
            f"{len(val_acc)} epochs (threshold: {collapse_acc_threshold:.2f}) -- the student "
            "never learned to separate classes at all, even briefly."
        )

    if val_patient_acc:
        unique_vals = set(round(v, 6) for v in val_patient_acc)
        if len(unique_vals) == 1 and next(iter(unique_vals)) < collapse_acc_threshold:
            reasons.append(
                f"val_patient_acc is STUCK at a single value ({next(iter(unique_vals)):.4f}) "
                f"across all {len(val_patient_acc)} epochs, below the collapse threshold -- "
                "looks like a degenerate 'always predicts the same class' pattern."
            )

    return {"collapsed": len(reasons) > 0, "reasons": reasons}


def launch_feature_fallback(config: dict, num_classes: int, student_name: str, device: torch.device):
    print(f"\n{'#'*70}\n# LAUNCHING FEATURE-LEVEL DISTILLATION FALLBACK\n{'#'*70}")
    print("Stopping alpha/temperature tuning on the logit-level track -- switching")
    print(f"to feature-level distillation with student='{student_name}'.\n")

    print("Loading dataset...")
    train_loader, val_loader, _test_loader = build_dataloaders(DATA_DIR, config)

    print("Loading teacher (frozen primary model)...")
    teacher_model = ViTfBCD(num_classes=num_classes)
    teacher_model.resize_position_embeddings()
    checkpoint = torch.load(TEACHER_CKPT_PATH, map_location=device)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    teacher_model.load_state_dict(state_dict, strict=False)

    print(f"Building fresh student: {student_name} (starting over, not continuing "
          "from the collapsed logit-KD checkpoint)...")
    student_model = build_student(student_name=student_name, num_classes=num_classes, pretrained=True)

    trainer = FeatureDistillationTrainer(
        teacher=teacher_model, student=student_model, config=config,
        device=device, output_dir=FALLBACK_OUTPUT_DIR, num_classes=num_classes,
    )
    trainer.fit(train_loader, val_loader)
    print(f"\n[OK] Feature-level fallback complete. Checkpoints/history under {FALLBACK_OUTPUT_DIR}/")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--history_path", default=DEFAULT_HISTORY_PATH)
    parser.add_argument("--min_loss_decrease", type=float, default=0.10,
                         help="Minimum required relative train_loss decrease over the whole "
                              "run before it's considered 'not learning'.")
    parser.add_argument("--collapse_acc_threshold", type=float, default=0.65,
                         help="If val_acc never exceeds this, or val_patient_acc is stuck "
                              "below this, the run is considered collapsed.")
    parser.add_argument("--student_name", default="deit_tiny")
    parser.add_argument("--num_classes", type=int, default=2)
    parser.add_argument("--image_size", type=int, default=384)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--feature_weight", type=float, default=0.5)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--force_fallback", action="store_true",
                         help="Skip the collapse check and launch the fallback directly.")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not args.force_fallback:
        history_path = Path(args.history_path)
        if not history_path.exists():
            raise FileNotFoundError(f"{history_path} not found -- run run_distill.py first, "
                                     "or pass --force_fallback to skip this check.")
        with open(history_path) as f:
            history = json.load(f)

        result = check_collapse(history, args.min_loss_decrease, args.collapse_acc_threshold)

        print(f"{'='*70}\nSTUDENT CONVERGENCE / COLLAPSE CHECK\n{'='*70}")
        if not result["collapsed"]:
            print("No collapse detected. The logit-level KD run looks like it's learning "
                  "normally -- no need to switch tracks. (If accuracy is merely unsatisfying "
                  "rather than collapsed, that's a tuning question, not a collapse -- use "
                  "monitor_student_convergence.py to check plateau status instead.)")
            return

        print("COLLAPSE DETECTED:")
        for reason in result["reasons"]:
            print(f"  - {reason}")
        print("\nPer instructions: NOT tuning alpha/temperature further. Switching "
              "immediately to the feature-level distillation fallback track.")
    else:
        print("--force_fallback set: skipping the collapse check, launching the fallback directly.")

    config = {
        "epochs": args.epochs, "patience": args.patience, "lr": args.lr,
        "feature_weight": args.feature_weight, "image_size": args.image_size,
        "batch_size": args.batch_size, "mode": "binary", "stain_method": "macenko",
        "label_smoothing": 0.1, "use_class_weights": False, "lr_schedule": "plateau",
        "lr_factor": 0.5, "lr_patience": 3,
    }
    launch_feature_fallback(config, args.num_classes, args.student_name, device)


if __name__ == "__main__":
    main()