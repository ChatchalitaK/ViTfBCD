import json
from pathlib import Path

import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix
from src.dataset import build_dataloaders
from src.model import ViTfBCD
from src.train_teacher import get_patient_predictions
from src.distillation import build_student

OUTPUT_DIR = Path("outputs/distill_results")
TEACHER_CKPT_PATH = Path("outputs/best_model.pt")
DATA_DIR = "/home/user/Proj-Ploy/vit_breast_cancer/data/BreaKHis_v1/histology_slides/breast/"


def patient_accuracy(patient_label: dict, patient_pred: dict) -> float:
    if not patient_label:
        return float("nan")
    correct = sum(int(patient_pred[pid] == patient_label[pid]) for pid in patient_label)
    return correct / len(patient_label)


def load_teacher(num_classes: int, device):
    model = ViTfBCD(num_classes=num_classes)
    model.resize_position_embeddings()
    ckpt = torch.load(TEACHER_CKPT_PATH, map_location=device)
    state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
    model.load_state_dict(state_dict, strict=False)
    return model.to(device).eval()


def load_student_fp32(student_name: str, num_classes: int, device):
    model = build_student(student_name=student_name, num_classes=num_classes, pretrained=False)
    ckpt = torch.load(OUTPUT_DIR / "best_student.pt", map_location=device)
    state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
    model.load_state_dict(state_dict, strict=False)
    return model.to(device).eval()


def load_student_int8(student_name: str, num_classes: int):
    fp32_model = build_student(student_name=student_name, num_classes=num_classes, pretrained=False)
    quantized_structure = torch.quantization.quantize_dynamic(
        fp32_model.to("cpu").eval(), {nn.Linear}, dtype=torch.qint8,
    )
    state_dict = torch.load(OUTPUT_DIR / f"{student_name}_int8.pt", map_location="cpu")
    quantized_structure.load_state_dict(state_dict, strict=False)
    return quantized_structure.eval()


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    compression_summary_path = OUTPUT_DIR / "compression_summary.json"
    if not compression_summary_path.exists():
        raise FileNotFoundError(f"{compression_summary_path} not found -- run run_distill.py first.")
    with open(compression_summary_path) as f:
        summary = json.load(f)
    student_name = summary.get("student_name", "deit_tiny")
    num_classes = 2  
    image_size = 384

    print("Building the properly held-out BreakHis TEST loader...")
    config = {"magnification": "all", "mode": "binary", "image_size": image_size,
              "seed": 42, "stain_method": "macenko", "batch_size": 32, "num_workers": 4}
    _train_loader, _val_loader, test_loader = build_dataloaders(DATA_DIR, config)
    print(f"Test set: {len(test_loader.dataset)} images")

    print("\nLoading teacher...")
    teacher = load_teacher(num_classes, device)
    print("Loading FP32 student...")
    raw_student_fp32 = load_student_fp32(student_name, num_classes, device)
    print("Loading INT8 student...")
    raw_student_int8 = load_student_int8(student_name, num_classes)

    class DeiTSafeWrapper(nn.Module):
        def __init__(self, base_model):
            super().__init__()
            self.base_model = base_model
        def forward(self, x):
            out = self.base_model(x)
            if isinstance(out, tuple):
                return (out[0] + out[1]) / 2.0
            return out

    student_fp32 = DeiTSafeWrapper(raw_student_fp32)
    student_int8 = DeiTSafeWrapper(raw_student_int8)

    results = {}
    for name, model, eval_device in (
        ("teacher", teacher, device),
        ("student_fp32", student_fp32, device),
        ("student_int8", student_int8, torch.device("cpu")),
    ):
        model_on_device = model.to(eval_device)
        
        patient_label, patient_pred = get_patient_predictions(
            model_on_device, test_loader, eval_device, num_classes=num_classes
        )
        
        y_true = [patient_label[pid] for pid in sorted(patient_label.keys())]
        y_pred = [patient_pred[pid] for pid in sorted(patient_label.keys())]
        
        p_acc = patient_accuracy(patient_label, patient_pred)
        
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        sensitivity = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0

        results[name] = {
            "test_patient_accuracy": p_acc,
            "n_patients": len(patient_label),
            "sensitivity": sensitivity,
            "specificity": specificity,
            "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}
        }
        
        print(f"  {name:<14s}: Acc={p_acc:.4f} | Sens={sensitivity:.4f} | Spec={specificity:.4f}")

    print(f"\n{'='*78}")
    print(f" CLINICAL COMPREHENSIVE COMPARISON (BreakHis TEST Set)")
    print(f"{'='*78}")
    print(f"{'Model':<20s}{'Patient Acc':>13s}{'Sensitivity':>15s}{'Specificity':>15s}{'CM (TN/FP/FN/TP)':>15s}")
    print(f"{'-'*78}")
    
    for name, label in (("teacher", "Teacher (ViT)"), 
                        ("student_fp32", f"Student FP32"),
                        ("student_int8", f"Student INT8")):
        res = results[name]
        cm_str = f"{res['confusion_matrix']['tn']}/{res['confusion_matrix']['fp']}/{res['confusion_matrix']['fn']}/{res['confusion_matrix']['tp']}"
        print(f"{label:<20s}{res['test_patient_accuracy']:>12.2%}{res['sensitivity']:>14.2%}{res['specificity']:>14.2%}{cm_str:>15s}")
    print(f"{'='*78}")

    out_path = OUTPUT_DIR / "distillation_test_set_comparison.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[OK] Saved all expanded metrics to -> {out_path}")

if __name__ == "__main__":
    main()