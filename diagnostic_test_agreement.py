"""
Diagnostic: are teacher / student_fp32 / student_int8 REALLY agreeing on every
patient (down to the raw probability), or did they just land on the same
argmax by coincidence while their confidence differs wildly?

This does NOT just re-check accuracy (already known: 94.12% for all three).
It prints, per patient, the raw P(malignant) from all three models side by
side, so you can visually confirm:
  - the FN patient is the SAME one for all three (genuine shared failure
    on a hard/borderline case) vs. different patients that happen to
    balance out to the same COUNT (would be a red flag)
  - fp32 vs int8 probabilities differ by a small amount (healthy, expected
    quantization noise) vs. being suspiciously bit-identical (would suggest
    quantize_dynamic didn't actually touch anything -- though the 22.5MB ->
    6.6MB size drop already argues against that)
  - how close to the 0.5 decision boundary each patient's prediction is,
    which tells you whether "identical predictions" is a fragile coincidence
    (everyone sitting near 0.5) or a robust one (everyone confidently far
    from it, one genuinely hard case aside)

Run AFTER evaluate_distill_on_test.py (reuses the same checkpoints).
"""

import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.dataset import build_dataloaders
from src.model import ViTfBCD
from src.train_teacher import get_patient_predictions
from src.distillation import build_student

OUTPUT_DIR = Path("outputs/distill_results")
TEACHER_CKPT_PATH = Path("outputs/best_model.pt")
DATA_DIR = "/home/user/Proj-Ploy/vit_breast_cancer/data/BreaKHis_v1/histology_slides/breast/"


def load_teacher(num_classes, device):
    model = ViTfBCD(num_classes=num_classes)
    model.resize_position_embeddings()
    ckpt = torch.load(TEACHER_CKPT_PATH, map_location=device)
    state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
    model.load_state_dict(state_dict, strict=False)
    return model.to(device).eval()


def load_student_fp32(student_name, num_classes, device):
    model = build_student(student_name=student_name, num_classes=num_classes, pretrained=False)
    ckpt = torch.load(OUTPUT_DIR / "best_student.pt", map_location=device)
    state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
    model.load_state_dict(state_dict, strict=False)
    return model.to(device).eval()


def load_student_int8(student_name, num_classes):
    fp32_model = build_student(student_name=student_name, num_classes=num_classes, pretrained=False)
    quantized = torch.quantization.quantize_dynamic(
        fp32_model.to("cpu").eval(), {nn.Linear}, dtype=torch.qint8,
    )
    state_dict = torch.load(OUTPUT_DIR / f"{student_name}_int8.pt", map_location="cpu")
    quantized.load_state_dict(state_dict, strict=False)
    return quantized.eval()


class DeiTSafeWrapper(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model

    def forward(self, x):
        out = self.base_model(x)
        if isinstance(out, tuple):
            return (out[0] + out[1]) / 2.0
        return out


def get_patient_probs(model, loader, device, num_classes):
    """Same aggregation as get_patient_predictions, but keeps the raw
    averaged probability vector per patient instead of collapsing to argmax."""
    model.eval()
    from collections import defaultdict
    patient_probs = defaultdict(lambda: torch.zeros(num_classes))
    patient_label = {}
    patient_n_images = defaultdict(int)
    idx = 0
    with torch.no_grad():
        for images, labels, *_ in loader:
            bs = images.size(0)
            paths_labels = loader.dataset.samples[idx: idx + bs]
            idx += bs
            images = images.to(device, non_blocking=True)
            probs = F.softmax(model(images), dim=1).cpu()
            for (path, label), p in zip(paths_labels, probs):
                from src.dataset import _parse_patient_id
                pid = _parse_patient_id(Path(path))
                patient_probs[pid] += p
                patient_label[pid] = label
                patient_n_images[pid] += 1
    # average, not sum, so the printed numbers are genuine probabilities
    patient_mean_probs = {pid: (patient_probs[pid] / patient_n_images[pid]) for pid in patient_probs}
    return patient_label, patient_mean_probs


def main():
    compression_summary_path = OUTPUT_DIR / "compression_summary.json"
    with open(compression_summary_path) as f:
        summary = json.load(f)
    student_name = summary.get("student_name", "deit_tiny")
    num_classes = 2
    image_size = 384

    config = {"magnification": "all", "mode": "binary", "image_size": image_size,
              "seed": 42, "stain_method": "macenko", "batch_size": 32, "num_workers": 4}
    _train_loader, _val_loader, test_loader = build_dataloaders(DATA_DIR, config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Loading teacher / fp32 student / int8 student ...")
    teacher = load_teacher(num_classes, device)
    student_fp32 = DeiTSafeWrapper(load_student_fp32(student_name, num_classes, device))
    student_int8 = DeiTSafeWrapper(load_student_int8(student_name, num_classes))

    label_t, probs_t = get_patient_probs(teacher, test_loader, device, num_classes)
    _, probs_fp32 = get_patient_probs(student_fp32, test_loader, device, num_classes)
    _, probs_int8 = get_patient_probs(student_int8, test_loader, torch.device("cpu"), num_classes)

    print(f"\n{'='*92}")
    print(" PER-PATIENT P(malignant=class 1) -- teacher vs fp32 vs int8")
    print(f"{'='*92}")
    print(f"{'Patient ID':<20}{'True':>6}{'Teacher':>12}{'FP32':>12}{'INT8':>12}{'FP32-INT8 diff':>18}")
    print(f"{'-'*92}")

    flagged = []
    for pid in sorted(label_t.keys()):
        true_label = label_t[pid]
        p_t = probs_t[pid][1].item()
        p_fp32 = probs_fp32[pid][1].item()
        p_int8 = probs_int8[pid][1].item()
        diff = abs(p_fp32 - p_int8)

        pred_t = int(p_t > 0.5)
        pred_fp32 = int(p_fp32 > 0.5)
        pred_int8 = int(p_int8 > 0.5)
        any_wrong = (pred_t != true_label) or (pred_fp32 != true_label) or (pred_int8 != true_label)
        near_boundary = min(abs(p_t - 0.5), abs(p_fp32 - 0.5), abs(p_int8 - 0.5)) < 0.15

        marker = ""
        if any_wrong:
            marker = "  <-- misclassified somewhere"
            flagged.append(pid)
        elif near_boundary:
            marker = "  (near decision boundary)"

        print(f"{pid:<20}{true_label:>6}{p_t:>12.4f}{p_fp32:>12.4f}{p_int8:>12.4f}{diff:>18.5f}{marker}")

    print(f"{'='*92}")
    print(f"\nPatients with a misclassification in at least one model: {flagged}")
    print("If that list has exactly ONE patient and all three models agree it's the")
    print("SAME one -- that's a genuine shared hard case, not a bug. If different")
    print("models disagree on WHICH patient(s) they get wrong while accuracy still")
    print("comes out equal, that would be the red flag to dig into further.")


if __name__ == "__main__":
    main()