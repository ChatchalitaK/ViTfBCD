"""
Post-Training Quantization (PTQ) for the distilled student model.

Takes the already-trained FP32 student checkpoint (produced by run_distill.py),
applies INT8 dynamic quantization, and measures everything you need to justify
it as an edge-deployment target:
  - model size on disk (MB)              -> deployability
  - GPU FP32 latency (ms/image)           -> "current" server-side baseline
  - CPU FP32 latency (ms/image)           -> isolates the quantization effect
                                             from the hardware-switch effect
  - CPU INT8 latency (ms/image)           -> the actual edge-deployment number
  - patient-level accuracy on TEST set    -> whether compression cost any
                                             real diagnostic accuracy

Why dynamic quantization specifically: for a ViT-style student, the
FLOPs/latency are dominated by nn.Linear layers (QKV projections, MLP blocks),
and torch.ao.quantization.quantize_dynamic quantizes exactly those to INT8
at inference time -- no calibration dataset or fake-quant training loop
needed, which is why this is "post-training" (as opposed to QAT).

IMPORTANT: quantize_dynamic's INT8 kernels (fbgemm/qnnpack) are CPU-only --
there is no CUDA path for them. So "INT8" here always means CPU, and the
GPU number is always FP32. This script reports GPU-FP32 vs CPU-INT8 as the
realistic deployment trade-off (what you'd actually be choosing between),
AND CPU-FP32 vs CPU-INT8 as the same-hardware control (what quantization
alone buys you, with the hardware switch factored out).

Run AFTER run_distill.py has produced outputs/distill_results/best_student.pt.
"""

import json
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.ao.quantization as ao_quant

from src.dataset import build_dataloaders
from src.model import ViTfBCD
from src.train_teacher import get_patient_predictions
from src.distillation import build_student

# ── Paths ──────────────────────────────────────────────────────────────────
DATA_DIR = "/home/user/Proj-Ploy/vit_breast_cancer/data/BreaKHis_v1/histology_slides/breast/"
OUTPUT_DIR = Path("outputs/distill_results")
COMPRESSION_SUMMARY_PATH = OUTPUT_DIR / "compression_summary.json"
FP32_CKPT_PATH = OUTPUT_DIR / "best_student.pt"
INT8_STATE_DICT_PATH = OUTPUT_DIR / "student_int8_ptq.pt"
RESULT_PATH = OUTPUT_DIR / "ptq_int8_efficiency.json"

N_WARMUP = 10
N_TIMED = 50


# ── Model loading ──────────────────────────────────────────────────────────
def load_fp32_student(student_name: str, num_classes: int) -> nn.Module:
    model = build_student(student_name=student_name, num_classes=num_classes, pretrained=False)
    ckpt = torch.load(FP32_CKPT_PATH, map_location="cpu")
    state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
    model.load_state_dict(state_dict, strict=False)
    return model.eval()


def quantize_to_int8(fp32_model: nn.Module) -> nn.Module:
    """
    Dynamic PTQ: only nn.Linear is quantized (weights -> qint8, activations
    quantized on-the-fly at inference). This is the standard choice for
    transformer-style students because Linear layers dominate their compute;
    Conv2d/patch-embed layers are left in FP32 (dynamic quantization doesn't
    support them well -- static/QAT would be needed for that, which is a
    heavier lift than this task calls for).
    """
    torch.backends.quantized.engine = "fbgemm"  # x86 server/desktop CPUs
    return ao_quant.quantize_dynamic(
        fp32_model.to("cpu").eval(), {nn.Linear}, dtype=torch.qint8,
    )


# ── Measurement helpers ────────────────────────────────────────────────────
def measure_latency_ms(model: nn.Module, image_size: int, device: torch.device) -> float:
    model.eval()
    dummy = torch.randn(1, 3, image_size, image_size, device=device)
    with torch.no_grad():
        for _ in range(N_WARMUP):
            model(dummy)
        start = time.perf_counter()
        for _ in range(N_TIMED):
            model(dummy)
        elapsed = time.perf_counter() - start
    return (elapsed / N_TIMED) * 1000.0  # ms/image


def model_size_mb(path: Path) -> float:
    return path.stat().st_size / (1024 * 1024)


def patient_level_accuracy(model: nn.Module, loader, device: torch.device, num_classes: int) -> dict:
    patient_label, patient_pred = get_patient_predictions(model, loader, device, num_classes=num_classes)
    if not patient_label:
        return {"accuracy": float("nan"), "n_patients": 0}
    correct = sum(int(patient_pred[pid] == patient_label[pid]) for pid in patient_label)
    return {"accuracy": correct / len(patient_label), "n_patients": len(patient_label)}


# ── Main ────────────────────────────────────────────────────────────────────
def main():
    if not FP32_CKPT_PATH.exists():
        raise FileNotFoundError(
            f"{FP32_CKPT_PATH} not found -- run run_distill.py first to produce a trained student."
        )
    if not COMPRESSION_SUMMARY_PATH.exists():
        raise FileNotFoundError(f"{COMPRESSION_SUMMARY_PATH} not found -- run run_distill.py first.")

    with open(COMPRESSION_SUMMARY_PATH) as f:
        summary = json.load(f)
    student_name = summary.get("student_name", "deit_tiny")
    num_classes = 2
    image_size = 384

    print(f"Loading FP32 student ({student_name}) from {FP32_CKPT_PATH} ...")
    fp32_model = load_fp32_student(student_name, num_classes)

    print("Applying INT8 dynamic post-training quantization (nn.Linear only) ...")
    int8_model = quantize_to_int8(fp32_model)
    torch.save(int8_model.state_dict(), INT8_STATE_DICT_PATH)
    print(f"[OK] Saved INT8 state_dict -> {INT8_STATE_DICT_PATH}")

    # -- Size on disk --
    fp32_mb = model_size_mb(FP32_CKPT_PATH)
    int8_mb = model_size_mb(INT8_STATE_DICT_PATH)

    # -- CPU latency (dynamic quantization is CPU-only) --
    cpu = torch.device("cpu")
    print(f"Benchmarking CPU latency ({N_WARMUP} warmup + {N_TIMED} timed runs) ...")
    fp32_latency_cpu = measure_latency_ms(fp32_model.to(cpu), image_size, cpu)
    int8_latency = measure_latency_ms(int8_model, image_size, cpu)

    # -- GPU FP32 latency (the "no compression at all" reference point) --
    # torch.ao.quantization.quantize_dynamic's INT8 kernels are CPU-only
    # (fbgemm/qnnpack) -- there is no CUDA INT8 path here, so INT8 is always
    # measured on CPU. What GPU FP32 latency buys you is the honest
    # deployment comparison: "current server-side FP32-on-GPU" vs
    # "candidate edge INT8-on-CPU", not just an isolated precision effect.
    gpu_available = torch.cuda.is_available()
    fp32_latency_gpu = None
    if gpu_available:
        gpu = torch.device("cuda")
        print(f"Benchmarking GPU (FP32) latency ({N_WARMUP} warmup + {N_TIMED} timed runs) ...")
        fp32_model_gpu = load_fp32_student(student_name, num_classes).to(gpu)  # fresh copy; fp32_model above gets moved to CPU below
        dummy = torch.randn(1, 3, image_size, image_size, device=gpu)
        with torch.no_grad():
            for _ in range(N_WARMUP):
                fp32_model_gpu(dummy)
            torch.cuda.synchronize()
            start = time.perf_counter()
            for _ in range(N_TIMED):
                fp32_model_gpu(dummy)
            torch.cuda.synchronize()
        fp32_latency_gpu = ((time.perf_counter() - start) / N_TIMED) * 1000.0
        del fp32_model_gpu
        torch.cuda.empty_cache()
    else:
        print("[Note] No CUDA device detected -- skipping GPU FP32 latency "
              "(will report CPU-only numbers).")

    # -- Patient-level accuracy on the held-out TEST set --
    print("Building held-out TEST loader ...")
    config = {
        "magnification": "all", "mode": "binary", "image_size": image_size,
        "seed": 42, "stain_method": "macenko", "batch_size": 32, "num_workers": 4,
    }
    _train_loader, _val_loader, test_loader = build_dataloaders(DATA_DIR, config)

    class DeiTSafeWrapper(nn.Module):
        """Matches evaluate_distill_on_test.py's handling of DeiT's
        (class_logits, distillation_logits) tuple output."""
        def __init__(self, base_model):
            super().__init__()
            self.base_model = base_model

        def forward(self, x):
            out = self.base_model(x)
            if isinstance(out, tuple):
                return (out[0] + out[1]) / 2.0
            return out

    print("Evaluating FP32 student on TEST set (patient-level) ...")
    fp32_acc = patient_level_accuracy(DeiTSafeWrapper(fp32_model), test_loader, cpu, num_classes)

    print("Evaluating INT8 student on TEST set (patient-level) ...")
    int8_acc = patient_level_accuracy(DeiTSafeWrapper(int8_model), test_loader, cpu, num_classes)

    results = {
        "student_name": student_name,
        "fp32_cpu": {
            "size_mb": fp32_mb,
            "latency_ms": fp32_latency_cpu,
            "test_patient_accuracy": fp32_acc["accuracy"],
            "n_patients": fp32_acc["n_patients"],
        },
        "fp32_gpu": {
            "size_mb": fp32_mb,  # quantization is what changes size, not device
            "latency_ms": fp32_latency_gpu,
            "test_patient_accuracy": fp32_acc["accuracy"],  # device doesn't change accuracy (up to fp rounding)
            "n_patients": fp32_acc["n_patients"],
            "note": "GPU-available" if gpu_available else "no CUDA device found -- not measured",
        },
        "int8_cpu": {
            "size_mb": int8_mb,
            "latency_ms": int8_latency,
            "test_patient_accuracy": int8_acc["accuracy"],
            "n_patients": int8_acc["n_patients"],
            "compression_vs_fp32": (fp32_mb / int8_mb) if int8_mb else None,
            "speedup_vs_fp32_cpu": (fp32_latency_cpu / int8_latency) if int8_latency else None,
            "speedup_vs_fp32_gpu": (fp32_latency_gpu / int8_latency) if (fp32_latency_gpu and int8_latency) else None,
            "accuracy_delta_vs_fp32": int8_acc["accuracy"] - fp32_acc["accuracy"],
        },
    }

    with open(RESULT_PATH, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*74}")
    print(" INT8 POST-TRAINING QUANTIZATION -- EDGE-DEPLOYMENT SUMMARY")
    print(f"{'='*74}")
    print(f"{'Model':<16}{'Size (MB)':>12}{'Latency (ms)':>16}{'Test Acc':>12}")
    print(f"{'-'*74}")
    print(f"{'FP32 (GPU)':<16}{fp32_mb:>12.2f}"
          f"{(f'{fp32_latency_gpu:.2f}' if fp32_latency_gpu else 'n/a'):>16}"
          f"{fp32_acc['accuracy']*100:>11.2f}%")
    print(f"{'FP32 (CPU)':<16}{fp32_mb:>12.2f}{fp32_latency_cpu:>16.2f}{fp32_acc['accuracy']*100:>11.2f}%")
    print(f"{'INT8 (CPU)':<16}{int8_mb:>12.2f}{int8_latency:>16.2f}{int8_acc['accuracy']*100:>11.2f}%")
    print(f"{'-'*74}")
    print(f"Compression (size):        {results['int8_cpu']['compression_vs_fp32']:.2f}x")
    if fp32_latency_gpu:
        gpu_vs_int8 = fp32_latency_gpu / int8_latency if int8_latency else None
        print(f"GPU FP32 vs CPU INT8 latency: {'faster on GPU' if gpu_vs_int8 and gpu_vs_int8 < 1 else 'faster on INT8/CPU'} "
              f"({fp32_latency_gpu:.2f} ms GPU-FP32  vs  {int8_latency:.2f} ms CPU-INT8, "
              f"ratio={gpu_vs_int8:.2f}x)")
    print(f"CPU speedup from quantization: {results['int8_cpu']['speedup_vs_fp32_cpu']:.2f}x "
          f"(isolates the quantization effect, same hardware)")
    print(f"Accuracy delta (INT8 - FP32): {results['int8_cpu']['accuracy_delta_vs_fp32']*100:+.2f} pts")
    print(f"{'='*74}")
    print(f"\n[OK] Saved -> {RESULT_PATH}")


if __name__ == "__main__":
    main()