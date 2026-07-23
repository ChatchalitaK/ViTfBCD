import json
import re
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.ao.quantization as ao_quant

from cross_validation import CONFIG as CV_CONFIG, load_model as load_teacher
from src.distillation import build_student

OUTPUT_DIR = Path("outputs/distill_results")
COMPRESSION_SUMMARY_PATH = OUTPUT_DIR / "compression_summary.json"
TEST_COMPARISON_PATH = OUTPUT_DIR / "distillation_test_set_comparison.json"
REPORT_PATH = Path("efficiency_section.md")
CACHED_RESULTS_PATH = OUTPUT_DIR / "efficiency_table_results.json"

N_WARMUP = 10
N_TIMED = 50


def measure_latency_ms(model, image_size: int, device: torch.device) -> float:
    model.eval()
    dummy = torch.randn(1, 3, image_size, image_size, device=device)
    with torch.no_grad():
        for _ in range(N_WARMUP):
            model(dummy)
        if device.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(N_TIMED):
            model(dummy)
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
    return (elapsed / N_TIMED) * 1000.0  # ms/image


def load_student_fp32(student_name: str, num_classes: int, ckpt_path: Path):
    model = build_student(student_name=student_name, num_classes=num_classes, pretrained=False)
    state_dict = torch.load(ckpt_path, map_location="cpu")
    if "model_state_dict" in state_dict:
        state_dict = state_dict["model_state_dict"]
    model.load_state_dict(state_dict, strict=False)
    return model.eval()


def load_student_int8(student_name: str, num_classes: int, ckpt_path: Path):
    fp32_model = build_student(student_name=student_name, num_classes=num_classes, pretrained=False)
    quantized_structure = ao_quant.quantize_dynamic(
        fp32_model.to("cpu").eval(), {nn.Linear}, dtype=torch.qint8,
    )
    state_dict = torch.load(ckpt_path, map_location="cpu")
    quantized_structure.load_state_dict(state_dict, strict=False)
    return quantized_structure.eval()


def _fmt(value, decimals=2):
    if value is None:
        return "n/a"
    return f"{value:.{decimals}f}"


def build_table(force_rebench: bool = False):
    if not COMPRESSION_SUMMARY_PATH.exists():
        raise FileNotFoundError(f"{COMPRESSION_SUMMARY_PATH} not found.")
    if not TEST_COMPARISON_PATH.exists():
        raise FileNotFoundError(f"{TEST_COMPARISON_PATH} not found.")

    with open(COMPRESSION_SUMMARY_PATH) as f:
        summary = json.load(f)
    with open(TEST_COMPARISON_PATH) as f:
        test_acc = json.load(f)

    student_name = summary.get("student_name", "student")

    # Reuse cached benchmark if available to keep efficiency_section and distillation_section synchronized
    if CACHED_RESULTS_PATH.exists() and not force_rebench:
        print(f"[OK] Loading cached benchmark results from {CACHED_RESULTS_PATH}")
        with open(CACHED_RESULTS_PATH) as f:
            rows = json.load(f)
        return student_name, rows

    num_classes = CV_CONFIG["num_classes"]
    image_size = CV_CONFIG["image_size"]
    device = torch.device("cpu") 

    print("Loading teacher (frozen primary model)...")
    teacher = load_teacher(CV_CONFIG, device)
    teacher_params_m = sum(p.numel() for p in teacher.parameters()) / 1e6

    fp32_path = OUTPUT_DIR / "best_student.pt"
    int8_path = OUTPUT_DIR / f"{student_name}_int8.pt"

    print("Loading FP32 student...")
    student_fp32 = load_student_fp32(student_name, num_classes, fp32_path)

    print("Loading INT8 student...")
    student_int8 = load_student_int8(student_name, num_classes, int8_path)

    print(f"Benchmarking live CPU latency ({N_WARMUP} warmup + {N_TIMED} timed runs)...")
    teacher_latency = measure_latency_ms(teacher, image_size, device)
    fp32_latency = measure_latency_ms(student_fp32, image_size, device)
    int8_latency = measure_latency_ms(student_int8, image_size, device)

    teacher_mb = summary.get("teacher_size_mb", 1038.0)
    fp32_mb = summary.get("fp32_size_mb", 22.5)
    int8_mb = summary.get("int8_size_mb", 6.6)

    student_params_m = summary.get("student_fp32", {}).get("params_m")
    if student_params_m is None:
        student_params_m = sum(p.numel() for p in student_fp32.parameters()) / 1e6

    rows = {
        "teacher": {
            "params_m": teacher_params_m, "size_mb": teacher_mb,
            "compression": 1.0, "val_acc": test_acc["teacher"]["test_patient_accuracy"],
            "latency_ms": teacher_latency,
        },
        "student_fp32": {
            "params_m": student_params_m, "size_mb": fp32_mb,
            "compression": (teacher_mb / fp32_mb) if teacher_mb and fp32_mb else None,
            "val_acc": test_acc["student_fp32"]["test_patient_accuracy"], "latency_ms": fp32_latency,
        },
        "student_int8": {
            "params_m": student_params_m, "size_mb": int8_mb,
            "compression": (teacher_mb / int8_mb) if teacher_mb and int8_mb else None,
            "val_acc": test_acc["student_int8"]["test_patient_accuracy"], "latency_ms": int8_latency,
        },
    }

    # Cache results so other scripts use identical latency metrics
    with open(CACHED_RESULTS_PATH, "w") as f:
        json.dump(rows, f, indent=2)
    print(f"[OK] Saved synchronized benchmark cache -> {CACHED_RESULTS_PATH}")

    return student_name, rows


def render_row(label: str, r: dict) -> str:
    if r["compression"] is not None and r["compression"] != 1.0:
        comp_str = f"{_fmt(r['compression'], 1)}x"
    else:
        comp_str = "-- (baseline)"
    val_acc_str = f"{r['val_acc']*100:.2f}%" if r.get("val_acc") is not None else "n/a"
    return (f"| {label} | {_fmt(r['params_m'])} | {_fmt(r['size_mb'], 1)} "
            f"| {comp_str} | {val_acc_str} | {_fmt(r['latency_ms'], 2)} |")


def main():
    student_name, rows = build_table()

    if REPORT_PATH.exists():
        text = REPORT_PATH.read_text(encoding="utf-8")
        text = re.sub(r"\| Teacher \(ViTfBCD, FP32\) \|.*\|",
                      render_row("Teacher (ViTfBCD, FP32)", rows["teacher"]), text)
        text = re.sub(r"\| Student \((?:deit_tiny|DeiT-Tiny), FP32\) \|.*\|",
                      render_row(f"Student ({student_name}, FP32)", rows["student_fp32"]), text)
        text = re.sub(r"\| Student \((?:deit_tiny|DeiT-Tiny), INT8\) \|.*\|",
                      render_row(f"Student ({student_name}, INT8)", rows["student_int8"]), text)
        REPORT_PATH.write_text(text, encoding="utf-8")
        print(f"[OK] Updated -> {REPORT_PATH}")


if __name__ == "__main__":
    main()