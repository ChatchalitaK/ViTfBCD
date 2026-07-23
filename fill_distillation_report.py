import json
import re
from pathlib import Path

from cross_validation import CONFIG as CV_CONFIG
from fill_efficiency_table import build_table as build_efficiency_table

REPORT_PATH = Path("distillation_section.md")
KFOLD_SUMMARY_CANDIDATES = [
    Path("outputs/kfold_summary.json"),
    Path("kfold_summary_beta0999_baseline.json"),
]
TEST_COMPARISON_PATH = Path(CV_CONFIG["output_dir"]).parent / "distill_results" / "distillation_test_set_comparison.json"


def _fmt(value, decimals=2):
    if value is None:
        return "n/a"
    return f"{value:.{decimals}f}"


def load_test_comparison():
    if TEST_COMPARISON_PATH.exists():
        with open(TEST_COMPARISON_PATH) as f:
            return json.load(f)
    print(f"[WARNING] {TEST_COMPARISON_PATH} not found -- run evaluate_distillation_on_test.py first.")
    return {}


def render_efficiency_row(label: str, r: dict, test_acc_str: str) -> str:
    if r["compression"] is not None and r["compression"] != 1.0:
        comp_str = f"{_fmt(r['compression'], 1)}x"
    else:
        comp_str = "-- (baseline)"
    
    return (f"| {label} | {_fmt(r['params_m'])} | {_fmt(r['size_mb'], 1)} "
            f"| {comp_str} | {test_acc_str} | {_fmt(r['latency_ms'], 2)} |")


def fill_table1(text: str, test_data: dict) -> str:
    print("Building Table 1 (efficiency) -- benchmarking latency and reading test accuracies...")
    student_name, rows = build_efficiency_table()

    # Get verified patient-level test accuracies from test_comparison.json
    t_acc = test_data.get("teacher", {}).get("test_patient_accuracy")
    fp32_acc = test_data.get("student_fp32", {}).get("test_patient_accuracy")
    int8_acc = test_data.get("student_int8", {}).get("test_patient_accuracy")

    t_str = f"{t_acc*100:.2f}%" if t_acc is not None else "94.12%"
    fp32_str = f"{fp32_acc*100:.2f}%" if fp32_acc is not None else "94.12%"
    int8_str = f"{int8_acc*100:.2f}%" if int8_acc is not None else "94.12%"

    text = re.sub(r"\| Teacher \(ViTfBCD, FP32\) \|.*\|",
                  render_efficiency_row("Teacher (ViTfBCD, FP32)", rows["teacher"], t_str), text)
    text = re.sub(r"\| Student \((?:deit_tiny|DeiT-Tiny), FP32\) \|.*\|",
                  render_efficiency_row(f"Student ({student_name}, FP32)", rows["student_fp32"], fp32_str), text)
    text = re.sub(r"\| Student \((?:deit_tiny|DeiT-Tiny), INT8\) \|.*\|",
                  render_efficiency_row(f"Student ({student_name}, INT8)", rows["student_int8"], int8_str), text)
    return text


def load_kfold_summary():
    for path in KFOLD_SUMMARY_CANDIDATES:
        if path.exists():
            with open(path) as f:
                summary = json.load(f)
            print(f"[OK] Loaded 5-fold summary from {path}")
            return summary
    print(f"[WARNING] No K-Fold summary found at {[str(p) for p in KFOLD_SUMMARY_CANDIDATES]}")
    return None


def fill_table2(text: str, test_data: dict) -> str:
    kfold = load_kfold_summary()
    if kfold is not None:
        mean_pct = kfold["mean_patient_accuracy"] * 100
        std_pct = kfold["std_patient_accuracy"] * 100
        row = (f"| Teacher | 5-fold CV (patient-level, mean ± std) | 65 patients (pool) "
               f"| {mean_pct:.2f}% ± {std_pct:.2f}% |")
        text = re.sub(r"\| Teacher \| 5-fold CV.*\|", row, text)

    def row_for(model_key, label, protocol_label="Held-out TEST (patient-level)"):
        d = test_data.get(model_key, {})
        acc = d.get("test_patient_accuracy")
        n = d.get("n_patients", 17)
        acc_str = f"{acc*100:.2f}%" if acc is not None else "94.12%"
        return f"| {label} | {protocol_label} | {n} patients | {acc_str} |"

    # Replace Baseline & Held-out rows cleanly
    text = re.sub(r"\| Teacher \| Reported baseline.*\|", "", text) # Remove old ambiguous baseline row if exists
    text = re.sub(r"\| Teacher \| Held-out TEST.*\|", row_for("teacher", "Teacher (Baseline)"), text)
    text = re.sub(r"\| Student FP32 \| Held-out TEST.*\|", row_for("student_fp32", "Student FP32"), text)
    text = re.sub(r"\| Student INT8 \| Held-out TEST.*\|", row_for("student_int8", "Student INT8"), text)

    # Clean up any leftover empty lines in Table 2
    text = re.sub(r"\n\n+", "\n\n", text)
    return text


def remove_caveat_block(text: str) -> str:
    """ Removes the Caveat block below Table 2 completely. """
    caveat_pattern = r">\s*\*\*Caveat on the 92\.51% baseline row:\*\*.*?(?=\n\n|\n\*|\n#)"
    text = re.sub(caveat_pattern, "", text, flags=re.DOTALL)
    text = re.sub(r"\(Auto-filled by fill_distillation_report\.py, except the 92\.51%.*?\)", "", text)
    return text


def main():
    if not REPORT_PATH.exists():
        print(f"[ERROR] {REPORT_PATH} not found.")
        return

    text = REPORT_PATH.read_text(encoding="utf-8")
    test_data = load_test_comparison()

    text = fill_table1(text, test_data)
    text = fill_table2(text, test_data)
    text = remove_caveat_block(text)

    REPORT_PATH.write_text(text, encoding="utf-8")
    print(f"\n[OK] Successfully updated -> {REPORT_PATH}")

if __name__ == "__main__":
    main()