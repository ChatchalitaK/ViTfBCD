import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

OUTPUT_DIR = Path("outputs/distill_results")
json_path = OUTPUT_DIR / "distillation_test_set_comparison.json"

if not json_path.exists():
    raise FileNotFoundError(f"Not found: {json_path} Please run evaluatioon script before")

with open(json_path, "r") as f:
    data = json.load(f)

models = ["teacher", "student_fp32", "student_int8"]
model_labels = ["Teacher (ViT)", "Student FP32", "Student INT8"]
metrics = ["test_patient_accuracy", "sensitivity", "specificity"]
metric_labels = ["Accuracy", "Sensitivity", "Specificity"]


plot_data = {metric: [data[model][metric] * 100 for model in models] for metric in metrics}


fig, ax = plt.subplots(figsize=(10, 6), dpi=150)

x = np.arange(len(model_labels)) 
width = 0.25  


rects1 = ax.bar(x - width, plot_data["test_patient_accuracy"], width, label="Patient Accuracy", color="#2b5c8f")
rects2 = ax.bar(x, plot_data["sensitivity"], width, label="Sensitivity (Recall)", color="#4682b4")
rects3 = ax.bar(x + width, plot_data["specificity"], width, label="Specificity", color="#b0c4de")


ax.set_ylabel("Percentage (%)", fontsize=12, fontweight="bold")
ax.set_title("Clinical Metrics Comparison on BreakHis Test Set\n(Patient-Level Analysis)", fontsize=14, fontweight="bold", pad=15)
ax.set_xticks(x)
ax.set_xticklabels(model_labels, fontsize=11, fontweight="bold")
ax.set_ylim(0, 115) 
ax.legend(loc="upper right", frameon=True, shadow=False)
ax.grid(axis="y", linestyle="--", alpha=0.5)


def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f"{height:.1f}%",
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3), 
                    textcoords="offset points",
                    ha="center", va="bottom", fontsize=9)

autolabel(rects1)
autolabel(rects2)
autolabel(rects3)

plt.tight_layout()

metrics_plot_path = OUTPUT_DIR / "clinical_metrics_comparison.png"
plt.savefig(metrics_plot_path, bbox_inches="tight")
print(f"[SUCCESS] Saved main metrics comparison plot to: {metrics_plot_path}")
plt.close()

fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), dpi=150)
fig.suptitle("Patient-Level Confusion Matrices", fontsize=14, fontweight="bold", y=1.05)

for i, model in enumerate(models):
    cm = data[model]["confusion_matrix"]
    matrix = np.array([[cm["tn"], cm["fp"]], [cm["fn"], cm["tp"]]])
    
    ax = axes[i]
    im = ax.imshow(matrix, cmap="Blues", alpha=0.6, vmin=0, vmax=11)
    
    ax.set_title(model_labels[i], fontsize=12, fontweight="bold")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Benign", "Malignant"])
    ax.set_yticklabels(["Benign", "Malignant"])
    ax.set_xlabel("Predicted Label")
    if i == 0:
        ax.set_ylabel("True Label")
        
    for row in range(2):
        for col in range(2):
            ax.text(col, row, str(matrix[row, col]),
                    ha="center", va="center", 
                    fontsize=14, fontweight="bold",
                    color="black" if matrix[row, col] < 7 else "white")

plt.tight_layout()


cm_plot_path = OUTPUT_DIR / "confusion_matrices_comparison.png"
plt.savefig(cm_plot_path, bbox_inches="tight")
print(f"[SUCCESS] Saved Confusion Matrices plot to: {cm_plot_path}")
plt.close()