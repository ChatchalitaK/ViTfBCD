import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

sns.set_theme(style="whitegrid")
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 11

SUMMARY_PATH = "/home/user/Proj-Ploy/vit_breast_cancer/outputs/distill_results/compression_summary.json"
OUTPUT_PATH = "/home/user/Proj-Ploy/vit_breast_cancer/outputs/model_compression_chart.png"

with open(SUMMARY_PATH) as f:
    summary = json.load(f)

student_name = summary.get("student_name", "Student")
teacher_mb = summary.get("teacher_size_mb")
fp32_mb = summary.get("fp32_size_mb")
int8_mb = summary.get("int8_size_mb")

if teacher_mb is None or fp32_mb is None or int8_mb is None:
    raise ValueError(
        f"compression_summary.json is missing one or more sizes "
        f"(teacher={teacher_mb}, fp32={fp32_mb}, int8={int8_mb}) -- "
        "make sure run_distill.py completed fully (teacher checkpoint, "
        "best_student.pt, and the INT8 export must all exist) before "
        "regenerating this chart."
    )

models = [
    f'Teacher Core\n(ViTfBCD - FP32)',
    f'Student Base\n({student_name} - FP32)',
    f'Compressed Student\n({student_name} - INT8)',
]
sizes = [teacher_mb, fp32_mb, int8_mb]

fig, ax = plt.subplots(figsize=(8, 6))

colors = ['#1f77b4', '#aec7e8', '#ff7f0e']
bars = ax.bar(models, sizes, color=colors, width=0.5, edgecolor='black', linewidth=0.7)

ax.set_yscale('log')
ax.set_ylim(min(sizes) * 0.5, max(sizes) * 3)
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda y, _: f'{y:g}'))

for bar in bars:
    height = bar.get_height()
    ax.annotate(f'{height:.1f} MB',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 8),
                textcoords="offset points",
                ha='center', va='bottom', fontweight='bold')

ax.set_title('Model Size Compression Comparison', fontsize=14, fontweight='bold', pad=20)
ax.set_ylabel('Model File Size (MB, log scale)', fontsize=12, fontweight='bold')

teacher_to_fp32_pct = (1 - fp32_mb / teacher_mb) * 100 if teacher_mb else 0
fp32_to_int8_pct = (1 - int8_mb / fp32_mb) * 100 if fp32_mb else 0

ax.annotate(f'\u2b07 -{teacher_to_fp32_pct:.1f}% Size Reduction',
            xy=(1, fp32_mb), xytext=(1, fp32_mb * 2.6),
            ha='center', color='#1e3d59', fontsize=10, fontweight='bold',
            arrowprops=dict(arrowstyle='-', color='#1e3d59', lw=1, alpha=0.6))
ax.annotate(f'\u2b07 -{fp32_to_int8_pct:.1f}% From Student',
            xy=(2, int8_mb), xytext=(2, int8_mb * 4.5),
            ha='center', color='#d9534f', fontsize=10, fontweight='bold',
            arrowprops=dict(arrowstyle='-', color='#d9534f', lw=1, alpha=0.6))

plt.tight_layout()

plt.savefig(OUTPUT_PATH, dpi=300)
print(f"[OK] success: {OUTPUT_PATH}")