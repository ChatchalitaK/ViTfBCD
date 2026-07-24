# ViTfBCD — Vision Transformer for Breast Cancer Detection

Vision Transformer (ViT) applied to breast cancer histopathological image classification.
Supports **Binary** (Benign/Malignant) and **Multi-class** (8 subtypes) on the **BreakHis** dataset.

---

## Project Structure

```
vit_breast_cancer/
├── src/
│   ├── model.py      # ViTfBCD architecture (Block 1–4)
│   ├── dataset.py    # BreakHis loader + augmentation + class rebalancing (seeded, reproducible)
│   ├── trainer.py    # Training loop, early stopping, checkpointing
│   ├── evaluate.py   # Metrics, confusion matrix, attention map visualization
|   ├── distillation.py # KD (soft-target loss) + smoothed patient-level checkpoint selection
|   └── uncertainty.py  # MC-Dropout + attention validation
├── run_distill.py                    # Train DeiT-Tiny student via distillation from the ViTfBCD teacher
├── evaluate_distill_on_test.py       # Patient-level teacher/FP32/INT8 comparison on the held-out TEST set
├── quantize student.py               # Standalone INT8 dynamic PTQ + GPU-FP32-vs-CPU-INT8 efficiency benchmark
├── diagnostic_test_agreement.py      # Per-patient probability audit (verifies KD/quantization agreement is genuine)
├── fill_distillation_report.py       # Auto-fills distillation_section.md from the JSON results above
├── distillation_section.md           # Full distillation + quantization write-up (methodology, results, limitations)
├── main.py           # Entry point
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
└── README.md
```

---

## Step 1 — Download the BreakHis Dataset

Download from: https://web.inf.ufpr.br/vri/databases/breast-cancer-histopathological-database-breakhis/

Extract so the folder structure looks like:

```
vit_breast_cancer/
└── data/
    └── BreaKHis_v1/
        └── histology_slides/
            └── breast/
                ├── benign/
                │   └── SOB/
                │       ├── adenosis/
                │       ├── fibroadenoma/
                │       ├── phyllodes_tumor/
                │       └── tubular_adenoma/
                └── malignant/
                    └── SOB/
                        ├── ductal_carcinoma/
                        ├── lobular_carcinoma/
                        ├── mucinous_carcinoma/
                        └── papillary_carcinoma/
```

---

## Step 2 — Build the Docker Image

```bash
cd vit_breast_cancer
docker build -t vitfbcd:latest .
```

For **Rootless Docker** (as recommended by your instructor):

```bash
# Rootless Docker uses the same commands — just ensure rootless is set up first:
# https://docs.docker.com/engine/security/rootless/
docker build -t vitfbcd:latest .
```

---

## Step 3 — Run Training

### Option A: docker-compose (recommended)

```bash
# Binary classification (Benign vs Malignant)
docker compose run vitfbcd-binary

# Multi-class (8 subtypes)
docker compose run vitfbcd-multiclass
```

### Option B: docker run directly

```bash
# Binary — ViTfBCD-Base, 40X magnification
docker run --gpus all \
  -v $(pwd)/data:/workspace/data \
  -v $(pwd)/outputs:/workspace/outputs \
  vitfbcd:latest \
  python main.py \
    --mode binary \
    --model_size base \
    --magnification 40X \
    --epochs 30 \
    --batch_size 16 \
    --visualize_attention

# Multi-class — ViTfBCD-Large, all magnifications
docker run --gpus all \
  -v $(pwd)/data:/workspace/data \
  -v $(pwd)/outputs:/workspace/outputs \
  vitfbcd:latest \
  python main.py \
    --mode multiclass \
    --model_size large \
    --magnification all \
    --epochs 50 \
    --batch_size 8
```

> **Rootless Docker + NVIDIA:** Make sure you've followed
> https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html#rootless-mode
> then use `--gpus all` as normal.

---

## Step 4 — Evaluate & Visualize Attention Maps

```bash
docker compose run vitfbcd-eval
```

Or:

```bash
docker run --gpus all \
  -v $(pwd)/data:/workspace/data \
  -v $(pwd)/outputs:/workspace/outputs \
  vitfbcd:latest \
  python main.py \
    --mode binary \
    --eval_only \
    --visualize_attention
```

---

## Step 5 — Knowledge Distillation, INT8 Quantization & Edge Deployment

The ViTfBCD teacher (86.49M params, 1038 MB) is distilled into a DeiT-Tiny
student (5.60M params) and further compressed with INT8 dynamic
quantization for CPU/edge inference — with no loss in patient-level test
accuracy, sensitivity, or specificity versus the teacher. Full methodology,
results, and limitations are written up in
[`distillation_section.md`](distillation_section.md); this section covers
how to reproduce it.

**5.1 — Train the student via distillation**

```bash
python run_distill.py
```

Trains DeiT-Tiny against the frozen ViTfBCD teacher using a soft-target
distillation loss (temperature-scaled, weighted between ground-truth
cross-entropy and matching the teacher's output distribution). Two
correctness fixes are built into this step:

- **Reproducible runs.** Training is fully seeded (`torch`, `numpy`,
  `random`, deterministic cuDNN, and a seeded generator for both the
  class-balancing sampler and every DataLoader worker) — re-running this
  command on unchanged code reproduces the same result rather than
  drifting by several accuracy points run-to-run.
- **Smoothed checkpoint selection.** With only 12 validation patients, one
  patient flipping correct/incorrect moves raw patient accuracy by 8.33
  points in a single epoch. Checkpoint saving and the `ReduceLROnPlateau`
  step use a 3-epoch moving average of patient accuracy instead, so a
  single noisy epoch is never mistaken for genuine improvement.

Outputs (`outputs/distill_results/`): `best_student.pt`,
`performance_curves.png`, `distill_history.json`, `compression_summary.json`.

**5.2 — Quantize to INT8 and benchmark edge efficiency**

```bash
python "quantize student.py"
```

Applies INT8 dynamic post-training quantization to all `nn.Linear` layers
of the trained student (CPU-only — no calibration set or retraining
needed), then benchmarks size, CPU latency, and (if a GPU is available)
GPU-FP32-vs-CPU-INT8 latency, alongside patient-level test accuracy.
Outputs: `student_int8_ptq.pt`, `ptq_int8_efficiency.json`.

**5.3 — Evaluate teacher / FP32 / INT8 on the held-out TEST set**

```bash
python evaluate_distill_on_test.py
```

Computes patient-level accuracy, sensitivity, specificity, and confusion
matrix for all three models on the same held-out 17-patient TEST set.
Output: `distillation_test_set_comparison.json`.

**5.4 — Verify the result is genuine, not a small-test-set coincidence**

```bash
python diagnostic_test_agreement.py
```

All three models agreeing on 16/17 test patients is a striking result on
such a small cohort. This script prints raw per-patient probabilities
(not just accuracy) for teacher / FP32 / INT8 side by side, to confirm the
one shared misclassification is a genuine hard case (and the same patient
across all three models) rather than an evaluation artifact, and that
FP32-vs-INT8 probability differences are small and evenly distributed
(healthy quantization noise, not a sign quantization silently failed).

**5.5 — Regenerate the write-up**

```bash
python fill_distillation_report.py
```

Auto-fills `distillation_section.md`'s results tables from the JSON files
produced above, keeping the write-up in sync with the actual run.

**Summary of results** (see `distillation_section.md` for full details):

| Model | Params (M) | Size (MB) | Compression vs. Teacher | Test Patient Acc. | CPU Latency (ms/img) |
|---|---|---|---|---|---|
| Teacher (ViTfBCD, FP32) | 86.49 | 1038.0 | — | 94.12% | 110.83 |
| Student (DeiT-Tiny, FP32) | 5.60 | 22.5 | 46.1× | 94.12% | 14.59 |
| Student (DeiT-Tiny, INT8) | 5.60 | 6.6 | 157.3× | 94.12% | 15.16 |

---

## Outputs

All results are saved to `./outputs/vitfbcd_<size>_<mode>_<magnification>/`:

| File | Description |
|---|---|
| `best_model.pt` | Best checkpoint (highest val accuracy) |
| `config.json` | Training configuration |
| `history.json` | Loss & accuracy per epoch |
| `test_metrics.json` | Accuracy, Precision, Recall, F1, AUC-ROC |
| `confusion_matrix.png` | Confusion matrix heatmap |
| `attention_batch.png` | Attention map overlays (8 samples) |

Distillation / quantization outputs saved to `./outputs/distill_results/`:

| File | Description |
|---|---|
| `best_student.pt` | Best DeiT-Tiny student checkpoint (smoothed patient-acc criterion) |
| `distill_history.json` | Per-epoch train/val loss and patient accuracy (raw + smoothed) |
| `performance_curves.png` | Loss curves + validation accuracy (broken-axis, smoothed vs. raw) |
| `compression_summary.json` | Teacher/FP32/INT8 params, size, and val accuracy |
| `student_int8_ptq.pt` | INT8-quantized student weights |
| `ptq_int8_efficiency.json` | Size, GPU/CPU latency, and test accuracy for FP32 vs. INT8 |
| `distillation_test_set_comparison.json` | Patient-level accuracy/sensitivity/specificity/confusion matrix, all 3 models |

---

## CLI Arguments

| Argument | Default | Description |
|---|---|---|
| `--mode` | `binary` | `binary` or `multiclass` |
| `--model_size` | `base` | `base` (86M) or `large` (307M params) |
| `--magnification` | `40X` | `40X`, `100X`, `200X`, `400X`, or `all` |
| `--epochs` | `30` | Number of training epochs |
| `--batch_size` | `16` | Batch size (reduce if OOM) |
| `--lr` | `1e-4` | Learning rate |
| `--eval_only` | `False` | Skip training, load checkpoint |
| `--visualize_attention` | `False` | Generate attention maps |

---

## Architecture Summary (ViTfBCD)

```
Input Histopathology Image (384×384)
    ↓
Block 1: Augmentation + Class Rebalancing (WeightedRandomSampler)
    ↓
Block 2: Patch Tokenization (16×16 patches → 576 tokens + CLS)
         + Learnable Positional Embedding
    ↓
Block 3: Transformer Encoder × N layers
         - Multi-Head Self-Attention
         - LayerNorm + Residual
         - MLP (GeLU)
    ↓
Block 4: Classification Head
         Flatten → BatchNorm → Dense(GeLU) → BatchNorm → Softmax
    ↓
Output: Benign/Malignant  
```

---

## References

- Yang et al. (2023). A Novel Vision Transformer Model for Skin Cancer Classification. *Neural Processing Letters*, 55, 9335–9351.
- Dosovitskiy et al. (2020). An Image is Worth 16x16 Words. *arXiv:2010.11929*.
- Spanhol et al. (2016). A Dataset for Breast Cancer Histopathological Image Classification. *IEEE TBME*.
