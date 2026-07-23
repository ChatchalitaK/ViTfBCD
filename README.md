# ViTfBCD — Vision Transformer for Breast Cancer Detection

Vision Transformer (ViT) applied to breast cancer histopathological image classification.
Supports **Binary** (Benign/Malignant) and **Multi-class** (8 subtypes) on the **BreakHis** dataset.

---

## Project Structure

```
vit_breast_cancer/
├── src/
│   ├── model.py      # ViTfBCD architecture (Block 1–4)
│   ├── dataset.py    # BreakHis loader + augmentation + class rebalancing
│   ├── trainer.py    # Training loop, early stopping, checkpointing
│   ├── evaluate.py   # Metrics, confusion matrix, attention map visualization
|   ├── distillation.py #NEW: KD + INT8
|   └── uncertainty.py  #NEW: MC-Dropout +attention validation 
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
Output: Benign/Malignant  OR  8 subtypes
```

---

## References

- Yang et al. (2023). A Novel Vision Transformer Model for Skin Cancer Classification. *Neural Processing Letters*, 55, 9335–9351.
- Dosovitskiy et al. (2020). An Image is Worth 16x16 Words. *arXiv:2010.11929*.
- Spanhol et al. (2016). A Dataset for Breast Cancer Histopathological Image Classification. *IEEE TBME*.
