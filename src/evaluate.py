"""
Evaluation & Attention Map Visualization for ViTfBCD
Computes: Accuracy, Precision, Recall, F1, Specificity, AUC-ROC
Generates: Attention map overlays on histopathological images
"""
import json
import numpy as np
import torch
import torch.nn.functional as F
import sys
import os
from torch.utils.data import DataLoader
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    accuracy_score, precision_score, recall_score, f1_score
)
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pathlib import Path
from PIL import Image
from torchvision import transforms

from src.model import build_model, ViTfBCD
from src.dataset import build_dataloaders, IDX_TO_BINARY, IDX_TO_SUBTYPE, VIT_IMAGE_SIZE


# ── Full evaluation on test set ───────────────────────────────────────────────
def evaluate_both_modes(model: ViTfBCD, loader: DataLoader, device: torch.device, output_dir: str = "/home/user/Proj-Ploy/vit_breast_cancer/outputs"):

    import os
    from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
    import seaborn as sns
    import matplotlib.pyplot as plt
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model.eval()
    all_preds_8 = []
    all_labels_8 = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            logits = model(images)
            preds = logits.argmax(dim=1).cpu().numpy()
            
            all_preds_8.extend(preds)
            all_labels_8.extend(labels.numpy())

    all_preds_8 = np.array(all_preds_8)
    all_labels_8 = np.array(all_labels_8)

    num_classes = model.num_classes if hasattr(model, "num_classes") else int(logits.shape[-1])
    names_8 = ["Adenosis", "Fibroadenoma", "Phyllodes Tumor", "Tubular Adenoma", 
               "Carcinoma", "Lobular Carcinoma", "Mucinous Carcinoma", "Papillary Carcinoma"]
    names_2 = ["Benign", "Malignant"]

    if num_classes == 8:
        # Model predicts the 8 BreakHis subtypes -- build both the native
        # subtype report AND a derived binary (benign/malignant) view via
        # the same <4 threshold dataset.py uses (0-3=benign, 4-7=malignant).
        all_preds_2 = np.where(all_preds_8 < 4, 0, 1)
        all_labels_2 = np.where(all_labels_8 < 4, 0, 1)

        print("\n" + "="*60)
        print("REPORT 1: SUBTYPE CLASSIFICATION PERFORMANCE (8 CLASSES)")
        print("="*60)
        print(classification_report(all_labels_8, all_preds_8, target_names=names_8, zero_division=0))

        cm_8 = confusion_matrix(all_labels_8, all_preds_8)
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(cm_8, annot=True, fmt="d", cmap="Blues", xticklabels=names_8, yticklabels=names_8, ax=ax, linewidths=0.5)
        ax.set_xlabel("Predicted Subtype", fontsize=11)
        ax.set_ylabel("True Subtype", fontsize=11)
        ax.set_title(f"Confusion Matrix — Subtype (8 Classes)\nAccuracy: {accuracy_score(all_labels_8, all_preds_8):.4f}", fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        fig.savefig(output_dir / "confusion_matrix_8class.png", dpi=150)
        plt.close(fig)
        print(f" Saved 8-class Confusion Matrix to {output_dir / 'confusion_matrix_8class.png'}")
    elif num_classes == 2:
        # Model IS the binary head -- no remap needed or valid. Using the
        # already-binary predictions/labels directly avoids the `<4` bug.
        all_preds_2 = all_preds_8
        all_labels_2 = all_labels_8
        print("\n[Info] Model has 2 output classes -- skipping the 8-class subtype "
              "report (there is nothing to remap; predictions are already binary).")
    else:
        raise ValueError(
            f"evaluate_both_modes() only supports 2-class (binary) or 8-class "
            f"(subtype) models -- got model.num_classes={num_classes}. Refusing "
            f"to guess a remap for an unexpected class count."
        )

    # ==============================================================================
    # 2 CLASS (BINARY)
    # ==============================================================================
    print("\n" + "="*60)
    print("REPORT 2: BINARY CLASSIFICATION PERFORMANCE (2 CLASSES)")
    print("="*60)
    print(classification_report(all_labels_2, all_preds_2, target_names=names_2, zero_division=0))
    
    cm_2 = confusion_matrix(all_labels_2, all_preds_2)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm_2, annot=True, fmt="d", cmap="Reds", xticklabels=names_2, yticklabels=names_2, ax=ax, linewidths=0.5)
    ax.set_xlabel("Predicted Category", fontsize=11)
    ax.set_ylabel("True Category", fontsize=11)
    ax.set_title(f"Confusion Matrix — Binary (2 Classes)\nAccuracy: {accuracy_score(all_labels_2, all_preds_2):.4f}", fontsize=12)
    plt.tight_layout()
    fig.savefig(output_dir / "confusion_matrix_2class.png", dpi=150)
    plt.close(fig)
    print(f"Saved 2-class Confusion Matrix to {output_dir / 'confusion_matrix_2class.png'}")

    metrics_out = {"acc_8": accuracy_score(all_labels_8, all_preds_8) if num_classes == 8 else None,
                   "acc_2": accuracy_score(all_labels_2, all_preds_2),
                   "num_classes": int(num_classes)}
    with open(output_dir / "test_metrics.json", "w") as f:
        json.dump(metrics_out, f, indent=2)
    print(f"Saved metrics -> {output_dir / 'test_metrics.json'}")
    return metrics_out
    prec = precision_score(all_labels_8, all_preds_8, average="weighted", zero_division=0)
    rec  = recall_score(all_labels_8, all_preds_8, average="weighted", zero_division=0)
    f1   = f1_score(all_labels_8, all_preds_8, average="weighted", zero_division=0)

    # Specificity per class (macro average)
    cm = confusion_matrix(all_labels_8, all_preds_8)
    specificities = []
    for i in range(len(class_names)):
        tn = cm.sum() - (cm[i, :].sum() + cm[:, i].sum() - cm[i, i])
        fp = cm[:, i].sum() - cm[i, i]
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        specificities.append(spec)
    specificity = np.mean(specificities)

    # AUC-ROC
    try:
        if mode == "binary":
            auc = roc_auc_score(all_labels_8, all_preds_8[:, 1])
        else:
            auc = roc_auc_score(all_labels_8, all_preds_8, multi_class="ovr", average="weighted")
    except Exception:
        auc = float("nan")

    print("\n" + "="*60)
    print("  EVALUATION RESULTS")
    print("="*60)
    print(f"  Accuracy    : {acc:.4f}")
    print(f"  Precision   : {prec:.4f}")
    print(f"  Sensitivity : {rec:.4f}")
    print(f"  Specificity : {specificity:.4f}")
    print(f"  F1-Score    : {f1:.4f}")
    print(f"  AUC-ROC     : {auc:.4f}")
    print("="*60)
    print("\nPer-class Report:")
    print(classification_report(all_labels_8, all_preds_8, target_names=class_names, zero_division=0))

    # ── Confusion matrix plot ────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=class_names, yticklabels=class_names,
        ax=ax, linewidths=0.5
    )
    ax.set_xlabel("Predicted Label", fontsize=12)
    ax.set_ylabel("True Label", fontsize=12)
    ax.set_title(f"Confusion Matrix — {mode.capitalize()} Classification\nAccuracy: {acc:.4f}", fontsize=13)
    plt.tight_layout()
    cm_path = output_dir / "confusion_matrix.png"
    fig.savefig(cm_path, dpi=150)
    plt.close(fig)
    print(f"Confusion matrix saved to {cm_path}")

    return {"accuracy": acc, "precision": prec, "sensitivity": rec,
            "specificity": specificity, "f1": f1, "auc": auc}


# ── Attention Map Visualization ───────────────────────────────────────────────
def visualize_attention(
    model: ViTfBCD,
    image_path: str,
    device: torch.device,
    image_size: int = VIT_IMAGE_SIZE,
    patch_size: int = 16,
    output_dir: str = "/workspace/outputs",
    head_fusion: str = "mean", 
):
    """
    Generate attention map overlay for a single histopathological image.
    Shows which regions the model focuses on when making a prediction.

    Args:
        head_fusion: how to combine attention across heads ('mean' recommended)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load & preprocess image
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    orig_img = Image.open(image_path).convert("RGB").resize((image_size, image_size))
    tensor   = transform(orig_img).unsqueeze(0).to(device)

    # Extract attention weights from last encoder block
    model.eval()
    attn_weights = model._extract_attention(tensor)
    # attn_weights: [1, num_heads, num_patches+1, num_patches+1]

    if attn_weights is None:
        print("Could not extract attention weights.")
        return

    attn = attn_weights[0]  # [num_heads, seq, seq]

    # Fuse heads
    if head_fusion == "mean":
        attn = attn.mean(dim=0)
    elif head_fusion == "max":
        attn = attn.max(dim=0).values
    elif head_fusion == "min":
        attn = attn.min(dim=0).values
    attn = attn.cpu().detach().numpy()

    # Attention Rollout: attention from CLS token to all patches
    cls_attn = attn[0, 1:]  # [num_patches]
    num_patches = cls_attn.shape[0]
    grid_size   = int(num_patches ** 0.5)

    attn_map = cls_attn.reshape(grid_size, grid_size)
    attn_map = (attn_map - attn_map.min()) / (attn_map.max() - attn_map.min() + 1e-8)

    # Upsample to image size
    attn_up = Image.fromarray((attn_map * 255).astype(np.uint8)).resize(
        (image_size, image_size), Image.BILINEAR
    )
    attn_up = np.array(attn_up) / 255.0

    # Overlay heatmap on original image
    orig_np = np.array(orig_img) / 255.0
    heatmap  = cm.jet(attn_up)[:, :, :3]
    overlay  = 0.55 * orig_np + 0.45 * heatmap
    overlay  = np.clip(overlay, 0, 1)

    # Save figure
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(orig_np);    axes[0].set_title("Original Image",   fontsize=13); axes[0].axis("off")
    axes[1].imshow(attn_up, cmap="jet"); axes[1].set_title("Attention Map", fontsize=13); axes[1].axis("off")
    axes[2].imshow(overlay);   axes[2].set_title("Overlay",           fontsize=13); axes[2].axis("off")

    plt.suptitle("ViTfBCD Attention Visualization — Diseased areas appear brighter", fontsize=12)
    plt.tight_layout()
    out_path = output_dir / f"attention_{Path(image_path).stem}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Attention map saved to {out_path}")
    return out_path


# ── Batch attention visualization ─────────────────────────────────────────────
def visualize_attention_batch(
    model: ViTfBCD,
    loader: DataLoader,
    dataset,
    device: torch.device,
    n_samples: int = 8,
    output_dir: str = "/workspace/outputs",
):
    """Visualize attention maps for n_samples images from a DataLoader."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model.eval()
    images_shown = 0
    fig, axes = plt.subplots(n_samples, 3, figsize=(12, 4 * n_samples))

    mean = torch.tensor([0.485, 0.456, 0.406])
    std  = torch.tensor([0.229, 0.224, 0.225])

    for images, labels in loader:
        for i in range(images.size(0)):
            if images_shown >= n_samples:
                break
            img_tensor = images[i:i+1].to(device)

            attn_weights = model._extract_attention(img_tensor)
            if attn_weights is None:
                continue

            attn = attn_weights[0].mean(dim=0).cpu().detach().numpy()
            cls_attn = attn[0, 1:]
            grid_size = int(cls_attn.shape[0] ** 0.5)
            attn_map = cls_attn.reshape(grid_size, grid_size)
            attn_map = (attn_map - attn_map.min()) / (attn_map.max() - attn_map.min() + 1e-8)
            h = images[i].shape[-1]
            attn_up = np.array(Image.fromarray((attn_map * 255).astype(np.uint8)).resize((h, h), Image.BILINEAR)) / 255.0

            # Denormalize image for display
            img_disp = images[i].permute(1, 2, 0) * std + mean
            img_disp = np.clip(img_disp.numpy(), 0, 1)
            heatmap  = cm.jet(attn_up)[:, :, :3]
            overlay  = np.clip(0.55 * img_disp + 0.45 * heatmap, 0, 1)

            label_name = dataset.class_names[labels[i].item()]

            axes[images_shown, 0].imshow(img_disp);  axes[images_shown, 0].set_title(f"Original\n[{label_name}]"); axes[images_shown, 0].axis("off")
            axes[images_shown, 1].imshow(attn_up, cmap="jet"); axes[images_shown, 1].set_title("Attention"); axes[images_shown, 1].axis("off")
            axes[images_shown, 2].imshow(overlay);   axes[images_shown, 2].set_title("Overlay"); axes[images_shown, 2].axis("off")
            images_shown += 1

        if images_shown >= n_samples:
            break

    plt.suptitle("ViTfBCD — Attention Map Visualization (Batch)", fontsize=14)
    plt.tight_layout()
    out_path = output_dir / "attention_batch.png"
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Batch attention maps saved to {out_path}")

if __name__ == "__main__":
    import os
    import sys
    import random

    config = {
        "magnification": "all",
        "mode": "binary",   
        "num_classes": 2,      
        "model_size": "base",
        "pretrained": False,
        "image_size": 224,
        "batch_size": 16,
        "num_workers": 4,
        "seed": 42
    }

    # set direction file
    DATA_DIR = "/home/user/Proj-Ploy/vit_breast_cancer/data/BreaKHis_v1/histology_slides/breast/"
    OUTPUT_DIR = "/home/user/Proj-Ploy/vit_breast_cancer/outputs"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"==========================================")
print(f"STARTING BREAST CANCER CLASSIFICATION")
print(f"Executing device selection: {device.type.upper()}")

print("\nloading dataset split for testing")
_, _, test_loader = build_dataloaders(DATA_DIR, config)
test_dataset = test_loader.dataset

print("\n Restoring Trained ViTfBCD Architecture")
model = build_model(config)
checkpoint_path = os.path.join(OUTPUT_DIR, "best_model.pt")

if not os.path.exists(checkpoint_path):
    print(f"Error: No file at checkpoint path")
    sys.exit(1)

checkpoint = torch.load(checkpoint_path, map_location=device)
model.load_state_dict(checkpoint["model_state_dict"])
model = model.to(device)
print(" Best weight loaed successfully")

metrics = evaluate_both_modes(
        model=model, 
        loader=test_loader, 
        device=device, 
        output_dir=OUTPUT_DIR
    )

print("\n Generating Batch Attention Map Visualization")
visualize_attention_batch(
    model=model,
    loader=test_loader,
    dataset=test_dataset,
    device=device,
    output_dir=OUTPUT_DIR
)

print("\n Generating Single High-Resolution Attention Overlay")
try:
    random.seed(config["seed"])
    sample_path, _ = random.choice(test_dataset.samples)
    visualize_attention(
        model=model,
        image_path=sample_path,
        device=device,
        image_size=config["image_size"],
        output_dir=OUTPUT_DIR
    )

except Exception as e:
    print("f Could not generate single attention map overlay: {e}")

print("\n Process completed successfully")

