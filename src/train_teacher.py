"""
train_teacher.py
Retrain the ViTfBCD teacher model using the LEAK-FIXED patient-wise splits in
dataset.py.

Two modes, controlled by config["k_folds"]:
  - config["k_folds"] = None  -> single train/val/test split (original behavior)
  - config["k_folds"] = 5     -> patient-level stratified K-fold CV over the
                                  train+val pool, with the SAME held-out test
                                  set used for a final ensemble evaluation.
    K-fold matters here because val/test only have 12/16 patients, so a
    single-split patient-level accuracy can only move in ~8% increments and
    is very noisy (two different configs landed on the exact same 66.67%/
    50.00% val/test numbers). K-fold gives a mean +/- std across folds, and
    ensembling the fold models on the untouched test set usually boosts
    accuracy for free by averaging out per-fold noise.

Usage:
    python train_teacher.py
"""

import os
import json
import random
import time
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.amp import autocast, GradScaler
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from PIL import Image
import matplotlib.pyplot as plt

from src.dataset import (
    build_dataloaders, BreakHisDataset, get_transforms, build_stain_normalizer,
    _parse_patient_id, IDX_TO_SUBTYPE, make_weighted_sampler,
)
from src.model import ViTfBCD
from src.distillation import compute_class_weights, evaluate_teacher


def evaluate_teacher_patient_level(model, loader, device):
    """
    Aggregate predictions per patient (mean softmax across all that patient's
    images/magnifications) before scoring. Multiple images from the same
    patient are correlated, not independent samples, so per-image accuracy
    understates real-world reliability. This gives the "patient diagnosis
    accuracy" figure that's standard in BreaKHis literature.
    """
    import torch.nn.functional as F
    from collections import defaultdict

    dataset = loader.dataset
    model.eval()

    patient_probs = defaultdict(lambda: torch.zeros(dataset.num_classes))
    patient_label = {}

    with torch.no_grad():
        idx = 0
        for images, labels in loader:
            bs = images.size(0)
            paths_labels = dataset.samples[idx: idx + bs]
            idx += bs

            images = images.to(device, non_blocking=True)
            probs = F.softmax(model(images), dim=1).cpu()

            for (path, label), p in zip(paths_labels, probs):
                pid = _parse_patient_id(Path(path))
                patient_probs[pid] += p
                patient_label[pid] = label

    correct = 0
    for pid, prob_sum in patient_probs.items():
        pred = prob_sum.argmax().item()
        correct += int(pred == patient_label[pid])
    acc = correct / len(patient_probs) if patient_probs else 0.0
    print(f"[Teacher patient-level] Patient Acc: {acc:.4f} ({len(patient_probs)} patients)")
    return acc


def freeze_backbone(model, num_trainable_blocks: int):
    """
    Freeze all ViT encoder blocks except the last `num_trainable_blocks`, plus
    always keep the classification head trainable. Drastically cuts the number
    of trainable params, which matters a lot when fine-tuning an 86M-param
    ViT-Base on only ~54 independent patients (overfitting risk is severe
    otherwise, as seen in the first training run: train loss 1.12->0.49 while
    val loss climbed 2.03->2.31 within a few epochs).
    """
    blocks = model.vit.encoder.layers
    total_blocks = len(blocks)
    num_trainable_blocks = min(num_trainable_blocks, total_blocks)

    for p in model.vit.parameters():
        p.requires_grad_(False)

    for block in blocks[total_blocks - num_trainable_blocks:]:
        for p in block.parameters():
            p.requires_grad_(True)

    if hasattr(model.vit.encoder, "ln"):
        for p in model.vit.encoder.ln.parameters():
            p.requires_grad_(True)
    for p in model.vit.heads.parameters():
        p.requires_grad_(True)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"  Frozen backbone: training last {num_trainable_blocks}/{total_blocks} encoder blocks + head "
          f"({trainable/1e6:.1f}M / {total/1e6:.1f}M params trainable)")


def save_teacher_curves(history, save_path):
    epochs = list(range(1, len(history["train_loss"]) + 1))

    plt.rcParams["font.family"] = "sans-serif"
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    ax1.plot(epochs, history["train_loss"], "o-", color="#1f77b4", linewidth=2, label="Train Loss")
    ax1.plot(epochs, history["val_loss"], "s-", color="#ff7f0e", linewidth=2, label="Validation Loss")
    ax1.set_title("Teacher Training Loss Curves", fontsize=12, fontweight="bold", pad=10)
    ax1.set_xlabel("Epochs", fontsize=11)
    ax1.set_ylabel("Loss Value", fontsize=11)
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(fontsize=10)

    val_acc_pct = [a * 100 for a in history["val_acc"]]
    ax2.plot(epochs, val_acc_pct, "^--", color="#2ca02c", linewidth=2, label="Val Accuracy (per-image)")

    if "val_patient_acc" in history and history["val_patient_acc"]:
        patient_acc_pct = [a * 100 for a in history["val_patient_acc"]]
        ax2.plot(epochs, patient_acc_pct, "d-", color="#d62728", linewidth=2, label="Val Accuracy (patient-level)")
        best_acc = max(patient_acc_pct)
        best_epoch = epochs[patient_acc_pct.index(best_acc)]
    else:
        best_acc = max(val_acc_pct)
        best_epoch = epochs[val_acc_pct.index(best_acc)]

    ax2.scatter(best_epoch, best_acc, color="black", s=100, zorder=5,
                label=f"Best Patient Acc: {best_acc:.2f}% (Ep {best_epoch})")
    ax2.set_title("Teacher (ViTfBCD) Validation Accuracy", fontsize=12, fontweight="bold", pad=10)
    ax2.set_xlabel("Epochs", fontsize=11)
    ax2.set_ylabel("Accuracy (%)", fontsize=11)
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend(fontsize=10, loc="lower right")

    plt.suptitle("Teacher Training Performance (ViTfBCD, leak-fixed splits)",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"\n[SUCCESS] -> {save_path}")
    plt.close()


def run_epoch(model, loader, device, optimizer, criterion, scaler, use_amp, training: bool):
    model.train() if training else model.eval()
    total_loss = correct = total = 0

    ctx = torch.enable_grad() if training else torch.no_grad()
    with ctx:
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            with autocast(device_type=device.type, enabled=use_amp):
                logits = model(images)
                loss = criterion(logits, labels)

            if training:
                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()

            bs = images.size(0)
            total_loss += loss.item() * bs
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += bs

    return total_loss / total, correct / total


class SampleListDataset(Dataset):
    """Lightweight Dataset wrapping an explicit list of (path, label) samples,
    used to build arbitrary per-fold train/val splits without touching
    dataset.py's own train/val/test assignment logic."""
    def __init__(self, samples, transform):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, label = self.samples[index]
        image = self.loader(path)
        try:
            image = self.transform(image)
        except TypeError:
            import torchvision.transforms.functional as F
            if isinstance(image, torch.Tensor):

                from PIL import Image
                image = F.to_pil_image(image)
            image = self.transform(image)
            
        if not isinstance(image, torch.Tensor):
            import torchvision.transforms as T
            image = T.ToTensor()(image)
            
        return image, label, str(path)

    @property
    def num_classes(self):
        return 8


def make_patient_folds(patient_to_subtype: dict, k: int, seed: int = 42):
    """Stratified K-fold split at the PATIENT level, grouped per subtype so
    each fold gets a proportional share of each subtype's patients."""
    subtype_to_patients = defaultdict(list)
    for pid, subtype in patient_to_subtype.items():
        subtype_to_patients[subtype].append(pid)

    rng = random.Random(seed)
    folds = [[] for _ in range(k)]
    for subtype, plist in subtype_to_patients.items():
        plist = plist[:]
        rng.shuffle(plist)
        for i, pid in enumerate(plist):
            folds[i % k].append(pid)
    return folds


def get_patient_predictions(model, loader, device, num_classes=8):
    """
    Aggregates per-image softmax predictions up to the patient level (mean
    over all of a patient's images), returning per-patient (true_label,
    predicted_label) dicts. Used by evaluate_patient_level_samples() below,
    and also by main.py's train_k_fold() to build an aggregate, full-coverage
    report across all K-Fold folds combined (see aggregate_kfold_report() in
    main.py) -- a single fold's val set can be missing a rare subtype
    entirely (e.g. a subtype with only 2 patients pooled across 5 folds), so
    per-fold metrics alone can't measure that subtype, while patient_label
    aggregated across ALL folds covers every patient in the pool exactly
    once, giving complete coverage of every subtype regardless of any one
    fold's composition.
    """
    model.eval()
    patient_probs = defaultdict(lambda: torch.zeros(num_classes))
    patient_label = {}
    with torch.no_grad():
        idx = 0
        for images, labels, *_ in loader:
            bs = images.size(0)
            paths_labels = loader.dataset.samples[idx: idx + bs]
            idx += bs
            images = images.to(device, non_blocking=True)
            probs = F.softmax(model(images), dim=1).cpu()
            for (path, label), p in zip(paths_labels, probs):
                pid = _parse_patient_id(Path(path))
                patient_probs[pid] += p
                patient_label[pid] = label
    patient_pred = {pid: prob.argmax().item() for pid, prob in patient_probs.items()}
    return patient_label, patient_pred


def evaluate_patient_level_samples(model, loader, device, num_classes=8):
    """Same aggregation as evaluate_teacher_patient_level, but works for any
    loader whose .dataset has a .samples list (BreakHisDataset or
    SampleListDataset)."""
    patient_label, patient_pred = get_patient_predictions(model, loader, device, num_classes)
    if not patient_label:
        return 0.0
    correct = sum(int(patient_pred[pid] == patient_label[pid]) for pid in patient_label)
    return correct / len(patient_label)


def make_fold_loaders(train_samples, val_samples, config):
    stain_norm = build_stain_normalizer(config.get("stain_method")) if config.get("stain_method") else None
    train_tf = get_transforms("train", image_size=config["image_size"], stain_normalizer=stain_norm)
    val_tf = get_transforms("val", image_size=config["image_size"], stain_normalizer=stain_norm)

    train_ds = SampleListDataset(train_samples, train_tf)
    val_ds = SampleListDataset(val_samples, val_tf)

    from collections import Counter
    counts = Counter(l for _, l in train_ds.samples)
    train_ds.class_counts = dict(counts)
    # Effective-number-of-samples weighting (same as dataset.py's
    # make_weighted_sampler) instead of raw inverse-frequency -- raw inverse
    # frequency oversamples very rare classes so hard the model just memorizes
    # the same few patients repeatedly.
    sampler = make_weighted_sampler(train_ds, beta=config.get("sampler_beta", 0.999))

    train_loader = DataLoader(train_ds, batch_size=config["batch_size"], sampler=sampler,
                               num_workers=config.get("num_workers", 4), pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=config["batch_size"], shuffle=False,
                             num_workers=config.get("num_workers", 4), pin_memory=True)
    return train_loader, val_loader


def num_classes_for(config: dict) -> int:
    """Single source of truth: derive num_classes from config['mode'] so the
    model and dataset can never silently disagree about class count."""
    return 2 if config.get("mode", "multiclass") == "binary" else 8


def build_teacher(config, device):
    model = ViTfBCD(num_classes=num_classes_for(config), dropout=config.get("dropout", 0.5)).to(device)
    model.resize_position_embeddings()
    if config.get("freeze_blocks") is not None:
        freeze_backbone(model, config["freeze_blocks"])
    return model


def train_one_fold(fold_idx, train_samples, val_samples, config, device, output_dir):
    train_loader, val_loader = make_fold_loaders(train_samples, val_samples, config)
    model = build_teacher(config, device)

    criterion = nn.CrossEntropyLoss(label_smoothing=config.get("label_smoothing", 0.1))
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(trainable_params, lr=config["lr"], weight_decay=config["weight_decay"])
    scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=config["lr_factor"],
                                   patience=config["lr_patience"], min_lr=config["min_lr"])
    scaler = GradScaler(enabled=(device.type == "cuda"))
    use_amp = device.type == "cuda"

    best_patient_acc = 0.0
    no_improve = 0
    fold_ckpt = output_dir / f"fold{fold_idx}_best.pt"

    print(f"\n----- Fold {fold_idx}: train={len(train_samples)} imgs | val={len(val_samples)} imgs -----")

    for epoch in range(1, config["epochs"] + 1):
        t0 = time.time()
        tr_loss, tr_acc = run_epoch(model, train_loader, device, optimizer, criterion, scaler, use_amp, True)
        vl_loss, vl_acc = run_epoch(model, val_loader, device, optimizer, criterion, scaler, use_amp, False)
        vl_patient_acc = evaluate_patient_level_samples(model, val_loader, device, num_classes=num_classes_for(config))
        scheduler.step(vl_patient_acc)
        elapsed = time.time() - t0

        print(f"  [Fold {fold_idx}] Ep [{epoch:3d}/{config['epochs']}] "
              f"Train Loss: {tr_loss:.4f} | Val Loss: {vl_loss:.4f} | "
              f"Val Acc(img): {vl_acc:.4f} | Val Acc(patient): {vl_patient_acc:.4f} | Time: {elapsed:.1f}s")

        if vl_patient_acc > best_patient_acc:
            best_patient_acc = vl_patient_acc
            torch.save({"model_state_dict": model.state_dict(),
                        "best_val_patient_acc": best_patient_acc,
                        "config": config}, fold_ckpt)
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= config["patience"]:
                print(f"  [Fold {fold_idx}] Early stopping at epoch {epoch}.")
                break

    print(f"----- Fold {fold_idx} complete. Best patient val acc: {best_patient_acc:.4f} -----")
    return fold_ckpt, best_patient_acc


def run_kfold(config, device):
    root_dir = "/home/user/Proj-Ploy/vit_breast_cancer/data/BreaKHis_v1/histology_slides/breast/"
    output_dir = Path("/home/user/Proj-Ploy/vit_breast_cancer/outputs/kfold")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Reuse the exact same seed=42 patient split dataset.py always uses, so
    # the test set here is identical to every single-split run -- results
    # stay directly comparable. train+val become the k-fold pool.
    kw = dict(magnification="all", mode=config["mode"], seed=config["seed"],
              stain_method=config["stain_method"], image_size=config["image_size"])
    print("Loading existing patient split (train+val become the k-fold pool; test stays held out)...")
    train_ds_full = BreakHisDataset(root_dir, split="train", **kw)
    val_ds_full = BreakHisDataset(root_dir, split="val", **kw)
    test_ds = BreakHisDataset(root_dir, split="test", **kw)

    pool_samples = train_ds_full.samples + val_ds_full.samples

    patient_to_samples = defaultdict(list)
    patient_to_subtype = {}
    for path, label in pool_samples:
        pid = _parse_patient_id(Path(path))
        patient_to_samples[pid].append((path, label))
        patient_to_subtype[pid] = IDX_TO_SUBTYPE[label]

    pool_patients = list(patient_to_subtype.keys())
    print(f"K-fold POOL: {len(pool_patients)} patients, {len(pool_samples)} images")
    print(f"Held-out TEST (untouched all fold training): {len(test_ds)} images")

    folds = make_patient_folds(patient_to_subtype, config["k_folds"], seed=config["seed"])
    for i, f in enumerate(folds):
        print(f"  Fold {i}: {len(f)} val patients")

    fold_results, fold_ckpts = [], []
    for fold_idx in range(config["k_folds"]):
        val_patients = set(folds[fold_idx])
        train_patients = set(pool_patients) - val_patients
        train_samples = [s for pid in train_patients for s in patient_to_samples[pid]]
        val_samples = [s for pid in val_patients for s in patient_to_samples[pid]]

        ckpt, acc = train_one_fold(fold_idx, train_samples, val_samples, config, device, output_dir)
        fold_results.append(acc)
        fold_ckpts.append(ckpt)

    mean_acc = sum(fold_results) / len(fold_results)
    std_acc = (sum((a - mean_acc) ** 2 for a in fold_results) / len(fold_results)) ** 0.5
    print(f"\n{'='*65}")
    print(f" K-Fold Results: {[round(a, 4) for a in fold_results]}")
    print(f" Mean patient-level val acc: {mean_acc:.4f} +/- {std_acc:.4f}")
    print(f"{'='*65}\n")

    # ── Ensemble evaluation on the held-out test set ──────────────────────
    print("Evaluating ensemble of all fold models on the held-out TEST set...")
    test_loader = DataLoader(test_ds, batch_size=config["batch_size"], shuffle=False,
                              num_workers=4, pin_memory=True)

    ensemble_patient_probs = defaultdict(lambda: torch.zeros(num_classes_for(config)))
    patient_label = {}
    model = build_teacher(config, device)
    for ckpt_path in fold_ckpts:
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()
        with torch.no_grad():
            idx = 0
            for images, labels in test_loader:
                bs = images.size(0)
                paths_labels = test_ds.samples[idx: idx + bs]
                idx += bs
                images = images.to(device, non_blocking=True)
                probs = F.softmax(model(images), dim=1).cpu()
                for (path, label), p in zip(paths_labels, probs):
                    pid = _parse_patient_id(Path(path))
                    ensemble_patient_probs[pid] += p
                    patient_label[pid] = label

    correct = sum(int(prob.argmax().item() == patient_label[pid])
                  for pid, prob in ensemble_patient_probs.items())
    ensemble_acc = correct / len(ensemble_patient_probs) if ensemble_patient_probs else 0.0
    print(f"\n[ENSEMBLE] Patient-level TEST accuracy across {config['k_folds']} folds: "
          f"{ensemble_acc:.4f} ({len(ensemble_patient_probs)} patients)")

    with open(output_dir / "kfold_results.json", "w") as f:
        json.dump({
            "fold_val_patient_acc": fold_results,
            "mean_val_patient_acc": mean_acc,
            "std_val_patient_acc": std_acc,
            "ensemble_test_patient_acc": ensemble_acc,
            "config": config,
        }, f, indent=2)
    print(f"\nResults saved -> {output_dir / 'kfold_results.json'}")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device {device}")

    config = {
        "lr": 3e-5,
        "epochs": 40,
        "patience": 15,
        "weight_decay": 5e-2,
        "image_size": 384,
        "batch_size": 32,
        "mode": "binary",
        "stain_method": "macenko",
        "label_smoothing": 0.1, 
        "use_class_weights": True,#False
        "lr_factor": 0.5,
        "lr_patience": 3,
        "min_lr": 1e-6,
        "freeze_blocks": 2,   
        "dropout": 0.5,       
        "seed": 42,
        "k_folds": None,
    }

    if config.get("k_folds"):
        run_kfold(config, device)
        return

    num_classes = num_classes_for(config)
    output_dir = Path("/home/user/Proj-Ploy/vit_breast_cancer/outputs")
    output_dir.mkdir(parents=True, exist_ok=True)
    best_model_path = output_dir / "best_model.pt"  

    print("loading dataset (leak-fixed patient-wise splits)")
    train_loader, val_loader, test_loader = build_dataloaders(
        root_dir="/home/user/Proj-Ploy/vit_breast_cancer/data/BreaKHis_v1/histology_slides/breast/",
        config=config,
    )
    images, labels = next(iter(train_loader))
    print("image Batch Shape:", images.shape)

    print("Building ViTfBCD teacher model")
    model = ViTfBCD(num_classes=num_classes, dropout=config.get("dropout", 0.3)).to(device)
    model.resize_position_embeddings()

    if config.get("freeze_blocks") is not None:
        freeze_backbone(model, config["freeze_blocks"])

    class_weights = None
    if config["use_class_weights"]:
        print("Computing class weights from train_loader...")
        class_weights = compute_class_weights(train_loader, num_classes, device)

    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=config["label_smoothing"])
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(trainable_params, lr=config["lr"], weight_decay=config["weight_decay"])
    scheduler = ReduceLROnPlateau(
        optimizer, mode="max", factor=config["lr_factor"],
        patience=config["lr_patience"], min_lr=config["min_lr"],
    )
    scaler = GradScaler(enabled=(device.type == "cuda"))
    use_amp = device.type == "cuda"

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "val_patient_acc": []}
    best_val_patient_acc = 0.0
    no_improve = 0

    print(f"\n{'='*65}")
    print(" Teacher (ViTfBCD) Training — leak-fixed splits")
    print(" Checkpoint selection: PATIENT-LEVEL val accuracy (more stable/meaningful")
    print(" than per-image accuracy, since a patient's multiple images/magnifications")
    print(" are correlated, not independent samples)")
    print(f"{'='*65}\n")

    for epoch in range(1, config["epochs"] + 1):
        t0 = time.time()
        tr_loss, tr_acc = run_epoch(model, train_loader, device, optimizer, criterion, scaler, use_amp, True)
        vl_loss, vl_acc = run_epoch(model, val_loader, device, optimizer, criterion, scaler, use_amp, False)
        vl_patient_acc = evaluate_teacher_patient_level(model, val_loader, device)
        scheduler.step(vl_patient_acc)
        elapsed = time.time() - t0

        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["val_loss"].append(vl_loss)
        history["val_acc"].append(vl_acc)
        history["val_patient_acc"].append(vl_patient_acc)

        lr = optimizer.param_groups[0]["lr"]
        print(f"Ep [{epoch:3d}/{config['epochs']}] | Train Loss: {tr_loss:.4f} | "
              f"Val Loss: {vl_loss:.4f} | Val Acc(img): {vl_acc:.4f} | "
              f"Val Acc(patient): {vl_patient_acc:.4f} | LR: {lr:.6f} | Time: {elapsed:.1f}s")

        if vl_patient_acc > best_val_patient_acc:
            best_val_patient_acc = vl_patient_acc
            torch.save({
                "model_state_dict": model.state_dict(),
                "best_val_acc": vl_acc,
                "best_val_patient_acc": best_val_patient_acc,
                "config": config,
            }, best_model_path)
            no_improve = 0
            print(f"   Best PATIENT val acc: {vl_patient_acc:.4f} — saved -> {best_model_path}")
        else:
            no_improve += 1
            if no_improve >= config["patience"]:
                print(f"\n  Early stopping at epoch {epoch}.")
                break

    hist_path = output_dir / "teacher_history.json"
    with open(hist_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"\nTeacher training complete. Best patient-level val acc: {best_val_patient_acc:.4f}")

    chart_path = output_dir / "teacher_performance_curves.png"
    save_teacher_curves(history, str(chart_path))

    # Load best checkpoint and do a final honest check on val AND test
    print("\nLoading best teacher checkpoint for final evaluation...")
    ckpt = torch.load(best_model_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])

    print("\nFinal teacher evaluation:")
    final_val_acc = evaluate_teacher(model, val_loader, device)
    final_test_acc = evaluate_teacher(model, test_loader, device)
    print(f"  Val Acc (per-image):  {final_val_acc:.4f}")
    print(f"  Test Acc (per-image): {final_test_acc:.4f}")
    if abs(final_val_acc - final_test_acc) > 0.10:
        print("[Warning] Large val/test gap (>10 points) — worth double-checking "
            "the split for remaining imbalance or rare-subtype coverage issues.")

    print("\nPatient-level aggregated evaluation (mean softmax per patient):")
    evaluate_teacher_patient_level(model, val_loader, device)
    evaluate_teacher_patient_level(model, test_loader, device)


if __name__ == "__main__":
    main()