"""
Main Orchestrator Script for ViTfBCD for training and evaluation.
Connects dataset, model, and trainer to begin full execution of the pipeline.
Supports both Normal training and (patient-level) K-Fold Cross Validation modes.
"""
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from PIL.Image import Image
import torch
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader

from src.dataset import (
    build_dataloaders, BreakHisDataset, _parse_patient_id, IDX_TO_SUBTYPE, IDX_TO_BINARY,
    make_weighted_sampler, assert_no_patient_leakage, verify_stain_normalization_active,
)
from src.model import build_model
from src.trainer import Trainer
from src.train_teacher import make_patient_folds, SampleListDataset, evaluate_patient_level_samples, get_patient_predictions

def main():
    config = {
        "magnification": "all",
        "mode": "binary",
        "num_classes": 2,
        "model_size":  "base",
        "pretrained":  True,
        "image_size": 384,
        "batch_size": 32, 
        "epochs": 40,
        "lr": 2e-5,
        "weight_decay": 5e-2,
        "lr_schedule": "plateau",
        "lr_factor": 0.1,
        "lr_patience": 3,
        "patience": 10,
        "use_focal_loss": True,            
        "focal_gamma": 2.0,                
        "focal_alpha_mode": "none",
        "mixup_alpha": 0.4,               
        "sampler_beta": 0.99,              
        "es_delta": 1e-4,
        "label_smoothing": 0.1,
        "num_workers": 4,
        "seed": 42,
        "stain_method": "macenko",
    }
    # Set directories
    DATA_DIR = "/home/user/Proj-Ploy/vit_breast_cancer/data/BreaKHis_v1/histology_slides/breast/"
    OUTPUT_DIR = "/home/user/Proj-Ploy/vit_breast_cancer/outputs"

    # MODE SELECTOR 
    RUN_K_FOLD = False
    RUN_FINAL_MODEL = True

    # GPU Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"==========================================")
    print(f"STARTING BREAST CANCER CLASSIFICATION")
    print(f"Executing device selection: {device.type.upper()}")
    if device.type == "cuda":
        print(f"└─ Card Name: {torch.cuda.get_device_name(0)}")
    print(f"==========================================")

    if RUN_FINAL_MODEL:
        print("\n[MODE] Training FINAL model on the full patient pool "
              "(use after K-Fold has validated the recipe)...")
        train_final_model(DATA_DIR, config, OUTPUT_DIR, device)
    elif RUN_K_FOLD:
        print("\n[MODE] Switching to K-Fold Cross Validation (patient-level)...")
        train_k_fold(DATA_DIR, config, OUTPUT_DIR, device, k_folds=5)
    else:
        print("\n[MODE] Running Standard Single Split Training...")
        # Load 3 splits data
        print("\n loading and splitting datasets")
        train_loader, val_loader, test_loader = build_dataloaders(DATA_DIR, config)
        
        # Vision transformer
        print("\n Constructing VitfBCD Architecture" )
        model = build_model(config)
        model.resize_position_embeddings()
        info = model.get_model_info()
        print(f"   ├─ Model Parameters: {info['total_params']}")
        print(f"   └─ Trainable Block : {info['trainable_params']}")

        print("\n Initializing Trainer Engine...")
        trainer = Trainer(
            model=model,
            config=config,
            device=device,
            output_dir=OUTPUT_DIR
        )

        print("\n Commencing training cycle...")
        history = trainer.fit(train_loader, val_loader)
        print("\n Process completed successfully!")

class SafeLoaderWrapper:
    def __init__(self, dataloader):
        self.dataloader = dataloader

    def __iter__(self):
        for batch in self.dataloader:
            yield batch[0], batch[1]

    def __len__(self):
        return len(self.dataloader)


def train_k_fold(root_dir, config, output_dir, device, k_folds=5):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    kw = dict(magnification=config["magnification"], mode=config["mode"], seed=config["seed"],
              stain_method=config.get("stain_method"), image_size=config["image_size"])
    print("Loading existing patient split (train+val become the k-fold pool; test stays held out)...")
    train_ds_full = BreakHisDataset(root_dir, split="train", **kw)
    val_ds_full = BreakHisDataset(root_dir, split="val", **kw)
    test_ds = BreakHisDataset(root_dir, split="test", **kw)

    pool_samples = train_ds_full.samples + val_ds_full.samples

    from collections import defaultdict
    patient_to_samples = defaultdict(list)
    patient_to_subtype = {}
    for path, label in pool_samples:
        pid = _parse_patient_id(Path(path))
        patient_to_samples[pid].append((path, label))
        patient_to_subtype[pid] = IDX_TO_SUBTYPE[label]

    pool_patients = list(patient_to_subtype.keys())
    print(f" Dataset pool: {len(pool_patients)} patients, {len(pool_samples)} images "
          f"mapped for {k_folds}-Fold Split.\n")
    print(f" Held-out TEST (untouched across all folds): {len(test_ds)} images")

    folds = make_patient_folds(patient_to_subtype, k_folds, seed=config["seed"])
    for i, f in enumerate(folds):
        print(f"  Fold {i}: {len(f)} val patients")

    fold_results = []
    oof_patient_label, oof_patient_pred = {}, {}

    from src.dataset import get_transforms, build_stain_normalizer
    from torch.utils.data import WeightedRandomSampler
    from collections import Counter

    stain_norm = build_stain_normalizer(config.get("stain_method")) if config.get("stain_method") else None
    if stain_norm is not None:
        verify_stain_normalization_active(stain_norm, pool_samples[0][0])
    train_tf = get_transforms("train", image_size=config["image_size"], stain_normalizer=stain_norm)
    val_tf = get_transforms("val", image_size=config["image_size"], stain_normalizer=stain_norm)

    for fold in range(k_folds):
        print(f"\n " + "="*45)
        print(f" Starting Fold {fold + 1} / {k_folds}")
        print("="*48)

        val_patients = set(folds[fold])
        train_patients = set(pool_patients) - val_patients
        train_samples = [s for pid in train_patients for s in patient_to_samples[pid]]
        val_samples = [s for pid in val_patients for s in patient_to_samples[pid]]

        from PIL import Image

        train_ds = SampleListDataset(train_samples, train_tf)
        val_ds = SampleListDataset(val_samples, val_tf)

        train_ds.loader = lambda path: Image.open(path).convert("RGB")
        val_ds.loader = lambda path: Image.open(path).convert("RGB")

        assert_no_patient_leakage({"fold_train": train_ds, "fold_val": val_ds, "held_out_test": test_ds})

        counts = Counter(l for _, l in train_ds.samples)
        train_ds.class_counts = dict(counts)

        sampler = make_weighted_sampler(train_ds, beta=config.get("sampler_beta", 0.999))

        train_loader = DataLoader(
            train_ds, batch_size=config["batch_size"],
            sampler=sampler, num_workers=config["num_workers"], pin_memory=True
        )
        val_loader = DataLoader(
            val_ds, batch_size=config["batch_size"],
            shuffle=False, num_workers=config["num_workers"], pin_memory=True
        )

        model = build_model(config)
        model.resize_position_embeddings() 

        fold_output_dir = os.path.join(output_dir, f"fold_{fold + 1}")
        os.makedirs(fold_output_dir, exist_ok=True)

        trainer = Trainer(
            model=model,
            config=config,
            device=device,
            output_dir=fold_output_dir
        )

        print(f"-> Training Fold {fold + 1} with {len(train_samples)} train and "
            f"{len(val_samples)} val images ({len(train_patients)}/{len(val_patients)} patients)...")
        history = trainer.fit(train_loader, val_loader)
        trainer.load_best()

        best_img_acc = max(history["val_acc"])
        patient_label, patient_pred = get_patient_predictions(
            model, val_loader, device, num_classes=config["num_classes"]
        )
        patient_acc = (
            sum(int(patient_pred[pid] == patient_label[pid]) for pid in patient_label) / len(patient_label)
            if patient_label else 0.0
        )
        oof_patient_label.update(patient_label)
        oof_patient_pred.update(patient_pred)

        print(f" Fold {fold + 1} Complete! Best Val Acc (image): {best_img_acc:.4f} | "
            f"Val Acc (patient): {patient_acc:.4f}")
        fold_results.append(patient_acc)

    print("\n" + "═"*60)
    print(f"FINAL K-FOLD VALIDATION REPORT SUMMARY (patient-level accuracy)")
    print("═"*60)
    for i, res in enumerate(fold_results):
        print(f" ├─ Fold {i + 1} Patient Accuracy : {res:.4f} ({res*100:.2f}%)")
    print("─"*60)
    print(f"Mean System Accuracy : {np.mean(fold_results):.4f} ({np.mean(fold_results)*100:.2f}%)")
    print(f"Standard Deviation  : ±{np.std(fold_results):.4f}")
    print("═"*60)

    aggregate_kfold_report(oof_patient_label, oof_patient_pred, config["num_classes"], output_dir,
                            fold_results=fold_results, config=config)


def _report_classification(y_true: list, y_pred: list, num_classes: int, title: str,
                            out_path: Path, extra_note: str = ""):
    """
    Shared classification_report + confusion-matrix plot used by both
    aggregate_kfold_report() (out-of-fold, all folds combined) and
    train_final_model()'s test-set evaluation. Returns overall macro-F1.
    """
    try:
        from sklearn.metrics import f1_score, classification_report, confusion_matrix
    except ImportError:
        print("\n[Warning] scikit-learn not installed -- skipping report. "
              "Install with: pip install scikit-learn")
        return None
    import matplotlib.pyplot as plt

    if not y_true:
        print("\n[Warning] No predictions collected -- skipping report.")
        return None

    if num_classes == 2:
        names = [IDX_TO_BINARY[i] for i in range(num_classes)]
    else:
        names = [IDX_TO_SUBTYPE[i] for i in range(num_classes)]

    print("\n" + "═"*60)
    print(title)
    print("═"*60)
    if extra_note:
        print(extra_note)

    overall_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    report_dict = classification_report(y_true, y_pred, target_names=names,
                                         labels=list(range(num_classes)), zero_division=0,
                                         output_dict=True)
    print(f"\nMacro-F1: {overall_f1:.4f}\n")
    print(classification_report(y_true, y_pred, target_names=names,
                                 labels=list(range(num_classes)), zero_division=0))

    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1)

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(names)))
    ax.set_yticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha="right")
    ax.set_yticklabels(names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"{title}\nMacro-F1: {overall_f1:.4f}")
    for i in range(len(names)):
        for j in range(len(names)):
            ax.text(j, i, f"{cm[i, j]}", ha="center", va="center",
                     color="white" if cm_norm[i, j] > 0.5 else "black", fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f"[OK] Saved {out_path}")
    return overall_f1, report_dict


def aggregate_kfold_report(oof_patient_label: dict, oof_patient_pred: dict, num_classes: int,
                            output_dir: Path, fold_results: list = None, config: dict = None):
    pids = list(oof_patient_label.keys())
    y_true = [oof_patient_label[p] for p in pids]
    y_pred = [oof_patient_pred[p] for p in pids]
    overall_f1, report_dict = _report_classification(
        y_true, y_pred, num_classes,
        title="OUT-OF-FOLD AGGREGATE REPORT (patient-level, all folds combined)",
        out_path=Path(output_dir) / "oof_confusion_matrix.png",
        extra_note=f"Total patients evaluated: {len(pids)} (should equal the full pool size)",
    )

    if fold_results is not None:
        import json
        summary = {
            "sampler_beta": (config or {}).get("sampler_beta"),
            "seed": (config or {}).get("seed"),
            "mode": (config or {}).get("mode"),
            "per_fold_patient_accuracy": [float(r) for r in fold_results],
            "mean_patient_accuracy": float(np.mean(fold_results)),
            "std_patient_accuracy": float(np.std(fold_results)),
            "oof_macro_f1": float(overall_f1) if overall_f1 is not None else None,
            "oof_n_patients": len(pids),
            "oof_classification_report": report_dict,
        }
        out_json = Path(output_dir) / "kfold_summary.json"
        with open(out_json, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"[OK] Saved {out_json}  (use this to compare against other K-Fold runs, "
              "e.g. different sampler_beta values)")


def build_stratified_holdout(patient_to_subtype: dict, holdout_frac: float = 0.12, seed: int = 42):
    import random
    from collections import defaultdict
    rng = random.Random(seed)

    by_subtype = defaultdict(list)
    for pid, subtype in patient_to_subtype.items():
        by_subtype[subtype].append(pid)

    train_patients, val_patients = set(), set()
    for subtype, pids in by_subtype.items():
        pids = sorted(pids)
        rng.shuffle(pids)
        n = len(pids)
        if n <= 3:
            n_holdout = 0
        else:
            n_holdout = max(1, int(round(n * holdout_frac)))
            n_holdout = min(n_holdout, n - 2) 
        val_patients.update(pids[:n_holdout])
        train_patients.update(pids[n_holdout:])
    return train_patients, val_patients


def train_final_model(root_dir, config, output_dir, device):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    kw = dict(magnification=config["magnification"], mode=config["mode"], seed=config["seed"],
              stain_method=config.get("stain_method"), image_size=config["image_size"])
    print("Loading existing patient split (train+val become the full training pool; test stays held out)...")
    train_ds_full = BreakHisDataset(root_dir, split="train", **kw)
    val_ds_full = BreakHisDataset(root_dir, split="val", **kw)
    test_ds = BreakHisDataset(root_dir, split="test", **kw)

    pool_samples = train_ds_full.samples + val_ds_full.samples

    from collections import defaultdict
    patient_to_samples = defaultdict(list)
    patient_to_subtype = {}
    for path, label in pool_samples:
        pid = _parse_patient_id(Path(path))
        patient_to_samples[pid].append((path, label))
        patient_to_subtype[pid] = IDX_TO_SUBTYPE[label]

    train_patients, val_patients = build_stratified_holdout(patient_to_subtype, seed=config["seed"])
    train_samples = [s for pid in train_patients for s in patient_to_samples[pid]]
    val_samples = [s for pid in val_patients for s in patient_to_samples[pid]]
    print(f" Full-pool training: {len(train_patients)} train patients, "
          f"{len(val_patients)} checkpoint-selection patients "
          f"({len(train_samples)}/{len(val_samples)} images)")
    print(f" Held-out TEST (untouched, used for final report): {len(test_ds)} images\n")

    from src.dataset import get_transforms, build_stain_normalizer
    stain_norm = build_stain_normalizer(config.get("stain_method")) if config.get("stain_method") else None
    if stain_norm is not None:
        verify_stain_normalization_active(stain_norm, train_samples[0][0])
    train_tf = get_transforms("train", image_size=config["image_size"], stain_normalizer=stain_norm)
    eval_tf = get_transforms("val", image_size=config["image_size"], stain_normalizer=stain_norm)

    from PIL import Image
    from collections import Counter

    train_ds = SampleListDataset(train_samples, train_tf)
    val_ds = SampleListDataset(val_samples, eval_tf)
    train_ds.loader = lambda path: Image.open(path).convert("RGB")
    val_ds.loader = lambda path: Image.open(path).convert("RGB")

    assert_no_patient_leakage({"train": train_ds, "checkpoint_val": val_ds, "held_out_test": test_ds})

    counts = Counter(l for _, l in train_ds.samples)
    train_ds.class_counts = dict(counts)
    sampler = make_weighted_sampler(train_ds, beta=config.get("sampler_beta", 0.999))

    train_loader = DataLoader(train_ds, batch_size=config["batch_size"], sampler=sampler,
                               num_workers=config["num_workers"], pin_memory=True)
    if len(val_ds) == 0:
        raise ValueError("build_stratified_holdout() left an empty checkpoint-selection set -- "
                          "lower the threshold that forces small classes fully into train, "
                          "or increase holdout_frac.")
    val_loader = DataLoader(val_ds, batch_size=config["batch_size"], shuffle=False,
                             num_workers=config["num_workers"], pin_memory=True)

    model = build_model(config)
    model.resize_position_embeddings()

    trainer = Trainer(model=model, config=config, device=device, output_dir=str(output_dir))
    print("Training final model on the full pool...")
    trainer.fit(train_loader, val_loader)
    trainer.load_best()

    # Final, trustworthy numbers -- computed ONLY on the held-out test set.
    test_loader = DataLoader(test_ds, batch_size=config["batch_size"], shuffle=False,
                              num_workers=config["num_workers"], pin_memory=True)
    test_label, test_pred = get_patient_predictions(model, test_loader, device, num_classes=config["num_classes"])
    _report_classification(
        [test_label[p] for p in test_label], [test_pred[p] for p in test_label], config["num_classes"],
        title="FINAL MODEL — HELD-OUT TEST SET REPORT (patient-level)",
        out_path=output_dir / "final_test_confusion_matrix.png",
        extra_note=f"Total test patients evaluated: {len(test_label)}",
    )


if __name__ == "__main__":
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    main()