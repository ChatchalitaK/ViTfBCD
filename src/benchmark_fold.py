import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

from dataset import (
    BreakHisDataset, _parse_patient_id, IDX_TO_SUBTYPE,
    assert_no_patient_leakage,
)
from train_teacher import make_patient_folds, SampleListDataset


def _patient_hash(patient_ids) -> str:
    """Order-independent checksum of a patient ID set, to detect drift."""
    joined = "|".join(sorted(patient_ids))
    return hashlib.sha256(joined.encode()).hexdigest()[:16]


def _build_pool(root_dir: str, config: dict):
    kw = dict(magnification=config["magnification"], mode=config["mode"], seed=config["seed"],
              stain_method=config.get("stain_method"), image_size=config["image_size"])
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
    return patient_to_samples, patient_to_subtype, test_ds


def freeze(root_dir: str, config: dict, out_path: str, k_folds: int = 5, fold_index: int = 0):
    patient_to_samples, patient_to_subtype, test_ds = _build_pool(root_dir, config)
    pool_patients = list(patient_to_subtype.keys())

    folds = make_patient_folds(patient_to_subtype, k_folds, seed=config["seed"])
    val_patients = sorted(folds[fold_index])
    train_patients = sorted(set(pool_patients) - set(val_patients))
    test_patients = sorted({_parse_patient_id(Path(p)) for p, _ in test_ds.samples})

    # Belt-and-suspenders: verify disjointness before ever writing this to disk.
    train_ds_check = SampleListDataset([s for pid in train_patients for s in patient_to_samples[pid]], None)
    val_ds_check = SampleListDataset([s for pid in val_patients for s in patient_to_samples[pid]], None)
    assert_no_patient_leakage({"benchmark_train": train_ds_check, "benchmark_val": val_ds_check, "held_out_test": test_ds})

    frozen = {
        "k_folds": k_folds,
        "fold_index": fold_index,
        "seed": config["seed"],
        "magnification": config["magnification"],
        "mode": config["mode"],
        "train_patients": train_patients,
        "val_patients": val_patients,
        "test_patients": test_patients,
        "train_patient_hash": _patient_hash(train_patients),
        "val_patient_hash": _patient_hash(val_patients),
        "test_patient_hash": _patient_hash(test_patients),
        "n_train_images": sum(len(patient_to_samples[p]) for p in train_patients),
        "n_val_images": sum(len(patient_to_samples[p]) for p in val_patients),
    }

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(frozen, f, indent=2)

    print(f"[FROZEN] Benchmark fold {fold_index}/{k_folds} written -> {out_path}")
    print(f"  train: {len(train_patients)} patients / {frozen['n_train_images']} images")
    print(f"  val  : {len(val_patients)} patients / {frozen['n_val_images']} images")
    print(f"  test : {len(test_patients)} patients (untouched, held out)")
    print("  This file is now the standardized benchmark. Commit it -- do not regenerate it "
          "by re-running make_patient_folds() elsewhere.")
    return frozen


def load_frozen_fold(root_dir: str, config: dict, frozen_path: str):
    """
    Reconstructs train/val SampleListDatasets STRICTLY from the patient ID
    lists saved in frozen_path -- never re-derives the split. Also verifies
    the current on-disk data still matches the frozen patient hashes, so
    silent dataset drift (added/removed images, re-run split logic, etc.)
    is caught immediately instead of quietly comparing runs against a
    benchmark that no longer means what it used to.
    """
    with open(frozen_path) as f:
        frozen = json.load(f)

    patient_to_samples, patient_to_subtype, test_ds = _build_pool(root_dir, config)

    for name, key in (("train", "train_patients"), ("val", "val_patients")):
        current_pids = set(frozen[key]) & set(patient_to_subtype.keys())
        missing = set(frozen[key]) - set(patient_to_subtype.keys())
        if missing:
            raise AssertionError(
                f"[BENCHMARK DRIFT] {len(missing)} patient(s) recorded in the frozen "
                f"'{name}' split are no longer present in the dataset on disk: "
                f"{sorted(missing)[:5]}... The frozen fold no longer matches the data."
            )
        actual_hash = _patient_hash(frozen[key])
        if actual_hash != frozen[f"{name}_patient_hash"]:
            raise AssertionError(f"[BENCHMARK DRIFT] hash mismatch for '{name}' patients.")

    train_samples = [s for pid in frozen["train_patients"] for s in patient_to_samples[pid]]
    val_samples = [s for pid in frozen["val_patients"] for s in patient_to_samples[pid]]

    train_ds = SampleListDataset(train_samples, None)
    val_ds = SampleListDataset(val_samples, None)
    assert_no_patient_leakage({"benchmark_train": train_ds, "benchmark_val": val_ds, "held_out_test": test_ds})

    print(f"[OK] Loaded frozen benchmark fold {frozen['fold_index']}/{frozen['k_folds']} "
          f"({len(frozen['train_patients'])} train / {len(frozen['val_patients'])} val patients). "
          "Split verified unchanged from the frozen file.")
    return train_samples, val_samples, test_ds


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="/home/user/Proj-Ploy/vit_breast_cancer/data/BreaKHis_v1/histology_slides/breast/")
    parser.add_argument("--out", default="/home/user/Proj-Ploy/vit_breast_cancer/outputs/benchmark_fold.json")
    parser.add_argument("--k_folds", type=int, default=5)
    parser.add_argument("--fold_index", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--freeze", action="store_true")
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()

    config = {
        "magnification": "all", "mode": "binary", "image_size": 384,
        "seed": args.seed, "stain_method": None,
    }

    if args.freeze:
        freeze(args.data_dir, config, args.out, k_folds=args.k_folds, fold_index=args.fold_index)
    elif args.verify:
        load_frozen_fold(args.data_dir, config, args.out)
    else:
        parser.print_help()