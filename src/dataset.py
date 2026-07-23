"""
BreakHis Dataset Loader
Block 1: Data Augmentation & Class Rebalancing
Block 2: Image Restructuring (patch tokenization handled by ViT internally)
Note: Patient-wise splits, stain normalization, BACH/ICIAR2018 loader for cross-dataset validation

BreakHis folder structure expected:
    BreaKHis_v1/
    └── histology_slides/
        └── breast/
            ├── benign/
            │   └── SOB/
            │       ├── adenosis/         -> label: 'A'(binary: benign)
            │       ├── fibroadenoma/     -> label: 'F'
            │       ├── phyllodes_tumor/  -> label: 'PT'
            │       └── tubular_adenoma/  -> label: 'TA'
            └── malignant/
                └── SOB/
                    ├── ductal_carcinoma/     -> label: 'DC' (binary: malignant)
                    ├── lobular_carcinoma/    -> label: 'LC'
                    ├── mucinous_carcinoma/   -> label: 'MC'
                    └── papillary_carcinoma/  -> label: 'PC'
"""

import os
import sys
import random
import re
from pathlib import Path
from collections import Counter, defaultdict
from typing import Optional, Tuple, List, Dict
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms

# Stain normalization setup
try:
    import torchstain
    _TORCHSTAIN_AVAILABLE = True
except ImportError:
    _TORCHSTAIN_AVAILABLE = False
    print("[Warning] torchstain not installed. stain normalization disabled.")
    print("Install with: pip install torchstain")


# ── Label mappings ────────────────────────────────────────────────────────────
SUBTYPE_TO_IDX = {
    "adenosis": 0,
    "fibroadenoma": 1,
    "phyllodes_tumor": 2,
    "tubular_adenoma": 3,
    "ductal_carcinoma": 4,
    "lobular_carcinoma": 5,
    "mucinous_carcinoma": 6,
    "papillary_carcinoma": 7,
}
IDX_TO_SUBTYPE = {v: k for k, v in SUBTYPE_TO_IDX.items()}

BENIGN_SUBTYPES = {"adenosis", "fibroadenoma", "phyllodes_tumor", "tubular_adenoma"}
MALIGNANT_SUBTYPES = {"ductal_carcinoma", "lobular_carcinoma", "mucinous_carcinoma", "papillary_carcinoma"}

BINARY_TO_IDX = {"benign": 0, "malignant": 1}
IDX_TO_BINARY = {0: "benign", 1: "malignant"}

VIT_IMAGE_SIZE = 384

_PATIENT_RE = re.compile(r"SOB_[BM]_[A-Z0-9]+-(\d+-\d+)-\d+X-\d+", re.IGNORECASE)

def _parse_patient_id(path: Path) -> str:
    """Extract patient ID from BreakHis filename; fallback to parent structure if naming deviates."""
    m = _PATIENT_RE.match(path.stem)
    if m:
        return m.group(1)
    return path.parent.parent.name

DEFAULT_STAIN_REFERENCE = str(Path(__file__).resolve().parent / "assets" / "stain_reference.png")


def build_stain_normalizer(method: str = "macenko", reference_image_path: Optional[str] = None):
    """
    Return a callable norm(pil_image) -> PIL Image that applies stain
    normalization against a single fitted reference/template image.

    IMPORTANT: previously this function fell through without a `return`
    statement (so it always evaluated to None), and the inner `_normalize`
    never called the normalizer at all -- it just cast the image to a uint8
    tensor and handed it back UNCHANGED. Every caller that checked
    `if stain_normalizer is not None` therefore silently skipped
    normalization entirely, even with stain_method="macenko" set in config.
    This version fits once against a real reference image and actually
    normalizes every call.
    """
    if not _TORCHSTAIN_AVAILABLE:
        raise RuntimeError(
            "stain_method was requested but torchstain is not installed, so "
            "stain normalization CANNOT run. Install with `pip install torchstain` "
            "or explicitly pass stain_method=None if you intend to train without it. "
            "(Refusing to silently continue without normalization.)"
        )

    method = method.lower()
    if method == "macenko":
        normalizer = torchstain.normalizers.MacenkoNormalizer(backend="torch")
    elif method == "vahadane":
        normalizer = torchstain.normalizers.VahadaneNormalizer(backend="torch")
    elif method == "reinhard":
        normalizer = torchstain.normalizers.ReinhardNormalizer(backend="torch")
    else:
        raise ValueError(f"Unknown stain method '{method}'. Use macenko/vahadane/reinhard.")

    ref_path = Path(reference_image_path or DEFAULT_STAIN_REFERENCE)
    if not ref_path.exists():
        raise FileNotFoundError(
            f"Stain reference image not found at {ref_path}. Macenko/Vahadane/Reinhard "
            "normalization needs ONE fixed, representative, well-stained tile to fit "
            "against -- pick one BreakHis tile (e.g. a clean 40x sample) and pass its "
            "path via reference_image_path (or place it at the default path above). "
            "Every image, in every split and every external dataset (BACH etc.), must "
            "be normalized against this SAME reference for results to be comparable."
        )

    _to_tensor_255 = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x * 255.0),
    ])

    ref_img = Image.open(ref_path).convert("RGB")
    normalizer.fit(_to_tensor_255(ref_img))

    def _normalize(pil_img):
        if not isinstance(pil_img, Image.Image):
            raise TypeError(f"stain normalizer expects a PIL Image, got {type(pil_img)}")
        t = _to_tensor_255(pil_img)
        try:
            if method == "reinhard":
                norm = normalizer.normalize(I=t)
            else:
                norm, _H, _E = normalizer.normalize(I=t, stains=True)
        except Exception as e:
            # Tile has too little tissue / OD too low for Macenko's eigen
            # decomposition to converge (common on near-blank background
            # patches). Fall back to the original unnormalized tile for 
            # just this one image instead of crashing the whole DataLoader worker over it.
            return pil_img

        norm = norm.clamp(0, 255)
        if norm.shape[-1] == 3 and norm.shape[0] != 3:
            norm = norm.permute(2, 0, 1)
        return transforms.functional.to_pil_image(norm.byte())

    _normalize.method = method
    _normalize.reference_image_path = str(ref_path)
    print(f"[Stain Normalization] '{method}' normalizer FITTED and ACTIVE "
          f"(reference: {ref_path})")
    return _normalize


def verify_stain_normalization_active(stain_normalizer, sample_image_path: str,
                                       min_mean_abs_diff: float = 1.0) -> bool:
    """
    Sanity check that the stain-normalization pipeline actually changes pixels,
    instead of trusting that "not None" means "working" (which is exactly how
    the previous silent-no-op bug went unnoticed). Loads one real image, runs
    it through the normalizer, and asserts the output differs from the input
    by more than a trivial amount.
    """
    if stain_normalizer is None:
        raise AssertionError("verify_stain_normalization_active() called with no normalizer configured.")

    img = Image.open(sample_image_path).convert("RGB")
    normed = stain_normalizer(img)

    arr_before = np.asarray(img, dtype=np.float32)
    arr_after = np.asarray(normed.resize(img.size), dtype=np.float32)
    mean_abs_diff = float(np.mean(np.abs(arr_before - arr_after)))

    if mean_abs_diff < min_mean_abs_diff:
        raise AssertionError(
            f"[STAIN NORMALIZATION NOT ACTIVE] mean|before-after| = {mean_abs_diff:.4f} "
            f"(< {min_mean_abs_diff}) on {sample_image_path} -- the normalizer ran but "
            "barely touched the pixels, which is the signature of the earlier silent "
            "no-op bug. Investigate before trusting any stain-normalized results."
        )
    print(f"[OK] Stain normalization verified ACTIVE: mean|before-after| = {mean_abs_diff:.4f} "
          f"on {sample_image_path}")
    return True


def get_transforms(split: str, image_size: int = VIT_IMAGE_SIZE, stain_normalizer=None, strong: bool = False):
    """
    strong=True -> heavier augmentation used for minority-class samples during training,
    so rare classes see more visual diversity per epoch instead of the same few images
    repeated over and over by the sampler.
    """
    mean = [0.485, 0.456, 0.406]
    std  = [0.229, 0.224, 0.225]

    base = []
    if stain_normalizer is not None:
        base.append(transforms.Lambda(stain_normalizer))

    if split == "train":
        aug = [
            transforms.RandomCrop(512, pad_if_needed=True),
            transforms.Resize((image_size, image_size)),
            transforms.RandomRotation(180, fill=245),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
            transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.25, hue=0.08),
        ]
        if strong:
            aug += [
                transforms.RandomPerspective(distortion_scale=0.2, p=0.5, fill=245),
                transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.05),
            ]
        aug += [transforms.ToTensor()]
        if strong:
            aug += [transforms.RandomErasing(p=0.3, scale=(0.02, 0.1))]
        aug += [transforms.Normalize(mean=mean, std=std)]
        return transforms.Compose(base + aug)
    else:
        return transforms.Compose(base + [
            transforms.CenterCrop(512), 
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])

class BreakHisDataset(Dataset):
    def __init__(
        self,
        root_dir: str,
        magnification: str = all,
        mode: str = "binary",
        split: str = "train",
        split_ratio: Tuple[float, float] = (0.70, 0.15),
        seed: int = 42,
        transform=None,
        stain_method: Optional[str] = None,
        image_size: int = VIT_IMAGE_SIZE,
    ):
        self.root_dir = Path(root_dir)
        self.magnification = magnification.upper() if magnification != "all" else "all"
        self.mode = mode.lower().strip()
        self.split = split.lower().strip()
        self.seed = seed

        stain_norm = build_stain_normalizer(stain_method) if stain_method else None
        self._user_transform = transform  
        self._stain_norm = stain_norm
        self._image_size = image_size
        self.transform = transform or get_transforms(self.split, image_size=image_size, stain_normalizer=stain_norm)
        self.strong_transform = None
        self.minority_classes: set = set()

        assert self.mode in ("binary", "multiclass"), f"Invalid mode '{mode}'."
        assert self.split in ("train", "val", "test"), f"Invalid split '{split}'."
        assert 0.0 <= split_ratio[0] + split_ratio[1] <= 1.0, "Split ratios must sum to <= 1.0"

        self.samples: List[Tuple[str, int]] = []
        self._load_patient_wise(split_ratio, self.seed)

        if self.split == "train" and self._user_transform is None:
            counts = self.class_counts
            if counts:
                mean_count = sum(counts.values()) / len(counts)
                self.minority_classes = {c for c, n in counts.items() if n < 0.5 * mean_count}
                if self.minority_classes:
                    self.strong_transform = get_transforms(
                        self.split, image_size=self._image_size,
                        stain_normalizer=self._stain_norm, strong=True,
                    )
                    names = [IDX_TO_SUBTYPE.get(c, IDX_TO_BINARY.get(c, c)) for c in self.minority_classes]
                    print(f"[{self.split.upper():5s}] minority classes getting stronger augmentation: {names}")

    def _load_patient_wise(self, split_ratio, seed):
        patient_to_subtype = {}
        patient_to_samples = defaultdict(list)

        for category in ["benign", "malignant"]:
            cat_dir = self.root_dir / category / "SOB"
            if not cat_dir.exists():
                print(f"[WARNING] Not found: {cat_dir}")
                continue
            for subtype_dir in sorted(cat_dir.iterdir()):
                subtype = subtype_dir.name.lower()
                if subtype not in SUBTYPE_TO_IDX:
                    continue
                
                if self.magnification == "all":
                    img_paths = sorted(subtype_dir.rglob("*.png"))
                else:
                    img_paths = []
                    for d in sorted(subtype_dir.glob(f"**/{self.magnification}")):
                        img_paths.extend(sorted(d.glob("*.png")))

                label = (BINARY_TO_IDX["benign" if subtype in BENIGN_SUBTYPES else "malignant"]
                        if self.mode == "binary" else SUBTYPE_TO_IDX[subtype])

                for p in img_paths:
                    pid = _parse_patient_id(p)
                    patient_to_samples[pid].append((str(p), label))
                    patient_to_subtype[pid] = subtype

        if not patient_to_samples:
            raise RuntimeError(f"No images found under {self.root_dir}.")
        
        subtype_to_patients = defaultdict(list)
        for pid, subtype in patient_to_subtype.items():
            subtype_to_patients[subtype].append(pid)

        chosen_patients = []
        rng = random.Random(seed)
        leakage_warnings = []

        for subtype in sorted(subtype_to_patients.keys()):
            p_list = sorted(subtype_to_patients[subtype])
            rng.shuffle(p_list)
            n = len(p_list)
            
            if n >= 3:
                t_end = int(n * split_ratio[0])
                v_end = t_end + int(n * split_ratio[1])

                t_end = max(t_end, 1)
                if v_end <= t_end:
                    v_end = t_end + 1
                if v_end >= n:
                    v_end = n - 1
                    if v_end <= t_end:
                        t_end = v_end - 1

                train_patients = p_list[:t_end]
                val_patients = p_list[t_end:v_end]
                test_patients = p_list[v_end:]
            elif n == 2:
                train_patients = [p_list[0]]
                val_patients = [p_list[1]]
                test_patients = []
                leakage_warnings.append(
                    f"  '{subtype}': only 2 patients -> train=1, val=1, test=0 (no test coverage for this subtype)"
                )
            else:
                train_patients = [p_list[0]]
                val_patients = []
                test_patients = []
                leakage_warnings.append(
                    f"  '{subtype}': only 1 patient -> train=1, val=0, test=0 (no val/test coverage for this subtype)"
                )

            if self.split == "train":
                chosen_patients.extend(train_patients)
            elif self.split == "val":
                chosen_patients.extend(val_patients)
            else:
                chosen_patients.extend(test_patients)

        if leakage_warnings and self.split == "train":
            print("[WARNING] Some subtypes have too few patients for a clean 3-way split.")
            print("          These were previously LEAKING the same patient across splits")
            print("          (inflating val/test accuracy). Now fixed to avoid overlap:")
            for w in leakage_warnings:
                print(w)
            print("          Consider merging rare subtypes, gathering more patients,")
            print("          or using k-fold cross-validation instead of a fixed split.")

        for pid in chosen_patients:
            self.samples.extend(patient_to_samples[pid])

        print(f"[{self.split.upper():5s}] {len(self.samples):5d} images | {len(chosen_patients):3d} patients | mode={self.mode}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        image = Image.open(path).convert("RGB")
        if label in self.minority_classes and self.strong_transform is not None:
            image = self.strong_transform(image)
        elif self.transform:
            image = self.transform(image)
        return image, label

    @property
    def class_counts(self):
        return Counter(s[1] for s in self.samples)

    @property
    def num_classes(self):
        return 2 if self.mode == "binary" else 8

    @property
    def class_names(self):
        return list(IDX_TO_BINARY.values()) if self.mode == "binary" else list(IDX_TO_SUBTYPE.values())

class BACHdataset(Dataset):
    _LABEL_MAP = {
        "normal": 0, "benign": 0,
        "insitu": 1, "in situ": 1,
        "invasive": 1,
    }

    def __init__(
            self,
            bach_root: str,
            transform = None,
            stain_method: Optional[str] = None,
            extensions: Tuple[str, ...] = ("*.tif", "*.tiff", "*.png", "*.jpg"),
    ):
        self.root = Path(bach_root)
        stain_norm = build_stain_normalizer(stain_method) if stain_method else None
        self.transform = transform or get_transforms("test", stain_normalizer=stain_norm)
        self.samples: List[Tuple[str, int]] = []
        self._load(extensions)

    def _load(self, extensions):
        if not self.root.exists():
            print(f"[WARNING] BACH directory layout missing at: {self.root}")
            return

        for class_dir in sorted(self.root.iterdir()):
            if not class_dir.is_dir():
                continue
    
            dirname_cleaned = class_dir.name.lower().replace("_", "").strip()
            if dirname_cleaned in self._LABEL_MAP:
                target_label = self._LABEL_MAP[dirname_cleaned]
            else:
                continue

            for ext in extensions:
                for p in class_dir.glob(ext):
                    self.samples.append((str(p), target_label))
        
        if not self.samples:
            raise RuntimeError(f"No valid images found under {self.root}.")
        
        counts = self.class_counts
        print(f"[BACH] {len(self.samples)} images | benign={counts[0]} malignant={counts[1]}")
        
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        path, label = self.samples[idx]
        image = Image.open(path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label

    @property
    def class_names(self):
        return ["benign", "malignant"]
    
    @property
    def class_counts(self):
        return Counter(s[1] for s in self.samples)


def assert_no_patient_leakage(named_datasets: Dict[str, "BreakHisDataset"], strict: bool = True) -> bool:
    """
    Guarantees zero patient-wise leakage across an arbitrary number of splits.

    named_datasets: e.g. {"train": train_ds, "val": val_ds, "test": test_ds}
    strict=True  -> raises AssertionError on ANY overlap (use this in real runs)
    strict=False -> prints a warning and returns False instead of raising
                    (useful only for exploratory/debug scripts)
    """
    split_patients = {
        name: {_parse_patient_id(Path(p)) for p, _ in ds.samples}
        for name, ds in named_datasets.items()
    }

    names = list(split_patients.keys())
    leaks = {}
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            overlap = split_patients[names[i]] & split_patients[names[j]]
            if overlap:
                leaks[(names[i], names[j])] = overlap

    if leaks:
        lines = ["[LEAKAGE DETECTED] The same patient ID appears in more than one split:"]
        for (a, b), pids in leaks.items():
            sample = sorted(pids)[:5]
            lines.append(f"  {a} <-> {b}: {len(pids)} shared patient(s), e.g. {sample}")
        msg = "\n".join(lines)
        if strict:
            raise AssertionError(msg)
        print("[WARNING] " + msg)
        return False

    summary = ", ".join(f"{n}={len(p)} patients" for n, p in split_patients.items())
    print(f"[OK] Zero patient-wise leakage confirmed across {len(names)} splits ({summary}).")
    return True


def make_weighted_sampler(dataset: BreakHisDataset, beta: float = 0.999, seed: int = 42) -> WeightedRandomSampler:
    """
    Effective Number of Samples weighting (Cui et al., 'Class-Balanced Loss', 2019)
    instead of plain inverse frequency. Plain inverse frequency (total/count) makes
    very rare classes (e.g. adenosis) get resampled so aggressively that the model
    just memorizes the same handful of patients. Effective-number weighting still
    upweights rare classes but with diminishing returns as beta -> 1, which keeps
    the sampler from oversampling to the extreme.
    beta closer to 1.0 (e.g. 0.999) = softer correction; closer to 0.0 = plain inverse freq.

    NOTE: WeightedRandomSampler draws with replacement every epoch, so its
    output depends on whatever RNG it's handed. Previously no generator was
    passed, so it silently fell back to torch's GLOBAL default generator --
    fine if that's seeded, but a hidden source of run-to-run drift if it's
    not (which it wasn't, upstream in run_distill.py). Giving it its own
    seeded generator makes epoch-by-epoch sample order reproducible
    independent of whatever else touches the global RNG.
    """
    counts = dataset.class_counts
    eff_num = {cls: 1.0 - beta ** n for cls, n in counts.items()}
    cw = {cls: (1.0 - beta) / eff_num[cls] for cls in counts}
    sw = [cw[label] for _, label in dataset.samples]
    g = torch.Generator()
    g.manual_seed(seed)
    return WeightedRandomSampler(weights=sw, num_samples=len(sw), replacement=True, generator=g)


def _seed_worker(worker_id):
    """Reseed python's `random` and numpy per DataLoader worker process.

    torch reseeds its OWN per-worker RNG automatically, but `random.random()`
    and numpy's global RNG are NOT reseeded by default -- workers forked from
    the same parent can end up sharing correlated augmentation randomness
    (torchvision's RandomRotation/ColorJitter/etc. use `random`, not only
    torch), which is another silent source of run-to-run variance.
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def build_dataloaders(root_dir: str, config: dict) -> Tuple[DataLoader, DataLoader, DataLoader]:
    mag   = config.get("magnification", "all")
    mode  = config.get("mode", "multiclass")
    bs    = config.get("batch_size", 32)  
    nw    = config.get("num_workers", 4)
    isz   = config.get("image_size", VIT_IMAGE_SIZE)  
    seed  = config.get("seed", 42)
    stain_method = config.get("stain_method", None)
    sampler_beta = config.get("sampler_beta", 0.99)

    kw = dict(magnification=mag, mode=mode, seed=seed, stain_method=stain_method, image_size=isz)

    train_ds = BreakHisDataset(root_dir, split="train", **kw)
    val_ds   = BreakHisDataset(root_dir, split="val", **kw)
    test_ds  = BreakHisDataset(root_dir, split="test", **kw)

    assert_no_patient_leakage({"train": train_ds, "val": val_ds, "test": test_ds})

    if stain_method:
        verify_stain_normalization_active(
            train_ds._stain_norm, train_ds.samples[0][0]
        )

    sampler = make_weighted_sampler(train_ds, beta=sampler_beta, seed=seed)

    loader_gen = torch.Generator()
    loader_gen.manual_seed(seed)

    train_loader = DataLoader(train_ds, batch_size=bs, sampler=sampler, num_workers=nw, pin_memory=True,
                               worker_init_fn=_seed_worker, generator=loader_gen)
    val_loader   = DataLoader(val_ds,   batch_size=bs, shuffle=False, num_workers=nw, pin_memory=True,
                               worker_init_fn=_seed_worker, generator=loader_gen)
    test_loader  = DataLoader(test_ds,  batch_size=bs, shuffle=False, num_workers=nw, pin_memory=True,
                               worker_init_fn=_seed_worker, generator=loader_gen)

    return train_loader, val_loader, test_loader


def build_bach_loader(bach_root: str, config: dict) -> DataLoader:
    bs           = config.get("batch_size", 32)
    nw           = config.get("num_workers", 4)
    stain_method = config.get("stain_method", None)

    ds = BACHdataset(bach_root, stain_method=stain_method)
    return DataLoader(ds, batch_size=bs, shuffle=False, num_workers=nw, pin_memory=True)


target_splits = ["train", "val", "test"]

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    DATA_DIR = "/home/user/Proj-Ploy/vit_breast_cancer/data/BreaKHis_v1/histology_slides/breast/"
    MODE = "binary"  
    MAG = "all"   
    
    print("=" * 55)
    print("BreakHis Dataset Diagnostics")
    print("=" * 55)

    os.makedirs("/home/user/Proj-Ploy/vit_breast_cancer/outputs", exist_ok=True)
    
    config = {"magnification": MAG, "mode": MODE, "batch_size": 32, "image_size": 384}
    train_loader, val_loader, test_loader = build_dataloaders(DATA_DIR, config)
    
    splits = {
        "train": train_loader.dataset,
        "val": val_loader.dataset,
        "test": test_loader.dataset
    }

    print(f"\n{'Split':<8} {'Total': >7} Class Distribution")
    print("-" * 50)
    
    for split_name, ds in splits.items(): 
        counts = ds.class_counts
        dist = " ".join(f"{ds.class_names[i]}: {counts.get(i, 0)}" for i in range(len(ds.class_names)))
        print(f"{split_name:<8} {len(ds): >7} {dist}")

    train_ds = splits["train"]
    print(f"\nGenerating plotting diagrams to outputs folder...")

    fig, axes = plt.subplots(2, 4, figsize=(14, 7))
    axes = axes.flatten()

    mean = torch.tensor([0.485, 0.456, 0.406])
    std = torch.tensor([0.229, 0.224, 0.225])

    indices = random.sample(range(len(train_ds)), k=8)
    for i, idx in enumerate(indices):
        img, label = train_ds[idx]
        img_disp = img.permute(1, 2, 0) * std + mean
        img_disp = np.clip(img_disp.numpy(), 0, 1)
        
        class_name = train_ds.class_names[label]
        axes[i].imshow(img_disp)
        axes[i].set_title(f"[{class_name}]", color="darkblue", fontsize=12, fontweight="bold")
        axes[i].axis("off")

    plt.suptitle(f"Training Dataset Sample Images | mode={MODE} | mag={MAG}", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig("/home/user/Proj-Ploy/vit_breast_cancer/outputs/dataset_sample.png")
    plt.close()

    fig2, ax2 = plt.subplots(1, 3, figsize=(18, 5))
    for i, split_name in enumerate(target_splits):
        ds = splits[split_name]
        counts = ds.class_counts
        class_names = [ds.class_names[k] for k in sorted(counts.keys())]
        class_counts = [counts[k] for k in sorted(counts.keys())]
        
        ax2[i].bar(class_names, class_counts, color=["#4c72b0", "#55a868"])
        ax2[i].set_title(f"{split_name.capitalize()} Set", fontsize=12, fontweight="bold")
        ax2[i].set_xlabel("Class")
        ax2[i].set_ylabel("Samples")
        
        ax2[i].set_xticks(range(len(class_names)))
        ax2[i].set_xticklabels(class_names, rotation=45, ha="right")

    plt.suptitle(f"Patient-Wise Class Distribution Profile", fontsize=14, fontweight="bold")
    plt.tight_layout() 
    plt.savefig("/home/user/Proj-Ploy/vit_breast_cancer/outputs/dataset_distribution.png")
    plt.close()

    print("\n[SUCCESS] Script ran perfectly. Diagnostics visualization charts generated successfully.")