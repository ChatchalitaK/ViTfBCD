"""
external_datasets.py
Dataset wrappers for BACH and PCam, standardized to the same
(image_tensor, binary_label) interface as BreakHisDataset in binary mode
(0=benign, 1=malignant) -- for cross-dataset generalization checks only.
Subtype-level comparison isn't meaningful across datasets since BACH/PCam
don't share BreakHis's 8 subtype labels.
"""
from pathlib import Path
from typing import Callable, Optional

from PIL import Image
from torch.utils.data import Dataset


BACH_CLASS_TO_BINARY = {
    "Normal": 0,     # benign
    "Benign": 0,     # benign
    "InSitu": 1,     # malignant
    "Invasive": 1,   # malignant
}


class BACHDataset(Dataset):
    """
    Expects the folder layout produced by download_external_datasets.py:
        root/ICIAR2018_BACH_Challenge/Photos/{Normal,Benign,InSitu,Invasive}/*.tif
    """
    def __init__(self, root: str, transform: Optional[Callable] = None):
        self.root = Path(root)
        photos_dir = self.root / "ICIAR2018_BACH_Challenge" / "Photos"
        if not photos_dir.exists():
            photos_dir = self.root  # allow pointing straight at Photos/ too
        if not photos_dir.exists():
            raise FileNotFoundError(
                f"Could not find BACH's Photos directory under {self.root}. "
                "Run download_external_datasets.py first."
            )
        self.transform = transform
        self.samples = []
        for cls_name, binary_label in BACH_CLASS_TO_BINARY.items():
            cls_dir = photos_dir / cls_name
            if not cls_dir.exists():
                continue
            for ext in ("*.tif", "*.tiff", "*.png"):
                for img_path in sorted(cls_dir.glob(ext)):
                    self.samples.append((str(img_path), binary_label))
        if not self.samples:
            raise FileNotFoundError(f"No images found under {photos_dir} -- check the download.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        image = Image.open(path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label

    @property
    def class_counts(self):
        from collections import Counter
        return Counter(label for _, label in self.samples)


class PCamBinaryDataset(Dataset):
    """
    Thin wrapper around torchvision.datasets.PCAM standardizing its
    (image, label) output through the same transform pipeline used
    elsewhere in this project, so preprocessing matches what the trained
    model expects. PCam's label is already binary (1=tumor tissue present,
    treated here as "malignant"; 0=no tumor, treated as "benign") --
    matches this project's binary convention (0=benign, 1=malignant).
    """
    def __init__(self, root: str, split: str = "test", transform: Optional[Callable] = None, download: bool = False):
        from torchvision.datasets import PCAM
        self._ds = PCAM(root=root, split=split, download=download)
        self.transform = transform

    def __len__(self):
        return len(self._ds)

    def __getitem__(self, idx):
        image, label = self._ds[idx]
        if self.transform:
            image = self.transform(image)
        return image, int(label)