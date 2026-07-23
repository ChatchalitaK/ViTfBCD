import argparse
import hashlib
import zipfile
from pathlib import Path

BACH_URL = "https://zenodo.org/records/3632035/files/ICIAR2018_BACH_Challenge.zip?download=1"
BACH_MD5 = "8ae1801334aa943c44627c1eef3631b2"


def _download_with_progress(url: str, dest: Path):
    import requests
    from tqdm import tqdm

    if dest.exists():
        print(f"[SKIP] {dest.name} already exists.")
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    resp = requests.get(url, stream=True, timeout=30)
    resp.raise_for_status()
    total = int(resp.headers.get("content-length", 0))
    tmp = dest.with_suffix(dest.suffix + ".part")
    with open(tmp, "wb") as f, tqdm(total=total, unit="B", unit_scale=True, desc=dest.name) as bar:
        for chunk in resp.iter_content(chunk_size=1024 * 1024):
            f.write(chunk)
            bar.update(len(chunk))
    tmp.rename(dest)


def _md5(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def download_bach(output_dir: Path):
    bach_dir = output_dir / "BACH"
    bach_dir.mkdir(parents=True, exist_ok=True)
    zip_path = bach_dir / "ICIAR2018_BACH_Challenge.zip"

    print("Downloading BACH dataset (~10.4 GB, CC BY-NC-ND license -- "
          "cite Aresta et al. 2019 if used in a report/paper)...")
    _download_with_progress(BACH_URL, zip_path)

    print("Verifying checksum...")
    actual_md5 = _md5(zip_path)
    if actual_md5 != BACH_MD5:
        print(f"[Warning] MD5 mismatch (expected {BACH_MD5}, got {actual_md5}). "
              "The file may be corrupted, or Zenodo updated it -- proceeding "
              "anyway, but double-check the extracted images look right.")

    extracted_marker = bach_dir / "ICIAR2018_BACH_Challenge" / "Photos"
    if extracted_marker.exists():
        print("[SKIP] Already extracted.")
    else:
        print("Extracting (this can take a few minutes)...")
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(bach_dir)

    photos_dir = bach_dir / "ICIAR2018_BACH_Challenge" / "Photos"
    if not photos_dir.exists():
        raise RuntimeError(f"Expected {photos_dir} after extraction -- Zenodo's zip layout "
                            "may have changed; check the extracted folder manually.")
    for cls in ["Normal", "Benign", "InSitu", "Invasive"]:
        n = len(list((photos_dir / cls).glob("*.tif")))
        print(f"  {cls:10s}: {n} images")
    print(f"[OK] BACH ready at {photos_dir}")


def download_pcam(output_dir: Path):
    """Downloads via torchvision (handles Google Drive fetch + checksum internally)."""
    try:
        from torchvision.datasets import PCAM
    except ImportError:
        raise RuntimeError("torchvision is required for PCam download.")

    pcam_dir = output_dir / "PCam"
    pcam_dir.mkdir(parents=True, exist_ok=True)
    print("Downloading PCam (train/val/test splits, ~7-8 GB total)...")
    for split in ["train", "val", "test"]:
        print(f"  {split}...")
        PCAM(root=str(pcam_dir), split=split, download=True)
    print(f"[OK] PCam ready at {pcam_dir / 'pcam'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output_dir", default="/home/user/Proj-Ploy/vit_breast_cancer/data/external")
    parser.add_argument("--datasets", nargs="+", choices=["bach", "pcam"], default=["bach", "pcam"])
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if "bach" in args.datasets:
        download_bach(out_dir)
    if "pcam" in args.datasets:
        download_pcam(out_dir)