import argparse
import random
import shutil
from pathlib import Path

import numpy as np
from PIL import Image

DEFAULT_OUT_PATH = Path(__file__).resolve().parent / "src" / "assets" / "stain_reference.png"


def _score_candidate(path: Path) -> float:
    """
    Lower is more 'typical': penalizes tiles that are too dark, too bright,
    too low-contrast (likely background/whitespace), or too saturated
    (likely a staining artifact) -- crude but reasonable heuristics for
    picking one broadly representative tile without hand-inspecting hundreds.
    """
    try:
        img = Image.open(path).convert("RGB")
    except Exception:
        return float("inf")
    arr = np.asarray(img.resize((128, 128)), dtype=np.float32) / 255.0

    mean_brightness = arr.mean()
    contrast = arr.std()
    # H&E should have real color variance (not near-grayscale background/whitespace)
    saturation_proxy = (arr.max(axis=-1) - arr.min(axis=-1)).mean()

    brightness_penalty = abs(mean_brightness - 0.55)   # H&E tiles trend mid-bright
    contrast_penalty = max(0.0, 0.12 - contrast)         # too flat = probably blank/background
    saturation_penalty = max(0.0, 0.08 - saturation_proxy)  # too gray = probably background

    return brightness_penalty + contrast_penalty + saturation_penalty


def auto_pick(data_dir: str, n_candidates: int = 60, seed: int = 42) -> Path:
    data_dir = Path(data_dir)
    candidates = sorted(data_dir.rglob("*40X*/*.png")) or sorted(data_dir.rglob("*40*/*.png"))
    if not candidates:
        candidates = sorted(data_dir.rglob("*.png"))
    if not candidates:
        raise FileNotFoundError(f"No .png images found under {data_dir} -- check the path.")

    rng = random.Random(seed)
    sample = rng.sample(candidates, min(n_candidates, len(candidates)))

    scored = sorted(((_score_candidate(p), p) for p in sample), key=lambda t: t[0])
    best_score, best_path = scored[0]
    print(f"Scanned {len(sample)} candidate tiles, best score={best_score:.4f}")
    print(f"Picked: {best_path}")
    return best_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manual", type=str, default=None, help="Path to a specific image to use as-is.")
    parser.add_argument("--auto", action="store_true", help="Auto-pick from --data_dir (default mode).")
    parser.add_argument("--data_dir", type=str,
                         default="/home/user/Proj-Ploy/vit_breast_cancer/data/BreaKHis_v1/histology_slides/breast/")
    parser.add_argument("--out", type=str, default=str(DEFAULT_OUT_PATH))
    parser.add_argument("--n_candidates", type=int, default=60)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if args.manual:
        src_path = Path(args.manual)
        if not src_path.exists():
            raise FileNotFoundError(src_path)
    else:
        src_path = auto_pick(args.data_dir, n_candidates=args.n_candidates, seed=args.seed)

    img = Image.open(src_path).convert("RGB")
    img.save(out_path)
    print(f"\n[OK] Saved reference -> {out_path}")
    print(f"     Source tile: {src_path}")
    print("     Open the saved PNG and eyeball it once -- it should look like a "
          "normal, clearly-stained H&E tile, not background/whitespace or an outlier.")
    print("     Do not change this file later without re-running everything that "
          "depends on stain normalization (training AND evaluation) -- a different "
          "reference makes results incomparable to anything computed before the change.")


if __name__ == "__main__":
    main()