"""
freeze_bach_results.py
Freezes the BACH generalization results (from cross_validation.py's
cross_dataset_results.json) into an immutable outputs/cross_dataset/
bach_results_FINAL.json, then updates cross_validation_section.md to state
the table is FINALIZED (with a checksum/timestamp/checkpoint provenance
trail) rather than a draft waiting to be filled.

"Freeze" here means the same thing it means for benchmark_fold.py's patient
split: once written, this file should not be silently regenerated. Re-running
cross_validation.py and re-filling the report is fine during exploration, but
once a result is reported in the manuscript, it should be pinned to a
specific checkpoint + specific numbers with a visible paper trail -- if the
checkpoint changes later (retrained, fine-tuned, etc.), that must show up as
an explicit, visible mismatch here, not a silent overwrite.

Usage:
    python freeze_bach_results.py
    python freeze_bach_results.py --force --reason "retrained with sampler_beta=0.99"
"""
import argparse
import hashlib
import json
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path

RESULTS_PATH = Path("outputs/cross_dataset/cross_dataset_results.json")
FROZEN_PATH = Path("outputs/cross_dataset/bach_results_FINAL.json")
REPORT_MD_PATH = Path("cross_validation_section.md")


def _hash_dict(d: dict) -> str:
    canonical = json.dumps(d, sort_keys=True)
    return hashlib.sha256(canonical.encode()).hexdigest()[:16]


def _hash_file(path: Path) -> str:
    if not path.exists():
        return "file_not_found"
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def _git_commit() -> str:
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=5)
        if out.returncode == 0:
            return out.stdout.strip()
    except Exception:
        pass
    return "unknown (not a git repo, or git unavailable)"


def _checkpoint_path_from_cross_validation() -> str:
    """Reads CONFIG['checkpoint'] out of cross_validation.py without importing torch."""
    try:
        from cross_validation import CONFIG
        return CONFIG.get("checkpoint", "unknown")
    except Exception:
        return "unknown (could not import cross_validation.CONFIG)"


def freeze(results_path: Path, frozen_path: Path, force: bool, reason: str):
    if not results_path.exists():
        raise FileNotFoundError(
            f"{results_path} not found -- run cross_validation.py first."
        )
    with open(results_path) as f:
        results = json.load(f)

    bach = results.get("BACH", {})
    no_norm = bach.get("no_stain_norm")
    with_norm = bach.get("macenko")

    missing = [name for name, m in (("no_stain_norm", no_norm), ("macenko", with_norm)) if not m]
    if missing:
        raise ValueError(
            f"Cannot freeze -- BACH condition(s) missing or empty: {missing}. "
            "Run cross_validation.py with both stain_methods_to_compare entries first."
        )
    for name, m in (("no_stain_norm", no_norm), ("macenko", with_norm)):
        bad_fields = [k for k in ("accuracy", "precision", "sensitivity", "specificity", "f1_macro")
                      if m.get(k) is None or m.get(k) != m.get(k)]  # None or NaN
        if bad_fields:
            raise ValueError(f"Cannot freeze -- BACH/{name} has missing/NaN fields: {bad_fields}. "
                              "Fix the underlying run before finalizing the manuscript table.")

    bach_hash = _hash_dict(bach)

    if frozen_path.exists():
        with open(frozen_path) as f:
            existing = json.load(f)
        existing_hash = existing.get("bach_metrics_hash")
        if existing_hash == bach_hash:
            print(f"[OK] {frozen_path} already frozen with IDENTICAL numbers (hash={bach_hash}). "
                  "Nothing to do.")
            return existing
        if not force:
            raise RuntimeError(
                f"{frozen_path} already exists with DIFFERENT numbers "
                f"(existing hash={existing_hash}, new hash={bach_hash}).\n"
                "Overwriting a previously frozen, manuscript-reported result is a real "
                "change to what's being claimed -- re-run with --force --reason \"<why>\" "
                "if this is intentional (e.g. retrained checkpoint, bug fix)."
            )
        print(f"[WARNING] Overwriting frozen BACH results (hash {existing_hash} -> {bach_hash}). "
              f"Reason: {reason or '(none given)'}")

    checkpoint_path = _checkpoint_path_from_cross_validation()
    frozen = {
        "bach_metrics_hash": bach_hash,
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "checkpoint_path": checkpoint_path,
        "checkpoint_file_hash": _hash_file(Path(checkpoint_path)) if checkpoint_path != "unknown" else "unknown",
        "reason": reason or None,
        "bach": bach,
    }
    frozen_path.parent.mkdir(parents=True, exist_ok=True)
    with open(frozen_path, "w") as f:
        json.dump(frozen, f, indent=2)
    print(f"[OK] Froze BACH results -> {frozen_path}  (hash={bach_hash})")
    return frozen


def _fmt(value, pct=False, decimals=4):
    if value is None or value != value:
        return "n/a"
    return f"{value*100:.1f}%" if pct else f"{value:.{decimals}f}"


def _table_row(label: str, m: dict) -> str:
    return (
        f"| {label} | {m.get('n_samples', 'n/a')} "
        f"| {_fmt(m.get('accuracy'), pct=True)} | {_fmt(m.get('precision'), pct=True)} "
        f"| {_fmt(m.get('sensitivity'), pct=True)} | {_fmt(m.get('specificity'), pct=True)} "
        f"| {_fmt(m.get('f1_macro'))} | {_fmt(m.get('auc_roc'))} |"
    )


def update_manuscript(frozen: dict, report_path: Path):
    if not report_path.exists():
        print(f"[SKIP] {report_path} not found -- skipping manuscript update.")
        return

    bach = frozen["bach"]
    no_norm, with_norm = bach["no_stain_norm"], bach["macenko"]
    recovery_pts = (with_norm["accuracy"] - no_norm["accuracy"]) * 100

    text = report_path.read_text(encoding="utf-8")
    text = re.sub(r"\| No stain normalization \|.*\|", _table_row("No stain normalization", no_norm), text)
    text = re.sub(r"\| Macenko stain normalization \|.*\|", _table_row("Macenko stain normalization", with_norm), text)
    text = re.sub(
        r"\*\*Recovery from normalization\*\*:.*percentage\s*\npoints of accuracy\.",
        f"**Recovery from normalization**: {recovery_pts:+.2f} percentage points of accuracy.",
        text,
    )

    text = re.sub(
        r"\*\(This table is auto-filled by `fill_cross_dataset_report\.py`.*?can't drift from what was actually computed\.\)\*",
        "*(This table was filled and then frozen by `freeze_bach_results.py` -- see the "
        "FINALIZED banner above for the exact checkpoint and hash it corresponds to. "
        "It will not change on a routine `cross_validation.py` re-run.)*",
        text, flags=re.DOTALL,
    )

    banner = (
        f"\n> **FINALIZED** — this table is frozen (hash `{frozen['bach_metrics_hash']}`, "
        f"checkpoint `{frozen['checkpoint_path']}`, checkpoint file hash "
        f"`{frozen['checkpoint_file_hash']}`, frozen {frozen['frozen_at_utc']}"
        + (f", git commit `{frozen['git_commit']}`" if frozen['git_commit'] != "unknown" else "")
        + "). Re-running `cross_validation.py` will NOT change this table -- "
        "re-run `freeze_bach_results.py --force --reason \"...\"` explicitly if the "
        "underlying checkpoint or numbers genuinely change.\n"
    )
    if "**FINALIZED**" in text:
        text = re.sub(r"\n> \*\*FINALIZED\*\*.*?\n", banner, text, flags=re.DOTALL)
    else:
        text = text.replace("### Results\n", "### Results\n" + banner, 1)

    report_path.write_text(text, encoding="utf-8")
    print(f"[OK] Updated {report_path} with the finalized, frozen table.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default=str(RESULTS_PATH))
    parser.add_argument("--frozen", default=str(FROZEN_PATH))
    parser.add_argument("--report", default=str(REPORT_MD_PATH))
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--reason", default="")
    args = parser.parse_args()

    frozen = freeze(Path(args.results), Path(args.frozen), args.force, args.reason)
    update_manuscript(frozen, Path(args.report))


if __name__ == "__main__":
    main()