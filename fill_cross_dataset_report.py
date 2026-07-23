"""
fill_cross_dataset_report.py
Reads outputs/cross_dataset/cross_dataset_results.json (written by
cross_validation.py) and fills the BACH table + interpretation blank in
cross_validation_section.md -- so the report always reflects exactly what
was measured, never a hand-copied (and possibly stale/typo'd) number.

Usage:
    python fill_cross_dataset_report.py \
        --results outputs/cross_dataset/cross_dataset_results.json \
        --template cross_validation_section.md \
        --out cross_validation_section.md   # overwrite in place, or point elsewhere
"""
import argparse
import json
import re
from pathlib import Path

RECOVERY_TRIGGER_PTS = 5.0  # matches the ">=5 points" convention already stated in the .md


def _fmt(value, pct=False, decimals=4):
    if value is None or value != value:  # None or NaN
        return "n/a"
    return f"{value*100:.1f}%" if pct else f"{value:.{decimals}f}"


def build_table_row(condition_label: str, m: dict) -> str:
    if m is None:
        return f"| {condition_label} | n/a | n/a | n/a | n/a | n/a | n/a | n/a |"
    return (
        f"| {condition_label} "
        f"| {m.get('n_samples', 'n/a')} "
        f"| {_fmt(m.get('accuracy'), pct=True)} "
        f"| {_fmt(m.get('precision'), pct=True)} "
        f"| {_fmt(m.get('sensitivity'), pct=True)} "
        f"| {_fmt(m.get('specificity'), pct=True)} "
        f"| {_fmt(m.get('f1_macro'))} "
        f"| {_fmt(m.get('auc_roc'))} |"
    )


def build_interpretation(no_norm: dict, with_norm: dict) -> str:
    if not no_norm or not with_norm:
        return ("One or both BACH conditions are missing from cross_dataset_results.json -- "
                "re-run cross_validation.py with both stain_methods_to_compare entries before "
                "drawing a conclusion here.")

    recovery_pts = (with_norm["accuracy"] - no_norm["accuracy"]) * 100

    if recovery_pts >= RECOVERY_TRIGGER_PTS:
        return (
            f"Stain normalization recovered {recovery_pts:.1f} accuracy points on BACH, "
            f"meeting this project's \u2265{RECOVERY_TRIGGER_PTS:.0f}-point 'meaningful recovery' "
            "threshold. This is consistent with stain/color appearance driving a real part of "
            "the no-normalization gap -- the model likely relies on BreakHis-specific staining "
            "cues that Macenko normalization helps correct for on an external dataset."
        )
    elif recovery_pts <= -RECOVERY_TRIGGER_PTS:
        return (
            f"Stain normalization REDUCED accuracy by {abs(recovery_pts):.1f} points on BACH "
            "rather than helping. This matches the caution already noted in cross_validation.py: "
            "a Macenko reference fitted only on BreakHis tiles can distort BACH's different "
            "staining/scanner characteristics rather than correct for them. Consider reporting "
            "the no-normalization BACH result as primary, and/or fitting a BACH-specific "
            "reference tile if normalization is still desired for this dataset."
        )
    else:
        return (
            f"Stain normalization changed BACH accuracy by only {recovery_pts:+.1f} points -- "
            "not a clear signal in either direction. If a domain-shift gap exists between "
            "in-domain BreakHis and BACH accuracy, it is likely driven by something other than "
            "stain appearance (e.g. scanner resolution, tissue preparation, or the different "
            "benign/malignant boundary definitions noted in the caveats below), and stain "
            "normalization alone should not be expected to close it."
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default="outputs/cross_dataset/cross_dataset_results.json")
    parser.add_argument("--template", default="cross_validation_section.md")
    parser.add_argument("--out", default=None, help="Defaults to overwriting --template in place.")
    args = parser.parse_args()

    results_path = Path(args.results)
    template_path = Path(args.template)
    out_path = Path(args.out) if args.out else template_path

    if not results_path.exists():
        raise FileNotFoundError(
            f"{results_path} not found -- run cross_validation.py first (it writes this file "
            "at the end of main())."
        )
    if not template_path.exists():
        raise FileNotFoundError(f"{template_path} not found.")

    with open(results_path) as f:
        results = json.load(f)

    bach = results.get("BACH", {})
    no_norm = bach.get("no_stain_norm")
    with_norm = bach.get("macenko")

    if no_norm is None or with_norm is None:
        print(f"[WARNING] Missing BACH condition(s) in {results_path} "
              f"(no_stain_norm={'present' if no_norm else 'MISSING'}, "
              f"macenko={'present' if with_norm else 'MISSING'}). "
              "Filling what's available; re-run cross_validation.py with both stain "
              "methods in CONFIG['stain_methods_to_compare'] for a complete section.")

    text = template_path.read_text(encoding="utf-8")

    # Replace the two table rows (they're the only two lines starting with
    # "| No stain normalization" / "| Macenko stain normalization").
    no_norm_row = build_table_row("No stain normalization", no_norm)
    with_norm_row = build_table_row("Macenko stain normalization", with_norm)

    text = re.sub(r"\| No stain normalization \|.*\|", no_norm_row, text)
    text = re.sub(r"\| Macenko stain normalization \|.*\|", with_norm_row, text)

    # Recovery line.
    if no_norm and with_norm:
        recovery_pts = (with_norm["accuracy"] - no_norm["accuracy"]) * 100
        recovery_str = f"{recovery_pts:+.2f}"
    else:
        recovery_str = "n/a"
    text = re.sub(
        r"\*\*Recovery from normalization\*\*: `<fill: results\.summary\.recovery_pts>` percentage\s*\npoints of accuracy\.",
        f"**Recovery from normalization**: {recovery_str} percentage points of accuracy.",
        text,
    )

    # Interpretation blank.
    interpretation = build_interpretation(no_norm, with_norm)
    text = re.sub(
        r"- `<fill: one or two sentences stating which of these patterns was observed,\s*\n\s*once the numbers above are in>`",
        f"- {interpretation}",
        text,
    )

    out_path.write_text(text, encoding="utf-8")
    print(f"[OK] Filled -> {out_path}")
    if no_norm and with_norm:
        print(f"     no_norm accuracy={no_norm['accuracy']:.4f}  "
              f"macenko accuracy={with_norm['accuracy']:.4f}  "
              f"recovery={recovery_str} pts")


if __name__ == "__main__":
    main()