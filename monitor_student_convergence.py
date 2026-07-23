"""
monitor_student_convergence.py
Automates the same convergence check we did by hand for the BACH mixed
fine-tuning runs (is the tracked metric still climbing at the last epoch,
or has it genuinely plateaued?) and applies it to the student distillation
training history that DistillationTrainer.fit() saves to
outputs/distill_results/distill_history.json.

"Still improving at cutoff" is a real problem for a distillation run: if
the student's val_patient_acc is still climbing when training stops (max
epochs reached, no early stop triggered), the reported student accuracy is
an underestimate, and the "best" checkpoint kept is just whatever the last
epoch happened to be -- not necessarily anything close to the student's
ceiling.

Works on ANY history dict with parallel per-epoch lists (not hardcoded to
distillation only) -- pass --history_path to point it at a different run's
history.json (e.g. main.py's per-fold history.json files) if useful.

Usage:
    python monitor_student_convergence.py
    python monitor_student_convergence.py --history_path outputs/fold_1/history.json --metric val_acc
"""
import argparse
import json
from pathlib import Path

import numpy as np

DEFAULT_HISTORY_PATH = "outputs/distill_results/distill_history.json"
DEFAULT_METRIC = "val_patient_acc"


def analyze_convergence(values: list, metric_name: str, plateau_window: int = 5,
                         plateau_epsilon: float = 0.002, min_epochs_for_verdict: int = 8):
    """
    values: per-epoch list of the tracked metric (higher = better assumed;
            pass negated values for a loss-type metric where lower is better).
    plateau_window: how many of the most recent epochs to use for the
                    "has it flattened out" trend check.
    plateau_epsilon: per-epoch average change below this magnitude counts
                     as "flat" rather than "still trending".
    """
    n = len(values)
    result = {"n_epochs": n, "metric": metric_name}

    if n < 2:
        result["verdict"] = "insufficient_data"
        result["message"] = f"Only {n} epoch(s) recorded -- not enough to assess convergence."
        return result

    deltas = [values[i] - values[i - 1] for i in range(1, n)]
    best_epoch = int(np.argmax(values)) + 1  # 1-indexed for readability
    best_value = float(max(values))
    final_value = float(values[-1])

    window = min(plateau_window, n - 1)
    recent_deltas = deltas[-window:]
    recent_trend = float(np.mean(recent_deltas))
    recent_range = float(max(values[-window - 1:]) - min(values[-window - 1:]))

    is_best_near_end = best_epoch >= n - 1  # best epoch is the last or second-to-last
    still_trending = abs(recent_trend) > plateau_epsilon

    result.update({
        "best_epoch": best_epoch,
        "best_value": best_value,
        "final_epoch_value": final_value,
        "recent_window": window,
        "recent_avg_delta_per_epoch": recent_trend,
        "recent_range": recent_range,
        "best_epoch_is_near_end": is_best_near_end,
    })

    unique_values = sorted(set(round(v, 6) for v in values))
    is_coarse_metric = len(unique_values) <= max(3, n // 4)
    result["n_unique_values"] = len(unique_values)
    result["is_coarse_metric"] = is_coarse_metric

    if n < min_epochs_for_verdict:
        result["verdict"] = "too_early_to_tell"
        result["message"] = (f"Only {n} epochs so far -- wait for at least "
                              f"{min_epochs_for_verdict} before trusting a plateau/still-climbing verdict.")
    elif is_best_near_end and still_trending and recent_trend > 0:
        result["verdict"] = "still_improving"
        result["message"] = (
            f"Best {metric_name} ({best_value:.4f}) is at epoch {best_epoch}/{n} (the "
            f"last or second-to-last epoch), and the recent {window}-epoch trend is still "
            f"climbing (+{recent_trend:.4f}/epoch on average). Training was cut off before "
            "converging -- rerun with more epochs (and/or higher patience) to find the true "
            "ceiling before reporting this number as final."
        )
    elif still_trending and recent_trend < 0:
        result["verdict"] = "degrading"
        result["message"] = (
            f"{metric_name} has been getting WORSE over the last {window} epochs "
            f"({recent_trend:+.4f}/epoch on average), even though the best value "
            f"({best_value:.4f}) was reached at epoch {best_epoch}. This looks like "
            "overfitting past the best epoch -- as long as the best checkpoint (not the "
            "final epoch) is what gets loaded/reported, this is expected and fine; just "
            "don't report the LAST epoch's number."
        )
    else:
        result["verdict"] = "plateaued"
        result["message"] = (
            f"{metric_name} has flattened out over the last {window} epochs "
            f"(avg change {recent_trend:+.4f}/epoch, range {recent_range:.4f}) -- this looks "
            f"like genuine convergence around {best_value:.4f} (epoch {best_epoch}/{n}). "
            "Safe to treat this as the model's ceiling for this configuration."
        )

    if is_coarse_metric:
        result["message"] += (
            f" CAVEAT: this metric only took {len(unique_values)} distinct value(s) "
            f"({unique_values}) across all {n} epochs -- with this few achievable levels "
            "(typical of patient-level accuracy on a small validation set), 'plateaued' or "
            "'still improving' verdicts are not very meaningful; a single sample flipping "
            "prediction can swing the whole metric by a large step. Prefer a finer-grained "
            "companion metric (e.g. image-level val_acc, or val_loss) to judge convergence, "
            "and treat this metric as a coarse sanity check rather than the primary signal."
        )

    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--history_path", default=DEFAULT_HISTORY_PATH)
    parser.add_argument("--metric", default=DEFAULT_METRIC,
                         help="Key in the history JSON to analyze (e.g. val_patient_acc, "
                              "val_acc, val_loss). For loss-type metrics (lower=better), "
                              "pass --lower_is_better.")
    parser.add_argument("--lower_is_better", action="store_true",
                         help="Set this for loss-type metrics, where a decreasing trend is "
                              "the good direction (values are negated internally so the "
                              "'higher=better' logic above still applies).")
    parser.add_argument("--plateau_window", type=int, default=5)
    parser.add_argument("--plateau_epsilon", type=float, default=0.002)
    args = parser.parse_args()

    history_path = Path(args.history_path)
    if not history_path.exists():
        raise FileNotFoundError(
            f"{history_path} not found -- run the training script first "
            "(for the default path, that's run_distill.py, which DistillationTrainer.fit() "
            "saves distill_history.json from)."
        )

    with open(history_path) as f:
        history = json.load(f)

    if args.metric not in history:
        raise KeyError(f"'{args.metric}' not found in {history_path}. "
                        f"Available keys: {list(history.keys())}")

    values = history[args.metric]
    if args.lower_is_better:
        values = [-v for v in values]

    result = analyze_convergence(values, args.metric, args.plateau_window, args.plateau_epsilon)

    print(f"\n{'='*66}\nCONVERGENCE CHECK: {history_path} [{args.metric}]\n{'='*66}")
    for i, v in enumerate(history[args.metric], 1):
        marker = "  <- best" if i == result.get("best_epoch") else ""
        print(f"  Epoch {i:>3d}: {v:.4f}{marker}")

    print(f"\nVerdict: {result['verdict'].upper()}")
    print(result["message"])

    out_path = history_path.parent / f"convergence_check_{args.metric}.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\n[OK] Saved -> {out_path}")


if __name__ == "__main__":
    main()

    