"""Standalone cost-critic calibration audit from critic_pairs.csv.

Usage:
  python critic_calibration.py --input critic_pairs.csv --output-dir .

Recomputes Pearson, Spearman, MAE, RMSE, bias, decile calibration bins and a
high-cost miss rate directly from the raw (predicted, target) pairs.
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def rankdata(x):
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x), dtype=float)
    ranks[order] = np.arange(1, len(x) + 1, dtype=float)
    return ranks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output-dir", default=".")
    ap.add_argument("--bins", type=int, default=10)
    args = ap.parse_args()

    pred, real = [], []
    with open(args.input, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            pred.append(float(row["predicted_cost"]))
            real.append(float(row["actual_cost_target"]))
    pred = np.asarray(pred, dtype=float)
    real = np.asarray(real, dtype=float)

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    report = {
        "n_pairs": int(len(pred)),
        "pred_mean": float(pred.mean()),
        "pred_std": float(pred.std()),
        "target_mean": float(real.mean()),
        "target_std": float(real.std()),
        "pearson": float(np.corrcoef(pred, real)[0, 1]) if len(pred) > 1 else None,
        "spearman": float(np.corrcoef(rankdata(pred), rankdata(real))[0, 1])
        if len(pred) > 1
        else None,
        "mae": float(np.mean(np.abs(pred - real))),
        "rmse": float(np.sqrt(np.mean((pred - real) ** 2))),
        "bias_pred_minus_target": float(np.mean(pred - real)),
    }

    # decile calibration by predicted-risk quantile
    order = np.argsort(pred, kind="mergesort")
    chunks = np.array_split(order, min(args.bins, len(order)))
    bins = []
    for i, idx in enumerate(chunks):
        bins.append(
            {
                "bin": i,
                "n": int(len(idx)),
                "pred_mean": float(pred[idx].mean()),
                "target_mean": float(real[idx].mean()),
                "target_p90": float(np.quantile(real[idx], 0.9)),
            }
        )
    report["calibration_bins"] = bins

    # high-target-cost detection: top 25% of actual targets
    thr_real = float(np.quantile(real, 0.75))
    thr_pred = float(np.quantile(pred, 0.75))
    high = real >= thr_real
    flagged = pred >= thr_pred
    report["high_cost_threshold_target_q75"] = thr_real
    report["high_cost_threshold_pred_q75"] = thr_pred
    report["high_cost_count"] = int(high.sum())
    report["high_cost_detection_rate"] = float(flagged[high].mean()) if high.any() else None
    report["high_cost_miss_rate"] = (
        float(1.0 - flagged[high].mean()) if high.any() else None
    )
    report["false_alarm_rate"] = (
        float(flagged[~high].mean()) if (~high).any() else None
    )

    (out / "critic_calibration_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    with open(out / "critic_calibration_bins.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(bins[0].keys()))
        w.writeheader()
        w.writerows(bins)
    print(json.dumps({k: v for k, v in report.items() if k != "calibration_bins"}, indent=2))


if __name__ == "__main__":
    main()
