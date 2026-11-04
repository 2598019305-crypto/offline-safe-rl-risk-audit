#!/usr/bin/env python3
"""Recompute the CarCircle six-policy closed-loop table from episodes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


ARMS = ("BC-Safe", "BC-All", "standard_IQL", "Reward-only", "D10", "E100")
BASELINE = "Reward-only"


def cvar_upper(values: np.ndarray, fraction: float = 0.25) -> float:
    ordered = np.sort(np.asarray(values, dtype=float))
    count = max(1, int(np.ceil(fraction * len(ordered))))
    return float(ordered[-count:].mean())


def mean_ci(values: np.ndarray) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    if len(values) < 2:
        return float("nan"), float("nan")
    half = stats.t.ppf(0.975, len(values) - 1) * values.std(ddof=1) / np.sqrt(len(values))
    return float(values.mean() - half), float(values.mean() + half)


def analyze(csv_path: Path) -> tuple[pd.DataFrame, list[dict]]:
    frame = pd.read_csv(csv_path)
    required = {
        "arm",
        "train_seed",
        "env_seed",
        "reward_sum",
        "cost_sum",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"missing columns: {sorted(missing)}")
    if set(frame["arm"]) != set(ARMS):
        raise ValueError(f"unexpected arms: {sorted(set(frame['arm']))}")

    rows = []
    for (arm, seed), group in frame.groupby(["arm", "train_seed"], sort=True):
        costs = group["cost_sum"].to_numpy(dtype=float)
        returns = group["reward_sum"].to_numpy(dtype=float)
        rows.append(
            {
                "arm": arm,
                "train_seed": int(seed),
                "n_episodes": len(group),
                "cost_mean": float(costs.mean()),
                "cvar25": cvar_upper(costs),
                "return_mean": float(returns.mean()),
            }
        )
    seed_table = pd.DataFrame(rows)

    summary_rows = []
    for arm in ARMS:
        part = seed_table[seed_table["arm"] == arm]
        summary_rows.append(
            {
                "arm": arm,
                "n_train_seeds": int(len(part)),
                "cost_mean": float(part["cost_mean"].mean()),
                "cost_sd": float(part["cost_mean"].std(ddof=1)),
                "cvar25_mean": float(part["cvar25"].mean()),
                "cvar25_sd": float(part["cvar25"].std(ddof=1)),
                "return_mean": float(part["return_mean"].mean()),
                "return_sd": float(part["return_mean"].std(ddof=1)),
            }
        )

    pivots = {
        metric: seed_table.pivot(index="train_seed", columns="arm", values=metric)
        for metric in ("cost_mean", "cvar25", "return_mean")
    }
    comparisons = []
    for arm in ARMS:
        if arm == BASELINE:
            continue
        item = {"arm": arm, "reference": BASELINE}
        for metric, pivot in pivots.items():
            differences = (pivot[arm] - pivot[BASELINE]).to_numpy(dtype=float)
            low, high = mean_ci(differences)
            t_result = stats.ttest_1samp(differences, 0.0)
            item[f"delta_{metric}"] = float(differences.mean())
            item[f"delta_{metric}_sd"] = float(differences.std(ddof=1))
            item[f"delta_{metric}_ci95"] = [low, high]
            item[f"delta_{metric}_t_p"] = float(t_result.pvalue)
            item[f"delta_{metric}_better"] = int(
                np.sum(differences < 0) if metric != "return_mean" else np.sum(differences > 0)
            )
            item[f"delta_{metric}_worse"] = int(
                np.sum(differences > 0) if metric != "return_mean" else np.sum(differences < 0)
            )
        comparisons.append(item)

    for comparator in ("BC-Safe", "D10", "E100", "standard_IQL"):
        item = {"arm": comparator, "reference": "BC-All"}
        for metric, pivot in pivots.items():
            differences = (pivot[comparator] - pivot["BC-All"]).to_numpy(dtype=float)
            low, high = mean_ci(differences)
            item[f"delta_{metric}"] = float(differences.mean())
            item[f"delta_{metric}_sd"] = float(differences.std(ddof=1))
            item[f"delta_{metric}_ci95"] = [low, high]
            item[f"delta_{metric}_t_p"] = float(
                stats.ttest_1samp(differences, 0.0).pvalue
            )
            item[f"delta_{metric}_better"] = int(
                np.sum(differences < 0)
                if metric != "return_mean"
                else np.sum(differences > 0)
            )
            item[f"delta_{metric}_worse"] = int(
                np.sum(differences > 0)
                if metric != "return_mean"
                else np.sum(differences < 0)
            )
        comparisons.append(item)

    return pd.DataFrame(summary_rows), comparisons


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("csv", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    summary, comparisons = analyze(args.csv)
    summary.to_csv(args.output_dir / "carcircle_closed_loop_summary.csv", index=False)
    (args.output_dir / "carcircle_closed_loop_comparisons.json").write_text(
        json.dumps(comparisons, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(summary.to_string(index=False))
    print(json.dumps(comparisons, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
