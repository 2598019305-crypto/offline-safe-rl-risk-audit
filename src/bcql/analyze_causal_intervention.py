"""Audit the A6 Monte-Carlo-cost intervention against the frozen BCQ-Lag runs.

This script never trains a model. It reads the original 100-episode evaluation
files and the A6 100-episode files, recomputes episode-level metrics from raw
records, and performs a two-stage paired cluster bootstrap:

1. resample evaluation seeds 149000..149099 within each training seed;
2. resample the three training seeds across training runs.

The 300 evaluation episodes are therefore never treated as 300 independent
training seeds.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ARMS = (
    ("constrained", "BCQ-Lag"),
    ("lambda0", "BCQ-Lag (lambda=0)"),
    ("a6", "A6 MC-cost intervention"),
)
METRIC_COLUMNS = (
    "mean_cost",
    "cvar25_cost",
    "safe_rate",
    "mean_return",
    "normalized_cost_mean",
    "normalized_reward_mean",
    "mean_length",
)


def cvar25(values: Iterable[float]) -> float:
    arr = np.sort(np.asarray(list(values), dtype=float))[::-1]
    k = max(1, int(math.ceil(0.25 * len(arr))))
    return float(arr[:k].mean())


def summarize(records: list[dict]) -> dict:
    costs = np.asarray([float(r["episode_cost"]) for r in records])
    returns = np.asarray([float(r["episode_return"]) for r in records])
    norm_costs = np.asarray([float(r["normalized_cost"]) for r in records])
    norm_returns = np.asarray(
        [float(r["normalized_reward"]) for r in records]
    )
    lengths = np.asarray([float(r["episode_length"]) for r in records])
    return {
        "n_episodes": int(len(records)),
        "mean_cost": float(costs.mean()),
        "cvar25_cost": cvar25(costs),
        "safe_rate": float(np.mean(costs <= 10.0)),
        "mean_return": float(returns.mean()),
        "normalized_cost_mean": float(norm_costs.mean()),
        "normalized_reward_mean": float(norm_returns.mean()),
        "mean_length": float(lengths.mean()),
    }


def load_evaluation(path: Path, arm: str, training_seed: int) -> list[dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    records = payload.get("records", [])
    if not records:
        raise ValueError(f"no episode records in {path}")
    aliases = {
        "episode_return": "reward",
        "episode_cost": "cost",
        "episode_length": "length",
    }
    result = []
    for raw in records:
        row = dict(raw)
        for target, source in aliases.items():
            if target not in row and source in row:
                row[target] = row[source]
        row.update(
            {
                "arm": arm,
                "training_seed": int(training_seed),
                "eval_seed": int(row["eval_seed"]),
                "safe": int(float(row["episode_cost"]) <= 10.0),
            }
        )
        result.append(row)
    unique = {r["eval_seed"] for r in result}
    if len(unique) != len(result):
        raise ValueError(f"duplicate evaluation seeds in {path}")
    return result


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def paired_delta(left: list[dict], right: list[dict]) -> dict:
    lmap = {int(r["eval_seed"]): r for r in left}
    rmap = {int(r["eval_seed"]): r for r in right}
    keys = sorted(lmap.keys() & rmap.keys())
    if not keys:
        raise ValueError("paired evaluation has no common seeds")
    le = np.asarray([float(lmap[k]["episode_cost"]) for k in keys])
    re = np.asarray([float(rmap[k]["episode_cost"]) for k in keys])
    lr = np.asarray([float(lmap[k]["episode_return"]) for k in keys])
    rr = np.asarray([float(rmap[k]["episode_return"]) for k in keys])
    lnc = np.asarray([float(lmap[k]["normalized_cost"]) for k in keys])
    rnc = np.asarray([float(rmap[k]["normalized_cost"]) for k in keys])
    lnr = np.asarray([float(lmap[k]["normalized_reward"]) for k in keys])
    rnr = np.asarray([float(rmap[k]["normalized_reward"]) for k in keys])
    return {
        "n_pairs": len(keys),
        "delta_cost": float((le - re).mean()),
        "delta_cvar25": cvar25(le) - cvar25(re),
        "delta_safe_rate": float(np.mean(le <= 10.0) - np.mean(re <= 10.0)),
        "delta_return": float((lr - rr).mean()),
        "delta_normalized_cost": float((lnc - rnc).mean()),
        "delta_normalized_reward": float((lnr - rnr).mean()),
    }


def metric_vector(left: list[dict], right: list[dict]) -> dict[str, float]:
    return paired_delta(left, right)


def hierarchical_bootstrap(
    by_arm_seed: dict[tuple[str, int], list[dict]],
    left_arm: str,
    right_arm: str,
    reps: int,
    rng: np.random.Generator,
) -> tuple[dict, list[dict]]:
    seeds = sorted(
        {seed for arm, seed in by_arm_seed if arm == left_arm}
        & {seed for arm, seed in by_arm_seed if arm == right_arm}
    )
    if not seeds:
        raise ValueError(f"no training seeds for {left_arm} vs {right_arm}")
    draws = []
    seed_metric = {
        (arm, seed): metric_vector(
            by_arm_seed[(arm, seed)], by_arm_seed[(right_arm, seed)]
        )
        for arm in (left_arm,)
        for seed in seeds
    }
    metric_names = [
        key for key in next(iter(seed_metric.values())) if key.startswith("delta_")
    ]
    for _ in range(reps):
        outer = rng.choice(seeds, size=len(seeds), replace=True)
        rep_values = {name: [] for name in metric_names}
        for seed in outer:
            left = by_arm_seed[(left_arm, int(seed))]
            right = by_arm_seed[(right_arm, int(seed))]
            rmap = {int(r["eval_seed"]): r for r in right}
            eval_seeds = np.asarray([int(r["eval_seed"]) for r in left])
            sampled = rng.choice(eval_seeds, size=len(eval_seeds), replace=True)
            left_boot = [
                next(r for r in left if int(r["eval_seed"]) == int(sample))
                for sample in sampled
            ]
            # Keep a stable eval_seed key when the same evaluation seed is
            # sampled repeatedly. Cost/return comparisons are independent of it.
            for duplicate, row in enumerate(left_boot):
                row = dict(row)
                row["eval_seed"] = duplicate
                left_boot[duplicate] = row
            right_lookup = {}
            for sample in sampled:
                key = int(sample)
                if key not in right_lookup:
                    right_lookup[key] = rmap[key]
            # Use the same multiplicities as the left resample.
            right_boot = []
            for duplicate, sample in enumerate(sampled):
                row = dict(right_lookup[int(sample)])
                row["eval_seed"] = duplicate
                right_boot.append(row)
            local = metric_vector(left_boot, right_boot)
            for name in metric_names:
                rep_values[name].append(local[name])
        draws.append({name: float(np.mean(vals)) for name, vals in rep_values.items()})

    summary = {}
    point = {
        name: float(np.mean([seed_metric[(left_arm, seed)][name] for seed in seeds]))
        for name in metric_names
    }
    for name in metric_names:
        vals = np.asarray([draw[name] for draw in draws])
        summary[name] = {
            "estimate": point[name],
            "ci95_low": float(np.quantile(vals, 0.025)),
            "ci95_high": float(np.quantile(vals, 0.975)),
        }
    return summary, draws


def plot_training(a6_root: Path, output: Path) -> list[dict]:
    history_rows = []
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for seed in (20261216, 20261217, 20261218):
        rows = json.loads((a6_root / str(seed) / "l1_history.json").read_text())
        for row in rows:
            history_rows.append({"training_seed": seed, **row})
        steps = [r["step"] for r in rows]
        axes[0].plot(steps, [r["lagrangian"] for r in rows], label=str(seed))
        axes[1].plot(steps, [r["actor_cost_q_mean"] for r in rows], label=str(seed))
    axes[0].set_title("A6 Lagrange multiplier")
    axes[0].set_ylabel("lambda")
    axes[1].set_title("Actor cost-Q estimate")
    axes[1].set_ylabel("Q_cost")
    for axis in axes:
        axis.set_xlabel("Training step")
        axis.grid(alpha=0.25)
        axis.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output / "A6_training_response.png", dpi=180)
    plt.close(fig)
    return history_rows


def plot_closed_loop(
    metrics_by_arm_seed: dict[tuple[str, int], dict], output: Path
) -> None:
    seeds = sorted({seed for _, seed in metrics_by_arm_seed})
    metrics = (
        ("mean_cost", "Mean cumulative cost", "cost"),
        ("cvar25_cost", "CVaR@25 of episode cost", "cost"),
        ("safe_rate", "Safe-episode rate", "rate"),
        ("normalized_reward_mean", "Normalized reward", "reward"),
    )
    fig, axes = plt.subplots(1, 4, figsize=(15, 4.2))
    x = np.arange(len(seeds))
    width = 0.25
    for axis, (metric, title, unit) in zip(axes, metrics):
        for offset, (arm, label) in enumerate(ARMS):
            values = [metrics_by_arm_seed[(arm, seed)][metric] for seed in seeds]
            axis.bar(x + (offset - 1) * width, values, width, label=label)
        axis.set_xticks(x, [str(seed) for seed in seeds])
        axis.set_title(title)
        axis.set_xlabel("Training seed")
        axis.set_ylabel(unit)
        axis.grid(axis="y", alpha=0.25)
    axes[0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(output / "closed_loop_paired_seeds.png", dpi=180)
    plt.close(fig)


def plot_episode_distributions(episodes: list[dict], output: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    bins = np.arange(0, 181, 5)
    for arm, label in ARMS:
        values = [
            float(r["episode_cost"]) for r in episodes if r["arm"] == arm
        ]
        axes[0].hist(values, bins=bins, density=True, histtype="step", label=label)
        sorted_values = np.sort(np.asarray(values))
        cdf = np.arange(1, len(sorted_values) + 1) / len(sorted_values)
        axes[1].plot(sorted_values, cdf, label=label)
    axes[0].set_title("Episode cost distribution")
    axes[0].set_xlabel("Cumulative episode cost")
    axes[0].set_ylabel("Density")
    axes[1].axvline(
        10.0,
        color="#b22222",
        linestyle="--",
        linewidth=1,
        label="Safety threshold",
    )
    axes[1].set_title("Episode cost ECDF")
    axes[1].set_xlabel("Cumulative episode cost")
    axes[1].set_ylabel("Cumulative fraction")
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output / "closed_loop_cost_distribution.png", dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--original-root", type=Path, required=True)
    parser.add_argument("--a6-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-reps", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=20261031)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    by_arm_seed: dict[tuple[str, int], list[dict]] = {}
    source_files = {}
    for seed in (20261216, 20261217, 20261218):
        constrained_path = args.original_root / f"{seed}_evaluation_100.json"
        lambda0_path = args.original_root / f"{seed}_control_evaluation_100.json"
        a6_path = args.a6_root / str(seed) / "evaluation_100.json"
        for arm, path in (
            ("constrained", constrained_path),
            ("lambda0", lambda0_path),
            ("a6", a6_path),
        ):
            by_arm_seed[(arm, seed)] = load_evaluation(path, arm, seed)
            source_files[f"{arm}_{seed}"] = str(path.resolve())

    episode_rows = []
    model_rows = []
    metrics_by_arm_seed = {}
    for arm, seed in sorted(by_arm_seed):
        records = by_arm_seed[(arm, seed)]
        for record in records:
            episode_rows.append(
                {
                    "arm": arm,
                    "training_seed": seed,
                    "eval_seed": record["eval_seed"],
                    "episode_return": record["episode_return"],
                    "episode_cost": record["episode_cost"],
                    "episode_length": record["episode_length"],
                    "normalized_reward": record["normalized_reward"],
                    "normalized_cost": record["normalized_cost"],
                    "safe": record["safe"],
                }
            )
        metrics = summarize(records)
        metrics_by_arm_seed[(arm, seed)] = metrics
        model_rows.append({"arm": arm, "training_seed": seed, **metrics})

    write_csv(
        args.output_dir / "episode_level_all.csv",
        episode_rows,
        [
            "arm",
            "training_seed",
            "eval_seed",
            "episode_return",
            "episode_cost",
            "episode_length",
            "normalized_reward",
            "normalized_cost",
            "safe",
        ],
    )
    write_csv(
        args.output_dir / "model_metrics.csv",
        model_rows,
        ["arm", "training_seed", "n_episodes", *METRIC_COLUMNS],
    )

    pair_rows = []
    for left_arm, right_arm, pair_name in (
        ("a6", "constrained", "A6_minus_BCQ-Lag"),
        ("a6", "lambda0", "A6_minus_lambda0"),
        ("constrained", "lambda0", "BCQ-Lag_minus_lambda0"),
    ):
        for seed in (20261216, 20261217, 20261218):
            row = {
                "comparison": pair_name,
                "training_seed": seed,
                **paired_delta(
                    by_arm_seed[(left_arm, seed)],
                    by_arm_seed[(right_arm, seed)],
                ),
            }
            pair_rows.append(row)
    write_csv(
        args.output_dir / "paired_by_seed.csv",
        pair_rows,
        [
            "comparison",
            "training_seed",
            "n_pairs",
            "delta_cost",
            "delta_cvar25",
            "delta_safe_rate",
            "delta_return",
            "delta_normalized_cost",
            "delta_normalized_reward",
        ],
    )

    rng = np.random.default_rng(args.bootstrap_seed)
    bootstrap_summary = {}
    bootstrap_draws = {}
    for left_arm, right_arm, pair_name in (
        ("a6", "constrained", "A6_minus_BCQ-Lag"),
        ("a6", "lambda0", "A6_minus_lambda0"),
        ("constrained", "lambda0", "BCQ-Lag_minus_lambda0"),
    ):
        summary, draws = hierarchical_bootstrap(
            by_arm_seed,
            left_arm,
            right_arm,
            args.bootstrap_reps,
            rng,
        )
        bootstrap_summary[pair_name] = summary
        bootstrap_draws[pair_name] = draws

    bootstrap_rows = []
    for comparison, summary in bootstrap_summary.items():
        for metric, values in summary.items():
            bootstrap_rows.append({"comparison": comparison, "metric": metric, **values})
    write_csv(
        args.output_dir / "paired_hierarchical_bootstrap.csv",
        bootstrap_rows,
        ["comparison", "metric", "estimate", "ci95_low", "ci95_high"],
    )

    history_rows = plot_training(args.a6_root, args.output_dir)
    write_csv(
        args.output_dir / "A6_training_history_long.csv",
        history_rows,
        list(history_rows[0].keys()),
    )
    plot_closed_loop(metrics_by_arm_seed, args.output_dir)
    plot_episode_distributions(episode_rows, args.output_dir)

    summary_payload = {
        "protocol": {
            "evaluation_seeds": [149000, 149099],
            "episodes_per_model": 100,
            "training_seeds": [20261216, 20261217, 20261218],
            "safe_definition": "raw cumulative episode cost <= 10",
            "cvar25_definition": "mean of highest ceil(25%) episode costs",
            "bootstrap": (
                "two-stage paired cluster bootstrap: evaluation seeds nested "
                "within training seeds, then training seeds resampled"
            ),
            "bootstrap_reps": args.bootstrap_reps,
            "bootstrap_seed": args.bootstrap_seed,
        },
        "source_files": source_files,
        "paired_hierarchical_bootstrap": bootstrap_summary,
    }
    (args.output_dir / "analysis_summary.json").write_text(
        json.dumps(summary_payload, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary_payload, indent=2))


if __name__ == "__main__":
    main()
