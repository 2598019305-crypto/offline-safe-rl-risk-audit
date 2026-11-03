"""Apply the observed terminal A6 PID multiplier to static gradient probes.

The original probe rebuilds a fresh controller and therefore does not retain the
PID error state stored outside model_state. A6 checkpoints intentionally save
only model weights, so the probe's reconstructed lambda is not used. The
unweighted reward/cost gradients are state-dependent but lambda-independent;
this script combines them with the final lambda recorded in l1_history.json.
"""

import argparse
import csv
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--a6-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for seed in (20261216, 20261217, 20261218):
        raw_path = args.raw_dir / f"{seed}.raw.json"
        history_path = args.a6_root / str(seed) / "l1_history.json"
        raw = json.loads(raw_path.read_text(encoding="utf-8"))
        history = json.loads(history_path.read_text(encoding="utf-8"))
        terminal_lambda = float(history[-1]["lagrangian"])
        output = dict(raw)
        output["reconstructed_fresh_controller_lagrangian"] = float(
            raw["lagrangian"]
        )
        output["lagrangian"] = terminal_lambda
        output["lagrangian_source"] = str(history_path.resolve())
        output["lagrangian_history_step"] = int(history[-1]["step"])
        output["grad_cost_scaled_norm"] = (
            terminal_lambda * float(raw["grad_cost_term_norm"])
        )
        output["grad_ratio_cost_over_reward"] = (
            output["grad_cost_scaled_norm"]
            / (float(raw["grad_reward_norm"]) + 1e-12)
        )
        (args.output_dir / f"{seed}.json").write_text(
            json.dumps(output, indent=2), encoding="utf-8"
        )
        rows.append(output)

    fields = [
        "checkpoint",
        "n_states",
        "q_reward_mean",
        "qc_min_mean",
        "qc_mean_agg_mean",
        "qc_negative_ratio",
        "qc_threshold",
        "lagrangian",
        "grad_reward_norm",
        "grad_cost_term_norm",
        "grad_cost_scaled_norm",
        "grad_ratio_cost_over_reward",
        "grad_cosine_reward_cost",
        "data_cost_to_go_mean",
        "data_cost_to_go_p90",
    ]
    with (args.output_dir / "A6_L3_summary.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({field: row[field] for field in fields} for row in rows)


if __name__ == "__main__":
    main()
