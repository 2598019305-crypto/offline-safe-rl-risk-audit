"""Post-process the final L2 hazard-clearance rollout."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, pearsonr, spearmanr


def cvar25(values):
    arr = np.sort(np.asarray(values, dtype=float))[::-1]
    count = max(1, int(np.ceil(0.25 * len(arr))))
    return float(arr[:count].mean())


def summarize(frame):
    safe = frame["safe_total_episode_cost_le_10"].astype(bool)
    prospective = frame["future_cost_after_candidate"] <= 10.0
    result = {
        "n": int(len(frame)),
        "safe_episode_rate": float(safe.mean()),
        "prospective_cost_le_10_rate": float(prospective.mean()),
        "mean_total_episode_cost": float(frame["total_episode_cost"].mean()),
        "cvar25_total_episode_cost": cvar25(frame["total_episode_cost"]),
        "mean_future_cost_after_candidate": float(
            frame["future_cost_after_candidate"].mean()
        ),
        "cvar25_future_cost_after_candidate": cvar25(
            frame["future_cost_after_candidate"]
        ),
        "mean_future_reward_after_candidate": float(
            frame["future_reward_after_candidate"].mean()
        ),
        "mean_Q_cost": float(frame["Q_cost"].mean()),
    }
    if frame["Q_cost"].nunique() > 1 and prospective.nunique() > 1:
        result["pearson_Qcost_vs_future_cost"] = float(
            pearsonr(
                frame["Q_cost"], frame["future_cost_after_candidate"]
            ).statistic
        )
        result["spearman_Qcost_vs_future_cost"] = float(
            spearmanr(
                frame["Q_cost"], frame["future_cost_after_candidate"]
            ).statistic
        )
        rho = spearmanr(frame["Q_cost"], prospective).statistic
        result["spearman_Qcost_vs_prospective_safety"] = float(rho)
        positive = frame.loc[prospective, "Q_cost"]
        negative = frame.loc[~prospective, "Q_cost"]
        if len(positive) and len(negative):
            u = mannwhitneyu(positive, negative, alternative="two-sided").statistic
            result["auc_Qcost_safe_ranking"] = float(
                u / (len(positive) * len(negative))
            )
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rollout", type=Path, required=True)
    parser.add_argument("--state-bank", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    frame = pd.read_csv(args.rollout)
    bank = pd.read_csv(args.state_bank)
    frame = frame.merge(
        bank[["state_id", "cumulative_cost_before"]],
        on="state_id",
        how="left",
        validate="many_to_one",
    )
    frame["remaining_budget_positive"] = (
        frame["cumulative_cost_before"] < 10.0
    )
    frame["prospective_cost_le_10"] = (
        frame["future_cost_after_candidate"] <= 10.0
    )

    rows = []
    for scope, scoped in (
        ("all_states", frame),
        ("positive_remaining_budget", frame[frame["remaining_budget_positive"]]),
    ):
        for (risk, source), group in scoped.groupby(["risk_group", "source"]):
            rows.append(
                {
                    "scope": scope,
                    "risk_group": risk,
                    "source": source,
                    **summarize(group),
                }
            )
    by_risk_source = pd.DataFrame(rows)
    by_risk_source.to_csv(
        args.out / "L2_by_risk_source.csv", index=False, encoding="utf-8-sig"
    )

    rows = []
    for (time_bin, risk, source), group in frame.groupby(
        ["time_bin", "risk_group", "source"]
    ):
        rows.append(
            {
                "time_bin": int(time_bin),
                "risk_group": risk,
                "source": source,
                **summarize(group),
            }
        )
    pd.DataFrame(rows).to_csv(
        args.out / "L2_by_time_risk_source.csv",
        index=False,
        encoding="utf-8-sig",
    )

    grouped = frame.groupby(["state_id", "rollout_index"])
    state_rollout = []
    for (state_id, rollout_index), group in grouped:
        policy = group[group["policy_selected"].astype(bool)]
        if len(policy) != 1:
            raise ValueError(
                f"expected one policy-selected row for {state_id}/{rollout_index}"
            )
        any_safe = bool(group["safe_total_episode_cost_le_10"].any())
        policy_safe = bool(policy.iloc[0]["safe_total_episode_cost_le_10"])
        any_prospective = bool(group["prospective_cost_le_10"].any())
        policy_prospective = bool(policy.iloc[0]["prospective_cost_le_10"])
        state_rollout.append(
            {
                "state_id": int(state_id),
                "rollout_index": int(rollout_index),
                "risk_group": group.iloc[0]["risk_group"],
                "time_bin": int(group.iloc[0]["time_bin"]),
                "remaining_budget_positive": bool(
                    group.iloc[0]["remaining_budget_positive"]
                ),
                "any_safe_candidate": any_safe,
                "policy_safe": policy_safe,
                "safe_available_but_policy_unsafe": any_safe and not policy_safe,
                "any_prospective_candidate": any_prospective,
                "policy_prospective_safe": policy_prospective,
                "prospective_available_but_policy_unsafe": (
                    any_prospective and not policy_prospective
                ),
            }
        )
    state_rollout = pd.DataFrame(state_rollout)
    state_rollout.to_csv(
        args.out / "L2_state_rollout_decisions.csv",
        index=False,
        encoding="utf-8-sig",
    )

    rollup = {}
    for scope, group in (
        ("all_states", state_rollout),
        (
            "positive_remaining_budget",
            state_rollout[state_rollout["remaining_budget_positive"]],
        ),
    ):
        rollup[scope] = {
            "n_state_rollouts": int(len(group)),
            "fraction_with_any_safe_candidate": float(
                group["any_safe_candidate"].mean()
            ),
            "policy_selected_safe_rate": float(group["policy_safe"].mean()),
            "fraction_safe_available_but_policy_unsafe": float(
                group["safe_available_but_policy_unsafe"].mean()
            ),
            "fraction_with_any_prospective_candidate": float(
                group["any_prospective_candidate"].mean()
            ),
            "policy_prospective_safe_rate": float(
                group["policy_prospective_safe"].mean()
            ),
            "fraction_prospective_available_but_policy_unsafe": float(
                group["prospective_available_but_policy_unsafe"].mean()
            ),
        }

    summary = {
        "notes": {
            "safe_episode": (
                "total episode cost <= 10, including cost already incurred "
                "before the selected state"
            ),
            "prospective_safe": (
                "future cost after candidate <= 10; this separates local "
                "candidate quality from an already-exhausted past budget"
            ),
            "integrity": {
                "rows": int(len(frame)),
                "states": int(frame["state_id"].nunique()),
                "policy_rows": int(frame["policy_selected"].sum()),
                "max_state_replay_abs_obs_error": float(
                    frame["state_replay_max_abs_obs_error"].max()
                ),
            },
        },
        "state_rollout_rollup": rollup,
    }
    (args.out / "L2_analysis_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    major = by_risk_source[by_risk_source["scope"] == "all_states"].copy()
    sources = ["data", "vae", "actor"]
    risks = ["low", "mid", "high"]
    x = np.arange(len(risks))
    width = 0.24
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.3))
    for offset, source in enumerate(sources):
        selected = major[major["source"] == source].set_index("risk_group")
        values = [selected.loc[risk, "safe_episode_rate"] for risk in risks]
        axes[0].bar(x + (offset - 1) * width, values, width, label=source)
        values = [
            selected.loc[risk, "prospective_cost_le_10_rate"] for risk in risks
        ]
        axes[1].bar(x + (offset - 1) * width, values, width, label=source)
    for axis, title in zip(
        axes,
        (
            "Full-episode safety (including past cost)",
            "Prospective safety (future cost <= 10)",
        ),
    ):
        axis.set_xticks(x, risks)
        axis.set_ylim(0, 1)
        axis.set_xlabel("Hazard-clearance stratum")
        axis.set_ylabel("Safe rate")
        axis.set_title(title)
        axis.grid(axis="y", alpha=0.25)
        axis.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(args.out / "L2_candidate_feasibility.png", dpi=180)
    plt.close(fig)

    rows = []
    for scope, scoped in (
        ("all_states", state_rollout),
        (
            "positive_remaining_budget",
            state_rollout[state_rollout["remaining_budget_positive"]],
        ),
    ):
        for risk, group in scoped.groupby("risk_group"):
            rows.append(
                {
                    "scope": scope,
                    "risk_group": risk,
                    "n_state_rollouts": len(group),
                    "any_safe_candidate": float(group["any_safe_candidate"].mean()),
                    "policy_safe": float(group["policy_safe"].mean()),
                    "safe_available_policy_unsafe": float(
                        group["safe_available_but_policy_unsafe"].mean()
                    ),
                }
            )
    pd.DataFrame(rows).to_csv(
        args.out / "L2_state_rollout_by_risk.csv",
        index=False,
        encoding="utf-8-sig",
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
