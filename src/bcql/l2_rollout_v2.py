"""Real-environment L2 rollout with hazard-clearance-balanced state bank.

For each state and common random-number index, this script restores the state by
replaying the original episode, checks observation equality, applies one
candidate action, then continues the frozen behavior policy to episode end.
Safety is evaluated against the full remaining raw cost budget of 10.
"""

import argparse
import csv
import json
import os
import random
import sys
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "disabled")
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("DSRL_SUPPRESS_IMPORT_ERROR", "1")

import gymnasium as gym
import numpy as np
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR / "osrl_min"))
sys.path.insert(0, str(SCRIPT_DIR))

import dsrl  # noqa: F401
import dsrl.offline_safety_gymnasium  # noqa: F401
from dsrl.offline_env import OfflineEnvWrapper, wrap_env
from osrl.algorithms.bcql import BCQL


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_model(checkpoint, device):
    model = BCQL(
        state_dim=72,
        action_dim=2,
        max_action=1.0,
        a_hidden_sizes=[256, 256],
        c_hidden_sizes=[256, 256],
        vae_hidden_sizes=400,
        sample_action_num=10,
        gamma=0.99,
        tau=0.005,
        phi=0.05,
        lmbda=0.75,
        beta=0.5,
        PID=[0.1, 0.003, 0.001],
        num_q=2,
        num_qc=2,
        cost_limit=10,
        episode_len=1000,
        device=device,
    )
    model.load_state_dict(torch.load(checkpoint, map_location=device)["model_state"])
    model.eval()
    return model


def make_env():
    env = gym.make("OfflineCarGoal1Gymnasium-v0")
    env.set_target_cost(10)
    env = wrap_env(env, reward_scale=0.1)
    return OfflineEnvWrapper(env)


def read_rows(path):
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--l2-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--rollouts", type=int, default=5)
    parser.add_argument("--horizon", type=int, default=0)
    parser.add_argument("--max-obs-mismatch", type=float, default=1e-5)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    args.out.mkdir(parents=True, exist_ok=True)
    seed_all(0)
    model = build_model(args.checkpoint, device)
    env = make_env()

    bank = read_rows(args.l2_dir / "state_bank.csv")
    selections = read_rows(args.l2_dir / "candidate_selection.csv")
    candidate_rows = {
        source: {
            (int(row["state_id"]), int(row["candidate_id"])): row
            for row in read_rows(args.l2_dir / filename)
        }
        for source, filename in (
            ("data", "data_candidates.csv"),
            ("vae", "vae_candidates.csv"),
            ("actor", "actor_candidates.csv"),
        )
    }
    selected = defaultdict(list)
    for row in selections:
        selected[int(row["state_id"])].append(
            (row["source"], int(row["candidate_id"]), int(row["policy_selected"]))
        )

    results = []
    max_mismatch_observed = 0.0
    for state in bank:
        state_id = int(state["state_id"])
        eval_seed = int(state["eval_seed"])
        target_step = int(state["step"])
        target_obs = np.asarray(json.loads(state["observation"]), dtype=np.float32)
        expected_before = float(state["cumulative_cost_before"])
        for source, candidate_id, policy_selected in selected[state_id]:
            candidate = candidate_rows[source][(state_id, candidate_id)]
            candidate_action = np.asarray(
                [float(candidate["action_0"]), float(candidate["action_1"])],
                dtype=np.float32,
            )
            for rollout_index in range(args.rollouts):
                seed_all(eval_seed)
                observation, _ = env.reset()
                actual_before = 0.0
                replayed = True
                for _ in range(target_step):
                    action, _ = model.act(observation)
                    observation, _, terminated, truncated, info = env.step(action)
                    actual_before += float(info["cost"])
                    if terminated or truncated:
                        replayed = False
                        break
                mismatch = float(np.max(np.abs(observation - target_obs)))
                max_mismatch_observed = max(max_mismatch_observed, mismatch)
                if not replayed or mismatch > args.max_obs_mismatch:
                    raise RuntimeError(
                        f"state replay mismatch state={state_id} step={target_step} "
                        f"max_abs_obs={mismatch}"
                    )

                common_seed = 700000 + state_id * 100 + rollout_index
                seed_all(common_seed)
                observation, reward, terminated, truncated, info = env.step(
                    candidate_action
                )
                future_cost = float(info["cost"])
                future_reward = float(reward)
                steps = 1
                while not (terminated or truncated):
                    if args.horizon > 0 and steps >= args.horizon:
                        break
                    action, _ = model.act(observation)
                    observation, reward, terminated, truncated, info = env.step(action)
                    future_cost += float(info["cost"])
                    future_reward += float(reward)
                    steps += 1

                total_cost = actual_before + future_cost
                results.append(
                    {
                        "state_id": state_id,
                        "risk_group": state["risk_group"],
                        "time_bin": int(state["time_bin"]),
                        "remaining_steps_at_state": int(state["remaining_steps"]),
                        "hazard_clearance": float(state["hazard_clearance"]),
                        "source": source,
                        "candidate_id": candidate_id,
                        "policy_selected": policy_selected,
                        "rollout_index": rollout_index,
                        "common_rollout_seed": common_seed,
                        "Q_reward": float(candidate["Q_reward"]),
                        "Q_cost": float(candidate["Q_cost"]),
                        "actual_cost_before_candidate": actual_before,
                        "future_cost_after_candidate": future_cost,
                        "total_episode_cost": total_cost,
                        "future_reward_after_candidate": future_reward,
                        "future_steps_executed": steps,
                        "safe_total_episode_cost_le_10": int(total_cost <= 10.0),
                        "state_replay_max_abs_obs_error": mismatch,
                    }
                )
        print(f"state {state_id} {state['risk_group']} complete", flush=True)

    out_csv = args.out / "rollout_results.csv"
    with out_csv.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)

    grouped = defaultdict(list)
    for row in results:
        grouped[(row["risk_group"], row["source"])].append(
            row["safe_total_episode_cost_le_10"]
        )
    summary = {
        f"{risk}/{source}": {
            "n": len(values),
            "safe_rate": float(np.mean(values)),
        }
        for (risk, source), values in sorted(grouped.items())
    }

    by_state_rollout = defaultdict(list)
    for row in results:
        by_state_rollout[(row["state_id"], row["rollout_index"])].append(row)
    available = []
    selected_safe = []
    available_safe_selected_unsafe = []
    for rows in by_state_rollout.values():
        safe_available = any(
            row["safe_total_episode_cost_le_10"] for row in rows
        )
        policy_rows = [row for row in rows if row["policy_selected"]]
        if not policy_rows:
            raise RuntimeError("policy-selected actor action missing")
        policy_safe = bool(policy_rows[0]["safe_total_episode_cost_le_10"])
        available.append(safe_available)
        selected_safe.append(policy_safe)
        if safe_available and not policy_safe:
            available_safe_selected_unsafe.append(True)
        elif not safe_available:
            available_safe_selected_unsafe.append(False)

    summary["state_rollout_analysis"] = {
        "n_state_rollouts": len(available),
        "fraction_with_any_safe_candidate": float(np.mean(available)),
        "policy_selected_safe_rate": float(np.mean(selected_safe)),
        "fraction_safe_available_but_policy_unsafe": float(
            np.mean(available_safe_selected_unsafe)
        ),
    }
    summary["integrity"] = {
        "max_state_replay_abs_obs_error": max_mismatch_observed,
        "horizon_mode": "episode_end" if args.horizon == 0 else args.horizon,
        "cost_limit": 10.0,
    }
    (args.out / "feasibility_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
