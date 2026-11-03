"""Improved L2 pilot with hazard-clearance risk and balanced elapsed time.

The original pilot stratified states by cumulative episode cost, which is
confounded by elapsed time. This version records hazard clearance directly from
the environment and samples low/mid/high clearance states inside each of four
remaining-horizon bands.
"""

import argparse
import csv
import json
import os
import random
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "disabled")
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("DSRL_SUPPRESS_IMPORT_ERROR", "1")

import gymnasium as gym
import h5py
import numpy as np
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
DATA_DIR = Path(os.environ.get("BCQ_DATA_DIR", REPO_ROOT / "data" / "raw"))
H5 = Path(
    os.environ.get(
        "BCQ_CARGOAL_HDF5",
        DATA_DIR / "SafetyCarGoal1Gymnasium-v0-120-1671.hdf5",
    )
)
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


def find_base_env(env):
    while hasattr(env, "env"):
        env = env.env
    return env


def hazard_clearance(env):
    base = find_base_env(env)
    task = base.task
    agent_xy = np.asarray(task.agent.pos[:2], dtype=float)
    clearances = []
    for hazard_xy in task.hazards.pos:
        xy = np.asarray(hazard_xy[:2], dtype=float)
        clearances.append(float(np.linalg.norm(agent_xy - xy) - task.hazards.size))
    return min(clearances)


def make_env():
    env = gym.make("OfflineCarGoal1Gymnasium-v0")
    env.set_target_cost(10)
    env = wrap_env(env, reward_scale=0.1)
    return OfflineEnvWrapper(env)


def qvals(model, states, acts):
    n_states, n_candidates, _ = acts.shape
    repeated_states = (
        states[:, None, :]
        .expand(n_states, n_candidates, states.shape[-1])
        .reshape(n_states * n_candidates, -1)
    )
    q1, q2, _, _ = model.critic.predict(repeated_states, acts.reshape(-1, 2))
    c1, c2, _, _ = model.cost_critic.predict(repeated_states, acts.reshape(-1, 2))
    return (
        torch.min(q1, q2).reshape(n_states, n_candidates),
        torch.min(c1, c2).reshape(n_states, n_candidates),
    )


def dump_candidates(
    path, states, actions, rewards, costs, shortlisted_ids, policy_id=None
):
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "state_id",
                "candidate_id",
                "action_0",
                "action_1",
                "Q_reward",
                "Q_cost",
                "policy_selected",
                "evaluation_shortlisted",
            ]
        )
        for state_id in range(actions.shape[0]):
            for candidate_id in range(actions.shape[1]):
                writer.writerow(
                    [
                        state_id,
                        candidate_id,
                        float(actions[state_id, candidate_id, 0]),
                        float(actions[state_id, candidate_id, 1]),
                        float(rewards[state_id, candidate_id]),
                        float(costs[state_id, candidate_id]),
                        int(policy_id is not None and candidate_id == policy_id),
                        int(candidate_id in shortlisted_ids[state_id]),
                    ]
                )


def shortlist_by_cost(costs, policy_id=0):
    order = np.argsort(costs)
    chosen = {int(order[0]), int(order[len(order) // 2]), int(order[-1])}
    chosen.add(int(policy_id))
    return sorted(chosen)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--episodes", type=int, default=24)
    parser.add_argument("--seed-start", type=int, default=149000)
    parser.add_argument("--states-per-risk-bin", type=int, default=3)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    seed_all(0)
    model = build_model(args.checkpoint, device)
    env = make_env()

    all_states = []
    for episode in range(args.episodes):
        eval_seed = args.seed_start + episode
        seed_all(eval_seed)
        observation, _ = env.reset()
        cumulative_cost = 0.0
        for step in range(1000):
            action, _ = model.act(observation)
            all_states.append(
                {
                    "episode": episode,
                    "eval_seed": eval_seed,
                    "step": step,
                    "time_bin": min(3, step // 250),
                    "remaining_steps": 999 - step,
                    "hazard_clearance": hazard_clearance(env),
                    "cumulative_cost_before": cumulative_cost,
                    "observation": observation.astype(np.float32).tolist(),
                    "behavior_action": action.astype(np.float32).tolist(),
                }
            )
            observation, _, terminated, truncated, info = env.step(action)
            cumulative_cost += float(info["cost"])
            if terminated or truncated:
                break
    print(f"rollout_states={len(all_states)}", flush=True)

    bank = []
    for time_bin in range(4):
        pool = [row for row in all_states if row["time_bin"] == time_bin]
        pool.sort(key=lambda row: row["hazard_clearance"])
        thirds = np.array_split(np.asarray(pool, dtype=object), 3)
        for risk_group, group_rows in zip(("low", "mid", "high"), thirds):
            count = args.states_per_risk_bin
            if count == 1:
                indices = [len(group_rows) // 2]
            else:
                indices = np.linspace(0, len(group_rows) - 1, count).round().astype(int)
            for index in sorted(set(int(i) for i in indices)):
                row = dict(group_rows[index])
                row["risk_group"] = risk_group
                row["time_bin"] = time_bin
                bank.append(row)
    for state_id, row in enumerate(bank):
        row["state_id"] = state_id

    with (output / "state_bank.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        fields = [
            "state_id",
            "risk_group",
            "time_bin",
            "remaining_steps",
            "episode",
            "eval_seed",
            "step",
            "hazard_clearance",
            "cumulative_cost_before",
            "behavior_action",
            "observation",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(bank)

    states = torch.tensor(
        [row["observation"] for row in bank], dtype=torch.float32, device=device
    )
    with h5py.File(H5, "r") as handle:
        observations = handle["observations"][:].astype(np.float32)
        data_actions = handle["actions"][:].astype(np.float32)
    dataset_states = torch.tensor(observations, device=device)
    dataset_actions = torch.tensor(data_actions, device=device)

    nearest_indices = []
    for start in range(0, len(states), 10):
        distances = torch.cdist(states[start : start + 10], dataset_states) ** 2
        nearest_indices.append(
            torch.topk(distances, k=10, largest=False).indices
        )
    nearest_indices = torch.cat(nearest_indices, dim=0)
    data_candidates = dataset_actions[nearest_indices.reshape(-1)].reshape(
        len(states), 10, 2
    )

    with torch.no_grad():
        mean_latent = model.vae.decode(states)
        canonical_action = model.actor(states, mean_latent)
        vae_candidates = [mean_latent]
        actor_candidates = [canonical_action]
        for _ in range(31):
            z = torch.randn(
                (len(states), model.latent_dim), device=device
            ).clamp(-0.5, 0.5)
            latent = model.vae.decode(states, z)
            vae_candidates.append(latent)
            actor_candidates.append(model.actor(states, latent))
        vae_candidates = torch.stack(vae_candidates, dim=1)
        actor_candidates = torch.stack(actor_candidates, dim=1)

    with torch.no_grad():
        q_data, c_data = qvals(model, states, data_candidates)
        q_vae, c_vae = qvals(model, states, vae_candidates)
        q_actor, c_actor = qvals(model, states, actor_candidates)

    data_shortlist = []
    vae_shortlist = []
    actor_shortlist = []
    for state_id in range(len(states)):
        data_shortlist.append(
            shortlist_by_cost(c_data[state_id].detach().cpu().numpy())
        )
        vae_shortlist.append(
            shortlist_by_cost(c_vae[state_id].detach().cpu().numpy())
        )
        actor_shortlist.append(
            shortlist_by_cost(c_actor[state_id].detach().cpu().numpy(), policy_id=0)
        )

    dump_candidates(
        output / "data_candidates.csv",
        states.cpu().numpy(),
        data_candidates.cpu().numpy(),
        q_data.cpu().numpy(),
        c_data.cpu().numpy(),
        data_shortlist,
    )
    dump_candidates(
        output / "vae_candidates.csv",
        states.cpu().numpy(),
        vae_candidates.cpu().numpy(),
        q_vae.cpu().numpy(),
        c_vae.cpu().numpy(),
        vae_shortlist,
    )
    dump_candidates(
        output / "actor_candidates.csv",
        states.cpu().numpy(),
        actor_candidates.cpu().numpy(),
        q_actor.cpu().numpy(),
        c_actor.cpu().numpy(),
        actor_shortlist,
        policy_id=0,
    )

    with (output / "candidate_selection.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "state_id",
                "source",
                "candidate_id",
                "Q_cost",
                "policy_selected",
            ]
        )
        for source, costs, shortlists in (
            ("data", c_data, data_shortlist),
            ("vae", c_vae, vae_shortlist),
            ("actor", c_actor, actor_shortlist),
        ):
            scores = costs.cpu().numpy()
            for state_id, candidate_ids in enumerate(shortlists):
                for candidate_id in candidate_ids:
                    writer.writerow(
                        [
                            state_id,
                            source,
                            candidate_id,
                            float(scores[state_id, candidate_id]),
                            int(source == "actor" and candidate_id == 0),
                        ]
                    )

    summary = {
        "n_states": len(bank),
        "states_per_risk_time_bin": args.states_per_risk_bin,
        "risk_definition": (
            "within each 250-step elapsed-time bin, sort by minimum hazard-center "
            "distance minus hazard radius; low/mid/high denote clearance tertiles"
        ),
        "horizon_balancing": "3 states from each risk group in each time bin",
        "data_candidates": 10,
        "vae_candidates": 32,
        "actor_candidates": 32,
        "shortlisted_per_source": 3,
        "behavior_policy_candidate": "actor candidate_id 0",
        "data_q_cost_mean": float(c_data.mean()),
        "vae_q_cost_mean": float(c_vae.mean()),
        "actor_q_cost_mean": float(c_actor.mean()),
        "data_q_reward_mean": float(q_data.mean()),
        "vae_q_reward_mean": float(q_vae.mean()),
        "actor_q_reward_mean": float(q_actor.mean()),
    }
    (output / "offline_candidate_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
