"""Load an official BCQ-Lag checkpoint and evaluate it on the frozen CarGoal1
environment with fixed evaluation seeds (149000..149000+N-1 by default).

Reports normalized reward/cost, safe-episode rate, CVaR@25, and a cost-critic
calibration summary (predicted cost-to-go vs realized discounted cost-to-go).
"""

import argparse
import csv
import json
import math
import os
import random
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "disabled")
os.environ.setdefault("DSRL_SUPPRESS_IMPORT_ERROR", "1")

import gymnasium as gym
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

import dsrl  # noqa: F401
import dsrl.offline_safety_gymnasium  # noqa: F401
from dsrl.offline_env import OfflineEnvWrapper, wrap_env
from osrl.algorithms.bcql import BCQL, BCQLTrainer


DEFAULT_CFG = {
    "task": "OfflineCarGoal1Gymnasium-v0",
    "reward_scale": 0.1,
    "cost_scale": 1.0,
    "a_hidden_sizes": [256, 256],
    "c_hidden_sizes": [256, 256],
    "vae_hidden_sizes": 400,
    "sample_action_num": 10,
    "gamma": 0.99,
    "tau": 0.005,
    "phi": 0.05,
    "lmbda": 0.75,
    "beta": 0.5,
    "PID": [0.1, 0.003, 0.001],
    "num_q": 2,
    "num_qc": 2,
    "cost_limit": 10,
    "episode_len": 1000,
}


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_model(cfg, env, device):
    return BCQL(
        state_dim=env.observation_space.shape[0],
        action_dim=env.action_space.shape[0],
        max_action=float(env.action_space.high[0]),
        a_hidden_sizes=list(cfg["a_hidden_sizes"]),
        c_hidden_sizes=list(cfg["c_hidden_sizes"]),
        vae_hidden_sizes=int(cfg["vae_hidden_sizes"]),
        sample_action_num=int(cfg["sample_action_num"]),
        gamma=float(cfg["gamma"]),
        tau=float(cfg["tau"]),
        phi=float(cfg["phi"]),
        lmbda=float(cfg["lmbda"]),
        beta=float(cfg["beta"]),
        PID=list(cfg["PID"]),
        num_q=int(cfg["num_q"]),
        num_qc=int(cfg["num_qc"]),
        cost_limit=float(cfg["cost_limit"]),
        episode_len=int(cfg["episode_len"]),
        device=device,
    )


def make_env(cfg):
    env = gym.make(cfg["task"])
    env.set_target_cost(cfg["cost_limit"])
    env = wrap_env(env, reward_scale=cfg["reward_scale"])
    env = OfflineEnvWrapper(env)
    return env


def rollout_with_calibration(model, env, gamma, device):
    obs, info = env.reset()
    ep_ret = 0.0
    ep_cost = 0.0
    ep_len = 0
    steps_cost = []
    calib = []
    for _ in range(model.episode_len):
        obs_t = torch.tensor(obs[None, ...], dtype=torch.float32).to(device)
        act = model.act(obs)[0]
        with torch.no_grad():
            act_t = torch.tensor(act[None, ...], dtype=torch.float32).to(device)
            qc1, qc2, _, _ = model.cost_critic.predict(obs_t, act_t)
            qc_pred = float(torch.min(qc1, qc2).item())
        obs_next, reward, terminated, truncated, info = env.step(act)
        cost = info["cost"]
        ep_ret += reward
        ep_cost += cost
        ep_len += 1
        steps_cost.append(float(cost))
        calib.append((qc_pred, float(cost)))
        obs = obs_next
        if terminated or truncated:
            break
    # realized discounted cost-to-go for each step
    T = len(steps_cost)
    realized = [0.0] * T
    for t in range(T - 1, -1, -1):
        realized[t] = steps_cost[t] + gamma * (realized[t + 1] if t + 1 < T else 0.0)
    return ep_ret, ep_cost, ep_len, calib, realized


def cvar25(values):
    arr = np.sort(np.asarray(values, dtype=float))[::-1]
    k = max(1, int(math.ceil(0.25 * len(arr))))
    return float(arr[:k].mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--config", default=None)
    ap.add_argument("--episodes", type=int, default=100)
    ap.add_argument("--seed-start", type=int, default=149000)
    ap.add_argument("--output", required=True)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    cfg = dict(DEFAULT_CFG)
    if args.config and Path(args.config).exists():
        cfg.update(json.loads(Path(args.config).read_text(encoding="utf-8")))
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    env = make_env(cfg)
    model = build_model(cfg, env, device)
    state = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state["model_state"])
    model.to(device)
    model.eval()

    ref_min_reward = float(env.min_episode_reward)
    ref_max_reward = float(env.max_episode_reward)
    target_cost = float(cfg["cost_limit"])
    epsilon = 0.0 if target_cost != 0 else 1.0

    records = []
    all_calib = []
    for i in range(args.episodes):
        seed = args.seed_start + i
        seed_everything(seed)
        ret, cost, length, calib, realized = rollout_with_calibration(
            model, env, float(cfg["gamma"]), device
        )
        ret_u = ret / float(cfg["reward_scale"])
        cost_u = cost / float(cfg["cost_scale"])
        norm_ret = (ret_u - ref_min_reward) / (ref_max_reward - ref_min_reward)
        norm_cost = (cost_u + epsilon) / (target_cost + epsilon)
        records.append(
            {
                "eval_seed": seed,
                "reward": ret_u,
                "cost": cost_u,
                "length": length,
                "normalized_reward": norm_ret,
                "normalized_cost": norm_cost,
            }
        )
        for step_in_episode, ((qc_pred, _step_cost), future) in enumerate(
            zip(calib, realized)
        ):
            all_calib.append(
                {
                    "training_seed": cfg.get("seed"),
                    "eval_seed": seed,
                    "step_in_episode": step_in_episode,
                    "state_id": f"{seed}:{step_in_episode}",
                    "action_id": 0,
                    "predicted_cost": qc_pred,
                    "actual_cost_target": future,
                }
            )

    norm_costs = [r["normalized_cost"] for r in records]
    norm_rets = [r["normalized_reward"] for r in records]
    safe_rate = float(np.mean([1.0 if c <= 1.0 else 0.0 for c in norm_costs]))

    pred = np.asarray(
        [x["predicted_cost"] for x in all_calib], dtype=float
    ).reshape(-1)
    real = np.asarray(
        [x["actual_cost_target"] for x in all_calib], dtype=float
    ).reshape(-1)
    calib_summary = None
    if len(pred) > 1 and len(pred) == len(real) and pred.std() > 0 and real.std() > 0:
        pearson = float(np.corrcoef(pred, real)[0, 1])
        sp = float(np.corrcoef(np.argsort(np.argsort(pred)), np.argsort(np.argsort(real)))[0, 1])
        mae = float(np.mean(np.abs(pred - real)))
        calib_summary = {
            "n_steps": int(len(pred)),
            "pearson": pearson,
            "spearman": sp,
            "mae": mae,
            "pred_mean": float(pred.mean()),
            "realized_mean": float(real.mean()),
        }

    out = {
        "checkpoint": args.checkpoint,
        "eval_seed_start": args.seed_start,
        "eval_seed_end": args.seed_start + args.episodes - 1,
        "n_episodes": args.episodes,
        "n_unique_seeds": len({r["eval_seed"] for r in records}),
        "normalized_reward_mean": float(np.mean(norm_rets)),
        "normalized_cost_mean": float(np.mean(norm_costs)),
        "safe_episode_rate": safe_rate,
        "cost_cvar25": cvar25(norm_costs),
        "cost_critic_calibration": calib_summary,
        "records": records,
    }
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")

    pair_fields = [
        "training_seed",
        "eval_seed",
        "step_in_episode",
        "state_id",
        "action_id",
        "predicted_cost",
        "actual_cost_target",
    ]
    with open(out_path.parent / "critic_pairs.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=pair_fields)
        w.writeheader()
        w.writerows(all_calib)
    with open(out_path.parent / "critic_predictions.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f, fieldnames=["state_id", "action_id", "predicted_cost"]
        )
        w.writeheader()
        w.writerows(
            {k: row[k] for k in ("state_id", "action_id", "predicted_cost")}
            for row in all_calib
        )
    with open(out_path.parent / "critic_targets.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f, fieldnames=["state_id", "action_id", "actual_cost_target"]
        )
        w.writeheader()
        w.writerows(
            {k: row[k] for k in ("state_id", "action_id", "actual_cost_target")}
            for row in all_calib
        )
    print(json.dumps({k: v for k, v in out.items() if k != "records"}, indent=2))


if __name__ == "__main__":
    main()
