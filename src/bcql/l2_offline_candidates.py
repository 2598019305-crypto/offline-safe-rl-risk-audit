"""L2 offline stage: build 200-state bank and three candidate sets."""

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

CFG = dict(
    task="OfflineCarGoal1Gymnasium-v0", reward_scale=0.1, cost_scale=1.0,
    cost_limit=10, episode_len=1000,
)


def seed_all(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


def build_model(checkpoint, device):
    m = BCQL(
        state_dim=72, action_dim=2, max_action=1.0,
        a_hidden_sizes=[256, 256], c_hidden_sizes=[256, 256],
        vae_hidden_sizes=400, sample_action_num=10, gamma=0.99, tau=0.005,
        phi=0.05, lmbda=0.75, beta=0.5, PID=[0.1, 0.003, 0.001],
        num_q=2, num_qc=2, cost_limit=10, episode_len=1000, device=device,
    )
    ck = torch.load(checkpoint, map_location=device)
    m.load_state_dict(ck["model_state"])
    m.eval()
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--seed-start", type=int, default=149000)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    seed_all(0)
    model = build_model(args.checkpoint, device)

    env = gym.make(CFG["task"])
    env.set_target_cost(CFG["cost_limit"])
    env = wrap_env(env, reward_scale=CFG["reward_scale"])
    env = OfflineEnvWrapper(env)

    rows = []
    for ep in range(args.episodes):
        seed = args.seed_start + ep
        seed_all(seed)
        obs, _ = env.reset()
        cum = 0.0
        for t in range(CFG["episode_len"]):
            a, _ = model.act(obs)
            rows.append({"episode": ep, "eval_seed": seed, "step": t,
                         "obs": obs.astype(np.float32).tolist(),
                         "cumulative_cost": cum, "action": a.astype(np.float32).tolist()})
            obs, r, term, trunc, info = env.step(a)
            cum += float(info["cost"])
            if term or trunc:
                break
    print("rollout_states", len(rows), flush=True)

    # stratify 50 low / 100 mid / 50 high by cumulative cost
    rows.sort(key=lambda x: x["cumulative_cost"])
    n = len(rows)
    low = rows[: int(n * 0.25)][:50]
    high = rows[-int(n * 0.25):][-50:]
    mid_pool = rows[int(n * 0.25): -int(n * 0.25)]
    mid = mid_pool[:: max(1, len(mid_pool) // 100)][:100]
    bank = low + mid + high
    for i, r in enumerate(bank):
        r["state_id"] = i
        r["risk_group"] = "low" if i < 50 else ("mid" if i < 150 else "high")
    with open(out / "state_bank.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["state_id", "risk_group", "episode", "eval_seed", "step",
                    "cumulative_cost", "action", "observation"])
        for r in bank:
            w.writerow([r["state_id"], r["risk_group"], r["episode"], r["eval_seed"],
                        r["step"], r["cumulative_cost"], r["action"], r["obs"]])

    states = torch.tensor([r["obs"] for r in bank], dtype=torch.float32, device=device)

    # dataset nearest-neighbour actions (10 per state)
    with h5py.File(H5, "r") as f:
        dobs = f["observations"][:].astype(np.float32)
        dact = f["actions"][:].astype(np.float32)
    d_t = torch.tensor(dobs, device=device)
    d_a = torch.tensor(dact, device=device)
    nn_idx = []
    for i in range(0, len(states), 10):
        s = states[i:i + 10]
        d2 = torch.cdist(s, d_t) ** 2
        nn_idx.append(torch.topk(d2, k=10, largest=False).indices)
    nn_idx = torch.cat(nn_idx, 0)

    vae_c, actor_c = [], []
    with torch.no_grad():
        for _ in range(64):
            z = torch.randn((len(states), model.latent_dim), device=device).clamp(-0.5, 0.5)
            v = model.vae.decode(states, z)
            a = model.actor(states, v)
            vae_c.append(v); actor_c.append(a)
    vae_c = torch.stack(vae_c, 1)   # [N,64,2]
    actor_c = torch.stack(actor_c, 1)

    def qvals(obs, acts):
        N, K, A = acts.shape
        o = obs[:, None, :].expand(N, K, obs.shape[-1]).reshape(N * K, -1)
        a = acts.reshape(N * K, -1)
        q1, q2, _, _ = model.critic.predict(o, a)
        c1, c2, _, _ = model.cost_critic.predict(o, a)
        q = torch.min(q1, q2).reshape(N, K)
        c = torch.min(c1, c2).reshape(N, K)
        return q, c

    with torch.no_grad():
        q_data, c_data = qvals(
            states, d_a[nn_idx.reshape(-1)].reshape(len(states), 10, 2))
        q_vae, c_vae = qvals(states, vae_c)
        q_act, c_act = qvals(states, actor_c)

    def dump(name, acts, q, c):
        with open(out / name, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["state_id", "candidate_id", "action_0", "action_1",
                        "Q_reward", "Q_cost"])
            for i in range(acts.shape[0]):
                for k in range(acts.shape[1]):
                    w.writerow([i, k, float(acts[i, k, 0]), float(acts[i, k, 1]),
                                float(q[i, k]), float(c[i, k])])

    dump("data_actions.csv", d_a[nn_idx.reshape(-1)].reshape(len(states), 10, 2),
         q_data, c_data)
    dump("vae_candidates.csv", vae_c, q_vae, c_vae)
    dump("actor_candidates.csv", actor_c, q_act, c_act)

    summary = {
        "n_states": len(states),
        "data_candidates": 10,
        "vae_candidates": 64,
        "actor_candidates": 64,
        "data_q_cost_mean": float(c_data.mean()),
        "vae_q_cost_mean": float(c_vae.mean()),
        "actor_q_cost_mean": float(c_act.mean()),
        "data_q_reward_mean": float(q_data.mean()),
        "vae_q_reward_mean": float(q_vae.mean()),
        "actor_q_reward_mean": float(q_act.mean()),
    }
    (out / "offline_candidate_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
