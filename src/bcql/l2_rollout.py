"""L2 real-environment rollout pilot with replay-based state restoration."""

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
import numpy as np
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR / "osrl_min"))
sys.path.insert(0, str(SCRIPT_DIR))

import dsrl  # noqa: F401
import dsrl.offline_safety_gymnasium  # noqa: F401
from dsrl.offline_env import OfflineEnvWrapper, wrap_env
from osrl.algorithms.bcql import BCQL


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
    ap.add_argument("--l2-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--rollouts", type=int, default=5)
    ap.add_argument("--horizon", type=int, default=200)
    ap.add_argument("--max-states", type=int, default=12)
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    seed_all(0)
    model = build_model(args.checkpoint, device)

    def make_env():
        e = gym.make("OfflineCarGoal1Gymnasium-v0")
        e.set_target_cost(10)
        e = wrap_env(e, reward_scale=0.1)
        return OfflineEnvWrapper(e)

    env = make_env()
    bank = list(csv.DictReader(open(Path(args.l2_dir) / "state_bank.csv",
                                    newline="", encoding="utf-8")))
    # 4 low / 4 mid / 4 high
    sel = [r for r in bank if r["risk_group"] == "low"][:4]
    sel += [r for r in bank if r["risk_group"] == "mid"][:4]
    sel += [r for r in bank if r["risk_group"] == "high"][:4]
    sel = sel[: args.max_states]

    def cands(name, state_id, ids):
        rows = list(csv.DictReader(open(Path(args.l2_dir) / name, newline="",
                                        encoding="utf-8")))
        return [r for r in rows if int(r["state_id"]) == state_id and int(r["candidate_id"]) in ids]

    results = []
    for st in sel:
        sid = int(st["state_id"]); ep = int(st["episode"]); tgt = int(st["step"])
        eval_seed = int(st["eval_seed"]); budget = max(0.0, 10.0 - float(st["cumulative_cost"]))
        group = st["risk_group"]
        pool = []
        pool += [(r, "data") for r in cands("data_actions.csv", sid, {0, 7})]
        pool += [(r, "vae") for r in cands("vae_candidates.csv", sid, {0, 21, 42})]
        pool += [(r, "actor") for r in cands("actor_candidates.csv", sid, {0, 21, 42})]
        for cand, src in pool:
            act0 = np.array([float(cand["action_0"]), float(cand["action_1"])], dtype=np.float32)
            for k in range(args.rollouts):
                seed = 500000 + 100 * sid + 10 * k
                seed_all(eval_seed)
                obs, _ = env.reset()
                for _ in range(tgt):
                    a, _ = model.act(obs)
                    obs, r, term, trunc, info = env.step(a)
                    if term or trunc:
                        break
                seed_all(seed)
                obs, r, term, trunc, info = env.step(act0)
                fc = float(info["cost"]); fr = float(r); steps = 1
                for _ in range(args.horizon - 1):
                    if term or trunc:
                        break
                    a, _ = model.act(obs)
                    obs, r, term, trunc, info = env.step(a)
                    fc += float(info["cost"]); fr += float(r); steps += 1
                results.append({
                    "state_id": sid, "risk_group": group, "source": src,
                    "candidate_id": int(cand["candidate_id"]),
                    "rollout_seed": seed, "future_cost": fc,
                    "future_reward": fr, "steps": steps,
                    "remaining_budget": budget,
                    "safe": int(fc <= budget),
                })
        print(f"state {sid} {group} done", flush=True)

    with open(out / "rollout_results.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        w.writeheader(); w.writerows(results)
    # feasibility by state/source
    summary = {}
    for r in results:
        key = (r["risk_group"], r["source"])
        summary.setdefault(key, []).append(r["safe"])
    out_summary = {f"{g}/{s}": {"n": len(v), "safe_rate": sum(v) / len(v)}
                   for (g, s), v in summary.items()}
    (out / "feasibility_summary.json").write_text(json.dumps(out_summary, indent=2),
                                                  encoding="utf-8")
    print(json.dumps(out_summary, indent=2))


if __name__ == "__main__":
    main()
