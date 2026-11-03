"""Train a single L1 ablation variant (A0/A1/A2/A3) for 20k steps."""

import argparse
import hashlib
import json
import os
import random
import time
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "disabled")
os.environ.setdefault("MPLBACKEND", "Agg")
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

import sys

sys.path.insert(0, str(SCRIPT_DIR / "osrl_min"))
sys.path.insert(0, str(SCRIPT_DIR))

import dsrl  # noqa: F401
import dsrl.offline_safety_gymnasium  # noqa: F401
from dsrl.offline_env import OfflineEnvWrapper, wrap_env
from osrl.algorithms.bcql import BCQLTrainer
from osrl.common.dataset import TransitionDataset
from bcql_ablation import BCQLAblation

CFG = {
    "task": "OfflineCarGoal1Gymnasium-v0",
    "reward_scale": 0.1,
    "cost_scale": 1.0,
    "actor_lr": 0.001,
    "critic_lr": 0.001,
    "vae_lr": 0.001,
    "phi": 0.05,
    "lmbda": 0.75,
    "beta": 0.5,
    "cost_limit": 10,
    "episode_len": 1000,
    "batch_size": 512,
    "a_hidden_sizes": [256, 256],
    "c_hidden_sizes": [256, 256],
    "vae_hidden_sizes": 400,
    "sample_action_num": 10,
    "gamma": 0.99,
    "tau": 0.005,
    "num_q": 2,
    "num_qc": 2,
    "PID": [0.1, 0.003, 0.001],
}

VARIANTS = {
    "A0": (False, "min", None, False, False),
    "A1": (True, "min", None, False, False),
    "A2": (False, "mean", None, False, False),
    "A3": (True, "mean", None, False, False),
    "A4": (False, "min", 0.0, False, False),
    "A5": (False, "min", None, True, False),
    "A6": (False, "min", None, False, True),
}


class CaptureLogger:
    def __init__(self):
        self.buf = []

    def store(self, **kwargs):
        self.buf.append(kwargs)

    def write(self, *a, **k):
        pass

    def write_without_reset(self, *a, **k):
        pass


class MCDataset(torch.utils.data.IterableDataset):
    def __init__(self, data, mc, reward_scale, cost_scale):
        self.obs = data["observations"]; self.nxt = data["next_observations"]
        self.act = data["actions"]; self.rew = data["rewards"]
        self.cost = data["costs"]; self.mc = mc
        self.done = np.logical_or(data["terminals"], data["timeouts"]).astype(np.float32)
        self.rs = reward_scale; self.cs = cost_scale; self.n = len(self.obs)

    def __iter__(self):
        while True:
            i = np.random.choice(self.n)
            yield (self.obs[i], self.nxt[i], self.act[i],
                   self.rew[i] * self.rs, self.cost[i] * self.cs,
                   self.done[i], self.mc[i])


class AblationTrainer(BCQLTrainer):
    def train_one_step(self, obs, nxt, act, rew, cost, done, mc):
        self.model.vae_loss(obs, act)
        self.model.critic_loss(obs, nxt, act, rew, done)
        self.model.cost_critic_loss_mc(obs, act, mc)
        self.model.actor_loss(obs)
        self.model.sync_weight()


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def param_hash(model):
    h = hashlib.sha256()
    for _, p in sorted(model.state_dict().items()):
        h.update(p.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True, choices=sorted(VARIANTS))
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--log-every", type=int, default=1000)
    ap.add_argument("--eval-episodes", type=int, default=20)
    ap.add_argument("--output-dir", required=True)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    mask, agg, floor, nonneg, cost_mc = VARIANTS[args.variant]

    seed_everything(args.seed)
    env = gym.make(CFG["task"])
    data = env.get_dataset(h5path=str(H5))
    env.set_target_cost(CFG["cost_limit"])
    data = env.pre_process_data(data, None, None, None, None, 1.0)
    env = wrap_env(env, reward_scale=CFG["reward_scale"])
    env = OfflineEnvWrapper(env)

    model = BCQLAblation(
        state_dim=env.observation_space.shape[0],
        action_dim=env.action_space.shape[0],
        max_action=float(env.action_space.high[0]),
        a_hidden_sizes=CFG["a_hidden_sizes"],
        c_hidden_sizes=CFG["c_hidden_sizes"],
        vae_hidden_sizes=CFG["vae_hidden_sizes"],
        sample_action_num=CFG["sample_action_num"],
        gamma=CFG["gamma"],
        tau=CFG["tau"],
        phi=CFG["phi"],
        lmbda=CFG["lmbda"],
        beta=CFG["beta"],
        PID=CFG["PID"],
        num_q=CFG["num_q"],
        num_qc=CFG["num_qc"],
        cost_limit=CFG["cost_limit"],
        episode_len=CFG["episode_len"],
        device=device,
        cost_terminal_mask=mask,
        actor_cost_agg=agg,
        cost_target_floor=floor,
        cost_nonneg=nonneg,
    )
    model.cost_mc = cost_mc
    init_hash = param_hash(model)
    logger = CaptureLogger()
    trainer_cls = AblationTrainer if cost_mc else BCQLTrainer
    trainer = trainer_cls(
        model, env, logger=logger,
        actor_lr=CFG["actor_lr"], critic_lr=CFG["critic_lr"], vae_lr=CFG["vae_lr"],
        reward_scale=CFG["reward_scale"], cost_scale=CFG["cost_scale"], device=device,
    )

    cfg = dict(CFG)
    cfg.update({"variant": args.variant, "seed": args.seed, "steps": args.steps,
                "cost_terminal_mask": mask, "actor_cost_agg": agg,
                "cost_target_floor": floor,
                "cost_nonneg": nonneg,
                "cost_mc": cost_mc,
                "initial_param_sha256": init_hash})
    (out / "config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")

    if cost_mc:
        done = np.logical_or(data["terminals"], data["timeouts"])
        mc = np.zeros(len(data["costs"]), dtype=np.float32)
        start = 0
        for end in np.where(done)[0]:
            acc = 0.0
            for t in range(int(end), start - 1, -1):
                acc = float(data["costs"][t]) + CFG["gamma"] * acc
                mc[t] = acc
            start = int(end) + 1
        dataset = MCDataset(data, mc, CFG["reward_scale"], CFG["cost_scale"])
    else:
        dataset = TransitionDataset(data, reward_scale=CFG["reward_scale"], cost_scale=CFG["cost_scale"])
    loader = torch.utils.data.DataLoader(dataset, batch_size=CFG["batch_size"], num_workers=0)
    it = iter(loader)

    history = []
    t0 = time.time()
    for step in range(args.steps):
        logger.buf.clear()
        batch = [b.to(device) for b in next(it)]
        trainer.train_one_step(*batch)
        if (step + 1) % args.log_every == 0:
            rec = {"step": step + 1, "elapsed": time.time() - t0}
            rec.update(model.last_diag)
            history.append(rec)
            (out / "l1_history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")

    torch.save({"model_state": model.state_dict(), "step": args.steps}, out / "model.pt")
    ret, cost, length = trainer.evaluate(args.eval_episodes)
    eval_rec = {"episodes": args.eval_episodes, "reward": float(ret),
                "cost": float(cost), "length": float(length)}
    (out / "evaluation_20.json").write_text(json.dumps(eval_rec, indent=2), encoding="utf-8")
    (out / "status.json").write_text(json.dumps({
        "status": "completed", "step": args.steps, "variant": args.variant,
        "initial_param_sha256": init_hash, "eval": eval_rec,
    }, indent=2), encoding="utf-8")
    print(json.dumps({"variant": args.variant, "seed": args.seed,
                      "init_sha256": init_hash, "eval": eval_rec}, indent=2))


if __name__ == "__main__":
    main()
