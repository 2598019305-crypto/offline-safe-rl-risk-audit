"""Official-equivalent BCQ-Lag (BCQL) training entry for OfflineCarGoal1Gymnasium-v0.

Algorithm core (BCQL / BCQLTrainer / networks) is imported unchanged from the
frozen OSRL commit. This wrapper only replaces the wandb/logger and the dataset
download with a local frozen HDF5, and adds JSON logging + full-state resume.
"""

import argparse
import json
import math
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

import dsrl  # noqa: F401
import dsrl.offline_safety_gymnasium  # noqa: F401
from dsrl.offline_env import OfflineEnvWrapper, wrap_env
from osrl.algorithms.bcql import BCQL, BCQLTrainer
from osrl.common.dataset import TransitionDataset


CONFIG = {
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


def seed_everything(seed):
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


def save_state(path, model, step):
    torch.save(
        {
            "model_state": model.state_dict(),
            "actor_optim": model.actor_optim.state_dict(),
            "critic_optim": model.critic_optim.state_dict(),
            "cost_critic_optim": model.cost_critic_optim.state_dict(),
            "vae_optim": model.vae_optim.state_dict(),
            "controller": {
                "error_old": float(model.controller.error_old),
                "error_integral": float(model.controller.error_integral),
            },
            "step": step,
            "rng": {
                "python": random.getstate(),
                "numpy": np.random.get_state(),
                "torch": torch.get_rng_state(),
            },
        },
        path,
    )


def load_state(path, model, device):
    ck = torch.load(path, map_location=device)
    model.load_state_dict(ck["model_state"])
    model.actor_optim.load_state_dict(ck["actor_optim"])
    model.critic_optim.load_state_dict(ck["critic_optim"])
    model.cost_critic_optim.load_state_dict(ck["cost_critic_optim"])
    model.vae_optim.load_state_dict(ck["vae_optim"])
    model.controller.error_old = ck["controller"]["error_old"]
    model.controller.error_integral = ck["controller"]["error_integral"]
    random.setstate(ck["rng"]["python"])
    np.random.set_state(ck["rng"]["numpy"])
    torch.set_rng_state(ck["rng"]["torch"])
    return int(ck["step"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--steps", type=int, default=100000)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--eval-every", type=int, default=2500)
    ap.add_argument("--eval-episodes", type=int, default=10)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--device", default=None)
    ap.add_argument(
        "--disable-cost",
        action="store_true",
        help="Experimental control arm: zero out the Lagrangian cost penalty "
        "at runtime without modifying the frozen bcql.py update formulas.",
    )
    args = ap.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    out = Path(args.output_dir)
    ckpt_dir = out / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    cfg = dict(CONFIG)
    cfg["seed"] = args.seed
    cfg["update_steps"] = args.steps
    cfg["batch_size"] = args.batch_size
    cfg["eval_every"] = args.eval_every
    cfg["eval_episodes"] = args.eval_episodes
    cfg["disable_cost"] = args.disable_cost

    env = gym.make(cfg["task"])
    data = env.get_dataset(h5path=str(H5))
    env.set_target_cost(cfg["cost_limit"])
    data = env.pre_process_data(data, None, None, None, None, 1.0)
    env = wrap_env(env, reward_scale=cfg["reward_scale"])
    env = OfflineEnvWrapper(env)

    model = build_model(cfg, env, device)
    trainer = BCQLTrainer(
        model,
        env,
        logger=type("L", (), {"store": lambda *a, **k: None})(),
        actor_lr=cfg["actor_lr"],
        critic_lr=cfg["critic_lr"],
        vae_lr=cfg["vae_lr"],
        reward_scale=cfg["reward_scale"],
        cost_scale=cfg["cost_scale"],
        device=device,
    )

    start_step = 0
    model_path = ckpt_dir / "model.pt"
    best_path = ckpt_dir / "model_best.pt"
    if model_path.exists():
        start_step = load_state(model_path, model, device)
    if args.disable_cost:
        def _zero_control(qc):
            return torch.zeros((), device=qc.device)

        model.controller.control = _zero_control

    if start_step == 0:
        seed_everything(args.seed)
        (out / "config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")

    dataset = TransitionDataset(data, reward_scale=cfg["reward_scale"], cost_scale=cfg["cost_scale"])
    loader = torch.utils.data.DataLoader(dataset, batch_size=cfg["batch_size"], num_workers=0)
    it = iter(loader)

    best_cost = math.inf
    best_reward = -math.inf
    history = []
    t0 = time.time()

    for step in range(start_step, args.steps):
        batch = [b.to(device) for b in next(it)]
        trainer.train_one_step(*batch)

        if (step + 1) % cfg["eval_every"] == 0 or step == args.steps - 1:
            ret, cost, length = trainer.evaluate(cfg["eval_episodes"])
            norm_ret, norm_cost = env.get_normalized_score(ret, cost)
            save_state(model_path, model, step + 1)
            if cost < best_cost or (cost == best_cost and ret > best_reward):
                best_cost = cost
                best_reward = ret
                save_state(best_path, model, step + 1)
            record = {
                "step": step + 1,
                "reward": float(ret),
                "cost": float(cost),
                "length": float(length),
                "normalized_reward": float(norm_ret),
                "normalized_cost": float(norm_cost),
                "best_cost": float(best_cost),
                "elapsed": time.time() - t0,
            }
            history.append(record)
            (out / "status.json").write_text(
                json.dumps(
                    {
                        "status": "completed" if step == args.steps - 1 else "running",
                        "step": step + 1,
                        "history": history,
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
            print(json.dumps(record))

    print("TRAIN_DONE")


if __name__ == "__main__":
    main()
