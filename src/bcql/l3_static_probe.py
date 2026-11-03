"""Static L3 probe: reward vs cost actor gradients and PID multiplier."""

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "disabled")
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("DSRL_SUPPRESS_IMPORT_ERROR", "1")

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

from osrl.algorithms.bcql import BCQL

CFG = dict(
    state_dim=72, action_dim=2, max_action=1.0,
    a_hidden_sizes=[256, 256], c_hidden_sizes=[256, 256],
    vae_hidden_sizes=400, sample_action_num=10, gamma=0.99, tau=0.005,
    phi=0.05, lmbda=0.75, beta=0.5, PID=[0.1, 0.003, 0.001],
    num_q=2, num_qc=2, cost_limit=10, episode_len=1000, device="cpu",
)


def flat_grad(loss, params, retain_graph=False):
    gs = torch.autograd.grad(loss, params, retain_graph=retain_graph, allow_unused=True)
    return torch.cat([(g if g is not None else torch.zeros_like(p)).reshape(-1)
                      for g, p in zip(gs, params)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--n", type=int, default=1024)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    torch.manual_seed(0)
    np.random.seed(0)
    with h5py.File(H5, "r") as f:
        obs_all = f["observations"][:]
        cost_all = f["costs"][:]
        term_all = f["terminals"][:]
        tout_all = f["timeouts"][:]
    idx = np.random.choice(len(obs_all), args.n, replace=False)
    obs = torch.tensor(obs_all[idx], dtype=torch.float32)

    model = BCQL(**CFG)
    ck = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(ck["model_state"])
    if "controller" in ck:
        model.controller.error_old = ck["controller"]["error_old"]
        model.controller.error_integral = ck["controller"]["error_integral"]
    model.eval()

    with torch.no_grad():
        z = torch.randn((obs.shape[0], model.latent_dim)).clamp(-0.5, 0.5)
        a = model.actor(obs, model.vae.decode(obs, z))
        q1, q2, _, _ = model.critic.predict(obs, a)
        qc1, qc2, _, _ = model.cost_critic.predict(obs, a)
        q_pi = torch.min(q1, q2)
        qc_min = torch.min(qc1, qc2)
        qc_mean = 0.5 * (qc1 + qc2)
        err_old, err_int = model.controller.error_old, model.controller.error_integral
        multiplier = model.controller.control(qc_min)
        model.controller.error_old, model.controller.error_integral = err_old, err_int
        lam = float(multiplier.item())
        thres = float(model.qc_thres)

    params = [p for p in model.actor.parameters() if p.requires_grad]
    z = torch.randn((obs.shape[0], model.latent_dim)).clamp(-0.5, 0.5)
    a = model.actor(obs, model.vae.decode(obs, z))
    q1, q2, _, _ = model.critic.predict(obs, a)
    qc1, qc2, _, _ = model.cost_critic.predict(obs, a)
    q_pi = torch.min(q1, q2)
    qc_min = torch.min(qc1, qc2)
    L_R = -q_pi.mean()
    L_C = (qc_min - thres).mean()
    g_R = flat_grad(L_R, params, retain_graph=True)
    g_C = flat_grad(L_C, params, retain_graph=False)
    nR, nC = float(g_R.norm().item()), float(g_C.norm().item())
    cos = float(torch.nn.functional.cosine_similarity(g_R, g_C, dim=0).item())

    # data-based discounted cost-to-go (horizon 100)
    horizon = 100
    g = 0.99
    targets = []
    for i in idx:
        s = 0.0
        for k in range(horizon):
            j = i + k
            if j >= len(cost_all):
                break
            s += (g ** k) * float(cost_all[j])
            if term_all[j] or tout_all[j]:
                break
        targets.append(s)
    targets = np.asarray(targets)

    result = {
        "checkpoint": args.checkpoint,
        "n_states": args.n,
        "q_reward_mean": float(q_pi.mean().item()),
        "qc_min_mean": float(qc_min.mean().item()),
        "qc_mean_agg_mean": float(qc_mean.mean().item()),
        "qc_negative_ratio": float((qc_min < 0).float().mean().item()),
        "qc_threshold": thres,
        "lagrangian": lam,
        "grad_reward_norm": nR,
        "grad_cost_term_norm": nC,
        "grad_cost_scaled_norm": abs(lam) * nC,
        "grad_ratio_cost_over_reward": abs(lam) * nC / (nR + 1e-12),
        "grad_cosine_reward_cost": cos,
        "data_cost_to_go_mean": float(targets.mean()),
        "data_cost_to_go_p90": float(np.quantile(targets, 0.9)),
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
