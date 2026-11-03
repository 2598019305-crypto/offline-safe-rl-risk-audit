#!/usr/bin/env python3
"""Evaluate frozen P3 D10/E100 checkpoints on the pre-registered sealed seeds."""

import argparse
import hashlib
import importlib.util
import json
import os
import time
from pathlib import Path

import numpy as np
import torch

BASE = os.environ.get("HALFCHEETAH_WORKDIR", str(Path(__file__).resolve().parent))
EARLY = os.path.join(BASE, "论文核验输出_P3稳定IQL_DE端点")
HC = os.path.join(BASE, "论文核验输出_HC补充_20261016")
OUT = os.path.join(BASE, "论文核验输出_P3独立封存测试_20261020")

P3_PATH = os.path.join(BASE, "工具_Day21_P3_稳定IQL_DE端点复验_v1_1.py")
P3_EXT_PATH = os.path.join(BASE, "工具_Day21_P3_稳定IQL_DE端点复验_扩种子v1.py")
HC_PATH = os.path.join(BASE, "工具_HC_异质驱动.py")
STAGE5_PATH = os.path.join(BASE, "阶段5_HalfCheetah机制验证.py")
FIXED_PATH = os.path.join(
    os.environ.get(
        "FIXED_BATCH_ENGINE",
        os.path.join(BASE, "..", "shared", "公共_定批回放引擎.py"),
    )
)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def import_local(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def model_path(arm, seed):
    if seed <= 2:
        return os.path.join(EARLY, f"模型_P3_{arm}_种子{seed}_40000步.pt")
    return os.path.join(
        HC, f"P3_{arm}_seed{seed}", f"模型_P3_{arm}_种子{seed}_40000步.pt"
    )


def load_model(p3, arm, seed):
    path = model_path(arm, seed)
    ck = torch.load(path, map_location="cpu", weights_only=False)
    if ck.get("arm") != arm or ck.get("seed") != seed or ck.get("steps") != 40000:
        raise RuntimeError(f"checkpoint metadata mismatch: {path} {ck.get('arm')} {ck.get('seed')} {ck.get('steps')}")
    model = p3.StablePolicy(
        np.zeros((1, 17), dtype=np.float32), np.ones((1, 17), dtype=np.float32)
    )
    model.load_state_dict(ck["state_dict"], strict=True)
    model.eval()
    return model, path, ck


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=["D10", "E100"])
    ap.add_argument("--seed", type=int, required=True, choices=list(range(8)))
    ap.add_argument("--episodes", type=int, default=100)
    ap.add_argument("--seed-start", type=int, default=125000)
    ap.add_argument("--parallel", type=int, default=6)
    ap.add_argument("--outdir", default=OUT)
    ap.add_argument("--label", default="final_sealed_test")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    torch.set_num_threads(2)
    os.makedirs(args.outdir, exist_ok=True)

    p3 = import_local(P3_PATH, "p3_frozen_eval")
    hc_drv = import_local(HC_PATH, "p3_hc_driver_eval")
    hc = hc_drv.装好(0.75)
    pre = p3.环境预飞(hc)
    if not pre["passed"]:
        raise SystemExit(f"environment cost preflight failed: {pre}")

    model, mpath, ck = load_model(p3, args.arm, args.seed)
    t0 = time.time()
    print(
        f"[start] {args.arm} seed{args.seed} episodes={args.episodes} "
        f"seed_start={args.seed_start} parallel={args.parallel}",
        flush=True,
    )
    ev = hc.评估(
        model,
        回合=args.episodes,
       并行=args.parallel,
        种子=args.seed_start,
    )
    elapsed = time.time() - t0
    result = {
        "protocol": "P3_FINAL_SEALED_TEST_20261020",
        "evaluation_type": args.label,
        "arm": args.arm,
        "training_seed": args.seed,
        "training_steps": 40000,
        "model_path": mpath,
        "model_sha256": sha256(mpath),
        "model_metadata": {
            "arm": ck.get("arm"),
            "seed": ck.get("seed"),
            "steps": ck.get("steps"),
        },
        "seed_start": args.seed_start,
        "num_episodes": args.episodes,
        "parallel": args.parallel,
        "cost_threshold": float(hc.数据阈值[0]),
        "horizon": int(hc.固定视界),
        "action": "deterministic_mean_action",
        "environment_cost_preflight": pre,
        "data_path": hc.数据路,
        "data_sha256": "e0850f2e929a3892806f99109937ca323a74f0d624f4fd5fcc027f7c111e8179",
        "code_sha256": {
            "p3_v1_1": sha256(P3_PATH),
            "p3_expanded_seed_v1": sha256(P3_EXT_PATH),
            "hc_driver": sha256(HC_PATH),
            "stage5": sha256(STAGE5_PATH),
            "fixed_batch_engine": sha256(FIXED_PATH),
        },
        "return_mean": float(ev["回报均值"]),
        "cost_mean": float(ev["成本均值"]),
        "episode_CVaR25": float(ev["成本CVaR25"]),
        "returns": np.asarray(ev["逐回合回报"]).tolist(),
        "episode_costs": np.asarray(ev["逐回合成本"]).tolist(),
        "seconds": round(elapsed, 2),
    }
    name = f"结果_{args.label}_{args.arm}_训练种子{args.seed}_{args.episodes}回合.json"
    path = os.path.join(args.outdir, name)
    mode = "w" if args.overwrite else "x"
    with open(path, mode, encoding="utf-8") as fh:
        json.dump(result, fh, ensure_ascii=False, indent=2)
    print(
        f"[done] {args.arm} seed{args.seed} cost={result['cost_mean']:.6f} "
        f"return={result['return_mean']:.6f} cvar={result['episode_CVaR25']:.6f} "
        f"seconds={elapsed:.1f} -> {path}",
        flush=True,
    )


if __name__ == "__main__":
    main()
