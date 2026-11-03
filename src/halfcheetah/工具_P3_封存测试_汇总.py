import csv
import hashlib
import json
import math
import os
import statistics
from pathlib import Path

import numpy as np
from scipy import stats

BASE = os.environ.get("HALFCHEETAH_WORKDIR", str(Path(__file__).resolve().parent))
OUT = os.environ.get(
    "P3_TEST_DIR",
    os.path.join(BASE, "论文核验输出_P3独立封存测试_20261020"),
)
EXPECTED_SEED_START = int(os.environ.get("P3_SEED_START", "125000"))


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


with open(os.path.join(OUT, "PRE_REG_FINAL_TEST.json"), encoding="utf-8") as fh:
    pre = json.load(fh)
expected_models = {(m["arm"], m["training_seed"]): m for m in pre["model_checkpoints"]}

rows = []
for arm in ("D10", "E100"):
    for seed in range(8):
        path = os.path.join(
            OUT, f"结果_final_sealed_test_{arm}_训练种子{seed}_100回合.json"
        )
        with open(path, encoding="utf-8") as fh:
            d = json.load(fh)
        costs = np.asarray(d["episode_costs"], dtype=float)
        returns = np.asarray(d["returns"], dtype=float)
        assert len(costs) == len(returns) == 100
        assert np.isfinite(costs).all() and np.isfinite(returns).all()
        rec_cost = float(costs.mean())
        rec_return = float(returns.mean())
        rec_cvar = float(np.sort(costs)[-25:].mean())
        model_expected = expected_models[(arm, seed)]
        checks = {
            "model_sha": d["model_sha256"] == model_expected["sha256"],
            "seed_start": d["seed_start"] == EXPECTED_SEED_START,
            "episodes": d["num_episodes"] == 100,
            "eval_type": d["evaluation_type"] == "final_sealed_test",
            "cost_mean": abs(rec_cost - d["cost_mean"]) < 1e-6,
            "return_mean": abs(rec_return - d["return_mean"]) < 1e-4,
            "cvar": abs(rec_cvar - d["episode_CVaR25"]) < 1e-6,
        }
        if not all(checks.values()):
            raise RuntimeError(f"check failed {arm} {seed}: {checks}")
        rows.append(
            {
                "arm": arm,
                "training_seed": seed,
                "cost_mean": rec_cost,
                "return_mean": rec_return,
                "cvar25": rec_cvar,
                "seconds": d["seconds"],
                "result_path": path,
                "result_sha256": sha256(path),
                "model_sha256": d["model_sha256"],
            }
        )


def arm_summary(arm):
    xs = [r for r in rows if r["arm"] == arm]
    out = {}
    for key in ("cost_mean", "return_mean", "cvar25"):
        vals = [r[key] for r in xs]
        out[key] = {
            "mean": statistics.fmean(vals),
            "sd": statistics.stdev(vals),
            "values": vals,
        }
    return out


summaries = {arm: arm_summary(arm) for arm in ("D10", "E100")}
paired = {}
for key in ("cost_mean", "return_mean", "cvar25"):
    d10 = np.array([r[key] for r in rows if r["arm"] == "D10"])
    e100 = np.array([r[key] for r in rows if r["arm"] == "E100"])
    diff = d10 - e100
    n = len(diff)
    mean = float(diff.mean())
    sd = float(diff.std(ddof=1))
    se = sd / math.sqrt(n)
    tcrit = float(stats.t.ppf(0.975, n - 1))
    ttest = stats.ttest_rel(d10, e100)
    try:
        wil = stats.wilcoxon(d10, e100)
        wil_p = float(wil.pvalue)
    except Exception:
        wil_p = None
    paired[key] = {
        "mean_diff_d10_minus_e100": mean,
        "sd_diff": sd,
        "ci95_low": mean - tcrit * se,
        "ci95_high": mean + tcrit * se,
        "t_stat": float(ttest.statistic),
        "t_p_two_sided": float(ttest.pvalue),
        "wilcoxon_p_two_sided": wil_p,
        "d10_better_seeds": int((diff < 0).sum() if key != "return_mean" else (diff > 0).sum()),
        "per_seed_diff": diff.tolist(),
    }

aggregate = {
    "protocol": pre["protocol"],
    "eval_type": "final_sealed_test",
    "seed_start": EXPECTED_SEED_START,
    "episodes_per_model": 100,
    "n_training_seeds_per_arm": 8,
    "models": 16,
    "all_consistency_checks": True,
    "per_seed": rows,
    "summaries": summaries,
    "paired_D10_minus_E100": paired,
}
agg_path = os.path.join(OUT, "FINAL_TEST_AGGREGATE.json")
with open(agg_path, "x", encoding="utf-8") as fh:
    json.dump(aggregate, fh, ensure_ascii=False, indent=2)

csv_path = os.path.join(OUT, "FINAL_TEST_PER_SEED.csv")
with open(csv_path, "x", newline="", encoding="utf-8-sig") as fh:
    w = csv.writer(fh)
    w.writerow(["arm", "training_seed", "cost_mean", "return_mean", "cvar25", "seconds", "result_sha256"])
    for r in rows:
        w.writerow([r["arm"], r["training_seed"], r["cost_mean"], r["return_mean"], r["cvar25"], r["seconds"], r["result_sha256"]])

print(f"=== FINAL SEALED TEST (seeds {EXPECTED_SEED_START}-{EXPECTED_SEED_START+99}) ===")
for arm in ("D10", "E100"):
    s = summaries[arm]
    print(
        f"{arm}: cost {s['cost_mean']['mean']:.4f} (sd {s['cost_mean']['sd']:.4f}) | "
        f"return {s['return_mean']['mean']:.4f} (sd {s['return_mean']['sd']:.4f}) | "
        f"CVaR25 {s['cvar25']['mean']:.4f} (sd {s['cvar25']['sd']:.4f})"
    )
for key in ("cost_mean", "return_mean", "cvar25"):
    p = paired[key]
    print(
        f"{key}: diff={p['mean_diff_d10_minus_e100']:.4f} "
        f"CI95=[{p['ci95_low']:.4f},{p['ci95_high']:.4f}] "
        f"t_p={p['t_p_two_sided']:.6g} wilcoxon_p={p['wilcoxon_p_two_sided']} "
        f"better={p['d10_better_seeds']}/8"
    )
print("wrote", agg_path)
print("wrote", csv_path)
