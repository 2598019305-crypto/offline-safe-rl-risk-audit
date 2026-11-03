import csv
import json
import os
import statistics
from pathlib import Path

import numpy as np
from scipy import stats

BASE = os.environ.get("HALFCHEETAH_WORKDIR", str(Path(__file__).resolve().parent))
EARLY = os.path.join(BASE, "论文核验输出_P3稳定IQL_DE端点")
HC = os.path.join(BASE, "论文核验输出_HC补充_20261016")
OUT = os.path.join(BASE, "论文核验输出_P3独立封存测试_20261020")


def path_for(arm, seed):
    if seed <= 2:
        return os.path.join(EARLY, f"结果_P3_{arm}_种子{seed}_40000步.json")
    return os.path.join(
        HC, f"P3_{arm}_seed{seed}", f"结果_P3_{arm}_种子{seed}_40000步.json"
    )


rows = []
for arm in ("D10", "E100"):
    for seed in range(8):
        d = json.load(open(path_for(arm, seed), encoding="utf-8"))
        ev = d["evaluation"]
        costs = np.asarray(ev["episode_costs"], dtype=float)
        returns = np.asarray(ev["returns"], dtype=float)
        assert ev["type"] == "validation_not_final_test"
        assert ev["seed_start"] == 115000 and len(costs) == len(returns) == 100
        rows.append(
            {
                "arm": arm,
                "training_seed": seed,
                "cost_mean": float(costs.mean()),
                "return_mean": float(returns.mean()),
                "cvar25": float(np.sort(costs)[-25:].mean()),
            }
        )


def arm_summary(arm):
    xs = [r for r in rows if r["arm"] == arm]
    return {
        key: {
            "mean": statistics.fmean(r[key] for r in xs),
            "sd": statistics.stdev(r[key] for r in xs),
            "values": [r[key] for r in xs],
        }
        for key in ("cost_mean", "return_mean", "cvar25")
    }


summaries = {arm: arm_summary(arm) for arm in ("D10", "E100")}
paired = {}
for key in ("cost_mean", "return_mean", "cvar25"):
    a = np.array([r[key] for r in rows if r["arm"] == "D10"])
    b = np.array([r[key] for r in rows if r["arm"] == "E100"])
    diff = a - b
    n = len(diff)
    mean = float(diff.mean())
    sd = float(diff.std(ddof=1))
    se = sd / np.sqrt(n)
    tcrit = float(stats.t.ppf(0.975, n - 1))
    t = stats.ttest_rel(a, b)
    try:
        w = stats.wilcoxon(a, b)
        wp = float(w.pvalue)
    except Exception:
        wp = None
    paired[key] = {
        "mean_diff_d10_minus_e100": mean,
        "sd_diff": sd,
        "ci95_low": mean - tcrit * se,
        "ci95_high": mean + tcrit * se,
        "t_stat": float(t.statistic),
        "t_p_two_sided": float(t.pvalue),
        "wilcoxon_p_two_sided": wp,
        "d10_better_seeds": int((diff < 0).sum() if key != "return_mean" else (diff > 0).sum()),
        "per_seed_diff": diff.tolist(),
    }

agg = {
    "eval_type": "validation_not_final_test",
    "seed_start": 115000,
    "episodes_per_model": 100,
    "n_training_seeds_per_arm": 8,
    "per_seed": rows,
    "summaries": summaries,
    "paired_D10_minus_E100": paired,
}
with open(os.path.join(OUT, "VALIDATION_8SEED_AGGREGATE.json"), "x", encoding="utf-8") as fh:
    json.dump(agg, fh, ensure_ascii=False, indent=2)
with open(os.path.join(OUT, "VALIDATION_8SEED_PER_SEED.csv"), "x", newline="", encoding="utf-8-sig") as fh:
    w = csv.writer(fh)
    w.writerow(["arm", "training_seed", "cost_mean", "return_mean", "cvar25"])
    for r in rows:
        w.writerow([r["arm"], r["training_seed"], r["cost_mean"], r["return_mean"], r["cvar25"]])

for arm in ("D10", "E100"):
    s = summaries[arm]
    print(
        arm,
        "cost %.4f return %.4f cvar %.4f"
        % (s["cost_mean"]["mean"], s["return_mean"]["mean"], s["cvar25"]["mean"]),
    )
for key, p in paired.items():
    print(key, p["mean_diff_d10_minus_e100"], p["d10_better_seeds"])
