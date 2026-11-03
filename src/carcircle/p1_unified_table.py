#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""P1：统一 8 训练种子 × 8 开发环境种子的主表（10k 演员更新，审计评估协议）。

数据来源：
  Reward-only / D10 / E100 / standard-IQL → results_8seed_fixed/eval_episode.csv（既有，复用）
  BC-All / BC-Safe                      → results_p1_bc/eval_episode.csv（本轮新增 10k × 8 种子）
输出 results_p1_unified/：unified_table.csv、unified_table.json、per_seed.csv
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import night_lib as L  # noqa: E402

OUT = L.WORK / 'results_p1_unified'
ARMS = [('BC-All', L.WORK / 'results_p1_bc' / 'eval_episode.csv'),
        ('BC-Safe', L.WORK / 'results_p1_bc' / 'eval_episode.csv'),
        ('Reward-only', L.WORK / 'results_8seed_fixed' / 'eval_episode.csv'),
        ('D10', L.WORK / 'results_8seed_fixed' / 'eval_episode.csv'),
        ('E100', L.WORK / 'results_8seed_fixed' / 'eval_episode.csv'),
        ('standard_IQL', L.WORK / 'results_8seed_fixed' / 'eval_episode.csv')]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    per_seed = []
    for arm, p in ARMS:
        d = pd.read_csv(p)
        d = d[(d.arm == arm) & (d.milestone == 10000)]
        for ts, g in d.groupby('train_seed'):
            costs = np.sort(g.cost_sum.values)
            per_seed.append(dict(arm=arm, train_seed=int(ts), n_env=len(g),
                                 cost_mean=float(g.cost_sum.mean()),
                                 cost_median=float(np.median(g.cost_sum)),
                                 cost_min=float(costs[0]), cost_max=float(costs[-1]),
                                 cost_q1=float(np.percentile(g.cost_sum, 25)),
                                 reward_mean=float(g.reward_sum.mean()),
                                 cvar25=float(costs[:max(1, int(0.25 * len(costs)))].mean()),
                                 cost_after_10pct_mean=float(g.cost_after_10pct.mean())))
    ps = pd.DataFrame(per_seed)
    for arm, g in ps.groupby('arm', sort=False):
        rows.append(dict(arm=arm, n_train_seeds=len(g),
                         cost_mean_across_seeds=float(g.cost_mean.mean()),
                         cost_sd_across_seeds=float(g.cost_mean.std(ddof=1)),
                         cost_median_across_seeds=float(g.cost_mean.median()),
                         cost_min_seed=float(g.cost_mean.min()),
                         cost_max_seed=float(g.cost_mean.max()),
                         cvar25_mean=float(g.cvar25.mean()),
                         reward_mean_across_seeds=float(g.reward_mean.mean()),
                         reward_sd_across_seeds=float(g.reward_mean.std(ddof=1))))
    t = pd.DataFrame(rows)
    t.to_csv(OUT / 'unified_table.csv', index=False, encoding='utf-8-sig')
    ps.to_csv(OUT / 'per_seed.csv', index=False, encoding='utf-8-sig')
    (OUT / 'unified_table.json').write_text(json.dumps(
        dict(milestone=10000, dev_env_seeds=[510, 517],
             note='训练种子为统计单位；外部 CPQ 参考另表（缩减预算 10k，官方为 100k）',
             table=t.to_dict(orient='records')), ensure_ascii=False, indent=2), encoding='utf-8')
    print(t.to_string(index=False))
    print()
    print(ps.pivot(index='train_seed', columns='arm', values='cost_mean').round(1).to_string())


if __name__ == '__main__':
    main()
