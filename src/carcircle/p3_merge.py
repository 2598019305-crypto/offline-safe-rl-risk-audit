#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""P3：合并封存评估分片，输出主表（成本均值 / 回报均值 / CVaR@25 上尾）+ 标记封存段已使用。"""
from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import night_lib as L  # noqa: E402

OUT = L.WORK / 'results_p3_sealed'
FS = L.WORK / 'final_stage_20261015'


def cvar_top(costs, q=0.25):
    c = np.sort(np.asarray(costs, float))[::-1]
    return float(c[:max(1, int(round(q * len(c))))].mean())


def main():
    files = sorted(OUT.glob('sealed_raw_shard*.csv'))
    if not files:
        raise SystemExit('没有分片结果')
    d = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    d = d.drop_duplicates(subset=['tag', 'arm', 'train_seed', 'env_seed'], keep='last')
    d.to_csv(OUT / 'sealed_raw.csv', index=False, encoding='utf-8-sig')
    rows = []
    for (tag, arm, seed), g in d.groupby(['tag', 'arm', 'train_seed']):
        rows.append(dict(tag=tag, arm=arm, train_seed=int(seed), n_episodes=len(g),
                         cost_mean=float(g.cost_sum.mean()),
                         cost_median=float(g.cost_sum.median()),
                         cvar25=cvar_top(g.cost_sum.values),
                         reward_mean=float(g.reward_sum.mean()),
                         cost_after_10pct_mean=float(g.cost_after_10pct.mean())))
    ps = pd.DataFrame(rows)
    ps.to_csv(OUT / 'sealed_per_seed.csv', index=False, encoding='utf-8-sig')
    t = ps.groupby('arm').agg(n_train_seeds=('train_seed', 'size'),
                              cost_mean=('cost_mean', 'mean'),
                              cost_sd=('cost_mean', 'std'),
                              cvar25_mean=('cvar25', 'mean'),
                              reward_mean=('reward_mean', 'mean')).reset_index()
    t.to_csv(OUT / 'sealed_unified_table.csv', index=False, encoding='utf-8-sig')
    (OUT / 'sealed_summary.json').write_text(json.dumps(dict(
        n_episodes=len(d), n_policies=len(ps), sealed_env_seeds=[125000, 125099],
        cvar_definition='上尾：成本最高的 25% 回合均值',
        note='一次性封存评估；执行后该种子区间标记为已使用',
        table=t.to_dict(orient='records')), ensure_ascii=False, indent=2), encoding='utf-8')
    reg = FS / 'SEAL_REGISTRY.json'
    if reg.is_file():
        r = json.loads(reg.read_text(encoding='utf-8'))
        r['sealed_env_seeds']['status'] = '已使用（%s，P3 一次性执行）' % time.strftime('%Y-%m-%d %H:%M')
        r['sealed_env_seeds']['used_for'] = 'P3 最终封存评估'
        reg.write_text(json.dumps(r, ensure_ascii=False, indent=2), encoding='utf-8')
    print(t.to_string(index=False))


if __name__ == '__main__':
    main()
