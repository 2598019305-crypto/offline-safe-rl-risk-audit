#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""把分片评估结果合并为 targets_raw.csv 与 targets_policy_summary.csv。"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'carcircle'))
import night_lib as L  # noqa: E402

OUT = L.WORK / 'costpred'
SEEDS = list(range(600, 632))


def main():
    files = sorted(OUT.glob('targets_raw_shard*.csv'))
    if not files:
        raise SystemExit('没有分片结果')
    d = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    d = d.drop_duplicates(subset=['policy_id', 'env_seed'], keep='last')
    ros = json.loads((OUT / 'policy_roster.json').read_text(encoding='utf-8'))
    want = sum(1 for p in ros['policies'] if p['loadable']) * len(SEEDS)
    if len(d) != want:
        raise SystemExit('回合数不完整：%d / %d' % (len(d), want))
    d = d.sort_values(['policy_id', 'env_seed'])
    d.to_csv(OUT / 'targets_raw.csv', index=False, encoding='utf-8-sig')
    g = d.groupby(['policy_id', 'arm', 'split'])
    summ = g.agg(n_episodes=('cost_sum', 'size'), mean_cost=('cost_sum', 'mean'),
                 std_cost=('cost_sum', 'std'), mean_reward=('reward_sum', 'mean'),
                 mean_cost_after_10pct=('cost_after_10pct', 'mean'),
                 mean_cost_events=('cost_events', 'mean')).reset_index()
    summ['cvar25'] = [float(np.sort(g.get_group(k)['cost_sum'].values)[
        :max(1, int(0.25 * len(g.get_group(k))))].mean()) for k in g.groups]
    summ.to_csv(OUT / 'targets_policy_summary.csv', index=False, encoding='utf-8-sig')
    print('合并完成：%d 条回合，%d 个策略' % (len(d), len(summ)))


if __name__ == '__main__':
    main()
