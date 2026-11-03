#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""COSTPRED_V1：在未使用过的环境回合 600–631 上评估 57 个策略的真实成本（目标变量）。

只读检查点；每回合新建环境 + reset 前播种；动作取 net(obs) 的确定性均值并裁剪。
支持断点续跑：逐 (policy, env_seed) 追加写 CSV，重跑时跳过已完成项。
"""
from __future__ import annotations

import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'carcircle'))
import iql          # noqa: E402
import night_lib as L  # noqa: E402

import warnings
warnings.filterwarnings('ignore')
import gymnasium as gym                      # noqa: E402
import dsrl.offline_bullet_safety_gym  # noqa: E402,F401

OUT = L.WORK / 'costpred'
SEEDS = list(range(600, 632))


class NetWrap(nn.Module):
    """把确定性 BC 网络包成只有 .net 的接口，便于按同一评估代码回放。"""

    def __init__(self, net):
        super().__init__()
        self.net = net

    def log_prob(self, *a, **k):      # 评估不使用
        raise NotImplementedError


def load_policy(path, kind):
    ck = torch.load(path, map_location='cpu', weights_only=False)
    sd = ck['actor'] if 'actor' in ck else (ck['bc'] if 'bc' in ck else ck)
    if kind == 'stochastic_gaussian':
        a = iql.Actor(8, 2)
        a.load_state_dict(sd)
        a.eval()
        return a
    net = iql.mlp(8, 2, 256, nn.Tanh)
    net.load_state_dict(sd)
    net.eval()
    return NetWrap(net)


def eval_episode(policy, env_id, seed):
    env = gym.make(env_id)
    try:
        np.random.seed(int(seed) % (2 ** 31 - 1))
        obs, _ = env.reset(seed=seed)
        r_tot = c_tot = 0.0
        events = []
        t = 0
        done = False
        while not done:
            with torch.no_grad():
                a = policy.net(torch.from_numpy(np.asarray(obs, np.float32)[None, :])).numpy()[0]
            a = np.clip(a, env.action_space.low, env.action_space.high).astype(np.float32)
            obs, r, c, done, _ = L.step_env(env, a)
            r_tot += r
            c_tot += c
            if c > 0:
                events.append(t)
            t += 1
        cut = int(np.floor(0.1 * t))
        return dict(steps=t, reward_sum=r_tot, cost_sum=c_tot, cost_events=len(events),
                    cost_after_10pct=sum(1 for i in events if i >= cut))
    finally:
        env.close()


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--shard', type=int, default=0)
    ap.add_argument('--nshards', type=int, default=1)
    args = ap.parse_args()
    ros = json.loads((OUT / 'policy_roster.json').read_text(encoding='utf-8'))
    raw = OUT / ('targets_raw_shard%d.csv' % args.shard)
    fields = ['policy_id', 'arm', 'split', 'env_seed', 'steps', 'reward_sum', 'cost_sum',
              'cost_events', 'cost_after_10pct']
    done = set()
    rows = []
    if raw.is_file():
        with raw.open(encoding='utf-8-sig', newline='') as f:
            for r in csv.DictReader(f):
                rows.append(r)
                done.add((r['policy_id'], int(r['env_seed'])))
        print('续跑：已有 %d 条回合记录' % len(rows), flush=True)
    t0 = time.time()
    subset = [p for i, p in enumerate(ros['policies']) if i % args.nshards == args.shard]
    for p in subset:
        if not p['loadable']:
            print('跳过不可加载策略', p['policy_id'], flush=True)
            continue
        need = [s for s in SEEDS if (p['policy_id'], s) not in done]
        if not need:
            continue
        policy = load_policy(p['path'], p['kind'])
        for s in need:
            rec = eval_episode(policy, L.ENV_A, s)
            rows.append(dict(policy_id=p['policy_id'], arm=p['arm'], split=p['split'],
                             env_seed=s, **rec))
            if len(rows) % 32 == 0:
                with raw.open('w', encoding='utf-8-sig', newline='') as f:
                    w = csv.DictWriter(f, fieldnames=fields)
                    w.writeheader()
                    w.writerows(rows)
                print('  已写 %d 条（%.0fs）' % (len(rows), time.time() - t0), flush=True)
        print('  完成策略 %s（累计 %d 条，%.0fs）'
              % (p['policy_id'], len(rows), time.time() - t0), flush=True)
    with raw.open('w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    if args.nshards > 1:
        print('分片 %d/%d 完成（%d 条）' % (args.shard, args.nshards, len(rows)))
        return
    # 汇总
    import pandas as pd
    d = pd.read_csv(raw)
    g = d.groupby(['policy_id', 'arm', 'split'])
    summ = g.agg(n_episodes=('cost_sum', 'size'),
                 mean_cost=('cost_sum', 'mean'),
                 std_cost=('cost_sum', 'std'),
                 mean_reward=('reward_sum', 'mean'),
                 mean_cost_after_10pct=('cost_after_10pct', 'mean'),
                 mean_cost_events=('cost_events', 'mean')).reset_index()
    summ['cvar25'] = [float(np.sort(g.get_group(k)['cost_sum'].values)[:max(1, int(0.25 * len(g.get_group(k))))].mean())
                      for k in g.groups]
    summ.to_csv(OUT / 'targets_policy_summary.csv', index=False, encoding='utf-8-sig')
    print('写出 %s（%d 个策略，%d 条回合，%.0fs）'
          % (OUT / 'targets_policy_summary.csv', len(summ), len(rows), time.time() - t0))


if __name__ == '__main__':
    main()
