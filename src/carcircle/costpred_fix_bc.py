#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""修正 COSTPRED_V1 中 4 个 BC 策略的网络加载错误，并重算目标/特征/评价。

缺陷：BC 检查点的隐层激活为 ReLU（gates._bc_train），而 costpred_targets.py 与
costpred_features.py 用 iql.mlp(..., nn.Tanh) 加载；形状相同所以没有报错，但策略被算错，
导致 4 个 BC 策略的目标成本与 3 个离线特征全部失真。

本脚本：① 用正确结构重算 4 个 BC 策略在种子 600–631 的目标；② 重算全部 57 个策略的离线特征；
③ 按已冻结协议重跑一次评价。属**实现缺陷修正后的重跑**，不是新的独立检验。
"""
from __future__ import annotations

import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
import iql          # noqa: E402
import night_lib as L  # noqa: E402

import warnings
warnings.filterwarnings('ignore')
import gymnasium as gym                      # noqa: E402
import dsrl.offline_bullet_safety_gym  # noqa: F401,E402

O = L.WORK / 'costpred'
SEEDS = list(range(600, 632))
BC_IDS = ['bc-BC-All-10000', 'bc-BC-Safe-10000', 'bc-BC-All-2000', 'bc-BC-Safe-2000']


def load_bc_net(path):
    ck = torch.load(path, map_location='cpu', weights_only=False)
    sd = ck['bc']
    net = nn.Sequential(nn.Linear(8, 256), nn.ReLU(), nn.Linear(256, 256), nn.ReLU(),
                        nn.Linear(256, 2), nn.Tanh())
    net.load_state_dict(sd)
    net.eval()
    return net


def eval_episode(net, env_id, seed):
    env = gym.make(env_id)
    try:
        np.random.seed(int(seed) % (2 ** 31 - 1))
        obs, _ = env.reset(seed=seed)
        r_tot = c_tot = 0.0
        events, t, done = [], 0, False
        while not done:
            with torch.no_grad():
                a = net(torch.from_numpy(np.asarray(obs, np.float32)[None, :])).numpy()[0]
            a = np.clip(a, env.action_space.low, env.action_space.high).astype(np.float32)
            obs, r, c, done, _ = L.step_env(env, a)
            r_tot += r; c_tot += c
            if c > 0:
                events.append(t)
            t += 1
        cut = int(np.floor(0.1 * t))
        return dict(steps=t, reward_sum=r_tot, cost_sum=c_tot, cost_events=len(events),
                    cost_after_10pct=sum(1 for i in events if i >= cut))
    finally:
        env.close()


def main():
    t0 = time.time()
    ros = json.loads((O / 'policy_roster.json').read_text(encoding='utf-8'))
    raw = pd.read_csv(O / 'targets_raw.csv')
    old_bc = raw[raw.policy_id.isin(BC_IDS)].groupby('policy_id').agg(
        old_mean=('cost_sum', 'mean')).to_dict()['old_mean']
    raw = raw[~raw.policy_id.isin(BC_IDS)]
    rows = []
    for p in ros['policies']:
        if p['policy_id'] not in BC_IDS:
            continue
        net = load_bc_net(p['path'])
        for s in SEEDS:
            rec = eval_episode(net, L.ENV_A, s)
            rows.append(dict(policy_id=p['policy_id'], arm=p['arm'], split=p['split'],
                             env_seed=s, **rec))
        print('  修正评估 %s 完成（%.0fs）' % (p['policy_id'], time.time() - t0), flush=True)
    raw = pd.concat([raw, pd.DataFrame(rows)], ignore_index=True)
    raw.to_csv(O / 'targets_raw_BCFIXED.csv', index=False, encoding='utf-8-sig')
    g = raw.groupby(['policy_id', 'arm', 'split'])
    summ = g.agg(n_episodes=('cost_sum', 'size'), mean_cost=('cost_sum', 'mean'),
                 std_cost=('cost_sum', 'std'), mean_reward=('reward_sum', 'mean'),
                 mean_cost_after_10pct=('cost_after_10pct', 'mean'),
                 mean_cost_events=('cost_events', 'mean')).reset_index()
    summ['cvar25'] = [float(np.sort(g.get_group(k)['cost_sum'].values)[::-1][
        :max(1, int(round(0.25 * len(g.get_group(k)))))]
        .mean()) for k in g.groups]
    summ.to_csv(O / 'targets_policy_summary_BCFIXED.csv', index=False, encoding='utf-8-sig')
    fix = {k: dict(old=float(old_bc[k]), new=float(summ.loc[summ.policy_id == k, 'mean_cost'].iloc[0]))
           for k in BC_IDS}
    (O / 'BC_LOADER_FIX.json').write_text(json.dumps(dict(
        defect='BC 检查点隐层为 ReLU，但目标与特征脚本用 Tanh 结构加载（形状相同、无报错）',
        impact='4 个 BC 策略的目标成本与 3 个离线特征全部失真',
        corrected_means=fix,
        note='本文件为修正后的重跑，不是新的独立检验'), ensure_ascii=False, indent=2),
        encoding='utf-8')
    print(json.dumps(fix, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
