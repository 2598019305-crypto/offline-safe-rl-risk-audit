#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""P3：一次性封存评估——预先锁定的全部模型 × 100 个封存环境种子（125000–125099）。

规则（见 SEAL_REGISTRY.json 与 CPQ_FULL_BUDGET_LOCK.json）：
  - 每回合新建环境对象 + reset 前 np.random.seed(env_seed)（已审计协议）
  - 只运行预先指定的模型，不因结果挑选或替换
  - 指标：每训练种子的 100 回合原生成本均值、回报均值、CVaR@25（**上尾**：成本最高的 25% 回合均值）
  - 执行后立即把封存段标记为"已使用"
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
import iql          # noqa: E402
import night_lib as L  # noqa: E402

import warnings
warnings.filterwarnings('ignore')
import gymnasium as gym                      # noqa: E402
import dsrl.offline_bullet_safety_gym  # noqa: F401,E402

OUT = L.WORK / 'results_p3_sealed'
SEALED = list(range(125000, 125100))
FS = L.WORK / 'final_stage_20261015'


class Wrap(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net


def load_policy(path):
    ck = torch.load(path, map_location='cpu', weights_only=False)
    sd = ck['actor'] if 'actor' in ck else ck.get('bc')
    if sd is None:
        raise RuntimeError('unknown checkpoint format: %s' % path)
    if 'log_std' in sd:
        a = iql.Actor(8, 2)
        a.load_state_dict(sd)
        a.eval()
        return a
    net = nn.Sequential(nn.Linear(8, 256), nn.ReLU(), nn.Linear(256, 256), nn.ReLU(),
                        nn.Linear(256, 2), nn.Tanh())
    net.load_state_dict(sd)
    net.eval()
    return Wrap(net)


def eval_episode(policy, env_id, seed):
    env = gym.make(env_id)
    try:
        np.random.seed(int(seed) % (2 ** 31 - 1))
        obs, _ = env.reset(seed=seed)
        r_tot = c_tot = 0.0
        events, t, done = [], 0, False
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


def policy_list():
    """预先锁定的模型清单：统一主表六臂 + CPQ-10k（若已完成）。"""
    pol = []
    for arm in ('BC-All', 'BC-Safe'):
        for s in range(8):
            p = L.WORK / 'checkpoints_p1_bc' / ('bc_%s_seed%d_10000.pt' % (arm, s))
            if p.is_file():
                pol.append(('p1bc', arm, s, p))
    for arm in ('Reward-only', 'D10', 'E100', 'standard_IQL'):
        for s in range(8):
            p = (L.WORK / 'checkpoints_fixed' / ('%s_seed%d' % (arm, s)) / 'actor_step10000.pt')
            if p.is_file():
                pol.append(('fixed', arm, s, p))
    for s in range(8):
        p = L.WORK / 'checkpoints_p1_cpq' / ('cpq_seed%d.pt' % s)
        if p.is_file():
            pol.append(('cpq10k', 'CPQ-10k', s, p))
    return pol


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--shard', type=int, default=0)
    ap.add_argument('--nshards', type=int, default=1)
    ap.add_argument('--limit-policies', type=int, default=0)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    pol = policy_list()
    if a.limit_policies:
        pol = pol[:a.limit_policies]
    my = [p for i, p in enumerate(pol) if i % a.nshards == a.shard]
    raw = OUT / ('sealed_raw_shard%d.csv' % a.shard)
    rows = []
    if raw.is_file():
        with raw.open(encoding='utf-8-sig', newline='') as f:
            rows = list(csv.DictReader(f))
    done = {(r['tag'], r['arm'], r['train_seed'], int(r['env_seed'])) for r in rows}
    print('本分片模型数 %d；已有 %d 条回合' % (len(my), len(rows)), flush=True)
    for tag, arm, seed, path in my:
        need = [s for s in SEALED if (tag, arm, str(seed), s) not in done]
        if not need:
            continue
        policy = load_policy(path)
        for s in need:
            rec = eval_episode(policy, L.ENV_A, s)
            rows.append(dict(tag=tag, arm=arm, train_seed=seed, env_seed=s, **rec))
        with raw.open('w', encoding='utf-8-sig', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        print('  %s/%s seed%d 完成（累计 %d 条，%.0fs）'
              % (tag, arm, seed, len(rows), time.time() - t0), flush=True)
    print('分片 %d 完成：%d 条' % (a.shard, len(rows)))


if __name__ == '__main__':
    main()
