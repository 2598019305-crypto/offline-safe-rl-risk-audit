#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""P2：门控单因素消融训练（B1–B5 × 训练种子 0–2）。

同一 Q/V 主干（P1_seed0/1/2 按种子对应）、同一 actor 初始化、同一 minibatch 流、
固定 β=0.5 λ=1 clip=100、相同候选枚举（稳定最近优先截断）与同一风险数组
（八折交叉拟合口径，20261017 折规则），仅门控 g 不同：
  B1 g≡0（Reward-only）  B2 g≡1（D10）  B3 support-only  B4 完整门控
  B5 B4 门控在 minibatch 内保持开启数置乱
评估：开发种子 510–517，里程碑 0/2000/10000，审计协议。
"""
from __future__ import annotations

import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import iql          # noqa: E402
import night_lib as L  # noqa: E402

import warnings
warnings.filterwarnings('ignore')
import gymnasium as gym                      # noqa: E402
import dsrl.offline_bullet_safety_gym  # noqa: E402,F401

OUT = L.WORK / 'results_p2'
CKPT = L.WORK / 'checkpoints_p2'
ARMS = ('B1_reward_only', 'B2_gate_full_on', 'B3_support_only', 'B4_full_gate', 'B5_shuffled_gate')
SEEDS = (0, 1, 2)
ENVSEEDS = list(range(510, 518))
MILESTONES = (0, 2000, 10000)
SHUFFLE_SEED = 20261014


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
    OUT.mkdir(parents=True, exist_ok=True)
    CKPT.mkdir(parents=True, exist_ok=True)
    log = L.Log('p2_train')
    data = iql.load_and_split(L.ENV_A)
    tr = data['train_idx'].astype(np.int64)
    cf = np.load(L.WORK / 'results_crossfit' / 'd10_crossfit_train.npy')
    risk_full = np.zeros(len(data['S']), np.float64)
    risk_full[tr] = np.nan_to_num(cf, nan=0.0)
    gz = np.load(L.WORK / 'agrw' / 'gates_p2.npz')
    gates = {k: {s: gz['gate_%s_seed%d' % (k, s)].astype(np.float64) for s in SEEDS}
             for k in ('B3', 'B4')}
    rows = []
    for seed in SEEDS:
        crit = L.WORK / 'checkpoints' / ('P1_seed%d' % seed) / 'critic_step40000.pt'
        gf_b3 = np.zeros(len(data['S']), np.float64); gf_b3[tr] = gates['B3'][seed]
        gf_b4 = np.zeros(len(data['S']), np.float64); gf_b4[tr] = gates['B4'][seed]
        plan = {'B1_reward_only': (None, None, None),
                'B2_gate_full_on': (risk_full, None, None),
                'B3_support_only': (risk_full, gf_b3, None),
                'B4_full_gate': (risk_full, gf_b4, None),
                'B5_shuffled_gate': (risk_full, gf_b4, SHUFFLE_SEED + seed)}
        for arm in ARMS:
            risk, gate, shuf = plan[arm]

            def cb(actor, step, seed=seed, arm=arm):
                if step not in MILESTONES:
                    return
                for es in ENVSEEDS:
                    rows.append(dict(arm=arm, train_seed=seed, milestone=step, env_seed=es,
                                     **eval_episode(actor, L.ENV_A, es)))
                with (OUT / 'eval_episode.csv').open('w', encoding='utf-8-sig', newline='') as f:
                    w = csv.DictWriter(f, fieldnames=list(rows[0]))
                    w.writeheader(); w.writerows(rows)
                sel = [r for r in rows if r['arm'] == arm and r['train_seed'] == seed
                       and r['milestone'] == step]
                log('  eval %s seed%d step%d: 成本 %.2f 回报 %.2f'
                    % (arm, seed, step, float(np.mean([x['cost_sum'] for x in sel])),
                       float(np.mean([x['reward_sum'] for x in sel]))))

            def extra(j, p, g_used):
                if g_used is None:
                    return dict(gate_sample_share=None, gate_weight_share=None)
                gj = np.asarray(g_used) > 0.5
                return dict(gate_sample_share=float(gj.mean()),
                            gate_weight_share=float(p[gj].sum()) if gj.any() else 0.0)

            log('=== P2 %s seed%d ===' % (arm, seed))
            iql.train_actor(crit, data, seed, arm, log, risk=risk, operator='mech',
                            updates=10000, eval_cb=cb, ckpt_root=CKPT, gate_full=gate,
                            extra_log=extra, gate_shuffle_seed=shuf)
    print('P2 训练完成，用时 %.0fs' % (time.time() - t0))


if __name__ == '__main__':
    main()
