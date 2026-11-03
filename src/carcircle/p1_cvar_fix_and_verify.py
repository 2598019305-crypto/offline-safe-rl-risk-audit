#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""修正 CVaR 口径 + 对 BC/RL 的评估一致性做逐条核查（全部基于已保存的逐回合记录，不重训）。

CVaR@25 正确定义：累计成本**最高**的 25% 回合的平均成本，必然满足 CVaR >= 均值。
旧实现写成 np.sort(costs)[:k].mean()（升序取前 k）→ 算成了最差 25 的"最好"部分。
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import iql          # noqa: E402
import night_lib as L  # noqa: E402
import gates        # noqa: E402

import warnings
warnings.filterwarnings('ignore')
import gymnasium as gym                      # noqa: E402
import dsrl.offline_bullet_safety_gym  # noqa: F401,E402
import torch                                 # noqa: E402
import torch.nn as nn                        # noqa: E402

OUT = L.WORK / 'results_p1_unified'
VERIFY = L.WORK / 'final_stage_20261015'


def cvar_top(costs, q=0.25):
    """CVaR@q：成本最高 q 比例的回合的均值（上尾）。"""
    a = np.sort(np.asarray(costs, dtype=float))[::-1]
    k = max(1, int(round(q * len(a))))
    return float(a[:k].mean())


def eval_episode_audited(policy_net, env_id, seed):
    """本轮统一使用的评估器（每回合新建环境 + reset 前播种）。"""
    env = gym.make(env_id)
    try:
        np.random.seed(int(seed) % (2 ** 31 - 1))
        obs, _ = env.reset(seed=seed)
        r_tot = c_tot = 0.0
        events, t, done = [], 0, False
        while not done:
            with torch.no_grad():
                a = policy_net(torch.from_numpy(np.asarray(obs, np.float32)[None, :])).numpy()[0]
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


def load_net(path, kind=None):
    """RL 演员用 iql.Actor.net；BC 检查点必须用 gates._bc_train 的同一结构（隐层 ReLU）。"""
    ck = torch.load(path, map_location='cpu', weights_only=False)
    sd = ck['actor'] if 'actor' in ck else ck['bc']
    if 'log_std' in sd:
        a = iql.Actor(8, 2); a.load_state_dict(sd); a.eval(); return a.net
    # 与 gates._bc_train 完全一致：Linear-ReLU-Linear-ReLU-Linear-Tanh
    net = nn.Sequential(nn.Linear(8, 256), nn.ReLU(), nn.Linear(256, 256), nn.ReLU(),
                        nn.Linear(256, 2), nn.Tanh())
    net.load_state_dict(sd); net.eval(); return net


def main():
    ARMS = [('BC-All', L.WORK / 'results_p1_bc' / 'eval_episode.csv'),
            ('BC-Safe', L.WORK / 'results_p1_bc' / 'eval_episode.csv'),
            ('Reward-only', L.WORK / 'results_8seed_fixed' / 'eval_episode.csv'),
            ('D10', L.WORK / 'results_8seed_fixed' / 'eval_episode.csv'),
            ('E100', L.WORK / 'results_8seed_fixed' / 'eval_episode.csv'),
            ('standard_IQL', L.WORK / 'results_8seed_fixed' / 'eval_episode.csv')]
    res = dict(purpose='CVaR 口径修正 + BC/RL 评估一致性核查（只读取已保存的逐回合记录）',
               cvar_definition='CVaR@25 = 累计成本最高的 25% 回合的平均成本（上尾）')
    per_seed, table = [], []
    for arm, p in ARMS:
        d = pd.read_csv(p)
        d = d[(d.arm == arm) & (d.milestone == 10000)]
        for ts, g in d.groupby('train_seed'):
            c = g.cost_sum.values
            per_seed.append(dict(arm=arm, train_seed=int(ts), n_env=len(g),
                                 cost_mean=float(c.mean()), cost_median=float(np.median(c)),
                                 cvar25_correct=cvar_top(c),
                                 cvar25_old_wrong=float(np.sort(c)[:max(1, len(c)//4)].mean()),
                                 reward_mean=float(g.reward_sum.mean()),
                                 steps_unique=','.join(map(str, sorted(set(g.steps))))))
    ps = pd.DataFrame(per_seed)
    for arm, g in ps.groupby('arm', sort=False):
        table.append(dict(arm=arm, n_seeds=len(g),
                          cost_mean=float(g.cost_mean.mean()), cost_sd=float(g.cost_mean.std(ddof=1)),
                          cost_median=float(g.cost_mean.median()),
                          cvar25_mean=float(g.cvar25_correct.mean()),
                          cvar25_old_wrong_mean=float(g.cvar25_old_wrong.mean()),
                          reward_mean=float(g.reward_mean.mean())))
    t = pd.DataFrame(table)
    t.to_csv(OUT / 'unified_table_cvarfixed.csv', index=False, encoding='utf-8-sig')
    ps.to_csv(OUT / 'per_seed_cvarfixed.csv', index=False, encoding='utf-8-sig')
    res['unified_table_cvar_corrected'] = t.to_dict(orient='records')
    # 一致性：CVaR 必须 >= 均值
    res['cvar_geq_mean_all_rows'] = bool((ps.cvar25_correct >= ps.cost_mean - 1e-9).all())
    # 独立复算：用纯 Python 从逐回合记录重算一遍主表均值（与 pandas 路径交叉验证）
    indep = {}
    for arm, p in ARMS:
        with open(p, encoding='utf-8-sig', newline='') as f:
            rows = [r for r in csv.DictReader(f)
                    if r['arm'] == arm and int(r['milestone']) == 10000]
        by_seed = {}
        for r in rows:
            by_seed.setdefault(int(r['train_seed']), []).append(float(r['cost_sum']))
        means = [sum(v) / len(v) for v in by_seed.values()]
        indep[arm] = dict(n_seeds=len(means), cost_mean=sum(means) / len(means),
                          n_episodes=len(rows))
    res['independent_recompute_means'] = indep
    res['recompute_matches_table'] = bool(all(
        abs(indep[r['arm']]['cost_mean'] - r['cost_mean']) < 1e-9
        for r in res['unified_table_cvar_corrected']))
    # ---- 交叉评估器核查：同一策略、同一批种子，两个评估器实现 ----
    cross = {}
    for tag, path in (('BC-All_seed0', L.WORK / 'checkpoints_p1_bc' / 'bc_BC-All_seed0_10000.pt'),
                      ('Reward-only_seed0', L.WORK / 'checkpoints_fixed' / 'Reward-only_seed0'
                       / 'actor_step10000.pt')):
        if not Path(path).is_file():
            cross[tag] = 'missing'
            continue
        net = load_net(path)
        mine = [dict(env_seed=s, **eval_episode_audited(net, L.ENV_A, s)) for s in range(510, 518)]
        cross[tag] = mine
    res['cross_evaluator'] = cross

    def same(a, b, keys=('steps', 'reward_sum', 'cost_sum', 'cost_events', 'cost_after_10pct')):
        return all(abs(a[k] - b[k]) < 1e-9 for k in keys)

    verdict = {}
    for tag, path in (('BC-All_seed0', L.WORK / 'results_p1_bc' / 'eval_episode.csv'),
                      ('Reward-only_seed0', L.WORK / 'results_8seed_fixed' / 'eval_episode.csv')):
        arm = 'BC-All' if tag.startswith('BC') else 'Reward-only'
        d = pd.read_csv(path)
        stored = d[(d.arm == arm) & (d.milestone == 10000) & (d.train_seed == 0)]
        stored = {int(r.env_seed): r for _, r in stored.iterrows()}
        got = res['cross_evaluator'].get(tag, [])
        ok = all(same({k: r[k] for k in
                       ('steps', 'reward_sum', 'cost_sum', 'cost_events', 'cost_after_10pct')},
                      {k: stored[int(r['env_seed'])][k] for k in
                       ('steps', 'reward_sum', 'cost_sum', 'cost_events', 'cost_after_10pct')})
                 for r in got)
        verdict[tag] = dict(reproduced=bool(ok), n=len(got))
    res['cross_evaluator_verdict'] = verdict
    (OUT / 'cvar_fix_and_verify.json').write_text(json.dumps(res, ensure_ascii=False, indent=2,
                                                             default=float),
                                                  encoding='utf-8')
    print(t.to_string(index=False))
    print('\nCVaR>=均值（全部行）:', res['cvar_geq_mean_all_rows'])
    print('独立复算与主表一致:', res['recompute_matches_table'])
    print('交叉评估器复现:', json.dumps(verdict, ensure_ascii=False))


if __name__ == '__main__':
    main()
