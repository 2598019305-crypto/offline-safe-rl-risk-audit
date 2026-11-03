#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""E3e：SafetySwimmerVelocity 的 20 回合离线数据 pilot（只采集与质量核验，不训练）。

行为策略（**事前固定**，比例 50/50，按固定 RNG 逐步选择）：
  模式 R：a ~ Uniform(-1,1)^2
  模式 G：周期步态 a = (sin(2πt/20), sin(2πt/20 + π))
采集 20 回合 × 1000 步，逐步保存 (s_t, a_t, r_t, c_t, s_{t+1}, done_t)，
并单独统计首次进入成本、持续接触、动作饱和比例、每回合累计成本、corr(cost,‖a‖²)。
"""
from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np

EID = 'SafetySwimmerVelocity-v1'
EPISODES = 20
HORIZON = 1000
POLICY_SEED = 20261013
MIX_R = 0.5
GAIT_PERIOD = 20


def main():
    import safety_gymnasium
    project = Path(__file__).resolve().parent
    out = project / '论文核验输出_E3e_Swimmer离线数据pilot_20261013'
    if out.exists():
        raise FileExistsError('输出目录已存在: ' + str(out))
    out.mkdir(parents=True)
    t0 = time.time()
    env = safety_gymnasium.make(EID)
    o, _ = env.reset(seed=0)
    no = int(np.asarray(o).shape[0]); na = int(env.action_space.shape[0])
    rng = np.random.RandomState(POLICY_SEED)
    S, A, R, C, S2, D, T, EP, MD = [], [], [], [], [], [], [], [], []
    for ep in range(EPISODES):
        obs, _ = env.reset(seed=ep)
        obs = np.asarray(obs, dtype=np.float64)
        for t in range(HORIZON):
            if rng.rand() < MIX_R:
                a = rng.uniform(-1, 1, size=na); mode = 0
            else:
                ph = 2 * math.pi * t / GAIT_PERIOD
                a = np.array([math.sin(ph), math.sin(ph + math.pi)]); mode = 1
            o2, r, c, te, tr, _ = env.step(a)
            o2 = np.asarray(o2, dtype=np.float64)
            done = bool(te or tr)
            S.append(obs); A.append(np.asarray(a, dtype=np.float64)); R.append(float(r))
            C.append(float(c)); S2.append(o2); D.append(done); T.append(int(t)); EP.append(int(ep)); MD.append(int(mode))
            obs = o2
            if done:
                break
    env.close()
    S = np.array(S); A = np.array(A); R = np.array(R); C = np.array(C); S2 = np.array(S2)
    D = np.array(D); T = np.array(T); EP = np.array(EP); MD = np.array(MD)
    sq = np.sum(A ** 2, axis=1)
    # 事件类型：首次进入（本步计费且上一步不计费且同一回合）vs 持续接触
    prevC = np.zeros(len(C))
    for i in range(1, len(C)):
        prevC[i] = C[i - 1] if EP[i] == EP[i - 1] else 0.0
    entry = (C > 0) & (prevC == 0)
    persist = (C > 0) & (prevC > 0)
    per_ep_cost = np.array([C[EP == k].sum() for k in range(EPISODES)])
    per_ep_len = np.array([(EP == k).sum() for k in range(EPISODES)])
    res = dict(env=EID, episodes=EPISODES, steps=int(len(C)),
               policy=f'50% uniform random + 50% periodic gait(period={GAIT_PERIOD}), policy_seed={POLICY_SEED}',
               policy_mode_R_rate=float((MD == 0).mean()),
               reward_mean=float(R.mean()), reward_std=float(R.std()),
               per_episode_return=[float(R[EP == k].sum()) for k in range(EPISODES)],
               per_episode_return_mean=float(per_ep_cost.mean() * 0 + np.mean([R[EP == k].sum() for k in range(EPISODES)])),
               cost_event_rate=float((C > 0).mean()),
               first_entry_count=int(entry.sum()), persistent_count=int(persist.sum()),
               first_entry_per_episode_mean=float(entry.sum() / EPISODES),
               per_episode_cost_mean=float(per_ep_cost.mean()),
               per_episode_cost_std=float(per_ep_cost.std(ddof=1)),
               per_episode_cost=[float(x) for x in per_ep_cost],
               episode_length_mean=float(per_ep_len.mean()),
               action_saturation_ratio=float((np.abs(A) > 0.99).mean()),
               action_norm2_mean=float(sq.mean()),
               corr_cost_actnorm2=(float(np.corrcoef(sq, C)[0, 1]) if (sq.std() > 0 and C.std() > 0) else None),
               obs_mean=[float(x) for x in S.mean(0)], obs_std=[float(x) for x in S.std(0)],
               has_next_state=True,
               next_state_cross_episode_invalidated=True,
               elapsed_sec=round(time.time() - t0, 1))
    np.savez_compressed(out / '数据_SwimmerVelocity_自采pilot.npz',
                        obs=S.astype(np.float32), act=A.astype(np.float32), rew=R.astype(np.float32),
                        cost=C.astype(np.float32), next_obs=S2.astype(np.float32),
                        done=D, t=T, episode=EP, mode=MD.astype(np.int8))
    with (out / '核验与边界.json').open('x', encoding='utf8') as f:
        json.dump(res, f, ensure_ascii=False, indent=2, default=float)
    for k, v in res.items():
        if k not in ('per_episode_cost', 'per_episode_return', 'obs_mean', 'obs_std'):
            print(f'  {k} = {v}')
    print('\n输出:', out)


if __name__ == '__main__':
    main()
