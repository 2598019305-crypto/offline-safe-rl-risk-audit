#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""E4-2：重做 Swimmer 离线数据设计 + 用固定参照检验奖励可学性（不训练演员）。

动机：E4-1c/1d/1e 已把在线/EMA 目标 V、截断掩码、expectile 方向三处都排除，
     固定参照显示 Q 几乎不超过平凡基线 ⇒ 问题在数据侧（奖励信号太弱、覆盖不足）。

阶段 probe  ：在一组**事前固定**的周期步态里挑"平均每步奖励最高"的一支（选择规则先写死）。
阶段 collect：用 75% 最优步态 + 25% 均匀随机（固定 RNG）采 20 回合 × 1000 步，
             保存 (s,a,r,c,s',done,terminated,truncated)，并给出奖励/覆盖/事件质量。
阶段 learn  ：前 14 条拟合、后 6 条诊断，τ_V=0.3、EMA 目标 V、截断当终止（已确立的最佳配置），
             2000 步；**主判据** = 留出轨迹上 Q 对 MC 回报的 RMSE ÷ 常数基线 ≤ 0.8。
"""
from __future__ import annotations

import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

EID = 'SafetySwimmerVelocity-v1'
EPISODES = 20
HORIZON = 1000
REF_TRAJ = 14
GAITS = [(10, 1.0, math.pi), (20, 1.0, math.pi), (40, 1.0, math.pi),
         (20, 0.5, math.pi), (20, 1.0, math.pi / 2), (30, 1.0, math.pi)]
MIX_GAIT = 0.75
POLICY_SEED = 20261014
GAMMA, LR, EMA, CLIP, BATCH = 0.99, 3e-4, 0.005, 10.0, 256
TAU_V = 0.3
MILESTONES = (0, 500, 1000, 2000)
阈值 = 0.8


def net(i, o):
    return nn.Sequential(nn.Linear(i, 256), nn.ReLU(), nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, o))


class QV(nn.Module):
    def __init__(self, no, na):
        super().__init__()
        self.q1 = net(no + na, 1); self.q2 = net(no + na, 1); self.vr = net(no, 1)
        self.register_buffer('obs_mean', torch.zeros(1, no))
        self.register_buffer('obs_std', torch.ones(1, no))

    def normalized(self, s):
        return (s - self.obs_mean) / self.obs_std


def expectile(diff, tau):
    return (torch.where(diff > 0, tau, 1 - tau) * diff.square()).mean()


def 步态(a, na, t, period, amp, phase):
    ph = 2 * math.pi * t / period
    if na == 2:
        return np.array([amp * math.sin(ph), amp * math.sin(ph + phase)])
    return amp * np.sin(ph + np.linspace(0, 2 * phase, na))


def main():
    import safety_gymnasium
    project = Path(__file__).resolve().parent
    out = project / '论文核验输出_E4_2_Swimmer数据重设计_20261013'
    if out.exists():
        raise FileExistsError('输出目录已存在: ' + str(out))
    out.mkdir(parents=True)
    t0 = time.time()
    env = safety_gymnasium.make(EID)
    o, _ = env.reset(seed=0)
    no, na = int(np.asarray(o).shape[0]), int(env.action_space.shape[0])
    # ---------- 阶段 probe：步态标定（选择规则：3 个固定种子上平均每步奖励最高） ----------
    probe = []
    for (period, amp, phase) in GAITS:
        rets = []
        for s in (300, 301, 302):
            env.reset(seed=s); tot = 0.0
            for t in range(HORIZON):
                oo, r, c, te, tr, _ = env.step(步态(None, na, t, period, amp, phase))
                tot += float(r)
                if te or tr:
                    break
            rets.append(tot / HORIZON)
        probe.append(dict(period=period, amp=amp, phase=round(phase, 4),
                          mean_reward_per_step=float(np.mean(rets))))
    best = max(probe, key=lambda x: x['mean_reward_per_step'])
    print('步态标定：', [(p['period'], p['amp'], round(p['mean_reward_per_step'], 4)) for p in probe])
    print('选定步态（规则先写死：平均每步奖励最高）：', best, flush=True)
    # ---------- 阶段 collect ----------
    env.close(); env = safety_gymnasium.make(EID)
    rng = np.random.RandomState(POLICY_SEED)
    S, A, R, C, S2, D, TE, TR, EP = [], [], [], [], [], [], [], [], []
    for ep in range(EPISODES):
        obs_, _ = env.reset(seed=ep)
        obs_ = np.asarray(obs_, dtype=np.float64)
        for t in range(HORIZON):
            if rng.rand() < MIX_GAIT:
                a = 步态(None, na, t, best['period'], best['amp'], best['phase'])
            else:
                a = rng.uniform(-1, 1, size=na)
            o2, r, c, te, tr, _ = env.step(np.asarray(a, dtype=np.float64))
            o2 = np.asarray(o2, dtype=np.float64)
            S.append(obs_); A.append(a); R.append(float(r)); C.append(float(c))
            S2.append(o2); D.append(bool(te or tr)); TE.append(bool(te)); TR.append(bool(tr)); EP.append(ep)
            obs_ = o2
            if te or tr:
                break
    env.close()
    S = np.array(S); A = np.array(A); R = np.array(R); C = np.array(C); S2 = np.array(S2)
    D = np.array(D); TE = np.array(TE); TR = np.array(TR); EP = np.array(EP)
    per_ep = np.array([R[EP == k].sum() for k in range(EPISODES)])
    sq = np.sum(A ** 2, axis=1)
    qual = dict(steps=int(len(R)), episodes=EPISODES,
                reward_per_step_mean=float(R.mean()), reward_per_step_std=float(R.std()),
                per_episode_return_mean=float(per_ep.mean()), per_episode_return_std=float(per_ep.std(ddof=1)),
                per_episode_return=[float(x) for x in per_ep],
                cost_event_rate=float((C > 0).mean()),
                action_saturation=float((np.abs(A) > 0.99).mean()),
                corr_cost_actnorm2=float(np.corrcoef(sq, C)[0, 1]) if sq.std() > 0 else None,
                terminated_count=int(TE.sum()), truncated_count=int(TR.sum()),
                alignment_ok=bool(all(np.array_equal(S2[np.where(EP == k)[0][:-1]], S[np.where(EP == k)[0][1:]])
                                      for k in range(EPISODES))))
    np.savez_compressed(out / '数据_SwimmerVelocity_重设计pilot.npz',
                        obs=S.astype(np.float32), act=A.astype(np.float32), rew=R.astype(np.float32),
                        cost=C.astype(np.float32), next_obs=S2.astype(np.float32),
                        done=D, terminated=TE, truncated=TR, episode=EP)
    print('采集质量：', {k: (round(v, 4) if isinstance(v, float) else v) for k, v in qual.items()
                     if k != 'per_episode_return'}, flush=True)
    # ---------- 阶段 learn ----------
    ref = EP < REF_TRAJ
    obs_mean = torch.from_numpy(S[ref].mean(0, keepdims=True))
    obs_std = torch.from_numpy(np.maximum(S[ref].std(0, keepdims=True), 1e-8))
    St = torch.from_numpy(S.astype(np.float32)); At = torch.from_numpy(A.astype(np.float32))
    Rt = torch.from_numpy(R.astype(np.float32)).reshape(-1, 1)
    St2 = torch.from_numpy(S2.astype(np.float32))
    Dt = torch.from_numpy(D.astype(np.float32)).reshape(-1, 1)
    ref_idx = np.where(ref)[0]; ho_idx = np.where(~ref)[0]
    mc = np.zeros(len(R), dtype=np.float32)
    for k in range(EPISODES):
        m_ = np.where(EP == k)[0]
        g = 0.0
        for i in range(len(m_) - 1, -1, -1):
            g = R[m_[i]] + GAMMA * g
            mc[m_[i]] = g
    mc_const = float(np.sqrt(np.mean((mc[ho_idx] - mc[ref_idx].mean()) ** 2)))
    rng2 = np.random.RandomState(0)
    批次 = [rng2.choice(ref_idx, size=BATCH) for _ in range(MILESTONES[-1])]
    torch.manual_seed(0)
    m = QV(no, na); mt = QV(no, na)
    m.obs_mean.copy_(obs_mean); m.obs_std.copy_(obs_std)
    mt.load_state_dict(m.state_dict())
    oq = torch.optim.Adam(list(m.q1.parameters()) + list(m.q2.parameters()), lr=LR)
    ov = torch.optim.Adam(m.vr.parameters(), lr=LR)
    stages = []

    def diag(step):
        m.eval()
        with torch.no_grad():
            sn = m.normalized(St); sn2 = m.normalized(St2)
            q1 = m.q1(torch.cat([sn, At], -1)); v = m.vr(sn)
            y = Rt + GAMMA * (1 - Dt) * mt.vr(mt.normalized(St2))
            adv = q1 - v
        q1n = q1.numpy().reshape(-1); vn = v.numpy().reshape(-1); advn = adv.numpy().reshape(-1)
        rmse = float(np.sqrt(np.mean((q1n[ho_idx] - mc[ho_idx]) ** 2)))
        return dict(step=step, q_absmax=float(np.abs(q1n).max()), v_absmax=float(np.abs(vn).max()),
                    mc_Q_rmse=rmse, mc_const_rmse=mc_const, 主判据比值=float(rmse / mc_const),
                    corr_Q_vs_MC=float(np.corrcoef(q1n[ho_idx], mc[ho_idx])[0, 1]),
                    corr_adv_vs_MC=float(np.corrcoef(advn[ho_idx], mc[ho_idx])[0, 1]))

    stages.append(diag(0))
    step = 0
    for target in MILESTONES[1:]:
        m.train()
        while step < target:
            j = torch.from_numpy(批次[step])
            with torch.no_grad():
                tv = Rt[j] + GAMMA * (1 - Dt[j]) * mt.vr(mt.normalized(St2[j]))
                qt = torch.minimum(mt.q1(torch.cat([mt.normalized(St[j]), At[j]], -1)),
                                   mt.q2(torch.cat([mt.normalized(St[j]), At[j]], -1)))
            q1 = m.q1(torch.cat([m.normalized(St[j]), At[j]], -1))
            q2 = m.q2(torch.cat([m.normalized(St[j]), At[j]], -1))
            loss = (q1 - tv).square().mean() + (q2 - tv).square().mean() \
                   + expectile(qt - m.vr(m.normalized(St[j])), TAU_V)
            oq.zero_grad(); ov.zero_grad(); loss.backward()
            for n_ in (m.q1, m.q2, m.vr):
                nn.utils.clip_grad_norm_(n_.parameters(), CLIP)
            oq.step(); ov.step()
            with torch.no_grad():
                for a_, b_ in ((m.q1, mt.q1), (m.q2, mt.q2), (m.vr, mt.vr)):
                    for pa, pb in zip(a_.parameters(), b_.parameters()):
                        pb.mul_(1 - EMA).add_(pa, alpha=EMA)
            step += 1
        stages.append(diag(step))
        torch.save({'step': step, 'state_dict': m.state_dict()}, out / f'模型_E4_2_种子0_{step}步.pt')
        print(f'learn stage {step} 完成', flush=True)
    final = stages[-1]
    res = dict(data_is_self_collected=True, gait_probe=probe, best_gait=best,
               policy=f'{int(MIX_GAIT*100)}% 最优周期步态 + {int((1-MIX_GAIT)*100)}% 均匀随机, seed={POLICY_SEED}',
               quality=qual, mc_const_rmse_reference=mc_const,
               primary_criterion=f'mc_Q_rmse ÷ mc_const_rmse ≤ {阈值}',
               stages=stages, passed=bool(final['主判据比值'] <= 阈值),
               elapsed_sec=round(time.time() - t0, 1))
    with (out / '核验与边界.json').open('x', encoding='utf8') as f:
        json.dump(res, f, ensure_ascii=False, indent=2, default=float)
    for s in stages:
        print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in s.items()})
    print('\n主判据达标:', res['passed'], '| 输出:', out, '| 用时', res['elapsed_sec'], 's')


if __name__ == '__main__':
    main()
