#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""E4-3：Swimmer 最小 reward-only 行为策略训练 + 同参照验收（一次跑完，后台可续）。

训练：PPO（obs→高斯策略），单种子；上限 100,000 次环境交互或 90 分钟墙钟，先到为止。
      奖励只用环境原生 reward（前向速度 − 控制代价）；**不使用任何安全风险信号**。
验收：用训练后的策略采 20 条轨迹 → 保存 (s,a,r,c,s',done,terminated,truncated)
      → 前 14 条拟合、后 6 条诊断，τ_V=0.3 + EMA 目标 V、2000 步，
      主判据 = 留出轨迹上 Q 对 MC 回报的 RMSE ÷ 常数基线 ≤ 0.8。
"""
from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

EID = 'SafetySwimmerVelocity-v1'
MAX_STEPS = 100_000
MAX_MINUTES = 90
ROLLOUT = 2048
EPOCHS, MINIBATCH, LR, CLIP_EPS, GAMMA, LAM = 4, 256, 3e-4, 0.2, 0.99, 0.95
SEED = 0
COLLECT_EP = 20
HORIZON = 1000
REF_TRAJ = 14
TAU_V, EMA_QV, LR_QV, CLIP_QV, BATCH_QV = 0.3, 0.005, 3e-4, 10.0, 256
MILESTONES = (0, 500, 1000, 2000)
阈值 = 0.8


def mlp(i, o, h=64):
    return nn.Sequential(nn.Linear(i, h), nn.Tanh(), nn.Linear(h, h), nn.Tanh(), nn.Linear(h, o))


class ActorCritic(nn.Module):
    def __init__(self, no, na):
        super().__init__()
        self.pi = mlp(no, na); self.v = mlp(no, 1)
        self.log_std = nn.Parameter(torch.full((na,), -0.7))

    def dist(self, s):
        mu = self.pi(s)
        return torch.distributions.Normal(mu, self.log_std.exp().expand_as(mu))


def main():
    import safety_gymnasium
    project = Path(__file__).resolve().parent
    out = project / '论文核验输出_E4_3_行为策略训练与验收_20261013'
    if out.exists():
        raise FileExistsError('输出目录已存在: ' + str(out))
    out.mkdir(parents=True)
    t0 = time.time()
    torch.manual_seed(SEED); np.random.seed(SEED)
    env = safety_gymnasium.make(EID)
    o, _ = env.reset(seed=0)
    no, na = int(np.asarray(o).shape[0]), int(env.action_space.shape[0])
    net = ActorCritic(no, na)
    opt = torch.optim.Adam(net.parameters(), lr=LR)
    obs_, _ = env.reset(seed=SEED)
    obs_ = np.asarray(obs_, dtype=np.float32)
    total = 0
    hist = []
    while total < MAX_STEPS and (time.time() - t0) / 60 < MAX_MINUTES:
        S, A, LP, R, D, V = [], [], [], [], [], []
        for _ in range(ROLLOUT):
            s = torch.from_numpy(obs_)
            with torch.no_grad():
                d = net.dist(s.unsqueeze(0)); a = d.sample(); lp = d.log_prob(a).sum(-1)
                v = net.v(s.unsqueeze(0)).squeeze(-1)
            an = a.squeeze(0).numpy()
            o2, r, c, te, tr, _ = env.step(np.clip(an, -1, 1))
            S.append(obs_); A.append(an); LP.append(float(lp)); R.append(float(r))
            done = bool(te or tr); D.append(done); V.append(float(v))
            obs_ = np.asarray(o2, dtype=np.float32); total += 1
            if done:
                obs_, _ = env.reset(seed=1000 + total)
                obs_ = np.asarray(obs_, dtype=np.float32)
        S = torch.from_numpy(np.array(S)); A = torch.from_numpy(np.array(A))
        LP = torch.from_numpy(np.array(LP)); R = torch.from_numpy(np.array(R))
        D = torch.from_numpy(np.array(D, dtype=np.float32)); V = torch.from_numpy(np.array(V))
        with torch.no_grad():
            adv = torch.zeros_like(R); last = 0.0
            for t in range(len(R) - 1, -1, -1):
                nv = V[t + 1] if t + 1 < len(R) else 0.0
                delta = R[t] + GAMMA * (1 - D[t]) * nv - V[t]
                last = delta + GAMMA * LAM * (1 - D[t]) * last
                adv[t] = last
            ret = adv + V
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        idx = np.arange(ROLLOUT)
        for _ in range(EPOCHS):
            np.random.shuffle(idx)
            for b in range(0, ROLLOUT, MINIBATCH):
                j = torch.from_numpy(idx[b:b + MINIBATCH])
                d = net.dist(S[j]); lp = d.log_prob(A[j]).sum(-1)
                ratio = (lp - LP[j]).exp()
                pg = -torch.min(ratio * adv[j], ratio.clamp(1 - CLIP_EPS, 1 + CLIP_EPS) * adv[j]).mean()
                vl = (net.v(S[j]).squeeze(-1) - ret[j]).square().mean()
                opt.zero_grad(); (pg + 0.5 * vl).backward()
                nn.utils.clip_grad_norm_(net.parameters(), 0.5); opt.step()
        v_now = net.v(torch.from_numpy(obs_).unsqueeze(0)).item()
        hist.append(dict(steps=total, v=float(v_now), mean_return_in_rollout=float(R.mean())))
        print(f'  PPO 步数 {total}  rollout 平均奖励 {R.mean():+.4f}  V {v_now:+.2f}  '
              f'已用 {(time.time()-t0)/60:.1f} 分钟', flush=True)
    torch.save({'state_dict': net.state_dict(), 'steps': total}, out / '模型_E4_3_行为策略.pt')
    print(f'训练结束：{total} 步环境交互，用时 {(time.time()-t0)/60:.1f} 分钟', flush=True)
    # ---------- 用训练后的策略采数据 ----------
    rng = np.random.RandomState(20261015)
    S2, A2, R2, C2, N2, D2, TE2, TR2, EP2 = [], [], [], [], [], [], [], [], []
    for ep in range(COLLECT_EP):
        ob, _ = env.reset(seed=ep)
        ob = np.asarray(ob, dtype=np.float32)
        for t in range(HORIZON):
            with torch.no_grad():
                a = net.dist(torch.from_numpy(ob).unsqueeze(0)).sample().squeeze(0).numpy()
            a = np.clip(a, -1, 1)
            o2, r, c, te, tr, _ = env.step(a)
            o2 = np.asarray(o2, dtype=np.float32)
            S2.append(ob); A2.append(a); R2.append(float(r)); C2.append(float(c))
            N2.append(o2); D2.append(bool(te or tr)); TE2.append(bool(te)); TR2.append(bool(tr)); EP2.append(ep)
            ob = o2
            if te or tr:
                break
    env.close()
    S2 = np.array(S2); A2 = np.array(A2); R2 = np.array(R2); C2 = np.array(C2); N2 = np.array(N2)
    D2 = np.array(D2); TE2 = np.array(TE2); TR2 = np.array(TR2); EP2 = np.array(EP2)
    per_ep = np.array([R2[EP2 == k].sum() for k in range(COLLECT_EP)])
    qual = dict(steps=int(len(R2)), reward_per_step_mean=float(R2.mean()), reward_per_step_std=float(R2.std()),
                per_episode_return_mean=float(per_ep.mean()), per_episode_return_std=float(per_ep.std(ddof=1)),
                per_episode_return=[float(x) for x in per_ep], cost_event_rate=float((C2 > 0).mean()),
                action_saturation=float((np.abs(A2) > 0.99).mean()),
                terminated_count=int(TE2.sum()), truncated_count=int(TR2.sum()),
                alignment_ok=bool(all(np.array_equal(N2[np.where(EP2 == k)[0][:-1]], S2[np.where(EP2 == k)[0][1:]])
                                      for k in range(COLLECT_EP))))
    np.savez_compressed(out / '数据_SwimmerVelocity_行为策略20回合.npz',
                        obs=S2, act=A2, rew=R2, cost=C2, next_obs=N2, done=D2,
                        terminated=TE2, truncated=TR2, episode=EP2)
    print('新数据质量：', {k: (round(v, 4) if isinstance(v, float) else v) for k, v in qual.items()
                        if k != 'per_episode_return'}, flush=True)
    # ---------- 同参照验收 ----------
    ref = EP2 < REF_TRAJ
    om = torch.from_numpy(S2[ref].mean(0, keepdims=True)); os_ = torch.from_numpy(np.maximum(S2[ref].std(0, keepdims=True), 1e-8))
    St = torch.from_numpy(S2); At = torch.from_numpy(A2); Rt = torch.from_numpy(R2).reshape(-1, 1)
    St2 = torch.from_numpy(N2); Dt = torch.from_numpy(D2.astype(np.float32)).reshape(-1, 1)
    ref_idx = np.where(ref)[0]; ho_idx = np.where(~ref)[0]
    mc = np.zeros(len(R2), dtype=np.float32)
    for k in range(COLLECT_EP):
        m_ = np.where(EP2 == k)[0]; g = 0.0
        for i in range(len(m_) - 1, -1, -1):
            g = R2[m_[i]] + GAMMA * g; mc[m_[i]] = g
    mc_const = float(np.sqrt(np.mean((mc[ho_idx] - mc[ref_idx].mean()) ** 2)))
    q = type('Q', (nn.Module,), {})( )
    q.q1 = mlp(no + na, 1, 256); q.q2 = mlp(no + na, 1, 256); q.vr = mlp(no, 1, 256)
    q.register_buffer('obs_mean', om); q.register_buffer('obs_std', os_)
    norm = lambda s: (s - q.obs_mean) / q.obs_std
    qt_ = type('Q', (nn.Module,), {})(); qt_.q1 = mlp(no + na, 1, 256); qt_.q2 = mlp(no + na, 1, 256); qt_.vr = mlp(no, 1, 256)
    qt_.register_buffer('obs_mean', om); qt_.register_buffer('obs_std', os_)
    qt_.load_state_dict(q.state_dict())
    oq = torch.optim.Adam(list(q.q1.parameters()) + list(q.q2.parameters()), lr=LR_QV)
    ov = torch.optim.Adam(q.vr.parameters(), lr=LR_QV)
    rngb = np.random.RandomState(0)
    批次 = [rngb.choice(ref_idx, size=BATCH_QV) for _ in range(MILESTONES[-1])]
    stages = []

    def diag(step):
        with torch.no_grad():
            q1 = q.q1(torch.cat([norm(St), At], -1)); v = q.vr(norm(St))
        q1n = q1.numpy().reshape(-1)
        rmse = float(np.sqrt(np.mean((q1n[ho_idx] - mc[ho_idx]) ** 2)))
        return dict(step=step, q_absmax=float(np.abs(q1n).max()),
                    mc_Q_rmse=rmse, mc_const_rmse=mc_const, 主判据比值=float(rmse / mc_const),
                    corr_Q_vs_MC=float(np.corrcoef(q1n[ho_idx], mc[ho_idx])[0, 1]))

    stages.append(diag(0)); step = 0
    for target in MILESTONES[1:]:
        while step < target:
            j = torch.from_numpy(批次[step])
            with torch.no_grad():
                tv = Rt[j] + GAMMA * (1 - Dt[j]) * qt_.vr(norm(St2[j]))
                tq = torch.minimum(qt_.q1(torch.cat([norm(St[j]), At[j]], -1)),
                                   qt_.q2(torch.cat([norm(St[j]), At[j]], -1)))
            l = (q.q1(torch.cat([norm(St[j]), At[j]], -1)) - tv).square().mean() \
                + (q.q2(torch.cat([norm(St[j]), At[j]], -1)) - tv).square().mean() \
                + ((torch.where((tq - q.vr(norm(St[j]))) > 0, TAU_V, 1 - TAU_V)
                    * (tq - q.vr(norm(St[j]))).square()).mean())
            oq.zero_grad(); ov.zero_grad(); l.backward()
            for n_ in (q.q1, q.q2, q.vr):
                nn.utils.clip_grad_norm_(n_.parameters(), CLIP_QV)
            oq.step(); ov.step()
            with torch.no_grad():
                for a_, b_ in ((q.q1, qt_.q1), (q.q2, qt_.q2), (q.vr, qt_.vr)):
                    for pa, pb in zip(a_.parameters(), b_.parameters()):
                        pb.mul_(1 - EMA_QV).add_(pa, alpha=EMA_QV)
            step += 1
        stages.append(diag(step))
        torch.save({'step': step, 'state_dict': q.state_dict()}, out / f'模型_E4_3_QV_{step}步.pt')
        print(f'  QV stage {step} 完成', flush=True)
    res = dict(env=EID, seed=SEED, policy='PPO reward-only, 无任何风险信号',
               env_steps_used=int(total), train_minutes=round((time.time() - t0) / 60, 1),
               caps=dict(max_steps=MAX_STEPS, max_minutes=MAX_MINUTES),
               train_history=hist[-5:], quality=qual,
               mc_const_rmse_reference=mc_const, primary_criterion=f'mc_Q_rmse ÷ mc_const_rmse ≤ {阈值}',
               qv_stages=stages, passed=bool(stages[-1]['主判据比值'] <= 阈值),
               elapsed_sec=round(time.time() - t0, 1))
    with (out / '核验与边界.json').open('x', encoding='utf8') as f:
        json.dump(res, f, ensure_ascii=False, indent=2, default=float)
    print(json.dumps({k: v for k, v in res.items() if k not in ('train_history', 'quality')},
                     ensure_ascii=False, indent=1, default=float))
    print('\n主判据达标:', res['passed'], '| 输出:', out)


if __name__ == '__main__':
    main()
