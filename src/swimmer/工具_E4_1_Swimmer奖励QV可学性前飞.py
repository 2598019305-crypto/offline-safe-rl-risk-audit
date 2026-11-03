#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""E4-1：SafetySwimmerVelocity-v1 单种子奖励 Q/V 可学性前飞（只用种子 0 的一条奖励链）。

严格按任务单：数据只用 E3e 自采 pilot（SHA 硬校验），前 14 条轨迹拟合、后 6 条只做离线诊断；
标准化量只在前 14 条估计；不引入任何安全风险代理、不训练演员；≤20,000 次更新，里程碑 0/5k/10k/20k。
奖励支路沿用已验证的 P2/P3 口径：双 Q MSE TD + 目标网络 EMA、V 用 min(目标双 Q) 的 expectile τ=0.7、
分离 Adam、输入标准化、梯度裁剪。
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

DATA_SHA = '4edffe8b0178c1220a3d158521cbeafccd2ca479dac07fa8e38430d6086202fa'
GAMMA = 0.99
TAU_V = 0.7
LR = 3e-4
EMA = 0.005
CLIP = 10.0
BATCH = 256
REF_TRAJ = 14
MILESTONES = (0, 5000, 10000, 20000)


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


def expectile(diff, tau=TAU_V):
    return (torch.where(diff > 0, tau, 1 - tau) * diff.square()).mean()


def main():
    project = Path(__file__).resolve().parent
    npz = project / '论文核验输出_E3e_Swimmer离线数据pilot_20261013' / '数据_SwimmerVelocity_自采pilot.npz'
    h = hashlib.sha256(npz.read_bytes()).hexdigest()
    assert h == DATA_SHA, '数据 SHA 不符，停止'
    out = project / '论文核验输出_E4_1_Swimmer奖励QV前飞_20261013'
    if out.exists():
        raise FileExistsError('输出目录已存在: ' + str(out))
    out.mkdir(parents=True)
    t0 = time.time()
    z = np.load(npz)
    obs = z['obs'].astype(np.float32); act = z['act'].astype(np.float32)
    rew = z['rew'].astype(np.float32); nxt = z['next_obs'].astype(np.float32)
    done = z['done'].astype(bool); ep = z['episode']
    no, na = obs.shape[1], act.shape[1]
    ref = ep < REF_TRAJ
    res = dict(data_sha256=h, n_obs=int(no), n_act=int(na), n_steps=int(len(obs)),
               n_ref_traj=REF_TRAJ, n_holdout_traj=int(ep.max() + 1 - REF_TRAJ),
               n_ref=int(ref.sum()), n_holdout=int((~ref).sum()),
               reward_mean=float(rew.mean()), reward_std=float(rew.std()),
               hyper=dict(gamma=GAMMA, tau_v=TAU_V, lr=LR, ema=EMA, clip=CLIP, batch=BATCH,
                          milestones=list(MILESTONES), seed=0, arch='256-256 ReLU'),
               alignment=dict(reward_finite=bool(np.isfinite(rew).all()),
                              next_obs_finite=bool(np.isfinite(nxt).all()),
                              done_at_episode_end=bool(all(done[np.where(ep == k)[0][-1]] for k in range(int(ep.max() + 1))))))
    torch.manual_seed(0)
    m = QV(no, na); mt = QV(no, na); mt.load_state_dict(m.state_dict())
    m.obs_mean.copy_(torch.from_numpy(obs[ref].mean(0, keepdims=True)))
    m.obs_std.copy_(torch.from_numpy(np.maximum(obs[ref].std(0, keepdims=True), 1e-8)))
    mt.obs_mean.copy_(m.obs_mean); mt.obs_std.copy_(m.obs_std)
    oq = torch.optim.Adam(list(m.q1.parameters()) + list(m.q2.parameters()), lr=LR)
    ov = torch.optim.Adam(m.vr.parameters(), lr=LR)
    S = torch.from_numpy(obs); A = torch.from_numpy(act); R = torch.from_numpy(rew).reshape(-1, 1)
    S2 = torch.from_numpy(nxt); D = torch.from_numpy(done.astype(np.float32)).reshape(-1, 1)
    rng = np.random.RandomState(0)
    ref_idx = np.where(ref)[0]; ho_idx = np.where(~ref)[0]
    # 后 6 条轨迹的蒙特卡洛回报（只用于诊断，不参与训练）
    mc = np.zeros(len(obs), dtype=np.float32)
    for k in range(int(ep.max() + 1)):
        m_ = np.where(ep == k)[0]
        g = 0.0
        for i in range(len(m_) - 1, -1, -1):
            g = rew[m_[i]] + GAMMA * g
            mc[m_[i]] = g

    def diag(stage):
        m.eval()
        with torch.no_grad():
            sn = m.normalized(S); sn2 = m.normalized(S2)
            q1 = m.q1(torch.cat([sn, A], -1)); q2 = m.q2(torch.cat([sn, A], -1))
            v = m.vr(sn)
            qt = torch.minimum(mt.q1(torch.cat([mt.normalized(S), A], -1)),
                               mt.q2(torch.cat([mt.normalized(S), A], -1)))
            td = (R + GAMMA * (1 - D) * mt.vr(mt.normalized(S2))) - qt
            adv = torch.minimum(q1, q2) - v
        q1n, q2n, vn, tdn, advn = [x.numpy().reshape(-1) for x in (q1, q2, v, td, adv)]
        ho = ho_idx; rr = ref_idx
        d = dict(stage=stage,
                 q_min=float(q1n.min()), q_max=float(q1n.max()), q_absmax=float(np.abs(q1n).max()),
                 q_p50=float(np.median(q1n)), q_p05=float(np.percentile(q1n, 5)), q_p95=float(np.percentile(q1n, 95)),
                 v_p50=float(np.median(vn)), v_absmax=float(np.abs(vn).max()),
                 finite=bool(np.isfinite(q1n).all() and np.isfinite(vn).all() and np.isfinite(tdn).all()),
                 td_rmse_train=float(np.sqrt(np.mean(tdn[rr] ** 2))),
                 td_rmse_holdout=float(np.sqrt(np.mean(tdn[ho] ** 2))),
                 adv_holdout_mean=float(advn[ho].mean()), adv_holdout_std=float(advn[ho].std()),
                 adv_holdout_p05=float(np.percentile(advn[ho], 5)),
                 adv_holdout_p95=float(np.percentile(advn[ho], 95)))
        # 与"常数/零预测器"的对照：预测奖励均值 与 预测 0
        base_mean = float(rew[rr].mean())
        d['reward_const_baseline_rmse'] = float(np.sqrt(np.mean((rew[ho] - base_mean) ** 2)))
        d['reward_zero_baseline_rmse'] = float(np.sqrt(np.mean(rew[ho] ** 2)))
        d['td_rmse_holdout_vs_best_const_ratio'] = float(d['td_rmse_holdout'] / max(d['reward_const_baseline_rmse'], 1e-9))
        # 可学性信号：优势 vs 蒙特卡洛回报（留出轨迹）
        if advn[ho].std() > 0 and mc[ho].std() > 0:
            d['corr_adv_vs_MC_return_holdout'] = float(np.corrcoef(advn[ho], mc[ho])[0, 1])
            d['corr_Q_vs_MC_return_holdout'] = float(np.corrcoef(q1n[ho], mc[ho])[0, 1])
        # 状态群覆盖：按 obs[3]（速度分量）四分位分组后在留出轨迹上的优势分布
        qs = np.quantile(obs[ho][:, 3], [0.25, 0.5, 0.75])
        grp = np.digitize(obs[ho][:, 3], qs)
        d['adv_by_state_quartile_holdout'] = [dict(q=int(i), n=int((grp == i).sum()),
                                                   adv_mean=float(advn[ho][grp == i].mean()))
                                              for i in range(4)]
        return d

    stages = []
    for target in MILESTONES:
        while True:
            if 'step' not in dir() or True:
                pass
            break
        break
    step = 0
    stages.append(diag(0))
    print('stage 0 完成', flush=True)
    for target in MILESTONES[1:]:
        m.train()
        while step < target:
            j = torch.from_numpy(rng.choice(ref_idx, size=BATCH))
            with torch.no_grad():
                qt = torch.minimum(mt.q1(torch.cat([mt.normalized(S2[j]), A[j]], -1)),
                                   mt.q2(torch.cat([mt.normalized(S2[j]), A[j]], -1)))
                tv = R[j] + GAMMA * (1 - D[j]) * qt
                vt = mt.vr(mt.normalized(S2[j]))
            q1 = m.q1(torch.cat([m.normalized(S[j]), A[j]], -1))
            q2 = m.q2(torch.cat([m.normalized(S[j]), A[j]], -1))
            qloss = (q1 - tv).square().mean() + (q2 - tv).square().mean()
            vloss = expectile(qt - m.vr(m.normalized(S[j])))
            loss = qloss + vloss
            if not torch.isfinite(loss):
                raise FloatingPointError('非有限损失 step=%d' % step)
            oq.zero_grad(); ov.zero_grad(); loss.backward()
            for n_ in (m.q1, m.q2, m.vr):
                nn.utils.clip_grad_norm_(n_.parameters(), CLIP)
            oq.step(); ov.step()
            with torch.no_grad():
                for a_, b_ in ((m.q1, mt.q1), (m.q2, mt.q2)):
                    for pa, pb in zip(a_.parameters(), b_.parameters()):
                        pb.mul_(1 - EMA).add_(pa, alpha=EMA)
            step += 1
        stages.append(diag(target))
        torch.save({'step': step, 'state_dict': m.state_dict(), 'obs_mean': m.obs_mean,
                    'obs_std': m.obs_std, 'data_sha256': DATA_SHA}, out / f'模型_E4_1_种子0_{step}步.pt')
        print(f'stage {target} 完成', flush=True)
    res['stages'] = stages
    res['elapsed_sec'] = round(time.time() - t0, 1)
    with (out / '核验与边界.json').open('x', encoding='utf8') as f:
        json.dump(res, f, ensure_ascii=False, indent=2, default=float)
    for s in stages:
        print({k: (round(v, 5) if isinstance(v, float) else v) for k, v in s.items()
               if k not in ('adv_by_state_quartile_holdout',)})
    print('\n输出:', out, '用时', res['elapsed_sec'], 's')


if __name__ == '__main__':
    main()
