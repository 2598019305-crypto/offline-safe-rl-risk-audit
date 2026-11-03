#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""E4-4：演员与真实闭环安全验证（本机实现版）。

7 个分支（共享演员起点 + 共享批次序列 + 冻结 Q/V，只做 actor-only 更新）：
  A_BC              均匀权重行为克隆（无优势、无风险）
  B_reward_only     λ=0，仅优势加权
  C_D10_lam1        λ=1，风险=D10（留一种子外的严格留出评分）
  D_D10_lam10       λ=10，风险=D10
  E_E100            λ=10，风险=E100（同几何邻域均值）
  F_D10_emptyfix    λ=10，风险=D10，但**空邻域样本赋最坏值 1.0**（修 F3-Closure 边界 a）
  G_speed_control   λ=10，风险=|obs[3]|（速度分量，F3-Closure 发现的强基线）

评价：开发集环境种子 410–417，确定性均值动作，1000 步闭环；终点 = 逐回合奖励、
原生累计成本、相对共享起点的动作变化。不碰 125000–125099。
"""
from __future__ import annotations

import copy
import csv
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

AW, TAIL, KQ = 20.0, 0.10, 500
GAMMA, LR, EMA, CLIP, BATCH, STEPS = 0.99, 3e-4, 0.005, 10.0, 256, 2000
DEV_SEEDS = tuple(range(410, 418))
SEEDS8 = tuple(range(400, 408))
EID = 'SafetySwimmerVelocity-v1'


def mlp(i, o, h=64):
    return nn.Sequential(nn.Linear(i, h), nn.Tanh(), nn.Linear(h, h), nn.Tanh(), nn.Linear(h, o))


class AC(nn.Module):
    def __init__(self, no, na):
        super().__init__()
        self.pi = mlp(no, na); self.v = mlp(no, 1)
        self.log_std = nn.Parameter(torch.zeros(na))


def z(x):
    return (x - x.mean()) / (x.std() + 1e-6)


def main():
    import safety_gymnasium
    project = Path(__file__).resolve().parent
    src = project / '论文核验输出_E4_3b_多模态pilot_20261013' / '数据_SwimmerVelocity_多模态32条.npz'
    pol = project / '论文核验输出_E4_3_行为策略训练与验收_20261013' / '模型_E4_3_行为策略.pt'
    qvp = project / '论文核验输出_E4_3_行为策略训练与验收_20261013' / '模型_E4_3_QV_2000步.pt'
    for p in (src, pol, qvp):
        if not p.is_file():
            raise FileNotFoundError('缺输入: ' + str(p))
    h = hashlib.sha256(src.read_bytes()).hexdigest()
    out = project / '论文核验输出_E4_4_演员闭环验证_20261013'
    if out.exists():
        raise FileExistsError('输出目录已存在: ' + str(out))
    out.mkdir(parents=True)
    t0 = time.time()
    zz = np.load(src)
    S, A, C, EP = zz['obs'], zz['act'], zz['cost'].astype(np.float64), zz['episode']
    no, na = S.shape[1], A.shape[1]
    ESEED = 400 + (EP % 8)
    # 留一种子外的 D10 / E100 + 空邻域标记
    from scipy.spatial import cKDTree
    D10 = np.zeros(len(S)); E100 = np.zeros(len(S)); EMPTY = np.zeros(len(S), bool)
    for sd in SEEDS8:
        te = ESEED == sd; tr = ~te
        om, os_ = S[tr].mean(0), np.maximum(S[tr].std(0), 1e-8)
        am, asd = A[tr].mean(0), np.maximum(A[tr].std(0), 1e-8)
        F = lambda o_, a_: np.concatenate([(o_ - om) / os_, AW * (a_ - am) / asd], -1)
        Fr, Fq = F(S[tr], A[tr]), F(S[te], A[te])
        tree = cKDTree(Fr); k = min(KQ, int(tr.sum()) - 1)
        dsf, _ = tree.query(Fr, k=k + 1, workers=-1); rad = float(np.median(dsf[:, -1]))
        nb = tree.query_ball_point(Fq, rad, workers=-1)
        d10 = np.zeros(len(nb)); e100 = np.zeros(len(nb)); emp = np.zeros(len(nb), bool); rc = C[tr]
        for i, ids in enumerate(nb):
            if not ids:
                emp[i] = True; continue
            v = rc[np.asarray(ids, np.intp)]; e100[i] = v.mean()
            kk = max(1, int(math.ceil(TAIL * len(v)))); d10[i] = np.sort(v)[-kk:].mean()
        idx = np.where(te)[0]
        D10[idx] = d10; E100[idx] = e100; EMPTY[idx] = emp
    D10_fix = np.where(EMPTY, 1.0, D10)
    speed = np.abs(S[:, 3]).astype(np.float64)
    res = dict(data_sha256=h, geometry='G_impl = (z(s), 20·z(a))', n=int(len(S)),
               empty_n=int(EMPTY.sum()), steps=STEPS, batch=BATCH, dev_seeds=list(DEV_SEEDS))
    # 冻结 Q/V → 优势
    ck = torch.load(qvp, map_location='cpu', weights_only=True)['state_dict']
    st = lambda p: {k[len(p):]: v for k, v in ck.items() if k.startswith(p)}
    q1 = mlp(no + na, 1, 256); q2 = mlp(no + na, 1, 256); vr = mlp(no, 1, 256)
    q1.load_state_dict(st('q1.')); q2.load_state_dict(st('q2.')); vr.load_state_dict(st('vr.'))
    nm = lambda s: (s - ck['obs_mean']) / ck['obs_std']
    St = torch.from_numpy(S.astype(np.float32)); At = torch.from_numpy(A.astype(np.float32))
    with torch.no_grad():
        Q = torch.minimum(q1(torch.cat([nm(St), At], -1)), q2(torch.cat([nm(St), At], -1))).numpy().reshape(-1)
        V = vr(nm(St)).numpy().reshape(-1)
    adv = Q - V
    # 共享批次序列
    rng = np.random.RandomState(20261017)
    批次 = [rng.choice(len(adv), BATCH, replace=False) for _ in range(STEPS)]
    # 共享演员起点
    base = AC(no, na); base.load_state_dict(torch.load(pol, map_location='cpu', weights_only=True)['state_dict'])
    起点 = copy.deepcopy(base.state_dict())
    分支 = [('A_BC', 'bc', 0.0), ('B_reward_only', 'adv', 0.0), ('C_D10_lam1', 'd10', 1.0),
            ('D_D10_lam10', 'd10', 10.0), ('E_E100', 'e100', 10.0),
            ('F_D10_emptyfix', 'd10fix', 10.0), ('G_speed_control', 'speed', 10.0)]
    env = safety_gymnasium.make(EID)
    summary, per_ep = [], []
    for name, kind, lam in 分支:
        m = AC(no, na); m.load_state_dict(起点)
        opt = torch.optim.Adam(m.pi.parameters(), lr=LR)
        for stp in range(STEPS):
            j = torch.from_numpy(批次[stp])
            a_ = adv[j]
            if kind == 'bc':
                w = torch.ones(len(j))
            else:
                risk = {'adv': np.zeros(len(adv)), 'd10': D10, 'e100': E100,
                        'd10fix': D10_fix, 'speed': speed}[kind][j]
                u = z(z(a_) - lam * z(risk)); lw = 3.0 * u
                e = np.exp(lw - lw.max()); w = torch.from_numpy(np.clip(e / e.mean(), 0, 100.0)).float()
            mu = m.pi(St[j]); std = m.log_std.exp()
            logp = (-0.5 * ((At[j] - mu) / std).square() - torch.log(std) - 0.9189).sum(-1)
            loss = -(w * logp).mean()
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(m.pi.parameters(), CLIP); opt.step()
        m.eval()
        # 闭环回放（确定性均值动作）
        rets, costs, dacts = [], [], []
        for sd in DEV_SEEDS:
            ob, _ = env.reset(seed=sd); ob = np.asarray(ob, dtype=np.float32)
            R_, C_ = 0.0, 0.0; diffs = []
            for t in range(1000):
                with torch.no_grad():
                    a_new = m.pi(torch.from_numpy(ob).unsqueeze(0)).squeeze(0).numpy()
                    a_old = base.pi(torch.from_numpy(ob).unsqueeze(0)).squeeze(0).numpy()
                diffs.append(float(np.linalg.norm(np.clip(a_new, -1, 1) - np.clip(a_old, -1, 1))))
                o2, r, c, te, tr, _ = env.step(np.clip(a_new, -1, 1))
                R_ += float(r); C_ += float(c); ob = np.asarray(o2, dtype=np.float32)
                if te or tr:
                    break
            rets.append(R_); costs.append(C_); dacts.append(float(np.mean(diffs)) if diffs else 0.0)
            per_ep.append(dict(branch=name, env_seed=sd, return_sum=R_, cost_sum=C_, action_diff_mean=float(np.mean(diffs))))
        summary.append(dict(branch=name, kind=kind, lam=lam,
                            return_mean=float(np.mean(rets)), return_sd=float(np.std(rets, ddof=1)),
                            cost_mean=float(np.mean(costs)), cost_sd=float(np.std(costs, ddof=1)),
                            action_diff_mean=float(np.mean(dacts)),
                            returns=[float(x) for x in rets], costs=[float(x) for x in costs],
                            action_diffs=[float(x) for x in dacts]))
        torch.save({'branch': name, 'state_dict': m.state_dict()}, out / f'模型_E4_4_{name}.pt')
        print(f'{name}: 回报 {np.mean(rets):.2f}±{np.std(rets, ddof=1):.2f} | 成本 {np.mean(costs):.2f}±{np.std(costs, ddof=1):.2f} | 动作变化 {np.mean(dacts):.4f}', flush=True)
    env.close()
    with (out / '逐回合_闭环.csv').open('x', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(per_ep[0].keys())); w.writeheader(); w.writerows(per_ep)
    # 相对 reward-only 的配对差值
    def get(nm_, k):
        return {d['env_seed']: d[k] for d in per_ep if d['branch'] == nm_}
    base_r, base_c = get('B_reward_only', 'return_sum'), get('B_reward_only', 'cost_sum')
    paired = {}
    for name, _, _ in 分支:
        if name == 'B_reward_only':
            continue
        r_, c_ = get(name, 'return_sum'), get(name, 'cost_sum')
        dr = [r_[s] - base_r[s] for s in DEV_SEEDS]; dc = [c_[s] - base_c[s] for s in DEV_SEEDS]
        paired[name] = dict(d_return_mean=float(np.mean(dr)), d_return_sd=float(np.std(dr, ddof=1)),
                            d_cost_mean=float(np.mean(dc)), d_cost_sd=float(np.std(dc, ddof=1)),
                            d_return_all_same_sign=bool(len(set(np.sign(dr))) == 1),
                            d_cost_all_same_sign=bool(len(set(np.sign(dc))) == 1),
                            d_return=[float(x) for x in dr], d_cost=[float(x) for x in dc])
    res['summary'] = summary; res['paired_vs_reward_only'] = paired
    res['elapsed_sec'] = round(time.time() - t0, 1)
    with (out / '核验与边界.json').open('x', encoding='utf8') as f:
        json.dump(res, f, ensure_ascii=False, indent=2, default=float)
    print('\n配对（相对 reward-only）：')
    print(json.dumps(paired, ensure_ascii=False, indent=1, default=float))
    print('输出:', out, '用时', res['elapsed_sec'], 's')


if __name__ == '__main__':
    main()
