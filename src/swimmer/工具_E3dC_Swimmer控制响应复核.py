#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""E3d-C：Swimmer 的局部控制响应复核（同一初始状态、正负单轴、短时域）。

修正上一轮的方法缺陷：上一轮比的是"各轴连续执行 100 步后的观测位移"，
那会把动力学耦合与长时漂移算进来，不能代表局部控制响应。
本轮改为：同一初始状态 reset(seed) 固定，分别施加 +e_j 与 −e_j，取中心差分：
    R_j(N) = ( obs(+e_j, N 步) − obs(−e_j, N 步) ) / 2
再比较 R_0 与 R_1 的余弦、以及 [R_0 R_1] 的奇异值，判断两轴是否可分辨。
同时补齐上一轮漏掉的 corr(cost, ||a||^2)。
"""
from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np

EID = 'SafetySwimmerVelocity-v1'
HORIZONS = (1, 5, 20)
ROLL_STEPS = 3000


def 状态(env):
    """优先用 qpos（若拿得到），否则退回观测。"""
    for path in (('unwrapped', 'task', 'engine'), ('unwrapped', 'engine'),
                 ('unwrapped', 'model')):
        obj = env
        ok = True
        for a in path:
            obj = getattr(obj, a, None)
            if obj is None:
                ok = False
                break
        if not ok:
            continue
        if hasattr(obj, 'data') and hasattr(obj, 'model'):
            return 'qpos', np.asarray(obj.data.qpos, dtype=np.float64).copy()
    return 'obs', None


def 响应(env, na, N, seed=0):
    """返回每个轴的中心差分响应向量（在观测空间里量，任何任务都可比）。"""
    列 = []
    for j in range(na):
        outs = {}
        for sgn in (+1.0, -1.0):
            o0, _ = env.reset(seed=seed)
            a = np.zeros(na)
            a[j] = sgn
            oo = np.asarray(o0, dtype=np.float64).copy()
            for _ in range(N):
                oo, r, c, te, tr, _ = env.step(a)
                if te or tr:
                    break
            outs[sgn] = np.asarray(oo, dtype=np.float64).copy()
        列.append((outs[+1.0] - outs[-1.0]) / 2.0)
    return np.array(列)  # (na, obs_dim)


def main():
    import safety_gymnasium
    project = Path(__file__).resolve().parent
    out = project / '论文核验输出_E3dC_Swimmer控制响应复核_20261013'
    if out.exists():
        raise FileExistsError('输出目录已存在: ' + str(out))
    out.mkdir(parents=True)
    t0 = time.time()
    env = safety_gymnasium.make(EID)
    o, _ = env.reset(seed=0)
    na = int(env.action_space.shape[0])
    res = dict(env=EID, obs_dim=int(np.asarray(o).shape[0]), act_dim=na,
               method='同一初始状态 + 正负单轴中心差分，观测空间响应',
               horizons={})
    for N in HORIZONS:
        R = 响应(env, na, N)
        n = R / (np.linalg.norm(R, axis=1, keepdims=True) + 1e-12)
        cos = float(abs(n[0] @ n[1])) if na >= 2 else None
        sv = np.linalg.svd(R, compute_uv=False)
        res['horizons'][f'N{N}'] = dict(
            response_norms=[float(x) for x in np.linalg.norm(R, axis=1)],
            cosine_between_axes=cos,
            singular_values=[float(x) for x in sv],
            sv_ratio=float(sv[-1] / sv[0]) if sv[0] > 0 else None)
    # 成本与动作范数的相关（上一轮漏掉的检查，这里补齐）
    rng = np.random.RandomState(20261013)
    env.reset(seed=11)
    CW, SQ, RW = [], [], []
    for t in range(ROLL_STEPS):
        a = rng.uniform(-1, 1, size=na)
        oo, r, c, te, tr, _ = env.step(a)
        CW.append(float(c)); SQ.append(float(np.sum(a ** 2))); RW.append(float(r))
        if te or tr:
            env.reset(seed=11 + t + 1)
    CW = np.array(CW); SQ = np.array(SQ); RW = np.array(RW)
    res['random_rollout'] = dict(steps=int(len(CW)),
                                 cost_event_rate=float((CW > 0).mean()),
                                 cost_mean=float(CW.mean()),
                                 reward_std=float(RW.std()),
                                 corr_cost_actnorm2=(float(np.corrcoef(SQ, CW)[0, 1])
                                                     if (SQ.std() > 0 and CW.std() > 0) else None))
    env.close()
    # 判定：任一短时域上两轴可分辨（余弦 < 0.9）或响应矩阵两维奇异值比 > 0.2
    ok = False
    依据 = []
    for N in HORIZONS:
        h = res['horizons'][f'N{N}']
        if h['cosine_between_axes'] is not None and h['cosine_between_axes'] < 0.9:
            ok = True; 依据.append(f'N{N} 轴间余弦 {h["cosine_between_axes"]:.3f} < 0.9')
        if h['sv_ratio'] is not None and h['sv_ratio'] > 0.2:
            ok = True; 依据.append(f'N{N} 奇异值比 {h["sv_ratio"]:.3f} > 0.2')
    res['axes_distinguishable'] = bool(ok)
    res['verdict_reason'] = 依据 if 依据 else ['三个时域上两轴都高度共线且奇异值比很小']
    res['elapsed_sec'] = round(time.time() - t0, 1)
    with (out / '验收结果.json').open('x', encoding='utf8') as f:
        json.dump(res, f, ensure_ascii=False, indent=2, default=float)
    print(json.dumps(res, ensure_ascii=False, indent=1, default=float))
    print('\n结论: 两轴可分辨 =', res['axes_distinguishable'], '|', res['verdict_reason'])


if __name__ == '__main__':
    main()
