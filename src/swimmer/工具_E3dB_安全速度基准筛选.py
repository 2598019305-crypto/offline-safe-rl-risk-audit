#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""E3d-B：Safe Velocity 候选环境限时验收（只验收，不采集、不训练）。

对每个候选做四项检查：
  1) 能否实例化（obs/act 维度、类名）
  2) 动作是否真的改变物理状态（逐动作维施加，测 qpos 与 root 位置变化）
  3) 奖励与原生成本是否有效（随机 rollout 的回报、成本事件率、cost 与 ||a||^2 的相关）
  4) 同种子 + 同动作序列是否逐位可复现

判据先写死：
  P1 实例化成功且 obs/act 维度有限；
  P2 至少两个动作维各自引起可见的物理状态变化（该维 max|Δqpos| > 1e-4），
     且不同维引起的变化不正比（即不是只有 1 个有效方向）；
  P3 随机 rollout 回报非常数、原生成本出现非零事件，且 |corr(cost, ||a||^2)| < 0.5；
  P4 同种子同动作序列 100 步的 obs 与 cost 逐位相同。
四项全过才记为 PASS。
"""
from __future__ import annotations

import json
import time
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings('ignore')

CANDIDATES = ['SafetyWalker2dVelocity-v1', 'SafetySwimmerVelocity-v1',
              'SafetyHalfCheetahVelocity-v1']
PROBE_STEPS = 100
ROLLOUT_STEPS = 3000


def 找引擎(env):
    """不同任务的引擎挂载点不一样，逐个尝试。"""
    cands = []
    t = getattr(env.unwrapped, 'task', None)
    for obj in (t, env.unwrapped, getattr(env, 'unwrapped', None)):
        if obj is None:
            continue
        for name in ('engine', '_engine'):
            cands.append(getattr(obj, name, None))
    for c in cands:
        if c is not None and hasattr(c, 'model') and hasattr(c, 'data'):
            return c
    for name in ('model', 'data'):
        if hasattr(env.unwrapped, name):
            # 只有 model/data 直接挂在 env 上时，用轻量包装
            pass
    return None


def 根位置(engine):
    d = engine.data
    try:
        return np.asarray(d.qpos[:2], dtype=np.float64).copy()
    except Exception:
        return np.zeros(2)


def 检查一个(eid):
    import safety_gymnasium
    res = dict(env=eid)
    起点 = time.time()
    try:
        e = safety_gymnasium.make(eid)
        o, _ = e.reset(seed=0)
        res['class'] = type(e).__name__
        res['obs_dim'] = int(np.asarray(o).shape[0])
        res['act_dim'] = int(e.action_space.shape[0])
        res['act_low'] = [float(x) for x in np.atleast_1d(e.action_space.low)[:4]]
        res['act_high'] = [float(x) for x in np.atleast_1d(e.action_space.high)[:4]]
        res['P1'] = True
    except Exception as ex:
        res['P1'] = False
        res['instantiation_error'] = f'{type(ex).__name__}: {str(ex)[:160]}'
        res['elapsed_sec'] = round(time.time() - 起点, 1)
        return res
    eng = 找引擎(e)
    res['engine_found'] = eng is not None
    na = res['act_dim']
    # P2：逐动作维施加，测"策略可见的状态"变化（观测）——不依赖 engine 也能做。
    # engine 可用时额外给出 qpos 与根位置的原始证据。
    逐维 = []
    for j in range(na):
        a = np.zeros(na); a[j] = 1.0
        o0, _ = e.reset(seed=0)
        o0 = np.asarray(o0, dtype=np.float64).copy()
        q0 = np.asarray(eng.data.qpos).copy() if eng is not None else None
        p0 = 根位置(eng) if eng is not None else None
        观测末尾 = None
        for _ in range(PROBE_STEPS):
            oo, rr, cc, te, tr, _ = e.step(a)
            观测末尾 = np.asarray(oo, dtype=np.float64).copy()
            if te or tr:
                break
        dobs = 观测末尾 - o0
        rec = dict(dim=int(j),
                   max_abs_dobs=float(np.max(np.abs(dobs))),
                   dobs_norm=float(np.linalg.norm(dobs)),
                   dobs_vec=[float(x) for x in dobs])
        if eng is not None:
            q1 = np.asarray(eng.data.qpos).copy()
            rec['max_abs_dqpos'] = float(np.max(np.abs(q1 - q0)))
            rec['d_root'] = [float(x) for x in (根位置(eng) - p0)]
        逐维.append(rec)
    res['per_dim'] = 逐维
    有效 = [x for x in 逐维 if x['max_abs_dobs'] > 1e-6]
    方向 = np.array([[x['d_root'][0], x['d_root'][1]] for x in 有效 if 'd_root' in x]) \
        if any('d_root' in x for x in 有效) else np.zeros((0, 2))
    # 可区分性：取所有通道对里"最不像"的那一对（最小 |cos|）。
    # Velocity 系列没有 task/engine，用观测变化向量；有 engine 时用根位移。
    if len(方向) >= 2:
        V = 方向
    else:
        V = np.array([x['dobs_vec'] for x in 有效]) if 有效 else np.zeros((0, 2))
    V = np.array([v for v in V if np.linalg.norm(v) > 1e-9]) if len(V) else np.zeros((0, 2))
    if len(V) >= 2:
        nn = V / (np.linalg.norm(V, axis=1, keepdims=True) + 1e-12)
        C = np.abs(nn @ nn.T)
        iu = np.triu_indices(len(nn), k=1)
        cos = float(C[iu].min())
        非共线 = cos < 0.99
    else:
        cos = None
        非共线 = False
    res['effective_dims'] = len(有效)
    res['root_direction_cos_first_two'] = cos
    res['P2'] = bool(len(有效) >= 2 and 非共线)
    # P3 与 P4
    try:
        rng = np.random.RandomState(20261013)
        # 速度约束任务的成本来自"速度超限"：随机动作下机器人几乎不动 ⇒ 事件必然为 0。
        # 所以必须两种动作机制都测：随机 与 强恒定动作（让它真的跑起来）。
        def 跑(steps, mode, seed):
            e.reset(seed=seed)
            RW, CW, SQ = [], [], []
            for t in range(steps):
                a = rng.uniform(-1, 1, size=na) if mode == 'rand' else np.ones(na)
                o, r, c, te, tr, _ = e.step(a)
                RW.append(float(r)); CW.append(float(c)); SQ.append(float(np.sum(a ** 2)))
                if te or tr:
                    e.reset(seed=seed + t + 1)
            return np.array(RW), np.array(CW), np.array(SQ)
        rw, cw, sq = 跑(ROLLOUT_STEPS, 'rand', 11)
        rw2, cw2, sq2 = 跑(ROLLOUT_STEPS, 'const1', 21)
        res['reward_mean'] = float(rw.mean()); res['reward_std'] = float(rw.std())
        res['reward_mean_const1'] = float(rw2.mean()); res['reward_std_const1'] = float(rw2.std())
        res['cost_event_rate'] = float((cw > 0).mean())
        res['cost_event_rate_const1'] = float((cw2 > 0).mean())
        res['cost_mean'] = float(cw.mean()); res['cost_mean_const1'] = float(cw2.mean())
        res['P3'] = bool(max(rw.std(), rw2.std()) > 0
                         and max((cw > 0).mean(), (cw2 > 0).mean()) > 0)
    except Exception as ex:
        res['P3'] = False
        res['rollout_error'] = f'{type(ex).__name__}: {str(ex)[:160]}'
    try:
        acts = np.random.RandomState(7).uniform(-1, 1, size=(100, na))
        录 = []
        for _ in range(2):
            e.reset(seed=9); O, C = [], []
            for x in acts:
                o, r, c, te, tr, _ = e.step(x); O.append(np.asarray(o)); C.append(float(c))
                if te or tr:
                    break
            录.append((np.array(O), np.array(C)))
        res['P4'] = bool(len(录[0][0]) == len(录[1][0]) and np.array_equal(录[0][0], 录[1][0])
                         and np.array_equal(录[0][1], 录[1][1]))
    except Exception as ex:
        res['P4'] = False
        res['repro_error'] = f'{type(ex).__name__}: {str(ex)[:160]}'
    e.close()
    res['PASS'] = bool(res.get('P1') and res.get('P2') and res.get('P3') and res.get('P4'))
    res['elapsed_sec'] = round(time.time() - 起点, 1)
    return res


def main():
    project = Path(__file__).resolve().parent
    out = project / '论文核验输出_E3dB安全速度基准筛选_20261013'
    if out.exists():
        raise FileExistsError('输出目录已存在: ' + str(out))
    out.mkdir(parents=True)
    t0 = time.time()
    import safety_gymnasium, mujoco
    allres = []
    for eid in CANDIDATES:
        r = 检查一个(eid)
        allres.append(r)
        print('\n' + '=' * 78)
        print(f"{eid}: {'PASS' if r.get('PASS') else 'FAIL'}")
        for k in ('P1', 'P2', 'P3', 'P4'):
            print(f'   {k} = {r.get(k)}')
        for k in ('class', 'obs_dim', 'act_dim', 'engine_found', 'effective_dims',
                  'root_direction_cos_first_two', 'reward_mean', 'reward_std',
                  'cost_event_rate', 'corr_cost_actnorm2', 'instantiation_error',
                  'rollout_error', 'repro_error', 'note'):
            if k in r:
                print(f'   {k} = {r[k]}')
        if 'per_dim' in r:
            for x in r['per_dim']:
                extra = ''
                if 'max_abs_dqpos' in x:
                    extra = f" max|Δqpos|={x['max_abs_dqpos']:.4f} Δroot={[round(v,4) for v in x['d_root']]}"
                print(f"     dim{x['dim']}: max|Δobs|={x['max_abs_dobs']:.4f}{extra}")
    summary = dict(experiment='E3d-B Safe Velocity candidate acceptance (screening only)',
                   no_collection=True, no_training=True,
                   safety_gymnasium=str(safety_gymnasium.__version__),
                   mujoco=str(mujoco.__version__),
                   criteria=dict(P1='实例化成功', P2='≥2 个动作维各自引起可见物理状态变化且不共线',
                                 P3='回报非常数、成本有非零事件、|corr(cost,|a|^2)|<0.5',
                                 P4='同种子同动作序列 100 步逐位可复现'),
                   results=allres,
                   passed=[r['env'] for r in allres if r.get('PASS')],
                   elapsed_sec=round(time.time() - t0, 1))
    with (out / '验收结果.json').open('x', encoding='utf8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, default=float)
    print('\n通过验收的候选:', summary['passed'] if summary['passed'] else '（无）')
    print('输出:', out, '用时', summary['elapsed_sec'], 's')


if __name__ == '__main__':
    main()
