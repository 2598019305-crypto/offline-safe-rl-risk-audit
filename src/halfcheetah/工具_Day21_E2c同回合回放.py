#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""E2c：同回合 MuJoCo 回放（探索性）——按 Day21_E2c同回合回放_协议_v1.0.md 执行。

主对比 12 对：3 种子 × 2 演员起点 × 2 条批次序列之内的 D10 vs 保序变换（λ=10、500 步）。
次要对照 6 对：同一风险 D10 在两条批次序列之间（"训练批次差异"的同一度量参考）。
每对用相同 10 个开发集回放种子 115000–115009，确定性均值动作。

边界：不改动历史结果、不碰 125000–125099、不重训任何网络（演员直接取自 E2b 阶段 II 归档）。
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

TAU = 3.9079
STEPS = 500
LAM = 10.0
SEEDS = (0, 1, 2)
STARTS = ('D10', 'NORM2')
PAIR_BATCHES = (20261013, 20261014)
RISKS = ('D10', 'D10_MONOTONE')
EVAL_START_SEED = 115000
EPISODES = 10
PARALLEL = 6
HORIZON = 1000


def net(i, o):
    return nn.Sequential(nn.Linear(i, 256), nn.ReLU(), nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, o))


class StablePolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.q1 = net(23, 1); self.q2 = net(23, 1); self.vr = net(17, 1)
        self.演员 = net(17, 12)
        self.register_buffer('obs_mean', torch.zeros(1, 17))
        self.register_buffer('obs_std', torch.ones(1, 17))

    def normalized(self, s):
        return (s - self.obs_mean) / self.obs_std

    def params(self, s_norm):
        z = self.演员(s_norm)
        return z[:, :6], z[:, 6:].clamp(-5.0, 2.0)

    def 均值动作(self, raw):
        mu, _ = self.params(self.normalized(raw))
        return torch.tanh(mu)


def _载(path, name):
    spec = importlib.util.spec_from_file_location(name, str(path))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


class 记录器:
    """包住环境，记录每条回合的步数与动作平方范数累加。"""
    def __init__(self, env, slot, sink):
        self.env = env; self.slot = slot; self.sink = sink
        self.steps = 0; self.sumsq = 0.0

    def reset(self, **kw):
        self._flush()
        return self.env.reset(**kw)

    def step(self, a):
        self.steps += 1
        self.sumsq += float(np.sum(np.asarray(a, dtype=np.float64) ** 2))
        return self.env.step(a)

    def _flush(self):
        if self.steps > 0:
            self.sink.append((self.slot, self.steps, self.sumsq))
            self.steps = 0; self.sumsq = 0.0

    def close(self):
        self._flush()
        return self.env.close()

    def __getattr__(self, k):
        return getattr(self.env, k)


def 主函数():
    ap = argparse.ArgumentParser()
    ap.add_argument('--project', type=Path, default=Path('.'))
    ap.add_argument('--stage', choices=['check', 'rollout'], default='check')
    ap.add_argument('--out', type=Path, default=None)
    args = ap.parse_args()
    torch.set_num_threads(2)
    project = args.project.resolve()

    arch = project / '论文核验输出_E2b阶段II_多起点多种子_20261013' / '步500演员检查点.npz'
    if not arch.is_file():
        raise FileNotFoundError('缺 E2b 阶段 II 演员归档: ' + str(arch))
    # 说明：E2b 阶段 II 归档里每条分支存的是一个"参数字典对象"，因此这里必须 allow_pickle=True。
    # 这是当时打包方式留下的缺陷（已单独产出扁平化的安全版本作修正）。
    z = np.load(arch, allow_pickle=True)

    hc_drv = _载(project / '工具_HC_异质驱动.py', 'E2c_HC')
    hc = hc_drv.装好(0.75)
    hc.数据阈值[0] = TAU                      # 必须显式设，否则成本会按 ‖a‖² 算（P3 v1.1 的老坑）
    act_dim = hc.W2.动作维
    env = hc.建环境()
    try:
        env.reset(seed=12345)
        _, _, _, _, i0 = env.step(np.zeros(act_dim, dtype=np.float32))
        c0 = float(i0.get('cost', float('nan')))
        env.reset(seed=12345)
        _, _, _, _, i1 = env.step(np.ones(act_dim, dtype=np.float32))
        c1 = float(i1.get('cost', float('nan')))
    finally:
        env.close()
    ok0 = abs(c0 - 0.0) <= 1e-4
    ok1 = abs(c1 - (6.0 - TAU)) <= 1e-4
    print(f'预飞: cost(零动作)={c0:.6f} {"PASS" if ok0 else "FAIL"} | '
          f'cost(全1动作)={c1:.6f} 期望 {6.0-TAU:.6f} {"PASS" if ok1 else "FAIL"}', flush=True)
    if not (ok0 and ok1):
        raise SystemExit('环境成本口径预飞失败，停止')
    meta = dict(arch=dict(path=str(arch), sha256=__import__('hashlib').sha256(arch.read_bytes()).hexdigest()),
                cost_threshold=float(hc.数据阈值[0]), eval_start_seed=EVAL_START_SEED,
                episodes=EPISODES, parallel=PARALLEL, horizon=HORIZON, lam=LAM, steps=STEPS,
                pair_batches=list(PAIR_BATCHES), seeds=list(SEEDS), starts=list(STARTS))
    print('CHECK PASS:', json.dumps(meta, ensure_ascii=False)[:300], flush=True)
    if args.stage == 'check':
        return

    out = (args.out or project / '论文核验输出_E2c同回合回放_20261013').resolve()
    if out.exists():
        raise FileExistsError('输出目录已存在: ' + str(out))
    out.mkdir(parents=True)
    定批 = _载(project / '公共_定批回放引擎.py', 'E2c_定批')
    t0 = time.time()
    per_ep, ep_rows = [], []

    def 跑一个(start, seed, bs, rk):
        key = f'{start}|seed{seed}|bs{bs}|lam{LAM}|{rk}'
        model = StablePolicy()
        # 先完整加载起点检查点（含 obs_mean/obs_std 与 Q/V），再用归档的分支演员参数覆盖演员。
        # 只加载演员会丢掉观测标准化参数，动作会整体算错——这是必须显式处理的坑。
        if start == 'D10':
            ck_path = project / '输出_P3稳定IQL_DE端点' / f'模型_P3_D10_种子{seed}_40000步.pt'
        else:
            ck_path = project / '论文核验输出_P4动作范数平方对照' / f'模型_P4_NORM2_种子{seed}_40000步.pt'
        ck = torch.load(ck_path, map_location='cpu', weights_only=True)
        model.load_state_dict(ck['state_dict'], strict=True)
        model.演员.load_state_dict({k: torch.from_numpy(v) for k, v in z[key].item().items()}, strict=True)
        model.eval()
        sink = []
        slots = {'i': 0}

        def 工厂():
            i = slots['i']; slots['i'] += 1
            return 记录器(hc.建环境(), i, sink)

        class 零主干:
            def predict(self, 批):
                return np.zeros((len(批), act_dim), dtype=np.float32)

        def 决策(批, _):
            with torch.no_grad():
                return model.均值动作(torch.tensor(批, dtype=torch.float32)).numpy()

        res = 定批.定批回放(零主干(), 决策, 回合数=EPISODES, 并行数=PARALLEL,
                            起始种子=EVAL_START_SEED, 环境工厂=工厂,
                            最长步数=HORIZON, 观察维=hc.W2.观察维, 动作维=act_dim)
        lens = [s for _, s, _ in sink]
        sqs = [q / s for _, s, q in sink]
        if len(lens) != EPISODES:
            raise RuntimeError(f'{key} 回合记录数 {len(lens)} != {EPISODES}')
        映射精确 = len(set(lens)) == 1        # 步数全同时，完成顺序 = 派发顺序 = 回合编号
        for ep in range(EPISODES):
            ep_rows.append(dict(start=start, seed=seed, batch_seed=bs, risk=rk, episode=ep,
                                env_seed=EVAL_START_SEED + ep,
                                ret=float(res['逐回合回报'][ep]), cost=float(res['逐回合成本'][ep]),
                                length=int(lens[ep]), action_norm2_mean=float(sqs[ep]),
                                length_mapping_exact=bool(映射精确)))
        print(f'  {key}: 回报均值 {res["回报均值"]:.1f} 成本均值 {res["成本均值"]:.2f} '
              f'用时 {res["秒数"]}s 步数集合 {sorted(set(lens))}', flush=True)
        return {r['episode']: r for r in ep_rows if r['start'] == start and r['seed'] == seed
                and r['batch_seed'] == bs and r['risk'] == rk}

    # 主对比：12 对
    主对 = []
    for start in STARTS:
        for seed in SEEDS:
            for bs in PAIR_BATCHES:
                A = 跑一个(start, seed, bs, 'D10')
                B = 跑一个(start, seed, bs, 'D10_MONOTONE')
                for ep in range(EPISODES):
                    a, b = A[ep], B[ep]
                    主对.append(dict(type='main_shape', start=start, seed=seed, batch_seed=bs, episode=ep,
                                     env_seed=a['env_seed'],
                                     d_ret=b['ret'] - a['ret'], d_cost=b['cost'] - a['cost'],
                                     d_action_norm2=b['action_norm2_mean'] - a['action_norm2_mean'],
                                     d_length=b['length'] - a['length']))
    # 次要对照：6 对（同一风险 D10，两条批次序列之间）
    次对 = []
    for start in STARTS:
        for seed in SEEDS:
            A = None
            for bs in PAIR_BATCHES:
                cur = 跑一个(start, seed, bs, 'D10')
                if A is None:
                    A = (bs, cur)
                else:
                    for ep in range(EPISODES):
                        a, b = A[1][ep], cur[ep]
                        次对.append(dict(type='control_batch', start=start, seed=seed,
                                         batch_seed=A[0], batch_seed_b=bs, episode=ep,
                                         env_seed=a['env_seed'],
                                         d_ret=b['ret'] - a['ret'], d_cost=b['cost'] - a['cost'],
                                         d_action_norm2=b['action_norm2_mean'] - a['action_norm2_mean'],
                                         d_length=b['length'] - a['length']))
    with (out / '逐回合指标.csv').open('x', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(ep_rows[0].keys())); w.writeheader(); w.writerows(ep_rows)
    with (out / '同回合配对差_主.csv').open('x', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(主对[0].keys())); w.writeheader(); w.writerows(主对)
    with (out / '同回合配对差_次对照.csv').open('x', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(次对[0].keys())); w.writeheader(); w.writerows(次对)
    summary = dict(experiment='E2c exploratory same-episode MuJoCo rollout',
                   status='exploratory_NOT_significance_test', input=meta,
                   n_main_pairs=12, n_control_pairs=6, episodes_per_pair=EPISODES,
                   actors_evaluated=len({(r['start'], r['seed'], r['batch_seed'], r['risk']) for r in ep_rows}),
                   independent_test_seeds_125000_125099_used=False,
                   limits=['10 episodes per cell; not CVaR@25 and no significance claim',
                           'single task, dev set 115000-115009',
                           'cross-start D10 vs NORM2 is not a paired comparison'],
                   elapsed_sec=round(time.time() - t0, 1))
    with (out / '核验与边界.json').open('x', encoding='utf8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print('E2c PASS, output:', out, 'elapsed:', summary['elapsed_sec'], flush=True)


if __name__ == '__main__':
    主函数()
