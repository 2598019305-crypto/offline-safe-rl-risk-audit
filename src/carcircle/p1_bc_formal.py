#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""P1：BC-All / BC-Safe 的正式预算对照（10,000 次演员更新 × 8 个训练种子）。

配置在运行前已冻结（final_stage_20261015/BASELINE_CONFIG_LOCK.json）：
  - 网络：MLP(obs→256→256→act)，ReLU 隐层，Tanh 输出（与既有 BC 完全一致）
  - 损失：确定性 MSE 行为克隆；Adam lr=3e-4；batch 256；updates=10000
  - 数据：BC-All = 训练划分全部 1023 条轨迹；BC-Safe = 训练划分中总成本最低的 50% 轨迹
    （并列按 traj_hash(7, tid) 打破，与 G2 修复轮完全相同的规则）
  - 训练种子 0–7：torch.manual_seed(seed) + RandomState(1000+seed) 采样流
  - 评估：仅开发种子 510–517；**每回合新建环境 + reset 前 np.random.seed(env_seed)**
输出：checkpoints_p1_bc/、results_p1_bc/eval_episode.csv、bc_train_log.csv、SHA 清单
"""
from __future__ import annotations

import csv
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
import iql          # noqa: E402
import night_lib as L  # noqa: E402
import gates        # noqa: E402

import warnings
warnings.filterwarnings('ignore')
import gymnasium as gym                      # noqa: E402
import dsrl.offline_bullet_safety_gym  # noqa: E402,F401

CK = L.WORK / 'checkpoints_p1_bc'
OUT = L.WORK / 'results_p1_bc'
UPDATES = 10000
SEEDS = tuple(range(8))
MILESTONES = (2000, 10000)
ENVSEEDS = list(range(510, 518))


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for c in iter(lambda: f.read(1 << 20), b''):
            h.update(c)
    return h.hexdigest()


class NetWrap(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net


def eval_episode(policy, env_id, seed):
    env = gym.make(env_id)
    try:
        np.random.seed(int(seed) % (2 ** 31 - 1))
        obs, _ = env.reset(seed=seed)
        r_tot = c_tot = 0.0
        events = []
        t = 0
        done = False
        while not done:
            with torch.no_grad():
                a = policy.net(torch.from_numpy(np.asarray(obs, np.float32)[None, :])).numpy()[0]
            a = np.clip(a, env.action_space.low, env.action_space.high).astype(np.float32)
            obs, r, c, done, _ = L.step_env(env, a)
            r_tot += r
            c_tot += c
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
    CK.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    log = L.Log('p1_bc')
    data = iql.load_and_split(L.ENV_A)
    bnd = data['bounds']
    S, A, C = data['S'], data['A'], data['C']
    split = {'train': [], 'dev': [], 'critic_only': []}
    for tid in range(len(bnd)):
        h = L.traj_hash(L.RNG_SEED, tid)
        split['train' if h < 700 else ('dev' if h < 850 else 'critic_only')].append(tid)
    cost_ret = {t: float(C[bnd[t][0]:bnd[t][1]].sum()) for t in split['train']}
    safe = sorted(sorted(split['train'], key=lambda t: (cost_ret[t], L.traj_hash(7, t)))[
                  :max(1, len(split['train']) // 2)])

    def stack(tids):
        return (np.concatenate([S[bnd[t][0]:bnd[t][1]] for t in tids]).astype(np.float32),
                np.concatenate([A[bnd[t][0]:bnd[t][1]] for t in tids]).astype(np.float32))

    meta = dict(config='BASELINE_CONFIG_LOCK.json', updates=UPDATES, seeds=list(SEEDS),
                train_trajectories=len(split['train']), safe_trajectories=len(safe),
                train_transitions=int(len(data['train_idx'])),
                safe_transitions=int(sum(bnd[t][1] - bnd[t][0] for t in safe)),
                dev_env_seeds=ENVSEEDS, eval_protocol='新建环境+reset前np.random.seed')
    log('P1 BC 正式预算启动：%s' % json.dumps(meta, ensure_ascii=False))
    rows, traini = [], []
    for tag, tids in (('BC-All', split['train']), ('BC-Safe', safe)):
        X, Y = stack(tids)
        for seed in SEEDS:
            t1 = time.time()
            net = gates._bc_train(X, Y, log, '%s-s%d' % (tag, seed), updates=UPDATES, seed=seed)
            torch.save(dict(bc=net.state_dict(), tag=tag, updates=UPDATES, seed=seed,
                            split='train' if tag == 'BC-All' else 'lowest_cost_50pct'),
                       CK / ('bc_%s_%d.pt' % (tag, UPDATES)))
            torch.save(dict(bc=net.state_dict(), tag=tag, updates=UPDATES, seed=seed,
                            split='train' if tag == 'BC-All' else 'lowest_cost_50pct'),
                       CK / ('bc_%s_seed%d_%d.pt' % (tag, seed, UPDATES)))
            net.eval()
            policy = NetWrap(net)
            for es in ENVSEEDS:
                rows.append(dict(arm=tag, train_seed=seed, milestone=UPDATES, env_seed=es,
                                 **eval_episode(policy, L.ENV_A, es)))
            traini.append(dict(arm=tag, train_seed=seed, updates=UPDATES,
                               seconds=round(time.time() - t1, 1),
                               n_transitions=int(len(X)), ckpt_sha256=sha(CK / ('bc_%s_seed%d_%d.pt'
                                                                                % (tag, seed, UPDATES)))))
            with (OUT / 'eval_episode.csv').open('w', encoding='utf-8-sig', newline='') as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0]))
                w.writeheader(); w.writerows(rows)
            log('  %s seed%d 完成（%.0fs）' % (tag, seed, time.time() - t1))
    with (OUT / 'bc_train_log.csv').open('w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(traini[0]))
        w.writeheader(); w.writerows(traini)
    summ = {}
    import statistics as st
    for tag in ('BC-All', 'BC-Safe'):
        for seed in SEEDS:
            sel = [r for r in rows if r['arm'] == tag and r['train_seed'] == seed]
            summ['%s_seed%d' % (tag, seed)] = dict(
                cost_mean=float(st.mean(r['cost_sum'] for r in sel)),
                reward_mean=float(st.mean(r['reward_sum'] for r in sel)))
    (OUT / 'p1_bc_summary.json').write_text(json.dumps(
        dict(meta=meta, per_arm_seed=summ, seconds=round(time.time() - t0, 1)), ensure_ascii=False,
        indent=2), encoding='utf-8')
    print(json.dumps(summ, ensure_ascii=False, indent=2))
    print('P1 BC 完成，用时 %.0fs' % (time.time() - t0))


if __name__ == '__main__':
    main()
