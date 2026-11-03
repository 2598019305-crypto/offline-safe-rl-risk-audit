#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""成本预测实验（COSTPRED_V1）：只读盘点可用策略检查点，生成策略清单与哈希。

在冻结协议**之前**运行；只读取文件是否存在、能否加载，不计算任何成本目标值。
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'carcircle'))
import iql  # noqa: E402
import night_lib as L  # noqa: E402

FIXED = L.WORK / 'checkpoints_fixed'
GA2 = L.WORK / 'checkpoints_ga2'
GATEFIX = L.WORK / 'checkpoints_ga2_gatefix'
P2 = L.WORK / 'checkpoints_ga2_p2'
G2 = L.WORK / 'checkpoints_g2'
OUT = L.WORK / 'costpred'

TRAIN_FAMILIES = ('Reward-only', 'D10', 'E100', 'standard_IQL')
TEST_FAMILIES = ('AG-RW', 'SHUFFLE-G', 'AG-RWcv2', 'E-M1_Shuffled-D10', 'E-M2_D10-lam0.5')


def sha(p: Path) -> str:
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for c in iter(lambda: f.read(1 << 20), b''):
            h.update(c)
    return h.hexdigest()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    pol = []
    # 训练家族：checkpoints_fixed 的四臂 × 8 训练种子（主终点 10000）
    for arm in TRAIN_FAMILIES:
        for seed in range(8):
            p = FIXED / ('%s_seed%d' % (arm, seed)) / 'actor_step10000.pt'
            if p.is_file():
                pol.append(dict(policy_id='fixed-%s-s%d' % (arm, seed), arm=arm, train_seed=seed,
                                path=str(p), split='train', source='checkpoints_fixed/step10000'))
    # 训练家族：GA2 目录下的 Reward-only / D10（同家族、另一批训练）
    for arm in ('Reward-only', 'D10'):
        for seed in (0, 1, 2):
            p = GA2 / ('%s_seed%d' % (arm, seed)) / 'actor_step10000.pt'
            if p.is_file():
                pol.append(dict(policy_id='ga2-%s-s%d' % (arm, seed), arm=arm, train_seed=seed,
                                path=str(p), split='train', source='checkpoints_ga2/step10000'))
    # 测试家族：AG-RW / SHUFFLE-G
    for arm in ('AG-RW', 'SHUFFLE-G'):
        for seed in (0, 1, 2):
            p = GA2 / ('%s_seed%d' % (arm, seed)) / 'actor_step10000.pt'
            if p.is_file():
                pol.append(dict(policy_id='ga2-%s-s%d' % (arm, seed), arm=arm, train_seed=seed,
                                path=str(p), split='test', source='checkpoints_ga2/step10000'))
    # 测试家族：门控修正 / 置乱风险 / λ=0.5
    for seed in (0, 1, 2):
        p = GATEFIX / ('AG-RWcv2_seed%d' % seed) / 'actor_step10000.pt'
        if p.is_file():
            pol.append(dict(policy_id='gatefix-AG-RWcv2-s%d' % seed, arm='AG-RWcv2',
                            train_seed=seed, path=str(p), split='test',
                            source='checkpoints_ga2_gatefix/step10000'))
    for arm in ('E-M1_Shuffled-D10', 'E-M2_D10-lam0.5'):
        for seed in (0, 1, 2):
            p = P2 / ('%s_seed%d' % (arm, seed)) / 'actor_step10000.pt'
            if p.is_file():
                pol.append(dict(policy_id='p2-%s-s%d' % (arm, seed), arm=arm, train_seed=seed,
                                path=str(p), split='test', source='checkpoints_ga2_p2/step10000'))
    # 测试家族：行为克隆（BC-All / BC-Safe，两种预算都放测试）
    for tag, p in (('BC-All-10000', FIXED / 'bc_BC-All_10000.pt'),
                   ('BC-Safe-10000', FIXED / 'bc_BC-Safe_10000.pt'),
                   ('BC-All-2000', G2 / 'bc_BC-All_2000.pt'),
                   ('BC-Safe-2000', G2 / 'bc_BC-Safe_2000.pt')):
        if p.is_file():
            pol.append(dict(policy_id='bc-%s' % tag, arm='BC-' + tag.split('-')[1],
                            train_seed=None, path=str(p), split='test', source=str(p.parent.name)))
    # 加载校验
    bad = []
    for r in pol:
        try:
            ck = torch.load(r['path'], map_location='cpu', weights_only=False)
            sd = ck['actor'] if 'actor' in ck else (ck['bc'] if 'bc' in ck else ck)
            if 'log_std' in sd:
                a = iql.Actor(8, 2)
                a.load_state_dict(sd)
                r['kind'] = 'stochastic_gaussian'
            else:
                # 行为克隆检查点只保存确定性网络（键为 "0.weight" 等），没有 log_std
                import torch.nn as nn
                net = iql.mlp(8, 2, 256, nn.Tanh)
                net.load_state_dict(sd)
                net.eval()
                r['kind'] = 'deterministic_net'
            r['loadable'] = True
            r['sha256'] = sha(Path(r['path']))
        except Exception as e:                      # noqa: BLE001
            r['loadable'] = False
            r['error'] = repr(e)
            bad.append(r['policy_id'])
    res = dict(n_policies=len(pol), n_train=sum(1 for r in pol if r['split'] == 'train'),
               n_test=sum(1 for r in pol if r['split'] == 'test'),
               by_arm={}, unloadable=bad, policies=pol)
    for r in pol:
        res['by_arm'][r['arm']] = res['by_arm'].get(r['arm'], 0) + 1
    (OUT / 'policy_roster.json').write_text(json.dumps(res, ensure_ascii=False, indent=2),
                                            encoding='utf-8')
    print(json.dumps({k: v for k, v in res.items() if k != 'policies'},
                     ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
