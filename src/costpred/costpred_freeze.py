#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""冻结 COSTPRED_V1 协议：登记协议文本、策略清单、评估回合与代码哈希。
在计算任何目标成本值之前运行。
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / 'src' / 'carcircle'))
import night_lib as L  # noqa: E402

OUT = L.WORK / 'costpred'
WS = Path(os.environ.get('REPRO_WORK_DIR', REPO_ROOT))


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for c in iter(lambda: f.read(1 << 20), b''):
            h.update(c)
    return h.hexdigest()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    ros = json.loads((OUT / 'policy_roster.json').read_text(encoding='utf-8'))
    fr = dict(
        protocol='COSTPRED_V1',
        frozen_at=time.strftime('%Y-%m-%d %H:%M:%S'),
        purpose=('用完全离线特征预测策略在未使用回合上的真实平均原生成本；'
                 '测试集为训练中未见过的策略家族'),
        target_episodes=dict(seeds=list(range(600, 632)), n=32,
                             note='本机从未用于评估或模型选择；旧开发种子 510-517 不作目标集'),
        sealed_seeds_untouched='125000-125099（继续不访问）',
        eval_protocol='每回合新建环境对象 + reset 前 np.random.seed(env_seed)；动作取 net(obs) 确定性均值并裁剪',
        primary_target='mean_cost over 32 episodes',
        secondary_targets=['cvar25', 'mean_cost_after_10pct'],
        roster=dict(n=ros['n_policies'], n_train=ros['n_train'], n_test=ros['n_test'],
                    by_arm=ros['by_arm'], roster_sha256=sha(OUT / 'policy_roster.json'),
                    split_rule='按策略家族切分：训练={Reward-only,D10,E100,standard_IQL}；'
                               '测试={AG-RW,SHUFFLE-G,AG-RWcv2,E-M1,E-M2,BC-All,BC-Safe}'),
        features=dict(
            f_cost_surrogate='mean_{s in S*} p_hat(c>0|s,a_pi(s))，逻辑回归在数据集训练划分上拟合',
            f_action_magnitude='mean_{s in S*} ||a_pi(s)||_2',
            f_action_ood='mean_{s in S*} min_j ||a_pi(s)-a_j||_2，j 取 z(s) 最近 64 个数据状态',
            state_set='S[train_idx][::25]'),
        models=dict(FULL_3='StandardScaler+Ridge(alpha=1.0) on 3 features',
                    single_feature='同结构单特征',
                    MEAN_TRAIN='训练策略目标均值常数',
                    BEST_SINGLE_ON_TRAIN='在训练策略上选最优单特征'),
        primary_metric='Spearman rho on 19 test policies',
        decision_bands={'usable': 'rho>=0.60', 'weak': '0.30<=rho<0.60', 'failed': 'rho<0.30'},
        secondary_analysis=dict(random_split='固定种子随机 70/30，仅作较弱对照',
                                note='同家族可能同时出现在两侧，不作为外推证据'),
        prohibitions=['不得在测试集上做特征/模型选择', '不得重复运行最终评价以挑结果',
                      '不得称为外部独立确认', '不得主张 CVaR 保证'],
        code_sha256={n: sha(Path(n)) for n in
                     [str(REPO_ROOT / 'src' / 'costpred' / 'costpred_roster.py')]},
        protocol_text_sha256=sha(WS / 'PROTOCOL_COSTPRED_V1.md'),
        protocol_text_path=str(WS / 'PROTOCOL_COSTPRED_V1.md'),
    )
    (OUT / 'COSTPRED_V1_frozen.json').write_text(json.dumps(fr, ensure_ascii=False, indent=2),
                                                 encoding='utf-8')
    print(json.dumps(fr, ensure_ascii=False, indent=2)[:2000])


if __name__ == '__main__':
    main()
