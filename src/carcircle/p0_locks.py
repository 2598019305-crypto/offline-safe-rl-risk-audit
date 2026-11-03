#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""P0：建立 final_stage_20261015 目录并写出四个锁文件 + 复现身份清单。

- BASELINE_CONFIG_LOCK.json：P1 基线（含 BC 正式预算）冻结配置
- ABLATION_CONFIG_LOCK.json：P2 门控消融（B1..B5）冻结配置
- SEAL_REGISTRY.json：P3 封存评估登记（125000–125099 保持未使用）
- MANIFEST_SHA256.json：本目录全部文件的哈希（先建目录与锁，运行产物随后追加）
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
WS = Path(os.environ.get('REPRO_WORK_DIR', REPO_ROOT))
FS = Path(os.environ.get('REPRO_OUTPUT_DIR', REPO_ROOT / 'outputs')) / 'final_stage_20261015'
WIN = Path(os.environ.get('CAR_CIRCLE_DATA_DIR', REPO_ROOT / 'data' / 'raw' / 'car_circle'))


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for c in iter(lambda: f.read(1 << 20), b''):
            h.update(c)
    return h.hexdigest()


def main():
    FS.mkdir(parents=True, exist_ok=True)
    (FS / 'protocols').mkdir(exist_ok=True)
    # 复制既有协议文本（只读副本，不覆盖原件）
    for src, name in ((WS / 'PROTOCOL_COSTPRED_V1.md', 'PROTOCOL_COSTPRED_V1.md'),
                      (WS / 'MECHANISM_AUDIT_20261014' / '13_P0_A1_RISK_DEFINITION_AUDIT.md',
                       'P0_1_risk_definition_audit.md')):
        if src.is_file():
            shutil.copy2(src, FS / 'protocols' / name)
    for n in ('README_执行与后续使用.md', 'PROTOCOL_DIAG_VALIDITY_V1.md'):
        s = REPO_ROOT / 'docs' / 'protocols' / n
        if s.is_file():
            shutil.copy2(s, FS / 'protocols' / n)

    baseline = dict(
        protocol='BASELINE_CONFIG_LOCK v1', frozen_at=time.strftime('%Y-%m-%d %H:%M:%S'),
        purpose='P1 开发集基线对比（仅用开发环境种子 510-517；不触碰封存 125000-125099）',
        reuse_from_existing_ledger=dict(
            Reward_only='checkpoints_fixed/Reward-only_seed0..7 actor_step10000（8 训练种子）',
            D10='checkpoints_fixed/D10_seed0..7 actor_step10000',
            E100='checkpoints_fixed/E100_seed0..7 actor_step10000',
            standard_IQL='checkpoints_fixed/standard_IQL_seed0..7 actor_step10000',
            note='IQL 家族的 expectile/β 配方与其余臂不同，属算法级参考；Reward-only 才是风险项的单因素对照'),
        new_bc=dict(arms=['BC-All', 'BC-Safe'], updates=10000, train_seeds=list(range(8)),
                    net='MLP(8->256->256->2, ReLU, Tanh)', loss='deterministic MSE behavior cloning',
                    optimizer='Adam lr=3e-4 batch=256',
                    data_all='train split 1023 trajectories (306900 transitions)',
                    data_safe='lowest-total-cost 50% of train trajectories (ties by traj_hash(7,tid))',
                    seeding='torch.manual_seed(seed); batch stream RandomState(1000+seed)',
                    eval='dev env seeds 510-517, new env per episode, np.random.seed before reset',
                    script='p1_bc_formal.py'),
        external_safe_rl=dict(candidates=['CPQ', 'FISOR'], status='选型中（需先核验可移植性）',
                              rule='必须使用官方/作者实现，不得自行重写后仍称官方基线'),
        metrics=['原生累计成本', '回报', '最差 25% 回合成本均值', '训练时间与预算'],
        prohibitions=['不得用开发表现挑选或删除参赛者',
                      '不得把 2k 的 G2 探针当作正式算法排名',
                      '不得在 P3 之前访问 125000-125099'])
    (FS / 'BASELINE_CONFIG_LOCK.json').write_text(
        json.dumps(baseline, ensure_ascii=False, indent=2), encoding='utf-8')

    ablation = dict(
        protocol='ABLATION_CONFIG_LOCK v1', frozen_at=time.strftime('%Y-%m-%d %H:%M:%S'),
        purpose='P2 门控单因素消融：B1..B5，同一 Q/V 主干、同一 actor 初始化、同一 minibatch 流',
        fixed=dict(beta=0.5, lam=1.0, clip=100.0, batch=256, updates=10000,
                   risk_definition='cross-fit 八折（20261017 折规则）用于新臂；'
                                   '旧 self-inclusive 的 AG-RW 结果单独标注，不当作仅改门控的一因子对照',
                   candidate_enumeration='稳定最近优先截断（新合规执行），不追认旧 G-A2',
                   train_seeds=[0, 1, 2], dev_env_seeds=list(range(510, 518))),
        arms=dict(
            B1='g≡0（Reward-only）', B2='g≡1（D10）',
            B3='仅数据支持条件门控（support-only）：只检查原行为与候选是否各达 k0、'
               '风险评分有限、候选优势可接受；不检查风险下降门槛',
            B4='完整门控：支持 + 风险改善差 ≥0.10 + 奖励容差',
            B5='B4 门控标签在 minibatch 内保持开启数置乱'),
        operator_consistency_gate=['逐样本 A,P,g,w,logpi 形状均为 [256]',
                                   'g 全零严格回到 B1；g 全一严格回到 B2',
                                   '空邻域 UNKNOWN 与演员风险 NaN→0 两种语义分开说明'],
        readings=['静态门控率', '危险样本开启率', '支持/风险差/奖励容差逐层淘汰', '裁剪后总质量', 'TV',
                  '样本 ESS', '成本事件权重比', '相对梯度向量差与余弦', '相同状态动作 L2',
                  '开发集部署成本/回报'],
        exit_rule='B3 vs B4 能实证隔离"只是支持度"与"还需要风险可分辨性与奖励容差"；无差异则如实报负结果')
    (FS / 'ABLATION_CONFIG_LOCK.json').write_text(
        json.dumps(ablation, ensure_ascii=False, indent=2), encoding='utf-8')

    seal = dict(
        protocol='SEAL_REGISTRY v1', frozen_at=time.strftime('%Y-%m-%d %H:%M:%S'),
        sealed_env_seeds=dict(range=[125000, 125099], status='未使用（保持封存）',
                              note='P3 之前不得访问；启用需单独的明确批准与预先冻结方案'),
        consumed_seeds=dict(costpred=[600, 631],
                            note='COSTPRED_V1 已消耗，不能当作新的封存数据重复使用'),
        p3_prereq=['P0/P1/P2 完成', '算法名单/检查点 SHA/超参/指标/失败标准/输出格式全部锁定',
                   '与既有冻结封存协议兼容性核对完成'],
        p3_design=dict(n_env_seeds=100, per_seed_metrics=['原生成本均值', '回报均值', 'CVaR25'],
                       unit='训练种子层配对；不把 100×训练种子个回合当独立训练重复',
                       one_shot=True, retry_rule='只允许原配置/原种子的技术重试并透明登记'),
        prohibited_claims=['门控保证更低成本', '支持度代表安全', '局部 Pq 即轨迹 CVaR',
                           '单个上界保证多步安全'])
    (FS / 'SEAL_REGISTRY.json').write_text(json.dumps(seal, ensure_ascii=False, indent=2),
                                           encoding='utf-8')

    man = {}
    for p in sorted(FS.rglob('*')):
        if p.is_file() and p.name != 'MANIFEST_SHA256.json':
            man[str(p.relative_to(FS)).replace('\\', '/')] = dict(sha256=sha(p),
                                                                  bytes=p.stat().st_size)
    # 关键身份：数据、环境代码、结果目录
    ident = dict(
        dataset=dict(hdf5=sha(L.DATASET['OfflineCarCircle-v0']) if False else None,
                     note='见下方 data_block'),
        created=time.strftime('%Y-%m-%d %H:%M:%S'), files=man)
    try:
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        import night_lib as L  # noqa: E402
        d = L.DATASET['OfflineCarCircle-v0']
        ident['data_block'] = dict(path=str(d), sha256=sha(d), bytes=Path(d).stat().st_size)
        bsg = REPO_ROOT / 'third_party' / 'bullet_safety_gym'
        ident['env'] = dict(env_id='OfflineCarCircle-v0',
                            seeding_fix='每回合新建环境 + reset 前 np.random.seed(env_seed)')
    except Exception as e:      # noqa: BLE001
        ident['data_block'] = repr(e)
    (FS / 'MANIFEST_SHA256.json').write_text(
        json.dumps(ident, ensure_ascii=False, indent=2), encoding='utf-8')
    print('final_stage 目录:', FS)
    print(json.dumps({k: v for k, v in ident.items() if k != 'files'},
                     ensure_ascii=False, indent=2)[:1200])


if __name__ == '__main__':
    main()
