#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""P0 健康检查：检查点清单与哈希、数据划分、播种修复复验、算子一致性断言、无效记录登记。

只读；不训练、不采样、不触碰封存种子。
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

sys.path.insert(0, str(Path(__file__).resolve().parent))
import iql          # noqa: E402
import night_lib as L  # noqa: E402

FS = L.WORK / 'final_stage_20261015'
FS.mkdir(parents=True, exist_ok=True)


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for c in iter(lambda: f.read(1 << 20), b''):
            h.update(c)
    return h.hexdigest()


def inventory():
    rows = []
    roots = [('checkpoints_fixed', L.WORK / 'checkpoints_fixed'),
             ('checkpoints_ga2', L.WORK / 'checkpoints_ga2'),
             ('checkpoints_ga2_gatefix', L.WORK / 'checkpoints_ga2_gatefix'),
             ('checkpoints_ga2_p2', L.WORK / 'checkpoints_ga2_p2'),
             ('checkpoints_g2', L.WORK / 'checkpoints_g2'),
             ('checkpoints_p1_bc', L.WORK / 'checkpoints_p1_bc')]
    for tag, root in roots:
        if not root.is_dir():
            continue
        for p in sorted(root.rglob('*.pt')):
            try:
                ck = torch.load(p, map_location='cpu', weights_only=False)
                sub = ck.get('cfg', {}) if isinstance(ck, dict) else {}
                rows.append(dict(root=tag, rel=str(p.relative_to(root)).replace('\\', '/'),
                                 arm=str(ck.get('arm', ck.get('tag', '?'))) if isinstance(ck, dict) else '?',
                                 seed=int(ck['seed']) if isinstance(ck, dict) and 'seed' in ck else '',
                                 step=int(ck.get('step', ck.get('updates', -1))) if isinstance(ck, dict) else -1,
                                 keys=','.join(sorted(ck.keys())) if isinstance(ck, dict) else 'tensor',
                                 operator=str(sub.get('operator', '')), beta=str(sub.get('beta', '')),
                                 lam=str(sub.get('lam', '')), bytes=p.stat().st_size,
                                 sha256=sha(p)))
            except Exception as e:                      # noqa: BLE001
                rows.append(dict(root=tag, rel=str(p.relative_to(root)), arm='LOAD_ERROR',
                                 sha256='', bytes=p.stat().st_size, keys=repr(e)))
    return rows


def main():
    t0 = time.time()
    data = iql.load_and_split(L.ENV_A)
    bnd = data['bounds']
    split_sizes = {k: len(v) for k, v in data['split'].items()}
    counts = dict(n_traj=len(bnd), n_transitions_total=int(len(data['S'])),
                  train_transitions=int(len(data['train_idx'])),
                  critic_transitions=int(len(data['critic_idx'])),
                  split_trajectories=split_sizes,
                  dev_transitions=int(sum(bnd[t][1] - bnd[t][0] for t in data['split']['dev'])))
    # 播种与确定性：同一策略、同一 env 种子，两次独立回合应完全一致
    import gymnasium as gym
    import dsrl.offline_bullet_safety_gym  # noqa: F401

    def ep(policy_net, seed):
        env = gym.make(L.ENV_A)
        try:
            np.random.seed(seed % (2 ** 31 - 1))
            obs, _ = env.reset(seed=seed)
            cs, rs, done, t = 0.0, 0.0, False, 0
            while not done:
                with torch.no_grad():
                    a = policy_net(torch.from_numpy(np.asarray(obs, np.float32)[None, :])).numpy()[0]
                a = np.clip(a, env.action_space.low, env.action_space.high).astype(np.float32)
                obs, r, c, done, _ = L.step_env(env, a)
                cs += c; rs += r; t += 1
            return dict(cost=cs, reward=rs, steps=t)
        finally:
            env.close()

    ck = torch.load(L.WORK / 'checkpoints_ga2' / 'Reward-only_seed0' / 'actor_step10000.pt',
                    map_location='cpu', weights_only=False)
    a1 = iql.Actor(8, 2); a1.load_state_dict(ck['actor']); a1.eval()
    r1 = ep(a1.net, 511); r2 = ep(a1.net, 511)
    # 算子一致性：形状、g=0 回到 Reward-only、g=1 回到 D10
    ck2 = torch.load(L.WORK / 'checkpoints' / 'P1_seed0' / 'critic_step40000.pt',
                     map_location='cpu', weights_only=False)
    q1, q2, v = iql.mlp(10, 1), iql.mlp(10, 1), iql.mlp(8, 1)
    q1.load_state_dict(ck2['q1']); q2.load_state_dict(ck2['q2']); v.load_state_dict(ck2['v'])
    for m in (q1, q2, v):
        m.eval()
        for p in m.parameters():
            p.requires_grad_(False)
    rng = np.random.RandomState(20261013)
    tr = data['train_idx'].astype(np.int64)
    b = rng.choice(len(tr), 256)
    j = tr[b]
    risk = np.load(L.WORK / 'results_crossfit' / 'd10_selfinclusive_full.npy')[j]
    st = torch.from_numpy(data['S'][j]); at = torch.from_numpy(data['A'][j])
    om, osd = torch.from_numpy(data['obs_mean']), torch.from_numpy(data['obs_std'])
    with torch.no_grad():
        sj = (st - om) / osd
        adv = (torch.minimum(q1(torch.cat([sj, at], -1)), q2(torch.cat([sj, at], -1)))
               - v(sj)).reshape(-1).numpy().astype(np.float64)
    a0 = iql.Actor(8, 2); a0.load_state_dict(ck['actor'])
    lp = a0.log_prob(sj, at)
    w_ro = iql.weights_mechanism(adv, None)
    w_d10 = iql.weights_mechanism(adv, risk)
    w_g0 = iql.weights_mechanism(adv, risk, gate=np.zeros(256))
    w_g1 = iql.weights_mechanism(adv, risk, gate=np.ones(256))
    opt_chk = dict(
        logpi_shape=list(lp.shape), n_samples=int(lp.numel()),
        scalar_loss=bool(torch.tensor(-(torch.from_numpy(w_ro).float() * lp).mean()).dim() == 0),
        weights_shape=list(w_ro.shape), risk_shape=list(risk.shape),
        g0_equals_reward_only=bool(np.allclose(w_g0, w_ro, atol=1e-12)),
        g1_equals_d10=bool(np.allclose(w_g1, w_d10, atol=1e-12)),
        w_mean_ro=float(w_ro.mean()), w_mean_d10=float(w_d10.mean()),
        clip_hits_ro=float((w_ro >= L.WEIGHT_CAP - 1e-7).mean()))
    inv = dict(
        note='以下为已知无效或非严格复现的记录，不得作为有效证据引用',
        items=[
            dict(id='INVALID-1', what='加权广播缺陷版演员（w (n,1)×log_prob (n,) → (n,n)）',
                 reason='逐样本加权被抹掉；作废于 INCIDENT_weight_broadcast',
                 files='results/、results_phase2/ 下旧产物'),
            dict(id='INVALID-2', what='环境 reset(seed) 无效版全部闭环评估',
                 reason='回合初条件由全局 RNG 决定，配对无效',
                 files='results/_VOID/、*UNCONTROLLED_void*'),
            dict(id='INVALID-3', what='不正确 critic 门控派生版（critic0 损失配 critic-k 的 σ）',
                 reason='critic 种子 1/2 的 AG-RW 臂并非逐 critic 配对（INCIDENT-11）',
                 files='agrw/gates_per_seed.npz（旧）、results_ga2/'),
            dict(id='INVALID-4', what='G-A2 实际执行版 AG-RW',
                 reason='与冻结 v1.0 不一致（门控派生缺陷 + 旧播种），不得标作严格复现',
                 files='results_ga2/、G_A2_最终实验报告.md'),
            dict(id='INVALID-5', what='P1_M1_SATURATION.json / P1_M2_M3_M4.json（v1）',
                 reason='M2 拒绝计数恒 0、M3 空、M4 全 NaN（INCIDENT-08/09/10）',
                 files='mech_audit/ 同名文件，仅留档'),
            dict(id='INVALID-6', what='COSTPRED_V1 的目标回合 600–631',
                 reason='已被消耗，不得再当作新的封存数据', files='costpred/'),
        ])
    inv_p = FS / 'INVALID_REGISTRY.json'
    inv_p.write_text(json.dumps(inv, ensure_ascii=False, indent=2), encoding='utf-8')
    rows = inventory()
    with (FS / 'CHECKPOINT_INVENTORY.csv').open('w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    tally = {}
    for r in rows:
        tally[r['root']] = tally.get(r['root'], 0) + 1
    hc = dict(
        ran_at=time.strftime('%Y-%m-%d %H:%M:%S'), seconds=round(time.time() - t0, 1),
        data_identity=dict(hdf5=str(L.DATASET['OfflineCarCircle-v0']),
                           sha256=sha(L.DATASET['OfflineCarCircle-v0']),
                           bytes=Path(L.DATASET['OfflineCarCircle-v0']).stat().st_size),
        split=counts, checkpoint_tally=tally, n_checkpoint_files=len(rows),
        seed_determinism=dict(seed=511, run1=r1, run2=r2, identical=bool(r1 == r2)),
        operator_consistency=opt_chk,
        invalid_registry=str(inv_p),
        sealed_seeds='125000–125099 未访问',
        env=dict(python=sys.version.split()[0], numpy=np.__version__, torch=torch.__version__),
        verbatim_checks=dict(
            shapes_ok=bool(opt_chk['logpi_shape'] == [256] and opt_chk['weights_shape'] == [256]),
            g0_to_reward_only=opt_chk['g0_equals_reward_only'],
            g1_to_d10=opt_chk['g1_equals_d10'],
            scalar_loss=opt_chk['scalar_loss']))
    (FS / 'P0_HEALTHCHECK.json').write_text(json.dumps(hc, ensure_ascii=False, indent=2),
                                            encoding='utf-8')
    print(json.dumps({k: v for k, v in hc.items() if k != 'checkpoint_tally'},
                     ensure_ascii=False, indent=2))
    print('检查点分类计数:', json.dumps(tally, ensure_ascii=False))


if __name__ == '__main__':
    main()
