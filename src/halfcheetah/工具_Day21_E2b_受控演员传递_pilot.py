#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""E2b 阶段 I：受控 actor-only 传递 pilot（只更新演员，Q/V 完全冻结）。

按 `E2b_下一轮实验执行安排_20261013.md` 执行：
  · 起点：P3 D10 seed0 已保存检查点；奖励 Q1/Q2/V 与观测标准化参数冻结。
  · 风险输入：D10 / E100 / NORM2 / D10 严格保序非线性变换，λ ∈ {0, 1, 10} ⇒ 12 个分支。
  · 每臂仅 200 次演员更新；同一初始演员、同一优化器初始化、同一批次序列与随机流。
  · 评估只看固定离线状态集上的确定性平均动作，不做环境回放。

演员更新逐字复刻 P3 v1.1：weights = exp(3·z(z(adv) − λ·z(risk))) 归一化后截断 100；
loss = −mean(w · logp_tanhGaussian)；Adam(lr=3e-4)；clip_grad_norm 5.0。
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

DATA_SHA = 'e0850f2e929a3892806f99109937ca323a74f0d624f4fd5fcc027f7c111e8179'
LR = 3e-4
CLIP_ACTOR = 5.0
BETA = 3.0
W_MAX = 100.0
BATCH = 256
EVAL_N = 4096          # 固定离线状态集（动作评估用）
DIAG_N = 1024          # 固定诊断批次（梯度相似度用）
RECORD_STEPS = (0, 50, 100, 150, 200)
LAMBDAS = (0.0, 1.0, 10.0)
RISKS = ('D10', 'E100', 'NORM2', 'D10_MONOTONE')
BATCH_SEEDS = (20261013, 20261014)


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def file_sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for blk in iter(lambda: f.read(1 << 20), b''):
            h.update(blk)
    return h.hexdigest()


def net(i, o):
    return nn.Sequential(nn.Linear(i, 256), nn.ReLU(), nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, o))


class StablePolicy(nn.Module):
    """与 P3 v1.1 / E2a 完全相同的结构与参数名，保证 strict 载入。"""
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


def standardize(x):
    return (x - x.mean()) / (x.std() + 1e-6)


def actor_weights(adv, risk, lam):
    """逐字复刻 P3 v1.1 的 actor_weights。"""
    z_r = standardize(adv)
    z_p = standardize(risk)
    u = standardize(z_r - lam * z_p)
    w = torch.exp(BETA * u)
    return (w / w.mean().clamp_min(1e-6)).clamp(max=W_MAX)


def risk_batch(kind, labels, sq, ix):
    """取该分支在这一批上的风险数值（严格保序变换在批内做，与 E2a 一致）。"""
    if kind == 'D10':
        r = labels['D10'][ix]
    elif kind == 'E100':
        r = labels['E100'][ix]
    elif kind == 'NORM2':
        r = sq[ix]
    elif kind == 'D10_MONOTONE':
        base = labels['D10'][ix]
        r = np.square(base - base.min() + 0.1)
    else:
        raise ValueError(kind)
    return torch.from_numpy(np.ascontiguousarray(r, dtype=np.float32)).view(-1, 1)


def load_checkpoint(project: Path):
    p3 = project / '输出_P3稳定IQL_DE端点'
    ck_path = p3 / '模型_P3_D10_种子0_40000步.pt'
    ck = torch.load(ck_path, map_location='cpu', weights_only=True)
    if ck.get('seed') != 0 or ck.get('steps') != 40000 or ck.get('arm') != 'D10':
        raise ValueError('检查点身份异常')
    return ck_path, ck


def checks(project: Path):
    data = project / '每日结果' / 'Day20_路径二状态相关代价' / '数据_HalfCheetah_异质.npz'
    p3 = project / '输出_P3稳定IQL_DE端点'
    need = [data, p3 / '标签_P3_D10_原始几何.npz', p3 / '标签_P3_E100_原始几何.npz']
    miss = [str(p) for p in need if not p.exists()]
    if miss:
        raise FileNotFoundError('缺少 E2b 只读输入: ' + '; '.join(miss))
    if file_sha256(data) != DATA_SHA:
        raise ValueError('Day21 数据哈希不符')
    z = np.load(data, allow_pickle=False)
    obs = np.ascontiguousarray(z['obs'], dtype=np.float32)
    act = np.ascontiguousarray(z['act'], dtype=np.float32)
    if obs.shape != (200000, 17) or act.shape != (200000, 6):
        raise ValueError('数据维度不符')
    labels = {}
    for arm in ('D10', 'E100'):
        with np.load(p3 / f'标签_P3_{arm}_原始几何.npz', allow_pickle=False) as v:
            l = np.asarray(v['labels'], dtype=np.float32)
            if l.shape != (200000,) or not np.isfinite(l).all():
                raise ValueError('标签异常 ' + arm)
            labels[arm] = l
    sq = np.sum(act.astype(np.float64) ** 2, axis=1).astype(np.float32)
    ck_path, ck = load_checkpoint(project)
    model = StablePolicy()
    model.load_state_dict(ck['state_dict'], strict=True)
    meta = dict(data=str(data), data_sha256=DATA_SHA, p3_dir=str(p3),
                checkpoint=str(ck_path), checkpoint_sha256=file_sha256(ck_path),
                label_means={k: float(v.mean()) for k, v in labels.items()},
                action_norm2_mean=float(sq.mean()),
                actor_param_count=sum(p.numel() for p in model.演员.parameters()),
                qv_param_count=sum(p.numel() for p in list(model.q1.parameters()) + list(model.q2.parameters()) + list(model.vr.parameters())))
    return meta, obs, act, sq, labels, ck


def selftest():
    torch.manual_seed(1)
    adv = torch.randn(256, 1)
    risk = torch.randn(256, 1)
    assert torch.allclose(actor_weights(adv, risk, 0.0), actor_weights(adv, risk * 3 + 7, 0.0), atol=2e-5)
    for lam in (1.0, 10.0):
        err = float((actor_weights(adv, risk, lam) - actor_weights(adv, 2 * risk + 3, lam)).abs().max())
        if err > 1e-3:
            raise AssertionError(f'正仿射不变性失败 {err}')
    sh = (risk - risk.min() + 0.1)
    w1, w2 = actor_weights(adv, sh, 10.0), actor_weights(adv, sh.square(), 10.0)
    assert float((w1 - w2).abs().max()) > 1e-4, '保序变换未产生权重差异'
    m = StablePolicy()
    s = torch.randn(32, 17); a = torch.rand(32, 6) * 2 - 1
    mu, ls = m.params(m.normalized(s))
    az = torch.atanh(a.clamp(-.999999, .999999))
    logp = (-.5 * ((az - mu) / ls.exp()).square() - ls - .9189).sum(-1, keepdim=True)
    loss = -(actor_weights(torch.randn(32, 1), torch.randn(32, 1), 1.0) * logp).mean()
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.演员.parameters())
    print('E2b selftest PASS: λ=0 不变 / 正仿射不变 / 保序形状改变权重 / 演员梯度通路可用')


def qv_fingerprint(model):
    h = hashlib.sha256()
    for p in list(model.q1.parameters()) + list(model.q2.parameters()) + list(model.vr.parameters()):
        h.update(p.detach().numpy().tobytes())
    return h.hexdigest()


def actor_weights_bytes(model):
    h = hashlib.sha256()
    for p in model.演员.parameters():
        h.update(p.detach().numpy().tobytes())
    return h.hexdigest()


def weight_metrics(w, true_cost):
    a = w.detach().flatten().numpy().astype(np.float64)
    c = true_cost.detach().flatten().numpy().astype(np.float64)
    k = max(1, int(math.ceil(0.1 * len(c))))
    top = np.argsort(c)[-k:]
    tot = float(a.sum())
    return dict(ess=float(tot ** 2 / np.sum(a * a)),
                weight_p50=float(np.median(a)), weight_p95=float(np.percentile(a, 95)),
                clip_fraction=float(np.mean(a >= W_MAX - 1e-3)),
                weight_top_cost_share=float(a[top].sum() / tot),
                weighted_true_cost=float(np.sum(a * c) / tot))


def diag_gradient(model, s, a, w):
    for p in model.演员.parameters():
        p.grad = None
    mu, ls = model.params(model.normalized(s))
    az = torch.atanh(a.clamp(-.999999, .999999))
    logp = (-.5 * ((az - mu) / ls.exp()).square() - ls - .9189).sum(-1, keepdim=True)
    loss = -(w.detach() * logp).mean()
    loss.backward()
    g = torch.cat([p.grad.detach().flatten() for p in model.演员.parameters()])
    for p in model.演员.parameters():
        p.grad = None
    return g


def cos(x, y):
    d = float(x.norm() * y.norm())
    return float((x @ y) / d) if d > 1e-12 else float('nan')


def run_branch(ck, obs, act, sq, labels, risk_kind, lam, batch_seed, steps,
               eval_ix, diag_ix, out_rows, grad_store, act_store):
    model = StablePolicy()
    model.load_state_dict(ck['state_dict'], strict=True)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    for p in model.演员.parameters():
        p.requires_grad_(True)
    qv0 = qv_fingerprint(model)
    init_actor_sha = actor_weights_bytes(model)
    optim = torch.optim.Adam(model.演员.parameters(), lr=LR)

    s_eval = torch.from_numpy(obs[eval_ix])
    with torch.no_grad():
        a_eval0 = model.均值动作(s_eval).clone()
    s_diag = torch.from_numpy(obs[diag_ix]); a_diag = torch.from_numpy(act[diag_ix])
    rng = np.random.RandomState(batch_seed)
    batches = [rng.randint(0, len(obs), size=BATCH) for _ in range(steps)]
    batch_seq_sha = sha256_bytes(np.ascontiguousarray(np.stack(batches)).tobytes())

    def record(step):
        with torch.no_grad():
            a_now = model.均值动作(s_eval)
            d = (a_now - a_eval0)
            per_row = d.norm(dim=1)
            cost_now = torch.clamp(a_now.square().sum(-1) - 3.9079, min=0)
            r_d = risk_batch(risk_kind, labels, sq, diag_ix)
            snd = model.normalized(s_diag)
            adv_d = torch.minimum(model.q1(torch.cat([snd, a_diag], -1)),
                                  model.q2(torch.cat([snd, a_diag], -1))) - model.vr(snd)
            w_d = actor_weights(adv_d, r_d, lam)
            action_stats = dict(action_mean_l2_vs_init=float(per_row.mean()),
                                action_max_l2_vs_init=float(per_row.max()),
                                action_p95_l2_vs_init=float(np.percentile(per_row.numpy(), 95)),
                                action_norm2_mean=float(a_now.square().sum(-1).mean()),
                                action_cost_mean=float(cost_now.mean()))
            w_stats = weight_metrics(w_d, torch.clamp(a_diag.square().sum(-1, keepdim=True) - 3.9079, min=0))
        # 诊断梯度必须在 no_grad 之外计算（需要 autograd 图）
        g_d = diag_gradient(model, s_diag, a_diag, w_d)
        grad_store[(batch_seed, lam, risk_kind, step)] = g_d
        act_store[(batch_seed, lam, risk_kind, step)] = a_now.numpy().copy()
        out_rows.append(dict(batch_seed=batch_seed, lambda_fixed=lam, risk=risk_kind, step=step,
                             diag_grad_norm=float(g_d.norm()),
                             **action_stats, **w_stats))

    for step in range(steps + 1):
        if step in RECORD_STEPS or step == steps:
            record(step)
        if step == steps:
            break
        ix = batches[step]
        s = torch.from_numpy(obs[ix]); a = torch.from_numpy(act[ix])
        with torch.no_grad():
            sn = model.normalized(s)
            adv = torch.minimum(model.q1(torch.cat([sn, a], -1)), model.q2(torch.cat([sn, a], -1))) - model.vr(sn)
            w = actor_weights(adv, risk_batch(risk_kind, labels, sq, ix), lam)
        mu, logstd = model.params(sn)
        az = torch.atanh(a.clamp(-.999999, .999999))
        logp = (-.5 * ((az - mu) / logstd.exp()).square() - logstd - .9189).sum(-1, keepdim=True)
        loss = -(w * logp).mean()
        if not torch.isfinite(loss):
            raise FloatingPointError(f'非有限演员损失 seed={batch_seed} λ={lam} 风险={risk_kind} step={step}')
        optim.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(model.演员.parameters(), CLIP_ACTOR)
        optim.step()
    qv1 = qv_fingerprint(model)
    final = {k: v.detach().numpy().copy() for k, v in model.演员.state_dict().items()}
    return qv0, qv1, init_actor_sha, actor_weights_bytes(model), batch_seq_sha, final


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--project', type=Path, default=Path('.'))
    ap.add_argument('--stage', choices=['selftest', 'check', 'pilot'], default='selftest')
    ap.add_argument('--steps', type=int, default=200)
    ap.add_argument('--out', type=Path, default=None)
    args = ap.parse_args()
    torch.set_num_threads(2)
    selftest()
    if args.stage == 'selftest':
        return
    project = args.project.resolve()
    meta, obs, act, sq, labels, ck = checks(project)
    print('CHECK PASS:', json.dumps(meta, ensure_ascii=False)[:400], flush=True)
    if args.stage == 'check':
        return
    if args.steps < 1 or args.steps > 2000:
        raise ValueError('steps 必须在 1~2000')
    out = (args.out or project / '论文核验输出_E2b受控演员传递_20261013').resolve()
    if out.exists():
        raise FileExistsError('输出目录已存在，拒绝覆盖: ' + str(out))
    out.mkdir(parents=True)
    t0 = time.time()
    rng = np.random.RandomState(20261013)
    eval_ix = rng.choice(len(obs), size=EVAL_N, replace=False)
    diag_ix = rng.choice(len(obs), size=DIAG_N, replace=False)
    rows, grads, acts = [], {}, {}
    ident_rows, final_actors = [], {}
    qv_proof, actor_hashes = [], {}
    for bs in BATCH_SEEDS:
        for lam in LAMBDAS:
            for rk in RISKS:
                qv0, qv1, init_sha, final_sha, bseq_sha, final_sd = run_branch(
                    ck, obs, act, sq, labels, rk, lam, bs, args.steps,
                    eval_ix, diag_ix, rows, grads, acts)
                ident_rows.append(dict(batch_seed=bs, lambda_fixed=lam, risk=rk,
                                       optimizer='Adam', actor_lr=LR,
                                       betas='(0.9, 0.999)', eps='1e-08',
                                       actor_grad_clip=CLIP_ACTOR, batch_size=BATCH, steps=args.steps,
                                       batch_sequence_sha256=bseq_sha,
                                       actor_init_sha256=init_sha, actor_final_sha256=final_sha,
                                       actor_param_count=int(sum(v.size for v in final_sd.values()))))
                if bs == BATCH_SEEDS[0]:
                    for k, v in final_sd.items():
                        final_actors[f'lam{lam}|{rk}|{k}'] = v
                qv_proof.append(dict(batch_seed=bs, lambda_fixed=lam, risk=rk,
                                     qv_sha_before=qv0, qv_sha_after=qv1, qv_unchanged=(qv0 == qv1)))
                actor_hashes[f'{bs}_{lam}_{rk}'] = final_sha
                if not (qv0 == qv1):
                    raise AssertionError(f'Q/V 被改动: seed={bs} λ={lam} {rk}')
        print('E2b 批次种子', bs, '完成', flush=True)
    # 梯度相似度：同 λ、同步、同批次种子下与 D10 比
    for r in rows:
        g = grads[(r['batch_seed'], r['lambda_fixed'], r['risk'], r['step'])]
        gd = grads[(r['batch_seed'], r['lambda_fixed'], 'D10', r['step'])]
        r['grad_cos_vs_D10'] = cos(g, gd)

    # 闸门 1：λ=0 四个分支必须数值一致（动作完全一样）
    gate0 = []
    for bs in BATCH_SEEDS:
        base = {r['step']: r for r in rows if r['batch_seed'] == bs and r['lambda_fixed'] == 0.0 and r['risk'] == 'D10'}
        for rk in RISKS:
            cur = {r['step']: r for r in rows if r['batch_seed'] == bs and r['lambda_fixed'] == 0.0 and r['risk'] == rk}
            for st in base:
                d = abs(base[st]['action_mean_l2_vs_init'] - cur[st]['action_mean_l2_vs_init'])
                gate0.append(dict(batch_seed=bs, risk=rk, step=st, action_mean_absdiff=d))
    max_l0 = max(g['action_mean_absdiff'] for g in gate0)
    if max_l0 > 1e-9:
        raise AssertionError(f'λ=0 负对照失败，动作差异 {max_l0}')
    with (out / '批次指标.csv').open('x', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    with (out / 'QV冻结证明.csv').open('x', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(qv_proof[0].keys())); w.writeheader(); w.writerows(qv_proof)
    with (out / 'lambda0负对照.csv').open('x', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(gate0[0].keys())); w.writeheader(); w.writerows(gate0)

    # 配对动作距离：直接比较分支之间在固定离线状态集上的确定性平均动作
    pair_rows = []
    for bs in BATCH_SEEDS:
        for lam in LAMBDAS:
            for st in RECORD_STEPS:
                ref = acts[(bs, lam, 'D10', st)]
                for rk in RISKS:
                    cur = acts[(bs, lam, rk, st)]
                    d = np.linalg.norm(cur - ref, axis=1)
                    pair_rows.append(dict(batch_seed=bs, lambda_fixed=lam, step=st,
                                          arm_a='D10', arm_b=rk,
                                          pair_action_mean_l2=float(d.mean()),
                                          pair_action_max_l2=float(d.max()),
                                          pair_action_p95_l2=float(np.percentile(d, 95)),
                                          cos_of_mean_actions=float(
                                              (cur.ravel() @ ref.ravel())
                                              / (np.linalg.norm(cur) * np.linalg.norm(ref) + 1e-12))))
            # 与 λ=0 安全基线比（同批次种子、同风险）
            for rk in RISKS:
                base0 = acts[(bs, 0.0, rk, st)]
                cur = acts[(bs, lam, rk, st)]
                d = np.linalg.norm(cur - base0, axis=1)
                pair_rows.append(dict(batch_seed=bs, lambda_fixed=lam, step=st,
                                      arm_a='λ=0', arm_b=rk,
                                      pair_action_mean_l2=float(d.mean()),
                                      pair_action_max_l2=float(d.max()),
                                      pair_action_p95_l2=float(np.percentile(d, 95)),
                                      cos_of_mean_actions=float(
                                          (cur.ravel() @ base0.ravel())
                                          / (np.linalg.norm(cur) * np.linalg.norm(base0) + 1e-12))))
        print('E2b 配对动作距离完成', bs, flush=True)
    with (out / '配对动作距离.csv').open('x', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(pair_rows[0].keys())); w.writeheader(); w.writerows(pair_rows)
    np.savez_compressed(out / '动作向量.npz',
                        **{f'bs{bs}|lam{lam}|{rk}|step{st}': v
                           for (bs, lam, rk, st), v in acts.items()})
    with (out / '分支身份与检查点.csv').open('x', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(ident_rows[0].keys())); w.writeheader(); w.writerows(ident_rows)
    np.savez_compressed(out / '步200演员检查点_批次种子20261013.npz', **final_actors)

    # 汇总：动作差异（相对各自第 0 步）与可重复性
    def act_delta(bs, lam, rk, st=200):
        for r in rows:
            if r['batch_seed'] == bs and r['lambda_fixed'] == lam and r['risk'] == rk and r['step'] == st:
                return r['action_mean_l2_vs_init']
        return None
    summary = dict(experiment='E2b stage-I controlled actor-only transfer pilot',
                   status='actor_only_NOT_full_training', input=meta,
                   steps=args.steps, batch_seeds=list(BATCH_SEEDS), lambdas=list(LAMBDAS), risks=list(RISKS),
                   batch_size=BATCH, actor_lr=LR, actor_grad_clip=CLIP_ACTOR, beta=BETA, weight_max=W_MAX,
                   eval_states=EVAL_N, diag_states=DIAG_N, record_steps=list(RECORD_STEPS),
                   gate_lambda0_max_absdiff=max_l0, qv_frozen_all=True,
                   independent_test_seeds_125000_125099_used=False,
                   action_mean_l2_step200={f'bs{bs}_lam{lam}_{rk}': act_delta(bs, lam, rk)
                                           for bs in BATCH_SEEDS for lam in LAMBDAS for rk in RISKS},
                   elapsed_sec=round(time.time() - t0, 1))
    with (out / '核验与边界.json').open('x', encoding='utf8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print('E2b PASS, output:', out, 'elapsed:', summary['elapsed_sec'], flush=True)


if __name__ == '__main__':
    main()
