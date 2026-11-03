#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""E2b 阶段 II：多种子 × 双演员起点 × 4 条批次序列的受控 actor-only 传递实验。

设计（按 2026-10-13 批准的阶段 II 安排）：
  原始训练种子 0/1/2 ｜ 演员起点 D10、NORM2 ｜ 随机批次序列 4 条
  主干预：D10 vs D10 的严格保序非线性变换 (P−min+0.1)²
  固定风险强度 λ=10（主）、λ=1（敏感性）；另设 λ=0 负对照
  更新 0/200/500 步；主评价 = 动作差异与跨批次波动
  主分支 3×2×4×2×2 = 96 条，λ=0 负对照 3×2×4×2 = 48 条，共 144 条 500 步 actor-only 分支。

边界：只更新演员；Q/V 完全冻结；不重构邻域、不改 q、不扫 ρ、不碰 125000–125099；不覆盖历史输出。
演员更新逐字复刻 P3 v1.1（权重 exp(3·z(z(adv)−λ·z(risk))) 归一化截断 100；
loss = −mean(w·logp_tanhGaussian)；Adam lr=3e-4；clip_grad_norm 5.0）。
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
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
EVAL_N = 4096
DIAG_N = 1024
STEPS = 500
RECORD_STEPS = (0, 200, 500)
LAMBDAS = (1.0, 10.0)
LAM0 = 0.0
RISKS = ('D10', 'D10_MONOTONE')
STARTS = ('D10', 'NORM2')
SEEDS = (0, 1, 2)
BATCH_SEEDS = (20261013, 20261014, 20261015, 20261016)


def sha256_bytes(b):
    return hashlib.sha256(b).hexdigest()


def file_sha256(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for blk in iter(lambda: f.read(1 << 20), b''):
            h.update(blk)
    return h.hexdigest()


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


def standardize(x):
    return (x - x.mean()) / (x.std() + 1e-6)


def actor_weights(adv, risk, lam):
    z_r = standardize(adv); z_p = standardize(risk)
    u = standardize(z_r - lam * z_p)
    w = torch.exp(BETA * u)
    return (w / w.mean().clamp_min(1e-6)).clamp(max=W_MAX)


def risk_batch(kind, label_d10, ix):
    if kind == 'D10':
        r = label_d10[ix]
    elif kind == 'D10_MONOTONE':
        base = label_d10[ix]
        r = np.square(base - base.min() + 0.1)
    else:
        raise ValueError(kind)
    return torch.from_numpy(np.ascontiguousarray(r, dtype=np.float32)).view(-1, 1)


def start_checkpoint(project, start, seed):
    if start == 'D10':
        return project / '输出_P3稳定IQL_DE端点' / f'模型_P3_D10_种子{seed}_40000步.pt'
    if start == 'NORM2':
        return project / '论文核验输出_P4动作范数平方对照' / f'模型_P4_NORM2_种子{seed}_40000步.pt'
    raise ValueError(start)


def checks(project):
    data = project / '每日结果' / 'Day20_路径二状态相关代价' / '数据_HalfCheetah_异质.npz'
    p3 = project / '输出_P3稳定IQL_DE端点'
    need = [data, p3 / '标签_P3_D10_原始几何.npz']
    for start in STARTS:
        for seed in SEEDS:
            need.append(start_checkpoint(project, start, seed))
    miss = [str(p) for p in need if not p.exists()]
    if miss:
        raise FileNotFoundError('缺少阶段 II 只读输入: ' + '; '.join(miss))
    if file_sha256(data) != DATA_SHA:
        raise ValueError('Day21 数据哈希不符')
    z = np.load(data, allow_pickle=False)
    obs = np.ascontiguousarray(z['obs'], dtype=np.float32)
    act = np.ascontiguousarray(z['act'], dtype=np.float32)
    if obs.shape != (200000, 17) or act.shape != (200000, 6):
        raise ValueError('数据维度不符')
    with np.load(p3 / '标签_P3_D10_原始几何.npz', allow_pickle=False) as v:
        lab = np.asarray(v['labels'], dtype=np.float32)
        if lab.shape != (200000,) or not np.isfinite(lab).all():
            raise ValueError('D10 标签异常')
    sq = np.sum(act.astype(np.float64) ** 2, axis=1).astype(np.float32)
    cks, ck_meta = {}, {}
    for start in STARTS:
        for seed in SEEDS:
            p = start_checkpoint(project, start, seed)
            ck = torch.load(p, map_location='cpu', weights_only=True)
            if ck.get('seed') != seed or ck.get('steps') != 40000:
                raise ValueError('检查点身份异常 ' + str(p))
            m = StablePolicy(); m.load_state_dict(ck['state_dict'], strict=True)
            cks[(start, seed)] = ck
            ck_meta[f'{start}_seed{seed}'] = dict(path=str(p), sha256=file_sha256(p), arm_field=ck.get('arm'))
    meta = dict(data=str(data), data_sha256=DATA_SHA, d10_label_mean=float(lab.mean()),
                checkpoints=ck_meta,
                grid=dict(starts=list(STARTS), seeds=list(SEEDS), batch_seeds=list(BATCH_SEEDS),
                          risks=list(RISKS), lambdas=list(LAMBDAS), steps=STEPS,
                          record_steps=list(RECORD_STEPS),
                          neg_control_lambda=LAM0),
                main_branches=len(STARTS) * len(SEEDS) * len(BATCH_SEEDS) * len(RISKS) * len(LAMBDAS),
                neg_control_branches=len(STARTS) * len(SEEDS) * len(BATCH_SEEDS) * len(RISKS))
    return meta, obs, act, sq, lab, cks


def selftest():
    torch.manual_seed(1)
    adv = torch.randn(256, 1); risk = torch.randn(256, 1)
    assert torch.allclose(actor_weights(adv, risk, 0.0), actor_weights(adv, risk * 3 + 7, 0.0), atol=2e-5)
    for lam in (1.0, 10.0):
        err = float((actor_weights(adv, risk, lam) - actor_weights(adv, 2 * risk + 3, lam)).abs().max())
        if err > 1e-3:
            raise AssertionError(f'正仿射不变性失败 {err}')
    sh = (risk - risk.min() + 0.1)
    assert float((actor_weights(adv, sh, 10.0) - actor_weights(adv, sh.square(), 10.0)).abs().max()) > 1e-4
    m = StablePolicy()
    s = torch.randn(32, 17); a = torch.rand(32, 6) * 2 - 1
    mu, ls = m.params(m.normalized(s))
    az = torch.atanh(a.clamp(-.999999, .999999))
    logp = (-.5 * ((az - mu) / ls.exp()).square() - ls - .9189).sum(-1, keepdim=True)
    (-(actor_weights(torch.randn(32, 1), torch.randn(32, 1), 1.0) * logp).mean()).backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.演员.parameters())
    print('E2b-II selftest PASS: λ=0 不变 / 正仿射不变 / 保序形状改变权重 / 演员梯度通路可用')


def fprint(model, which):
    h = hashlib.sha256()
    ps = (list(model.q1.parameters()) + list(model.q2.parameters()) + list(model.vr.parameters())
          if which == 'qv' else list(model.演员.parameters()))
    for p in ps:
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
    (-(w.detach() * logp).mean()).backward()
    g = torch.cat([p.grad.detach().flatten() for p in model.演员.parameters()])
    for p in model.演员.parameters():
        p.grad = None
    return g


def cos(x, y):
    d = float(x.norm() * y.norm())
    return float((x @ y) / d) if d > 1e-12 else float('nan')


def run_branch(ck, label_d10, obs, act, sq, risk_kind, lam, batch_seed, eval_ix, diag_ix):
    """跑一条分支，返回 (qv前, qv后, 初始演员hash, 末态演员hash, 批次序列hash, 行, 梯度, 各步动作向量, 末态演员参数字典)。"""
    model = StablePolicy()
    model.load_state_dict(ck['state_dict'], strict=True)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    for p in model.演员.parameters():
        p.requires_grad_(True)
    qv0 = fprint(model, 'qv'); init_sha = fprint(model, 'actor')
    optim = torch.optim.Adam(model.演员.parameters(), lr=LR)
    s_eval = torch.from_numpy(obs[eval_ix])
    with torch.no_grad():
        a_eval0 = model.均值动作(s_eval).clone()
    s_diag = torch.from_numpy(obs[diag_ix]); a_diag = torch.from_numpy(act[diag_ix])
    rng = np.random.RandomState(batch_seed)
    batches = [rng.randint(0, len(obs), size=BATCH) for _ in range(STEPS)]
    bseq_sha = sha256_bytes(np.ascontiguousarray(np.stack(batches)).tobytes())
    rows, grads, vecs = [], {}, {}

    def record(step):
        with torch.no_grad():
            a_now = model.均值动作(s_eval)
            per_row = (a_now - a_eval0).norm(dim=1)
            cost_now = torch.clamp(a_now.square().sum(-1) - 3.9079, min=0)
            r_d = risk_batch(risk_kind, label_d10, diag_ix)
            snd = model.normalized(s_diag)
            adv_d = torch.minimum(model.q1(torch.cat([snd, a_diag], -1)),
                                  model.q2(torch.cat([snd, a_diag], -1))) - model.vr(snd)
            w_d = actor_weights(adv_d, r_d, lam)
            ast = dict(action_mean_l2_vs_init=float(per_row.mean()),
                       action_max_l2_vs_init=float(per_row.max()),
                       action_p95_l2_vs_init=float(np.percentile(per_row.numpy(), 95)),
                       action_norm2_mean=float(a_now.square().sum(-1).mean()),
                       action_cost_mean=float(cost_now.mean()))
            wst = weight_metrics(w_d, torch.clamp(a_diag.square().sum(-1, keepdim=True) - 3.9079, min=0))
            vecs[step] = a_now.numpy().copy()
            actor_sha_here = fprint(model, 'actor')
        g_d = diag_gradient(model, s_diag, a_diag, w_d)
        grads[step] = g_d
        rows.append(dict(batch_seed=batch_seed, lambda_fixed=lam, risk=risk_kind, step=step,
                         actor_sha256_at_step=actor_sha_here, diag_grad_norm=float(g_d.norm()),
                         **ast, **wst))

    for step in range(STEPS + 1):
        if step in RECORD_STEPS:
            record(step)
        if step == STEPS:
            break
        ix = batches[step]
        s = torch.from_numpy(obs[ix]); a = torch.from_numpy(act[ix])
        with torch.no_grad():
            sn = model.normalized(s)
            adv = torch.minimum(model.q1(torch.cat([sn, a], -1)), model.q2(torch.cat([sn, a], -1))) - model.vr(sn)
            w = actor_weights(adv, risk_batch(risk_kind, label_d10, ix), lam)
        mu, logstd = model.params(sn)
        az = torch.atanh(a.clamp(-.999999, .999999))
        logp = (-.5 * ((az - mu) / logstd.exp()).square() - logstd - .9189).sum(-1, keepdim=True)
        loss = -(w * logp).mean()
        if not torch.isfinite(loss):
            raise FloatingPointError(f'非有限损失 bs={batch_seed} λ={lam} {risk_kind} step={step}')
        optim.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(model.演员.parameters(), CLIP_ACTOR)
        optim.step()
    final_sd = {k: v.detach().numpy().copy() for k, v in model.演员.state_dict().items()}
    return qv0, fprint(model, 'qv'), init_sha, fprint(model, 'actor'), bseq_sha, rows, grads, vecs, final_sd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--project', type=Path, default=Path('.'))
    ap.add_argument('--stage', choices=['selftest', 'check', 'run'], default='selftest')
    ap.add_argument('--out', type=Path, default=None)
    args = ap.parse_args()
    torch.set_num_threads(2)
    selftest()
    if args.stage == 'selftest':
        return
    project = args.project.resolve()
    meta, obs, act, sq, lab, cks = checks(project)
    print('CHECK PASS:', json.dumps(meta, ensure_ascii=False)[:400], flush=True)
    if args.stage == 'check':
        return
    out = (args.out or project / '论文核验输出_E2b阶段II_多起点多种子_20261013').resolve()
    if out.exists():
        raise FileExistsError('输出目录已存在，拒绝覆盖: ' + str(out))
    out.mkdir(parents=True)
    t0 = time.time()
    rng = np.random.RandomState(20261013)
    eval_ix = rng.choice(len(obs), size=EVAL_N, replace=False)
    diag_ix = rng.choice(len(obs), size=DIAG_N, replace=False)
    all_rows, pair_rows, ident_rows, qv_rows, neg_rows = [], [], [], [], []
    actors_500, vecs_500, vecs_0 = {}, {}, {}
    for start in STARTS:
        for seed in SEEDS:
            ck = cks[(start, seed)]
            for bs in BATCH_SEEDS:
                局部 = {}
                for lam in (LAM0,) + LAMBDAS:
                    for rk in RISKS:
                        (q0, q1_, isha, fsha, bsha, rws, grs, vcs, fsd) = run_branch(
                            ck, lab, obs, act, sq, rk, lam, bs, eval_ix, diag_ix)
                        局部[(lam, rk)] = dict(rows=rws, grads=grs, vecs=vcs, init=isha, final=fsha)
                        ident_rows.append(dict(start=start, seed=seed, batch_seed=bs, lambda_fixed=lam,
                                               risk=rk, optimizer='Adam', actor_lr=LR, betas='(0.9, 0.999)',
                                               eps='1e-08', actor_grad_clip=CLIP_ACTOR, batch_size=BATCH,
                                               steps=STEPS, batch_sequence_sha256=bsha,
                                               actor_init_sha256=isha, actor_final_sha256=fsha))
                        qv_rows.append(dict(start=start, seed=seed, batch_seed=bs, lambda_fixed=lam,
                                            risk=rk, qv_sha_before=q0, qv_sha_after=q1_,
                                            qv_unchanged=(q0 == q1_)))
                        if q0 != q1_:
                            raise AssertionError('Q/V 被改动')
                        vecs_0.setdefault((start, seed, bs), vcs[0])
                        if lam != LAM0:
                            actors_500[f'{start}|seed{seed}|bs{bs}|lam{lam}|{rk}'] = fsd
                            vecs_500[f'{start}|seed{seed}|bs{bs}|lam{lam}|{rk}'] = vcs[500]
                # 梯度余弦 + 行落表（同 λ、同步下与 D10 比）
                for (lam, rk), pack in 局部.items():
                    for i, r in enumerate(pack['rows']):
                        g = pack['grads'][r['step']]
                        gd = 局部[(lam, 'D10')]['grads'][r['step']]
                        r['grad_cos_vs_D10'] = cos(g, gd)
                        r['start'] = start; r['seed'] = seed
                        all_rows.append(r)
                # 配对动作距离 + λ=0 负对照
                for st in RECORD_STEPS:
                    for lam in (LAM0,) + LAMBDAS:
                        ref = 局部[(lam, 'D10')]['vecs'][st]
                        for rk in RISKS:
                            cur = 局部[(lam, rk)]['vecs'][st]
                            dd = np.linalg.norm(cur - ref, axis=1)
                            pair_rows.append(dict(start=start, seed=seed, batch_seed=bs, lambda_fixed=lam,
                                                  step=st, arm_a='D10', arm_b=rk,
                                                  pair_action_mean_l2=float(dd.mean()),
                                                  pair_action_max_l2=float(dd.max()),
                                                  pair_action_p95_l2=float(np.percentile(dd, 95))))
                        if lam != LAM0:
                            base0 = 局部[(LAM0, 'D10')]['vecs'][st]
                            cur = 局部[(lam, 'D10')]['vecs'][st]
                            dd = np.linalg.norm(cur - base0, axis=1)
                            pair_rows.append(dict(start=start, seed=seed, batch_seed=bs, lambda_fixed=lam,
                                                  step=st, arm_a='lam0_baseline', arm_b='D10',
                                                  pair_action_mean_l2=float(dd.mean()),
                                                  pair_action_max_l2=float(dd.max()),
                                                  pair_action_p95_l2=float(np.percentile(dd, 95))))
                    a0 = 局部[(LAM0, 'D10')]['vecs'][st]
                    b0 = 局部[(LAM0, 'D10_MONOTONE')]['vecs'][st]
                    neg_rows.append(dict(start=start, seed=seed, batch_seed=bs, step=st,
                                         max_absdiff=float(np.abs(a0 - b0).max())))
            print(f'E2b-II {start} seed{seed} 完成  已用 {(time.time()-t0)/60:.1f} 分钟', flush=True)
    maxneg = max(r['max_absdiff'] for r in neg_rows)
    if maxneg > 1e-9:
        raise AssertionError(f'λ=0 负对照失败 {maxneg}')
    if not all(r['qv_unchanged'] for r in qv_rows):
        raise AssertionError('存在 Q/V 被改动的分支')
    # 跨批次波动汇总（主评价）
    wob = []
    for start in STARTS:
        for seed in SEEDS:
            for lam in LAMBDAS:
                for rk in RISKS:
                    for st in RECORD_STEPS:
                        v = [r['action_mean_l2_vs_init'] for r in all_rows
                             if r['start'] == start and r['seed'] == seed and r['lambda_fixed'] == lam
                             and r['risk'] == rk and r['step'] == st]
                        wob.append(dict(start=start, seed=seed, lambda_fixed=lam, risk=rk, step=st,
                                        n_batches=len(v), mean=float(np.mean(v)),
                                        sd=float(np.std(v, ddof=1)) if len(v) > 1 else 0.0,
                                        min=float(np.min(v)), max=float(np.max(v))))
    with (out / '批次指标.csv').open('x', encoding='utf-8-sig', newline='') as f:
        cols = ['start', 'seed', 'batch_seed', 'lambda_fixed', 'risk', 'step', 'actor_sha256_at_step',
                'action_mean_l2_vs_init', 'action_max_l2_vs_init', 'action_p95_l2_vs_init',
                'action_norm2_mean', 'action_cost_mean', 'ess', 'weight_p50', 'weight_p95',
                'clip_fraction', 'weight_top_cost_share', 'weighted_true_cost', 'diag_grad_norm',
                'grad_cos_vs_D10']
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader(); w.writerows(all_rows)
    for fn, data in (('配对动作距离.csv', pair_rows), ('分支身份与检查点.csv', ident_rows),
                     ('QV冻结证明.csv', qv_rows), ('lambda0负对照.csv', neg_rows),
                     ('跨批次波动汇总.csv', wob)):
        with (out / fn).open('x', encoding='utf-8-sig', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(data[0].keys())); w.writeheader(); w.writerows(data)
    np.savez_compressed(out / '动作向量_步500.npz', **vecs_500)
    np.savez_compressed(out / '动作向量_初始_各起点种子批次.npz',
                        **{f'{s}|seed{sd}|bs{bs}': v for (s, sd, bs), v in vecs_0.items()})
    np.savez_compressed(out / '步500演员检查点.npz', **actors_500)
    summary = dict(experiment='E2b stage-II multi-seed multi-start controlled actor-only transfer',
                   status='actor_only_NOT_full_training', input=meta,
                   steps=STEPS, record_steps=list(RECORD_STEPS), batch_size=BATCH,
                   actor_lr=LR, actor_grad_clip=CLIP_ACTOR, beta=BETA, weight_max=W_MAX,
                   eval_states=EVAL_N, diag_states=DIAG_N,
                   gate_lambda0_max_absdiff=maxneg,
                   qv_frozen_all=bool(all(r['qv_unchanged'] for r in qv_rows)),
                   independent_test_seeds_125000_125099_used=False,
                   archival_note='步0/200/500 的演员 SHA 记在 批次指标.csv；'
                                 '实际检查点只存档步500（0/200 可由同一确定性脚本复算）',
                   limits=['actor-only; no environment rollout',
                           'no claim about trajectory return/cost/CVaR',
                           'D10 vs NORM2 starting points are NOT a paired comparison',
                           'D10 label = joint state-action neighbourhood; E1 quantity = state-only 50NN'],
                   elapsed_sec=round(time.time() - t0, 1))
    with (out / '核验与边界.json').open('x', encoding='utf8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print('E2b-II PASS, output:', out, 'elapsed:', summary['elapsed_sec'], flush=True)


if __name__ == '__main__':
    main()
