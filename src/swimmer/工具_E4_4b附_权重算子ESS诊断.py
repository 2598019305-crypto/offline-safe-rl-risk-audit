#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""E4-4b 附：只读诊断——演员优势权重算子的有效样本量（ESS）从哪里来。

E4-4b 报告 batch_ess_mean ≈ 1.78（批大小 256）。本脚本区分两种互斥解释：
  (A) 数据缺乏动作级优势信息，导致权重无法分散；
  (B) 算子本身（β=3 的双重标准化 softmax × 批大小）在数学上就必然把 ESS 压到约 2。

做法：只读。使用同一份冻结 32k 数据（哈希硬校验）与同一 E4-3 Q/V 2000 步检查点，
复现 E4-4/E4-4b 的批次抽样顺序（RandomState(20261017)），
  1) 扫描 β ∈ {0.5,1,2,3,5}（批 256）；
  2) 扫描批大小 ∈ {32,64,128,256,512,1024,2048}（β=3）；
  3) 以同批大小的合成标准正态优势作为“算子内在 ESS”对照；
  4) 报告权重落到回合前 40 步的份额（这一项才是真正由数据决定的）。

不训练演员、不回放环境、不触碰 125000-125099、不写入任何历史结果目录。
"""
from __future__ import annotations
import argparse, csv, hashlib, io, json, zipfile
from pathlib import Path
import numpy as np
import torch
from torch import nn

SHA_E43 = '7949c493745923e28f67bb0a21586e4f52f889d594bc60dc2e49993c38840686'
SHA_DATA = 'e0bbcb750bba27daf13f5d13d411ac22fedcc2c49f46603af03b38a187120b9c'
BATCH = 256
CLIP = 10.0          # 与 E4-4/E4-4b 保持一致（用于可复现说明，未参与有效计算）
GAMMA = 0.99
CAP = 100.0
RNG_SEED = 20261017


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def exact_member(z: zipfile.ZipFile, suffix: str) -> bytes:
    names = [n for n in z.namelist() if n.endswith(suffix)]
    if len(names) != 1:
        raise ValueError(f'Archive expected unique member {suffix}: {names}')
    return z.read(names[0])


def pt_from_zip(z: zipfile.ZipFile, suffix: str):
    obj = torch.load(io.BytesIO(exact_member(z, suffix)), map_location='cpu', weights_only=True)
    return obj['state_dict']


def mlp(i, o, h):
    return nn.Sequential(nn.Linear(i, h), nn.Tanh(), nn.Linear(h, h), nn.Tanh(), nn.Linear(h, o))


def dumpcsv(path: Path, rows):
    if not rows:
        return
    fields = []
    for r in rows:
        for k in r:
            if k not in fields:
                fields.append(k)
    with path.open('x', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields, restval='')
        w.writeheader()
        w.writerows(rows)


def actor_weights(adv_z: np.ndarray, beta: float) -> np.ndarray:
    """复现论文/E4-4 的演员权重算子：双重标准化后 β·softmax，上限 100，再归一。"""
    lw = beta * adv_z
    p = np.exp(lw - lw.max())
    p = p / p.mean()
    p = np.minimum(p, CAP)
    return p / p.sum()


def ess(p: np.ndarray) -> float:
    return float(1.0 / np.sum(p * p))


def zscore(x: np.ndarray) -> np.ndarray:
    s = x.std()
    if s <= 0:
        return np.zeros_like(x)
    return (x - x.mean()) / s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--e4-3-zip', required=True, type=Path)
    ap.add_argument('--dataset', required=True, type=Path)
    ap.add_argument('--out', required=True, type=Path)
    ap.add_argument('--reps', type=int, default=2000, help='批次抽样次数')
    args = ap.parse_args()

    if sha(args.e4_3_zip).lower() != SHA_E43:
        raise RuntimeError('E4-3 ZIP SHA mismatch')
    got = sha(args.dataset)
    if got != SHA_DATA:
        raise RuntimeError('32k dataset SHA mismatch: ' + got)
    if args.out.exists():
        raise FileExistsError(f'Output exists, refusing overwrite: {args.out}')
    args.out.mkdir(parents=True)

    torch.set_num_threads(4)
    with zipfile.ZipFile(args.e4_3_zip) as z3:
        ck = pt_from_zip(z3, '模型_E4_3_QV_2000步.pt')
    sub = lambda p: {k[len(p):]: v for k, v in ck.items() if k.startswith(p)}
    data = np.load(args.dataset, allow_pickle=False)
    S = np.asarray(data['obs'], np.float32)
    A = np.asarray(data['act'], np.float32)
    t = np.asarray(data['t'])
    N = len(S)

    no, na = S.shape[1], A.shape[1]
    q1, q2, v = mlp(no + na, 1, 256), mlp(no + na, 1, 256), mlp(no, 1, 256)
    q1.load_state_dict(sub('q1.')); q2.load_state_dict(sub('q2.')); v.load_state_dict(sub('vr.'))
    q1.eval(); q2.eval(); v.eval()
    mean, std = ck['obs_mean'], ck['obs_std']
    qs, vs = [], []
    with torch.no_grad():
        for off in range(0, N, 4096):
            x = torch.from_numpy(S[off:off + 4096]); a = torch.from_numpy(A[off:off + 4096])
            ns = (x - mean) / std
            q = torch.minimum(q1(torch.cat([ns, a], -1)), q2(torch.cat([ns, a], -1))).flatten()
            qs.append(q.numpy()); vs.append(v(ns).flatten().numpy())
    Q = np.concatenate(qs).astype(np.float64)
    V = np.concatenate(vs).astype(np.float64)
    Adv = Q - V

    early = t < 40
    mid = (t >= 40) & (t < 100)
    print(f'数据 {N} 条；回合前40步占比 {early.mean():.4%}；40-99步 {mid.mean():.4%}', flush=True)
    print(f'Adv 均值 {Adv.mean():.4f} 标准差 {Adv.std():.4f}；'
          f'前40步内 Adv 均值 {Adv[early].mean():.4f}，100步后 {Adv[t >= 100].mean():.4f}', flush=True)

    rng = np.random.RandomState(RNG_SEED)

    def run_config(beta: float, bsz: int, reps: int):
        r = np.random.RandomState(RNG_SEED)
        e_list, early_share, argmax_early, top1_share, r2_early = [], [], [], [], []
        for _ in range(reps):
            idx = r.choice(N, bsz, replace=False)
            z = zscore(Adv[idx])
            p = actor_weights(z, beta)
            e_list.append(ess(p))
            early_share.append(float(p[early[idx]].sum()))
            argmax_early.append(bool(early[idx][int(np.argmax(p))]))
            top1_share.append(float(p.max()))
            if early[idx].any() and (~early[idx]).any():
                r2_early.append(float(np.corrcoef(z, early[idx].astype(float))[0, 1] ** 2))
        return dict(
            beta=beta, batch=int(bsz), reps=reps,
            ess_mean=float(np.mean(e_list)), ess_median=float(np.median(e_list)),
            ess_p90=float(np.percentile(e_list, 90)),
            weight_share_t_lt_40_mean=float(np.mean(early_share)),
            argmax_is_early_frac=float(np.mean(argmax_early)),
            top1_weight_mean=float(np.mean(top1_share)),
            r2_advantage_vs_early_mean=float(np.mean(r2_early)) if r2_early else None,
        )

    rows = []
    print('\n[1] β 扫描（批 256，真实数据）', flush=True)
    for beta in (0.5, 1.0, 2.0, 3.0, 5.0):
        rec = run_config(beta, BATCH, args.reps)
        rows.append(rec)
        print(f'  β={beta:<4} ESS={rec["ess_mean"]:.3f}  '
              f'权重落到前40步={rec["weight_share_t_lt_40_mean"]:.4%}  '
              f'权重最大样本占比={rec["top1_weight_mean"]:.3f}', flush=True)

    print('\n[2] 批大小扫描（β=3，真实数据）', flush=True)
    for bsz in (32, 64, 128, 256, 512, 1024, 2048):
        rec = run_config(3.0, bsz, args.reps)
        rows.append(rec)
        print(f'  B={bsz:<5} ESS={rec["ess_mean"]:.3f}  '
              f'权重落到前40步={rec["weight_share_t_lt_40_mean"]:.4%}', flush=True)

    synth = []
    print('\n[3] 合成对照：同批大小的标准正态优势（算子内在 ESS，与数据无关）', flush=True)
    for beta in (0.5, 1.0, 2.0, 3.0, 5.0):
        r = np.random.RandomState(RNG_SEED + 1)
        vals = []
        for _ in range(args.reps):
            vals.append(ess(actor_weights(zscore(r.randn(BATCH)), beta)))
        synth.append(dict(kind='synthetic_N01', beta=beta, batch=BATCH,
                          reps=args.reps, ess_mean=float(np.mean(vals)),
                          ess_median=float(np.median(vals)), ess_p90=float(np.percentile(vals, 90))))
        print(f'  β={beta:<4} 合成 ESS={np.mean(vals):.3f}', flush=True)
    for bsz in (64, 256, 1024):
        r = np.random.RandomState(RNG_SEED + 2)
        vals = []
        for _ in range(args.reps):
            vals.append(ess(actor_weights(zscore(r.randn(bsz)), 3.0)))
        synth.append(dict(kind='synthetic_N01', beta=3.0, batch=bsz,
                          reps=args.reps, ess_mean=float(np.mean(vals)),
                          ess_median=float(np.median(vals)), ess_p90=float(np.percentile(vals, 90))))
        print(f'  B={bsz:<5} 合成 ESS={np.mean(vals):.3f}（β=3）', flush=True)

    rows2 = []
    print('\n[4] 稳定区对照：只在 100 步之后的样本内部做标准化与加权（β=3）', flush=True)
    for beta in (1.0, 3.0):
        r = np.random.RandomState(RNG_SEED)
        vals, shares = [], []
        for _ in range(args.reps):
            idx = r.choice(N, BATCH, replace=False)
            idx = idx[t[idx] >= 100]
            if len(idx) < 16:
                continue
            p = actor_weights(zscore(Adv[idx]), beta)
            vals.append(ess(p))
            shares.append(float(len(idx)))
        rec = dict(region='t_ge_100_only', beta=beta, batch=BATCH,
                   n_mean=float(np.mean(shares)),
                   ess_mean=float(np.mean(vals)), ess_median=float(np.median(vals)))
        rows2.append(rec)
        print(f'  β={beta:<4} 仅稳定区样本 ESS={np.mean(vals):.3f}', flush=True)

    summary = dict(
        dataset_sha256=SHA_DATA, e4_3_zip_sha256=SHA_E43, batch=BATCH, reps=args.reps,
        intent='诊断 ESS≈1.78 来自数据还是来自算子（只读）',
        pool_size=int(N),
        early40_sample_share=float(early.mean()),
        advantage_mean=float(Adv.mean()), advantage_sd=float(Adv.std()),
        advantage_mean_early40=float(Adv[early].mean()),
        advantage_mean_t_ge_100=float(Adv[t >= 100].mean()),
        beta_sweep=[r for r in rows if r['batch'] == BATCH and r['beta'] in (0.5, 1.0, 2.0, 3.0, 5.0)],
        batch_sweep=[r for r in rows if r['beta'] == 3.0 and r['batch'] != BATCH],
        synthetic_intrinsic=synth,
        stable_region_only=rows2,
    )
    (args.out / '诊断_权重算子ESS.json').write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    dumpcsv(args.out / 'ESS扫描_真实数据.csv', rows)
    dumpcsv(args.out / 'ESS扫描_合成对照与稳定区.csv', synth + rows2)
    print('\nSUCCESS:', args.out, flush=True)


if __name__ == '__main__':
    main()
