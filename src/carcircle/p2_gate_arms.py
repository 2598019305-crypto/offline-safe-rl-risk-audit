#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""P2：计算 B3（仅数据支持门控）与 B4（完整门控）在全部训练样本上的标签。

判据（在看结果前冻结，见 ABLATION_CONFIG_LOCK.json）：
  候选级公共条件：联合支撑 ≥5 且行为风险有限
  B3（support-only）：候选优势损失 ≤0.25σ_k（不检查风险下降门槛）
  B4（完整门控）    ：再要求风险改善差 ≥0.10
风险定义：留一轨迹口径（gates_cache.npz 使用的同一去泄漏定义），保证与既有门控可比。

输出 mech_audit/gates_p2.npz：gate_B3_seed{k}、gate_B4_seed{k}，并与既有
gates_per_seed_v2.npz 的 B4 对照，作为独立复算校验。
"""
from __future__ import annotations

import json
import math
import sys
import time

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import iql          # noqa: E402
import night_lib as L  # noqa: E402
from scipy.spatial import cKDTree

NB, MAXC, MINS, MING, RND = 64, 16, 5, 0.10, 3
OUT = L.WORK / 'agrw'


class Shim:
    def __call__(self, m):
        print('  [risk]', m, flush=True)

    def sync(self):
        pass


def tail_mean(c, tail=0.10):
    take = max(1, int(math.ceil(tail * len(c))))
    return float(np.partition(c, -take)[-take:].mean())


def main():
    t0 = time.time()
    data = iql.load_and_split(L.ENV_A)
    risk = iql.compute_risk_proxies(data, Shim())
    tr = risk['train_idx'].astype(np.int64)
    radius = float(risk['meta']['radius'])
    traj_of = np.full(len(data['S']), -1, np.int32)
    for tid, (s, e) in enumerate(data['bounds']):
        traj_of[s:e] = tid
    S, A, C = data['S'], data['A'], data['C']
    sm, ss = S[tr].mean(0), np.maximum(S[tr].std(0), 1e-8)
    am, asd = A[tr].mean(0), np.maximum(A[tr].std(0), 1e-8)
    sf = (S[tr] - sm) / ss
    feat = np.concatenate([sf, 20.0 * (A[tr] - am) / asd], axis=-1)
    tree, stree = cKDTree(feat), cKDTree(sf)
    cost_tr, traj_tr = C[tr], traj_of[tr]
    critics, sd3 = [], None

    def adv_all3(states, actions):
        with torch.no_grad():
            s = (torch.from_numpy(states) - om) / osd
            a = torch.from_numpy(actions)
            return [(torch.minimum(q1(torch.cat([s, a], -1)), q2(torch.cat([s, a], -1)))
                     - v(s)).reshape(-1).numpy().astype(np.float64) for q1, q2, v in critics]

    for k in (0, 1, 2):
        ck = torch.load(L.WORK / 'checkpoints' / ('P1_seed%d' % k) / 'critic_step40000.pt',
                        map_location='cpu', weights_only=False)
        q1, q2, v = iql.mlp(10, 1), iql.mlp(10, 1), iql.mlp(8, 1)
        q1.load_state_dict(ck['q1']); q2.load_state_dict(ck['q2']); v.load_state_dict(ck['v'])
        for m in (q1, q2, v):
            m.eval()
            for p in m.parameters():
                p.requires_grad_(False)
        critics.append((q1, q2, v))
    om, osd = torch.from_numpy(data['obs_mean']), torch.from_numpy(data['obs_std'])
    sd3 = [float(a.std()) for a in adv_all3(S[tr], A[tr])]
    n = len(tr)
    d10_beh = np.load(OUT / 'gates_cache.npz')['d10_beh']     # 留一轨迹口径
    gateB3 = np.zeros((3, n), bool)
    gateB4 = np.zeros((3, n), bool)
    CHUNK = 1024
    for c0 in range(0, n, CHUNK):
        c1 = min(n, c0 + CHUNK)
        m = c1 - c0
        nn = stree.query(sf[c0:c1], k=NB + 1, workers=4)[1][:, 1:]
        ci, ca, cb = [], [], []
        for li in range(m):
            acts = np.round(A[tr[nn[li]]], RND)
            uniq, inv = np.unique(acts, axis=0, return_inverse=True)
            own_a = np.round(A[tr[c0 + li]], RND)
            own_t = int(traj_tr[c0 + li])
            for k in range(min(len(uniq), MAXC)):
                if np.allclose(uniq[k], own_a):
                    continue
                ci.append(li); ca.append(uniq[k])
                cb.append({own_t} | {int(traj_tr[nn[li][t]]) for t in np.where(inv == k)[0]})
        K = len(ci)
        if K == 0:
            continue
        qpts = np.empty((m + K, feat.shape[1]))
        qpts[:m] = feat[c0:c1]
        cac = np.asarray(ca)
        qpts[m:] = np.concatenate([sf[[c0 + i for i in ci]], 20.0 * (cac - am) / asd], axis=-1)
        nbs = tree.query_ball_point(qpts, radius, workers=4)
        for li in range(m):     # 行为风险（留一轨迹口径）已由缓存给出，这里只补缺失
            pass
        kept = []               # (li, cand_action, support, gap, risk_finite)
        for j in range(K):
            nb = np.asarray(nbs[m + j], np.intp)
            ba = np.fromiter(cb[j], np.int32, len(cb[j]))
            keep = ~np.isin(traj_tr[nb], ba)
            sup = int(keep.sum())
            if sup < MINS:
                continue
            li = ci[j]
            rb = d10_beh[c0 + li]
            if not np.isfinite(rb):
                continue
            gap = rb - tail_mean(cost_tr[nb[keep]])
            kept.append((li, ca[j], sup, gap, np.isfinite(rb)))
        if not kept:
            continue
        sts = np.stack([S[tr[c0 + li]] for li, _, _, _, _ in kept])
        acs = np.stack([c for _, c, _, _, _ in kept]).astype(np.float32)
        ab = adv_all3(sts, np.stack([A[tr[c0 + li]] for li, _, _, _, _ in kept]).astype(np.float32))
        ac = adv_all3(sts, acs)
        for t, (li, c, sup, gap, fin) in enumerate(kept):
            for k in range(3):
                if ab[k][t] - ac[k][t] <= 0.25 * sd3[k]:
                    gateB3[k, c0 + li] = True
                    if gap >= MING:
                        gateB4[k, c0 + li] = True
        if (c0 // CHUNK) % 25 == 0:
            print('  进度 %d/%d（%.1f%%）%.0fs  B3率=%.4f B4率=%.4f'
                  % (c1, n, 100 * c1 / n, time.time() - t0, gateB3[0].mean(), gateB4[0].mean()),
                  flush=True)
    np.savez_compressed(L.WORK / 'agrw' / 'gates_p2.npz', train_idx=tr,
                        **{'gate_B3_seed%d' % k: gateB3[k] for k in range(3)},
                        **{'gate_B4_seed%d' % k: gateB4[k] for k in range(3)},
                        adv_sd=np.asarray(sd3))
    # 与既有 B4（gates_per_seed_v2）对照
    ref = np.load(L.WORK / 'agrw' / 'gates_per_seed_v2.npz')
    agree = {}
    for k in range(3):
        r = ref['gate_seed%d' % k]
        agree['seed%d' % k] = dict(b4_rate=float(gateB4[k].mean()), ref_rate=float(r.mean()),
                                   agreement=float((gateB4[k] == r).mean()),
                                   b3_rate=float(gateB3[k].mean()),
                                   b3_minus_b4=float(gateB3[k].mean() - gateB4[k].mean()))
    res = dict(adv_sd=sd3, note='留一轨迹风险口径；B3=支持+有限+优势；B4=B3+风险改善差≥0.10',
               per_seed=agree, seconds=round(time.time() - t0, 1))
    (L.WORK / 'mech_audit' / 'P2_GATE_ARMS.json').write_text(
        json.dumps(res, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(res, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
