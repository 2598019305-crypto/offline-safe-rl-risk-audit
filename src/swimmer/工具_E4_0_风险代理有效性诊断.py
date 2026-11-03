#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""E4-0：Swimmer 二值速度成本下的风险代理有效性诊断（不训练任何网络）。

检查：D10 饱和率与有效排序、E100/D10 关系、局部成本事件比例分布、
状态-动作邻域支撑度、以及"独立轨迹上是否仍有风险信息"（含简单基线对照）。
几何沿用 P3 的口径：标准化 obs 与 act（动作权重 20）拼成邻域特征，半径取 k 近邻距离中位数，
参照库只由 14 条轨迹构成，其余 6 条轨迹作为独立查询。
"""
from __future__ import annotations

import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np

DATA_SHA = '4edffe8b0178c1220a3d158521cbeafccd2ca479dac07fa8e38430d6086202fa'
ACTION_WEIGHT = 20.0
WINDOW = 10.0          # 邻域半径分位数用的 k
REF_TRAJ = 14


def main():
    project = Path(__file__).resolve().parent
    npz = project / '论文核验输出_E3e_Swimmer离线数据pilot_20261013' / '数据_SwimmerVelocity_自采pilot.npz'
    h = hashlib.sha256(npz.read_bytes()).hexdigest()
    print('数据 SHA256 =', h)
    assert h == DATA_SHA, '数据 SHA256 与放行文件不一致，停止'
    out = project / '论文核验输出_E4_0风险代理有效性诊断_20261013'
    if out.exists():
        raise FileExistsError('输出目录已存在: ' + str(out))
    out.mkdir(parents=True)
    t0 = time.time()
    z = np.load(npz)
    obs, act, cost, nxt, done = z['obs'].astype(np.float64), z['act'].astype(np.float64), z['cost'], z['next_obs'].astype(np.float64), z['done'].astype(bool)
    ep = z['episode']
    轨迹数 = int(ep.max() + 1)
    res = dict(data_sha256=h, n_steps=int(len(obs)), n_traj=轨迹数, cost_pos_rate=float((cost > 0).mean()))
    # 时间对齐复核
    对齐 = []
    for k in range(轨迹数):
        m = np.where(ep == k)[0]
        if len(m) > 1:
            对齐.append(bool(np.array_equal(nxt[m[:-1]], obs[m[1:]])))
    res['next_obs_alignment_all_true'] = bool(all(对齐)) and len(对齐) == 轨迹数
    # 邻域参照库只用前 REF_TRAJ 条
    ref_m = ep < REF_TRAJ
    q_m = ~ref_m
    om, os_ = obs[ref_m].mean(0), np.maximum(obs[ref_m].std(0), 1e-8)
    am, asd = act[ref_m].mean(0), np.maximum(act[ref_m].std(0), 1e-8)
    def feat(o, a):
        return np.concatenate([(o - om) / os_, ACTION_WEIGHT * (a - am) / asd], axis=-1)
    Fref = feat(obs[ref_m], act[ref_m]); Fq = feat(obs[q_m], act[q_m])
    from scipy.spatial import cKDTree
    tree = cKDTree(Fref)
    k = min(500, len(Fref) - 1)
    d_self, _ = tree.query(Fref, k=k + 1, workers=-1)
    radius = float(np.median(d_self[:, -1]))
    res['radius'] = radius; res['ref_n'] = int(ref_m.sum()); res['query_n'] = int(q_m.sum())
    nbrs = tree.query_ball_point(Fq, radius, workers=-1)
    cref = cost[ref_m].astype(np.float64)
    E100 = np.zeros(len(nbrs)); D10 = np.zeros(len(nbrs)); P = np.zeros(len(nbrs)); SSTAR = np.zeros(len(nbrs))
    for i, ids in enumerate(nbrs):
        if not ids:
            continue
        v = cref[np.asarray(ids, dtype=np.intp)]
        m = len(v)
        E100[i] = v.mean()
        kq = max(1, int(math.ceil(0.10 * m)))
        D10[i] = np.sort(v)[-kq:].mean()
        P[i] = float((v > 0).mean())
        SSTAR[i] = m
    res['empty_fraction'] = float(np.mean([len(x) == 0 for x in nbrs]))
    res['neighbor_count_mean'] = float(SSTAR.mean())
    # D10 / E100 分布与饱和
    def 分布(x):
        return dict(mean=float(x.mean()), std=float(x.std()), min=float(x.min()), max=float(x.max()),
                    unique=int(len(np.unique(np.round(x, 6)))),
                    frac_eq_1=float(np.mean(np.isclose(x, 1.0))),
                    frac_eq_0=float(np.mean(np.isclose(x, 0.0))),
                    frac_in_0_1=float(np.mean((x > 1e-9) & (x < 1 - 1e-9))),
                    p50=float(np.median(x)), p95=float(np.percentile(x, 95)))
    res['D10'] = 分布(D10); res['E100'] = 分布(E100); res['local_event_frac_p'] = 分布(P)
    res['local_frac_p_gt_10pct'] = float(np.mean(P > 0.10))
    # D10 的理论值 min(1,10p) 与实测的偏差
    res['corr_D10_E100'] = float(np.corrcoef(D10, E100)[0, 1])
    res['corr_D10_min1_10p'] = float(np.corrcoef(D10, np.minimum(1.0, 10 * P))[0, 1])
    # 并列度：随机取 20000 对，统计 D10/E100 相等的比例
    rng = np.random.RandomState(20261013)
    ii = rng.randint(0, len(D10), size=20000); jj = rng.randint(0, len(D10), size=20000)
    res['tie_rate_D10'] = float(np.mean(np.isclose(D10[ii], D10[jj])))
    res['tie_rate_E100'] = float(np.mean(np.isclose(E100[ii], E100[jj])))
    # 独立轨迹上"是否有风险信息"：用代理预测本步是否计费
    sq = np.sum(act[q_m] ** 2, axis=1)
    y = (cost[q_m] > 0).astype(int)
    from sklearn.metrics import roc_auc_score, average_precision_score
    def auc(x):
        try:
            return dict(auc=float(roc_auc_score(y, x)), pr_auc=float(average_precision_score(y, x)))
        except Exception:
            return dict(auc=None, pr_auc=None)
    res['predict_step_cost'] = dict(base_rate=float(y.mean()), D10=auc(D10), E100=auc(E100),
                                    actnorm2=auc(-sq), state_only=auc(obs[q_m][:, 3]))
    res['elapsed_sec'] = round(time.time() - t0, 1)
    with (out / '诊断结果.json').open('x', encoding='utf8') as f:
        json.dump(res, f, ensure_ascii=False, indent=2, default=float)
    print(json.dumps(res, ensure_ascii=False, indent=1, default=float))
    print('\n输出:', out)


if __name__ == '__main__':
    main()
