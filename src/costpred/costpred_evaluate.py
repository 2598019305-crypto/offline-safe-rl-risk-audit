#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""COSTPRED_V1：一次性评价（按冻结协议：分组切分 + 固定模型 + 固定指标）。"""
from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT = Path(os.environ.get('REPRO_OUTPUT_DIR', REPO_ROOT / 'outputs')) / 'costpred'
FEATS = ['f_cost_surrogate', 'f_action_magnitude', 'f_action_ood']
SEED = 20261015


def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for c in iter(lambda: f.read(1 << 20), b''):
            h.update(c)
    return h.hexdigest()


def model():
    return make_pipeline(StandardScaler(), Ridge(alpha=1.0))


def metrics(y, yhat):
    y = np.asarray(y, float); yhat = np.asarray(yhat, float)
    rho = float(spearmanr(y, yhat).statistic) if len(set(y)) > 1 else float('nan')
    r = float(pearsonr(y, yhat).statistic) if len(set(y)) > 1 else float('nan')
    mae = float(np.mean(np.abs(y - yhat)))
    rmse = float(np.sqrt(np.mean((y - yhat) ** 2)))
    return dict(spearman=rho, pearson=r, mae=mae, rmse=rmse,
                mae_over_target_sd=float(mae / np.std(y, ddof=1)) if np.std(y, ddof=1) > 0 else None)


def main():
    fr = json.loads((OUT / 'COSTPRED_V1_frozen.json').read_text(encoding='utf-8'))
    fe = pd.read_csv(OUT / 'features_offline.csv')
    tg = pd.read_csv(OUT / 'targets_policy_summary.csv')
    d = fe.merge(tg[['policy_id', 'n_episodes', 'mean_cost', 'cvar25',
                     'mean_cost_after_10pct', 'mean_reward', 'std_cost']],
                 on='policy_id', how='inner')
    d = d.dropna(subset=['mean_cost'] + FEATS)
    tr = d[d.split == 'train'].reset_index(drop=True)
    te = d[d.split == 'test'].reset_index(drop=True)
    res = dict(protocol='COSTPRED_V1',
               frozen_sha256=sha(OUT / 'COSTPRED_V1_frozen.json'),
               features_sha256=sha(OUT / 'features_offline.csv'),
               targets_sha256=sha(OUT / 'targets_policy_summary.csv'),
               n_train=len(tr), n_test=len(te),
               target='mean_cost over env seeds 600-631',
               target_sd_test=float(te.mean_cost.std(ddof=1)),
               models={}, per_family_mae={}, random_split_reference=None)
    ytr = tr.mean_cost.values
    yte = te.mean_cost.values
    preds = {}
    # 主模型
    m = model().fit(tr[FEATS].values, ytr)
    preds['FULL_3'] = m.predict(te[FEATS].values)
    res['models']['FULL_3'] = metrics(yte, preds['FULL_3'])
    # 单特征基线
    for f in FEATS:
        mm = model().fit(tr[[f]].values, ytr)
        preds[f] = mm.predict(te[[f]].values)
        res['models'][f] = metrics(yte, preds[f])
    # 常数基线
    preds['MEAN_TRAIN'] = np.full(len(te), ytr.mean())
    res['models']['MEAN_TRAIN'] = metrics(yte, preds['MEAN_TRAIN'])
    # 训练集上最优单特征
    best, best_rho = None, -9
    for f in FEATS:
        mm = model().fit(tr[[f]].values, ytr)
        rho = spearmanr(ytr, mm.predict(tr[[f]].values)).statistic
        if rho > best_rho:
            best, best_rho = f, rho
    res['best_single_on_train'] = dict(feature=best, train_spearman=float(best_rho))
    res['models']['BEST_SINGLE_ON_TRAIN'] = res['models'][best]
    # 置换检验（主模型）
    rng = np.random.default_rng(SEED)
    obs = res['models']['FULL_3']['spearman']
    perm = []
    for _ in range(1000):
        yp = rng.permutation(yte)
        perm.append(spearmanr(yp, preds['FULL_3']).statistic)
    perm = np.asarray(perm)
    res['permutation_p_two_sided'] = float(np.mean(np.abs(perm) >= abs(obs)))
    res['permutation_note'] = '1000 次置换，描述性'
    # 分家族 MAE
    te2 = te.copy()
    te2['pred_FULL_3'] = preds['FULL_3']
    for arm, g in te2.groupby('arm'):
        res['per_family_mae'][arm] = dict(n=len(g),
                                          mae=float(np.mean(np.abs(g.mean_cost - g.pred_FULL_3))),
                                          mean_cost_true=float(g.mean_cost.mean()))
    # 次要：随机 70/30（同家族可能跨侧，不作为外推证据）
    rng2 = np.random.default_rng(SEED)
    idx = rng2.permutation(len(d))
    n_tr = int(0.7 * len(d))
    a, b = d.iloc[idx[:n_tr]], d.iloc[idx[n_tr:]]
    mm = model().fit(a[FEATS].values, a.mean_cost.values)
    res['random_split_reference'] = dict(
        n_train=len(a), n_test=len(b),
        **metrics(b.mean_cost.values, mm.predict(b[FEATS].values)),
        note='随机切分，同家族可能同时出现在两侧，不作为外推证据')
    # 写出
    (OUT / 'costpred_results.json').write_text(json.dumps(res, ensure_ascii=False, indent=2),
                                               encoding='utf-8')
    out = te.copy()
    for k, v in preds.items():
        out['pred_%s' % k] = v
    out.to_csv(OUT / 'costpred_predictions.csv', index=False, encoding='utf-8-sig')
    tr_out = tr.copy()
    tr_out['pred_train_fit'] = model().fit(tr[FEATS].values, ytr).predict(tr[FEATS].values)
    tr_out.to_csv(OUT / 'costpred_train_policies.csv', index=False, encoding='utf-8-sig')
    print(json.dumps(res, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
