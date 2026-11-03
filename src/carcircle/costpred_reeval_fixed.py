#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""COSTPRED_V1 修正后重算：特征（正确 BC 网络）+ 冻结评价逻辑。
（实现缺陷修正后的重跑，不是新的独立检验。）
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.stats import spearmanr, pearsonr
from scipy.spatial import cKDTree
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
import iql          # noqa: E402
import night_lib as L  # noqa: E402

O = L.WORK / 'costpred'
FEATS = ['f_cost_surrogate', 'f_action_magnitude', 'f_action_ood']
SEED = 20261015


def load_net(path, kind):
    ck = torch.load(path, map_location='cpu', weights_only=False)
    sd = ck['actor'] if 'actor' in ck else ck['bc']
    if 'log_std' in sd:
        a = iql.Actor(8, 2); a.load_state_dict(sd); a.eval(); return a.net
    net = nn.Sequential(nn.Linear(8, 256), nn.ReLU(), nn.Linear(256, 256), nn.ReLU(),
                        nn.Linear(256, 2), nn.Tanh())
    net.load_state_dict(sd); net.eval(); return net


def main():
    t0 = time.time()
    data = iql.load_and_split(L.ENV_A)
    tr = data['train_idx'].astype(np.int64)
    S, A, C = data['S'], data['A'], data['C']
    om = torch.from_numpy(data['obs_mean']); osd = torch.from_numpy(data['obs_std'])
    Sstar = S[tr][::25]
    Xtr = np.concatenate([S[tr], A[tr]], axis=1)
    ytr = (C[tr] > 0).astype(int)
    sc = StandardScaler().fit(Xtr)
    clf = LogisticRegression(C=1, max_iter=2000, random_state=SEED).fit(sc.transform(Xtr), ytr)
    sm = S[tr].mean(0); ss = np.maximum(S[tr].std(0), 1e-8)
    tree = cKDTree((S[tr] - sm) / ss)
    nn_idx = tree.query((Sstar - sm) / ss, k=64, workers=-1)[1]
    st = torch.from_numpy(Sstar.astype(np.float32))
    norm_s = (st - om) / osd
    ros = json.loads((O / 'policy_roster.json').read_text(encoding='utf-8'))
    rows = []
    for p in ros['policies']:
        net = load_net(p['path'], p['kind'])
        with torch.no_grad():
            ap = net(norm_s).numpy().astype(np.float64)
        f1 = float(clf.predict_proba(sc.transform(np.concatenate([Sstar, ap], axis=1)))[:, 1].mean())
        f2 = float(np.linalg.norm(ap, axis=1).mean())
        nb = A[tr][nn_idx]
        f3 = float(np.linalg.norm(nb - ap[:, None, :], axis=2).min(axis=1).mean())
        rows.append(dict(policy_id=p['policy_id'], arm=p['arm'], split=p['split'],
                         f_cost_surrogate=f1, f_action_magnitude=f2, f_action_ood=f3))
    fe = pd.DataFrame(rows)
    fe.to_csv(O / 'features_offline_BCFIXED.csv', index=False, encoding='utf-8-sig')
    tg = pd.read_csv(O / 'targets_policy_summary_BCFIXED.csv')
    d = fe.merge(tg[['policy_id', 'n_episodes', 'mean_cost', 'cvar25', 'mean_reward']],
                 on='policy_id', how='inner')
    trd = d[d.split == 'train'].reset_index(drop=True)
    ted = d[d.split == 'test'].reset_index(drop=True)

    def model():
        return make_pipeline(StandardScaler(), Ridge(alpha=1.0))

    def metrics(y, yhat):
        y = np.asarray(y, float); yhat = np.asarray(yhat, float)
        rho = float(spearmanr(y, yhat).statistic) if len(set(y)) > 1 else float('nan')
        return dict(spearman=rho, pearson=float(pearsonr(y, yhat).statistic),
                    mae=float(np.mean(np.abs(y - yhat))),
                    rmse=float(np.sqrt(np.mean((y - yhat) ** 2))),
                    mae_over_target_sd=float(np.mean(np.abs(y - yhat)) / np.std(y, ddof=1)))

    ytr_ = trd.mean_cost.values; yte_ = ted.mean_cost.values
    res = dict(protocol='COSTPRED_V1', status='CORRECTED_RERUN（BC 网络加载缺陷修正后重跑）',
               n_train=len(trd), n_test=len(ted), target='mean_cost over env seeds 600-631',
               target_sd_test=float(ted.mean_cost.std(ddof=1)), models={})
    preds = {}
    preds['FULL_3'] = model().fit(trd[FEATS].values, ytr_).predict(ted[FEATS].values)
    res['models']['FULL_3'] = metrics(yte_, preds['FULL_3'])
    for f in FEATS:
        preds[f] = model().fit(trd[[f]].values, ytr_).predict(ted[[f]].values)
        res['models'][f] = metrics(yte_, preds[f])
    preds['MEAN_TRAIN'] = np.full(len(ted), ytr_.mean())
    res['models']['MEAN_TRAIN'] = metrics(yte_, preds['MEAN_TRAIN'])
    best, br = None, -9
    for f in FEATS:
        r = spearmanr(ytr_, model().fit(trd[[f]].values, ytr_).predict(trd[[f]].values)).statistic
        if r > br:
            best, br = f, r
    res['best_single_on_train'] = dict(feature=best, train_spearman=float(br))
    rng = np.random.default_rng(SEED)
    perm = [spearmanr(rng.permutation(yte_), preds['FULL_3']).statistic for _ in range(1000)]
    res['permutation_p_two_sided'] = float(np.mean(np.abs(perm) >= abs(res['models']['FULL_3']['spearman'])))
    res['per_family_mae'] = {a: dict(n=len(g), mae=float(np.mean(np.abs(
        g.mean_cost - g.pred_FULL_3))), mean_cost_true=float(g.mean_cost.mean()))
        for a, g in ted.assign(pred_FULL_3=preds['FULL_3']).groupby('arm')}
    (O / 'costpred_results_BCFIXED.json').write_text(json.dumps(res, ensure_ascii=False, indent=2),
                                                    encoding='utf-8')
    out = ted.copy()
    for k, v in preds.items():
        out['pred_%s' % k] = v
    out.to_csv(O / 'costpred_predictions_BCFIXED.csv', index=False, encoding='utf-8-sig')
    print(json.dumps(res, ensure_ascii=False, indent=2))
    print('用时 %.0fs' % (time.time() - t0))


if __name__ == '__main__':
    main()
